from __future__ import print_function

import torch
from torch.autograd import Variable
import networks
from util import random_size, get_scale_weights
import os
import warnings
import numpy as np


class LRPolicy(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __call__(self, citer):
        return 1. - max(0., float(citer - self.start) / float(self.end - self.start))


# noinspection PyAttributeOutsideInit
class InGAN:
    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf
        self.cur_iter = 0
        self.max_iters = conf.max_iters

        # Define input tensor
        self.input_tensor = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()
        self.real_example = torch.FloatTensor(1, 3, conf.output_crop_size, conf.output_crop_size).cuda()

        # Define networks
        self.G = networks.Generator(conf.G_base_channels, conf.G_num_resblocks, conf.G_num_downscales, conf.G_use_bias,
                                    conf.G_skip)
        self.D = networks.MultiScaleDiscriminator(conf.output_crop_size,  self.conf.D_max_num_scales,
                                                  self.conf.D_scale_factor, self.conf.D_base_channels)
        self.GAN_loss_layer = networks.GANLoss()
        self.Reconstruct_loss = networks.WeightedMSELoss(use_L1=conf.use_L1)
        self.RandCrop = networks.RandomCrop([conf.input_crop_size, conf.input_crop_size], must_divide=conf.must_divide)
        self.SwapCrops = networks.SwapCrops(conf.crop_swap_min_size, conf.crop_swap_max_size)

        # Make all networks run on GPU
        self.G.cuda()
        self.D.cuda()
        self.GAN_loss_layer.cuda()
        self.Reconstruct_loss.cuda()
        self.RandCrop.cuda()
        self.SwapCrops.cuda()

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward
        self.criterionReconstruction = self.Reconstruct_loss.forward

        # Keeping track of losses- prepare tensors
        self.losses_G_gan = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_D_real = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_D_fake = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_G_reconstruct = torch.FloatTensor(conf.print_freq).cuda()
        if self.conf.reconstruct_loss_stop_iter > 0:
            self.losses_D_reconstruct = torch.FloatTensor(conf.print_freq).cuda()

        # Initialize networks
        self.G.apply(networks.weights_init)
        self.D.apply(networks.weights_init)

        # Initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))

        # Learning rate scheduler
        # First define linearly decaying functions (decay starts at a special iter)
        start_decay = conf.lr_start_decay_iter
        end_decay = conf.max_iters
        # def lr_function(n_iter):
        #     return 1 - max(0, 1.0 * (n_iter - start_decay) / (conf.max_iters - start_decay))
        lr_function = LRPolicy(start_decay, end_decay)
        # Define learning rate schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_function)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_function)

        # # do we resume from checkpoint?
        # if self.conf.resume:
        #     print('resuming checkpoint {}'.format(self.conf.resume))
        #     self.resume(self.conf.resume)

    def save(self, citer=None):
        if citer is None:
            filename = 'snapshot.pth.tar'
        elif isinstance(citer, str):
            filename = citer
        else:
            filename = 'snapshot-{:05d}.pth.tar'.format(citer)
        torch.save({'G': self.G.state_dict(),
                    'D': self.D.state_dict(),
                    'optim_G': self.optimizer_G.state_dict(),
                    'optim_D': self.optimizer_D.state_dict(),
                    'sched_G': self.lr_scheduler_G.state_dict(),
                    'sched_D': self.lr_scheduler_D.state_dict(),
                    'loss': self.GAN_loss_layer.state_dict(),
                    'iter': citer if citer else self.cur_iter},
                   os.path.join(self.conf.output_dir_path, filename))

    def resume(self, resume_path, test_flag=False):
        resume = torch.load(resume_path, map_location={'cuda:5': 'cuda:0'})
        missing = []
        if 'G' in resume:
            self.G.load_state_dict(resume['G'])
        else:
            missing.append('G')
        if 'D' in resume:
            self.D.load_state_dict(resume['D'])
        else:
            missing.append('D')
        if not test_flag:
            if 'optim_G' in resume:
                self.optimizer_G.load_state_dict(resume['optim_G'])
            else:
                missing.append('optimizer G')
            if 'optim_D' in resume:
                self.optimizer_D.load_state_dict(resume['optim_D'])
            else:
                missing.append('optimizer D')
            if 'sched_G' in resume:
                self.lr_scheduler_G.load_state_dict(resume['sched_G'])
            else:
                missing.append('lr scheduler G')
            if 'sched_D' in resume:
                self.lr_scheduler_D.load_state_dict(resume['sched_D'])
            else:
                missing.append('lr scheduler G')
            if 'loss' in resume:
                self.GAN_loss_layer.load_state_dict(resume['loss'])
            else:
                missing.append('GAN loss')
        if len(missing):
            warnings.warn('Missing the following state dicts from checkpoint: {}'.format(', '.join(missing)))

        print('resuming checkpoint {}'.format(self.conf.resume))

    def test(self, input_tensor, output_size, rand_affine, input_size, run_d_pred=True, run_reconstruct=True):
        with torch.no_grad():
            self.G_pred = self.G.forward(Variable(input_tensor.detach()), output_size=output_size, random_affine=rand_affine)
            if run_d_pred:
                scale_weights_for_output = get_scale_weights(i=self.cur_iter,
                                                             max_i=self.conf.D_scale_weights_iter_for_even_scales,
                                                             start_factor=self.conf.D_scale_weights_sigma,
                                                             input_shape=self.G_pred.shape[2:],
                                                             min_size=self.conf.D_min_input_size,
                                                             num_scales_limit=self.conf.D_max_num_scales,
                                                             scale_factor=self.conf.D_scale_factor)
                scale_weights_for_input = get_scale_weights(i=self.cur_iter,
                                                            max_i=self.conf.D_scale_weights_iter_for_even_scales,
                                                            start_factor=self.conf.D_scale_weights_sigma,
                                                            input_shape=input_tensor.shape[2:],
                                                            min_size=self.conf.D_min_input_size,
                                                            num_scales_limit=self.conf.D_max_num_scales,
                                                            scale_factor=self.conf.D_scale_factor)
                self.D_preds = [self.D.forward(Variable(input_tensor.detach()), scale_weights_for_input),
                                self.D.forward(Variable(self.G_pred.detach()), scale_weights_for_output)]
            else:
                self.D_preds = None

            self.G_preds = [input_tensor, self.G_pred]

            self.reconstruct = self.G.forward(self.G_pred, output_size=input_size, random_affine=-rand_affine) if run_reconstruct else None

        return self.G_preds, self.D_preds, self.reconstruct

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        # Determine output size of G (dynamic change)
        output_size, random_affine = random_size(orig_size=self.input_tensor.shape[2:],
                                                 curriculum=self.conf.curriculum,
                                                 i=self.cur_iter,
                                                 iter_for_max_range=self.conf.iter_for_max_range,
                                                 must_divide=self.conf.must_divide,
                                                 min_scale=self.conf.min_scale,
                                                 max_scale=self.conf.max_scale,
                                                 max_transform_magniutude=self.conf.max_transform_magnitude)

        # Add noise to G input for better generalization (make it ignore the 1/255 binning)
        self.input_tensor_noised = self.input_tensor + (torch.rand_like(self.input_tensor) - 0.5) * 2.0 / 255

        # Generator forward pass
        self.G_pred = self.G.forward(self.input_tensor_noised, output_size=output_size, random_affine=random_affine)

        # Run generator result through discriminator forward pass
        self.scale_weights = get_scale_weights(i=self.cur_iter,
                                               max_i=self.conf.D_scale_weights_iter_for_even_scales,
                                               start_factor=self.conf.D_scale_weights_sigma,
                                               input_shape=self.G_pred.shape[2:],
                                               min_size=self.conf.D_min_input_size,
                                               num_scales_limit=self.conf.D_max_num_scales,
                                               scale_factor=self.conf.D_scale_factor)
        d_pred_fake = self.D.forward(self.G_pred, self.scale_weights)

        # If reconstruction-loss is used, run through decoder to reconstruct, then calculate reconstruction loss
        if self.conf.reconstruct_loss_stop_iter > self.cur_iter:
            self.reconstruct = self.G.forward(self.G_pred, output_size=self.input_tensor.shape[2:], random_affine=-random_affine)
            self.loss_G_reconstruct = self.criterionReconstruction(self.reconstruct, self.input_tensor, self.loss_mask)

        # Calculate generator loss, based on discriminator prediction on generator result
        self.loss_G_GAN = self.criterionGAN(d_pred_fake, is_d_input_real=True)

        # Generator final loss
        # Weighted average of the two losses (if indicated to use reconstruction loss)
        if self.conf.reconstruct_loss_stop_iter < self.cur_iter:
            self.loss_G = self.loss_G_GAN
        else:
            self.loss_G = (self.conf.reconstruct_loss_proportion * self.loss_G_reconstruct + self.loss_G_GAN)

        # Calculate gradients
        # Note that the gradients are propagated from the loss through discriminator and then through generator
        self.loss_G.backward()

        # Update weights
        # Note that only generator weights are updated (by definition of the G optimizer)
        self.optimizer_G.step()

        # Extra training for the inverse G. The difference between this and the reconstruction is the .detach() which
        # makes the training only for the inverse G and not for regular G.
        if self.cur_iter > self.conf.G_extra_inverse_train_start_iter:
            for _ in range(self.conf.G_extra_inverse_train):
                self.optimizer_G.zero_grad()
                self.inverse = self.G.forward(self.G_pred.detach(), output_size=self.input_tensor.shape[2:], random_affine=-random_affine)
                self.loss_G_inverse = (self.criterionReconstruction(self.inverse, self.input_tensor, self.loss_mask) *
                                       self.conf.G_extra_inverse_train_ratio)
                self.loss_G_inverse.backward()
                self.optimizer_G.step()

        # Update learning rate scheduler
        self.lr_scheduler_G.step()

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()

        # Adding noise to D input to prevent overfitting to 1/255 bins
        real_example_with_noise = self.real_example + (torch.rand_like(self.real_example[-1]) - 0.5) * 2.0 / 255.0

        # Discriminator forward pass over real example
        self.d_pred_real = self.D.forward(real_example_with_noise, self.scale_weights)

        # Adding noise to D input to prevent overfitting to 1/255 bins
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_pred_with_noise = self.G_pred.detach() + (torch.rand_like(self.G_pred) - 0.5) * 2.0 / 255

        # Discriminator forward pass over generated example example
        self.d_pred_fake = self.D.forward(g_pred_with_noise, self.scale_weights)

        # Calculate discriminator loss
        self.loss_D_fake = self.criterionGAN(self.d_pred_fake, is_d_input_real=False)
        self.loss_D_real = self.criterionGAN(self.d_pred_real, is_d_input_real=True)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        # Calculate gradients
        # Note that gradients are not propagating back through generator
        # noinspection PyUnresolvedReferences
        self.loss_D.backward()

        # Update weights
        # Note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

        # Update learning rate scheduler
        self.lr_scheduler_D.step()

    def train_one_iter(self, cur_iter, input_tensors):
        # Set inputs as random crops
        input_crops = []
        mask_crops = []
        real_example_crops = []
        mask_flag = False
        for input_tensor in input_tensors:
            real_example_crops += self.RandCrop.forward([input_tensor])

            if np.random.rand() < self.conf.crop_swap_probability:
                swapped_input_tensor, loss_mask = self.SwapCrops.forward(input_tensor)
                [input_crop, mask_crop] = self.RandCrop.forward([swapped_input_tensor, loss_mask])
                input_crops.append(input_crop)
                mask_crops.append(mask_crop)
                mask_flag = True
            else:
                input_crops.append(real_example_crops[-1])

        self.input_tensor = torch.cat(input_crops)
        self.real_example = torch.cat(real_example_crops)
        self.loss_mask = torch.cat(mask_crops) if mask_flag else None

        # Update current iteration
        self.cur_iter = cur_iter

        # Run a single forward-backward pass on the model and update weights
        # One global iteration includes several iterations of generator and several of discriminator
        # (not necessarily equal)
        # noinspection PyRedeclaration
        for _ in range(self.conf.G_iters):
            self.train_g()

        # noinspection PyRedeclaration
        for _ in range(self.conf.D_iters):
            self.train_d()

        # Accumulate stats
        # Accumulating as cuda tensors is much more efficient than passing info from GPU to CPU at every iteration
        self.losses_G_gan[cur_iter % self.conf.print_freq] = self.loss_G_GAN.item()
        self.losses_D_fake[cur_iter % self.conf.print_freq] = self.loss_D_fake.item()
        self.losses_D_real[cur_iter % self.conf.print_freq] = self.loss_D_real.item()
        if self.conf.reconstruct_loss_stop_iter > self.cur_iter:
            self.losses_G_reconstruct[cur_iter % self.conf.print_freq] = self.loss_G_reconstruct.item()
