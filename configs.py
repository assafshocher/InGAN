import argparse
import torch
import os
from util import prepare_result_dir


# noinspection PyPep8
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # Paths
        self.parser.add_argument('--input_image_path', default=[os.path.dirname(os.path.abspath(__file__)) + '/14_s.jpg'], nargs='+', help='path to one specific image file')
        self.parser.add_argument('--output_dir_path', default=os.path.dirname(os.path.abspath(__file__)) + '/results', help='path to a directory to save results to')
        self.parser.add_argument('--name', default='dbg', help='name of current experiment, to be used for saving the results')
        self.parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
        self.parser.add_argument('--test_params_path', type=str, default=None, help='checkpoint for testing')

        # Architecture (Generator)
        self.parser.add_argument('--G_base_channels', type=int, default=64, help='# of base channels in G')
        self.parser.add_argument('--G_num_resblocks', type=int, default=6, help='# of resblocks in G\'s bottleneck')
        self.parser.add_argument('--G_num_downscales', type=int, default=3, help='# of downscaling layers in G')
        self.parser.add_argument('--G_use_bias', type=bool, default=True, help='Determinhes whether bias is used in G\'s conv layers')
        self.parser.add_argument('--G_skip', type=bool, default=True, help='Determines wether G uses skip connections (U-net)')

        # Architecture (Discriminator)
        self.parser.add_argument('--D_base_channels', type=int, default=64, help='# of base channels in D')
        self.parser.add_argument('--D_max_num_scales', type=int, default=99, help='Limits the # of scales for the multiscale D')
        self.parser.add_argument('--D_scale_factor', type=float, default=1.4, help='Determines the downscaling factor for multiscale D')
        self.parser.add_argument('--D_scale_weights_sigma', type=float, default=1.4, help='Determines the downscaling factor for multiscale D')
        self.parser.add_argument('--D_min_input_size', type=int, default=13, help='Determines the downscaling factor for multiscale D')
        self.parser.add_argument('--D_scale_weights_iter_for_even_scales', type=int, default=25000, help='Determines the downscaling factor for multiscale D')

        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=0.00005, help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=0.00005, help='initial learning rate for discriminator')
        self.parser.add_argument('--lr_start_decay_iter', type=float, default=20000, help='iteration from which linear decay of lr starts until max_iter')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--curriculum', type=bool, default=True, help='Enable curriculum learning')
        self.parser.add_argument('--iter_for_max_range', type=int, default=10000, help='In curriculum learning, when getting to this iteration all range is covered')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=256, help='input is cropped to this size')
        self.parser.add_argument('--output_crop_size', type=int, default=256, help='output is cropped to this size')
        self.parser.add_argument('--max_scale', type=float, default=2.25, help='max retargeting scale')
        self.parser.add_argument('--min_scale', type=float, default=0.15, help='min retargeting scale')
        self.parser.add_argument('--must_divide', type=int, default=8, help='In curriculum learning, when getting to this iteration all range is covered')
        self.parser.add_argument('--max_transform_magnitude', type=float, default=0.4, help='max manitude of geometric transformation')

        # Crop Swap
        self.parser.add_argument('--crop_swap_min_size', type=int, default=32, help='swapping crops augmnetation')
        self.parser.add_argument('--crop_swap_max_size', type=int, default=256, help='swapping crops augmnetation')
        self.parser.add_argument('--crop_swap_probability', type=float, default=0.0, help='probability for crop swapping to occur')

        # GPU
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id number')

        # Monitoring display frequencies
        self.parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
        self.parser.add_argument('--save_snapshot_freq', type=int, default=5000, help='frequency of saving the latest results')

        # Iterations
        self.parser.add_argument('--max_iters', type=int, default=75000, help='max # of iters')
        self.parser.add_argument('--G_iters', type=int, default=1, help='# of sub-iters for the generator per each global iteration')
        self.parser.add_argument('--D_iters', type=int, default=1, help='# of sub-iters for the discriminator per each global iteration')

        # Losses
        self.parser.add_argument('--reconstruct_loss_proportion', type=float, default=0.1, help='relative part of reconstruct-loss (out of 1)')
        self.parser.add_argument('--reconstruct_loss_stop_iter', type=int, default=200000, help='from this iter and on, reconstruct loss is deactivated')
        self.parser.add_argument('--G_extra_inverse_train', type=int, default=1, help='number of extra training iters for G on inverse direction')
        self.parser.add_argument('--G_extra_inverse_train_start_iter', type=int, default=10000, help='number of extra training iters for G on inverse direction')
        self.parser.add_argument('--G_extra_inverse_train_ratio', type=int, default=1.0, help='number of extra training iters for G on inverse direction')
        self.parser.add_argument('--use_L1', type=bool, default=True, help='Determine whether to use L1 or L2 for reconstruction')

        # Misc
        self.parser.add_argument('--create_code_copy', type=bool, default=True, help='when set to true, all .py files are saved to results directory to keep track')

    def parse(self, create_dir_flag=True):
        # Parse arguments
        self.conf = self.parser.parse_args()

        # set gpu ids
        torch.cuda.set_device(self.conf.gpu_id)

        # Create results dir if does not exist
        if create_dir_flag:
            self.conf.output_dir_path = prepare_result_dir(self.conf)

        return self.conf
