import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import util
from InGAN import InGAN
from configs import Config
import torch.nn as nn


def main():
    conf = Config().parse()
    gan = InGAN(conf)
    sd = torch.load('./results/building_MSDf_maxpool_decay_30_no_spec_norm_G_Oct_09_18_47_12/snapshot-100000.pth.tar')
    gan.G.load_state_dict(sd['G'])
    # gan.G.eval()
    gan.D.load_state_dict(sd['D'])
    # gan.D.eval()
    with torch.no_grad():
        ns = 1
        sf = np.sqrt(.5)
        nms = 1

        # for input_name, exp_name in zip(['input5.png'], ['bull']):
        for input_name, exp_name in zip(['input7.png', 'input7_mod1.png'], ['building', 'building-mod1']):
            for swap in [False]:
                # input_image = np.array(Image.open('./' + input_name).convert('RGB')) / 255.0
                pil_image = Image.open('./' + input_name).convert('RGB')
                ow, oh = pil_image.size
                # if swap:
                #     input_image = input_image[:, ::-1, :]
                for ratio in [(1., 1.), (1., 2.), (1, 0.5), (2., 1.)]:

                    sw = int(ow * sf**(ns-1))
                    sh = int(oh * sf**(ns-1))
                    sc_img = pil_image.resize((sw, sh), Image.ANTIALIAS)
                    rsf = (np.power(ratio[0], 1. / nms), np.power(ratio[1], 1./nms))
                    w, h = sc_img.size
                    input_image = np.array(sc_img) / 255.
                    for ms in range(nms):
                        # input_image = np.array(sc_img.resize((int(h * rsf[1]**(ms+1)), int(w*rsf[0]**(ms+1))), Image.ANTIALIAS)) / 255.
                        out_size = (input_image.shape[0] * rsf[0], input_image.shape[1] * rsf[1])
                        size = [int((out_size[i_] // 2) * 2) for i_ in range(2)]
                        # mp.output_ratio = ratio
                        g_pred, d_pred, _ = gan.test(util.im2tensor(input_image, False),
                                                     input_size=input_image.shape[:2],
                                                     output_size=size,
                                                     run_d_pred=True, run_reconstruct=True)

                        # img = util.image_concat(util.tensor2im(g_pred), util.tensor2im(d_pred))
                        # plt.imshow(np.clip(img, 0, 255), vmin=0, vmax=255)
                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].imshow(util.tensor2im(g_pred[0]))
                        # ax[1].imshow(util.tensor2im(g_pred[1]))
                        plt.imshow(util.tensor2im(g_pred[1]))
                        plt.title('{:0.2f}x{:0.2f}'.format(ratio[0], ratio[1]))
                        plt.pause(0.01)
                        plt.savefig('./sandbox-{}-{:0.2f}x{:0.2f}-{}_{}-{}_{}-eval.png'
                                    .format(exp_name, ratio[0], ratio[1], ns, ns, ms, nms))
                        input_image = util.tensor2im(g_pred[1])
                        # del fig
                    for ls in range(ns-1, 0, -1):
                        out_size = (input_image.shape[0] / sf, input_image.shape[1] / sf)
                        size = [int((out_size[i_] // 2) * 2) for i_ in range(2)]
                        # mp.output_ratio = ratio
                        g_pred, d_pred, _ = gan.test(util.im2tensor(input_image, False),
                                                     input_size=input_image.shape[:2],
                                                     output_size=size,
                                                     run_d_pred=True, run_reconstruct=True)

                        # img = util.image_concat(util.tensor2im(g_pred), util.tensor2im(d_pred))
                        # plt.imshow(np.clip(img, 0, 255), vmin=0, vmax=255)
                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].imshow(util.tensor2im(g_pred[0]))
                        # ax[1].imshow(util.tensor2im(g_pred[1]))
                        plt.imshow(util.tensor2im(g_pred[1]))
                        plt.title('{:0.2f}x{:0.2f}'.format(ratio[0], ratio[1]))
                        plt.pause(0.01)
                        plt.savefig('./sandbox-{}-{:0.2f}x{:0.2f}-{}_{}-{}_{}-eval.png'
                                    .format(exp_name, ratio[0], ratio[1], ls, ns, nms, nms))
                        input_image = util.tensor2im(g_pred[1])


        # g_pred, d_pred, _ = gan.test(util.im2tensor(input_image[100:196, 100:196, :], False), input_size=(96, 96), output_size=(96, 96))
        # g_pred, d_pred, _ = gan.test(g_pred[1][:, :, 100:196, 100:196], input_size=(96, 96),
        #                              output_size=(96, 96))
        # img = util.image_concat(util.tensor2im(g_pred[:3]), util.tensor2im(d_pred[:3]))
        # plt.imshow(np.clip(img, 0, 255), vmin=0, vmax=255)
        # plt.title('top left crop'.format(ratio[0], ratio[1]))
        # plt.pause(0.01)
        # plt.savefig('./sandbox-top-left-crop-eval.png')
    print('done')


if __name__ == '__main__':
    main()
