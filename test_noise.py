import numpy as np
import torch
import util
from InGAN import InGAN
from configs import Config
from traceback import print_exc
from skvideo.io import FFmpegWriter
import os
from networks import MoveCrop


def test_one(gan, input_tensor, scale=(1.0, 1.0), must_divide=8):
    with torch.no_grad():
        in_size = input_tensor.shape[2:]
        out_size = (np.uint32(np.floor(scale[0] * in_size[0] * 1.0 / must_divide) * must_divide),
                    np.uint32(np.floor(scale[1] * in_size[1] * 1.0 / must_divide) * must_divide))

        output_tensor, _, _ = gan.test(input_tensor=input_tensor,
                                       input_size=in_size,
                                       output_size=out_size,
                                       run_d_pred=False,
                                       run_reconstruct=False)
        return util.tensor2im(output_tensor[1])


def create_vid(gan, input_tensor, n_frames, output_dir_path, area=None, sigma=2.4, scale=(1.0, 1.0), must_divide=8):
    frame_shape = np.uint32(np.array(input_tensor.shape[2:]))
    frame_shape[0] += (frame_shape[0] % 2)
    frame_shape[1] += (frame_shape[1] % 2)
    frames = np.zeros([n_frames, frame_shape[0], frame_shape[1], 3])
    if area is not None:
        area_mask = torch.zeros_like(input_tensor)
        area_mask[:, :, area[0]:area[2], area[1]:area[3]] = 1.0

    for i in range(n_frames):
        noise = area_mask * torch.randn_like(input_tensor) * sigma if area_mask is not None \
            else torch.randn_like(input_tensor) * sigma
        in_tensor = input_tensor + noise
        output_image = test_one(gan, in_tensor, scale, must_divide)
        frames[i, 0:output_image.shape[0], 0:output_image.shape[1], :] = output_image
    writer = FFmpegWriter(output_dir_path + '/vid.mp4', verbosity=1, outputdict={'-b': '30000000', '-r': '100.0'})

    for i in range(n_frames):
        for _ in range(3):
            writer.writeFrame(frames[i, :, :, :])
    writer.close()


def create_flow_vid(gan, input_tensor, n_stops, n_frames_per_stop, output_dir_path,
                    area=None, sigma=3.0, scale=(1.0, 1.0), must_divide=8):

    if area is not None:
        area_mask = torch.zeros_like(input_tensor)
        area_mask[:, :, area[0]:area[2], area[1]:area[3]] = 1.0
        stops = [area_mask * torch.randn_like(input_tensor) * sigma for _ in range(n_stops)]
    else:
        stops = [make_pink_noise(input_tensor.shape, sigma) for _ in range(n_stops)]
        # stops = [torch.randn_like(input_tensor) * sigma for _ in range(n_stops)]
    n_frames = n_frames_per_stop * (n_stops - 1)

    frame_shape = np.uint32(np.array(input_tensor.shape[2:]))
    frame_shape[0] += (frame_shape[0] % 2)
    frame_shape[1] += (frame_shape[1] % 2)
    frames = np.zeros([n_frames, frame_shape[0], frame_shape[1], 3])

    for j, (stop_before, stop_after) in enumerate(zip(stops, stops[1:])):
        for i in range(n_frames_per_stop):
            in_tensor = input_tensor + stop_before + i * (stop_after - stop_before) / n_frames_per_stop
            output_image = test_one(gan, in_tensor, scale, must_divide)
            frames[j * n_frames_per_stop + i, 0:output_image.shape[0], 0:output_image.shape[1], :] = output_image

    writer = FFmpegWriter(output_dir_path + '/vid.mp4', verbosity=1, outputdict={'-b': '30000000', '-r': '100.0'})

    for i in range(n_frames):
        for _ in range(3):
            writer.writeFrame(frames[i, :, :, :])
    writer.close()


def reshuffle_video(gan, input_tensor, output_dir_path, orig_top_left_v, orig_top_left_h, cr_v_sz, cr_h_sz):
    move_crop = MoveCrop().forward
    H, W = input_tensor.shape[2:]

    frame_shape = np.uint32(np.array(input_tensor.shape[2:]))
    frame_shape[0] += (frame_shape[0] % 2)
    frame_shape[1] += (frame_shape[1] % 2)
    frames = np.zeros([H*W/16, frame_shape[0], frame_shape[1], 3])

    for i in range(0, H-1, 4):
        for j in range(0, W-1, 4):
            if np.any(np.array([i, j]) + np.array([cr_v_sz, cr_h_sz]) > np.array([H, W]) - 10):
                frames[i * W / 16 + j / 4, 0:output_image.shape[0], 0:output_image.shape[1], :] = 0
                continue

            in_tensor = move_crop(input_tensor, orig_top_left_v, orig_top_left_h, i, j, cr_v_sz, cr_h_sz)
            output_image = test_one(gan, in_tensor)
            frames[i * W / 16 + j / 4, 0:output_image.shape[0], 0:output_image.shape[1], :] = output_image

    writer = FFmpegWriter(output_dir_path + '/vid.mp4', verbosity=1, outputdict={'-b': '30000000', '-r': '100.0'})

    for i in range(H*W/16):
        for _ in range(3):
            writer.writeFrame(frames[i, :, :, :])
    writer.close()


def main():
    name_infile_ckptfile_list = [
        # ['fruits_resh', 'fruits_ss.png', '/fruits_ss_noise_sig_3_Mar_01_05_22_52'],
        # ['bull_resh', 'bull.png', '/bull_gini_mask_2_Mar_01_23_14_18'],
        ['farm_house_resh', 'farm_house_s.png', '/farm_house_s_gini_mask_2_Mar_01_19_59_37']

    ]

    snapshot_iters = [50000]
    n_files = len(name_infile_ckptfile_list)

    for i, (name, input_image_path, test_params_path) in enumerate(name_infile_ckptfile_list):
        for snapshot_iter in snapshot_iters:

            conf = Config().parse(create_dir_flag=False)
            conf.name = 'TEST_' + name + '_iter_%dk' % (snapshot_iter / 1000)
            conf.output_dir_path = util.prepare_result_dir(conf)
            conf.input_image_path = [os.path.dirname(os.path.abspath(__file__)) + '/' + input_image_path]
            conf.test_params_path = os.path.dirname(
                os.path.abspath(__file__)) + '/results/' + test_params_path + '/checkpoint_%07d.pth.tar' % snapshot_iter
            gan = InGAN(conf)

            try:
                gan.resume(conf.test_params_path)
                [input_tensor] = util.read_data(conf)

                # create_vid(gan, input_tensor, 100, conf.output_dir_path, [130, 0, 180, 120])
                # create_flow_vid(gan, input_tensor, 20, 5 , conf.output_dir_path, [130, 0, 180, 120])
                # create_flow_vid(gan, input_tensor, 20, 5, conf.output_dir_path)
                # reshuffle_video(gan, input_tensor, conf.output_dir_path, 140, 170, 80, 80)
                # reshuffle_video(gan, input_tensor, conf.output_dir_path, 45, 180, 100, 75)
                reshuffle_video(gan, input_tensor, conf.output_dir_path, 100, 200, 170, 200)

                print 'Done with %s (%d/%d), iter %dk' % (input_image_path, i, n_files, snapshot_iter)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print 'Something went wrong with %s (%d/%d), iter %dk' % (input_image_path, i, n_files, snapshot_iter)
                print_exc()


if __name__ == '__main__':
    main()
