from networks import GeoTransform
from PIL import Image
import util
from InGAN import InGAN
from configs import Config
from traceback import print_exc
from skvideo.io import FFmpegWriter
import os
from non_rect import *


def test_one_scale(gan, input_tensor, scale, must_divide, affine=None, return_tensor=False, size_instead_scale=False):
    with torch.no_grad():
        in_size = input_tensor.shape[2:]
        if size_instead_scale:
            out_size = scale
        else:
            out_size = (np.uint32(np.floor(scale[0] * in_size[0] * 1.0 / must_divide) * must_divide),
                        np.uint32(np.floor(scale[1] * in_size[1] * 1.0 / must_divide) * must_divide))

        output_tensor, _, _ = gan.test(input_tensor=input_tensor,
                                       input_size=in_size,
                                       output_size=out_size,
                                       rand_affine=affine,
                                       run_d_pred=False,
                                       run_reconstruct=False)
        if return_tensor:
            return output_tensor[1]
        else:
            return util.tensor2im(output_tensor[1])


def concat_images(images, margin, input_spot):
    # for i in range(len(images)):
    #     for j in range(len(images)):
    #         print images[i][j].shape

    h_sizes = [im.shape[0] for im in zip(*images)[0]]
    w_sizes = [im.shape[1] for im in images[0]]

    h_total_size = np.sum(h_sizes) + margin * (len(images) - 1)
    w_total_size = np.sum(w_sizes) + margin * (len(images) - 1)

    collage = np.ones([h_total_size, w_total_size, 3]) * 255
    for i in range(len(images)):
        for j in range(len(images)):
            top_left_corner_h = int(np.sum(h_sizes[:j]) + j * margin)
            top_left_corner_w = int(np.sum(w_sizes[:i]) + i * margin)
            bottom_right_corner_h = int(top_left_corner_h + h_sizes[j])
            bottom_right_corner_w = int(top_left_corner_w + w_sizes[i])

            if [i, j] == input_spot:
                collage[top_left_corner_h - margin/2: bottom_right_corner_h + margin/2,
                        top_left_corner_w - margin/2: bottom_right_corner_w + margin/2,
                        :] = [255, 0, 0]
            collage[top_left_corner_h:bottom_right_corner_h, top_left_corner_w:bottom_right_corner_w] = images[j][i]

    return collage


def generate_images_for_collage(gan, input_tensor, scales, must_divide):
    # NOTE: scales here is different from in the other funcs: here we only need 1d scales.
    # Prepare output images list
    output_images = [[[None] for _ in range(len(scales))] for _ in range(len(scales))]

    # Run over all scales and test the network for each one
    for i, scale_h in enumerate(scales):
        for j, scale_w in enumerate(scales):
            # t = time()
            if scale_h == 2.0:
                scale_h = 1.5
            if scale_h == 0.4:
                scale_h = 0.66
            output_images[i][j] = test_one_scale(gan, input_tensor, [scale_h, scale_w], must_divide)
            # print time() - t
    return output_images


def retarget_video(gan, input_tensor, scales, must_divide, output_dir_path, name):
    max_scale = np.max(np.array(scales))
    frame_shape = np.uint32(np.array(input_tensor.shape[2:]) * max_scale)
    frame_shape[0] += (frame_shape[0] % 2)
    frame_shape[1] += (frame_shape[1] % 2)
    frames = np.zeros([len(scales), frame_shape[0], frame_shape[1], 3])
    for i, (scale_h, scale_w) in enumerate(scales):
        output_image = test_one_scale(gan, input_tensor, [scale_h, scale_w], must_divide)
        frames[i, 0:output_image.shape[0], 0:output_image.shape[1], :] = output_image
    writer = FFmpegWriter(output_dir_path + '/vid.mp4', verbosity=1, outputdict={'-b': '30000000', '-r': '100.0'})

    for i, _ in enumerate(scales):
        for j in range(3):
            writer.writeFrame(frames[i, :, :, :])
    writer.close()


def define_video_scales():
    max_v = 2.2
    min_v = 0.1
    max_h = 2.2
    min_h = 0.1
    frames_per_resize = 10
    # max_v = 1.2
    # min_v = 0.8
    # max_h = 1.2
    # min_h = 0.8
    # frames_per_resize = 8

    x = np.concatenate([
                        np.linspace(1, max_v, frames_per_resize),
                        np.linspace(max_v, min_v, 2 * frames_per_resize),
                        np.linspace(min_v, max_v, 2 * frames_per_resize),
                        np.linspace(max_v, 1, frames_per_resize),
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, 1, 2 * frames_per_resize),
                        np.linspace(1, 1, 2 * frames_per_resize),
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, max_v, frames_per_resize),
                        np.linspace(max_v, min_v, 2 * frames_per_resize),
                        np.linspace(min_v, max_v, 2 * frames_per_resize),
                        np.linspace(max_v, 1, frames_per_resize),
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, max_v, frames_per_resize),
                        np.linspace(max_v, max_v, 2 * frames_per_resize),
                        np.linspace(max_v, min_v, 2 * frames_per_resize)])
    y = np.concatenate([
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, 1, 2 * frames_per_resize),
                        np.linspace(1, 1, 2 * frames_per_resize),
                        np.linspace(1, 1, frames_per_resize),
                        np.linspace(1, max_h, frames_per_resize),
                        np.linspace(max_h, min_h, 2 * frames_per_resize),
                        np.linspace(min_h, max_h, 2 * frames_per_resize),
                        np.linspace(max_h, 1, frames_per_resize),
                        np.linspace(1, max_h, frames_per_resize),
                        np.linspace(max_h, min_h, 2 * frames_per_resize),
                        np.linspace(min_h, max_h, 2 * frames_per_resize),
                        np.linspace(max_h, 1, frames_per_resize),
                        np.linspace(1, max_h, frames_per_resize),
                        np.linspace(max_h, max_h, frames_per_resize),
                        np.linspace(max_h, min_h, 2 * frames_per_resize),
                        np.linspace(min_h, min_h, 2 * frames_per_resize)])

    return zip(x, y)


def generate_collage_and_outputs(conf, gan, input_tensor):
    #scales = [1.0, 0.75]
    # scales = [2.0, 1.25, 1.0, 0.66, 0.33]
    # scales = [1.8, 1.7, 1.6, 1.5, 1.35, 1.2, 1.0, 0.9, 0.8, 0.7, 0.66, 0.6, 0.5, 0.4, 0.33]
    input_spot = [2, 2]

    scales = [2.0, 1.0, 0.4]
    input_spot = [1, 1]



    output_images = generate_images_for_collage(gan, input_tensor, scales, conf.must_divide)

    for i in range(len(output_images)):
        for j in range(len(output_images)):
            Image.fromarray(output_images[i][j], 'RGB').save(conf.output_dir_path + '/test_%d_%d.png' % (i, j))

    output_images[input_spot[0]][input_spot[1]] = util.tensor2im(input_tensor)

    collage = concat_images(output_images, margin=10, input_spot=input_spot)

    Image.fromarray(np.uint8(collage), 'RGB').save(conf.output_dir_path + '/test_collage.png')


def _make_homography_mask(in_mask, target_size, rand_h):
    theta = homography_based_on_top_corners_x_shift(rand_h)
    target_size4d = torch.Size([in_mask.shape[0], in_mask.shape[1], target_size[0], target_size[1]])
    grid = homography_grid(theta.expand(in_mask.shape[0], -1, -1), target_size4d)
    out = f.grid_sample(in_mask, grid, mode='bilinear', padding_mode='border')
    return out


def test_homo(conf, gan, input_tensor, snapshot_iter, must_divide=8):



    transform = GeoTransform().forward

    # scale_range = range(3, 10, 2)
    # scale_range = range(6, 18, 2)
    scale_range = [10]

    # shift_range = range(-4, 5)
    shift_range = range(-8, 10)


    total = (len(scale_range)*len(shift_range))**2

    ind = 0
    for scale1 in scale_range:
        for scale2 in scale_range:
            scale = [0.1*scale1, 0.1*scale2]
            for shift1 in shift_range:
                for shift2 in shift_range:
                    ind += 1
                    shifts = (0.1 * shift1, 0.1 * shift2)
                    sz = input_tensor.shape
                    out_pad = np.uint8(255*np.ones([np.uint32(np.floor(sz[2]*scale[0])), np.uint32(np.floor(3*sz[3]*scale[1])), 3]))

                    pad_l = np.abs(np.int(np.ceil(sz[3] * shifts[0])))
                    pad_r = np.abs(np.int(np.ceil(sz[3] * shifts[1])))

                    in_mask = torch.zeros(sz[0], sz[1], sz[2], pad_l + sz[3] + pad_r).cuda()
                    input_for_regular = torch.zeros(sz[0], sz[1], sz[2], pad_l + sz[3] + pad_r).cuda()


                    in_size = in_mask.shape[2:]

                    out_size = (np.uint32(np.floor(scale[0] * in_size[0] * 1.0 / must_divide) * must_divide),
                                np.uint32(np.floor(scale[1] * in_size[1] * 1.0 / must_divide) * must_divide))



                    if pad_r > 0:
                        in_mask[:,:, :, pad_l:-pad_r] = torch.ones_like(input_tensor)
                        input_for_regular[:, :, :, pad_l:-pad_r] = input_tensor
                    else:
                        in_mask[:, :, :, pad_l:] = torch.ones_like(input_tensor)
                        input_for_regular[:, :, :, pad_l:] = input_tensor

                    out = test_one_scale(gan, input_tensor, out_size, conf.must_divide, affine=shifts, return_tensor=True, size_instead_scale=True)
                    regular = transform(input_tensor, out_size, shifts)
                    out_mask = _make_homography_mask(in_mask, out_size, shifts)

                    out = util.tensor2im(out_mask * out + 1 - out_mask)
                    regular_out = util.tensor2im(out_mask * regular + 1 - out_mask)
                    # out_pad[:, sz[3] - pad_l:  sz[3] - pad_l + out_size[1], :] = out
                    shift_str = "{1:0{0}d}_{3:0{2}d}".format(2 if shift1>=0 else 3, shift1, 2 if shift2>=0 else 3, shift2)

                    # out = np.rot90(out, 3)
                    # regular_out = np.rot90(regular_out, 3)

                    Image.fromarray(out, 'RGB').save(conf.output_dir_path + '/scale_%02d_%02d_transform %s_ingan_iter_%03dk.png' % (scale1, scale2, shift_str, snapshot_iter))
                    Image.fromarray(regular_out, 'RGB').save(conf.output_dir_path + '/scale_%02d_%02d_transform %s_ref_iter_%03dk.png' % (scale1, scale2, shift_str, snapshot_iter))
                    print ind, '/', total, 'scale:', scale, 'shift:', shifts


def main():

    name_infile_ckptfile_list = [
                                 # ['farm_house_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'farm_house_s.png', '/farm_house_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_16_07_59'],
                                 # ['birds_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'birds_s.png', '/birds_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_08_35'],
                                 # ['statue_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'statue_s.png', '/statue_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_19_17'],
                                 # ['buffaloes_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'buffaloes_s.png', '/buffaloes_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_18_27'],
                                 # ['bull_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'bull.png', '/bull_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_16_07_41'],
                                 # ['carrots_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'carrots_s.png', '/carrots_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_16_07_23'],
                                 # ['cab_building_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'cab_building_s.png', '/cab_building_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_10_25'],
                                 # ['capitol_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'capitol.png', '/capitol_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_13_22'],
                                 # ['house_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'house.jpg', '/house_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_16_05_55'],
                                 # ['fruits_ss_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'fruits_ss.png', '/experiment_old_code_with_homo_2/results/fruits_ss_geo_new_pad_Mar_16_18_00_17'],
                                 # ['yard_house_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'yard_house_s.png', '/yard_house_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_08_20'],
                                 # ['panorama_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'panorama_s.png', '/panorama_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_16_06_15'],
                                 # ['rome_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20', 'rome_s.png', '/rome_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_09_19'],
                                 # ['train_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric', 'train.png', '/train_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_08_59'],
                                 # ['building_s_L1_Dfactor_14_WeightsEqualeThenFine_23_LRdecay_23', 'building_s.png', '/building_s_L1_Dfactor_14_WeightsEqualeThenFine_23_LRdecay_23_Nov_02_18_04_15'],
                                 # ['china_soldiers_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G', 'china_soldiers.png', '/china_soldiers_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_09_46_09'],
                                 # ['corn_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_10_29_00', 'corn.png', '/corn_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_10_29_00'],
                                 # ['fruits_ss_256_COARSE2FINE_extraInv_2_30_until60_killReconstruct_20', 'fruits_ss.png', '/fruits_ss_256_COARSE2FINE_extraInv_2_30_until60_killReconstruct_20_Oct_24_12_35_33']
                                 # ['sapa_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G', 'sapa.png', '/sapa_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_09_44_59'],
                                 # ['sushi_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G', 'sushi.png', '/sushi_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_07_47_39'],
                                 # ['nkorea_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G', 'nkorea.png', '/nkorea_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_07_48_00']
                                 # ['birds_s_ablation_no_rec', 'birds_s.png', '/birds_s_ablation_no_multiscale_Nov_12_14_57_47']
                                 # ['birds_s_ablation_no_multiscale', 'birds_s.png', '/birds_s_ablation_no_rec_Nov_12_16_56_14']
                                 # ['guitar3_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G', 'guitar3.png', '/guitar3_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_07_18_52_46']
                                 # ['penguins', 'penguins.png', '/penguins_Nov_13_16_26_14'],
                                 # ['fruits_resh', 'fruits_ssrr.png', '/emojis3_Feb_27_10_02_18']
                                 # ['fruits', 'fruits_ss.png', '/fruits_ss_fig1_Mar_12_12_20_51'],
                                 # ['fruits_resh', 'fruits_ssr.png', '/fruits_ss_ingan_Feb_28_03_42_12'],
                                 # ['fruits_resh', 'fruits_ssrrr.png', '/fruits_ss_noise_sig_3_Mar_01_05_22_52'],
                                 # ['bull_resh_1', 'bull_n_1.png', '/bull_reshuffle_1_Feb_28_12_37_20'],
                                 # ['bull_resh_2', 'bull_n_2.png', '/bull_reshuffle_1_Feb_28_12_37_20'],
                                 # ['bull_resh_3', 'bull_n_3.png', '/bull_reshuffle_1_Feb_28_12_37_20'],
                                 # ['bull_resh_4', 'bull_n_4.png', '/bull_reshuffle_1_Feb_28_12_37_20'],
                                 # ['bull_resh_5', 'bull_n_5.png', '/bull_reshuffle_1_Feb_28_12_37_20'],
                                 # ['farm', 'farm_house_s.png', '/farm_house_s_fig1_Mar_12_12_21_56'],
                                 # ['emojis2', 'emojis2.png', '/emojis2_Nov_18_16_38_21'],
                                 # ['bull_resh', 'bull_nn.png', '/bull_gini_mask_noise_crop_coarse_gauss_Mar_05_23_01_42'],
                                 #   ['bull_resh', 'bull_nn.png', '/bull_gini_mask_noise_crop_coarse_pink_2_Mar_04_00_58_36'],
                                 # ['emojis3', 'emojis3.png', '/emojis3_Nov_23_09_59_59']
                                 # ['farm_house_plethora', 'farm_house_s.png', '/farm_house_to_resume_Mar_18_20_41_37'],
                                 # ['building_plethora', 'building_s.png', '/building_s_plethora_75_Mar_18_04_15_25'],
                                 # ['fish_plethora', 'input/fish.png', '/fish_plethora_75_Mar_18_03_36_25'],
                                 # ['blueman_plethora', 'input/blueman.png', '/blueman_plethora_75_Mar_17_20_49_11'],
                                 # ['cab_building_plethora', 'cab_building_s.png', '/cab_building_s_plethora_75_Mar_18_14_11_41'],
                                 # ['ny_plethora', 'textures/ny.png', '/ny_texture_synth_Mar_19_04_51_14'],
                                 # ['ny_plethora_side', 'side/ny.png', '/ny_geo_side_Mar_20_13_25_40'],
                                 # ['fruits_plethora_side', 'side/fruits_ss.png', '/fruits_ss_geo_side_Mar_20_13_25_40'],
                                 # ['umbrella', 'umbrella.png', '/umbrella'],
                                 # ['quilt', 'quilt.png', '/quilt'],
                                 ['elect', 'elect.png', '/metal_circles_Apr_09_20_38_53']

    ]
    # name_infile_ckptfile_list = [['nrid_%d_075' % i, 'scaled_nird/ours_%d_scaled.jpg' % i, 'ours_%d' % i]
    #                              for i in range(1, 36)]


    snapshot_iters =[75000]
    n_files = len(name_infile_ckptfile_list)

    for i, (name, input_image_path, test_params_path) in enumerate(name_infile_ckptfile_list):
        for snapshot_iter in snapshot_iters:

            conf = Config().parse(create_dir_flag=False)
            conf.name = 'TEST_' + name #+ '_iter_%dk' % (snapshot_iter / 1000)
            conf.output_dir_path = util.prepare_result_dir(conf)
            conf.input_image_path = [os.path.dirname(os.path.abspath(__file__)) + '/' + input_image_path]
            conf.test_params_path = os.path.dirname(os.path.abspath(__file__)) + '/results/' + test_params_path + '/checkpoint_%07d.pth.tar' % snapshot_iter
            # conf.test_params_path = os.path.dirname(os.path.abspath(__file__)) + test_params_path + '/checkpoint_%07d.pth.tar' % snapshot_iter
            gan = InGAN(conf)

            try:
                gan.resume(conf.test_params_path)
                [input_tensor] = util.read_data(conf)

                retarget_video(gan, input_tensor, define_video_scales(), 8, conf.output_dir_path, conf.name)

                generate_collage_and_outputs(conf, gan, input_tensor)

                # test_homo(conf, gan, input_tensor, snapshot_iter / 1000)

                print 'Done with %s (%d/%d), iter %dk' % (input_image_path, i, n_files, snapshot_iter )

            except KeyboardInterrupt:
                raise
            except Exception as e:
                # print 'Something went wrong with %s (%d/%d), iter %dk' % (input_image_path, i, n_files, snapshot_iter)
                print_exc()


if __name__ == '__main__':
    main()
