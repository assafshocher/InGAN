from __future__ import print_function

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
    h_sizes = [im.shape[0] for im in [image[0] for image in images]]
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
                collage[top_left_corner_h - margin//2: bottom_right_corner_h + margin//2,
                        top_left_corner_w - margin//2: bottom_right_corner_w + margin//2,
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
            output_images[i][j] = test_one_scale(gan, input_tensor, [scale_h, scale_w], must_divide)
    return output_images


def retarget_video(gan, input_tensor, scales, must_divide, output_dir_path):
    max_scale = np.max(np.array(scales))
    frame_shape = np.uint32(np.array(input_tensor.shape[2:]) * max_scale)
    frame_shape[0] += (frame_shape[0] % 2)
    frame_shape[1] += (frame_shape[1] % 2)
    frames = np.zeros([len(scales[0]), frame_shape[0], frame_shape[1], 3])
    for i, (scale_h, scale_w) in enumerate(zip(*scales)):
        output_image = test_one_scale(gan, input_tensor, [scale_h, scale_w], must_divide)
        frames[i, 0:output_image.shape[0], 0:output_image.shape[1], :] = output_image
    writer = FFmpegWriter(output_dir_path + '/vid.mp4', verbosity=1, outputdict={'-b': '30000000', '-r': '100.0'})

    for i, _ in enumerate(zip(*scales)):
        for j in range(3):
            writer.writeFrame(frames[i, :, :, :])
    writer.close()


def define_video_scales(scales):
    max_v, min_v, max_h, min_h = scales
    frames_per_resize = 10

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
    return x, y


def generate_collage_and_outputs(conf, gan, input_tensor):
    output_images = generate_images_for_collage(gan, input_tensor, conf.collage_scales, conf.must_divide)

    for i in range(len(output_images)):
        for j in range(len(output_images)):
            Image.fromarray(output_images[i][j], 'RGB').save(conf.output_dir_path + '/test_%d_%d.png' % (i, j))

    input_spot = conf.collage_input_spot
    output_images[input_spot[0]][input_spot[1]] = util.tensor2im(input_tensor)

    collage = concat_images(output_images, margin=10, input_spot=input_spot)

    Image.fromarray(np.uint8(collage), 'RGB').save(conf.output_dir_path + '/test_collage.png')


def _make_homography_mask(in_mask, target_size, rand_h):
    theta = homography_based_on_top_corners_x_shift(rand_h)
    target_size4d = torch.Size([in_mask.shape[0], in_mask.shape[1], target_size[0], target_size[1]])
    grid = homography_grid(theta.expand(in_mask.shape[0], -1, -1), target_size4d)
    out = f.grid_sample(in_mask, grid, mode='bilinear', padding_mode='border')
    return out


def test_homo(conf, gan, input_tensor, must_divide=8):
    shift_range = np.arange(conf.non_rect_shift_range[0], conf.non_rect_shift_range[1], conf.non_rect_shift_range[2])
    total = (len(conf.non_rect_scales)*len(shift_range))**2
    ind = 0
    for scale1 in conf.non_rect_scales:
        for scale2 in conf.non_rect_scales:
            scale = [scale1, scale2]
            for shift1 in shift_range:
                for shift2 in shift_range:
                    ind += 1
                    shifts = (shift1, shift2)
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
                    # regular = transform(input_tensor, out_size, shifts)
                    out_mask = _make_homography_mask(in_mask, out_size, shifts)

                    out = util.tensor2im(out_mask * out + 1 - out_mask)
                    # regular_out = util.tensor2im(out_mask * regular + 1 - out_mask)
                    # out_pad[:, sz[3] - pad_l:  sz[3] - pad_l + out_size[1], :] = out
                    shift_str = "{1:0{0}d}_{3:0{2}d}".format(2 if shift1>=0 else 3, int(10*shift1), 2 if shift2>=0 else 3, int(10*shift2))

                    # out = np.rot90(out, 3)
                    # regular_out = np.rot90(regular_out, 3)

                    Image.fromarray(out, 'RGB').save(conf.output_dir_path + '/scale_%02d_%02d_transform %s_ingan.png' % (int(10*scale1), int(10*scale2), shift_str))
                    # Image.fromarray(regular_out, 'RGB').save(conf.output_dir_path + '/scale_%02d_%02d_transform %s_ref.png' % (scale1, scale2, shift_str))
                    print(ind, '/', total, 'scale:', scale, 'shift:', shifts)


def main():
    conf = Config().parse(create_dir_flag=False)
    conf.name = 'TEST_' + conf.name
    conf.output_dir_path = util.prepare_result_dir(conf)
    gan = InGAN(conf)

    try:
        gan.resume(conf.test_params_path, test_flag=True)
        [input_tensor] = util.read_data(conf)

        if conf.test_video:
            retarget_video(gan, input_tensor, define_video_scales(conf.test_vid_scales), 8, conf.output_dir_path)
        if conf.test_collage:
            generate_collage_and_outputs(conf, gan, input_tensor)
        if conf.test_non_rect:
            test_homo(conf, gan, input_tensor)

        print('Done with %s' % conf.input_image_path)

    except KeyboardInterrupt:
        raise
    except Exception as e:
        # print 'Something went wrong with %s (%d/%d), iter %dk' % (input_image_path, i, n_files, snapshot_iter)
        print_exc()


if __name__ == '__main__':
    main()
