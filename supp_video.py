import util
from InGAN import InGAN
from configs import Config
from skvideo.io import FFmpegWriter
import os
from non_rect import *
from SceneScripts import *


FRAME_SHAPE = [600, 1000]
MUST_DIVIDE = 8
VIDEO_SCRIPT = [  # [name, script_name, script_params=(min_v, max_v, min_h, max_h, max_t, repeat)]
    ['fruits', 'vertical_grow_shrink', (0.3, 2.0, 0.3, 2.0, None, 1)],
    ['fruits', 'resize_round', (0.3, 2.0, 0.3, 2.0, None, 1)],
    ['fruits', 'affine_dance', (None, None, None, None, 0.5, 1)],
    ['rome', 'horizontal_grow_shrink', (0.3, 2.0, 0.3, 1.75, None, 2)],
    ['ny', 'affine_dance', (None, None, None, None, 0.3, 2)],
]


def generate_one_frame(gan, input_tensor, frame_shape, scale, geo_shifts):
    with torch.no_grad():
        base_sz = input_tensor.shape
        in_size = base_sz[2:]
        out_pad = np.zeros(frame_shape + [3])

        out_mask, out_size = prepare_geometric(base_sz, scale, geo_shifts)

        output_tensor, _, _ = gan.test(input_tensor=input_tensor,
                                       input_size=in_size,
                                       output_size=out_size,
                                       rand_affine=geo_shifts,
                                       run_d_pred=False,
                                       run_reconstruct=False)

        out = out_mask * output_tensor[1] - 1 + out_mask
        margin = np.uint16((np.array(frame_shape) - np.array(out_size)) / 2)
        out_pad[margin[0]:margin[0] + out_size[0], margin[1]:margin[1] + out_size[1], :] = util.tensor2im(out)
        return out_pad


def generate_one_scene(gan, input_tensor, scene_script, frame_shape):
    frames = []
    for i, (scale_v, scale_h, shift_l, shift_r) in enumerate(scene_script):
        output_image = generate_one_frame(gan, input_tensor, frame_shape, [scale_v, scale_h], [shift_l, shift_r])
        frames.append(output_image)
    return frames


def generate_full_video(video_script, frame_shape):
    conf = Config().parse(create_dir_flag=False)
    conf.name = 'supp_vid'
    conf.output_dir_path = util.prepare_result_dir(conf)
    vid = []
    n_scenes = len(video_script)

    for i, (name, scene_script_name, scene_script_params) in enumerate(video_script):

        conf.input_image_path = [os.path.dirname(os.path.abspath(__file__)) + '/' + INPUT_DICT[name][0]]
        conf.test_params_path = os.path.dirname(os.path.abspath(__file__)) + INPUT_DICT[name][1]
        gan = InGAN(conf)
        gan.resume(conf.test_params_path)
        [input_tensor] = util.read_data(conf)

        scene_script = make_scene_script(scene_script_name, *scene_script_params)

        vid += generate_one_scene(gan, input_tensor, scene_script, frame_shape)

        print 'Done with %s,  (scene %d/%d)' % (name, i+1, n_scenes)

    writer = FFmpegWriter(conf.output_dir_path + '/vid.mp4', verbosity=1, outputdict={'-b': '30000000', '-r': '100.0'})
    for frame in vid:
        for j in range(3):
            writer.writeFrame(frame)
    writer.close()


def prepare_geometric(base_sz, scale, geo_shifts):
    pad_l = np.abs(np.int(np.ceil(base_sz[3] * geo_shifts[0])))
    pad_r = np.abs(np.int(np.ceil(base_sz[3] * geo_shifts[1])))
    in_mask = torch.zeros(base_sz[0], base_sz[1], base_sz[2], pad_l + base_sz[3] + pad_r).cuda()
    in_size = in_mask.shape[2:]
    out_size = (np.uint32(np.floor(scale[0] * in_size[0] * 1.0 / MUST_DIVIDE) * MUST_DIVIDE),
                np.uint32(np.floor(scale[1] * in_size[1] * 1.0 / MUST_DIVIDE) * MUST_DIVIDE))
    if pad_r > 0:
        in_mask[:, :, :, pad_l:-pad_r] = torch.ones(base_sz)
    else:
        in_mask[:, :, :, pad_l:] = torch.ones(base_sz)

    theta = homography_based_on_top_corners_x_shift(geo_shifts)
    target_size4d = torch.Size([in_mask.shape[0], in_mask.shape[1], out_size[0], out_size[1]])
    grid = homography_grid(theta.expand(in_mask.shape[0], -1, -1), target_size4d)
    out_mask = f.grid_sample(in_mask, grid, mode='bilinear', padding_mode='border')
    return out_mask, out_size





def main():
    generate_full_video(VIDEO_SCRIPT, FRAME_SHAPE)


if __name__ == '__main__':
    main()
