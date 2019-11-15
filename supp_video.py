from __future__ import print_function

import util
from InGAN import InGAN
from configs import Config
from skvideo.io import FFmpegWriter
import os
from non_rect import *
from SceneScripts import *


FRAME_SHAPE = [500, 1000]
MUST_DIVIDE = 8
VIDEO_SCRIPT = [  # [nameses, script_name, script_params=(min_v, max_v, min_h, max_h, max_t, repeat)]
[[['fruits'], ['fruits_old'], ['fruits_old'], ['fruits'], ['fruits']], ['horizontal_grow_shrink_slow', 'vertical_grow_shrink', 'resize_round', 'affine_dance', 'random'], [[0.55, None, 0.55, None, None, 1], [0.3, 1.8, 0.3, 2.0, None, 1, False], [0.3, 1.8, 0.3, 2.0, None, 1, False], [None, None, None, None, 0.45, 1, False], [0.3, 1.3, 0.3, 1.6, 0.45, 1, False]]],
['farm_house', 'special_resize_round', [0.45, None,  0.45, None, None, 2]],
['cab_building', 'resize_round', [0.5, None, 0.3, 2.5, None, 2]],
['rome', 'horizontal_grow_shrink', [0.3, None, 0.3, None, None, 3]],
[[['peacock', 'windows']], 'resize_round', [0.5, 2, 0.5, 1.75, None, 3]],
[[['soldiers', 'penguins']], 'horizontal_grow_shrink', [0.3, None, 0.3, None, None, 3]],
[[['nkorea', 'sapa']], 'horizontal_grow_shrink', [0.15, None, 0.15, None, None, 3]],
[[['quilt']] * 5, ['horizontal_grow_shrink', 'vertical_grow_shrink', 'resize_round', 'affine_dance', 'random'], [[0.55, None, 0.55, None, None, 1], [0.3, None, 0.3, None, None, 1, False], [0.3, None, 0.3, None, None, 1, False], [None, None, None, None, 0.45, 1, False], [0.6, 1.6, 0.6, 1.75, 0.55, 1, False]]],
[[['umbrella'], ['umbrella'], ['umbrella']], ['horizontal_grow_shrink', 'resize_round', 'trapezoids'], [[0.55, None, 0.55, None, None, 1], [0.55, None, 0.55, None, None, 1, False], [1, 1, 0.8, 1.2, 0.3, 1, False]]],
[[['metal_circles']] * 5, ['vertical_grow_shrink', 'random'], [[0.15, None, 0.55, None, None, 2], [0.15, 1.8, 0.15, 1.45, 0.55, 1, False]]],
[[['fish'], ['fish']], ['affine_dance', 'random'], [[1, 1, 1, 1, 0.4, 1], [1, 1, 1, 1, 0.5, 1, False]]],
['wood', 'special_zoom', [0.3, None, 0.3, None, None, 2]],
['ny', 'affine_dance', [None, None, None, None, 0.3, 2]],
['sushi', 'resize_round', [0.5, None, 0.3, None, None, 1]],
]


def generate_one_frame(gan, input_tensor, frame_shape, scale, geo_shifts, center):
    with torch.no_grad():
        base_sz = input_tensor.shape
        in_size = base_sz[2:]
        out_pad = np.uint8(np.zeros([frame_shape[0], frame_shape[1], 3]))

        if scale[0] == -1:
            output_tensor = [None, input_tensor]
            out_mask = torch.ones_like(output_tensor[1])
            out_size = in_size

        else:
            out_mask, out_size = prepare_geometric(base_sz, scale, geo_shifts)

            output_tensor, _, _ = gan.test(input_tensor=input_tensor,
                                           input_size=in_size,
                                           output_size=out_size,
                                           rand_affine=geo_shifts,
                                           run_d_pred=False,
                                           run_reconstruct=False)

        out = out_mask * output_tensor[1] - 1 + out_mask
        margin = np.uint16((frame_shape - np.array(out_size)) / 2) if center else [0, 0]
        out_pad[margin[0]:margin[0] + out_size[0], margin[1]:margin[1] + out_size[1], :] = util.hist_match(util.tensor2im(out), util.tensor2im(input_tensor), util.tensor2im(out_mask))
        return out_pad


def generate_one_scene(gan, input_tensor, scene_script, frame_shape, center):
    frames = []
    for i, (scale_v, scale_h, shift_l, shift_r) in enumerate(scene_script):
        output_image = generate_one_frame(gan, input_tensor, frame_shape, [scale_v, scale_h], [shift_l, shift_r], center)
        frames.append(output_image)
    return np.stack(frames, axis=0)


def generate_full_video(video_script, frame_shape):
    conf = Config().parse(create_dir_flag=False)
    conf.name = 'supp_vid'
    conf.output_dir_path = util.prepare_result_dir(conf)
    n_scenes = len(video_script)

    for i, (nameses, scene_script_names, scene_script_params) in enumerate(video_script):
        if not isinstance(nameses, list):
            nameses = [[nameses]]
        if not isinstance(scene_script_names, list):
            scene_script_names = [scene_script_names]
            scene_script_params = [scene_script_params]
        scenes = []
        for names, scene_script_name, scene_script_param in zip(nameses, scene_script_names, scene_script_params):
            partial_screen_scenes = []

            for name in names:
                conf.input_image_path = [os.path.dirname(os.path.abspath(__file__)) + '/' + INPUT_DICT[name][0]]
                conf.test_params_path = os.path.dirname(os.path.abspath(__file__)) + INPUT_DICT[name][1]
                gan = InGAN(conf)
                gan.G.load_state_dict(torch.load(conf.test_params_path, map_location='cuda:0')['G'])
                [input_tensor] = util.read_data(conf)

                cur_frame_shape = frame_shape[:]
                concat_axis = 2 if scene_script_name == 'resize_round' else 1
                if len(names) > 1:
                    cur_frame_shape[concat_axis - 1] /= 2

                cur_scene_script_param = scene_script_param[:]
                if scene_script_param[1] is None:
                    cur_scene_script_param[1] = cur_frame_shape[0] * 1.0 / input_tensor.shape[2]
                    print('max scale vertical:', cur_scene_script_param[1])
                if cur_scene_script_param[3] is None:
                    cur_scene_script_param[3] = cur_frame_shape[1] * 1.0 / input_tensor.shape[3]
                    print('max scale horizontal:', cur_scene_script_param[3])
    irint(type(images))

                scene_script = make_scene_script(scene_script_name, *cur_scene_script_param)

                center = (cur_scene_script_param[4] is not None)


                scene = generate_one_scene(gan, input_tensor, scene_script, np.array([cur_frame_shape[0], cur_frame_shape[1]]), center)
                partial_screen_scenes.append(scene)

                print('Done with %s,  (scene %d/%d)' % (name, i + 1, n_scenes))


            scene = np.concatenate(partial_screen_scenes, axis=concat_axis) if len(partial_screen_scenes) > 1 else partial_screen_scenes[0]
            scenes.append(scene)

        scene = np.concatenate(scenes, axis=0)

        outputdict = {'-b:v': '30000000', '-r': '100.0',
                      '-vf': 'drawtext="text=\'Input image\':fontcolor=red:fontsize=48:x=(w-text_w)/2:y=(h-text_h)*7/8:enable=\'between(t,0,2)\'"',
                      '-preset': 'slow', '-profile:v': 'high444', '-level:v': '4.0', '-crf': '22'}
        if len(names) > 1:
            outputdict['-vf'] = 'drawtext="text=\'Input images\':fontcolor=red:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2.5:enable=\'between(t,0,2)\'"'

        if not scene_script_params[-1]:
            outputdict['-vf'] = 'drawtext="text=\'Input images\':fontcolor=red:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2.5:enable=\'between(t,0,0)\'"'

        writer = FFmpegWriter(conf.output_dir_path + '/vid%d_%s.mp4' % (i, '_'.join(names)), verbosity=1,
                              outputdict=outputdict)
        for frame in scene:
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
    out_mask = f.grid_sample(in_mask, grid, mode='bilinear', padding_mode='zeros')
    return out_mask, out_size


def main():
    generate_full_video(VIDEO_SCRIPT, FRAME_SHAPE)


if __name__ == '__main__':
    main()
