from __future__ import print_function

import numpy as np


def make_scene_script(script_name, min_v, max_v, min_h, max_h, max_t, repeat, show_input=True, frames_per_resize=10):
    l = np.linspace

    if script_name == 'vertical_grow_shrink':
        size_v = np.concatenate([
            l(1, max_v, frames_per_resize),
            l(max_v, min_v, 2 * frames_per_resize),
            l(min_v, 1, frames_per_resize)])
        size_h = np.concatenate([
            l(1, 1, frames_per_resize),
            l(1, 1, 2 * frames_per_resize),
            l(1, 1, frames_per_resize)])
        shift_l = [0 for _ in size_v]
        shift_r = [0 for _ in size_v]

    elif script_name == 'horizontal_grow_shrink':
        size_v = np.concatenate([
            l(1, 1, frames_per_resize),
            l(1, 1, 2 * frames_per_resize),
            l(1, 1, frames_per_resize)])
        size_h = np.concatenate([
            l(1, max_h, frames_per_resize),
            l(max_h, min_h, 2 * frames_per_resize),
            l(min_h, 1, frames_per_resize)])
        shift_l = [0 for _ in size_v]
        shift_r = [0 for _ in size_v]

    elif script_name == 'horizontal_grow_shrink_slow':
        size_v = np.concatenate([
            l(1, 1, 2 *frames_per_resize),
            l(1, 1, 2 * frames_per_resize),
            l(1, 1, frames_per_resize)])
        size_h = np.concatenate([
            l(1, max_h, 2 * frames_per_resize),
            l(max_h, min_h, 2 * frames_per_resize),
            l(min_h, 1, frames_per_resize)])
        shift_l = [0 for _ in size_v]
        shift_r = [0 for _ in size_v]

    elif script_name == '2d_grow_shrink':
        size_v = np.concatenate([
            l(1, max_v, frames_per_resize),
            l(max_v, min_v, 2 * frames_per_resize),
            l(min_v, 1, frames_per_resize)])
        size_h = np.concatenate([
            l(1, max_h, frames_per_resize),
            l(max_h, min_h, 2 * frames_per_resize),
            l(min_h, 1, frames_per_resize)])
        shift_l = [0 for _ in size_v]
        shift_r = [0 for _ in size_v]

    elif script_name == 'resize_round':
        size_v = np.concatenate([
            l(1, 1, frames_per_resize),
            l(1, max_v, frames_per_resize),
            l(max_v, max_v, 2 * frames_per_resize),
            l(max_v, min_v, 2 * frames_per_resize),
            l(min_v, 1, frames_per_resize)])
        size_h = np.concatenate([
            l(1, max_h, frames_per_resize),
            l(max_h, max_h, frames_per_resize),
            l(max_h, min_h, 2 * frames_per_resize),
            l(min_h, min_h, 2 * frames_per_resize),
            l(min_h, 1, frames_per_resize)])
        shift_l = [0 for _ in size_v]
        shift_r = [0 for _ in size_v]

    elif script_name == 'special_resize_round':
        size_v = np.concatenate([
            l(1, 1, frames_per_resize/2),
            l(1, max_v, frames_per_resize),
            l(max_v, max_v, frames_per_resize),
            l(max_v, max_v, 2 * frames_per_resize),
            l(max_v, min_v, 2 * frames_per_resize),
            l(min_v, 1, frames_per_resize)])

        size_h = np.concatenate([
            l(1, max_h/2, frames_per_resize/2),
            l(max_h/2, max_h/2, frames_per_resize),
            l(max_h/2, max_h, frames_per_resize),
            l(max_h, min_h, 2 * frames_per_resize),
            l(min_h, min_h, 2 * frames_per_resize),
            l(min_h, 1, frames_per_resize)])
        shift_l = [0 for _ in size_v]
        shift_r = [0 for _ in size_v]

    elif script_name == 'special_zoom':
        size_v = np.concatenate([
            l(1, max_v, frames_per_resize),
            l(max_v, min_v, frames_per_resize),
            l(min_v, 1, frames_per_resize)])
        size_h = np.concatenate([
            l(1, max_v, frames_per_resize),
            l(max_v, min_v, frames_per_resize),
            l(min_v, 1, frames_per_resize)])
        shift_l = [0 for _ in size_v]
        shift_r = [0 for _ in size_v]

    elif script_name == 'affine_dance':
        shift_l = np.concatenate([
            l(0, max_t, frames_per_resize),
            l(max_t, - max_t, 2 * frames_per_resize),
            l(- max_t, 0, frames_per_resize)])
        shift_r = np.concatenate([
            l(0, - max_t, frames_per_resize),
            l(- max_t, max_t, 2 * frames_per_resize),
            l(max_t, 0, frames_per_resize)])
        size_v = [1for _ in shift_l]
        size_h = [1 for _ in shift_l]

    elif script_name == 'trapezoids':
        shift_l = np.concatenate([
            l(0, max_t, frames_per_resize),
            l(max_t, - max_t, 2 * frames_per_resize),
            l(- max_t, max_t, 2 * frames_per_resize),
            l(max_t, 0, frames_per_resize)])
        shift_r = np.concatenate([
            l(0, max_t, frames_per_resize),
            l(max_t, - max_t, 2 * frames_per_resize),
            l(- max_t, max_t, 2 * frames_per_resize),
            l(max_t, 0, frames_per_resize)])
        size_v = [1for _ in shift_l]
        size_h = [1 for _ in shift_l]

    elif script_name == 'trapezoids_vresize':
        shift_l = np.concatenate([
            l(0, max_t, frames_per_resize),
            l(max_t, - max_t, 2 * frames_per_resize),
            l(- max_t, max_t, 2 * frames_per_resize),
            l(max_t, 0, frames_per_resize)])
        shift_r = np.concatenate([
            l(0, max_t, frames_per_resize),
            l(max_t, - max_t, 2 * frames_per_resize),
            l(- max_t, max_t, 2 * frames_per_resize),
            l(max_t, 0, frames_per_resize)])
        size_v = np.concatenate([
            l(1, max_v, frames_per_resize),
            l(max_v, 1, frames_per_resize),
            l(1, max_v, frames_per_resize),
            l(max_v, 1, frames_per_resize),
            l(1, max_v, frames_per_resize),
            l(max_v, 1, frames_per_resize),
        ])
        size_h = np.concatenate([
              l(1, 1, 6*frames_per_resize)])

    elif script_name == 'flicker':
        size_h = np.concatenate([
            l(1, 1, 6 * frames_per_resize)])
        size_v = size_h
        shift_l = np.concatenate([
            l(max_t, max_t, frames_per_resize),
            l(-max_t, -max_t, frames_per_resize),
            l(max_t, max_t, frames_per_resize),
            l(-max_t, -max_t, frames_per_resize),
            l(max_t, max_t, frames_per_resize),
            l(-max_t, -max_t, frames_per_resize),])
        shift_r = np.concatenate([
            l(-max_t, -max_t, frames_per_resize),
            l(max_t, max_t, frames_per_resize),
            l(-max_t, -max_t, frames_per_resize),
            l(max_t, max_t, frames_per_resize),
            l(-max_t, -max_t, frames_per_resize),
            l(max_t, max_t, frames_per_resize)])

    elif script_name == 'homography':
        size_h = np.concatenate([
            l(1, 1, 6 * frames_per_resize)])
        size_v = size_h
        shift_l = np.concatenate([
            l(0, max_t, frames_per_resize),
            l(max_t, max_t, frames_per_resize),
            l(max_t, - max_t, 2 * frames_per_resize),
            l(- max_t, - max_t, 2 * frames_per_resize),
            l(- max_t, 0, frames_per_resize)])
        shift_r = np.concatenate([
            l(0, 0, frames_per_resize),
            l(0, max_t, frames_per_resize),
            l(max_t, max_t, 2 * frames_per_resize),
            l(max_t, - max_t, 2 * frames_per_resize),
            l(- max_t, 0, frames_per_resize)])



    elif script_name == 'random':
        stops = np.random.rand(10, 4) * np.array([max_v-min_v, max_h-min_h, 2*max_t, 2*max_t])[None, :] + np.array([min_v, min_h, -max_t, -max_t])[None, :]
        stops = np.vstack([stops, [1, 1, 0, 0]])
        print(stops)

        size_v = np.concatenate([l(stop_0[0], stop_1[0], frames_per_resize)
                                 for stop_0, stop_1 in zip(np.vstack(([1, 1, 0, 0], stops)), stops)])

        size_h = np.concatenate([l(stop_0[1], stop_1[1], frames_per_resize)
                                 for stop_0, stop_1 in zip(np.vstack(([1, 1, 0, 0], stops)), stops)])

        shift_l = np.concatenate([l(stop_0[2], stop_1[2], frames_per_resize)
                                 for stop_0, stop_1 in zip(np.vstack(([1, 1, 0, 0], stops)), stops)])

        shift_r = np.concatenate([l(stop_0[3], stop_1[3], frames_per_resize)
                                 for stop_0, stop_1 in zip(np.vstack(([1, 1, 0, 0], stops)), stops)])

    elif script_name == 'random_trapezoids':
        stops_l = np.random.rand(11) * 2 * max_t - max_t
        stops_l[-1] = 0
        stops_r = np.random.rand(11) * max_t * (stops_l / np.abs(stops_l))
        stops = zip(stops_l, stops_r)
        print(stops)

        size_h = np.concatenate([
            l(1, 1, 20 * frames_per_resize)])
        size_v = size_h

        shift_l = np.concatenate([l(stop_0[0], stop_1[0], frames_per_resize)
                                  for stop_0, stop_1 in zip(np.vstack(([0, 0], stops)), stops)])

        shift_r = np.concatenate([l(stop_0[1], stop_1[1], frames_per_resize)
                                  for stop_0, stop_1 in zip(np.vstack(([0, 0], stops)), stops)])


    return [[-1, -1, -1, -1]] * 20 + zip(size_v, size_h, shift_l, shift_r) * repeat if show_input else zip(size_v, size_h, shift_l, shift_r) * repeat


INPUT_DICT = {
                 'fruits': ['fruits_ss.png', '/experiment_old_code_with_homo_2/results/fruits_ss_geo_new_pad_Mar_16_18_00_17/checkpoint_0075000.pth.tar'],
                 'farm_house': ['farm_house_s.png', '/results/farm_house_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_16_07_59/checkpoint_0050000.pth.tar'],
                 'cab_building': ['cab_building_s.png', '/results/cab_building_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_10_25/checkpoint_0065000.pth.tar'],
                 'capitol': ['capitol.png', '/results/capitol_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_13_22/checkpoint_0055000.pth.tar'],
                 'rome': ['rome_s.png', '/results/rome_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_09_19/checkpoint_0045000.pth.tar'],
                 'soldiers': ['china_soldiers.png', '/results/china_soldiers_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_09_46_09/checkpoint_0075000.pth.tar'],
                 'corn': ['corn.png', '/results/corn_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_10_29_00/checkpoint_0075000.pth.tar'],
                 'sushi': ['sushi.png', '/results/sushi_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_07_47_39/checkpoint_0075000.pth.tar'],
                 'penguins': ['penguins.png', '/results/penguins_Nov_13_16_26_14/checkpoint_0075000.pth.tar'],
                 'emojis': ['emojis3.png', '/results/emojis3_Nov_23_09_59_59/checkpoint_0075000.pth.tar'],
                 'fish': ['input/fish.png', '/results/fish_plethora_75_Mar_18_03_36_25/checkpoint_0075000.pth.tar'],
                 'ny': ['textures/ny.png', '/results/ny_texture_synth_Mar_19_04_51_14/checkpoint_0075000.pth.tar'],
                 'metal_circles': ['metal_circles.jpg', '/results/metal_circles_Mar_26_20_04_11/checkpoint_0075000.pth.tar'],
                 'quilt': ['quilt.png', '/results/quilt/checkpoint_0075000.pth.tar'],
                 'sapa': ['sapa.png', '/results/sapa_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_09_44_59/checkpoint_0075000.pth.tar'],
                 'nkorea': ['nkorea.png', '/results/nkorea_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_07_48_00/checkpoint_0075000.pth.tar'],
                 'wood': ['wood.png', '/results/wood/checkpoint_0075000.pth.tar'],
                 'starry': ['starry.png', '/results/starry/checkpoint_0075000.pth.tar'],
                 'umbrella': ['umbrella.png', '/results/umbrella/checkpoint_0075000.pth.tar'],
                 'fruits_old': ['fruits_ss.png', '/results/fruits_ss_256_COARSE2FINE_extraInv_2_30_until60_killReconstruct_20_Oct_24_12_35_33/checkpoint_0040000.pth.tar'],
                 'peacock': ['scaled_nird/ours_1_scaled.jpg', '/results/ours_1/checkpoint_0050000.pth.tar'],
                 'windows': ['scaled_nird/ours_2_scaled.jpg', '/results/ours_2/checkpoint_0050000.pth.tar'],
                 'light_house': ['scaled_nird/ours_23_scaled.jpg', '/results/ours_23/checkpoint_0050000.pth.tar'],
                 'hats': ['scaled_nird/ours_26_scaled.jpg', '/results/ours_26/checkpoint_0050000.pth.tar'],
                 'nature': ['scaled_nird/ours_32_scaled.jpg', '/results/ours_32/checkpoint_0050000.pth.tar'],

}
