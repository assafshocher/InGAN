import numpy as np


def make_scene_script(script_name, min_v, max_v, min_h, max_h, max_t, repeat, frames_per_resize=10):
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
            l(1, max_v, frames_per_resize),
            l(max_v, min_v, 2 * frames_per_resize),
            l(min_v, 1, frames_per_resize)])
        shift_l = [0 for _ in size_v]
        shift_r = [0 for _ in size_v]

    elif script_name == '2d_grow_shrink':
        size_v = np.concatenate([
            l(1, 1, frames_per_resize),
            l(1, 1, 2 * frames_per_resize),
            l(1, 1, frames_per_resize)])
        size_h = np.concatenate([
            l(1, 1, frames_per_resize),
            l(1, 1, 2 * frames_per_resize),
            l(1, 1, frames_per_resize)])
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



    return zip(size_v, size_h, shift_l, shift_r) * repeat


INPUT_DICT = {
                 'fruits': ['fruits_ss.png', '/experiment_old_code_with_homo_2/results/fruits_ss_geo_new_pad_Mar_16_18_00_17/checkpoint_0075000.pth.tar'],
                 'farm_house': ['farm_house_s.png', 'results/farm_house_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_16_07_59/checkpoint_0075000.pth.tar'],
                 'cab_building': ['cab_building_s.png', '/results/cab_building_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_10_25/checkpoint_0075000.pth.tar'],
                 'capitol': ['capitol.png', '/results/capitol_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_13_22/checkpoint_0075000.pth.tar'],
                 'rome': ['rome_s.png', '/results/rome_s_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_Nov_03_18_09_19/checkpoint_0045000.pth.tar'],
                 'soldiers': ['china_soldiers.png', '/results/china_soldiers_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_09_46_09/checkpoint_0075000.pth.tar'],
                 'corn': ['corn.png', '/results/corn_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_10_29_00/checkpoint_0075000.pth.tar'],
                 'sushi': ['sushi.png', '/results/sushi_L1_Dfactor_14_WeightsEqualeThenFine_25_LRdecay_20_curric_NOISE2G_Nov_05_07_47_39/checkpoint_0075000.pth.tar'],
                 'penguins': ['penguins.png', '/results/penguins_Nov_13_16_26_14/checkpoint_0075000.pth.tar'],
                 'emojis': ['emojis3.png', '/results/emojis3_Nov_23_09_59_59/checkpoint_0075000.pth.tar'],
                 'fish': ['input/fish.png', '/results/fish_plethora_75_Mar_18_03_36_25/checkpoint_0075000.pth.tar'],
                 'ny': ['textures/ny.png', '/results/ny_texture_synth_Mar_19_04_51_14/checkpoint_0075000.pth.tar'],
}
