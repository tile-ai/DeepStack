from mosaic.arch import stacked_gpu_base, stacked_gpu_large_matrix, stacked_gpu_large_vector, stacked_gpu_high_l1, stacked_gpu_high_l2, stacked_gpu_high_noc, stacked_gpu_low_noc, stacked_gpu_reduced_sm, stacked_gpu_wgmma
from mosaic.noc.noc_config_set import torus_mesh_switch_1, torus_mesh_switch_2, torus_mesh_mesh_3, strong_torus_mesh_switch_4, weak_torus_mesh_switch_5, torus_mesh_switch_7, torus_mesh_switch_8, torus_mesh_switch_9

from mosaic.arch import H100_SCALED, H200_SCALED, H100, H200, B200
from mosaic.noc.noc_config_set import h100x32_strong, h100x32_medium
def arch_noc_combinations1():
    # arch_list = [stacked_gpu_base, stacked_gpu_large_matrix, stacked_gpu_large_vector, stacked_gpu_high_l1, stacked_gpu_high_l2, stacked_gpu_high_noc, stacked_gpu_low_noc]
    # noc_list = [torus_mesh_switch_1, torus_mesh_switch_2, torus_mesh_mesh_3, strong_strong_torus_mesh_switch_4, weak_weak_torus_mesh_switch_5]

    combinations = []



    arch_list = [stacked_gpu_base(), stacked_gpu_large_matrix(), stacked_gpu_large_vector(), stacked_gpu_high_l1(), stacked_gpu_high_l2()]
    noc_list = [torus_mesh_switch_1(), torus_mesh_switch_2(), torus_mesh_mesh_3(), torus_mesh_switch_7(), torus_mesh_switch_8()]
    for arch in arch_list:
        for noc in noc_list:
            combinations.append([arch, noc])

    combinations.append([stacked_gpu_high_noc(), strong_torus_mesh_switch_4()])
    combinations.append([stacked_gpu_low_noc(), weak_torus_mesh_switch_5()])

    return combinations

def arch_noc_combinations1110():
    combinations = []
    combinations.append([stacked_gpu_base(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_base(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_base(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_base(), torus_mesh_switch_9()])
    combinations.append([stacked_gpu_high_l1(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_high_l1(), torus_mesh_switch_2()])
    combinations.append([stacked_gpu_high_l1(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_high_l1(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_switch_7()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_switch_9()])
    combinations.append([stacked_gpu_reduced_sm(), torus_mesh_switch_2()])
    combinations.append([stacked_gpu_reduced_sm(), strong_torus_mesh_switch_4()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_switch_7()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_switch_9()])
    combinations.append([stacked_gpu_large_matrix(), torus_mesh_switch_2()])
    combinations.append([stacked_gpu_large_matrix(), strong_torus_mesh_switch_4()])
    combinations.append([stacked_gpu_low_noc(), weak_torus_mesh_switch_5()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_switch_7()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_switch_9()])

    return combinations
# def arch_noc_combinations2():
#     # arch_list = [stacked_gpu_base, stacked_gpu_large_matrix, stacked_gpu_large_vector, stacked_gpu_high_l1, stacked_gpu_high_l2, stacked_gpu_high_noc, stacked_gpu_low_noc]
#     # noc_list = [torus_mesh_switch_1, torus_mesh_switch_2, torus_mesh_mesh_3, strong_strong_torus_mesh_switch_4, weak_weak_torus_mesh_switch_5]

#     combinations = []



#     arch_list = [stacked_gpu_base(), stacked_gpu_large_matrix(), stacked_gpu_large_vector(), stacked_gpu_high_l1(), stacked_gpu_high_l2()]
#     noc_list = [torus_mesh_switch_1(), torus_mesh_switch_2(), torus_mesh_mesh_3(), torus_mesh_switch_6(), torus_mesh_switch_7(), torus_mesh_switch_8()]
#     for arch in arch_list:
#         for noc in noc_list:
#             combinations.append([arch, noc])

#     combinations.append([stacked_gpu_high_noc(), strong_strong_torus_mesh_switch_4()])
#     combinations.append([stacked_gpu_low_noc(), weak_weak_torus_mesh_switch_5()])

#     return combinations

def arch_noc_combinations_scale_64_512_nodes():
    from mosaic.noc.noc_config_set import stacked_gpu_4x4, stacked_gpu_4x8, stacked_gpu_4x16, stacked_gpu_8x8, stacked_gpu_8x16
    combinations = []

    # combinations.append([stacked_gpu_base(),stacked_gpu_4x4()])
    # combinations.append([stacked_gpu_base(),stacked_gpu_4x8()])
    # combinations.append([stacked_gpu_base(),stacked_gpu_4x16()])
    # combinations.append([stacked_gpu_base(),stacked_gpu_8x8()])
    # combinations.append([stacked_gpu_base(),stacked_gpu_8x16()])

    # combinations.append([stacked_gpu_wgmma(),stacked_gpu_4x4()])
    # combinations.append([stacked_gpu_wgmma(),stacked_gpu_4x8()])
    # combinations.append([stacked_gpu_wgmma(),stacked_gpu_4x16()])
    # combinations.append([stacked_gpu_wgmma(),stacked_gpu_8x8()])
    # combinations.append([stacked_gpu_wgmma(),stacked_gpu_8x16()])

    combinations.append([stacked_gpu_base(),stacked_gpu_4x16()])
    combinations.append([stacked_gpu_base(),stacked_gpu_8x8()])
    combinations.append([stacked_gpu_wgmma(),stacked_gpu_4x16()])
    combinations.append([stacked_gpu_wgmma(),stacked_gpu_8x8()])


    return combinations

def h100x32_arch_noc_combinations():
    combinations = []
    combinations.append([H100_SCALED(), h100x32_strong()])
    combinations.append([H200_SCALED(), h100x32_strong()])
    return combinations

def h100_8_to_64_arch_noc_combinations():
    from mosaic.noc.noc_config_set import h100_8x1_8, h100_8x2_16, h100_8x4_32, h100_8x8_64
    combinations = []
    combinations.append([H100(), h100_8x1_8()])
    combinations.append([H200(), h100_8x1_8()])

    combinations.append([H100(), h100_8x2_16()])
    combinations.append([H200(), h100_8x2_16()])

    combinations.append([H100(), h100_8x4_32()])
    combinations.append([H200(), h100_8x4_32()])

    combinations.append([H100(), h100_8x8_64()])
    combinations.append([H200(), h100_8x8_64()])
    return combinations

def b200_8x1_8_arch_noc_combinations():
    from mosaic.noc.noc_config_set import b200_8x1_8
    combinations = []
    combinations.append([B200(), b200_8x1_8()])
    return combinations


def b200_e2e_compare_arch_noc_combinations():
    """B200 NoC set used for multi-GPU validation.

    Covers one-node 4/8-GPU, two-node 16-GPU, and four-node 32-GPU
    configurations.
    """
    from mosaic.noc.noc_config_set import (
        b200_8x1_8,
        b200x4,
        b200x16,
        b200x32,
    )
    return [
        [B200(), b200_8x1_8()],
        [B200(), b200x4()],
        [B200(), b200x16()],
        [B200(), b200x32()],
    ]


def multi_gpu_validation_arch_noc_combinations():
    """Hardware topologies used by the optional multi-GPU validation driver."""
    from mosaic.noc.noc_config_set import (
        b200_8x1_8, b200x4, b200x16, b200x32,
        h200x8,
        A100x1, B6000x2,
        mi325x4, mi325x8, mi325x4_all2all, mi325x8_all2all,
    )
    from mosaic.arch import A100, B6000, H200, MI325X, MI325XSpec
    return [
        [B200(), b200_8x1_8()],
        [B200(), b200x4()],
        [B200(), b200x16()],
        [B200(), b200x32()],
        [H200(), h200x8()],
        [A100(), A100x1()],
        [B6000(), B6000x2()],
        [MI325XSpec(), mi325x4()],
        [MI325XSpec(), mi325x8()],
        [MI325XSpec(), mi325x4_all2all()],
        [MI325XSpec(), mi325x8_all2all()],
        [MI325X().set_to_microbench(), mi325x4()],
        [MI325X().set_to_microbench(), mi325x8()],
        [MI325X().set_to_microbench(), mi325x4_all2all()],
        [MI325X().set_to_microbench(), mi325x8_all2all()],
    ]


def h100x8_arch_noc_combinations():
    from mosaic.noc.noc_config_set import h100x8
    noc = h100x8()
    # Normalize the public profile name used by combination lookup.
    noc.name = "h100x8"
    return [[H100(), noc]]


def h200x8_arch_noc_combinations():
    from mosaic.noc.noc_config_set import h200x8
    return [[H200(), h200x8()]]


def h200x16_arch_noc_combinations():
    from mosaic.noc.noc_config_set import h200x16
    return [[H200(), h200x16()]]


def h200x32_arch_noc_combinations():
    from mosaic.noc.noc_config_set import h200x32
    return [[H200(), h200x32()]]


def h200_no_nvswitch_arch_noc_combinations():
    """H200 NVL machine (no NVSwitch): single 4-card NVLink islands."""
    from mosaic.noc.noc_config_set import (
        h200x4_direct_no_nvswitch,
        h200x8_direct_no_nvswitch,
    )
    return [
        [H200(), h200x4_direct_no_nvswitch()],
        [H200(), h200x8_direct_no_nvswitch()],
    ]


def h100x32_strong_arch_noc_combinations():
    return [[H100(), h100x32_strong()]]


def h100x32_medium_arch_noc_combinations():
    return [[H100(), h100x32_medium()]]


def b200_8x1_8_single_arch_noc_combinations():
    from mosaic.noc.noc_config_set import b200_8x1_8
    return [[B200(), b200_8x1_8()]]


def compare_stacked_gpu_vs_h200():
    from mosaic.noc.noc_config_set import b200_8x1_8, h100_8x4_32
    from mosaic.noc.noc_config_set import stacked_gpu_8x8

    combinations = []
    combinations.append([H200(), h100_8x4_32()])
    combinations.append([stacked_gpu_reduced_sm(), stacked_gpu_8x8()])
    return combinations



if __name__ == "__main__":
    combinations = arch_noc_combinations1()

    print("number of arch + noc combinations: ",len(combinations))
    for combination in combinations:
        print(combination[0].__class__.__name__, combination[1].__class__.__name__)

def stacked_gpu_dse_vs_h200_scaled_0309():
    combinations = []
    combinations.append([stacked_gpu_base(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_base(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_base(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_base(), torus_mesh_switch_9()])
    combinations.append([stacked_gpu_high_l1(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_high_l1(), torus_mesh_switch_2()])
    combinations.append([stacked_gpu_high_l1(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_high_l1(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_switch_7()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_high_l2(), torus_mesh_switch_9()])
    combinations.append([stacked_gpu_reduced_sm(), torus_mesh_switch_2()])
    combinations.append([stacked_gpu_reduced_sm(), strong_torus_mesh_switch_4()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_switch_7()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_large_vector(), torus_mesh_switch_9()])
    combinations.append([stacked_gpu_large_matrix(), torus_mesh_switch_2()])
    combinations.append([stacked_gpu_large_matrix(), strong_torus_mesh_switch_4()])
    combinations.append([stacked_gpu_low_noc(), weak_torus_mesh_switch_5()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_switch_1()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_mesh_3()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_switch_7()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_switch_8()])
    combinations.append([stacked_gpu_wgmma(), torus_mesh_switch_9()])

    combinations.append([H100_SCALED(), h100x32_strong()])
    combinations.append([H200_SCALED(), h100x32_strong()])

    return combinations
