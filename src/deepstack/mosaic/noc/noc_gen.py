"""Scalable binary-backed NoC profile generators."""

from .noc_config_set import make_generated_profile


def noc_gen_torus_mesh_switch(num_nodes: int):
    return make_generated_profile(num_nodes, False)


def noc_gen_torus_mesh_switch_variant(num_nodes: int):
    return make_generated_profile(num_nodes, True)
