import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Translation
from compas_assembly.datastructures import Block

from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import add_external_force
from compas_cra.equilibrium import density_setup
from compas_cra.equilibrium import external_force_setup

DENSITY = 2000.0  # [kg/m3]
W = DENSITY * 9.81  # weight of the unit cube [N]


def box_assembly():
    support = Box(1, 1, 1)
    free1 = Box(1, 1, 1, frame=Frame.worldXY().transformed(Translation.from_vector([0, 0, 1])))
    assembly = CRA_Assembly()
    assembly.add_block(Block.from_shape(support), node=0)
    assembly.add_block(Block.from_shape(free1), node=1)
    assembly.set_boundary_conditions([0])
    interface = Mesh()
    for i, c in enumerate([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]):
        interface.add_vertex(key=i, x=c[0], y=c[1], z=c[2])
    interface.add_face([0, 1, 2, 3])
    assembly.add_interfaces_from_meshes([interface], 0, 1)
    return assembly


def test_no_external_force_is_self_weight():
    p = external_force_setup(box_assembly(), density=DENSITY)
    assert p.shape == (6, 1)
    assert np.allclose(p.flatten(), [0, 0, -W, 0, 0, 0])
    # without a density and without external forces the weights are unit weights
    assert np.allclose(external_force_setup(box_assembly()).flatten(), [0, 0, -1, 0, 0, 0])


def test_external_force_adds_to_self_weight():
    assembly = box_assembly()
    add_external_force(assembly, 1, [2, 3, -4, 5, 6, 7])
    p = external_force_setup(assembly, density=DENSITY)
    assert np.allclose(p.flatten(), [2, 3, -W - 4, 5, 6, 7])


def test_external_force_accumulates():
    assembly = box_assembly()
    add_external_force(assembly, 1, [1, 0, 0, 0, 0, 0])
    add_external_force(assembly, 1, [1, 0, -2, 0, 0, 0])
    p = external_force_setup(assembly, density=DENSITY)
    assert np.allclose(p.flatten(), [2, 0, -W - 2, 0, 0, 0])


def test_external_force_on_support_is_ignored():
    assembly = box_assembly()
    add_external_force(assembly, 0, [9, 9, 9, 9, 9, 9])
    p = external_force_setup(assembly)  # and it does not make the density required
    assert np.allclose(p.flatten(), [0, 0, -1, 0, 0, 0])


def test_external_force_rejects_bad_input():
    assembly = box_assembly()
    with pytest.raises(ValueError):
        add_external_force(assembly, 1, [0, 0, -1])
    with pytest.raises(KeyError):
        add_external_force(assembly, 5, [0, 0, -1, 0, 0, 0])


def test_external_force_requires_density():
    assembly = box_assembly()
    add_external_force(assembly, 1, [100e3, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        external_force_setup(assembly)


def test_external_force_block_densities_are_enough():
    assembly = box_assembly()
    add_external_force(assembly, 1, [100e3, 0, 0, 0, 0, 0])
    density_setup(assembly, {1: DENSITY})
    p = external_force_setup(assembly)
    assert np.allclose(p.flatten(), [100e3, 0, -W, 0, 0, 0])


def test_gravity_of_the_assembly():
    assembly = box_assembly()
    assert assembly.gravity == 9.81
    assembly.gravity = 1.62
    p = external_force_setup(assembly, density=DENSITY)
    assert np.allclose(p.flatten(), [0, 0, -DENSITY * 1.62, 0, 0, 0])


def test_external_force_reaches_solver():
    pytest.importorskip("compas_cra._native")
    from compas_cra.equilibrium import cra_solve

    # in newtons, with a joint that needs friction: this only solves because the loads are normalised
    assembly = box_assembly()
    add_external_force(assembly, 1, [6000, 0, -5000, 0, 0, 0])
    cra_solve(assembly, density=DENSITY)

    interface = assembly.graph.edge_attribute((0, 1), "interfaces")[0]
    normal = sum(force["c_np"] - force["c_nn"] for force in interface.forces)
    shear = np.hypot(sum(force["c_u"] for force in interface.forces), sum(force["c_v"] for force in interface.forces))
    assert normal == pytest.approx(W + 5000, rel=1e-6)
    assert shear == pytest.approx(6000, rel=1e-6)
