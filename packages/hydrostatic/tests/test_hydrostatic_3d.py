import pytest
import trimesh

from hydrostatic.hydrostatic_3d import get_submerged_mesh, volume_difference, \
    find_vertical_offset_for_vertical_equilibrium, compute_righting_arm


def test_submerged_mesh():
    mesh = trimesh.primitives.Box(extents=[1, 1, 1])
    submerged_mesh = get_submerged_mesh(mesh)

    # Cube is centered
    assert submerged_mesh.volume==0.5

def test_volume_difference():
    mesh = trimesh.primitives.Box(extents=[1, 1, 1])
    diff = volume_difference(
        draft_offset=0, target_volume=0.5, mesh=mesh
    )

    # Cube is centered
    assert diff==0


def test_find_draft_offset_at_vertical_equilibrium_sinking():
    mesh = trimesh.primitives.Box(extents=[1, 1, 1])
    assert find_vertical_offset_for_vertical_equilibrium(
            target_displacement_volume=0.5,
            mesh=mesh,
        )==0.5


def test_compute_righting_arm():
    mesh = trimesh.primitives.Box(extents=[1, 1, 1])
    righting_arm = compute_righting_arm(
        mesh=mesh, target_volume=0.75, center_of_gravity=[0, 0], plot=False
    )
    assert righting_arm == pytest.approx(0)
# def test_find_equilibrium_points():
#     trimesh.primitives.Box
#     center_of_gravity = [0, 0.5]
#     target_area = 1.0  # Set the desired submerged area
#     eq = find_equilibrium_points(
#         curve_points=curve_points,
#         center_of_gravity=center_of_gravity,
#         target_area=target_area,
#         plot=False,
#     )
#     numpy.testing.assert_almost_equal(eq, [-90, 0, 90, 180])