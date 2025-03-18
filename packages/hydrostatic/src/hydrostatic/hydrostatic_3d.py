import trimesh

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import bisect

mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]])

import trimesh
from shapely.geometry import Polygon, MultiPolygon

def get_submerged_mesh(mesh):
    plane_normal = [0, 0, -1]
    plane_origin = [0, 0, 0]
    submerged_mesh = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin,
        cap=True,
    )
    return submerged_mesh

def volume_difference(
    draft_offset: float, target_volume: float, mesh: trimesh.Trimesh
) -> float:
    """
    Function for calculating the difference between the desired submerged volume and the volume for a given draft offset.

    Args:
        draft_offset (float): The draft offset for shifting the mesh.
        target_volume (float): The submerged volume we aim (related to mass and density)
        mesh (trimesh.Trimesh): mesh

    Returns:
        float: Difference between computed volume and target volume.
    """
    # Shift curve points by draft_offset
    mesh.apply_translation(np.array([0, 0, draft_offset]))
    return get_submerged_mesh(mesh).volume - target_volume


def find_vertical_offset_for_vertical_equilibrium(mesh, target_displacement_volume):
    """
        Find the vertical offset to get the draft which enables to get the displacement of the ship

        Args:
            target_displacement_area (float): The target displacement (area in 2D)
            curve_points (List[List[float]]: The points describing the 2D ship

        Returns:
            float: The vertical offset (positive to move geometry down, and increase draft and displacement)
        """
    draft_offset_min, draft_offset_max = (
        mesh.vertices[:2].min(),
        mesh.vertices[:2].max(),
    )  # Adjust bounds as needed
    try:
        draft_offset_equilibrium: float = bisect(
            volume_difference,
            draft_offset_min,
            draft_offset_max,
            args=(
                target_displacement_volume,
                mesh,
            ),
        )
    except ValueError as e:
        if str(e) == "f(a) and f(b) must have different signs":
            raise ValueError("Ship is sinking")
        else:
            # Reraise error otherwise
            raise ValueError(repr(e))

    return draft_offset_equilibrium




if __name__=="__main__":


    # https://github.com/mikedh/trimesh/issues/1350
    ext_polygon = Polygon([[0, 0], [0, 3], [3, 3], [3, 0]]) # exterior polygon
    int_polygon = Polygon([[1, 1], [2, 1], [2, 2], [1, 2]]) # interior polygon
    multi_poly = MultiPolygon([ext_polygon, int_polygon])

    # Plot each polygon shape directly
    for geom in multi_poly.geoms:
        plt.plot(*geom.exterior.xy)

    # Set (current) axis to be equal before showing plot
    plt.gca().axis("equal")
    # plt.show()

    vf = [trimesh.creation.triangulate_polygon(p) for p in multi_poly.geoms]

    vertices, f = trimesh.util.append_faces([i[0] for i in vf], [i[1] for i in vf])

    polygon = Polygon(shell=[[0, 0], [0, 3], [3, 3], [3, 0]], holes=[[[1, 1], [2, 1], [2, 2], [1, 2]]])
    vertices, faces = trimesh.creation.triangulate_polygon(polygon)
    mesh = trimesh.creation.extrude_triangulation(vertices= vertices, faces=faces, height =3)
    # mesh = trimesh.Trimesh(vertices=[np.hstack((v, [0])) for v in vertices],
    #                        faces=f)
    # mesh.show()

    plane_normal = [0, -1, 0]
    plane_origin = [0, 1.5, 0]
    submerged_mesh = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin,
        cap=False,
    )
    broken = trimesh.repair.broken_faces(submerged_mesh, color=[255,0,0,255])

    # trimesh.repair.fill_holes(submerged_mesh)
    submerged_mesh.show()

    # mesh = trimesh.load('../../tests/half_mini_alpha_wrap.stl', force='mesh')

    slice = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

    # Compute area of the intersection
    if slice:
        slice_2D, to_3D = slice.to_2D()
        slice_2D.show()
        area = slice_2D.area
    else:
        area = 0
