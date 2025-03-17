import trimesh

import numpy as np
from matplotlib import pyplot as plt

mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]])

import trimesh
from shapely.geometry import Polygon, MultiPolygon


mesh = trimesh.load('../../tests/half_mini_alpha_wrap.stl', force='mesh')
mesh.show()


# https://github.com/mikedh/trimesh/issues/1350
ext_polygon = Polygon([[0, 0], [0, 3], [3, 3], [3, 0]]) # exterior polygon
int_polygon = Polygon([[1, 1], [2, 1], [2, 2], [1, 2]]) # interior polygon
multi_poly = MultiPolygon([ext_polygon, int_polygon])

# Plot each polygon shape directly
for geom in multi_poly.geoms:
    plt.plot(*geom.exterior.xy)

# Set (current) axis to be equal before showing plot
plt.gca().axis("equal")
plt.show()

vf = [trimesh.creation.triangulate_polygon(p) for p in multi_poly.geoms]

vertices, f = trimesh.util.append_faces([i[0] for i in vf], [i[1] for i in vf])

polygon = Polygon(shell=[[0, 0], [0, 3], [3, 3], [3, 0]], holes=[[[1, 1], [2, 1], [2, 2], [1, 2]]])
vertices, faces = trimesh.creation.triangulate_polygon(polygon)
mesh = trimesh.creation.extrude_triangulation(vertices= vertices, faces=faces, height =3)
# mesh = trimesh.Trimesh(vertices=[np.hstack((v, [0])) for v in vertices],
#                        faces=f)
mesh.show()




# mesh = trimesh.creation.extrude_polygon
#
# trimesh.intersections.slice_mesh_plane(
#     to_trimesh(boat_mesh),
#     plane_normal=[0, 0, -1],
#     plane_origin=[0, 0, 0],
#     cap=True,
# )
#
# # Compute intersection
# slice = mesh.section(plane_origin, plane_normal)
#
# # Compute area of the intersection
# if slice:
#     slice_2D, to_3D = slice.to_planar()
#     area = slice_2D.area
# else:
#     area = 0
# return area