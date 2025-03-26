# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from geomdl import NURBS
import matplotlib.pyplot as plt

import sys
import os


# Ajouter le chemin 'packages/hydrostatic/src' au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../packages/hydrostatic/src')))

from hydrostatic.hydrostatic_2d import (
    compute_submerged_area_and_centroid,
    computed_submerged_points,
    find_draft_offset_at_vertical_equilibrium,
)
from mouse_interaction import get_mouse_clicks

import numpy as np
import matplotlib.pyplot as plt

def generate_culbuto_boat():
    
    """Génère les points d'un bateau avec un rectangle et un demi-cercle centré.
    
    Returns:
        list of tuples: Coordonnées des points définissant la coque du bateau.
        tuple: Centre de gravité du bateau (0,0)."""
    
    width = 4  # Largeur du rectangle
    height = 2  # Hauteur du rectangle
    radius = width / 2  # Rayon du demi-cercle
    draft_offset = 1  # On abaisse toutes les coordonnées Y de 1 m

    # Rectangle : base du bateau
    rect_x = np.linspace(-width / 2, width / 2, 10)  # 10 points sur le bas
    rect_bottom = [(x, -height / 2 - draft_offset) for x in rect_x]  # Décalage en Y
    rect_right = [(width / 2, y - draft_offset) for y in np.linspace(-height / 2, height / 2, 5)]
    rect_left = [(-width / 2, y - draft_offset) for y in np.linspace(height / 2, -height / 2, 5)]

    # Demi-cercle : centré sur le haut du rectangle
    theta = np.linspace(0, np.pi, 10)  # 10 points pour lisser l'arrondi
    semi_circle = [(radius * np.cos(t), height / 2 + radius * np.sin(t) - draft_offset) for t in theta]

    # Fusion des points
    boat_shape = rect_bottom + rect_right + semi_circle + rect_left

    # Centre de gravité au milieu du rectangle
    center_of_gravity = (0, -draft_offset)

    return boat_shape, center_of_gravity

# Générer la forme du bateau et le CG
boat_points, center_of_gravity = generate_culbuto_boat()

# Visualiser la forme du bateau
boat_x, boat_y = zip(*boat_points)
plt.plot(boat_x, boat_y, marker="o", linestyle="-", label="Boat Shape")
plt.fill(boat_x, boat_y, color="gray", alpha=0.5)

# Affichage du centre de gravité
plt.plot(center_of_gravity[0], center_of_gravity[1], 'ro', markersize=8, label="Center of Gravity")

plt.axhline(0, color="blue", linestyle="--", label="Waterline")
plt.legend()
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Boat Shape: Rectangle + Semi-circle (CG at Rectangle Center, Lowered by 1m)")
plt.grid()
plt.show()


def generate_circular_boat():
    """
    Génère les points d'un bateau en forme de cercle.
    
    Returns:
        list of tuples: Coordonnées des points définissant la coque du bateau.
        tuple: Centre de gravité du bateau.
    """
    radius = 2  # Rayon du cercle
    num_points = 50  # Nombre de points pour bien lisser le cercle
    draft_offset = 1  # On abaisse toutes les coordonnées Y de 1 m

    # Génération des points du cercle
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = [(radius * np.cos(t), radius * np.sin(t) - draft_offset) for t in theta]

    # Centre de gravité au centre du cercle
    center_of_gravity = (0, -draft_offset)

    return circle_points, center_of_gravity

def generate_square_boat():
    """
    Génère les points d'un bateau en forme de cercle.
    
    Returns:
        list of tuples: Coordonnées des points définissant la coque du bateau.
        tuple: Centre de gravité du bateau.
    """
    width = 4 # Largeur du rectangle
    height = 2  # Hauteur du rectangle

    num_points = 50  # Nombre de points pour bien lisser le cercle
    draft_offset = -1  # On abaisse toutes les coordonnées Y de 1 m

    # Génération des points du carré
    rect_x = np.linspace(-width / 2, width / 2, 10)  # 10 points sur le bas
    rect_bottom = [(x, -height / 2 - draft_offset) for x in rect_x]  # Décalage en Y
    rect_top = [(x, +height / 2 - draft_offset) for x in rect_x]  # Décalage en Y
    rect_right = [(width / 2, y - draft_offset) for y in np.linspace(-height / 2, height / 2, 5)]
    rect_left = [(-width / 2, y - draft_offset) for y in np.linspace(height / 2, -height / 2, 5)]
    
    # Centre de gravité au centre du carré
    center_of_gravity = (0, -draft_offset)
    boat_shape_square = rect_bottom + rect_right + rect_left + rect_top
    
    return boat_shape_square, center_of_gravity

# Générer la forme du bateau et le CG
"""boat_points, center_of_gravity = generate_square_boat()

# Visualiser la forme du bateau
boat_x, boat_y = zip(*boat_points)
plt.plot(boat_x, boat_y, linestyle="-", label="Boat Shape")
plt.fill(boat_x, boat_y, color="gray", alpha=0.5)

# Affichage du centre de gravité
plt.plot(center_of_gravity[0], center_of_gravity[1], 'ro', markersize=8, label="Center of Gravity")

plt.axhline(0, color="blue", linestyle="--", label="Waterline")
plt.legend()
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Boat Shape: square Hull (CG at Center, Lowered by 1m)")
plt.grid()
plt.axis("equal")  # Assure une échelle égale pour le cercle
plt.show()"""


# Step 1: Define a Closed NURBS Curve
curve = NURBS.Curve()
curve.degree = 3
curve.ctrlpts = [
    [0.0, 2.0],  # Control points
    [2.0, 3.0],
    [4.0, 2.0],
    [3.0, -2.0],
    [1.0, -2.0],
    [-1.0, 0.0],  # Close the curve by duplicating the starting control point
    [0.0, 2.0],
]
curve.knotvector = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]  # Closed curve knot vector
curve.delta = 0.01  # Set resolution for sampling

# Evaluate points on the curve
curve_points = curve.evalpts
"""input_curve_points = get_mouse_clicks(
    "Draw polygon by clicking on vertices and \n double click at center of gravity to finish."
)"""
input_curve_points, center_of_gravity = generate_circular_boat()
angles_deg = range(361)
GZs = []
for angle_deg in angles_deg:
    complex_points = [p[0] + p[1] * 1j for p in input_curve_points]
    complex_point_rotated = [
        c * np.exp(1j * np.radians(angle_deg)) for c in complex_points
    ]
    curve_points = [(c.real, c.imag) for c in complex_point_rotated]
    # Last point is center of gravity
    #center_of_gravity = curve_points.pop()

    # Duplicated first point in last position to get a polygon
    curve_points.append(curve_points[0])

    # Step 2: Set the target area and find draft_offset using bisection
    target_area = 0.1  # Set the desired submerged area

    draft_offset_equilibrium = find_draft_offset_at_vertical_equilibrium(
        target_displacement_area=target_area, curve_points=curve_points
    )

    # Apply the found draft_offset to compute the submerged area and centroid
    shifted_points = [[p[0], p[1] - draft_offset_equilibrium] for p in curve_points]
    area, cx, cy = compute_submerged_area_and_centroid(shifted_points)
    x, y = computed_submerged_points(shifted_points)
    GZ = cx - center_of_gravity[0]
    GZs.append(GZ)

# Output results
print(f"Submerged Area (Volume): {area}")
print(f"Center of buoyancy: ({cx}, {cy})")

# (Optional) Plot the curve and submerged region
curve_x, curve_y = zip(*shifted_points)
plt.fill(curve_x, curve_y, color="red", alpha=0.1, edgecolor="black")
plt.plot(curve_x, curve_y, color="black", label="Closed curve")

plt.plot(cx, cy, marker="o", label="Center of buoyancy")
plt.plot(
    center_of_gravity[0],
    center_of_gravity[1] - draft_offset_equilibrium,
    marker="o",
    markerfacecolor="red",
    label="Center of gravity",
)
left, right = plt.gca().get_xlim()
bottom, top = plt.gca().get_xlim()
plt.fill(
    [2 * left, 2 * left, 2 * right, 2 * right],
    [0, 2 * bottom, 2 * bottom, 0],
    color="blue",
    alpha=0.1,
    label="Dense fluid",
)
# plt.gca().set_xlim(left, right)
# plt.gca().set_ylim(bottom, top)
# plt.fill(x, y, color="blue", alpha=0.1, label="Submerged region")
plt.axhline(0, color="blue", linestyle="--", label="y=0 Line")
plt.legend()
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title(f"Vertical equilibrium.\nTarget area = {target_area}m², GZ = {GZ:.2f}m")
plt.axis('equal')
plt.show()

plt.title("GZ curve")
plt.plot(angles_deg, GZs, label="GZ")
plt.grid()
plt.xlabel("Angle of rotation [deg]")
plt.ylabel("Righting arm GZ [m]")
plt.show()
