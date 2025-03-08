import numpy as np
import matplotlib.pyplot as plt

from hydrostatic.hydrostatic_2d import join_polygons


def generate_arch_boat_inner_outer():
    print("Executing generate_boat_arch_inner_outer()")

    width = 4  # Width of the rectangle
    height = 2  # Height of the rectangle
    e = 0.4 # Thickness of the arch
    radius_sup = width / 2  # Radius of the semi-circle
    radius_inf = radius_sup - e
    draft_offset = 1  # Lowers all Y coordinates by 1 meter

    # Rectangle: base of the boat
    rect_x = np.linspace(-width / 2, width / 2, 10)
    rect_bottom = [(x, -height / 2 - draft_offset) for x in rect_x]
    rect_right = [(width / 2, y - draft_offset) for y in np.linspace(-height / 2, height / 2, 5)]
    rect_left = [(-width / 2, y - draft_offset) for y in np.linspace(height / 2, -height / 2, 5)]

    # Top semi-circle: centered at the top of the rectangle
    theta = np.linspace(0, np.pi, 10)
    semi_circle_sup = [(radius_sup * np.cos(t), height / 2 + radius_sup * np.sin(t) - draft_offset) for t in theta]

    # Bottom semi-cercle (inverted to avoid fillin in)
    semi_circle_inf = [(radius_inf * np.cos(t), height / 2 + radius_inf * np.sin(t) - draft_offset) for t in reversed(theta)]

    # Top semi-circle: centered at the top of the rectangle
    outer_curve_points = rect_left + rect_bottom  + rect_right+ semi_circle_sup

    # Bottom semi-circle: centered at the top of the rectangle
    inner_curve_points = list(reversed(semi_circle_inf))
    if inner_curve_points[0] != inner_curve_points[-1]:
        inner_curve_points.append(inner_curve_points[0])

    # Centre de gravité
    center_of_gravity = [0.0, -float(draft_offset)]
    
    outer_curve_points = [[float(x), float(y)] for x, y in outer_curve_points]
    inner_curve_points = [[float(x), float(y)] for x, y in inner_curve_points]
    boat_shape = join_polygons([outer_curve_points, list(reversed(inner_curve_points))])

    return boat_shape, center_of_gravity

def generate_arch_boat() -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Generates the points of a boat with a trapeze and a semi-circle arch centered at the top.

    Returns:
        list[tuple[float, float]:]: Coordinates of the points defining the boat's hull.
        tuple[float, float]: Center of gravity of the boat (0,0).
    """
    print("Executing generate_boat_arch()")
    width = 4  # Width of the rectangle
    height = 2  # Height of the rectangle
    e = 0.4 # Thickness of the arch
    radius_sup = width / 2  # Radius of the semi-circle
    radius_inf = radius_sup - e
    draft_offset = 1  # Lowers all Y coordinates by 1 meter

    # Rectangle: base of the boat
    rect_x = np.linspace(-width / 2, width / 2, 10)  # 10 points along the bottom
    rect_bottom = [(x, -height / 2 - draft_offset) for x in rect_x]  # Y offset
    rect_right = [
        (width / 2, y - draft_offset) for y in np.linspace(-height / 2, height / 2, 5)
    ]
    rect_left = [
        (-width / 2, y - draft_offset) for y in np.linspace(height / 2, -height / 2, 5)
    ]

    # Top semi-circle: centered at the top of the rectangle
    theta = np.linspace(0, np.pi, 10)  # 10 points for a smooth curve
    semi_circle_sup= [
        (+radius_sup * np.cos(t), height / 2 + radius_sup * np.sin(t) - draft_offset)
        for t in theta
    ]
    # Bottom semi-circle: centered at the top of the rectangle
    theta = np.linspace(0, np.pi, 10)  # 10 points for a smooth curve
    semi_circle_inf = [
        (-radius_inf * np.cos(t), height / 2 + radius_inf * np.sin(t) - draft_offset)
        for t in theta
    ]

    # Merge all points
    boat_shape = join_polygons([rect_left + rect_bottom + rect_right, semi_circle_inf + semi_circle_sup])

    # Center of gravity at the middle of the rectangle²
    center_of_gravity = (0, -draft_offset)

    return boat_shape, center_of_gravity

def generate_culbuto_boat() -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Generates the points of a boat with a rectangle and a semi-circle centered at the top.

    Returns:
        list[tuple[float, float]:]: Coordinates of the points defining the boat's hull.
        tuple[float, float]: Center of gravity of the boat (0,0).
    """
    width = 4  # Width of the rectangle
    height = 2  # Height of the rectangle
    radius = width / 2  # Radius of the semi-circle
    draft_offset = 1  # Lowers all Y coordinates by 1 meter

    # Rectangle: base of the boat
    rect_x = np.linspace(-width / 2, width / 2, 10)  # 10 points along the bottom
    rect_bottom = [(x, -height / 2 - draft_offset) for x in rect_x]  # Y offset
    rect_right = [
        (width / 2, y - draft_offset) for y in np.linspace(-height / 2, height / 2, 5)
    ]
    rect_left = [
        (-width / 2, y - draft_offset) for y in np.linspace(height / 2, -height / 2, 5)
    ]

    # Semi-circle: centered at the top of the rectangle
    theta = np.linspace(0, np.pi, 10)  # 10 points for a smooth curve
    semi_circle = [
        (radius * np.cos(t), height / 2 + radius * np.sin(t) - draft_offset)
        for t in theta
    ]

    # Merge all points
    boat_shape = rect_bottom + rect_right + semi_circle + rect_left

    # Center of gravity at the middle of the rectangle
    center_of_gravity = (0, -draft_offset)

    return boat_shape, center_of_gravity





def generate_circular_boat() -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """
    Generates the points of a boat in the shape of a circle.

    Returns:
        list[tuple[float, float]:]: Coordinates of the points defining the boat's hull.
        tuple[float, float]: Center of gravity of the boat.
    """
    radius = 2  # Radius of the circle
    num_points = 50  # Number of points to smooth the circle
    draft_offset = 1  # Lowers all Y coordinates by 1 meter

    # Generate circle points
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = [
        (radius * np.cos(t), radius * np.sin(t) - draft_offset) for t in theta
    ]

    # Center of gravity at the center of the circle
    center_of_gravity = (0, draft_offset)

    return circle_points, center_of_gravity


def generate_square_boat() -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """
    Generates the points of a boat in the shape of a square.

    Returns:
        list[tuple[float, float]:]: Coordinates of the points defining the boat's hull.
        tuple[float, float]: Center of gravity of the boat.
    """
    width = 4  # Width of the square
    height = 2  # Height of the square
    draft_offset = -1  # Lowers all Y coordinates by 1 meter

    # Generate square points
    x_min = -width / 2
    x_max = +width / 2
    y_min = -height / 2 - draft_offset
    y_max = +height / 2 - draft_offset

    # Center of gravity at the center of the square
    center_of_gravity = (0, -draft_offset)
    boat_shape_square = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]

    return boat_shape_square, center_of_gravity


if __name__ == "__main__":
    # outer_curve_points, inner_curve_points, center_of_gravity = generate_boat_arch_inner_outer()
    #
    # # Extraire les coordonnées
    # outer_x, outer_y = zip(*outer_curve_points)
    # inner_x, inner_y = zip(*inner_curve_points)
    #
    # # Remplissage extérieur
    # plt.fill(outer_x, outer_y, color="gray", alpha=0.5)
    #
    # # Trou intérieur (en blanc)
    # plt.fill(inner_x, inner_y, color="white")
    #
    # # Tracer les contours
    # plt.plot(outer_x, outer_y, "k-", label="Outer Shape")
    # plt.plot(inner_x, inner_y, "r-", label="Inner Arch (Hollow)")
    #
    # plt.axis("equal")
    # plt.legend()
    # plt.xlabel("X [m]")
    # plt.ylabel("Y [m]")
    # plt.title("Boat Shape with Hollow Arch")
    # plt.grid()
    # plt.show()
    for method in [generate_arch_boat_inner_outer, generate_arch_boat, generate_circular_boat, generate_square_boat]:
        # Generate boat shape and CG
        boat_points, center_of_gravity = method()

        # Visualize boat shape
        boat_x, boat_y = zip(*boat_points)
        plt.plot(boat_x, boat_y, marker="o", linestyle="-", label="Boat Shape")
        plt.fill(boat_x, boat_y, color="gray", alpha=0.5)

        # Display center of gravity
        plt.plot(
            center_of_gravity[0],
            center_of_gravity[1],
            "ro",
            markersize=8,
            label="Center of Gravity",
        )

        plt.axhline(0, color="blue", linestyle="--", label="Waterline")
        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(
            "Boat Shape: Rectangle + Semi-circle (CG at Rectangle Center, Lowered by 1m)"
        )
        plt.grid()
        plt.show()
