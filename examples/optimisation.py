import enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib

from shapely.geometry import Point, Polygon

from hydrostatic.hydrostatic_2d import (
    join_polygons,
    compute_righting_arm_curve,
    find_equilibrium_points,
)

DEBUG = False
VERTICAL_SYM = True




ANGLE_GZ_STEP_DEG = 5
center_of_gravity = [0, -0.2]
matplotlib.use("QtAgg")

Nfeval = 0

print("Demo catamaran")

# Define the hull polygons
hull_left = [[-1, 0], [-2, 0], [-2, -1], [-1, -1]]
hull_right = [[1, 0], [1, -1], [2, -1], [2, 0]]
my_boat = join_polygons([hull_left, hull_right])
my_boat.reverse()
polygon = Polygon(my_boat)


target_area = 0.9

class ShapeFamily(str, enum.Enum):
    CIRCLE= "CIRCLE"
    KIND_OF_ELLIPSE= "KIND_OF_ELLIPSE"
    ELLIPSE = "ELLIPSE"
    POLAR= "POLAR"


shape_family = ShapeFamily.ELLIPSE

def polar_vars_split(
    polar_vars: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """
    Optimisation variable vectors is composed of:
    -n-1 angles differences
    -n distance of lower arch points
    -n lower arch thickness
    Args:
        polar_vars:

    Returns:
        list[float]: n polar angles with respect to origin
        list[float]: n distance from reference point lower_arch point
        list[float: n thicknesses of arch
    """
    if len(polar_vars) == 2:
        # Circle case
        angles=np.deg2rad(np.arange(start=0, stop=91))
        lower_arch_radius = polar_vars[0] + 0*angles
        arch_thickness = polar_vars[1] + 0*angles
    elif len(polar_vars)==4:
        if shape_family==ShapeFamily.KIND_OF_ELLIPSE:
            # Ellipsis or close
            angles=np.deg2rad(np.arange(start=0, stop=91))
            lower_arch_radius = np.linspace(start=polar_vars[0], stop=polar_vars[1], num=len(angles))
            arch_thickness = np.linspace(start=polar_vars[2], stop=polar_vars[3], num=len(angles))
        elif shape_family==ShapeFamily.ELLIPSE:

            # from center of ellipse
            #https: // math.stackexchange.com / questions / 315386 / ellipse - in -polar - coordinates
            angles=np.deg2rad(np.arange(start=0, stop=91))
            if polar_vars[0]>=polar_vars[1]:
                a=polar_vars[0]
                b= polar_vars[1]
                e = np.sqrt(1-b**2/a**2)
                lower_arch_radius = b/np.sqrt(1-e**2*np.cos(angles)**2)
            else:
                a=polar_vars[1]
                b= polar_vars[0]
                e = np.sqrt(1-b**2/a**2)
                lower_arch_radius = b/np.sqrt(1-e**2*np.sin(angles)**2)
            if polar_vars[2]>=polar_vars[3]:
                a=polar_vars[2]
                b= polar_vars[3]
                e = np.sqrt(1-b**2/a**2)
                arch_thickness = b/np.sqrt(1-e**2*np.cos(angles)**2)
            else:
                a=polar_vars[3]
                b= polar_vars[2]
                e = np.sqrt(1-b**2/a**2)
                arch_thickness = b/np.sqrt(1-e**2*np.sin(angles)**2)
    else:
        n = (len(polar_vars) + 1) // 3
        angles = [0] + [sum(polar_vars[:i]) for i in range(1, n)]
        lower_arch_radius = polar_vars[n - 1 : 2 * n - 1]
        arch_thickness = polar_vars[2 * n - 1 : 3 * n - 1]
    return angles, lower_arch_radius, arch_thickness


def lower_arch(polar_vars: list[float]) -> list[list[float]]:
    """Generates the lower part of the arch based on polar variables.

    Args:
        polar_vars (list): A list of polar variables representing the boat's geometry.

    Returns:
        list: List of coordinates representing the lower arc.
    """
    angles, lower_arch_radius, arch_thickness = polar_vars_split(polar_vars)
    lower_arc = [
        list(lower_arch_radius[i] * np.array([np.cos(angles[i]), np.sin(angles[i])]))
        for i in range(len(angles))
    ]
    if VERTICAL_SYM:
        lower_arc = lower_arc + [
            list(
                lower_arch_radius[i] * np.array([-np.cos(angles[i]), np.sin(angles[i])])
            )
            for i in reversed(range(len(angles)))
        ]
    return lower_arc


def arch(polar_vars: list[float]) -> list[list[float]]:
    """Generates the full arch shape based on polar variables.

    Args:
        polar_vars (list): A list of polar variables representing the arch geometry.

    Returns:
        list: List of coordinates representing the full arch (upper + lower).
    """
    angles, lower_arch_radius, arch_thickness = polar_vars_split(polar_vars)
    lower_arc = lower_arch(polar_vars)
    upper_arch = [
        list(
            (arch_thickness[i] + lower_arch_radius[i])
            * np.array([np.cos(angles[i]), np.sin(angles[i])])
        )
        for i in range(len(angles))
    ]
    if VERTICAL_SYM:
        upper_arch = upper_arch + [
            list(
                (arch_thickness[i] + lower_arch_radius[i])
                * np.array([-np.cos(angles[i]), np.sin(angles[i])])
            )
            for i in reversed(range(len(angles)))
        ]
    arch = upper_arch + list(reversed(lower_arc))
    # x, y = Polygon(arch).exterior.xy
    # plt.plot(x, y)
    # plt.show()
    return list(reversed(arch))


def arch_area(polar_vars: list[float]) -> float:
    """Calculates the area of the arch.

    Args:
        polar_vars (list): A list of polar variables representing the arch geometry.

    Returns:
        float: The area of the arch (negative value, since optimization minimizes).
    """
    # righting_arm_curve(polar_vars)
    return Polygon(arch(polar_vars)).area

def objective(polar_vars: list[float]):
    angles_deg = np.arange(start=5, stop=175, step=ANGLE_GZ_STEP_DEG)
    stability_constraints = [
        stability_constraint(polar_vars, angle_deg) for angle_deg in angles_deg
    ]
    return arch_area(polar_vars)+ np.sum(np.clip(-np.array(stability_constraints)+0.1, a_min=0, a_max=None))


def angle_sum_constraint(polar_vars: list[float]) -> float:
    """Ensures that the total sum of angles equals 180 degrees.

    Args:
        polar_vars (list): A list of polar variables representing the arch geometry.

    Returns:
        float: Difference between 180 degrees and the sum of the angles (last point must lie at 180°)
    """
    angles, lower_arch_radius, arch_thickness = polar_vars_split(polar_vars)

    total_angle = np.pi
    if VERTICAL_SYM:
        total_angle = total_angle / 2

    return total_angle - angles[-1]


def outer_constraint(i: int, polar_vars: list[float]) -> float:
    """Ensures that each point is at least a certain distance from the polygon (boat hull).

    Args:
        i (int): Index of the point to check.
        polar_vars (list): A list of polar variables representing the arch geometry.

    Returns:
        float: Difference between the distance from the point to the polygon and the threshold.
    """
    lower_arc = lower_arch(polar_vars)

    def constraint(x, threshold=0.1):
        point = Point(x[0], x[1])
        dist = point.distance(polygon)
        return dist - threshold

    return constraint(lower_arc[i])


def stability_constraint(polar_vars: list[float], angle_deg: float
) -> float:
    """Ensures that the boat's righting arm curve is valid for stability at each angle.

    Args:
        polar_vars (list): A list of polar variables representing the arch geometry.
        angle_deg: angle at which righting arm must be evaluated

    Returns:
        list: Righting arm curve for the given angle.
    """

    arc = arch(polar_vars)
    new_boat = join_polygons([my_boat, arc])

    # eq = find_equilibrium_points(
    #     curve_points=new_boat,
    #     center_of_gravity=center_of_gravity,
    #     target_area=target_area,
    #     plot=True,
    # )

    try:
        righting_arm_curves = compute_righting_arm_curve(
            curve_points=new_boat,
            center_of_gravity=center_of_gravity,
            target_area=target_area,
            angles_deg=[angle_deg],
            plot=False,
        )
    except ValueError:
        righting_arm_curves = [0]
    # if righting_arm_curves[0] * np.sign(angles_deg[j]) < -0.001:
    # x, y = Polygon(new_boat).exterior.xy
    # plt.plot(x, y)
    # plt.show()
    return righting_arm_curves[0]


def righting_arm_curve(polar_vars: list[float]) -> None:
    center_of_gravity = [0, 0]
    arc = arch(polar_vars)
    new_boat = join_polygons([my_boat, arc])
    angles_deg = range(-180, 182)
    compute_righting_arm_curve(
        curve_points=new_boat,
        center_of_gravity=center_of_gravity,
        target_area=target_area,
        angles_deg=angles_deg,
        plot=True,
    )


def optimize_polygon(n: int, R: float = 1.0) -> tuple[list[list[float]], list[float]]:
    """Optimizes the placement of n points in polar coordinates to minimize arch polygon area
    while ensuring the GZ is positive for positive angles and negative for negative angles.

    Args:
        n (int): Number of points (vertices) in the polygon.
        R (float): Maximum allowed radius.

    Returns:
        tuple: Optimized polygon coordinates and the minimized area.
    """
    if VERTICAL_SYM:
        factor = 1 / 2.0
    else:
        factor = 1

    if shape_family == ShapeFamily.POLAR:
        angle_diffs = [np.pi * factor / (n - 1) for i in range(n - 1)]
        radii = [R for i in range(n)]
        thickness = radii
        bounds = (
                [(np.pi * factor / (n - 1) / 2, np.pi * factor) for _ in range(n - 1)]
                + [(0.1, R) for _ in range(n)]
                + [(0.1, R) for _ in range(n)]
        )
    elif shape_family== ShapeFamily.CIRCLE:
        angle_diffs = []  # For circle and ellipse
        radii = [1]
        thickness = [1]
        bounds = (
                [(0.1, R) for _ in range(1)]
                + [(0.1, R) for _ in range(1)]
        )
    elif shape_family== ShapeFamily.KIND_OF_ELLIPSE or shape_family== ShapeFamily.ELLIPSE:
        angle_diffs = []  # For circle and ellipse
        radii = [1, 1]
        thickness = [1, 1]
        bounds = (
                [(0.1, R) for _ in range(2)]
                + [(0.1, R) for _ in range(2)]
        )



    x0 = np.concatenate([angle_diffs, radii, thickness])

    # Lists to track optimization progress
    iteration_areas = []
    iteration_angle_constraints = []
    iteration_stability_constraints = []

    def callback(polar_vars: list[float]) -> None:
        """Callback function to track optimization progress.

        Args:
            polar_vars (list): The current values of the optimization variables (polar coordinates).
        """
        global Nfeval
        Nfeval += 1
        if DEBUG:
            arc = arch(polar_vars)
            new_boat = join_polygons([my_boat, arc])

            try:
                find_equilibrium_points(
                    curve_points=new_boat,
                    center_of_gravity=center_of_gravity,
                    target_area=target_area,
                    plot=True,
                )
            except ValueError:
                print("invalid solution")
        area = arch_area(polar_vars)
        angle_constraint = angle_sum_constraint(polar_vars)
        angles_deg = np.arange(start=5, stop=175, step=ANGLE_GZ_STEP_DEG)
        stability_constraints = [
            stability_constraint(polar_vars, angle_deg) for angle_deg in angles_deg
        ]
        if not (VERTICAL_SYM):
            angles_deg = np.arange(start=-5, stop=-175, step=-ANGLE_GZ_STEP_DEG)
            stability_constraints = stability_constraints + [
                -1 * stability_constraint(polar_vars, angles_deg) for angle_deg in angles_deg
            ]

        iteration_areas.append(area)
        iteration_angle_constraints.append(angle_constraint)
        iteration_stability_constraints.append(stability_constraints)
        print("Nfeval: ", Nfeval)
        print("area: ", area)
        print("angle constrain: ", angle_constraint)
        print("stability constrain: ", stability_constraints)
        print("polar var: ", polar_vars)
    callback(x0)



    # Constraints
    constraints = []
    if shape_family == ShapeFamily.POLAR:
        constraints.append({"type": "eq", "fun": angle_sum_constraint})
        for i in range(n):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda polar_vars, i=i: outer_constraint(i, polar_vars),
                }
            )
    angles_deg = np.arange(start=5, stop=175, step=ANGLE_GZ_STEP_DEG)
    for angle_deg in angles_deg:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda polar_vars : stability_constraint(
                    polar_vars, angle_deg=angle_deg
                ),
            }
        )
    if not( VERTICAL_SYM):
        angles_deg = np.arange(start=-5, stop=-175, step=-ANGLE_GZ_STEP_DEG)
        for angle_deg in angles_deg:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda polar_vars: -stability_constraint(
                        polar_vars, angle_deg=angle_deg
                    ),
                }
            )
    result = minimize(
        objective,
        x0,
        constraints=constraints,
        method="SLSQP",
        bounds=bounds,
        callback=callback,
        options={"disp": False},
    )
    if not result.success:
        print(result.message)

    optimized_poly = arch(result.x)
    new_boat = join_polygons([my_boat, optimized_poly])

    try:
        find_equilibrium_points(
            curve_points=new_boat,
            center_of_gravity=center_of_gravity,
            target_area=target_area,
            plot=True,
        )
    except ValueError:
        print("invalid solution")

    # Plot optimization progress
    plot_optimization_progress(
        iteration_areas,
        iteration_angle_constraints,
        iteration_stability_constraints,
    )

    return optimized_poly, result.fun


def plot_polygon(coords: list[list[float]], R: float) -> None:
    """Plots the optimized polygon and the reference circle.

    Args:
        coords (ndarray): Optimized polygon coordinates.
        R (float): Radius of the reference circle.
    """
    fig, ax = plt.subplots()

    circle = plt.Circle((0, 0), R, color="blue", fill=False, linestyle="dashed")
    ax.add_patch(circle)

    coords = np.vstack([coords, coords[0]])

    plt.plot(coords[:, 0], coords[:, 1], "ro-", label="Optimized Polygon")

    ax.set_xlim(-R - 0.1, R + 0.1)
    ax.set_ylim(-R - 0.1, R + 0.1)
    ax.set_aspect("equal")
    plt.legend()
    plt.grid()
    plt.title("Optimized Polygon in Polar Coordinates")
    plt.show()


def plot_optimization_progress(
    areas: list[float],
    angle_constraints: list[float],
    stability_constraints: list[list[float]],
) -> None:
    """Plots the progress of the solver objective and constraints in separate subplots.

    Args:
        areas (list): List of area values at each iteration.
        angle_constraints (list): List of angle constraint violations.
        stability_constraints (list): List of stability constraint violations.
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(areas, "b-o")
    axs[0].set_title("Minimized Area")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Area")

    axs[1].plot(angle_constraints, "r--o")
    axs[1].set_title("Angle Sum Constraint Violation")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Violation if negative")

    axs[2].plot(stability_constraints, "y--o")
    axs[2].set_title("Stability Constraints Violation")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Violation if negative")

    plt.tight_layout()
    plt.show()


# Example: Optimize for a hexagon
n = 10  # Number of vertices
R = 10.0  # Fixed radius
coords, min_area = optimize_polygon(n, R)

print("Optimized coordinates:")
print(coords)
print("Minimized area:", min_area)

# Plot the result
# plot_polygon(coords, R)
