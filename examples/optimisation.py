import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib

from shapely.geometry import Point, Polygon
import shapely

from hydrostatic.hydrostatic_2d import join_polygons, close_curve, find_equilibrium_points, compute_righting_arm_curve


NUM_GZ = 10
matplotlib.use('QtAgg')

print("Demo catamaran")
hull_left = [
    [-1, -1],
    [-2, -1],
    [-2, -2],
    [-1, -2]]
hull_right = [
    [1, -1],
    [1, -2],
    [2, -2],
    [2, -1]
]
my_boat = join_polygons([hull_left, hull_right])
my_boat.reverse()
polygon = Polygon(my_boat)
x,y = polygon.exterior.xy
plt.plot(x,y)
plt.show()
target_area = 1.9



def lower_arch(polar_vars):
    n = len(polar_vars) // 3
    lower_arc = [[polar_vars[i + n] * np.cos(sum(polar_vars[:i])), polar_vars[i + n] * np.sin(sum(polar_vars[:i]))] for
                  i in range(n)]
    return lower_arc


def arch(polar_vars):
    n=len(polar_vars)//3
    lower_arc = lower_arch(polar_vars)
    upper_arch = [[(polar_vars[i+n]+polar_vars[i+2*n])*np.cos(sum(polar_vars[:i])), (polar_vars[i+n]+polar_vars[i+2*n])*np.sin(sum(polar_vars[:i]))] for i in range(n)]
    lower_arc.reverse()
    arch= upper_arch + lower_arc
    x, y = Polygon(arch).exterior.xy
    plt.plot(x, y)
    # plt.show()
    return arch

def arch_area(polar_vars):
    return -Polygon(arch(polar_vars)).area


def angle_sum_constraint(polar_vars):
    """Ensures that the total sum of angles does not exceed 360 degrees."""
    return 2 * np.pi - np.sum(polar_vars[:len(polar_vars) // 2])


def radius_constraint(i, polar_vars, R):
    """Ensures that each radius is within the maximum allowed range."""
    return R - polar_vars[2*len(polar_vars) // 3 + i]

# Define the constraint function to ensure the point is a certain distance away from the polygon
def outer_constraint(i, polar_vars):
    lower_arc = lower_arch(polar_vars)

    def outer_constraint(x, threshold=0.1):
        point = Point(x[0], x[1])  # Create a point from the coordinates
        dist = point.distance(polygon)  # Calculate the distance from the point to the polygon
        return dist - threshold  # Ensure the point is at least `threshold` distance away from the polygon


    return outer_constraint(lower_arc[i])

def stability_constraint(j, polar_vars):

    center_of_gravity = [0, 0]
    # Duplicated first point in last position to get a polygon
    angles_deg = np.linspace(start=0, stop=180, num=NUM_GZ)


    arc = arch(polar_vars)
    new_boat = join_polygons([my_boat, arc])

    try:
        righting_arm_curves = compute_righting_arm_curve(
            curve_points=new_boat,
            center_of_gravity=center_of_gravity,
            target_area=target_area,
            angles_deg=[angles_deg[j]],
            plot=False,
        )
    except ValueError as e:
        righting_arm_curves=[0]
    return righting_arm_curves

def optimize_polygon(n, R=1.0):
    """Optimizes the placement of n points in polar coordinates to minimize arch polygon area
    while ensuring the GZ is positive for positive angle and negative for negative angle
    """
    # Initial guess: Uniform angle increments and max radius
    angle_diffs = [np.pi/n for i in range(n)]  # Equal angles initially
    radii = [R for i in range(n)]  # Maximum radius
    x0 = np.concatenate([angle_diffs, radii, radii])

    # Define bounds
    # Impose minimal bound on angle diff to avoid having points superposed
    bounds = [(2*np.pi/n/2, 2 * np.pi) for _ in range(n)] + [(0, R) for _ in range(n)]+ [(0, R) for _ in range(n)]

    # # Lists to track optimization progress
    # iteration_areas = []
    # iteration_angle_constraints = []
    # iteration_radius_constraints = []

    # def callback(polar_vars):
    #     """Callback function to track optimization progress."""
    #     area = -shoelace_area(polar_vars, R)
    #     angle_constraint = angle_sum_constraint(polar_vars)
    #     radius_constraints = [radius_constraint(i, polar_vars, R) for i in range(n)]
    #
    #     iteration_areas.append(area)
    #     iteration_angle_constraints.append(angle_constraint)
    #     iteration_radius_constraints.append(radius_constraints)

    # Constraints
    constraints = [{'type': 'ineq', 'fun': angle_sum_constraint}]
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda polar_vars, i=i: radius_constraint(i, polar_vars, R)})
        constraints.append({'type': 'ineq', 'fun': lambda polar_vars, i=i: outer_constraint(i, polar_vars)})
    for j in range(NUM_GZ):
        constraints.append({'type': 'ineq', 'fun': lambda polar_vars, j=j: stability_constraint(j, polar_vars)})


    # Optimize
    # SLSQP and COBYQA were tested, but COBYQA is much worse on this simple case.
    result = minimize(arch_area, x0, args=(), constraints=constraints, method='SLSQP', bounds=bounds)
    if not result.success:
        print(result.message)
    # Convert optimized polar variables to Cartesian coordinates
    optimized_angles = np.cumsum(result.x[:n])
    optimized_radii = np.clip(result.x[n:2*n], 0, R)
    optimized_thickness = np.clip(result.x[2*n:3 * n], 0, R)
    optimized_poly = arch([optimized_angles + optimized_radii + optimized_thickness])
    new_boat = join_polygons([my_boat, optimized_poly])
    plt.show()
    x, y = Polygon(new_boat).exterior.xy
    plt.plot(x, y)
    plt.show()



    # Plot optimization progress
    # plot_optimization_progress(iteration_areas, iteration_angle_constraints, iteration_radius_constraints)

    return optimized_poly, -result.fun


def plot_polygon(coords, R):
    """Plots the optimized polygon and the reference circle."""
    fig, ax = plt.subplots()

    # Plot the reference circle
    circle = plt.Circle((0, 0), R, color='blue', fill=False, linestyle='dashed')
    ax.add_patch(circle)

    # Close the polygon by appending the first point at the end
    coords = np.vstack([coords, coords[0]])

    # Plot the polygon
    plt.plot(coords[:, 0], coords[:, 1], 'ro-', label='Optimized Polygon')

    # Formatting
    ax.set_xlim(-R - 0.1, R + 0.1)
    ax.set_ylim(-R - 0.1, R + 0.1)
    ax.set_aspect('equal')
    plt.legend()
    plt.grid()
    plt.title("Optimized Polygon in Polar Coordinates")
    plt.show()


def plot_optimization_progress(areas, angle_constraints, radius_constraints):
    """Plots the progress of the solver objective and constraints in separate subplots."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(areas, 'b-o')
    axs[0].set_title("Maximized Area")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Area")

    axs[1].plot(angle_constraints, 'r--o')
    axs[1].set_title("Angle Sum Constraint Violation")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Violation if negative")

    axs[2].plot(radius_constraints, 'g--o')
    axs[2].set_title("Radius Constraints Violation")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Violation if negative")

    plt.tight_layout()
    plt.show()


# Example: Optimize for a hexagon
n = 10  # Number of vertices
R = 1.0  # Fixed radius
coords, max_area = optimize_polygon(n, R)

print("Optimized coordinates:")
print(coords)
print("Maximized area:", max_area)

# Plot the result
# plot_polygon(coords, R)
