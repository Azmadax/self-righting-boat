import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def shoelace_area(polar_vars, R):
    """Computes the polygon area using the Shoelace theorem."""
    angles = np.cumsum(polar_vars[:len(polar_vars) // 2])  # Cumulative sum of angle differences
    radii = np.clip(polar_vars[len(polar_vars) // 2:], 0, R)  # Ensure radii stay within range

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    n = len(x)
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    return -area  # Negative for maximization


def angle_sum_constraint(polar_vars):
    """Ensures that the total sum of angles does not exceed 360 degrees."""
    return 2 * np.pi - np.sum(polar_vars[:len(polar_vars) // 2])


def radius_constraint(i, polar_vars, R):
    """Ensures that each radius is within the maximum allowed range."""
    return R - polar_vars[len(polar_vars) // 2 + i]


def optimize_polygon(n, R=1.0):
    """Optimizes the placement of n points in polar coordinates to maximize polygon area."""
    # Initial guess: Uniform angle increments and max radius
    rng = np.random.default_rng()
    angle_diffs = rng.uniform(low=0, high=2*np.pi/n, size=n)  # Equal angles initially
    radii = rng.uniform(low=0.5*R, high=R, size=n)  # Maximum radius
    x0 = np.concatenate([angle_diffs, radii])

    # Define bounds
    bounds = [(0, 2 * np.pi) for _ in range(n)] + [(0, 2*R) for _ in range(n)]

    # Lists to track optimization progress
    iteration_areas = []
    iteration_angle_constraints = []
    iteration_radius_constraints = []

    def callback(polar_vars):
        """Callback function to track optimization progress."""
        area = -shoelace_area(polar_vars, R)
        angle_constraint = angle_sum_constraint(polar_vars)
        radius_constraints = [radius_constraint(i, polar_vars, R) for i in range(n)]

        iteration_areas.append(area)
        iteration_angle_constraints.append(angle_constraint)
        iteration_radius_constraints.append(radius_constraints)

    # Constraints
    constraints = [{'type': 'ineq', 'fun': angle_sum_constraint}]
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda polar_vars, i=i: radius_constraint(i, polar_vars, R)})

    # Optimize
    result = minimize(shoelace_area, x0, args=(R,), constraints=constraints, method='SLSQP', callback=callback, bounds=bounds)
    if not result.success:
        print(result.message)
    # Convert optimized polar variables to Cartesian coordinates
    optimized_angles = np.cumsum(result.x[:n])
    optimized_radii = np.clip(result.x[n:], 0, R)
    optimized_coords = np.column_stack(
        [optimized_radii * np.cos(optimized_angles), optimized_radii * np.sin(optimized_angles)])

    # Plot optimization progress
    plot_optimization_progress(iteration_areas, iteration_angle_constraints, iteration_radius_constraints)

    return optimized_coords, -result.fun


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
n = 20  # Number of vertices
R = 1.0  # Fixed radius
coords, max_area = optimize_polygon(n, R)

print("Optimized coordinates:")
print(coords)
print("Maximized area:", max_area)

# Plot the result
plot_polygon(coords, R)
