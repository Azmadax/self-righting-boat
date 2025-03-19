import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def shoelace_area(coords):
    """Computes the polygon area using the Shoelace theorem."""
    x, y = coords.reshape(2, -1)
    n = len(x)
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    return -area  # Negative for maximization


def point_distance_constraint(i, coords, R):
    """Ensures that each point remains within the circle's radius."""
    x, y = coords.reshape(2, -1)
    return R ** 2 - (x[i] ** 2 + y[i] ** 2)  # Must be >= 0


def edge_intersects(p1, p2, q1, q2):
    """Checks if two line segments (p1->p2 and q1->q2) intersect."""

    def cross_product(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def subtract(v1, v2):
        return (v1[0] - v2[0], v1[1] - v2[1])

    r, s = subtract(p2, p1), subtract(q2, q1)
    qp = subtract(q1, p1)
    denom = cross_product(r, s)

    if denom == 0:  # Parallel or collinear
        return False

    t = cross_product(qp, s) / denom
    u = cross_product(qp, r) / denom

    return 0 < t < 1 and 0 < u < 1  # True if segments intersect


def segment_non_crossing_constraint(i, j, coords):
    """Ensures that edges do not cross."""
    x, y = coords.reshape(2, -1)
    n = len(x)

    if j == (i + 1) % n:  # Ignore adjacent edges
        return 0  # No intersection

    p1, p2 = (x[i], y[i]), (x[(i + 1) % n], y[(i + 1) % n])
    q1, q2 = (x[j], y[j]), (x[(j + 1) % n], y[(j + 1) % n])

    return -1 if edge_intersects(p1, p2, q1, q2) else 0


def optimize_polygon(n, R=1.0):
    """Optimizes the placement of n points on a circle to maximize polygon area."""
    # Initial guess: Random points on the circle
    angles = np.random.uniform(0, 2 * np.pi, n)
    x0 = np.column_stack([0.5*R * np.cos(angles), 0.5*R * np.sin(angles)]).flatten()

    # Lists to track optimization progress
    iteration_areas = []
    iteration_distance_constraints = []
    iteration_non_crossing_constraints = []

    def callback(coords):
        """Callback function to track optimization progress."""
        area = -shoelace_area(coords)
        distance_constraints = sum(abs(point_distance_constraint(i, coords, R)) for i in range(n))
        non_crossing_constraints = sum(
            abs(segment_non_crossing_constraint(i, j, coords)) for i in range(n) for j in range(i + 2, n))

        iteration_areas.append(area)
        iteration_distance_constraints.append(distance_constraints)
        iteration_non_crossing_constraints.append(non_crossing_constraints)

    # Constraints
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda coords, i=i: point_distance_constraint(i, coords, R)})
    for i in range(n):
        for j in range(i + 2, n):
            constraints.append(
                {'type': 'ineq', 'fun': lambda coords, i=i, j=j: segment_non_crossing_constraint(i, j, coords)})

    # Optimize
    result = minimize(shoelace_area, x0, constraints=constraints, method='SLSQP', callback=callback)

    # Reshape optimized coordinates
    optimized_coords = result.x.reshape(2, -1).T

    # Plot optimization progress
    plot_optimization_progress(iteration_areas, iteration_distance_constraints, iteration_non_crossing_constraints)

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
    plt.title("Optimized Polygon on Fixed Radius Circle")
    plt.show()


def plot_optimization_progress(areas, distance_constraints, non_crossing_constraints):
    """Plots the progress of the solver objective and constraints in separate subplots."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(areas, 'b-o')
    axs[0].set_title("Maximized Area")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Area")

    axs[1].plot(distance_constraints, 'r--o')
    axs[1].set_title("Distance Constraints Violation")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Violation Sum")

    axs[2].plot(non_crossing_constraints, 'g--o')
    axs[2].set_title("Non-Crossing Constraints Violation")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Violation Sum")

    plt.tight_layout()
    plt.show()


# Example: Optimize for a hexagon
n = 6  # Number of vertices
R = 1.0  # Fixed radius
coords, max_area = optimize_polygon(n, R)

print("Optimized coordinates:")
print(coords)
print("Maximized area:", max_area)

# Plot the result
plot_polygon(coords, R)
