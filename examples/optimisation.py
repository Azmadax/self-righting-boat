import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def shoelace_area(coords):
    """Computes the polygon area using the Shoelace theorem."""
    x, y = coords.reshape(2, -1)  # Split into x and y
    n = len(x)
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    return -area  # Negative for maximization


def circle_constraint(coords, R):
    """Constraint: Each point must satisfy x^2 + y^2 = R^2."""
    x, y = coords.reshape(2, -1)
    return R ** 2 - (x ** 2 + y ** 2)


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


def non_crossing_constraint(coords):
    """Ensures that edges do not cross."""
    x, y = coords.reshape(2, -1)
    n = len(x)

    for i in range(n):
        for j in range(i + 2, n):  # Avoid consecutive edges
            if j == (i + 1) % n:  # Ignore adjacent edges
                continue

            p1, p2 = (x[i], y[i]), (x[(i + 1) % n], y[(i + 1) % n])
            q1, q2 = (x[j], y[j]), (x[(j + 1) % n], y[(j + 1) % n])

            if edge_intersects(p1, p2, q1, q2):
                return -1  # Intersection detected
    return 0  # No intersections


def optimize_polygon(n, R=1.0):
    """Optimizes the placement of n points on a circle to maximize polygon area."""
    # Initial guess: Random points on the circle
    angles = np.random.uniform(0, 2 * np.pi, n)
    x0 = np.column_stack([0.5*R * np.cos(angles), 0.5*R * np.sin(angles)]).flatten()

    # Lists to track optimization progress
    iteration_areas = []
    iteration_constraints = []

    def callback(coords):
        """Callback function to track optimization progress."""
        area = -shoelace_area(coords)
        constraint_vals = np.abs(circle_constraint(coords, R)).sum()
        iteration_areas.append(area)
        iteration_constraints.append(constraint_vals)

    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda coords: circle_constraint(coords, R)},
        {'type': 'ineq', 'fun': non_crossing_constraint}  # Enforce non-crossing
    ]

    # Optimize
    result = minimize(shoelace_area, x0, constraints=constraints, method='SLSQP', callback=callback)

    # Reshape optimized coordinates
    optimized_coords = result.x.reshape(2, -1).T

    # Plot optimization progress
    plot_optimization_progress(iteration_areas, iteration_constraints)

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


def plot_optimization_progress(areas, constraints):
    """Plots the progress of the solver objective (area) and constraints."""
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(areas, 'b-o', label='Maximized Area')
    ax2.plot(constraints, 'r--o', label='Constraint Violation')

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Polygon Area", color='b')
    ax2.set_ylabel("Constraint Violation", color='r')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title("Optimization Progress")
    plt.grid()
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
