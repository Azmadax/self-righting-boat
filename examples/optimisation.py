import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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


def optimize_polygon(n, R=1.0):
    """Optimizes the placement of n points on a circle to maximize polygon area."""
    # Initial guess: Regular n-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([R * np.cos(angles), R * np.sin(angles)]).flatten()

    # Constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda coords: circle_constraint(coords, R)
    }]

    # Optimize
    result = minimize(shoelace_area, x0, constraints=constraints, method='COBYQA')

    # Reshape optimized coordinates
    optimized_coords = result.x.reshape(2, -1).T
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


# Example: Optimize for a hexagon
n = 12  # Number of vertices
R = 1.0  # Fixed radius
coords, max_area = optimize_polygon(n, R)

print("Optimized coordinates:")
print(coords)
print("Maximized area:", max_area)

# Plot the result
plot_polygon(coords, R)