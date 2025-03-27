import numpy as np
from shapely.geometry import Point, Polygon
from scipy.optimize import minimize

# Define the polygon by a set of vertices
polygon_vertices = [(1, 1), (5, 1), (5, 5), (1, 5)]
polygon = Polygon(polygon_vertices)


# Define the constraint function to ensure the point is a certain distance away from the polygon
def constraint(x, threshold=0.1):
    point = Point(x[0], x[1])  # Create a point from the coordinates
    dist = point.distance(
        polygon
    )  # Calculate the distance from the point to the polygon
    return (
        dist - threshold
    )  # Ensure the point is at least `threshold` distance away from the polygon


# Example objective function (you can replace this with your own objective)
def objective(x):
    return x[0] ** 2 + x[1] ** 2  # A simple objective to minimize


# Define the initial guess for the optimization
initial_guess = np.array([3.0, 3.0])

# Set up the constraint dictionary for scipy.optimize
constraints = {"type": "ineq", "fun": lambda x: constraint(x)}

# Run the optimization
result = minimize(objective, initial_guess, constraints=constraints)

# Output the result
print("Optimized point:", result.x)
