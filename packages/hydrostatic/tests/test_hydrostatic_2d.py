import numpy as np
import numpy.testing
from scipy.optimize import bisect
import pytest

from hydrostatic import (
    compute_submerged_area_and_centroid,
    find_draft_offset_at_vertical_equilibrium,
    close_curve,
)
from hydrostatic.hydrostatic_2d import (
    compute_righting_arm,
    rotate,
    compute_righting_arm_curve,
    find_equilibrium_points,
    compute_submerged_points_and_segments,
    compute_flotation_segments_inertia,
    compute_area_and_centroid,
    suppress_duplicated_neighbours,
    join_polygons,
)
from hydrostatic.sample_boats_2d import generate_circular_boat


def test_suppress_duplicated_points_in_curve():
    assert suppress_duplicated_neighbours([]) == []
    assert suppress_duplicated_neighbours([1, 1, 2]) == [1, 2]
    assert suppress_duplicated_neighbours([[1, 1], [1, 1], [2, 0], [1, 1]]) == [
        [1, 1],
        [2, 0],
        [1, 1],
    ]


def test_join_polygons():
    hull_left = [[-1, -1], [-2, -1], [-2, -2], [-1, -2]]
    hull_right = [[1, -1], [1, -2], [2, -2], [2, -1]]
    assert join_polygons([hull_left, hull_right]) == [
        [-1, -1],
        [-2, -1],
        [-2, -2],
        [-1, -2],
        [-1, -1],
        [1, -1],
        [1, -2],
        [2, -2],
        [2, -1],
        [1, -1],
        [-1, -1],
    ]


def test_computed_submerged_points_no_points_below_zero():
    """Test when there are no points below y=0."""
    curve_points = [[-1, 1], [0, 2], [1, 3], [2, 4]]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == 0
    assert len(y) == 0
    assert len(segments) == 0  # No segments under y=0


def test_computed_submerged_points_all_points_below_zero():
    """Test when all points are below y=0."""
    curve_points = [[-2, -1], [0, -2], [2, -3], [3, -1]]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == len(curve_points)  # All points should be below y=0
    assert np.array_equal(y, np.array([-1, -2, -3, -1]))
    assert len(segments) == 0  # No segments, just points


def test_computed_submerged_points_curve_crossing_zero_once():
    """Test when the curve crosses y=0 exactly once."""
    curve_points = [[-2, -1], [0, 0], [2, 1]]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == 2  # There should be one point exactly on y=0
    assert np.array_equal(x, np.array([-2, 0]))  # x-coordinates of submerged points
    assert np.array_equal(y, np.array([-1, 0]))  # y-coordinates of submerged points

    # Do not check for segment as polygon is not closed


def test_computed_submerged_points_curve_crossing_zero_multiple_times():
    """Test when the curve crosses y=0 multiple times."""
    curve_points = [[-2, -2], [0, 2], [2, -2], [3, -1], [5, 1]]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == 6  # Points below y=0 should be found
    assert np.array_equal(
        x, np.array([-2, -1, 1, 2, 3, 4])
    )  # X-coordinates of submerged points
    assert np.array_equal(
        y, np.array([-2, 0, 0, -2, -1, 0])
    )  # Y-coordinates of submerged points

    # Do not check for segment as polygon is not closed


def test_computed_submerged_points_curve_no_intersection_with_zero():
    """Test when the curve does not intersect with y=0."""
    curve_points = [[-2, 1], [0, 2], [2, 3], [3, 4]]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == 0
    assert len(y) == 0
    assert len(segments) == 0  # No intersections


def test_computed_submerged_points_empty_input():
    """Test when the input list is empty."""
    curve_points = []
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == 0
    assert len(y) == 0
    assert len(segments) == 0  # No points or segments


def test_computed_submerged_points_single_point_on_zero():
    """Test when there is exactly one point on y=0."""
    curve_points = [[0, 0]]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == 1
    assert len(y) == 1
    assert len(segments) == 0  # No segment, just one point


def test_computed_submerged_points_points_on_y_zero_and_below():
    """Test when points are on y=0 and some below y=0."""
    curve_points = [[-1, 0], [0, -1], [1, 1], [2, -1]]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == 5  # Including points below and on y=0
    assert np.array_equal(
        x, np.array([-1, 0, 0.5, 1.5, 2])
    )  # Submerged points (includes points below)
    assert np.array_equal(y, np.array([0, -1, 0, 0, -1]))  # Correct y values below zero
    # Do not check for segment as polygon is not closed


def test_computed_submerged_points_double_square():
    """Test when points are on y=0 and some below y=0."""
    curve_points = [
        [-1, 0.5],
        [-2, 0.5],
        [-2, -0.5],
        [-1, -0.5],
        [-1, 0.5],
        [1, 0.5],
        [1, -0.5],
        [2, -0.5],
        [2, 0.5],
        [1, 0.5],
        [-1, 0.5],
    ]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert np.array_equal(
        x, np.array([-2.0, -2.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0])
    )  # Submerged points (includes points below)
    assert np.array_equal(
        y, np.array([0.0, -0.5, -0.5, 0.0, 0.0, -0.5, -0.5, 0.0])
    )  # Correct y values below zero
    assert len(segments) == 2
    assert segments == [(-2, -1), (1, 2)]


def test_computed_submerged_points_double_square_with_overlap():
    """Test when points are on y=0 and some below y=0."""
    curve_points = [
        [0.25, -1],
        [-0.75, -1],
        [-0.75, -2],
        [0.25, -2],
        [0.25, -1],
        [-0.25, -1],
        [-0.25, -2],
        [0.75, -2],
        [0.75, -1],
        [-0.25, -1],
    ]
    x, y, segments = compute_submerged_points_and_segments(curve_points)
    assert np.all(y <= 0)
    assert len(x) == len(curve_points)  # 0 is not considered submerged
    assert np.array_equal(
        x, np.array([0.25, -0.75, -0.75, 0.25, 0.25, -0.25, -0.25, 0.75, 0.75, -0.25])
    )  # Submerged points (includes points below)
    assert np.array_equal(
        y, np.array([-1, -1, -2, -2, -1, -1, -2, -2, -1, -1])
    )  # Correct y values below zero
    assert len(segments) == 0


# Test for curve above y=0 (no submerged area)
def test_curve_above_y0():
    curve_points_above = [[0.0, 2.0], [2.0, 3.0], [4.0, 2.0], [3.0, 4.0]]
    area, cx, cy, metacentric_radius = compute_submerged_area_and_centroid(
        close_curve(curve_points_above)
    )
    assert area == 0, f"Expected area = 0, but got {area}"


def test_compute_flotation_segments_inertia():
    assert (
        compute_flotation_segments_inertia(x_flotations=[(-1.0, 1.0)], x_center=0)
        == 2 / 3
    )


def test_compute_area_and_centroid_flat():
    assert compute_area_and_centroid(x=np.array([0, 0]), y=np.array([1.0, 2.0])) == (
        0,
        0,
        1,
    )


# Test for curve below y=0 (entire curve submerged)
def test_curve_below_y0():
    curve_points_submerged = [
        [0.0, -2.0],
        [1.0, -2.0],
        [1.0, -3.0],
        [0.0, -3.0],
    ]
    area, cx, cy, metacentric_radius = compute_submerged_area_and_centroid(
        close_curve(curve_points_submerged)
    )
    assert area > 0, f"Expected positive area, but got {area}"
    assert area == 1


# Test for curve with multiple intersections with y=0
def test_curve_intersects_y0():
    curve_points_intersect = [
        [0.0, 2.0],
        [2.0, 2.0],
        [2.0, -1.0],
        [0.0, -1.0],
    ]
    area, cx, cy, metacentric_radius = compute_submerged_area_and_centroid(
        close_curve(curve_points_intersect)
    )
    assert area > 0, f"Expected positive area, but got {area}"
    assert area == 2


# Test for bisection method
def test_bisection_method():
    target_area = 6.0
    curve_points = [
        [0.0, 2.0],
        [2.0, 3.0],
        [4.0, 2.0],
        [3.0, -2.0],
        [1.0, -2.0],
        [-1.0, 0.0],
        [0.0, 2.0],
    ]

    def area_diff(draft_offset: float):
        shifted_points = [[p[0], p[1] - draft_offset] for p in curve_points]
        area, _, _, _ = compute_submerged_area_and_centroid(shifted_points)
        return area - target_area

    draft_offset_min, draft_offset_max = -5.0, 5.0
    draft_offset_equilibrium = bisect(area_diff, draft_offset_min, draft_offset_max)
    assert draft_offset_equilibrium is not None, (
        "Bisection failed to find equilibrium offset"
    )

    shifted_points = [[p[0], p[1] - draft_offset_equilibrium] for p in curve_points]
    area, cx, cy, metacentric_radius = compute_submerged_area_and_centroid(
        shifted_points
    )
    assert np.isclose(area, target_area, atol=0.1), (
        f"Expected area ≈ {target_area}, but got {area}"
    )


# Test for edge case: Single point curve
def test_single_point_curve():
    curve_points_single = [[0.0, 0.0]]  # Single point
    area, cx, cy, _ = compute_submerged_area_and_centroid(
        close_curve(curve_points_single)
    )
    assert area == 0, f"Expected area = 0, but got {area}"
    assert cx == 0 and cy == 0, f"Expected centroid at (0, 0), but got ({cx}, {cy})"


# Test for edge case: Empty curve
def test_empty_curve():
    curve_points_empty = []
    area, cx, cy, _ = compute_submerged_area_and_centroid(
        close_curve(curve_points_empty)
    )
    assert area == 0, f"Expected area = 0, but got {area}"
    assert np.isnan(cx) and np.isnan(cy), (
        f"Expected centroid at (0, 0), but got ({cx}, {cy})"
    )


# Test for simple curve for which results where visually checked
def test_non_regression():
    curve_points_simple = [
        [0.0, 2.0],
        [2.0, 3.0],
        [4.0, 2.0],
        [3.0, -2.0],
        [1.0, -2.0],
        [-1.0, 0.0],
        [0.0, 2.0],
    ]
    area, cx, cy, _ = compute_submerged_area_and_centroid(
        close_curve(curve_points_simple)
    )
    assert area > 0, f"Expected area > 0, but got {area}"
    assert np.isclose(cx, 1.576, atol=0.1), f"Expected centroid x ≈ 1.0, but got {cx}"
    assert np.isclose(cy, -0.871, atol=0.1), f"Expected centroid y ≈ -1.0, but got {cy}"


def test_compute_submerged_area_and_centroid_double_square():
    curve_points = [
        [-1, -1],
        [-2, -1],
        [-2, -2],
        [-1, -2],
        [-1, -1],
        [1, -1],
        [1, -2],
        [2, -2],
        [2, -1],
        [1, -1],
    ]
    area, cx, cy, _ = compute_submerged_area_and_centroid(close_curve(curve_points))
    assert area == 2
    assert np.isclose(cx, 0, atol=0.1), f"Expected centroid x ≈ .0, but got {cx}"
    assert np.isclose(cy, -1.5, atol=0.1), f"Expected centroid y ≈ -1.5, but got {cy}"


def test_compute_submerged_area_and_centroid_double_square_with_overlap():
    curve_points = [
        [0.25, -1],
        [-0.75, -1],
        [-0.75, -2],
        [0.25, -2],
        [0.25, -1],
        [-0.25, -1],
        [-0.25, -2],
        [0.75, -2],
        [0.75, -1],
        [-0.25, -1],
    ]
    area, cx, cy, _ = compute_submerged_area_and_centroid(close_curve(curve_points))
    assert area == 2
    assert np.isclose(cx, 0, atol=0.1), f"Expected centroid x ≈ .0, but got {cx}"
    assert np.isclose(cy, -1.5, atol=0.1), f"Expected centroid y ≈ -1.5, but got {cy}"


def test_find_draft_offset_at_vertical_equilibrium_down():
    draft_offset = find_draft_offset_at_vertical_equilibrium(
        target_displacement_area=1,
        curve_points=close_curve([(-1, 2), (-1, 3), (1, 3), (1, 2)]),
    )
    assert draft_offset == 2.5


def test_find_draft_offset_at_vertical_equilibrium_skimming():
    draft_offset = find_draft_offset_at_vertical_equilibrium(
        target_displacement_area=1,
        curve_points=close_curve([(-1, 0), (-1, 1), (1, 1), (1, 0)]),
    )
    assert draft_offset == 0.5


def test_find_draft_offset_at_vertical_equilibrium_skimming_up():
    draft_offset = find_draft_offset_at_vertical_equilibrium(
        target_displacement_area=1,
        curve_points=close_curve([(-1, -2), (-1, -1), (1, -1), (1, -2)]),
    )
    assert draft_offset == -1.5


def test_find_draft_offset_at_vertical_equilibrium_sinking():
    with pytest.raises(ValueError):
        _ = find_draft_offset_at_vertical_equilibrium(
            target_displacement_area=10,
            curve_points=close_curve([(-1, -2), (-1, -1), (1, -1), (1, -2)]),
        )


def test_compute_righting_arm():
    half = [[1, 0], [2, 1], [1, 2]]
    sym = [[-p[0], p[1]] for p in half]
    sym.reverse()
    righting_arm, metacenter_radius = compute_righting_arm(
        curve_points=half + sym, target_area=1, center_of_gravity=[0, 0], plot=False
    )
    assert righting_arm == 0


def test_rotate():
    curve_points = [[0, 0], [1, 0], [0, 1]]
    np.testing.assert_almost_equal(
        rotate(points=curve_points, angle=np.pi / 2), [[0, 0], [0, 1], [-1, 0]]
    )


def test_compute_righting_arm_curve_symmetric_rectangle():
    curve_points = close_curve([[-1, 0], [1, 0], [1, 1], [-1, 1]])
    center_of_gravity = [0, 0.5]
    angles_deg = [-15, 0, 15]
    target_area = 1.0  # Set the desired submerged area
    righting_arm_curve = compute_righting_arm_curve(
        curve_points=curve_points,
        center_of_gravity=center_of_gravity,
        target_area=target_area,
        angles_deg=angles_deg,
        plot=False,
    )

    assert righting_arm_curve[1] == [0]

    # Check sign convention (GZ slope positive for stable equilibrium point)
    assert righting_arm_curve[0] < 0
    assert righting_arm_curve[2] > 0


def test_compute_righting_arm_curve_circular_boat():
    curve_points, center_of_gravity = generate_circular_boat()
    angles_deg = range(0, 360)
    target_area = 1.0  # Set the desired submerged area
    righting_arm_curve = compute_righting_arm_curve(
        curve_points=close_curve(curve_points),
        center_of_gravity=center_of_gravity,
        target_area=target_area,
        angles_deg=angles_deg,
        plot=False,
    )
    # For a circle with center of gravity at center, there are always symmetry and zero righting arm
    assert np.max(np.abs(righting_arm_curve)) < 1e-3
    righting_arm, metacentric_height = compute_righting_arm(
        curve_points=close_curve(curve_points),
        target_area=1,
        center_of_gravity=center_of_gravity,
        plot=False,
    )
    assert pytest.approx(metacentric_height, abs=0.003) == 0


def test_find_equilibrium_points():
    curve_points = close_curve([[-1, 0], [1, 0], [1, 1], [-1, 1]])
    center_of_gravity = [0, 0.5]
    target_area = 1.0  # Set the desired submerged area
    eq = find_equilibrium_points(
        curve_points=curve_points,
        center_of_gravity=center_of_gravity,
        target_area=target_area,
        plot=False,
    )
    numpy.testing.assert_almost_equal(eq, [-90, 0, 90, 180])
