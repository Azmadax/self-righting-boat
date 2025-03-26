import numpy as np
import matplotlib.pyplot as plt

def get_mouse_clicks(prompt):
    """
    Permet à l'utilisateur de dessiner un polygone en cliquant sur les sommets.
    Double-clic pour terminer le tracé.
    """
    print(prompt)
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Cliquez pour dessiner le polygone, double-cliquez pour terminer.")
    plt.grid()

    points = []

    def onclick(event):
        if event.dblclick:
            plt.close(fig)  # Ferme la figure lorsqu'un double-clic est détecté
            return
        points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, "ro")  # Trace le point cliqué
        plt.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show(block=True)  # Mode interactif pour Spyder
    return points

def align_polygon_with_water_surface(points):
    """
    Aligne le polygone avec la surface de l'eau en déplaçant verticalement ses points
    pour que la base (y minimale) corresponde à y=0.
    """
    points = np.array(points)
    y_coords = points[:, 1]
    offset = -np.min(y_coords)  # Décalage vertical pour amener la base à y=0
    aligned_points = [[x, y + offset] for x, y in points]
    return aligned_points

def compute_polygon_area_and_centroid(points):
    """
    Calcule l'aire et le centroïde d'un polygone défini par une liste de points.
    """
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Calcul de l'aire avec la formule du polygone
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Calcul du centroïde
    cx = (1 / (6 * area)) * np.sum((x + np.roll(x, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y))
    cy = (1 / (6 * area)) * np.sum((y + np.roll(y, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y))

    return area, cx, cy

def find_draft_offset_at_vertical_equilibrium(target_area, points):
    """
    Trouve le tirant d'eau pour atteindre une aire immergée cible.
    """
    def submerged_area_with_offset(offset):
        # Décale les points verticalement par `offset`
        shifted_points = [[x, y - offset] for x, y in points]
        # Calcule l'aire immergée et le centroïde
        area, _, _ = compute_polygon_area_and_centroid(shifted_points)
        return area

    # Méthode de bissection
    low, high = -10, 10
    while high - low > 1e-6:  # Précision souhaitée
        mid = (low + high) / 2
        if submerged_area_with_offset(mid) < target_area:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def plot_rotated_polygon(points, angle_deg=180):
    """
    Trace le polygone après une rotation de `angle_deg` degrés.

    :param points: Liste des points définissant le polygone.
    :param angle_deg: Angle de rotation en degrés (par défaut 180°).
    """
    # Rotation des points
    complex_points = [p[0] + p[1] * 1j for p in points]
    rotated_points = [
        c * np.exp(1j * np.radians(angle_deg)) for c in complex_points
    ]
    rotated_points = [(c.real, c.imag) for c in rotated_points]
    rotated_points.append(rotated_points[0])  # Ferme le polygone

    # Tracé du polygone
    rotated_points = np.array(rotated_points)
    plt.figure()
    plt.plot(rotated_points[:, 0], rotated_points[:, 1], marker="o", label=f"Rotation: {angle_deg}°")
    plt.title(f"Polygone après rotation de {angle_deg}°")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()

# Étape 1 : Dessiner le polygone et récupérer les points
input_curve_points = get_mouse_clicks(
    "Dessinez un polygone en cliquant sur les sommets.\nDouble-cliquez pour terminer."
)
input_curve_points.append(input_curve_points[0])  # Ferme le polygone

# Étape 2 : Aligner le polygone avec la surface de l'eau
input_curve_points = align_polygon_with_water_surface(input_curve_points)

# Étape 3 : Calcul de l'aire totale et définition de target_area
total_area, _, _ = compute_polygon_area_and_centroid(input_curve_points)
target_area = total_area / 3.0  # Un tiers de l'aire totale
print(f"Aire totale du polygone : {total_area:.2f}, Aire immergée cible : {target_area:.2f}")

# Étape 4 : Calcul de la courbe GZ
angles_deg = range(361)
GZs = []

for angle_deg in angles_deg:
    # Rotation des points
    complex_points = [p[0] + p[1] * 1j for p in input_curve_points]
    complex_point_rotated = [
        c * np.exp(1j * np.radians(angle_deg)) for c in complex_points
    ]
    curve_points = [(c.real, c.imag) for c in complex_point_rotated]

    # Centre de gravité
    center_of_gravity = curve_points.pop()
    curve_points.append(curve_points[0])  # Fermer le polygone

    # Calcul du tirant d'eau pour l'équilibre vertical
    draft_offset_equilibrium = find_draft_offset_at_vertical_equilibrium(
        target_area, curve_points
    )

    # Points immergés
    shifted_points = [[p[0], p[1] - draft_offset_equilibrium] for p in curve_points]
    _, cx, _ = compute_polygon_area_and_centroid(shifted_points)

    # Calcul du bras de levier GZ
    GZ = cx - center_of_gravity[0]
    GZs.append(GZ)

# Étape 5 : Affichage de la courbe GZ
plt.figure()
plt.plot(angles_deg, GZs, label="GZ")
plt.title("Courbe de stabilité (GZ)")
plt.xlabel("Angle d'inclinaison [°]")
plt.ylabel("Bras de levier GZ [m]")
plt.grid()
plt.legend()
plt.show()
