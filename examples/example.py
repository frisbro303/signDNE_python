import trimesh
import signDNE

mesh = trimesh.load('../data/normal.ply')

(
    local_DNE, local_curvature, DNE, positive_DNE, negative_DNE,
    surface_area, positive_surface_area, negative_surface_area
) = signDNE.aria_dne(mesh, 0.08)

print("normal:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))
