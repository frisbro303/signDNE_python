import trimesh
import signDNE

mesh = trimesh.load('data/normal.ply')
local_DNE, local_curvature, area, DNE, positive_DNE, negative_DNE = signDNE.ariaDNE(mesh, 0.08) 

print("normal:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))
