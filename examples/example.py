import trimesh
import sign_ariaDNE

mesh = trimesh.load('data/normal.ply')
local_DNE, curvature, DNE, positive_DNE, negative_DNE = sign_ariaDNE.ariaDNE(mesh, 0.08) 

print("normal:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))
