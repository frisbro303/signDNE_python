import signedAriaDNE
import trimesh

mesh = trimesh.load('data/normal.ply')
local_DNE, DNE, positive_DNE, negative_DNE, centroids = signedAriaDNE.ariaDNE(mesh, 0.08, dist_type='Euclidean')

print("normal:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))


mesh = trimesh.load('data/low.ply')
local_DNE, DNE, positive_DNE, negative_DNE, centroids = signedAriaDNE.ariaDNE(mesh, 0.08, dist_type='Euclidean')

print("low:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))


mesh = trimesh.load('data/high.ply')
local_DNE, DNE, positive_DNE, negative_DNE, centroids = signedAriaDNE.ariaDNE(mesh, 0.08, dist_type='Euclidean')

print("high:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))


mesh = trimesh.load('data/smooth.ply')
local_DNE, DNE, positive_DNE, negative_DNE, centroids = signedAriaDNE.ariaDNE(mesh, 0.08, dist_type='Euclidean')

print("smooth:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))


mesh = trimesh.load('data/noise1.ply')
local_DNE, DNE, positive_DNE, negative_DNE, centroids = signedAriaDNE.ariaDNE(mesh, 0.08, dist_type='Euclidean')

print("noise1:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))


mesh = trimesh.load('data/noise2.ply')
local_DNE, DNE, positive_DNE, negative_DNE, centroids = signedAriaDNE.ariaDNE(mesh, 0.08, dist_type='Euclidean')

print("noise2:")
print("DNE: " + str(DNE))
print("positive_DNE: " + str(positive_DNE))
print("negative_DNE: " + str(negative_DNE))
