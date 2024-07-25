import signedAriaDNE
import trimesh
import numpy as np
import open3d as o3d
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from trimesh.transformations import translation_matrix

# Function for drawing illustration with vertex, centroid and neighborhood
def create_illustration(index, vertices, centroids):
    neighborhood = trimesh.creation.icosphere(subdevisions=3, radius=0.08)
    translation = translation_matrix(vertices[index])
    neighborhood.apply_transform(translation)
    neighborhood_color = np.tile([0, 255, 0, 50], (len(neighborhood.vertices), 1))
    neighborhood.visual.vertex_colors = neighborhood_color 

    vertex = trimesh.creation.icosphere(subdevisions=3, radius=0.005)
    vertex.apply_transform(translation)
    vertex_color = np.tile([0, 0, 0, 255], (len(neighborhood.vertices), 1))
    vertex.visual.vertex_colors = vertex_color 

    centroid = trimesh.creation.icosphere(subdevisions=3, radius=0.005)
    translation = translation_matrix(centroids[index])
    centroid.apply_transform(translation)
    centroid_color = np.tile([0, 0, 255, 255], (len(neighborhood.vertices), 1))
    centroid.visual.vertex_colors = centroid_color 

    return [neighborhood, vertex, centroid]


mesh = trimesh.load('data/normal.ply')
local_DNE, DNE, positive_DNE, negative_DNE, centroids = signedAriaDNE.ariaDNE(mesh, 0.08, dist_type='Euclidean')


# Color the shape
normalized_values = (local_DNE - np.min(local_DNE)) / (np.max(local_DNE) - np.min(local_DNE))
colors = [(0, 0, 1), (247/255, 240/255, 213/255), (1, 0, 0)]  # Blue, White, Red
custom_cmap = LinearSegmentedColormap.from_list("custom_bwr", colors)
colors = custom_cmap(normalized_values)
mesh.visual.vertex_colors = np.hstack([(colors[:, :3] * 255).astype(np.uint8), np.full((len(mesh.vertices), 1), 0.6 * 255, dtype=np.uint8)])


scene = [mesh]

# Add figures for each desired vertex to the scene
vertex_indices = [400, 238, 1000]
for i in vertex_indices:
    figure = create_illustration(i, mesh.vertices, centroids)
    scene.extend(figure)

# Draw the hole scene
trimesh.Scene(scene).show()
