from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh
import numpy as np


def ComputeF2V(mesh):
    F = mesh.faces.T
    V = mesh.vertices.T
    nf = F.shape[1]
    nv = V.shape[1]
    I = np.hstack((F[0], F[1], F[2]))
    J = np.hstack((np.arange(nf), np.arange(nf), np.arange(nf)))
    S = np.ones(len(I))
    F2V = csr_matrix((S, (J, I)), shape=(nf, nv))
    return F2V


def Centralize(mesh, watertight_mesh, scale=True):
    center = np.sum(mesh.vertices, 0)/(mesh.vertices.shape[0])
    mesh.vertices -= center
    watertight_mesh.vertices -= center
    if scale:
        scale_factor = np.sqrt(1 / mesh.area)
        mesh.vertices = mesh.vertices * scale_factor 
        # Only rescale if not already scaled
        if not watertight_mesh is mesh:
            watertight_mesh.vertices = watertight_mesh.vertices * scale_factor 


def triangulation_to_adjacency_matrix(vertices, faces, numPoints):
    A = np.zeros((numPoints, numPoints))
    for face in faces:
        for i in range(3):
            j = (i+1) % 3
            v1 = face[i]
            v2 = face[j]
            dist = np.linalg.norm(vertices[v1] - vertices[v2])
            A[v1, v2] = dist
            A[v2, v1] = dist
    return A


def ariaDNE(mesh, watertight_mesh=None, bandwidth=0.08, cutoff=0, distance_type='Euclidean', precomputed_dist=None):
    '''
    This function computes the ariaDNE value of a mesh surface.
    ariaDNE is a robustly implemented algorithm for Dirichlet Normal
    Energy, which measures how much a surface deviates from a plane.

    Input:
          meshname     - the mesh .ply file
          bandwidth    - the epsilon value in the paper, which indicates
                         the size of local influence in the weight function

    Output:
          curvature   - local curvature values for each vertex
          dne        - ARIADNE value for the surface

    Author:
          Shan Shan (sshan.asc@gmail.com)
          June 09, 2023
    '''

    if not (isinstance(mesh, trimesh.base.Trimesh) or 
            isinstance(mesh, trimesh.base.Trimesh)):
        raise TypeError("mesh must be an instance of trimesh.base.Trimesh")

    if watertight_mesh == None:
        watertight_mesh = mesh
    
    Centralize(mesh, watertight_mesh, scale=True)

    face_area = mesh.area_faces
    F2V = ComputeF2V(mesh)
    vertex_area = (face_area.T @ F2V)/3

    # Calculate non-weighted vertex normals
    face_normals = mesh.face_normals
    vertex_normals = np.zeros(mesh.vertices.shape)

    for i, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_normals[vertex] += face_normals[i]
    vertex_normals = trimesh.util.unitize(vertex_normals)

    points = mesh.vertices
    faces = mesh.faces
    num_points = np.shape(points)[0]; 
    normals = np.zeros((num_points, 3))
    curvature = np.zeros(num_points)

    # Only added for debugging and visualizations
    centroids = np.zeros((num_points, 3))


    d_dist = None 

    if not precomputed_dist == None:
        if isinstance(precomputed_dist, np.ndarray) and precomputed_dist.shape == (num_points, num_points):
            d_dist = precomputed_dist
        else:
            raise TypeError("Variable precomputed_dist must be a square numpy array with size equal to the number of points")
    elif distance_type == 'Geodesic':
        d_dist = dijkstra(triangulation_to_adjacency_matrix(points, faces, num_points), directed=False)
    elif distance_type == 'Euclidean':
        d_dist = squareform(pdist(points))
    else:
        raise NameError("Provide valid precomputed_dist or set dist_type to either 'Geodeisic' or 'Euclidian'")


    K = np.exp(-d_dist ** 2 / bandwidth ** 2)

    # for each vertex in the mesh, estimate its curvature via PCA
    for jj in range(num_points):#range(4):#range(num_points):
        neighbour = np.where(K[jj, :] > cutoff)[0]
        num_neighbours = len(neighbour)
        if num_neighbours <= 3:
            print('ARIADNE.m: Too few neighbor on vertex %d. \n' % jj)
        p = np.tile(points[jj, :3], (num_neighbours, 1)) - points[neighbour, :3]
        w = K[jj, neighbour]

        # build weighted covariance matrix for PCA
        C = np.zeros((6,))
        C[0] = np.sum(p[:, 0] * (w.T) * p[:, 0], axis=0)
        C[1] = np.sum(p[:, 0] * (w.T) * p[:, 1], axis=0)
        C[2] = np.sum(p[:, 0] * (w.T) * p[:, 2], axis=0)
        C[3] = np.sum(p[:, 1] * (w.T) * p[:, 1], axis=0)
        C[4] = np.sum(p[:, 1] * (w.T) * p[:, 2], axis=0)
        C[5] = np.sum(p[:, 2] * (w.T) * p[:, 2], axis=0)
        C = C / np.sum(w)

        Cmat = np.array([[C[0], C[1], C[2]], [C[1], C[3], C[4]], [C[2], C[4], C[5]]])

        # compute its eigenvalues and eigenvectors
        d, v = np.linalg.eig(Cmat)

        # find the eigenvector that is closest to the vertex normal
        v_aug = np.hstack([v, -v])
        diff = v_aug - np.tile(vertex_normals[jj, :], (6, 1)).T
        q = np.sum(diff ** 2, axis=0)
        k = np.argmin(q)

        # use that eigenvector to give an updated estimate to the vertex normal
        normals[jj, :] = v_aug[:, k]
        k = k % 3

        # calculate weighted neighbourhood centroid
        neighbour_centroid = np.sum(points[neighbour, :] * w.T[:, np.newaxis], axis=0) / np.sum(w)

        # Only added for debugging and visualizations
        centroids[jj] = neighbour_centroid

        # determine if the centroid is insidbae or not in order find sign of curvature
        inside = watertight_mesh.ray.contains_points([neighbour_centroid])
        sign = int(inside)*2 - 1

        # use the eigenvalue of that eigenvector to estimate the curvature
        lambda_ = d[k]
        curvature[jj] = (lambda_ / np.sum(d))*sign

        #curvature_nn[jj] = np.count_nonzero(w > np.max(w) * 1e-4)

    # save the outputs
    local_DNE = np.multiply(curvature, vertex_area)

    DNE = np.sum(np.abs(local_DNE))

    positive_indices = np.where(local_DNE > 0)
    negative_indices = np.where(local_DNE < 0)

    positive_DNE = np.sum(local_DNE[positive_indices])
    negative_DNE = np.sum(local_DNE[negative_indices]) 

    return local_DNE, DNE, positive_DNE, negative_DNE#, centroids


