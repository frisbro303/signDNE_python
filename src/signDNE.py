from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh
import numpy as np
import pyvista as pv


def compute_f2v(mesh):
    F = mesh.faces.T
    V = mesh.vertices.T
    nf = F.shape[1]
    nv = V.shape[1]
    I = np.hstack((F[0], F[1], F[2]))
    J = np.hstack((np.arange(nf), np.arange(nf), np.arange(nf)))
    S = np.ones(len(I))
    F2V = csr_matrix((S, (J, I)), shape=(nf, nv))
    return F2V


def centralize(mesh):
    center = np.sum(mesh.vertices, 0) / mesh.vertices.shape[0]
    mesh.vertices -= center
    scale_factor = np.sqrt(1 / mesh.area)
    mesh.vertices *= scale_factor


def triangulation_to_adjacency_matrix(vertices, faces, num_points):
    A = np.zeros((num_points, num_points))
    for face in faces:
        for i in range(3):
            j = (i + 1) % 3
            v1 = face[i]
            v2 = face[j]
            dist = np.linalg.norm(vertices[v1] - vertices[v2])
            A[v1, v2] = dist
            A[v2, v1] = dist
    return A


def close_holes(tm_mesh):
    pv_mesh = pv.wrap(tm_mesh)
    filled_mesh = pv_mesh.fill_holes(hole_size=float('inf'))
    vertices = filled_mesh.points
    faces = filled_mesh.faces.reshape((-1, 4))[:, 1:]
    tm_closed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    tm_closed_mesh.fix_normals()
    return tm_closed_mesh


def aria_dne(
    mesh, bandwidth=0.08, cutoff=0, distance_type='Euclidean',
    precomputed_dist=None):
    """
    Compute the ariaDNE and signed ariaDNE values of a mesh surface.

    Parameters:
    mesh : trimesh.Trimesh
        The mesh to be analyzed.
    bandwidth : float, optional
        The epsilon value in the weight function (default is 0.08).
    cutoff : float, optional
        The cutoff distance for neighbors (default is 0).
    distance_type : str, optional
        Type of distance metric ('Euclidean' or 'Geodesic', default is 'Euclidean').
    precomputed_dist : numpy.ndarray, optional
        Precomputed distance matrix.

    Returns:
        local DNE, local curvature, DNE, positive DNE, negative DNE,
        surface area, positive surface area, and negative surface area.
    """

    if not isinstance(mesh, trimesh.base.Trimesh):
        raise TypeError("mesh must be an instance of trimesh.base.Trimesh")

    # Simple clean up
    mesh.fill_holes()
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-08))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    unnormalized_face_area = mesh.area_faces
    f2v = compute_f2v(mesh)
    unnormalized_vertex_area = (unnormalized_face_area.T @ f2v) / 3

    centralize(mesh)

    if mesh.is_watertight:
        watertight_mesh = mesh
    else:
        watertight_mesh = close_holes(mesh)

    face_area = mesh.area_faces
    vertex_area = (face_area.T @ f2v) / 3

    # Calculate non-weighted vertex normals
    face_normals = mesh.face_normals
    vertex_normals = np.zeros(mesh.vertices.shape)

    for i, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_normals[vertex] += face_normals[i]
    vertex_normals = trimesh.util.unitize(vertex_normals)

    points = mesh.vertices
    faces = mesh.faces
    num_points = np.shape(points)[0]
    normals = np.zeros((num_points, 3))
    local_curvature = np.zeros(num_points)

    if precomputed_dist is not None:
        if isinstance(precomputed_dist, np.ndarray) and precomputed_dist.shape == (num_points, num_points):
            d_dist = precomputed_dist
        else:
            raise TypeError(
                "Variable precomputed_dist must be a square numpy array "
                "with size equal to the number of points"
            )
    elif distance_type == 'Geodesic':
        d_dist = dijkstra(triangulation_to_adjacency_matrix(points, faces, num_points), directed=False)
    elif distance_type == 'Euclidean':
        d_dist = squareform(pdist(points))
    else:
        raise NameError(
            "Provide valid precomputed_dist or set distance_type to either "
            "'Geodesic' or 'Euclidean'"
        )

    K = np.exp(-d_dist ** 2 / bandwidth ** 2)

    # Estimate curvature via PCA for each vertex in the mesh
    for jj in range(num_points):
        neighbour = np.where(K[jj, :] > cutoff)[0]
        num_neighbours = len(neighbour)
        if num_neighbours <= 3:
            print(f'aria_dne: Too few neighbors on vertex {jj}.')
        p = np.tile(points[jj, :3], (num_neighbours, 1)) - points[neighbour, :3]
        w = K[jj, neighbour]

        # Build weighted covariance matrix for PCA
        C = np.zeros((6,))
        C[0] = np.sum(p[:, 0] * w.T * p[:, 0], axis=0)
        C[1] = np.sum(p[:, 0] * w.T * p[:, 1], axis=0)
        C[2] = np.sum(p[:, 0] * w.T * p[:, 2], axis=0)
        C[3] = np.sum(p[:, 1] * w.T * p[:, 1], axis=0)
        C[4] = np.sum(p[:, 1] * w.T * p[:, 2], axis=0)
        C[5] = np.sum(p[:, 2] * w.T * p[:, 2], axis=0)
        C /= np.sum(w)

        Cmat = np.array([
            [C[0], C[1], C[2]],
            [C[1], C[3], C[4]],
            [C[2], C[4], C[5]]
        ])

        # Compute eigenvalues and eigenvectors
        d, v = np.linalg.eig(Cmat)

        # Find the eigenvector closest to the vertex normal
        v_aug = np.hstack([v, -v])
        diff = v_aug - np.tile(vertex_normals[jj, :], (6, 1)).T
        q = np.sum(diff ** 2, axis=0)
        k = np.argmin(q)

        # Update the vertex normal using the eigenvector
        normals[jj, :] = v_aug[:, k]
        k %= 3

        # Calculate weighted neighborhood centroid
        neighbour_centroid = np.sum(points[neighbour, :] * w.T[:, np.newaxis], axis=0) / np.sum(w)

        # Determine if the centroid is inside or not to find the sign of curvature
        inside = watertight_mesh.ray.contains_points([neighbour_centroid])
        sign = int(inside) * 2 - 1

        # Estimate curvature using the eigenvalue
        lambda_ = d[k]
        local_curvature[jj] = (lambda_ / np.sum(d)) * sign

    # Save the outputs
    local_dne = np.multiply(local_curvature, vertex_area)
    dne = np.sum(np.abs(local_dne))

    positive_indices = np.where(local_dne >= 0)
    negative_indices = np.where(local_dne < 0)

    positive_dne = np.sum(local_dne[positive_indices])
    negative_dne = np.abs(np.sum(local_dne[negative_indices]))

    surface_area = np.sum(unnormalized_vertex_area)
    positive_surface_area = np.sum(unnormalized_vertex_area[positive_indices])
    negative_surface_area = np.sum(unnormalized_vertex_area[negative_indices])

    return (
        local_dne, local_curvature, dne, positive_dne, negative_dne,
        surface_area, positive_surface_area, negative_surface_area
    )
