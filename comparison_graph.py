import trimesh
import signedAriaDNE
import ariaDNE
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import signedAriaDNEold

'''
Currently this file does not work, since the signedAriaDNEold file
is not included.
This was the file used for finding the signs using gaussian mean
curvature.
'''

mesh_file_names = ['normal', 'low', 'high', 'smooth', 'noise1', 'noise2']

mesh_names = ['normal', '2k Tri', '20k Tri', 'Smooth', '$10^{-3}$ Noise', '$2 \cdot 10^{-3}$ Noise']

pDNEs = []
nDNEs = []
DNEs = []

pDNEsold = []
nDNEsold = []
DNEsold = []
for mesh_file_name in mesh_file_names:
    mesh = trimesh.load('data/' + mesh_file_name + '.ply')
    _, DNE, pDNE, nDNE = signedAriaDNE.ariaDNE(mesh, 0.08)#, 0.01, 0.08)
    _, DNEold, pDNEold, nDNEold, _ = signedAriaDNEold.ariaDNE(mesh, 0.08)

    pDNEs.append(pDNE)
    nDNEs.append(nDNE)

    pDNEsold.append(pDNEold)
    nDNEsold.append(nDNEold)

pDifferences = np.abs(pDNEs - pDNEs[0])
nDifferences = np.abs(nDNEs - nDNEs[0])

pDifferencesold = np.abs(pDNEsold - pDNEsold[0])
nDifferencesold = np.abs(nDNEsold - nDNEsold[0])


n_meshes = len(mesh_names[1:])

# X-axis positions for the groups
x = np.arange(n_meshes)

# Width of the bars
bar_width = 0.35


# Plot negative signed DNE
fig, ax = plt.subplots()

bar1 = ax.bar(x - bar_width/2, nDifferences[1:], bar_width, label='neighborhood determined sign')
bar2 = ax.bar(x + bar_width/2, nDifferencesold[1:], bar_width, label='mean curvature sign')

ax.set_ylabel('Negative signed DNE')
ax.set_title('Neighbourhood determined negative DNE deviations from sample with 8k faces')
ax.set_xticks(x)
ax.set_xticklabels(mesh_names[1:])
ax.legend()

plt.savefig('negative-signed-DNE.png')


# Plot positive signed DNE
fig, ax = plt.subplots()

bar1 = ax.bar(x - bar_width/2, pDifferences[1:], bar_width, label='neighborhood determined sign')
bar2 = ax.bar(x + bar_width/2, pDifferencesold[1:], bar_width, label='mean curvature sign')

ax.set_ylabel('Positive signed DNE')
ax.set_title('Neighbourhood determined positive DNE deviations from sample with 8k faces')
ax.set_xticks(x)
ax.set_xticklabels(mesh_names[1:])
ax.legend()

plt.savefig('positive-signed-DNE.png')
