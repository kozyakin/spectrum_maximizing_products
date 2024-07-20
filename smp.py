# -*- coding: utf-8 -*-
"""An extremal norm S and its images AS and BS

Created on Fri Jul 5 13:46:28 2024 +0300.

@author: Victor Kozyakin
"""

import os
import re
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

# Initialization
theta = (2 * np.pi) / 3
cos_a = np.cos(theta)
sin_a = np.sin(theta)
KAPPA = 1.1**3
AUXFACTOR = 0.96

MU = AUXFACTOR * (1 + KAPPA**2)**2 / (1 + KAPPA**2 + KAPPA**4)

MUR = f'-{round(MU, 2)*100:.0f}'

myfilename = os.path.splitext(Path(__file__).name)[0]
myfilename = re.sub('[_-][0-9]+', '', myfilename)
pdfout = re.sub('_', '-', myfilename) + MUR + '.pdf'

A0 = np.array([[0, -1 / KAPPA], [KAPPA, 2 * cos_a]])
B0 = np.array([[0, -KAPPA], [1 / KAPPA, 2 * cos_a]])

v = np.array([1, KAPPA / (1 + KAPPA**2)])
w = np.array([1, 0])

LAMBDA = KAPPA**2
A = A0 / LAMBDA**(1 / 3)
B = B0 / LAMBDA**(1 / 3)

# Vertices of a polygon formed by eigenvectors of normed matrices
# BAA & BBA and their cyclic permutations
v1 = MU * v
v2 = w
v3 = -A @ v1
v4 = -A @ v2
v5 = -A @ v3
v6 = -B @ v4
v7 = -v1
v8 = -v2
v9 = -v3
v10 = -v4
v11 = -v5
v12 = -v6

V = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12])
AV = V @ A.T
BV = V @ B.T

print('\nList of vertices:\n')
print(', \n'.join(f'({v[0]:9.6f}, {v[1]:9.6f})' for v in V))

# Plotting

S = Polygon(V, facecolor='#f0f0f0', edgecolor='k')
AS = Polygon(AV, linestyle='--', facecolor='none', edgecolor='r')
BS = Polygon(BV, linestyle='-.', facecolor='none', edgecolor='b')

fig, ax = plt.subplots(
    num='Unit ball S of extremal norm and its images AS and BS')
ax.add_patch(S)
ax.add_patch(AS)
ax.add_patch(BS)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.grid(linestyle=":")
ax.set_aspect('equal')
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)

ax.scatter(V[:, 0], V[:, 1], 4, 'k')
labels = ['$v_1$', '$v_2$', '$v_3$', '$v_4$', '$v_5$', '$v_6$',
          '$v_7$', '$v_8$', '$v_9$', '$v_{10}$', '$v_{11}$', '$v_{12}$']
dx = [0.1, 0.1, 0.1, 0.08, 0.0, -0.1,
      -0.1, -0.08, -0.11, 0.0, 0.0, 0.12]
dy = [0.0, -0.06, 0.0, -0.08, -0.09, 0.0,
      0.0, 0.08, 0.0, 0.1, 0.09, 0.0]
for i, label in enumerate(labels):
    ax.text(V[i, 0] + dx[i], V[i, 1] + dy[i], label,
            ha='center', va='center', usetex=True, fontsize='large')

ax.scatter(AV[:, 0], AV[:, 1], 4, 'r')
labels = ['$a_1$', '$a_2$', '$a_3$', '$a_4$', '$a_5$', '$a_6$',
          '$a_7$', '$a_8$', '$a_9$', '$a_{10}$', '$a_{11}$', '$a_{12}$']
adx = [0.07, 0.0, -0.04, -0.09, -0.1, -0.1,
       -0.08, -0.04, 0.06, 0.12, 0.14, 0.13]
ady = [-0.07, -0.09, -0.07, 0.01, 0.0, 0.02,
       0.07, 0.09, 0.08, 0.0, 0.0, 0.0]
for i, label in enumerate(labels):
    ax.text(AV[i, 0] + adx[i], AV[i, 1] + ady[i], label,
            ha='center', va='center', usetex=True, fontsize='large')

ax.scatter(BV[:, 0], BV[:, 1], 4, 'b')
labels = ['$b_1$', '$b_2$', '$b_3$', '$b_4$', '$b_5$', '$b_6$',
          '$b_7$', '$b_8$', '$b_9$', '$b_{10}$', '$b_{11}$', '$b_{12}$']
bdx = [0.12, 0.08, -0.0, -0.06, -0.1, -0.1,
       -0.12, -0.06, 0.03, 0.1, 0.13, 0.13]
bdy = [-0.05, -0.1, -0.1, -0.08, 0.0, 0.05,
       0.05, 0.08, 0.09, 0.06, 0.0, -0.05]
for i, label in enumerate(labels):
    ax.text(BV[i, 0] + bdx[i], BV[i, 1] + bdy[i], label,
            ha='center', va='center', usetex=True, fontsize='large')

fig.savefig(pdfout, bbox_inches='tight', format='pdf', dpi=1200)

plt.show()
