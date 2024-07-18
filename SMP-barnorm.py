# -*- coding: utf-8 -*-
"""Proving SMP for rotation-like matrices.

Created on Sat Sep 21 12:37:46 2019.
Last updated on Mon Jun 17 10:57:45 2024 +0300
Make compatible with Shapely v2.0

@author: Victor Kozyakin
"""
import math
import platform
import time
from importlib.metadata import version

import numpy as np
import shapely
import shapely.affinity
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
from shapely.geometry import LineString, MultiPoint


def polygonal_norm(_x, _y, _h):
    """Calculate the norm specified by a polygonal unit ball.

    Args:
        _x (real): x-coordinate of vector
        _y (real): y-coordinate of vector
        _h (MultiPoint): polygonal norm unit ball

    Returns:
        real: vector's norm
    """
    _hb = _h.bounds
    _scale = 0.5 * math.sqrt(((_hb[2] - _hb[0])**2 + (_hb[3] - _hb[1])**2) /
                             (_x**2 + _y**2))
    _ll = LineString([(0, 0), (_scale * _x, _scale * _y)])
    _p_int = _ll.intersection(_h).coords
    return math.sqrt((_x**2 + _y**2) / (_p_int[1][0]**2 + _p_int[1][1]**2))


def min_max_norms_quotent(_g, _h):
    """Calculate the min/max of the quotient g-norm/h-norm.

    Args:
        _g (MultiPoint): polygonal norm unit ball
        _h (MultiPoint): polygonal norm unit ball

    Returns:
        2x0-array: mimimum and maximum of g-norm/h-norm
    """
    _pg = _g.boundary.coords
    _dimg = len(_pg) - 1
    _sg = [1 / polygonal_norm(_pg[i][0], _pg[i][1], _h)
           for i in range(_dimg)]
    _ph = _h.boundary.coords
    _dimh = len(_ph) - 1
    _sh = [polygonal_norm(_ph[i][0], _ph[i][1], _g) for i in range(_dimh)]
    _sgh = _sg + _sh
    return (min(_sgh), max(_sgh))


def matrix_angular_coord(_a, _t):
    """Calculate the angular coordinate of vector Ax given vector x.

    Args:
        _a (2x2 np.array): input matrix A
        _t (nx1 np.array): array of input angles of x's

    Returns:
        [nx1 np.array]: array of output angles of Ax's
    """
    _cos_t = math.cos(_t)
    _sin_t = math.sin(_t)
    _vec_t = np.asarray([_cos_t, _sin_t])
    _vec_t_transpose = np.transpose(_vec_t)
    _rot_back = np.asarray([[_cos_t, _sin_t],  [-_sin_t, _cos_t]])
    _vec_a = np.matmul(np.matmul(_rot_back, _a), _vec_t_transpose)
    return _t + math.atan2(_vec_a[1], _vec_a[0])


# Initialization

t_tick = time.time()
T_BARNORM_COMP = 0.

TOL = 1e-8
ANGLE_STEP = 0.01
LEN_TRAJECTORY = 10000
NUM_SYMB = 50
L_BOUND = 0.2
U_BOUND = 2.2

THETA = (2. * math.pi) / 3.
COS_THETA = math.cos(THETA)
LAMBDA = 1.1**3

X_INIT_0 = 1.  # 0.5
X_INIT_1 = 1.

A = np.asarray([[0, -1 / LAMBDA], [LAMBDA, 2 * COS_THETA]])
B = np.asarray([[0, -LAMBDA], [1 / LAMBDA, 2 * COS_THETA]])

AT = np.transpose(A)
BT = np.transpose(B)

# Computation initialization

if ((np.linalg.det(A) == 0) or (np.linalg.det(B) == 0)):
    raise SystemExit("Set of matrices is degenerate. End of work!")

INV_A = np.linalg.inv(A)
INV_B = np.linalg.inv(B)
INV_AT = np.transpose(INV_A)
INV_BT = np.transpose(INV_B)

p0 = np.asarray([[1, -1], [1, 1]])
p0 = np.concatenate((p0, -p0), axis=0)
p0 = MultiPoint(p0)
h0 = p0.convex_hull

scale0 = 1 / max(h0.bounds[2], h0.bounds[3])
h0 = shapely.affinity.scale(h0, xfact=scale0, yfact=scale0)

t_ini = time.time() - t_tick

print('\n  #   rho_min    rho    rho_max  Num_edges\n')

# Computation iterations

NITER = 0.
while True:
    t_tick = time.time()

    tmp_geom = np.array(MultiPoint(h0.boundary.coords).geoms)
    tmp_list = []
    for pp in tmp_geom:
        tmp_list.append([pp.x, pp.y])
    p0 = np.array(tmp_list)

    p1 = MultiPoint(np.matmul(p0, INV_AT))
    h1 = p1.convex_hull

    p2 = MultiPoint(np.matmul(p0, INV_BT))
    h2 = p2.convex_hull

    h12 = h1.intersection(h2)
    p12 = MultiPoint(h12.boundary.coords)

    rho_minmax = min_max_norms_quotent(h12, h0)
    rho_max = rho_minmax[1]
    rho_min = rho_minmax[0]

    rho = (rho_max + rho_min) / 2

    h0 = h0.intersection(shapely.affinity.scale(h12, xfact=rho, yfact=rho))
    h0 = h0.simplify(tolerance=TOL)

    T_BARNORM_COMP += (time.time() - t_tick)

    NITER += 1
    print(f'{NITER:3.0f}.', f'{rho_min:.6f}', f'{rho:.6f}', f'{rho_max:.6f}',
          '   ', len(h0.boundary.coords) - 1)

    # Normalizing h0 by fitting it into the unit ball of the Euclidean norm

    scale0 = 1 / math.sqrt(np.max(p0[:, 0]**2+p0[:, 1]**2))
    h0 = shapely.affinity.scale(h0, xfact=scale0, yfact=scale0)

    if (rho_max - rho_min) < TOL:
        break

# Plotting Barabanov norm

t_tick = time.time()

h10 = shapely.affinity.scale(h1, xfact=rho, yfact=rho)
tmp_geom = np.array(MultiPoint(h10.boundary.coords).geoms)
tmp_list = []
for pp in tmp_geom:
    tmp_list.append([pp.x, pp.y])
p10 = np.array(tmp_list)


h20 = shapely.affinity.scale(h2, xfact=rho, yfact=rho)
tmp_geom = np.array(MultiPoint(h20.boundary.coords).geoms)
tmp_list = []
for pp in tmp_geom:
    tmp_list.append([pp.x, pp.y])
p20 = np.array(tmp_list)


bb = max(h0.bounds[2], h10.bounds[2], h20.bounds[2],
         h0.bounds[3], h10.bounds[3], h20.bounds[3])

pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

# =================================================================
# Tuning the LaTex preamble (e.g. for international support)
#
# pyplot.rcParams['text.latex.preamble'] = \
#     r'\usepackage[utf8]{inputenc}' + '\n' + \
#     r'\usepackage[russian]{babel}' + '\n' + \
#     r'\usepackage{amsmath}'
# =================================================================

# Plotting Barabanov's norm

fig1 = pyplot.figure(num="Barabanov norm")
ax1 = fig1.add_subplot(111)
ax1.set_xlim(-1.1 * bb, 1.1 * bb)
ax1.set_ylim(-1.1 * bb, 1.1 * bb)
ax1.set_aspect(1)
ax1.grid(True, linestyle=":")
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(1))

ax1.plot(p10[:, 0], p10[:, 1], ':', color='red', linewidth=1.25)
ax1.plot(p20[:, 0], p20[:, 1], '--', color='blue', linewidth=1)
ax1.plot(p0[:, 0], p0[:, 1], '-', color='black')

# Plotting lines of intersection of norms' unit spheres

pl10 = LineString(p10)
pl20 = LineString(p20)
h_int = shapely.affinity.scale(pl10.intersection(pl20), xfact=3, yfact=3)
tmp_geom = np.array(h_int.geoms)
tmp_list = []
for pp in tmp_geom:
    tmp_list.append([pp.x, pp.y])
p_int = np.array(tmp_list)

arr_switch_N = np.size(p_int[:, 0])
arr_switch_ang = np.empty(arr_switch_N)
for i in range(np.size(p_int[:, 0])):
    arr_switch_ang[i] = math.atan2(p_int[i, 1], p_int[i, 0])
    if arr_switch_ang[i] < 0:
        arr_switch_ang[i] = arr_switch_ang[i] + 2. * math.pi
    if p_int[i, 0] >= 0:
        ax1.plot([2 * p_int[i, 0], -2 * p_int[i, 0]],
                 [2 * p_int[i, 1], -2 * p_int[i, 1]],
                 dashes=[5, 2, 1, 2], color='green', linewidth=1)

t_plot_fig1 = time.time() - t_tick
pyplot.show()

# Plotting image of Barabanov norm's unit sphere under A and B action

t_tick = time.time()

AT0 = (1./rho_min) * AT
p10 = np.array(np.matmul(p0, AT0))

BT0 = (1./rho_min) * BT
p20 = np.array(np.matmul(p0, BT0))

# Plotting Barabanov's norm

fig2 = pyplot.figure(
    num="Unit ball S of Barabanov norm and its images AS and BS")
ax2 = fig2.add_subplot(111)
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax2.set_aspect(1)
ax2.grid(True, linestyle=":")
ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.yaxis.set_major_locator(MultipleLocator(1))

ax2.fill(p0[:, 0], p0[:, 1], '-', color='black', facecolor='#f0f0f0')
ax2.plot(p10[:, 0], p10[:, 1], '--', color='red')
ax2.plot(p20[:, 0], p20[:, 1], '--', color='blue')
ax2.scatter(p0[:, 0], p0[:, 1], color='black', s=10)


t_plot_fig2 = time.time() - t_tick
pyplot.show()

# Plotting an extremal trajectory

t_tick = time.time()

fig3 = pyplot.figure(num="Maximum growth rate trajectory")
ax3 = fig3.add_subplot(111)
bbb = 1.5*bb
ax3.set_xlim(-bbb, bbb)
ax3.set_ylim(-bbb, bbb)
ax3.set_aspect(1)
ax3.grid(True, linestyle=":")
ax3.xaxis.set_major_locator(MultipleLocator(1))
ax3.yaxis.set_major_locator(MultipleLocator(1))

# Plotting lines of intersection of norms' unit spheres

arr_switch_N = np.size(p_int[:, 0])
arr_switch_ang = np.empty(arr_switch_N)
for i in range(np.size(p_int[:, 0])):
    arr_switch_ang[i] = math.atan2(p_int[i, 1], p_int[i, 0])
    if arr_switch_ang[i] < 0:
        arr_switch_ang[i] = arr_switch_ang[i] + 2. * math.pi
    if p_int[i, 0] >= 0:
        ax3.plot([2 * p_int[i, 0], -2 * p_int[i, 0]],
                 [2 * p_int[i, 1], -2 * p_int[i, 1]],
                 dashes=[5, 2, 1, 2], color='green', linewidth=1)


# Plotting the trajectory

x = np.asarray([X_INIT_0, X_INIT_1])

if rho > 1:
    x = (L_BOUND / polygonal_norm(x[0], x[1], h0)) * x
else:
    x = (U_BOUND / polygonal_norm(x[0], x[1], h0)) * x

for i in range(LEN_TRAJECTORY):
    xprev = x
    x0 = np.matmul(x, AT)
    x1 = np.matmul(x, BT)
    if (polygonal_norm(x0[0], x0[1], h0) >
            polygonal_norm(x1[0], x1[1], h0)):
        x = x0
        ax3.arrow(xprev[0], xprev[1], x[0] - xprev[0], x[1] - xprev[1],
                  head_width=0.04, head_length=0.08, linewidth=0.75,
                  color='red', length_includes_head=True, zorder=-i)
    else:
        x = x1
        ax3.arrow(xprev[0], xprev[1], x[0] - xprev[0], x[1] - xprev[1],
                  head_width=0.04, head_length=0.08, linewidth=0.75,
                  color='blue', length_includes_head=True, zorder=-i)
    if ((polygonal_norm(x[0], x[1], h0) > U_BOUND) or
            (polygonal_norm(x[0], x[1], h0) < L_BOUND)):
        break

arr_switch_ang.sort()
ISPLIT = 0
for i in range(np.size(arr_switch_ang)):
    if arr_switch_ang[i] < math.pi:
        ISPLIT = i

arr_switch_ang = np.resize(arr_switch_ang, ISPLIT + 1)
arr_switch_N = np.size(arr_switch_ang)
arr_switches = np.insert(arr_switch_ang, 0, 0)
arr_switches = np.append(arr_switches, math.pi)
omegB = 0.8 * arr_switches[1] + 0.2 * arr_switches[2]
omega2 = omegB + math.pi / 2.
omega3 = omega2 + math.pi / 2.
omega4 = omega3 + math.pi / 2.
props = dict(boxstyle='round', facecolor='gainsboro', edgecolor='none',
             alpha=0.5)
p_label = np.array([math.cos(omegB), math.sin(omegB)])

if (polygonal_norm(p_label[0], p_label[1], h10) >
        polygonal_norm(p_label[0], p_label[1], h20)):
    ax3.text(0.9 * bbb * math.cos(omegB), 0.9 * bbb * math.sin(omegB),
             r'$x_{n+1}=A x_n$', ha='center', va='center',
             fontsize='large', bbox=props)
    ax3.text(0.9 * bbb * math.cos(omega2), 0.9 * bbb * math.sin(omega2),
             r'$x_{n+1}=B x_n$', ha='center', va='center',
             fontsize='large', bbox=props)
    ax3.text(0.9 * bbb * math.cos(omega3), 0.9 * bbb * math.sin(omega3),
             r'$x_{n+1}=A x_n$', ha='center', va='center',
             fontsize='large', bbox=props)
    ax3.text(0.9 * bbb * math.cos(omega4), 0.9 * bbb * math.sin(omega4),
             r'$x_{n+1}=B x_n$', ha='center', va='center',
             fontsize='large', bbox=props)
else:
    ax3.text(0.9 * bbb * math.cos(omegB), 0.9 * bbb * math.sin(omegB),
             r'$x_{n+1}=B x_n$', ha='center', va='center',
             fontsize='large', bbox=props)
    ax3.text(0.9 * bbb * math.cos(omega2), 0.9 * bbb * math.sin(omega2),
             r'$x_{n+1}=A x_n$', ha='center', va='center',
             fontsize='large', bbox=props)
    ax3.text(0.9 * bbb * math.cos(omega3), 0.9 * bbb * math.sin(omega3),
             r'$x_{n+1}=B x_n$', ha='center', va='center',
             fontsize='large', bbox=props)
    ax3.text(0.9 * bbb * math.cos(omega4), 0.9 * bb * math.sin(omega4),
             r'$x_{n+1}=A x_n$', ha='center', va='center',
             fontsize='large', bbox=props)

t_plot_fig3 = time.time() - t_tick
pyplot.show()

# Calculating matrix sequence

t_tick = time.time()

F0 = 0.
F1 = 0.
F00 = 0.
F01 = 0.
F10 = 0.
F11 = 0.
x = np.asarray([X_INIT_0, X_INIT_1])
matrix_seq = []

for i in range(LEN_TRAJECTORY):
    x = x / polygonal_norm(x[0], x[1], h0)
    x0 = np.matmul(x, AT)
    x1 = np.matmul(x, BT)
    if (polygonal_norm(x0[0], x0[1], h0) >
            polygonal_norm(x1[0], x1[1], h0)):
        x = x0
        matrix_seq.append('A')
        F0 += 1
    else:
        x = x1
        matrix_seq.append('B')
        F1 += 1
    if i > 0:
        if ((matrix_seq[i - 1] == 'A') and (matrix_seq[i] == 'A')):
            F00 += 1
        if ((matrix_seq[i - 1] == 'A') and (matrix_seq[i] == 'B')):
            F01 += 1
        if ((matrix_seq[i - 1] == 'B') and (matrix_seq[i] == 'A')):
            F10 += 1
        if ((matrix_seq[i - 1] == 'B') and (matrix_seq[i] == 'B')):
            F11 += 1

print('\nExtremal matrix sequence: ', end='')
for i in range(NUM_SYMB):
    print(matrix_seq[i], end='')

print('\n\nFrequences of matrices A, B, AA, AB etc. in the matrix sequence:',
      '\n\nMatrices:       A      B      AA     AB     BA     BB')

print('Frequences: ',
      f' {round(F0 / LEN_TRAJECTORY, 3):.3f}',
      f' {round(F1 / LEN_TRAJECTORY, 3):.3f}',
      f' {round(F00 / (LEN_TRAJECTORY - 1), 3):.3f}',
      f' {round(F01 / (LEN_TRAJECTORY - 1), 3):.3f}',
      f' {round(F10 / (LEN_TRAJECTORY - 1), 3):.3f}',
      f' {round(F11 / (LEN_TRAJECTORY - 1), 3):.3f}')
np.set_printoptions(suppress=True)
print(p0)

t_matrix_seq = time.time() - t_tick

# Saving plots to pdf-files

"""
fig1.savefig(f'bnorm-{THETA:.2f}-{THETA:.2f}-{LAMBDA:.2f}.pdf',
             bbox_inches='tight')
fig2.savefig(f'sball-{THETA:.2f}-{THETA:.2f}-{LAMBDA:.2f}.pdf',
             bbox_inches='tight')
fig3.savefig(f'etraj-{THETA:.2f}-{THETA:.2f}-{LAMBDA:.2f}.pdf',
             bbox_inches='tight')
"""

# Computation timing

t_compute = T_BARNORM_COMP + t_matrix_seq
t_plot = t_plot_fig1 + t_plot_fig2 + t_plot_fig3
t_total = t_ini + t_plot + t_compute


print('\nInitialization: ', f'{round(t_ini, 6):6.2f} sec.')
print('Computations:   ', f'{round(t_compute, 6):6.2f} sec.')
print('Plotting:       ', f'{round(t_plot, 6):6.2f} sec.')
print('Total:          ', f'{round(t_total, 6):6.2f} sec.')

print('\nModules used:  Python ' + platform.python_version(),
      'matplotlib ' + version('matplotlib'),
      'numpy ' + version('numpy'),
      'shapely ' + version('shapely'), sep=', ')
