%   SMP_JSRToolbox.m
%   Created by V. Kozyakin on 17.06.2024 10:09:40
%
%   This program depends on Optimization Toolbox & JSR Toolbox
%
%   For compatibility with MATLAB R2023a and later releases, the
%   patched version of jsr_norm_balancedRealPolytope.m from JSR Toolbox
%   should be placed in the home directory of this program
%

%% Initialization

% Closing of all graphs, clearing of all variables and command window
close all;
clc;
commandwindow;

theta = (2 * pi) / 3;
cos_a = cos(theta);
sin_a = sin(theta);
kappa = 1.1^3;

A0 = [0, -1 / kappa; kappa, 2 * cos_a];
B0 = [0, -kappa; 1 / kappa, 2 * cos_a];

C = A0 * B0 * A0;
[e, lambda] = eigs(C, 1);
M = {A0, B0};
M = cellDivide(M, lambda^(1 / 3));

A = M{1};
B = M{2};
C = A * B * A;
K = 3;

% Computation
[BOUNDS, V, INFO] = jsr_norm_balancedRealPolytope(M, C, K);

% Normalizing B by fitting it into the unit ball of the Euclidean norm
V = V / sqrt(max(V(:, 1).*V(:, 1)+V(:, 2).*V(:, 2)));

% Cyclicaly reorder verices from V.
S = convhull(polyshape(V));
V = S.Vertices;
%
fprintf('Normalized cyclically ordered list of vertices:\n\n');
fprintf('{%9.6f, %9.6f}\n', V.');

% Plotting
AV = V * transpose(A);
AS = convhull(polyshape(AV));

BV = V * transpose(B);
BS = convhull(polyshape(BV));

plot(S, 'FaceColor', '#e0e0e0', 'LineWidth', 1);
hold on
plot(AS, 'LineStyle', '--', 'EdgeColor', 'red', 'FaceColor', 'none');
hold on
plot(BS, 'LineStyle', '-.', 'EdgeColor', 'blue', 'FaceColor', 'none');
xticks([-1, 0, 1]);
yticks([-1, 0, 1]);
hold on

Vx = transpose(V(:, 1));
Vy = transpose(V(:, 2));
scatter(Vx, Vy, 8, 'black', 'filled');
sss = {'$v_7$', '$v_8$', '$v_9$', '$v_{10}$', '$v_{11}$', '$v_{12}$', ...
    '$v_1$', '$v_2$', '$v_3$', '$v_4$', '$v_5$', '$v_6$'};
dx = [-0.1, -0.1, -0.08, 0.0, 0.0, 0.04, ...
    0.04, 0.04, 0.025, 0.0, 0.0, -0.125];
dy = [0.0, 0.0, 0.05, 0.06, 0.05, 0.0, ...
    0.0, 0.0, -0.05, -0.05, -0.05, 0];
text(Vx+dx, Vy+dy, sss, 'Interpreter', 'latex');
hold on

AV = AS.Vertices;
AVx = transpose(AV(:, 1));
AVy = transpose(AV(:, 2));
scatter(AVx, AVy, 8, 'red', 'filled');
sss = {'$a_7$', '$a_8$', '$a_9$', '$a_{10}$', '$a_{11}$', '$a_{12}$', ...
    '$a_1$', '$a_2$', '$a_3$', '$a_4$', '$a_5$', '$a_6$'};
adx = [-0.075, -0.1, 0.0, 0.03, 0.025, 0.025, ...
    0.025, 0.0, -0.075, -0.125, -0.125, -0.125];
ady = [0.05, 0.05, 0.06, 0.0, 0.0, 0.0, ...
    -0.05, -0.05, -0.05, 0.015, 0.0, 0];
text(AVx+adx, AVy+ady, sss, 'Interpreter', 'latex');
hold on

BV = BS.Vertices;
BVx = transpose(BV(:, 1));
BVy = transpose(BV(:, 2));
scatter(BVx, BVy, 8, 'blue', 'filled');
sss = {'$b_7$', '$b_8$', '$b_9$', '$b_{10}$', '$b_{11}$', '$b_{12}$', ...
    '$b_1$', '$b_2$', '$b_3$', '$b_4$', '$b_5$', '$b_6$'};
bdx = [-0.125, -0.1, -0.03, 0.0, 0.025, 0.025, ...
    0.05, 0.0, -0.035, -0.0875, -0.125, -0.125];
bdy = [0.05, 0.05, 0.06, 0.05, 0.0, -0.05, ...
    -0.05, -0.06, -0.06, -0.06, 0.0, 0.025];
text(BVx+bdx, BVy+bdy, sss, 'Interpreter', 'latex');
grid on
pbaspect([1, 1, 1]);
axis([-1.1, 1.1, -1.1, 1.1]);

exportgraphics(gcf, 'SMP-JSRToolbox.pdf', 'Resolution', 1200);

ss = get(0, 'ScreenSize');
wx = 640;
wy = 485;
set(gcf, 'Position', [ss(3) - wx - 10, ss(4) - wy - 90, wx, wy]);

title('Unit ball S of extremal norm and its images AS and BS', ...
    '(Computed by JSR Toolbox)');

legend({'$$~~S=\{x:\|x\|\le1\}$$', '$$~~AS$$', '$$~~BS$$'}, ...
    'Interpreter', 'latex', 'Location', 'BestOutside');