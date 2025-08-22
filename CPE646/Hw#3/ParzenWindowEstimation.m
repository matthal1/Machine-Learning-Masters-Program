% === Load Data ===
load('hw3.mat');  % class w1: variable hw3_2_1 (Nx2)

X1 = hw3_2_1';   % samples from class ω1
X2 = hw3_2_2';   % samples from class ω2

h = 2;          % Parzen window width
d = 2;          % data dimension

% === Grid Definition ===
[x1, x2] = meshgrid(-4:0.1:8, -4:0.1:8);
grid_points = [x1(:), x2(:)];   % (P×2) matrix where each row is a point

% === Gaussian Kernel Function ===
gaussian_kernel = @(u) (1/(2*pi)^(d/2)) * exp(-0.5 * sum(u.^2, 2));

% === Estimate p(x|w1) and p(x|w2) over the grid ===
p_w1 = parzen_estimate(grid_points, X1, h);
p_w2 = parzen_estimate(grid_points, X2, h);

% === Reshape and Plot ===
p_w1_grid = reshape(p_w1, size(x1));
p_w2_grid = reshape(p_w2, size(x2));

figure;
subplot(1,2,1);
mesh(x1, x2, p_w1_grid);
title('p(x|\omega_1)');
xlabel('x1'); ylabel('x2'); zlabel('Density');

subplot(1,2,2);
mesh(x1, x2, p_w2_grid);
title('p(x|\omega_2)');
xlabel('x1'); ylabel('x2'); zlabel('Density');

% === Classify x = [1; -2] ===
x_test = [1, -2];
p1_x = parzen_estimate(x_test, X1, h);
p2_x = parzen_estimate(x_test, X2, h);

if p1_x > p2_x
    fprintf('Parzen Classifier: x = [1; -2] classified as class ω1\n');
else
    fprintf('Parzen Classifier: x = [1; -2] classified as class ω2\n');
end


% === Parzen Estimation Function ===
function p = parzen_estimate(grid_points, data, h)
    n = size(data, 1);
    m = size(grid_points, 1);
    p = zeros(m, 1);
    for i = 1:m
        diff = (data - grid_points(i,:)) / h;  % n×2
        kernel_vals = exp(-0.5 * sum(diff.^2, 2));
        p(i) = sum(kernel_vals);
    end
    p = p / (n * h^2 * (2*pi));
end