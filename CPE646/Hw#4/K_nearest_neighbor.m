% Load data
load('hw3.mat');  % Assuming variables are hw3_2_1 and hw3_2_2

X1 = hw3_2_1';  % class ω1 samples
X2 = hw3_2_2';  % class ω2 samples

% Parameters
k = 10;  % number of nearest neighbors
x_range = -4:0.1:8;
[y1, y2] = meshgrid(x_range, x_range);
[m, n] = size(y1);
p_w1 = zeros(m, n);
p_w2 = zeros(m, n);

% KNN Density Estimation
for i = 1:m
    for j = 1:n
        x = reshape([y1(i,j), y2(i,j)], 1, []);  % ensure row vector

        % Distance to class ω1 samples
        d1 = sqrt(sum((X1 - x).^2, 2));
        d1 = sort(d1);
        V1 = pi * d1(k)^2;  % area of circle with radius to kth neighbor
        p_w1(i,j) = k / (size(X1,1) * V1);
        
        % Distance to class ω2 samples
        d2 = sqrt(sum((X2 - x).^2, 2));
        d2 = sort(d2);
        V2 = pi * d2(k)^2;
        p_w2(i,j) = k / (size(X2,1) * V2);
    end
end

% Plotting the densities
figure;
mesh(x_range, x_range, p_w1);
title('p(x|ω₁) estimated using kNN');
xlabel('x_1'); ylabel('x_2'); zlabel('Density');

figure;
mesh(x_range, x_range, p_w2);
title('p(x|ω₂) estimated using kNN');
xlabel('x_1'); ylabel('x_2'); zlabel('Density');

% Classify x = [1, -2]^T
x_test = [1, -2];

% Estimate density at x_test
d1 = sqrt(sum((X1 - x_test).^2, 2));
d1 = sort(d1);
V1 = pi * d1(k)^2;
px_w1 = k / (size(X1,1) * V1);

d2 = sqrt(sum((X2 - x_test).^2, 2));
d2 = sort(d2);
V2 = pi * d2(k)^2;
px_w2 = k / (size(X2,1) * V2);

fprintf('p(x|w1) = %.5f\n', px_w1);
fprintf('p(x|w2) = %.5f\n', px_w2);

if px_w1 > px_w2
    fprintf('Classified as ω1\n');
else
    fprintf('Classified as ω2\n');
end