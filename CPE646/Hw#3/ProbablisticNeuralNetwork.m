% === Load data (if not already loaded) ===
load('hw3.mat');  % X1 = hw3_2_1, X2 = hw3_2_2

X1 = hw3_2_1;    % 2×N1
X2 = hw3_2_2;    % 2×N2

% === Classification point and sigma ===
x = [1; -2];     % 2×1
sigma = 0.2;

% === Convert to sample-wise columns for loop-friendly code ===
N1 = size(X1, 2);
N2 = size(X2, 2);

% === Compute g1(x) ===
g1 = 0;
for i = 1:N1
    diff = x - X1(:, i);
    g1 = g1 + exp(- (diff' * diff) / (2 * sigma^2));
end

% === Compute g2(x) ===
g2 = 0;
for i = 1:N2
    diff = x - X2(:, i);
    g2 = g2 + exp(- (diff' * diff) / (2 * sigma^2));
end

% === Decision ===
if g1 > g2
    fprintf('PNN Classifier: x = [1; -2] is classified as class ω1\n');
else
    fprintf('PNN Classifier: x = [1; -2] is classified as class ω2\n');
end