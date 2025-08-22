% =============================
% Load and prepare the data
% =============================
load('hw4.mat');  % hw4_2_1 and hw4_2_2: each is 2×100

X1 = hw4_2_1';  % 100×2 (class ω1)
X2 = hw4_2_2';  % 100×2 (class ω2)

% =============================
% 2.1 Plot original data in 2D
% =============================
figure;
plot(X1(:,1), X1(:,2), 'ro', 'DisplayName', '\omega_1'); hold on;
plot(X2(:,1), X2(:,2), 'bs', 'DisplayName', '\omega_2');
xlabel('x_1'); ylabel('x_2');
title('2D Data Plot');
legend;
grid on;

% =============================
% 2.2 φ Mapping and 3D Plot
% =============================
phi1 = [X1, X1(:,1).*X1(:,2)];  % φ(x) = [x1, x2, x1*x2]
phi2 = [X2, X2(:,1).*X2(:,2)];

figure;
plot3(phi1(:,1), phi1(:,2), phi1(:,3), 'ro', 'DisplayName', '\omega_1'); hold on;
plot3(phi2(:,1), phi2(:,2), phi2(:,3), 'bs', 'DisplayName', '\omega_2');
xlabel('x_1'); ylabel('x_2'); zlabel('x_1 x_2');
title('3D φ(x) Mapping');
legend;
grid on;
view(45, 30);  % Better view angle

% =============================
% 2.3 Batch Perceptron in φ space
% =============================

% Augment each sample: y(x) = [1; x1; x2; x1*x2]
aug1 = [ones(100,1), X1(:,1), X1(:,2), X1(:,1).*X1(:,2)];  % class ω1 → +1
aug2 = [ones(100,1), X2(:,1), X2(:,2), X2(:,1).*X2(:,2)];  % class ω2 → -1

aug2 = -aug2;  % negate samples from class ω2
Y = [aug1; aug2];  % total: 200×4

% Initialization
a = sum(Y, 1)';  % a(0) = sum of all y
eta = 1;         % learning rate
theta = 1;       % threshold
max_iter = 100;

% Batch Perceptron training loop
for epoch = 1:max_iter
    misclassified = false;
    delta_a = zeros(size(a));
    
    for i = 1:size(Y,1)
        y = Y(i,:)';
        if a' * y <= theta
            delta_a = delta_a + y;
            misclassified = true;
        end
    end

    a = a + eta * delta_a;

    if ~misclassified
        fprintf('Batch Perceptron converged at epoch %d\n', epoch);
        break;
    end
end

disp('Final weight vector a =');
disp(a);
