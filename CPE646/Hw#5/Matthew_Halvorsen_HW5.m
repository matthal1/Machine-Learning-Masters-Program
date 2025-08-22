function hw5_problem1
%% 1) Load & visualize
clc; clear; close all;

load('hw5.mat', 'hw5_1', 'hw5_2'); % each 2x100: columns are samples

X1 = hw5_1;                 % class ω1
X2 = hw5_2;                 % class ω2
X  = [X1, X2];              % 2 x N
N1 = size(X1,2); N2 = size(X2,2); N = N1+N2;

% Plot data (1.1)
figure; hold on; grid on; box on;
plot(X1(1,:), X1(2,:), 'bo', 'MarkerFaceColor','b');
plot(X2(1,:), X2(2,:), 'rs', 'MarkerFaceColor','r');
xlabel('x_1'); ylabel('x_2'); title('Problem 1: Data in 2D');
legend('\omega_1','\omega_2');

%% 2) Prep targets & standardize
% Two-output coding per statement:
t_w1 = [ 1; -1];
t_w2 = [-1;  1];

T = [repmat(t_w1,1,N1), repmat(t_w2,1,N2)];     % 2 x N target matrix

% Standardize inputs: z = (x - mean) ./ std  (feature-wise over ALL samples)
mu  = mean(X, 2);                 % 2x1
sig = std(X, 0, 2);               % 2x1
sig(sig==0) = 1;                  % guard
Z = (X - mu) ./ sig;              % 2 x N standardized inputs

%%  3) Network spec
nin  = 2;
nh   = 10;          % hidden nodes
nout = 2;

a = 1.716;          % activation scale
b = 2/3;            % activation slope
eta = 0.1;          % step size (learning rate)
mse_goal = 0.1;     % convergence criterion
max_epochs = 10000;

% Random uniform init (small) — fan-in aware
rng(0);
lim1 = sqrt(6/(nin+nh));
lim2 = sqrt(6/(nh+nout));
W1 = (rand(nh, nin)*2-1) * lim1;    b1 = (rand(nh,1)*2-1) * lim1;
W2 = (rand(nout, nh)*2-1) * lim2;   b2 = (rand(nout,1)*2-1) * lim2;

%% 4) Stochastic backprop training
idx_all = 1:N;
mse_hist = zeros(max_epochs,1);

for epoch = 1:max_epochs
    % shuffle each epoch
    idx = idx_all(randperm(N));

    sse = 0;
    for k = idx
        x = Z(:,k);         % 2x1
        t = T(:,k);         % 2x1

        % ---- forward pass ----
        n1 = W1*x + b1;                 % nhx1
        t1 = tanh(b*n1);                % nhx1
        y1 = a * t1;                    

        n2 = W2*y1 + b2;                % noutx1
        t2 = tanh(b*n2);                % noutx1
        y  = a * t2;                    % network output

        e = y - t;                      % error (noutx1)
        sse = sse + 0.5*sum(e.^2);

        % ---- backprop (using derivatives of a*tanh(b*net)) ----
        % dy/dn = a*b*(1 - tanh(b*net)^2) = a*b*(1 - t.^2) using t=tanh(b*net)
        delta2 = e .* (a*b*(1 - t2.^2));      % noutx1
        gradW2 = delta2 * y1';                % nout x nh
        gradb2 = delta2;                      

        delta1 = (W2' * delta2) .* (a*b*(1 - t1.^2));   % nhx1
        gradW1 = delta1 * x';                 % nh x nin
        gradb1 = delta1;

        % ---- SGD update ----
        W2 = W2 - eta * gradW2;   b2 = b2 - eta * gradb2;
        W1 = W1 - eta * gradW1;   b1 = b1 - eta * gradb1;
    end

    mse = sse / N;
    mse_hist(epoch) = mse;

    if mod(epoch,100)==0 || epoch==1
        fprintf('Epoch %5d | MSE = %.4f\n', epoch, mse);
    end
    if mse < mse_goal
        fprintf('Converged at epoch %d with MSE=%.4f\n', epoch, mse);
        mse_hist = mse_hist(1:epoch);
        break;
    end
end

figure; plot(mse_hist,'LineWidth',1.5); grid on; xlabel('Epoch'); ylabel('MSE'); title('Training MSE');

%%  5) Classify test vectors (1.3)
Dtest = [  2   -3   -2    3;
           2   -3    5   -4 ];      % each column is a test vector

% standardize test using TRAIN stats
Dtest_z = (Dtest - mu) ./ sig;

Ytest = zeros(nout, size(Dtest,2));
for k = 1:size(Dtest_z,2)
    x = Dtest_z(:,k);
    % forward only
    n1 = W1*x + b1;  t1 = tanh(b*n1);  y1 = a*t1;
    n2 = W2*y1 + b2; t2 = tanh(b*n2);  y  = a*t2;
    Ytest(:,k) = y;
end

% decision rule: closer (Euclidean) to [1;-1] => ω1, closer to [-1;1] => ω2
d1 = vecnorm(Ytest - t_w1, 2, 1);
d2 = vecnorm(Ytest - t_w2, 2, 1);
pred = ones(1,size(Dtest,2));           % 1 for ω1
pred(d2 < d1) = 2;                      % 2 for ω2

fprintf('\n=== Test results ===\n');
for k = 1:size(Dtest,2)
    fprintf('x = [%4.1f, %4.1f]^T  ->  y = [% .3f, % .3f]^T  ->  class ω%d\n', ...
        Dtest(1,k), Dtest(2,k), Ytest(1,k), Ytest(2,k), pred(k));
end
