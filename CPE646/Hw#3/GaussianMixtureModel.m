% === Load Data ===
load('hw3.mat');     % Contains hw3_1: 2×100
X = hw3_1;           % X is 2×100
X_data = X';         % Transpose to 100×2 (samples as rows)
N = size(X_data, 1); % Number of samples (100)

% === Initialization ===
rho = 0.5;

% Initial parameter estimates from first 50 and last 50 samples
X1 = X(:, 1:50);         % 2×50
X2 = X(:, 51:100);       % 2×50

mu1 = mean(X1, 2);       % 2×1
mu2 = mean(X2, 2);       % 2×1

Sigma1 = cov(X1');       % 2×2
Sigma2 = cov(X2');       % 2×2

% === EM Parameters ===
max_iters = 100;
tol = 1e-6;
log_likelihoods = zeros(max_iters, 1);
epsilon = 1e-6;  % regularization term for numerical stability

% === EM Algorithm ===
for iter = 1:max_iters
    % E-step: Compute soft assignments (responsibilities)
    p1 = mvnpdf(X_data, mu1', Sigma1);
    p2 = mvnpdf(X_data, mu2', Sigma2);

    gamma1 = rho * p1;
    gamma2 = (1 - rho) * p2;
    gamma_sum = gamma1 + gamma2;

    w1 = gamma1 ./ gamma_sum;
    w2 = gamma2 ./ gamma_sum;

    % Ensure column vectors
    w1 = w1(:);
    w2 = w2(:);

    % M-step: Update parameters
    N1 = sum(w1);
    N2 = sum(w2);

    mu1 = (X_data' * w1) / N1;
    mu2 = (X_data' * w2) / N2;

    % Centered data
    X_centered1 = X_data - mu1';
    X_centered2 = X_data - mu2';

    % Covariance update (loop version for clarity and stability)
    Sigma1 = zeros(2, 2);
    Sigma2 = zeros(2, 2);
    for i = 1:N
        Sigma1 = Sigma1 + w1(i) * (X_centered1(i,:)' * X_centered1(i,:));
        Sigma2 = Sigma2 + w2(i) * (X_centered2(i,:)' * X_centered2(i,:));
    end
    Sigma1 = Sigma1 / N1;
    Sigma2 = Sigma2 / N2;

    % Add regularization to prevent singular matrices
    Sigma1 = Sigma1 + epsilon * eye(2);
    Sigma2 = Sigma2 + epsilon * eye(2);

    % Update rho
    rho = N1 / N;

    % Compute log-likelihood (optional)
    log_likelihood = sum(log(rho * p1 + (1 - rho) * p2));
    log_likelihoods(iter) = log_likelihood;

    % Convergence check
    if iter > 1 && abs(log_likelihoods(iter) - log_likelihoods(iter - 1)) < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
end

% === Final Output ===
fprintf('\nFinal parameter estimates:\n');
disp('mu1 ='); disp(mu1);
disp('mu2 ='); disp(mu2);
disp('Sigma1 ='); disp(Sigma1);
disp('Sigma2 ='); disp(Sigma2);
disp(['rho = ', num2str(rho)]);