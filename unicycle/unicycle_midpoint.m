function unicycle_se2_midpoint
clc; close all;

% Time-scaled SE(2) Lie–Poisson model 
% We use H(μ) = 1/2 μ'Qμ + b'μ, so gradH(μ) = Qμ + b.
% For time-scaled dynamics (θ as clock), choose:
%   Q = diag([0, 1, 0])   and   b = [1; 0; 0]
% so that ω = ∂H/∂μ1 = 1, v_x = ∂H/∂μ2 = μ2, v_y = 0.

Q = diag([0, 1, 0]);
b = [1; 0; 0];

% Initial coadjoint state (choose realistic magnitudes)
mu0 = [1; 10; 1];     % [μ1; μ2; μ3], note μ1 is not the angular rate here; ω≡1 comes from b

% Initial pose g0 in SE(2) (homogeneous 3x3): (x,y,theta) = (0,0,0)
g0 = eye(3);

% Two runs (coarse vs fine)
run_case(mu0, g0, Q, b, 0.05,  400, 1);   % coarse: dt=0.05, T=20 s
run_case(mu0, g0, Q, b, 0.005, 4000, 2);  % fine:   dt=0.005, T=20 s
end


function run_case(mu0, g0, Q, b, dt, N, figbase)
T = N*dt;
fprintf('SE(2) midpoint (time-scaled): dt = %.6g, N = %d (T=%.3g s)\n', dt, N, T);

% Storage
t  = (0:N)*dt;
mu = zeros(3,N+1);     mu(:,1) = mu0;
g  = zeros(3,3,N+1);   g(:,:,1) = g0;

% Invariants
H = zeros(1,N+1);
C = zeros(1,N+1);
[H(1), C(1)] = invariants(mu0, Q, b);

% Time stepping (implicit midpoint on μ; exact exp reconstruction on g)
for k = 1:N
    muk = mu(:,k);

    % implicit midpoint step on μ via damped Newton
    munext = newton_midpoint_mu(muk, dt, Q, b);

    % pose reconstruction in SE(2)
    mubar = 0.5*(muk + munext);
    w     = gradH(mubar, Q, b);            % body twist xi = [ω; v_x; v_y]
    Xi    = hat_se2(w);                    % se(2) matrix
    g(:,:,k+1) = g(:,:,k) * expm(dt*Xi);   % left-invariant update

    % store
    mu(:,k+1) = munext;
    [H(k+1), C(k+1)] = invariants(munext, Q, b);
end

% Diagnostics & plots
driftH = (max(H)-min(H)) / max(1,abs(H(1))) * 100;
driftC = (max(C)-min(C)) / max(1,abs(C(1))) * 100;
fprintf('  Energy drift:  %.6g %%\n', driftH);
fprintf('  Casimir drift: %.6g %%\n', driftC);

% Extract pose (x,y,theta) from g
x  = squeeze(g(1,3,:));
y  = squeeze(g(2,3,:));
th = atan2( squeeze(g(2,1,:)), squeeze(g(1,1,:)) );

% Figures
figure(figbase*10+1); clf;
plot(t, mu(1,:), 'LineWidth', 1.2); hold on;
plot(t, mu(2,:), 'LineWidth', 1.2);
plot(t, mu(3,:), 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('\mu components');
legend('\mu_1','\mu_2','\mu_3','Location','best');
title(sprintf('SE(2) momenta (dt=%.3g, T=%.1f s)', dt, T)); grid on;

figure(figbase*10+2); clf;
plot(t, H, 'LineWidth', 1.2); hold on;
plot(t, C, 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Invariants');
legend('H = \mu_1 + \tfrac{1}{2}\mu_2^2','C = \tfrac{1}{2}(\mu_2^2+\mu_3^2)','Location','best');
title(sprintf('Invariants (midpoint)  drift: H %.3g%%, C %.3g%%', driftH, driftC));
grid on;

figure(figbase*10+3); clf;
plot(x, y, '-', 'LineWidth', 1.2);
axis equal; grid on;
xlabel('x (m)'); ylabel('y (m)');
title(sprintf('Planar trajectory (dt=%.3g, T=%.1f s)', dt, T));

figure(figbase*10+4); clf;
plot(t, th, 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('\theta (rad)');
title('\theta(t) from SE(2) reconstruction (time-scaled: \omega \equiv 1)'); grid on;

end

% Core
function munext = newton_midpoint_mu(mk, dt, Q, b)
% Solve F(m) = 0:   m - mk - dt * Lambda(mbar)*(Q*mbar + b) = 0
% w/ mbar = (mk + m)/2, via damped Newton with residual-based stopping.

munext  = mk;                 % initial guess (can also try one explicit step)
F       = F_midpoint_mu(munext, mk, dt, Q, b);
tol     = 1e-12 * (1 + norm(mk,2));
maxIter = 50;

for it = 1:maxIter
    F  = F_midpoint_mu(munext, mk, dt, Q, b);
    if norm(F,2) <= tol, break; end

    J  = J_midpoint_mu(munext, mk, dt, Q, b);
    dm = -J\F;

    % backtracking line search on residual
    alpha = 1.0;
    Fn = F_midpoint_mu(munext + alpha*dm, mk, dt, Q, b);
    while norm(Fn,2) > norm(F,2) && alpha > 1/64
        alpha = 0.5*alpha;
        Fn = F_midpoint_mu(munext + alpha*dm, mk, dt, Q, b);
    end
    munext = munext + alpha*dm;
end
end

function F = F_midpoint_mu(m, mk, dt, Q, b)
mbar = 0.5*(mk + m);
w    = gradH(mbar, Q, b);        % w = Q*mbar + b
F    = m - mk - dt * ( Lambda(mbar) * w );
end

function J = J_midpoint_mu(m, mk, dt, Q, b)
% Linearization:
% J = I - dt * (1/2) * [ Omega(w) + Lambda(mbar) Q ],
% where w = Q*mbar + b, and Omega(w) y = Lambda(y) w.
mbar = 0.5*(mk + m);
w    = gradH(mbar, Q, b);
J    = eye(3) - 0.5*dt * ( Omega_of_w(w) + Lambda(mbar)*Q );
end

% Model-specific bits 
function L = Lambda(mu)
% Lie–Poisson tensor for se(2): entries linear in μ.
% With basis {e1 (rot), e2 (trans x), e3 (trans y)} and left-trivialization.
L = [  0     ,  mu(3), -mu(2);
     -mu(3) ,   0    ,   0   ;
      mu(2) ,   0    ,   0   ];
end

function w = gradH(mu, Q, b)
% Affine gradient of H(μ) = 1/2 μ'Qμ + b'μ
w = Q*mu + b;
end

function Om = Omega_of_w(w)
% Build the linear operator Omega(w) such that Omega(w)*y = Lambda(y)*w.
% Using decomposition Lambda(y) = y1*R1 + y2*R2 + y3*R3 with constant Rk.
R1 = [0 0 0; 0 0 0; 0 0 0];                       % no μ1 term in Lambda
R2 = [0 0 -1; 0 0 0; 1 0 0];
R3 = [0 1  0; -1 0 0; 0 0 0];
Om = [R1*w, R2*w, R3*w];   % columns are Rk*w
end

function Xi = hat_se2(xi)
% xi = [omega; v_x; v_y] -> se(2) matrix (3x3 homogeneous)
omega = xi(1); vx = xi(2); vy = xi(3);
Xi = [ 0     -omega   vx;
       omega   0      vy;
       0       0       0 ];
end

function [H,C] = invariants(mu, Q, b)
% Hamiltonian and Casimir for se(2)
H = 0.5*(mu.'*Q*mu) + b.'*mu;     % e.g., μ1 + 1/2 μ2^2 for time-scaled choice
C = 0.5*(mu(2)^2 + mu(3)^2);      % standard se(2) Casimir
end
