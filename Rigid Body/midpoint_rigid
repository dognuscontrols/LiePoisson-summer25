function rigid_body_midpoint_demo
clc; close all;

I = diag([1,2,3]);                 % inertia
Iinv = diag(1./diag(I));
w0 = [1;10;1];                     % initial angular velocity
m0 = I*w0;                         % initial body momentum
A0 = eye(3);                       % initial attitude (inertial = body)

run_case(I,Iinv,m0,A0,0.05,400,1);     % coarse: dt=0.05, 400 steps
run_case(I,Iinv,m0,A0,0.005,4000,2);   % fine: dt=0.005, 4000 steps
end

function run_case(I,Iinv,m0,A0,dt,N,figbase)
fprintf('Running dt = %.6g, N = %d (T=%.3g s)\n',dt,N,dt*N);

% Storage
t = (0:N)*dt;
m = zeros(3,N+1);       m(:,1) = m0;
A = zeros(3,3,N+1);     A(:,:,1) = A0;

% For energy/Casimir curves
H = zeros(1,N+1);
C = zeros(1,N+1);
[H(1),C(1)] = energy_casimir(m0,Iinv);

% Time stepping
for k = 1:N
    mk   = m(:,k);

    % Implicit midpoint step for body momentum 
    % Solve F(mnext) = 0 by damped Newton:
    % F(m) = m - mk - dt * (mbar x I^{-1} mbar), mbar=(mk+m)/2

    mnext = mk;                         % initial guess
    F0    = F_midpoint(mnext,mk,dt,Iinv);
    g0    = abs(F0);                    % load vector magnitude per component
    g0(g0==0) = 1;                      % avoid zero threshold

    maxIter = 50;
    for it = 1:maxIter
        F  = F_midpoint(mnext,mk,dt,Iinv);
        J  = J_midpoint(mnext,mk,dt,Iinv);    % analytic Jacobian

        % Newton step with simple backtracking if residual grows
        dm  = -J\F;
        alpha = 1.0;
        Fn = F_midpoint(mnext + alpha*dm,mk,dt,Iinv);
        while norm(Fn,2) > norm(F,2) && alpha > 1/64
            alpha = alpha/2;
            Fn = F_midpoint(mnext + alpha*dm,mk,dt,Iinv);
        end
        mnext = mnext + alpha*dm;

        % stopping criterion (componentwise 1% of initial g)
        if all(abs(dm) <= 0.01*g0)
            break;
        end
        % If residual exploded, try smaller dt locally
        if it==maxIter && norm(Fn) > norm(F0)
            warning('Newton stagnation at k=%d; consider smaller dt.',k);
        end
    end

    m(:,k+1) = mnext;

    % Attitude reconstruction w/ Cayley transform
    % b_k = (dt/4) * I^{-1} (m_k + m_{k+1}) [Eq. (37b)]
    b  = (dt/4) * Iinv*(m(:,k)+m(:,k+1));
    Sb = skew(b);
    % Solve [I - Sb] X = [I + Sb] (avoid explicit inverse)
    X = (eye(3) - Sb) \ (eye(3) + Sb);
    A(:,:,k+1) = A(:,:,k) * X;

    % invariants
    [H(k+1),C(k+1)] = energy_casimir(m(:,k+1),Iinv);
end

% Plots 
% Fig. 4 style: Energy and Casimir (both should be flat)
figure(figbase*10+1); clf;
plot(t,H,'-','LineWidth',1.2); hold on;
plot(t,C,'-','LineWidth',1.2);
xlabel('Time (seconds)'); ylabel('Energy and Casimir');
legend('Energy','Casimir','Location','best');
title(sprintf('Rigid Body Motion  (dt = %.6g, N = %d)',dt,N));
grid on;

% Fig. 5/6 style: A(1,1) vs time
A11 = squeeze(A(1,1,:));
figure(figbase*10+2); clf;
plot(t,A11,'-','LineWidth',1.0);
xlabel('Time (seconds)'); ylabel('A(1,1)');
title(sprintf('A(1,1) vs time  (dt = %.6g, N = %d)',dt,N));
yline(1,'k:'); yline(-1,'k:'); grid on;

% Cycle stats (peaks and half-cycles) 
stats = analyze_cycles_half(t, A11);
cycles_from_Tmean = (t(end)-t(1)) / stats.Tmean_peaks;
fprintf('Cycles estimated from mean period: %.2f\n', cycles_from_Tmean);

fprintf(['Cycles (peaks)                 : %d\n' ...
         'Mean period from peaks (s)     : %.6f\n' ...
         'Cycles (peaks+troughs)/2       : %.1f\n' ...
         'Mean period from half-cycles(s): %.6f\n'], ...
        stats.cycles_peaks, stats.Tmean_peaks, ...
        stats.cycles_half,  stats.Tmean_from_half);
end


% Residual for midpoint equation
function F = F_midpoint(m, mk, dt, Iinv)
mbar = 0.5*(mk + m);
u    = Iinv*mbar;
F    = m - mk - dt*( cross(mbar,u) );
end

% Analytic Jacobian of F wrt m (see derivation in analysis)
function J = J_midpoint(m, mk, dt, Iinv)
mbar = 0.5*(mk + m);
u    = Iinv*mbar;

% derivative of c(m) = mbar x (Iinv mbar)
% \partialc = (1/2)[ -S(u) + S(mbar) Iinv ] \partialm
Jc = 0.5*( -skew(u) + skew(mbar)*Iinv );

J = eye(3) - dt*Jc;
end

% Skew-symmetric matrix for cross-product
function S = skew(v)
S = [   0   -v(3)  v(2);
      v(3)    0   -v(1);
     -v(2)  v(1)    0 ];
end

% Energy and Casimir
function [H,C] = energy_casimir(m,Iinv)
H = 0.5*m.'*(Iinv*m);
C = 0.5*(m.'*m);
end

% Rough cycle counter for A11 (peak counting)
function n = count_cycles(x)
% count zero crossings of derivative (peak count)
dx = diff(x);
zc = find(dx(1:end-1) > 0 & dx(2:end) < 0); % local maxima
n = numel(zc);
end

function stats = analyze_cycles_half(t, x)
% Half-cycle analysis with endpoint correction.
% - Detect extrema from derivative sign changes
% - Add ±0.5 cycle if we likely missed an endpoint extremum
% - Return peak-only and half-cycle counts + mean periods

t = t(:); x = x(:);

% derivative & interior timestamps
dx = diff(x) ./ diff(t);                  % length L
if numel(dx) < 2
    stats.cycles_peaks        = 0;
    stats.Tmean_peaks         = NaN;
    stats.cycles_half_raw     = 0;
    stats.cycles_half_corr    = 0;
    stats.Tmean_from_half_raw = NaN;
    stats.Tmean_from_half_corr= NaN;
    return;
end
tz = t(2:end-1);                           % length L-1

% extrema via sign changes in dx
dz1 = dx(1:end-1); dz2 = dx(2:end);
isMax = (dz1 > 0) & (dz2 < 0);            % + to -
isMin = (dz1 < 0) & (dz2 > 0);            % - to +
tmax  = tz(isMax);
tmin  = tz(isMin);

% peak-only cycles 
stats.cycles_peaks = numel(tmax);
Tp = diff(tmax);
stats.Tmean_peaks  = mean(Tp,'omitnan');

% half-cycles (raw)
te   = sort([tmax; tmin]);                 % all extrema times
Th   = diff(te);                           % half-period samples
Th_med = median(Th,'omitnan');             % robust half-period
if isempty(Th_med) || ~isfinite(Th_med)
    Th_med = (t(end)-t(1))/max(1,numel(te));  % fallback
end

cycles_half_raw  = numel(te)/2;
Tmean_half_raw   = 2*mean(Th,'omitnan');

% endpoint correction heuristic 
% If there's a long gap from t(1) to first extremum relative to half-period,
% likely "missed" one endpoint extremum -> add 0.5
% Same for the tail gap from last extremum to t(end)
lead_gap = te(1)   - t(1);
tail_gap = t(end)  - te(end);
add_head = lead_gap > 0.75*Th_med;         % threshold: 75% of half-period
add_tail = tail_gap > 0.75*Th_med;

cycles_half_corr = cycles_half_raw + 0.5*add_head + 0.5*add_tail;

% For a corrected mean period, pad the half-intervals using the median
Th_corr = Th;
if add_head, Th_corr = [Th_med; Th_corr]; end
if add_tail, Th_corr = [Th_corr; Th_med]; end
Tmean_half_corr = 2*mean(Th_corr,'omitnan');

% export
stats.cycles_half          = cycles_half_raw;      
stats.Tmean_from_half      = Tmean_half_raw;
stats.cycles_half_raw      = cycles_half_raw;
stats.cycles_half_corr     = cycles_half_corr;
stats.Tmean_from_half_corr = Tmean_half_corr;
end
