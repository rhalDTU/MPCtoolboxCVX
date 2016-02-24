% Linear MPC examples using CVX
% 1) Download and install CVX
%    cvxr.com/cvx and run 'cvx_setup' in Matlab
% 2) Run examples by running 'MPCtoolboxCVX'
%    Runs 'WindPowerStorageExample' by default
% Rasmus Halvgaard, rhal@dtu.dk, 2016-02-24

function MPCtoolboxCVX
  WindPowerStorageExample;
end

function WindPowerStorageExample
% Example of balancing wind power using a lossless storage unit
Ts = 1; % sampling period
Np = 100; % prediction horizon
w = rand(Np,1); % predicted wind power
d = 0.1*ones(Np,1); % demand
d(70:80) = 0.9; d(10:20) = 0.9;

% Storage unit constraints and model
umax = 1; umin = -1;  % charge/discharge power limits
ymax = 10*ones(Np,1); % storage limits
ymin = ones(Np,1); ymin(40:49) = 7;
x0 = 9;  % initial storage capacity
A = 0; B = 1; C = 1; % dx(t)/dt = A*x(t) + B*u(t), y=C*x(t)
[Ad,Bd] = css2dss(Ts,A,B,[]); % x(k+1) = Ad*x(k) + Bd*u(k), y=C*x(k)
[Hu,H0] = ImpulseResponse(Np,Ad,Bd,C,[]);

% Solve MPC
cvx_begin %quiet
    variables y(Np) u(Np) e(Np)
    minimize( norm(e,2)  ) % ||e||_2^2
    subject to
        e == w - d - u     % power balance
        y == Hu*u + H0*x0; % impulse response model
        ymin <= y <= ymax; % output constraints
        umin <= u <= umax; % input constraints
cvx_end

% Check 'cvx_status' if solved.
% For closed-loop feedback:
% Use only u(1) and re-solve optimization problem
% in next time step with updated information

% Results
% The storage unit charges/discharges according to 
% wind power production and demand while 
% maintaining storage capacity constraints
% (Time)  
% (10:20) The demand goes up and the storage unit discharges
% (20:70) The storage charges but also help maintain balance by discharging
% (40:50) The capacity is above the minimum constraint
% (70:80) The demand goes up it discharges
figure(1), clf
subplot(211)
  stairs(w), hold on
  stairs(u,'r')
  stairs(d,'m')
  stairs(e,'k--')
  legend('w','u','d','e')
  ylabel('Power')
subplot(212)
  stairs(y), hold on
  stairs(ymin,'--b')
  stairs(ymax,'--b')
  legend('y','ymin','ymax')
  ylabel('Storage capacity')
  xlabel('Time')
end

function [Hu,H0,Hd,nx,nu,ny,nd] = ImpulseResponse(N,A,B,C,E,D,F)
% Convert discrete time state space model:
%     x(k+1) = A*x(k) + B*u(k) + E*d(k)
%     y(k)   = C*x(k) + D*u(k) + F*d(k)
% to impulse response function N steps
%     Y = Hu*U + H0*x0 + Hd*D
if nargin <= 5, D = []; F = []; end
[nx,nu] = size(B);
nd = size(E,2);
ny = size(C,1);
H0 = zeros(N*ny,nx);
Hu = zeros(N*ny,N*nu);
Hd = zeros(N*ny,N*nd);

% Compute first column with all impulse response coefficients
T = C;
k1 = 1;
k2 = ny;
for k=1:N
   Hu(k1:k2,1:nu) = T*B;
   Hd(k1:k2,1:nd) = T*E;
   T = T*A;
   H0(k1:k2,1:nx) = T;
   k1 = k1+ny;
   k2 = k2+ny;
end

if any(any(D)) == 1 || any(any(F)) == 1 % if non-zero elements in D
    % Add extra row with direct output contributions for k = 0
    Hu = [D, zeros(size(D,1),size(Hu,2)-size(D,2)); Hu];
    Hd = [F, zeros(size(F,1),size(Hd,2)-size(F,2)); Hd];
    H0 = [C, zeros(size(C,1),size(H0,2)-size(C,2)); H0];
    N0 = N+1;
else
    N0 = N;
end

% Copy coefficients and fill out remaining columns
k1row = ny+1;
k2row = N0*ny;
k1col = nu+1;
k2col = nu+nu;
kk = N0*ny-ny;
for k=2:N
   Hu(k1row:k2row,k1col:k2col) = Hu(1:kk,1:nu);
   k1row = k1row+ny;
   k1col = k1col+nu;
   k2col = k2col+nu;
   kk = kk-ny;
end

k1row = ny+1;
k2row = N0*ny;
k1col = nd+1;
k2col = nd+nd;
kk = N0*ny-ny;
for k=2:N
   Hd(k1row:k2row,k1col:k2col) = Hd(1:kk,1:nd);
   k1row = k1row+ny;
   k1col = k1col+nd;
   k2col = k2col+nd;
   kk = kk-ny;
end
end

function [Ad,Bd,Ed] = css2dss(Ts,A,B,E)
% Discretize state space model (c2d)
[nx,nu] = size(B);
nd = size(E,2);

dss = expm([A B E; zeros(nu+nd,nx+nu+nd)]*Ts);

Ad = dss(1:nx,1:nx);
Bd = dss(1:nx,nx+1:nx+nu);
Ed = dss(1:nx,nx+nu+1:nx+nu+nd);
end



