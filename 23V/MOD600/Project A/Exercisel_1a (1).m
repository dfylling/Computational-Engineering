
function E_O_scheme

% Solution of u_t + f(u)_x =0


% Final Time
T = 1.5;
T = 2.5;

% Number of grid cells
%N=20;
%N=100;
N=200;
%N=500;

% Delta x
dx=5/N;

% define cell centers
x = -2.5+dx*0.5:dx:2.5-0.5*dx;

%define number of time steps
%NTime = 40
%NTime = 20
%NTime = 12
%NTime = 8
%NTime = 11
%NTime = 9
%NTime = 10

%NTime = 100*2*1.5
NTime = 100

% Time step dt
dt = T/NTime


% plotting of f,f+,f-
v=-1:0.01:+1;
subplot(1,2,1)
plot(v,f(v),'-g');
hold on
plot(v,f_pluss(v),'ob');
plot(v,f_minus(v),'or');
legend('f','f+','f-')
grid on
title('functions f, f+, f-')

%pause

 
% Define a vector that is useful for handling the different cells
J  = 1:N;    % number the cells of the domain
J1 = 2:N-1;  % the interior cells 
J2 = 1:N-1;  % numbering of the cell interfaces 

%define vector for initial data
u0 = zeros(1,N);

% case1
L1= x <= -1.0;
L2= x > -1.0   &  x<=1.0 ;
L3= x > 1.0; 

% (a)
u0(L1) = 1;
u0(L2) = 0;
u0(L3) = -1;

% (b)
%u0(L1) = -1;
%u0(L2) = 0;
%u0(L3) = +1;


% case2
% L1= x <= 1.0;
% L2= x > 1.0;
% u0(L1) = 1.0;
% u0(L2) = 1.1;


% case3
% L1= x <= 0.5;
% L2= x > 0.5 & x<= 1.5;
% L3= x > 1.5; 
% u0(L1) = 1.1;
% %u0(L2) = 0.1*(x(L2)-0.5) + 1.0;
% u0(L2) = -0.1*(x(L2)-0.5) + 1.1;
% u0(L3) = 1.0;


% plot initial data
subplot(1,2,2)
hold off
plot(x,u0,'--r','LineWidth',2)
xlabel('X')
ylabel('U(X)')
%axis([0 1 -0.1 +1.5])
title('Solution')
axis([-1.5 1.5 -1.1 +1.1])
grid on
hold on


% define vector for solution
u = zeros(1,N);
u_old = zeros(1,N);

% useful quantity for the discrete scheme
lambda = dt/dx


% is used for taking a pause
teller = 1; 

% calculate the numerical solution u by going through a loop of NTime number
% of time steps

% initial state
u_old = u0;

pause

for j = 1:NTime

    % Define the numerical flux at the cell-interfaces 3/2, 5/2, ... ,
    % N-1/2 = total of N-1 cell interfaces
    
    % Enquist-Osher
    
    
    % Case 1
    % f(u) = 1/2*u^2
    % f'(u) = u
    %f_pl = f, u>=0
    %f_pl = 0, u<0
    %f_mi = f, u<0
    %f_mi = 0, u>=0
    
    % check flow direction
    %Lp = u_old(J) >= 0;    % non-negative u flows from left to right
    %Ld = u_old(J) < 0;     % negative u flow from right towards left
    
    F_pl = zeros(1,N);
    F_mi = zeros(1,N);
    
    %F_pl(J(Lp)) = f(u_old(J(Lp)));
    %F_mi(J(Ld)) = f(u_old(J(Ld)));
    
    F_pl = f_pluss(u_old);
    F_mi = f_minus(u_old);
    
    % case 2
    % f(u) = u^2/(u^2 + M(1-u)^2)
    
    
    % General flux based on Engquist-Osher scheme
    U_half(J2) = F_pl(J2) + F_mi(J2+1);
    
    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%% 
    % the numerical scheme   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % calculate solution at interior part of domain, that is, cells j=2,...,N-1  
    u(J1) = u_old(J1) - lambda*(U_half(J1) - U_half(J1-1)); 
   
    % solution at left boundary
    u(1) = u0(1);
   
    % solution at right boundary
    u(N) = u0(N); 
    
    
    % plotting
    subplot(1,2,2)
    plot(x,u0,'--r','LineWidth',2)
    hold on
    plot(x,u,'-o')
    xlabel('X')
    ylabel('U(X)')
    grid on
    title('Solution of Burgers Equation')
    axis([-1.5 1.5 -1.1 +1.1])
    drawnow
    hold off
    
    % update "u_old" before you move forward to the next time level
    u_old = u;
    
     % should we take a pause?
    check = j/100;
    if (check == teller) 
     pause
     teller = teller +1;
    end
    
    
    
    
    
end


% calculate exact solution

v = zeros(1,N);

% case a)

if (T<=2)
   
    L1 = x <= -1 + 0.5*T;
    L2 = x>-1 + 0.5*T  &  x<= 1-0.5*T; 
    L3 = x>1-0.5*T; 
    
    v(L1) = 1;
    v(L2) = 0;
    v(L3) = -1;
    
else
    
    L4 = x<= 0;
    L5 = x>0;
    
    v(L4) = 1;
    v(L5) = -1;
    
end


% case b)




hold off


% plot exact solution 
plot(x,u0,'--r','LineWidth',2)
hold on
plot(x,u,'-o','LineWidth',2)
hold on
plot(x,v,'-g','LineWidth',2)
%plot(x,v,'-r')
axis([-1.5 1.5 -1.1 +1.1])
legend('initial data','numerical approximation','exact solution')
xlabel('X')
ylabel('U(X)')
grid on

hold off


function f=f(u)    
f = 0.5*u.^2;    
    
%M = 1;
%f = u.^2/(u.^2 + M*(1-u).^2);
end


function fpluss = f_pluss(u)    
f = 0.5*u.^2;    
L = u >= 0;

fpluss(L)  = f(L);
fpluss(~L) = 0;
%M = 1;
%f = u.^2/(u.^2 + M*(1-u).^2);
end


function fminus = f_minus(u)    
f = 0.5*u.^2;    
L = u >= 0;

fminus(L)  = 0;
fminus(~L) = f(~L);

%M = 1;
%f = u.^2/(u.^2 + M*(1-u).^2);
end


end