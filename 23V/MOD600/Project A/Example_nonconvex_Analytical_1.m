function Riemann

clf

%%%%%%%%%%%%%%%%%
%  initial data %
%%%%%%%%%%%%%%%%%

% grid cells
N=500;

% Delta x
dx=4/N;

% define cell centers
x = -0.5+dx*0.5:dx:3.5-0.5*dx;
    
%define vector for initial data
u0 = zeros(1,N);

% case1
L1= x <= 0;
L2= x >  0   &  x<=1.0 ;
L3= x > 1.0; 

% (a)
u0(L1) = 0;
u0(L2) = 0.75;
u0(L3) = 0;

% (b)
%u0(L1) = -1;
%u0(L2) = 0;
%u0(L3) = +1;


% plot initial data
subplot(1,2,2)
hold off
plot(x,u0,'--r','LineWidth',2)
xlabel('X')
ylabel('U(X)')
%axis([0 1 -0.1 +1.5])
title('Solution')
axis([-0.5 3.5 -0.1 +1.1])
grid on
hold on


%%%%%%%%%%%%%%%%%%%%%%%%
%  analytical solution %
%%%%%%%%%%%%%%%%%%%%%%%%

% Time
Time = 0.5;

% Number of values to divide [vl,vr] into
Nv = 50;

% Give Riemann data

%%%%%%%%%%%%%%%%%%
% Riemann at x=0 %
%%%%%%%%%%%%%%%%%%
xR  = 0;
v_l = 0;
v_r = 0.75;

    % v_l < v_r (increasin jump)
    % Determine the lower convex envelope
    dv = (v_r-v_l)/Nv;
    v  = v_l:dv:v_r;
    
    % compute the lower convex envelope for f on [v_l,v_r]
    P = zeros(length(v),2);
    P(:,1) = v;
    P(:,2) = f(v);
    
    % computes the convex hull in [v_l,v_r]
    [k,av] = convhull(P);
    
    % determine the lower convex be (ul,gl(ul))
    d = diff(P(k,1));   
    s = sign(d);

    L = (s==1);
    upper_L = min(find(L==0));    % the last index we need to describe the lower convex envelop
    
    % (ul,gl) is used to rep lower conv env
    ul = zeros(upper_L,1);
    gl = zeros(upper_L,1);

    L1 = 1:upper_L;
    
    % represents lower convex envelope
    ul(L1) = P(k(L1),1);
    gl(L1) = P(k(L1),2);

    subplot(1,2,1)
    % plot function 
    plot(P(:,1),P(:,2));
    hold on
    plot(ul,gl,'o-g')
    grid on
    title('Lower Convex Envelope')
    legend('function','lower convex')
     
    % compute a velocity s_i for each v_i based on gl
    LL1 = 2:upper_L-1;
    speed_l = zeros(upper_L,1);
    speed_l(1)       = (gl(2)-gl(1))/(ul(2)-ul(1));
    speed_l(LL1)     = (gl(LL1+1)-gl(LL1-1))./(ul(LL1+1)-ul(LL1-1)); 
    speed_l(upper_L) = (gl(upper_L)-gl(upper_L-1))/(ul(upper_L)-ul(upper_L-1));
    
    % compute the travelled distance from x_l
    xr = zeros(length(speed_l),1);
    xr = xR + speed_l*Time;
    
    subplot(1,2,2)
    plot(xr,ul)
    axis([-0.5 3.5 -0.1 +1.1])
 
    % represent the solution of RP at x=0 by (xM,uM)
    xM(1) = -0.5;
    xM(2:length(xr)+1) = xr;
    uM(1) = 0;
    uM(2:length(ul)+1) = ul;
    
pause    
    
%%%%%%%%%%%%%%%%%%
% Riemann at x=1 %
%%%%%%%%%%%%%%%%%% 
xR  = 1;
v_l = 0.75;
v_r = 0;
    
    % v_l > v_r (decreasing jump)
    % Determine the upper concave envelope

    dv = (v_l-v_r)/Nv;
     v = v_r:dv:v_l;
     
    % compute the upper concave envelope for f on [v_r,v_l]
    P = zeros(length(v),2);
    P(:,1) = v;
    P(:,2) = f(v);
    
    % computes the convex hull in [v_r,v_l]
    [k,av] = convhull(P);
    
    % determine the upper concav envelope (uu,gu(uu))
    d = diff(P(k,1));   
    s = sign(d);

    % the first points of "k" up to "upper_L" is associated with lower
    % convex env, the remaining with upper concave envelope
    L = (s==1);
    upper_L = min(find(L==0));
    
    L2 = upper_L:length(k);
    
    % upper concav envelope
    uu = zeros(length(L2),1);
    gu = zeros(length(L2),1);
    
    uu = P(k(L2),1);
    gu = P(k(L2),2);

    % plotting
    subplot(1,2,1)
    %plot(P(:,1),P(:,2));
    hold on
    plot(uu,gu,'+-c')
    %legend('upper concave')
    legend('function','lower convex','upper concave')
    title('Upper Concave Envelope')
    grid on
    
    % compute a velocity s_i for each v_i based on gl
    lower_U = length(k)-upper_L+1;
    LL2 = 2:lower_U-1;
    speed_u = zeros(lower_U,1);
    speed_u(1)       = (gu(2)-gu(1))/(uu(2)-uu(1));
    speed_u(LL2)     = (gu(LL2+1)-gu(LL2-1))./(uu(LL2+1)-uu(LL2-1)); 
    speed_u(lower_U) = (gu(lower_U)-gu(lower_U-1))/(uu(lower_U)-uu(lower_U-1));

    % compute the travelled distance from x_l
    xr = zeros(length(speed_u),1);
    xr = xR + speed_u*Time;
    
    subplot(1,2,2)
    plot(xr,uu)
    %axis([-0.5 3.5 -0.1 +1.1])

pause    
    
    % represent the solution of RP at x=1 by (xM,uM)
    % by continuing from the solution of RP at x=0 -> xM is extended to also describe RP at x=1 
    I = length(xM)+1:length(xM) + length(xr);
    xM(I) = xr;
    uM(I) = uu;
    xM(length(xM)+1) = 3.5;
    uM(length(uM)+1) = 0;
    
% construct a full solution on the grid x by interpolation from (xM,uM)
% In other words, project back to the original grid in space given by "x"
u_sol = zeros(1,N);
u_sol = interp1(xM,uM,x,'linear','extrap');
    
subplot(1,2,2)
hold off
plot(x,u0,'--r')
hold on
grid on
plot(x,u_sol,'-r','LineWidth',2)
legend('initial','solution')
axis([-0.5 3.5 -0.1 +1.1])    
title('Solution')
    

function f=f(x)

f = x.^2./(x.^2 + (1-x).^2);
%f = x.^4./(x.^5 + (1-x).^3);
%f = sin(pi*x)+x.^5;
end

end
