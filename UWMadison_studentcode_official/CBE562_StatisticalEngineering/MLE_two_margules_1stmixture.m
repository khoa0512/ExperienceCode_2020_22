%% two-margules function building
for i=1:100
dat1 = load("C:\Users\buing\Documents\MATLAB\CBE 562\meth(1)_eth(2).dat");
dat_rand = dat1;
dat1_rand_2 = dat1(4:40,2) + (- 0.002 + (0.004).*rand(length(dat1(4:40,2)),1));
dat1_rand_3 = dat1(2:38,3) + (- 0.002 + (0.004).*rand(length(dat1(2:38)),1));
dat_rand(4:40,2) = dat1_rand_2;
dat_rand(2:38,3) = dat1_rand_3;
dat = dat_rand;
Psat1 = 127.04;
Psat2 = 58.98 ;
x1 = dat(:,2);
y1 = dat(:,3);
x2 = 1.- dat(:,2);
opt = optimoptions('fmincon','Display','off','Algorithm','sqp');
P_1 = dat(:,1).*750;
%yP = xPsat gamma
%log(yP/xPsat) = x2^2*(A12 + 2(A21 - A12)*x1)
%lng1 = A12*(x2^2 - 2*x2^2*x1) + A21*(2*x2^2*x1)
Aeq1 = [x2.^2 - 2.*(x2.^2).*x1, 2*(x2.^2).*x1];
Beq1 = log(y1.*P_1./(x1.*Psat1));
theta1= fmincon(@func1,[2, 1],[],[],Aeq1,Beq1,[],[],[],opt);
tA12 = (theta1(1));
tA21 = (theta1(2));
%reverse to calculate g1,g2,P1,...
x1_pre = linspace(0.001,0.999,100);
x2_pre = 1.- x1_pre;
g1_m = exp(x2_pre.^2.*(tA12 + 2.*(tA21 - tA12).*x1_pre));
g2_m = exp(x1_pre.^2.*(tA21 + 2.*(tA12 - tA21).*x2_pre));
P1_pre = x1_pre.*g1_m.*Psat1 + x2_pre.*g2_m.*Psat2;
y1_pre = g1_m.*x1_pre.*Psat1./P1_pre;
plot(x1,P_1,':xr')
hold on
plot(y1,P_1,':xb')
plot(x1_pre,P1_pre,'-r')
plot(y1_pre,P1_pre,'-b')
title('two-parameter model')
legend('online data x1','online data y1','model predict x1', 'model predict y1','location','best','FontSize',15)
ylabel('P(mmHg)')
xlabel('x1,y1')
end

function LogL2 = func1(theta1)
global x2 x1
LogL2 = (theta1(1).*sum(x2.^2 - 2.*x2.^2.*x1) + theta1(2).*sum((2.*(x2.^2).*x1)));
end
