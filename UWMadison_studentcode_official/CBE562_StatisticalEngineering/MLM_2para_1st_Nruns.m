for i=1:100
dat1 = load("C:\Users\buing\Documents\MATLAB\CBE 562\meth(1)_eth(2).dat");
dat_rand = dat1;
dat1_rand_2 = dat1(4:40,2) + (- 0.002 + (0.004).*rand(length(dat1(4:40,2)),1));
dat1_rand_3 = dat1(2:38,3) + (- 0.002 + (0.004).*rand(length(dat1(2:38)),1)) ;
dat_rand(4:40,2) = dat1_rand_2;
dat_rand(2:38,3) = dat1_rand_3;
dat = dat_rand;
Psat1_in = 127.04;
Psat2_in = 58.98 ;
P_1 = dat(:,1).*750;
x1 = dat(:,2);
y1 = dat(:,3);
x2 = 1.- dat(:,2);
y2 = 1.- y1;
Psat1 = Psat1_in; %methanol(1) in bar
Psat2 = Psat2_in; %ethanol(2) in bar

%this is margules 2-parameter model:
%Ge/RT = x1x2(A21*x1+A12*x2)
%ln(g1) = x2^2(A12 + 2(A21 - A12)x1)
%ln(g2) = x1^2(A12 + 2(A21- A12)x2)
% => lng1 = A12*(x2^2 - 2*x2^2*x1) + A21*(2*x2^2*x1)
% we can do lng2 too to test, but we have found that this doesn't
% signficantly affect the result from one_margules.
%g1 = exp(x2^2(A12 + 2(A21 - A12)x1)))
%in the full form:
g1_data = y1.*P_1./(x1.*Psat1);
g2_data = y2.*P_1./(x2.*Psat2);
lng1 = log(g1_data);
lng2 = log(g2_data);
Y1 = lng1;
Y2 = lng2;
%from the book, we can rearrange
X = [x2.^2 - 2.*x2.^2.*x1, 2.*x2.^2.*x1];
theta_est = inv(X'*X)*X'*Y1;
A12_est = theta_est(1);
A21_est = theta_est(2);
%reverse this into gamma 1 and 2, and
%recalculate P with x1 and x2 with the gamma 1 and 2.
x1_pre = linspace(0.001,0.999,100);
x2_pre = 1.- x1_pre;
g1_pre = exp(x2_pre.^2.*(A12_est + 2.*(A21_est - A12_est).*x1_pre));
g2_pre = exp(x1_pre.^2.*(A21_est + 2.*(A12_est - A21_est).*x2_pre));
P1_pre = Psat1.*x1_pre.*g1_pre + Psat2.*x2_pre.*g2_pre;
y1_pre = g1_pre.*x1_pre.*Psat1./P1_pre;
plot(x1,P_1,':xr')
hold on
plot(y1,P_1,':xb')
plot(y1_pre,P1_pre,'-b')
plot(x1_pre,P1_pre,'-r')
title('two-parameter model')
legend('online data x1','online data y1','model predict x1', 'model predict y1','location','best')
ylabel('P(mmHg)')
xlabel('x1,y1')
end
%% A12 and A21
A12 = A12_est;
A21 = A21_est;
%% Gaussian
heis_theta = eigs(X'*X);
Gaussian = heis_theta;