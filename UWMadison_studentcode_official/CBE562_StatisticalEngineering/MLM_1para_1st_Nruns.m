dat1 = load("C:\Users\buing\Documents\MATLAB\CBE 562\meth(1)_eth(2).dat")
dat_rand = dat1;
dat1_rand_2 = dat1(4:40,2) + (- 0.002 + (0.004).*rand(length(dat1(4:40,2)),1));
dat1_rand_3 = dat1(2:38,3) + (- 0.002 + (0.004).*rand(length(dat1(2:38)),1));
dat_rand(4:40,2) = dat1_rand_2;
dat_rand(2:38,3) = dat1_rand_3;
dat = dat_rand;
Psat1_in = 127.04;
Psat2_in = 58.98;     

P_1 = dat(:,1).*750;
x1 = dat(:,2);
y1 = dat(:,3);
x2 = 1.- dat(:,2);
y2 = 1.- y1;
%For Psat data, I took from CBE 311 database of excel sheet. I find the
%Psat data for methanol and ethanol at 25*C from excel sheet of Antoine.xlsx. This excel
% is introduced to us in CBE 311 as part of our learnings. The author of the book %
%create this database, and is reliable enough! It is given in mmHG, and convert
%to bar to fit P data from the website.
Psat1 = Psat1_in; %methanol(1) in bar
Psat2 = Psat2_in; %ethanol(2) in bar
%say this set of data is collected and we aren't sure about the activity
%model. Let's pick out a few CBE 311 activities model and build them as
% theta estimate model.
%we consider that these solutions could be modeled with modified Rault's
%Law/
%general form is P = x1g1Psat1 + x2g2Psat2
%our specialty lies in the activities coefficient embedded in gamma 1 and 2, 
%as different activities coefficient models will yield different gamma1 and
% 2 depending on the mole fraction.

%this is margules 1-parameter model:
%Ge/RT = A12x1x2
%ln(g1) = A12(x2)^2 => g1 = exp(A12(x2)^2)
%ln(g2) = A12(x1)^2 => g2 = exp(A12(x1)^2)
%in the full form:
%P = x1*exp(A12(x2)^2)*Psat1 + x2*exp(A12(x1)^2)*Psat2
% mathematically this seems complicated, so we could approach it
% differently
%gamma 1 and 2 by Rault's law definition is also gamma = yP/xPsat
%we could set up a set of gamma data and then use ln(g1) = A12(x2)^2 or
%ln(g2) = A12(x1)^2. For our 1 parameter model, A12 should be equal to each
%other.
g1_data = y1.*P_1./(x1.*Psat1);
g2_data = y2.*P_1./(x2.*Psat2);
lng1 = log(g1_data);
Y1 = lng1;
X0_1 = x1.^2;
%lng2 = A12(x1^2) => Y = theta0*X01;
theta_est_1 = inv(X0_1'*X0_1)*X0_1'*Y1;
A12 = theta_est_1;
heis_theta_1 = eigs(X0_1'*X0_1);
%theta_est = A12
%back calculate g1 and g2
x1_pre = linspace(0.001,0.999,100);
x2_pre = 1.- x1_pre;
g1_pre = exp(theta_est_1*(x2_pre.^2));
g2_pre = exp(theta_est_1*(x1_pre.^2));
P1_pre = Psat1.*x1_pre.*g1_pre + Psat2.*x2_pre.*g2_pre;
y1_pre = g1_pre.*x1_pre.*Psat1./P1_pre;
%output 
%% plot1
plot(x1,P_1,':xr')
hold on
plot(y1,P_1,':xb')
plot(x1_pre,P1_pre,'-r')
plot(y1_pre,P1_pre,'-b')
title('one-parameter model')
legend('online data x1','online data y1','model predict x1', 'model predict y1','location','best')
ylabel('P(mmHg)')
xlabel('x1,y1')
%% A12
A12 = theta_est_1;
%% Gaussian
heis_theta = eigs(X0_1'*X0_1);
GaussianX = heis_theta; %will fix later