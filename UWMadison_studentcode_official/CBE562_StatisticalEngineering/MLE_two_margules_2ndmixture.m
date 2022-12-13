for i=1:100
dat2 = load("C:\Users\buing\Documents\MATLAB\CBE 562\isopropanol(1)-water(2).dat");
dat2_rand_2 = dat2(4:40,2) + (- 0.002 + (0.004).*rand(length(dat2(4:40,2)),1)) ;
dat2_rand_3 = dat2(2:38,3) + (- 0.002 + (0.004).*rand(length(dat2(2:38,3)),1)) ;
dat2_rand = dat2;
dat2_rand(4:40,2) = dat2_rand_2;
dat2_rand(2:38,3) = dat2_rand_3;
dat = dat2_rand;
Psat1 = 58;
Psat2 = 32.01;
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
%reverse plug for model data
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
A12_dat(i) = tA12;
A21_dat(i) = tA21;
end
%% fisher, gaussian, data of average
m_A12 = mean(A12_dat);
m_A21 = mean(A21_dat);
m_theta = [m_A12 m_A21];
X = Aeq1;
Y1 = Beq1;
heis_theta = eigs(X'*X);
Yhat = X*m_theta'; %Xtheta
e = Y1- Yhat;
var_theta = var(e);
pd = fitdist(e,'Normal');
%e has miu approx to 0 and sigma = 0.242488
%can plot histogram to confirm if gaussian or not
%curve looks like it does fit, so residual does follow a gaussian
%distribution
%theta hat would be unbias! doesn't have any systematic error (hopefully)
%% precision?
Cov_var_e = inv(X'*X)*var(e);
%the matrix reflect large numbers; indicating that the variances are big,
%and estimates have large uncertainty. Overall, not precise.
%% Fisher
sigma_est = sqrt(var(e));
I = inv(Cov_var_e); %fisher
%check eigenvalues
heis_I = eigs(I'*I);
function LogL2 = func1(theta1)
global x2 x1
LogL2 = -((theta1(1).*sum(x2.^2 - 2.*x2.^2.*x1) + theta1(2).*sum((2.*(x2.^2).*x1))));
end
