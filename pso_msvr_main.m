clear;clc;close all;
%% load Data;
data=xlsread('SVM训练数据-改.xls','负荷明细数据','B3:B6602');
% 6600/24=275天
%上一天24个数据作为输入 预测下一天的24个数据
for i=1:274
    X(i,:)=data((i-1)*24+1:i*24);
    Y(i,:)=data(i*24+1:(i+1)*24);
end
% 归一化
[inputn,inputps]=mapminmax(X',0,1);
X=inputn';
[outputn,outputps]=mapminmax(Y',0,1);
Y=outputn';

%% 划分数据集
rand('state',0)
r=randperm(size(X,1));
ntrain =size(X,1)*0.5 ;          % 50%的为训练集 剩下为测试集
Xtrain = X(r(1:ntrain),:);       % 训练集输入
Ytrain = Y(r(1:ntrain),:);       % 训练集输出
Xtest  = X(r(ntrain+1:end),:);   % 测试集输入
Ytest  = Y(r(ntrain+1:end),:);   % 测试集输出

%% 没优化的24输出msvm
% 随机产生惩罚参数与核参数
C    = 1000*rand;%惩罚参数
par  = 1000*rand;%核参数
ker  = 'rbf';
tol  = 1e-20;
epsi = 1;
% 训练
[Beta,NSV,Ktrain,i1] = msvr(Xtrain,Ytrain,ker,C,epsi,par,tol);
% 测试
Ktest = kernelmatrix(ker,Xtest',Xtrain',par);
Ypredtest = Ktest*Beta;

% 计算均方误差
mse_test=sum(sum((Ypredtest-Ytest).^2))/(size(Ytest,1)*size(Ytest,2))

% 反归一化
yuce=mapminmax('reverse',Ypredtest',outputps);yuce=yuce';
zhenshi=mapminmax('reverse',Ytest',outputps);zhenshi=zhenshi';
%% 粒子群优化多输出支持向量机
[y ,trace]=psoformsvm(Xtrain,Ytrain,Xtest,Ytest);
%% 利用得到最优惩罚参数与核参数重新训练一次支持向量机
C    = y(1);%惩罚参数
par  = y(2);%核参数
[Beta,NSV,Ktrain,i1] = msvr(Xtrain,Ytrain,ker,C,epsi,par,tol);
Ktest = kernelmatrix(ker,Xtest',Xtrain',par);
Ypredtest_pso = Ktest*Beta;
% 误差
pso_mse_test=sum(sum((Ypredtest_pso-Ytest).^2))/(size(Ytest,1)*size(Ytest,2))
% 反归一化
yuce_pso=mapminmax('reverse',Ypredtest_pso',outputps);yuce_pso=yuce_pso';

%% 画图
figure
plot(trace)
xlabel('迭代次数')
ylabel('适应度值')
title('psosvm适应度曲线（寻优曲线）')
%画出测试集中最后一天的数据（由于是随机划分的，因此并不是代表12月份最后一天）

figure;hold on;grid on;axis([0 23 -inf inf]);t=0:1:23;
plot(t,yuce(end,:),'-r*')
plot(t,zhenshi(end,:),'-ks')
legend('预测值','真实值')
title('优化前');xlabel('时刻');ylabel('负荷')

figure;hold on;grid on;axis([0 23 -inf inf]);t=0:1:23;
plot(t,yuce_pso(end,:),'-r*')
plot(t,zhenshi(end,:),'-ks')
legend('预测值','真实值')
title('优化后');xlabel('时刻');ylabel('负荷')

%合并出一个图
figure;hold on;grid on;t=0:1:23;axis([0 23 -inf inf])
plot(t,yuce(end,:),'-bp')
plot(t,yuce_pso(end,:),'-r*')
plot(t,zhenshi(end,:),'-ks')
legend('svm预测值','psosvm预测值','真实值')
title('优化前后');xlabel('时刻');ylabel('负荷')

save data_svm_psosvm 