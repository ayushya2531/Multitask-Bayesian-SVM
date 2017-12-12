clc;
clear;
close all;
Data=load('landmine_balanced.mat');
load('Workspace_lm70.mat');
X_test=Data.xTe;
X_train=Data.xTr;
Y_test=Data.yTe;
Y_train=Data.yTr;
T=size(X_train,2);
%[N,D]=size(X_train{1}); 
%pred = zeros(N,T);
acc_mtl = zeros(T,1);
for t =1:T
    N = size(X_test{t},1);
    pred = sign(sum(X_test{t}.*repmat(w(t,:),N,1),2));
    acc_mtl(t,1) = sum(pred==Y_test{t})/size(Y_test{t},1)*100; 
end
mean_acc_mtl = mean(acc_mtl);    
% Data=load('landmine_balanced.mat');
% sigma_22=100; %Regularizer value, lower more regularization, higher less
% X_test=Data.xTe;
% X_train=Data.xTr;
% Y_test=Data.yTe;
% Y_train=Data.yTr;
% T=size(X_train,2);
acc_stl = zeros(T,1);
for t = 1:T
    SVMModel = fitcsvm(X_train{t},Y_train{t});
    [Y_pred{t},score] = predict(SVMModel,X_test{t});
    acc_stl(t,1) = sum(Y_pred{t}==Y_test{t})/size(Y_test{t},1)*100;
end
mean_acc_stl = mean(acc_stl);

xnew= [1:T];
plot(xnew,acc_stl,'ro',xnew,acc_mtl,'*b');
legend('STL accuracy','MTL accuracy');
xlabel('Task');
ylabel('Accuracy');
title('Multi-task SVM on Landmine data for 5 clusters');