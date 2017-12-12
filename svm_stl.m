clc;
clear;
close all;
Data=load('landmine_balanced.mat');
sigma_22=100; %Regularizer value, lower more regularization, higher less
X_test=Data.xTe;
X_train=Data.xTr;
Y_test=Data.yTe;
Y_train=Data.yTr;
T=size(X_train,2);
acc = zeros(T,1);
for t = 1:T
    SVMModel = fitcsvm(X_train{t},Y_train{t});
    [Y_pred{t},score] = predict(SVMModel,X_test{t});
    acc(t,1) = sum(Y_pred{t}==Y_test{t})/size(Y_test{t},1)*100;
end
mean(acc);
