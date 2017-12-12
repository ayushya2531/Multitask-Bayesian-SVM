clc;
clear;
close all;
Data=load('mtl_synth.mat');
X_test=Data.xTe;
X_train=Data.xTr;
Y_test=Data.yTe;
Y_train=Data.yTr;
T=size(X_train,2);
[N,D]=size(X_train{1}); 
K = 5; % No of clusters
Pi = (1/K)*ones(K,1);% proprtion of K clusters
mu_k = zeros(K,D); % mean of K clusters
sigma_2 = ones(K,1); % variance of K clusters
%Lamda
w = zeros(T,D);
Gamma = zeros(T,K);
Zi = zeros(T,N);
conv = 1;
%for l=1:conv
%%E_Step
    for t=1:T
        for n=1:N
            Zi(t,n)=1/abs(1-Y_train{t}(n)*X_train{t}(n,:)*(w(t,:))');   %expectation of lambda inverse
        end
        for k=1:K
            Gamma(t,k)= Pi(k)*mvnpdf(w(t,:),mu_k(k,:),sigma_2(k)*eye(D))*montecarlo(Y_train{t}, X_train{t}, w, t);
        end
        Gamma(t,:)=Gamma(t,:)/sum(Gamma(t,:),2);
    end
    %M_step
    %N=sum(sum(Gamma(:,k))); 
    for k=1:K
        N_k=sum(Gamma(:,k));
        Pi(k)=N_k/T;
    end
    iter = 10;
    for i=1:iter
        for k=1:K
           sigma_2(k)=0;
           for t=1:T
           sigma_2(k)=sigma_2(k)+Gamma(t,k)*(w(t,:)-mu_k(k,:))*(w(t,:)-mu_k(k,:))';
           end
           sigma_2(k)=sigma_2(k)/N_k;
           mu_k(k,:)=0;
           for t=1:T
           mu_k(k,:)=mu_k(k,:)+Gamma(t,k)*w(t,:);
           end
           mu_k(k,:)=mu_k(k,:)/N_k;
        end
        for t=1:T
            P=zeros(1,D);
            Q=zeros(D,D);
            for k=1:K
            P = P + Gamma(t,k)*mu_k(k,:)/sigma_2(k);
            Q = Q + (Gamma(t,k)/sigma_2(k))*eye(D);
            end
            R=zeros(1,D);
            S=zeros(D,D);
            for n=1:N
                R=R+(1+Zi(t,n))*Y_train{t}(n)*X_train{t}(n,:);
                S=S+Zi(t,n)*(X_train{t}(n,:))'*(X_train{t}(n,:));
            end
            w(t,:)=inv(Q+S)*(P+R)';
        end
    end
% end