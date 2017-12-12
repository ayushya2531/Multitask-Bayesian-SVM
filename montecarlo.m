function monte = montecarlo(Y,X,W,t)
[N D]=size(X);
num = 50;
val = ones(num,1);
for iter = 1:num
    for n=1:N
       sam = gigrnd(0.5, 1, (1-Y(n)*X(n,:)*W(t,:)')^2,1); 
       val(iter,1) = 10*val(iter,1)*(sam^(-0.5))*exp((-0.5/sam)*(1+sam-Y(n)*X(n,:)*W(t,:)')^2);
    end
end
monte = mean(val,1);