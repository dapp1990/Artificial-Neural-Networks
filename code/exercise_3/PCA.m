clc;
clear;
close all;
load choles_all;
load threes -ASCII;

k = 50;
% x = randn(50,500);
results =(k+1);

for c = 1:k
x1 = mapstd(threes);
[x,PS] = processpca(x1,0.01);

conv_x = cov(x);
[E,s] = eig(conv_x);

vec = cumsum(s);

[s, indx] = sort(diag(s), 'descend');

E =E(:,indx);

projection_matrix = E(:,1:c)';

z = projection_matrix*x';

x_hat = projection_matrix'*z;


x_hat2 = processpca('reverse', x_hat',PS);

error = sqrt(mean(mean((x1-x_hat2).^2)));

results(c) = error;

end


x1 = mapstd(threes);
[x,PS] = processpca(x1,0.01);

conv_x = cov(x);
[E,s] = eig(conv_x);

vec = cumsum(s);


Mean = mean(threes);

colormap('gray');
% imagesc(reshape(threes(80,:),16,16),[0,1]);
% imagesc(reshape(x_hat2(10,:),16,16),[0,1]);
plot(results);
title('PCA of number 3')
xlabel('number of componentes (k)')
ylabel('mean square error')
