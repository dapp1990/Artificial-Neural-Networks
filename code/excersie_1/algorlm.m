clear;
clc;
close all;

%%%%%%%%%%%
% https://nl.mathworks.com/help/nnet/ug/create-configure-and-initialize-multilayer-neural-networks.html
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

%generation of examples and targets
x=0:0.2:3*pi; y=sin(x);
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%num_neurons = 5;
%num_neurons = 25;
num_neurons = 50;

%creation of networks
net1=feedforwardnet(num_neurons,'trainlm');
net2=feedforwardnet(num_neurons,'traingd');

net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};


%training and simulation
net2.trainParam.epochs=1;
net2=train(net2,p,t);
a12=sim(net2,0p);

%plots
figure
subplot(3,3,1);
plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','trainlm','traingd','Location', 'northeast');
subplot(3,3,2);
postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
subplot(3,3,3);
postregm(cell2mat(a21),y);
%
subplot(3,3,4);
plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g');
title('15 epochs');
legend('target','trainlm','traingd','Location', 'northeast');
subplot(3,3,5);
postregm(cell2mat(a12),y);
subplot(3,3,6);
postregm(cell2mat(a22),y);
%
subplot(3,3,7);
plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
title('1000 epochs');
legend('target','trainlm','traingd','Location', 'northeast');
subplot(3,3,8);
postregm(cell2mat(a13),y);
subplot(3,3,9);
postregm(cell2mat(a23),y);
