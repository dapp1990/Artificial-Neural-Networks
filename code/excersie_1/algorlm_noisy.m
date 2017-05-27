clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

%generation of examples and targets
x=0:0.1:3*pi; y=sin(x)+.1*randn(size(x));
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

num_neurons = 5;

%creation of networks
net1=feedforwardnet(num_neurons,'traingd');
net2=feedforwardnet(num_neurons,'traingda');
net3=feedforwardnet(num_neurons,'traincgf');
net4=feedforwardnet(num_neurons,'traincgp');
net5=feedforwardnet(num_neurons,'trainbfg');
net6=feedforwardnet(num_neurons,'trainlm');

net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};

net3.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net3.lw{2,1}=net1.lw{2,1};
net3.b{1}=net1.b{1};
net3.b{2}=net1.b{2};

net4.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net4.lw{2,1}=net1.lw{2,1};
net4.b{1}=net1.b{1};
net4.b{2}=net1.b{2};

net5.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net5.lw{2,1}=net1.lw{2,1};
net5.b{1}=net1.b{1};
net5.b{2}=net1.b{2};

net6.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net6.lw{2,1}=net1.lw{2,1};
net6.b{1}=net1.b{1};
net6.b{2}=net1.b{2};

%training and simulation
net1.trainParam.epochs=1;  % set the number of epochs for the training 
net2.trainParam.epochs=1;
net3.trainParam.epochs=1;
net4.trainParam.epochs=1;
net5.trainParam.epochs=1;
net6.trainParam.epochs=1;
net1=train(net1,p,t);   % train the networks
net2=train(net2,p,t);
net3=train(net3,p,t);
net4=train(net4,p,t);
net5=train(net5,p,t);
net6=train(net6,p,t);
a11=sim(net1,p); a21=sim(net2,p); a31=sim(net3,p); a41=sim(net4,p); a51=sim(net5,p); a61=sim(net6,p);  % simulate the networks with the input vector p

net1.trainParam.epochs=10;  % set the number of epochs for the training 
net2.trainParam.epochs=10;
net3.trainParam.epochs=10;
net4.trainParam.epochs=10;
net5.trainParam.epochs=10;
net6.trainParam.epochs=10;
net1=train(net1,p,t);   % train the networks
net2=train(net2,p,t);
net3=train(net3,p,t);
net4=train(net4,p,t);
net5=train(net5,p,t);
net6=train(net6,p,t);
a12=sim(net1,p); a22=sim(net2,p); a32=sim(net3,p); a42=sim(net4,p); a52=sim(net5,p); a62=sim(net6,p);  % simulate the networks with the input vector p


net1.trainParam.epochs=50;
net2.trainParam.epochs=50;
net3.trainParam.epochs=50;
net4.trainParam.epochs=50;
net5.trainParam.epochs=50;
net6.trainParam.epochs=50;
net1=train(net1,p,t);   % train the networks
net2=train(net2,p,t);
net3=train(net3,p,t);
net4=train(net4,p,t);
net5=train(net5,p,t);
net6=train(net6,p,t);
a13=sim(net1,p); a23=sim(net2,p); a33=sim(net3,p); a43=sim(net4,p); a53=sim(net5,p); a63=sim(net6,p); 


net1.trainParam.epochs=100;
net2.trainParam.epochs=100;
net3.trainParam.epochs=100;
net4.trainParam.epochs=100;
net5.trainParam.epochs=100;
net6.trainParam.epochs=100;
net1=train(net1,p,t);   % train the networks
net2=train(net2,p,t);
net3=train(net3,p,t);
net4=train(net4,p,t);
net5=train(net5,p,t);
net6=train(net6,p,t);
a14=sim(net1,p); a24=sim(net2,p); a34=sim(net3,p); a44=sim(net4,p); a54=sim(net5,p); a64=sim(net6,p); 

net1.trainParam.epochs=500;
net2.trainParam.epochs=500;
net3.trainParam.epochs=500;
net4.trainParam.epochs=500;
net5.trainParam.epochs=500;
net6.trainParam.epochs=500;
net1=train(net1,p,t);   % train the networks
net2=train(net2,p,t);
net3=train(net3,p,t);
net4=train(net4,p,t);
net5=train(net5,p,t);
net6=train(net6,p,t);
a15=sim(net1,p); a25=sim(net2,p); a35=sim(net3,p); a45=sim(net4,p); a55=sim(net5,p); a65=sim(net6,p); 

net1.trainParam.epochs=1000;
net2.trainParam.epochs=1000;
net3.trainParam.epochs=1000;
net4.trainParam.epochs=1000;
net5.trainParam.epochs=1000;
net6.trainParam.epochs=1000;
net1=train(net1,p,t);   % train the networks
net2=train(net2,p,t);
net3=train(net3,p,t);
net4=train(net4,p,t);
net5=train(net5,p,t);
net6=train(net6,p,t);
a16=sim(net1,p); a26=sim(net2,p); a36=sim(net3,p); a46=sim(net4,p); a56=sim(net5,p); a66=sim(net6,p); 

%plots
figure
subplot(2,3,1);
plot(x,y,'bx',x,cell2mat(a11),'y',x,cell2mat(a21),'m', x,cell2mat(a31),'c', x,cell2mat(a41),'r', x,cell2mat(a51),'g',x,cell2mat(a61),'k'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','traingd','traingda','traincgf','traincgp', 'trainbfg', 'trainlm');
%
subplot(2,3,2);
plot(x,y,'bx',x,cell2mat(a12),'y',x,cell2mat(a22),'m', x,cell2mat(a32),'c', x,cell2mat(a42),'r', x,cell2mat(a52),'g',x,cell2mat(a62),'k'); % plot the sine function and the output of the networks
title('10 epoch');
legend('target','traingd','traingda','traincgf','traincgp', 'trainbfg', 'trainlm');
%
subplot(2,3,3);
plot(x,y,'bx',x,cell2mat(a13),'y',x,cell2mat(a23),'m', x,cell2mat(a33),'c', x,cell2mat(a43),'r', x,cell2mat(a53),'g',x,cell2mat(a63),'k'); % plot the sine function and the output of the networks
title('50 epoch');
legend('target','traingd','traingda','traincgf','traincgp', 'trainbfg', 'trainlm');

subplot(2,3,4);
plot(x,y,'bx',x,cell2mat(a14),'y',x,cell2mat(a24),'m', x,cell2mat(a34),'c', x,cell2mat(a44),'r', x,cell2mat(a54),'g',x,cell2mat(a64),'k'); % plot the sine function and the output of the networks
title('100 epoch');
legend('target','traingd','traingda','traincgf','traincgp', 'trainbfg', 'trainlm');

subplot(2,3,5);
plot(x,y,'bx',x,cell2mat(a15),'y',x,cell2mat(a25),'m', x,cell2mat(a35),'c', x,cell2mat(a45),'r', x,cell2mat(a55),'g',x,cell2mat(a65),'k'); % plot the sine function and the output of the networks
title('500 epoch');
legend('target','traingd','traingda','traincgf','traincgp', 'trainbfg', 'trainlm');

subplot(2,3,6);
plot(x,y,'bx',x,cell2mat(a16),'y',x,cell2mat(a26),'m', x,cell2mat(a36),'c', x,cell2mat(a46),'r', x,cell2mat(a56),'g',x,cell2mat(a66),'k'); % plot the sine function and the output of the networks
title('1000 epoch');
legend('target','traingd','traingda','traincgf','traincgp', 'trainbfg', 'trainlm');

%plots
figure
title('1 epoch');
subplot(2,3,1);
postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
title('This is my Figure')
subplot(2,3,2);
postregm(cell2mat(a21),y);
subplot(2,3,3);
postregm(cell2mat(a31),y);
subplot(2,3,4);
postregm(cell2mat(a41),y);
subplot(2,3,5);
postregm(cell2mat(a51),y);
subplot(2,3,6);
postregm(cell2mat(a61),y);


figure
title('10 epoch');
subplot(2,3,1);
postregm(cell2mat(a12),y); % perform a linear regression analysis and plot the result
subplot(2,3,2);
postregm(cell2mat(a22),y);
subplot(2,3,3);
postregm(cell2mat(a32),y);
subplot(2,3,4);
postregm(cell2mat(a42),y);
subplot(2,3,5);
postregm(cell2mat(a52),y);
subplot(2,3,6);
postregm(cell2mat(a62),y);


figure
title('50 epoch');
subplot(2,3,1);
postregm(cell2mat(a13),y); % perform a linear regression analysis and plot the result
subplot(2,3,2);
postregm(cell2mat(a23),y);
subplot(2,3,3);
postregm(cell2mat(a33),y);
subplot(2,3,4);
postregm(cell2mat(a43),y);
subplot(2,3,5);
postregm(cell2mat(a53),y);
subplot(2,3,6);
postregm(cell2mat(a63),y);



figure
title('100 epoch');
subplot(2,3,1);
postregm(cell2mat(a14),y); % perform a linear regression analysis and plot the result
subplot(2,3,2);
postregm(cell2mat(a24),y);
subplot(2,3,3);
postregm(cell2mat(a34),y);
subplot(2,3,4);
postregm(cell2mat(a44),y);
subplot(2,3,5);
postregm(cell2mat(a54),y);
subplot(2,3,6);
postregm(cell2mat(a64),y);

figure
title('500 epoch');
subplot(2,3,1);
postregm(cell2mat(a15),y); % perform a linear regression analysis and plot the result
subplot(2,3,2);
postregm(cell2mat(a25),y);
subplot(2,3,3);
postregm(cell2mat(a35),y);
subplot(2,3,4);
postregm(cell2mat(a45),y);
subplot(2,3,5);
postregm(cell2mat(a55),y);
subplot(2,3,6);
postregm(cell2mat(a65),y);

figure
title('1000 epoch');
subplot(2,3,1);
postregm(cell2mat(a16),y); % perform a linear regression analysis and plot the result
subplot(2,3,2);
postregm(cell2mat(a26),y);
subplot(2,3,3);
postregm(cell2mat(a36),y);
subplot(2,3,4);
postregm(cell2mat(a46),y);
subplot(2,3,5);
postregm(cell2mat(a56),y);
subplot(2,3,6);
postregm(cell2mat(a66),y);
