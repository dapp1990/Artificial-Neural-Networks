load('Data_Problem1_regression.mat')

%student number r0605947

% linspace
% meshgrid
% TriScqtteredInterp
% mesh

Tnew = (9*T1 + 7*T2 + 6*T3 + 5*T4 + 4*T5) / (9+7+6+5+4);
Inputs = [X1 X2];

% Get independent samples
rng(97654); %Use to hqve the same indices to compare results
Indices = randperm(size(Tnew,1)); 

Itraning = Indices(1:1000);
Ivalidation = Indices(1001:2000);
Itest = Indices(2001:3000);

Xtraning = Inputs(Indices(1:1000),:);
Tvalidation = Tnew(Indices(1:1000),:);
Ttest = Tnew(Indices(1001:2000),:);

%plot the surface of training set
%points = 1:1000;
%scatter(Xtraning);

%creation of networks
net=feedforwardnet(5,'traingd');

%training and simulation
net.trainParam.epochs=5000;
net=train(net,Xtraning',Tvalidation');
%a12=sim(net,Ttest');

%plots
%figure
%subplot(2,2,1);
%scatter(points, Ttes);
%title('First NN');
%legend('Test set');
%plot(points,Ttes,'bx',p,cell2mat(a11),'r'); % plot the sine function and the output of the networks

%figure
%subplot(3,3,1);
%plot(p,cell2mat(a12),'r'); % plot the sine function and the output of the networks
%title('1 epoch');
%legend('target','trainlm','traingd','Location', 'northeast');

