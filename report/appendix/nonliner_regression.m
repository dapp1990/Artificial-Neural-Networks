clear;
clc;
close all;

load('Data_Problem1_regression.mat')

%student number r0605947

% *********************** Preprocessing ***********************

Tnew = (9*T1 + 7*T2 + 6*T3 + 5*T4 + 4*T5) / (9+7+6+5+4);
Inputs = [X1 X2];

% *********************** Task 1 ***********************
% Get independent samples
rng(97654); %Use to have the same indices to compare results 
Indices = randperm(size(Tnew,1)); 

Itraning = Indices(1:1000);
Ivalidation = Indices(1001:2000);
Itest = Indices(2001:3000);

Xtraning = Inputs(Indices(1:1000),:);
Tvalidation = Tnew(Indices(1:1000),:);

Xtest = Inputs(Indices(1001:2000),:);
Ttest = Tnew(Indices(1001:2000),:);

% Plot the surface of training set
figure;
plot(Xtraning(:,1),Xtraning(:,2),'o');
title('Training set surface');
xlabel('X1');
ylabel('X2');

% *********************** Task 2 ***********************

max_epochs = 1000;

% creation of networks - 5 neurons
net5=feedforwardnet(5,'trainlm');
% training
net5.trainParam.epochs=max_epochs;
[net5,tr5]=train(net5,Xtraning',Tvalidation');

%creation of networks - 10 neurons
net10=feedforwardnet(10,'trainlm');
%training
net10.trainParam.epochs=max_epochs;
[net10,tr10]=train(net10,Xtraning',Tvalidation');

%creation of networks - 15 neurons
net15=feedforwardnet(15,'trainlm');
%training
net15.trainParam.epochs=max_epochs;
[net15,tr15]=train(net15,Xtraning',Tvalidation');

%creation of networks - 20 neurons
net20=feedforwardnet(20,'trainlm');
%training
net20.trainParam.epochs=max_epochs;
[net20,tr20]=train(net20,Xtraning',Tvalidation');

%creation of networks - 25 neurons
net25=feedforwardnet(25,'trainlm');
%training
net25.trainParam.epochs=max_epochs;
[net25,tr25]=train(net25,Xtraning',Tvalidation');

%creation of networks - 30 neurons
net30=feedforwardnet(30,'trainlm');
%training
net30.trainParam.epochs=max_epochs;
[net30,tr30]=train(net30,Xtraning',Tvalidation');

figure;
plotperform(tr5);
title('5 neurons');
figure;
plotperform(tr10);
title('10 neurons');
figure;
plotperform(tr15);
title('15 neurons');
figure;
plotperform(tr20);
title('20 neurons');
figure;
plotperform(tr25);
title('25 neurons');
figure;
plotperform(tr30);
title('30 neurons');


% *********************** Task 3 ***********************

net=feedforwardnet(30,'trainlm');
%training
net.trainParam.epochs=350;
net=train(net,Xtraning',Tvalidation');

Y_eval_tr = net(Xtest');
Y_eval = Y_eval_tr';

figure;
tri = delaunay(Xtest(:,1),Xtest(:,2));
trisurf(tri,Xtest(:,1),Xtest(:,2), Y_eval)
hold on
tri = delaunay(Xtest(:,1),Xtest(:,2));
trisurf(tri,Xtest(:,1),Xtest(:,2), Ttest)
title('Regression prediction');
xlabel('X1');
ylabel('X2');
zlabel('Target');
hold off

err = immse(Ttest, Y_eval);
fprintf('\n The mean-squared error is %0.4f\n', err);

iterations = 100;
err_m = (iterations);

for i = 1:iterations
    rng(i); % change seed for each iteration
    Indices = randperm(size(Tnew,1)); 

    Itraning = Indices(1:1000);
    Ivalidation = Indices(1001:2000);
    Itest = Indices(2001:3000);

    Xtraning = Inputs(Indices(1:1000),:);
    Tvalidation = Tnew(Indices(1:1000),:);

    Xtest = Inputs(Indices(1001:2000),:);
    Ttest = Tnew(Indices(1001:2000),:);


    net=feedforwardnet(30,'trainlm');
    %training
    net.trainParam.epochs=350;
    net=train(net,Xtraning',Tvalidation');

    Y_eval_tr = net(Xtest');
    Y_eval = Y_eval_tr';

    err_m(i) = immse(Ttest, Y_eval);
end

avg_error = mean(err_m);
fprintf('\n The average mean-squared error is %0.4f using %d iterations\n', avg_error, iterations);

figure;
plot(err_m);
