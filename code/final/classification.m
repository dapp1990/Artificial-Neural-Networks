clear;
clc;
close all;


% **************** Preprocessing data ****************
rng(97654); %Use to hqve the same indices to compare results
% Load data
% Positive 4 and 5
% Negative 6

all_raw_set = importdata('/home/r0605947/Documents/MATLAB/final/winequality-red.csv',';');
all_set = all_raw_set.data;
positive_samples_4 = all_set(all_set(:,end) == 4, :);
positive_samples_5 = all_set(all_set(:,end) == 5, :);
negative_samples = all_set(all_set(:,end) == 6, :);

% Concatenate set as input set 
X = [positive_samples_4(:,1:end-1); 
    positive_samples_5(:,1:end-1);
    negative_samples(:,1:end-1)];

% Convert set into suitable format for output, 1 for positive and -1 for
% negative class and concatenate them as target set
n_positive = length(positive_samples_4) + length(positive_samples_5);
n_negative = length(negative_samples);
Y = [ones(n_positive,1);
    -ones(n_negative,1)];

% Split data into training set, validation set and test set.
% I used randperm to have an evenly distribution od the samples.
% Total oof samples are 1372 I roughly divided 80% for training set 
% (1000 samples) and 20% for test set (372 samples).randIdx

n_samples = n_positive + n_negative;
randIdx = randperm(n_samples);

training_set = X(randIdx(1:1000),:);
validation_set = Y(randIdx(1:1000),:);
test_set = X(randIdx(1001:end),:);
ground_truth =  Y(randIdx(1001:end),:);

% **************** task 1 ****************


% Create network without reduced dimensionality
result_matrix = zeros(50);

for neurons=5:5:50
net1=feedforwardnet(neurons,'trainlm');

% Training and simulation
net1.trainParam.epochs=1000;
net1=train(net1,training_set',validation_set');

% Performance on test set
predictions = net1(test_set');
ccr1 = (sum(sign(predictions) == ground_truth')*100)/length(ground_truth);
result_matrix(neurons) = ccr1;
end

result_matrix( ~any(result_matrix,2), : ) = [];
result_matrix( :, ~any(result_matrix,1) ) = [];

figure;
x = 5:5:50;
bar(x,result_matrix);

% **************** task 2 ****************

% Find the k optimal components
conv_x = cov(training_set);
[E,s] = eigs(conv_x);
cumsum_values = diag(cumsum(s,100));

figure;
plot(cumsum_values);
title('Comulative sum of the largest eigenvalues');
xlabel('k component');
ylabel('Comulative eigenvalues');


Process training and test sets to apply PCA
k = 11;

result_matrix = zeros(50,11);
for neurons=5:5:50
for k=1:11
% Apply PCA to training set
conv_x = cov(training_set);
[E,s] = eig(conv_x);
[s, indx] = sort(diag(s), 'descend');
E =E(:,indx);
projection_training = E(:,1:k)';
training_set_reduction = projection_training*training_set';


% Apply PCA to test set
conv_x = cov(test_set);
[E,s] = eig(conv_x);
[s, indx] = sort(diag(s), 'descend');
E =E(:,indx);
projection_test = E(:,1:k)';
test_set_reduction = projection_test*test_set';


% **************** task 3 ****************
% Create network with reduced dimensionality
net2=feedforwardnet(neurons,'trainlm');

% Training and simulation
net2.trainParam.epochs=1000;
net2=train(net2,training_set_reduction,validation_set');

% Performance on test set
predictions = net2(test_set_reduction);
ccr2 = (sum(sign(predictions) == ground_truth')*100)/length(ground_truth);
fprintf('neurons=%f k=%f CRR=%f \n',neurons,k, ccr2);
result_matrix(neurons,k) = ccr2;
end 
end

result_matrix( ~any(result_matrix,2), : ) = [];

figure;
x = 5:5:50;
bar(x, result_matrix);
