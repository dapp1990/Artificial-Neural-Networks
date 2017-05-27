clear;
clc;
close all;

% daniel perez -> danielprz


% **************** Preprocessing data ****************
capitals_letters = prprob;
lowercase_letters = lowercase;

X= [lowercase_letters,capitals_letters];
X(X==0)=-1;

% Print all letters
figure;
for i=1:35
    subplot(5,7,i);
    colormap(gray);
    imagesc(reshape(X(:,i),5,7)',[0,1]);
    axis image;
    set(gca,'xtick',[],'ytick',[]);
end

% **************** task 1 ****************
% Create Hopfield Network
num_characters=5;
T = X(:,1:num_characters);
net = newhop(T);

% Distor characters
err=[];

for lo=1:10000
    new_T = zeros(35, num_characters);
    num_pix=3;

    for i=1:num_characters
        indx = randperm(35);
        new_T(:,i) = T(:,i);
        for j=1:num_pix
            new_T(indx(j),i) = T(indx(j),i) * -1;
        end
    end

    [y,Pf,Af] = sim(net,{num_characters 50},{},new_T);

    prediction = sign(y{end});

    err = [err immse(prediction,T)];
end

mean_error = mean(err);

figure;
for i=1:5
    subplot(3,5,i); 
    colormap(gray);
    imagesc(reshape(T(:,i),5,7)',[0,1]);
    axis image;
    set(gca,'xtick',[],'ytick',[]);
    
    subplot(3,5,i+5);
    colormap(gray);
    imagesc(reshape(new_T(:,i),5,7)',[0,1]);
    axis image;
    set(gca,'xtick',[],'ytick',[]);
    
    subplot(3,5,i+10);
    colormap(gray);
    imagesc(reshape(prediction(:,i),5,7)',[0,1]);
    axis image;
    set(gca,'xtick',[],'ytick',[]);
end

% **************** task 2 ****************
erro_by_p = [];
spurious_by_p = [];
spurious_state = [];
for P=1:35
    % Create Hopfield Network
    T = X(:,1:P);
    net = newhop(T);
    err=[];
    inner_counter = 0;
    for iteration=1:100
        new_T = zeros(35, P);
        num_pix=3;
    
        for i=1:P
            indx = randperm(35);
            new_T(:,i) = T(:,i);
            for j=1:num_pix
                new_T(indx(j),i) = T(indx(j),i) * -1;
            end
        end
    
        [y,Pf,Af] = sim(net,{P 50},{},new_T);
        
        if (y{end} ~= y{end-1})
            fprint('Hopfield network does not converge \n');
            break;
        end
        
        prediction = sign(y{end});
    
        err = [err sum(sum(prediction ~= T))];
        
        for psub = 1:P
            ynext = repmat(prediction(:,psub),1,P);
            if(sum(sum(ynext ~= T)==0)==0)
                fprintf('Found spurious state \n');
                spurious_state = ...
                    [spurious_state prediction(:,psub)];
                inner_counter = inner_counter + 1;
            end
        end
    end
    
    spurious_by_p = [spurious_by_p (inner_counter/100)];
    erro_by_p = [erro_by_p mean(err)];
end

figure;
plot(erro_by_p);
hold on;
plot(spurious_by_p);

figure;
for i=1:10
    subplot(1,10,i);
    colormap(gray);
    imagesc(reshape(spurious_state(:,i),5,7)',[0,1]);
    axis image;
    set(gca,'xtick',[],'ytick',[]);
end

% Theoretical Heb-rule
figure;
Xs = 1:size(X,2);
sig = sqrt(Xs/size(X,1));
Z = 1./sig;
P_error = normcdf(ones(size(Xs))*Inf) - normcdf(Z);
plot(Xs,P_error);

% **************** task 3 ****************
num_characters=25;
T = X(:,1:num_characters);
Target = ...
    [T T T T T T T T T T T T T T T T ...
    T T T T T T T T T T T T T T T T T T T T T T T T];

t_length = length(Target);

% Training set
training_set = zeros(35, t_length);
num_pix=3;

for i=1:t_length
    indx = randperm(35);
    training_set(:,i) = Target(:,i);
    for j=1:num_pix
        training_set(indx(j),i) = Target(indx(j),i) * -1;
    end
end


randIdx = randperm(1000);
t_set = training_set(:,randIdx(1:800));
validation_set = Target(:,randIdx(1:800));
test_set = training_set(:,randIdx(801:end));
ground_truth =  Target(:,randIdx(801:end));


net1=feedforwardnet(10,'trainlm');

% Training and simulation
net1.trainParam.epochs=500;
net1=train(net1,t_set,validation_set);

% Performance on test set
predictions = net1(test_set);
predictions = sign(predictions);

ccr1 = (sum(sum(predictions == ground_truth)))/(200*35);

figure;
for i=1:5
    subplot(3,5,i+5);
    colormap(gray);
    imagesc(reshape(test_set(:,i),5,7)',[0,1]);
    axis image;
    set(gca,'xtick',[],'ytick',[]);

    subplot(3,5,i+10);
    colormap(gray);
    imagesc(reshape(predictions(:,i),5,7)',[0,1]);
    axis image;
    set(gca,'xtick',[],'ytick',[]);


    subplot(3,5,i);
    colormap(gray);
    imagesc(reshape(ground_truth(:,i),5,7)',[0,1]);
    axis image;
    set(gca,'xtick',[],'ytick',[]);
end