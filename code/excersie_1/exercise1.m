X = [ -0.1 -0.1 +0.1 +0.1;
      -0.1 +0.1 +0.1 -0.1];
T = [1 1 0 0];

plotpv(X,T);

Pnew = [-0.5;-0.4];

net = newp(X,T);

net = init(net);

[net, tr_descr] = train(net,X,T);

net.trainParam.epochs = 20;

plotpv(X,T);
plotpc(net.IW{1,1},net.b{1,1});

%sim(net, Pnew);

%net.IW{1,1};

