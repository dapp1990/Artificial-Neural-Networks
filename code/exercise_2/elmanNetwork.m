u(1)=randn;
x(1)=rand+sin(u(1));
y(1)=0.6*x(1);

for i=2:n
    u(i)=randn;
    x(i)=0.6*x(i-1)+sin(u(i));
    y(i)=x(i);
end

n_neurons = 5;

net = newelm(X,T,n_neurons);
