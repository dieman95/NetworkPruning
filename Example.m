%% Create a example network to test pruning techniques
[x,t] = simplefit_dataset;
net = feedforwardnet([30 30 10]);
net.inputs{1}.processFcns = {};
net.outputs{4}.processFcns = {};
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,y,t);
save('example_net','net');