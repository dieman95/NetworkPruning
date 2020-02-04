%% Create example networks to test pruning techniques
clc;clear;close all;
disp('Training NNs to test pruning techniques');
%% Simplefit
[x,t] = simplefit_dataset;
net = feedforwardnet([50 100 20]);
net.inputs{1}.processFcns = {};
net.outputs{4}.processFcns = {};
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y)
save('simplefit_net','net');
clear;
%% Vinyl
[x,t] = vinyl_dataset;
net = feedforwardnet([75 150 40]);
net.inputs{1}.processFcns = {};
net.outputs{4}.processFcns = {};
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y)
save('vinyl_net','net');
clear;
%% Abalone
[x,t] = abalone_dataset;
net = feedforwardnet([50 60 40]);
net.inputs{1}.processFcns = {};
net.outputs{4}.processFcns = {};
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y)
save('abalone_net','net');
clear;
%% Engine
[x,t] = engine_dataset;
net = feedforwardnet([75 150 40]);
net.inputs{1}.processFcns = {};
net.outputs{4}.processFcns = {};
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y)
save('engine_net','net');