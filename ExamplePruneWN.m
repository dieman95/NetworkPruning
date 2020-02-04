%% Model Pruning
% Take any pre-existing network and perform weight and neuron pruning
% Source code in tensoorflow/python: 
% https://colab.research.google.com/drive/1GBLFxyFQtTTve_EE5y1Ulo0RwnKk_h6J?usp=drive_open#scrollTo=gptJSUTOauhM
% Explanation here:
% https://towardsdatascience.com/pruning-deep-neural-network-56cae1ec5505
clc;clear;close all;
disp('Running model pruning example');
%% 1. Load the network (original)
load('example_net.mat');
view(net);
net_or = net;
% Record all outputs from the training set
[x,t] = simplefit_dataset;
y = net_or(x);
perf = perform(net_or,t,y);
perc = (abs(y-t)/t)*100;

%% 2. Weight pruning
perf2 = [];
perc2 = [];
for k = [.25, .50, .60, .70, .80, .90, .95, .97, .99]
    net = net_or;
    ranks = {};
    for i=1:length(net.layers)-1
        if i == 1
            w = net.IW{i};
        else
            w = net.LW{i,i-1};
        end
        ranks{i} = reshape(floor(tiedrank(abs(w))-1),size(w));
        lower_bound_rank = ceil(max(ranks{i}*k));
        ranks{i}(ranks{i} <= lower_bound_rank) = 0;
        ranks{i}(ranks{i} > lower_bound_rank) = 1;
        w = w.*ranks{i};
        if i == 1
            net.IW{i} = w;
        else
            net.LW{i,i-1} = w;
        end
    end
    % Test pruned network. Performance should decrease
    y2 = net(x);
    perf2 = [perf2 perform(net,t,y2)];
    perc2 = [perc2 (abs(y2-t)/t)*100];
end

%% 3. Neuron pruning
perf3 = [];
perc3 = [];
for k = [.25, .50, .60, .70, .80, .90, .95, .97, .99]
    net = net_or;
    ranks = {};
    for i=1:length(net.layers)-1
        if i == 1
            w = net.IW{i};
        else
            w = net.LW{i,i-1};
        end
        nor = vecnorm(w');
        %disp(size(nor));
        % nor = repmat(nor,size(w,1),1);
        %disp(size(nor));
        temp = floor(tiedrank(nor));
        temp = repmat(temp,size(w,1),1);
        ranks{i} = reshape(temp,size(w));
        %disp(size(ranks{i}));
        lower_bound_rank = ceil(max(max(ranks{i}*k)));
        ranks{i}(ranks{i} < lower_bound_rank) = 0;
        ranks{i}(ranks{i} >= lower_bound_rank) = 1;
        w = w.*ranks{i};
        if i == 1
            net.IW{i} = w;
        else
            net.LW{i,i-1} = w;
        end
    end
    % Test pruned network. Performance should decrease
    y2 = net(x);
    perf3 = [perf3 perform(net,t,y2)];
    perc3 = [perc3 (abs(y2-t)/t)*100];
end

%% Plot accuracy results
figure;
plot([0 .25, .50, .60, .70, .80, .90, .95, .97, .99],[perf perf2])
figure;
plot([0 .25, .50, .60, .70, .80, .90, .95, .97, .99],[perf perf3])


