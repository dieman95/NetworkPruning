%% Model Pruning
% Take any pre-existing network and perform weight and neuron pruning
% Source code in tensoorflow/python: 
% https://colab.research.google.com/drive/1GBLFxyFQtTTve_EE5y1Ulo0RwnKk_h6J?usp=drive_open#scrollTo=gptJSUTOauhM
% Explanation here:
% https://towardsdatascience.com/pruning-deep-neural-network-56cae1ec5505

% 1. Load the network
load('example_net.mat');
view(net);

% 2. Weight pruning
ranks = [];
for i=1:length(net.layers)
    if i == 1
        w = net.IW{i};
    else
        w = net.LW{i,i-1};
    end
    
end