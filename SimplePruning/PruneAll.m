%% Prune all NNs trained on this directory
clc;clear
% Load all names of NNs in directory
netfiles = dir('*.mat');
% Begin for loop trhu all of them
for i=1:length(netfiles)
    a = split(netfiles(i).name,'_');
    dataname = [char(a(1)) '_dataset'];
    % Load dataset
    data = load(dataname);
    x = data.([char(a(1)) 'Inputs']);
    t = data.([char(a(1)) 'Targets']);
    % Load network
    load(netfiles(i).name);
    run PruneAlgo.m
%     try
%         run PruneAlgo.m;
%     catch
%         warning('Problem when running neuron pruning. Assigning incorrect shapes');
%         warning('W shape = [%d,%d]', size(w,1), size(w,2));
%         warning('temp shape = [%d,%d]', size(temp,1), size(temp,2));
%     end
end
