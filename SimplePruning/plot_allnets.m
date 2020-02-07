function plot_allnets(directo,out_name,forma,interv)
%UNTITLED Summary of this function goes here
%   directo: directory where all the networks are, i.e 'abalone/*.mat'

% Load all the names of networks in directory
net_files = dir(directo);

if nargin > 3 
    k = min(interv(2),length(net_files));
    k1 = max(interv(1),1);
else
    k1 = 1;
    k = length(net_files);
end
for i=k1:k
    load([net_files(i).folder '/' net_files(i).name]);
    if i == k1
        f = figure('WindowState','maximize');
        plot(y,'DisplayName','Original');
        hold on;
        plot(y2,'DisplayName',net_files(i).name);
    else
        plot(y2,'DisplayName',net_files(i).name);
    end
end
legend('show');
saveas(f,[out_name '_' num2str(k1) '_' num2str(k)],forma)