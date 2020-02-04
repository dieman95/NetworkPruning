%% Pruning Algorithm (weight and neuron)
net_or = net;
y = net(x);
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
        temp = floor(tiedrank(nor));
        temp = repmat(temp,size(w,2),1);
        ranks{i} = temp';
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
title([a(1) ' Weight pruning mse values'])
figure;
plot([0 .25, .50, .60, .70, .80, .90, .95, .97, .99],[perc perc2])
title([a(1) ' Weight pruning percentage values'])

figure;
plot([0 .25, .50, .60, .70, .80, .90, .95, .97, .99],[perf perf3])
title([a(1) ' Neuron pruning mse values'])
figure;
plot([0 .25, .50, .60, .70, .80, .90, .95, .97, .99],[perc perc3])
title([a(1) ' Neuron pruning percentage values'])
