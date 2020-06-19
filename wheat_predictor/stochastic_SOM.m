function [grouped_bboxes,new_dim,new_net]=stochastic_SOM(selbox,iterations,dim_sq)
%the purpose of this script is to train the SOM multiple different times so
%we can converge on an optimal number of total nodes it can have, ideally
%this would be the number of wheat heads present on an image, but that
%would be crazy.

for i = 1:iterations
    [bboxes(i).vect,net{i}]=SOM_bboxes(selbox,dim_sq);
    number_nodes(i)=length(unique(bboxes(i).vect(:,:)));
    fprintf('SOM %d contains %d nodes \n',i,number_nodes(i))
end

%% histogram of the stochastic pass
%nice to visulise the best initial grouping for the SOM
histogram(number_nodes)
h=histogram(number_nodes);
[maxcount, whichbin] = max(h.Values);

%% create a SOM with the new dimensions after the histogram peak
%take this SOM if you new one isn't below the improve threshold that we
%determine. 
new_dim=floor(sqrt((h.BinEdges(whichbin))));
disp(new_dim)
new_net = selforgmap([new_dim new_dim]);

%% training of the SOM
% Train the Network
[new_net,~] = train(new_net,selbox');
[new_net,~] = train(new_net,selbox');

outputs = new_net(selbox');
vectorized_outputs=vec2ind(outputs);
grouped_bboxes = vectorized_outputs;

disp(length(unique(grouped_bboxes)))
new_dim_2=floor(sqrt(length(unique(grouped_bboxes))));
%% post process to obtain higher IoU on bounding boxes
%so this is where the magic is really going to be. Taking the extremeties
%of the smaller bounding boxes, than converting them into larger ones. 