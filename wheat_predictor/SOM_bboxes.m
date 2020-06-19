function [grouped_bboxes,net]=SOM_bboxes(selbox,dim_sq)
%the purpose of this script is to group the bounding boxes which arise from
%the wheat predictor. In the SSD that i've trained, it is very capable of
%locating all the wheat which is present. It is not however placing large
%bounding boxes around this wheat, but rather small ones. It is well
%covering the wheat. 
%What i'm going to do here, is take all bounding boxes which are predicted
%from the SSD and group them into a larger one wich fits the Intersection
%over Union citeria better. 

%% data from the SSD
%what we require here is the boxes from the ssd that have been passed over
%by the select strongest bounding box routine. This will ensure we do not
%have too larger a burden here. The SOM will proceed to group objects
%together. 
inputs = selbox;

%% dimensions of the SOM
% Create a Self-Organizing Map
dimension1 = dim_sq;
dimension2 = dim_sq;
net = selforgmap([dimension1 dimension2]);

%% training of the SOM
% Train the Network
[net,tr] = train(net,inputs');

% Test the Network
outputs = net(inputs');
vectorized_outputs=vec2ind(outputs);

grouped_bboxes = vectorized_outputs;
