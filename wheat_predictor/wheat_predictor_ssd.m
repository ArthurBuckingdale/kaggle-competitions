%the purpose of this script is to obtain the wheat data set and use it to
%train a single shot detector to identify their locations on on image.
%The original purpose of this being to make a competition submission. But I
%didn't realise the test data is hidden. It makes perfect sense, but I'm
%not a python master and I do not have the toolbox to export this into
%python and run it(It is not overly complicated to do so).

%temporary
close all
clear all


%% data pre processing for bounding boxes
%the first step will be to extract the bounding box information out of the
%csv and save that into a datastore. The next step is reading the images
%into a datastore. The final step being combining those into one datastore
%to train our ssd algorithm.

%create and size the tables
cell_one=readtable("G:\data-sets-kaggle\wheat-detector\train.csv");
unique_ids=unique(cell_one(:,1));
[vv,~]=size(cell_one);
[qq,~]=size(unique_ids);
cell_one=sortrows(cell_one,1);

%so because the data set is so large we need to perform some grouping by
%unique id. We fortunately already have the id's sorted from our CSV which
%makes this one less step. This is a bit messy since we must read the
%bounding boxes as strings and convert them to vectors
kk=1;
ll=1;
blds_prep{qq,1}=zeros();
blds_prep{qq,1}(1,4)=zeros();
for i=1:(vv-1)
    str_compare=strcmp(cell_one.image_id(i),cell_one.image_id(i+1));
    if str_compare == true
        blds_prep{kk,2}=cell_one.image_id(i);
        %blds_prep{kk,2}(ll,:)='wheat';
        str_vect=cell_one.bbox(i);
        blds_prep{kk,1}(ll,:)=str2num(str_vect{1});
        for u=1:4
            if blds_prep{kk,1}(ll,u)==0
                blds_prep{kk,1}(ll,u)=1;
            end
        end
        blds_prep{kk,1}(ll,:)=bboxresize(round(blds_prep{kk,1}(ll,:)),0.5);
        
        
        ll=ll+1;
    else
        blds_prep{kk,2}=cell_one.image_id(i);
        %blds_prep{kk,2}(ll,:)='wheat';
        str_vect=cell_one.bbox(i);
        blds_prep{kk,1}(ll,:)=str2num(str_vect{1});
        for u=1:4
            if blds_prep{kk,1}(ll,u)==0
                blds_prep{kk,1}(ll,u)=1;
            end
        end
        blds_prep{kk,1}(ll,:)=bboxresize(round(blds_prep{kk,1}(ll,:)),0.5);
        kk=kk+1;
        ll=1;
    end
end

%blds_prep=sortrows(blds_prep,2);

%table_blds=cell2table(blds_prep,'VariableNames',{'Wheat'});
%place our data into the bounding box datastore function



%% data preprocessing for images
%so we can perform some image augmentation here. I'm not going to add this
%on the first iteration.
imds=imageDatastore('G:\data-sets-kaggle\wheat-detector\train\','ReadFcn',@half_scale);

%% data combining for the computer vision application
%so computer vision requires the bouding boxes to be associated with an
%image. In this case, we need to add some bouding boxes. Our CSV with the
%bbox data does't have null entries, and we have some bbox-less images
%here.


%find the bbox values that we're missing
for j=1:length(imds.Files)
    diff_cell{j,1}=char(imds.Files{j,1}(end-12:end-4));
end

%compare the tables
unique_ids=table2cell(unique_ids);
difference_vals=ismember(diff_cell,unique_ids);

%fill in values that should be empty of bboxes
gg=1;
for k=1:length(diff_cell)
    if difference_vals(k,1) == 1
        blds_prep_two{k,1}=blds_prep{gg,1};
        gg=gg+1;
    else
        blds_prep_two{k,1}=[];
    end
end


%% scaling down the images and bounding boxes.
%unfortunately I do not have enough ram to use the stride of VGG16 with a
%1024x1024 image(increasae the stride and the intermediate data is smaller).
% For this reason, I am going to scale the images and
%bouding boxes down by a scale of 0.5.

%obtain bbox datastore
blds_prep_two=cell2table(blds_prep_two,'VariableNames',{'Wheat'});
blds=boxLabelDatastore(blds_prep_two);

%combine our two types of datastores
training_data = combine(imds,blds);


%% visual validation
%in computer vision, we want to visually validate some of our training
%images to ensure we've assigned the correct bounding boxes. I cannot
%stress enough the time saving this can have.

rand_array = randi([0 length(imds.Files)],10,1); %sample 10 random images
[pp,~]=size(rand_array);
close all
for i=1:pp
    idx=rand_array(i);
    image=readimage(training_data.UnderlyingDatastores{1, 1},i);
    bboxes=training_data.UnderlyingDatastores{1, 2}.LabelData{i,1};
    [nn,mm,gg]=size(image);
    [xx,~]=size(bboxes);
    figure
    hold on
    imshow(image)
    for j=1:xx
        rectangle('Position',bboxes(j,:))
    end
    hold off
end

close all
%% defining and setting up the SSD.
%the next step is the architecture for the detector and defining the number
%of anchor boxes. Fortunately, we have lots of tools to complete this
%process for us. A simple VGG16 is a great place to start for us
baseNetwork =layerGraph(vgg16('Weights','none'));
imageSize = [512 512 1];
numClasses = 1;
numAnchors=20;
disp('starting anchor box estimation. Wait forever now...')
[anchorBoxes,meanIoU] = estimateAnchorBoxes(training_data,numAnchors);
fprintf('The mean IoU for %d anchor boxes is %d \n',numAnchors,meanIoU)
for i=1:4
    predlyr{1,i}=anchorBoxes(:,:);
end
pred_lyr_name={'relu1_2','relu2_2','relu3_2','relu4_2'};
lgraph = ssdLayers(imageSize,numClasses,baseNetwork,predlyr,pred_lyr_name);
newlyr = imageInputLayer([512 512 1],'Name','input','Normalization','None')   ;
lgraph = replaceLayer(lgraph,'input',newlyr);
%analyzeNetwork(lgraph)



%% setting the training options
%allowing us to modify both the gradient descent algorithm and all of its
%parameters in one convinient location.

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 4, ....
    'InitialLearnRate',4e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 1, ...
    'LearnRateDropFactor', 0.8, ...
    'MaxEpochs', 4, ...
    'VerboseFrequency', 30, ...
    'ExecutionEnvironment','cpu');

%% training the network
%with the parallel computing toolkit ready to rock, we can execute this on
%local GPU and basically destroy the world.

[detector, info] = trainSSDObjectDetector(training_data,lgraph,options);

%% more training
%it seems we're doing well, more training is necessary. Note that we've
%just swapped out the layer graph for the detector itself to continue
%training with the work that is completed already.
[detector, info] = trainSSDObjectDetector(training_data,detector,options);

%% hyper parameter tuning
%note that there is a lot of performance gain possible by tuning the hyper
%parameters. It's just for me 4 epochs is taking 50+ hours and I don't want
%my PC running all day and night. So note room for improvement here.

%% running the test set.
%we will now fill our test set with images and pass them through the
%detector. We will decide on a confidence rating to maximise the possible
%false postitive to precision ratio.

image=half_scale("G:\data-sets-kaggle\wheat-detector\test\2fd875eaa.jpg");
[bbox,score]=detect(detector,image,'Threshold',0.585);
[selbox,selscore]=selectStrongestBbox(bbox,score,'RatioType','Min','OverlapThreshold',0.5);
hold on
imshow(image)
for i=1:length(selbox)
    rectangle('Position',selbox(i,:))
end
hold off
%% grouping the bounding boxes together.
%we have a pretty good object detection here. We're just lacking the
%correct intersection over union dimensions that will succeed a
%competition.

dim_of_init_SOM = 10;
num_iterations = 40;
acceptance_cond = 0;
i=2;
new_dim(1)=dim_of_init_SOM;
[grouped_bboxes,new_dim(2),new_net]=stochastic_SOM(selbox,num_iterations,dim_of_init_SOM);
while acceptance_cond == 0
    if new_dim(i) < new_dim(i-1)-3
        [grouped_bboxes,new_dim(i+1),new_net]=stochastic_SOM(selbox,num_iterations,new_dim(i));
        i=i+1;
    else
        final_SOM=new_net;
        final_bboxes=grouped_bboxes;
        acceptance_cond = 1;
        fprintf('Stochastic Estimation of Optimal Grouped BBoxes is %d \n',length(unique(grouped_bboxes)))
    end
end

%% combine these BBOXES into larger ones.
%now that we've estimated the optimal number of bounding boxes, we want to
%cobine the groupings returned from the SOM into larger ones. This function
%will take the bouding boxes. 

%display the vector which tells us which bin to place each of the selboxes
%in. We'll now create a nested for loop to bin all of these. There are
%faster ways to solve this with better routines than comparing two for
%loops. To increse speed however, we'll star from the end and count
%downwards so we check one less item per loop.
disp(final_bboxes)

for i=1:length(final_bboxes)
    for j=1:length(unique(grouped_bboxes))
        if final_bboxes(i) == j
            final_grouped_bboxes(j).bbox(i,:)=selbox(i,:);
        end
    end
end

%the last step here is to find the extremety of each column corresponding
%to the bouding box way of marking (xmin,ymin,width,height). Again, might
%be a more elegant way than nested for loops, but don't mess with
%simplicity in my opinion.
disp(final_grouped_bboxes)

for i=1:length(final_bboxes)
    for j=1:length(unique(grouped_bboxes))
        if ~isempty(final_grouped_bboxes(j).bbox)
            xmin(j)=min(nonzeros(final_grouped_bboxes(j).bbox(:,1)));
            ymin(j)=min(nonzeros(final_grouped_bboxes(j).bbox(:,2)));
            wid(j)=max(nonzeros(final_grouped_bboxes(j).bbox(:,1)))-min(nonzeros(final_grouped_bboxes(j).bbox(:,1)));
            height(j)=max(nonzeros(final_grouped_bboxes(j).bbox(:,2)))-min(nonzeros(final_grouped_bboxes(j).bbox(:,2)));
        end
    end
end

%place them in a more convinient form for drawing a bounding box.
last_bbox=[xmin',ymin',wid',height'];

hold on
imshow(image)
for i=1:length(last_bbox)
    rectangle('Position',last_bbox(i,:))
end
hold off
