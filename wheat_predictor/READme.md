# Wheat Predictor
The challenge here is creating a computer vision application to identify different wheat heads present in an image. Some images can have no wheat and the method of evaluation is 
mAP(mean average precision) for different levels of intersection over union. 

### The Network
For this challenge, I chose a Single Shot Detector. It is a cascade forward net for determining the bounding box locations on a image and performs faster than the YOLO algorithm.
In this case, the implementation is in MATLAB. The number of bouding boxes for this detector is set to 20. The network architecture is printed upon execution of the MATLAB code.
The layers present in the net are as follows: 
```
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
analyzeNetwork(lgraph)

```

Initially, we begin with a base network taken from VGG16. The MATLAB framework for NNs then allows us to remove the weights and alter the expected size of input images.
We also set the layers in which we perform the anchor box regression for the SSD. The function to estimate anchor box sizes is useful, it bases the boxes on the datset. Each 
anchor box can be selected by the user if they want. On top of this, in order to handle the scale problem, differen anchor boxes can be used at different points of the 
cascade forward process. The images came in a 1024x1024 RGB format. I changed them to RGB and scaled down by a factor of 2 in each dimension to accomodate the RAM requirements 
of my machine. Training took 58 hours for 4 epochs of about 3300 images. The number of classes here is the objects which we're searching for, in this case, wheat. 

### The Training Options
The training of this net used a standard SGDM(Stochastic Gradient Descent Momentum) method. The adam method can be chosen here. The options selected are specified by the MATLAB
function options are here:
```
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 4, ....
    'InitialLearnRate',4e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 1, ...
    'LearnRateDropFactor', 0.8, ...
    'MaxEpochs', 4, ...
    'VerboseFrequency', 30, ...
    'ExecutionEnvironment','cpu');
```
I had the misfortune of dealing with limited RAM on my GPU which forced me to use CPU only. This is the one painfull problem with using a cascade forward network, the RAM 
consuption is enormous. Even with 16 GB of RAM the batch size is still 4. Note that there is a learning rate drop after each epoch in order to reduce any drastic changes in
weights after the first epoch. The rate is set to 80% of each previous epoch. 

### Results
The network performed very well and also awfull simultaneously. It correctly identified all the areas which contained wheat heads, but used a multitude of bounding boxes to
obtain this. I attempted grouping these various bounding boxed using an SOM(self-organising map) which has interesting results. Below is an example showing an example of the
detection yielded from this detector.

### Conclusions
SSDs are interesting detectors and I expected the ability to use larger batch sizes for this challenge. Even with 512x512 images, the RAM is completely filled. 