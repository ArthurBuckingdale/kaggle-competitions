%the purpose of this script is to model which passengers would survive the
%crash of the Titanic. I will be implementing a cascade forward shallow neural
%network which will process the training data. A few dropout layers may
%potentially be added as we need to estimate the fate of 500 passengers
%with data from only 900. This should be interesting.

%% importing and pre processing data
%here, we are going to import the data and pre process it. We must remove
%things like names and room numbers. Categorical variables must be encoded.
%We must also perform a column by column feature scaling.
close all
clear all

train_table=readtable("C:\Users\Captain\Documents\kaggle Data\train.csv");
test_table=readtable("C:\Users\Captain\Documents\kaggle Data\test.csv");
 
%we first remove variables which have no effect like name,cabin ID and
%ticket number. The cabin Ids could be useful but there aren't enough here
%to be pertinent. The location where they embarked could be important if no
%ticket price or passenger class was available. 
train_table = removevars(train_table,{'Name'});
train_table = removevars(train_table,{'Ticket'});
train_table = removevars(train_table,{'Cabin'});
train_table = removevars(train_table,{'Embarked'});


%we now want to replace all NaN values in the data. This will be completed
%by taking the mean age. We'll also take this time to encode some
%catogrical variables. Remember we cannot create an ordinal nature for our
%categorical variables i.e man = 0 woman = 1. Because one is not more or
%less than another. We need to add to colums to the table. Also, i'm
%missing a MATLAB toolbox to deal with nan variables in a 

train_table.Male(1)=1; %add col
train_table.Female(1)=1; %add col 
train_table.class1(1)=1;
train_table.class2(1)=1;
train_table.class3(1)=1;
[zz,xx]=size(train_table); 
for i=1:zz
    age_mean=round(mean(train_table.Age(:),'omitnan'));
    if isnan(train_table.Age(i)) %adding mean age
        train_table.Age(i)=age_mean;
    end
    if strcmp(train_table.Sex(i),'male') == 1
        train_table.Male(i)=1;
        train_table.Female(i)=0;
    elseif strcmp(train_table.Sex(i),'female') == 1
        train_table.Male(i)=0;
        train_table.Female(i)=1;
    end
    if train_table.Pclass(i) == 1
        train_table.class1(i)=1;
        train_table.class2(i)=0;
        train_table.class3(i)=0;
    elseif train_table.Pclass(i) == 2
        train_table.class1(i)=0;
        train_table.class2(i)=1;
        train_table.class3(i)=0;
    elseif train_table.Pclass(i) == 3
        train_table.class1(i)=0;
        train_table.class2(i)=0;
        train_table.class3(i)=1;
    end
end

%removing the original column containing gender
train_table = removevars(train_table,{'Sex'});
train_table = removevars(train_table,{'Pclass'});
%splitting into the independent and dependent variables
%we have the dependent variable in column 2 of train_table we'll assign it
%to the ground truth and take it out.
ground_truth = train_table.Survived;
train_table = removevars(train_table,{'Survived'});

%last remove the column containing the passenger labels, it does nothing
%here
train_table = removevars(train_table,{'PassengerId'});


%finally we need to rescale the data in each column of the train_table. We
%first want to remove this from a table type and make it a matrix type.
train_data=table2array(train_table);
colmin=min(train_data);
colmax=max(train_data);
train_data=rescale(train_data,'InputMin',colmin,'InputMax',colmax);

%% defining the architecture for the model
%here is where we define the parameters for training the network. We will
%experiment with different types of activation layers batch sizes and so
%on. This will allow for the best results. 


net = cascadeforwardnet(300);
%net.numInputs = 9;
disp(net)
net = train(net,train_data',ground_truth');
%  y = net(x);
%  perf = perform(net,y,ground_truth)