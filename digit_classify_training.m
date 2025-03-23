
% Saurabh Kumar, Indian Institute of Technology, Indore (Communication and
% Signal Processing - Master's of Technology)
% Prediction of MNIST dataset

clc
clear all
close all
load("networkLayer.mat") % Loading Network Layer
path='C:\Saurabh\M.Tech CSP\Semester 3\My_Practice\Digit_data'; % MNIST dataset path

digitdata=imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');

[train_img, test_img]=splitEachLabel(digitdata,0.8,'randomized');
f=zeros(8000,784);
for i=1:numel(train_img.Files)
    p=cell2mat(train_img.Files(i));
    r=imread(p);
    
     r=double(r)/255;
 
    f(i,:)=reshape(r,1,28*28);
end
g=zeros(2000,784);
for i=1:numel(test_img.Files)
    p=cell2mat(test_img.Files(i));
    r=imread(p);
    r=double(r)/255;
    g(i,:)=reshape(r,1,28*28);
end
% Define the labels for training and testing sets
train_labels = train_img.Labels;
test_labels = test_img.Labels;

train_labels = categorical(train_labels);
test_labels = categorical(test_labels);


options = trainingOptions('adam', ...
    'InitialLearnRate', 0.005, ...
    'ValidationData',{g,test_labels}, ...
    'ValidationFrequency',32, ...
    'MiniBatchSize', 128, ...
    'MaxEpochs', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

net = trainNetwork(f,train_labels, layers_2,options);


