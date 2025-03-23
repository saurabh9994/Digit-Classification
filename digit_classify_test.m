% Saurabh Kumar, Indian Institute of Technology, Indore (Communication and
% Signal Processing - Master's of Technology)
% Train the model by running "digit_classify.m" and save the network,
% provide trained network path below.

clc
trainedNet=load("C:\Saurabh\M.Tech CSP\Semester 3\My_Practice\trainedNet.mat"); % Trained Network Path
[filename,filepath]=uigetfile('*.*','Select_GreyScale_Input_Image'); % Select input image to be tested
f=strcat(filepath,filename);
i=imread(f);
in=im2gray(i);
figure()
imshow(in)
l=classify(trainedNet.net,reshape(in,1,784));
title(['Recognised digit is ' char(l) ])