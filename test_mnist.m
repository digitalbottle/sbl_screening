clc;clear;
addpath(genpath('.'));
%% Load dataset
train_image = loadMNISTImages('./data/mnist/train-images.idx3-ubyte');
train_label = loadMNISTLabels('./data/mnist/train-labels.idx1-ubyte');
test_image = loadMNISTImages('./data/mnist/t10k-images.idx3-ubyte');
test_label = loadMNISTLabels('./data/mnist/t10k-labels.idx1-ubyte');
%% 