clc;clear;
addpath(genpath('.'));
%% prepare dataset
[dict_img, dict_pt, target_img, clean_img, target_pts] = load_data('./data');