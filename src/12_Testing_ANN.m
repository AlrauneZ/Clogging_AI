clear
close all
clc

%% Import data
data = readmatrix('MATLAB_fav_CN.csv');
% Favorable condition: fav_CN.csv: Coordination number; fav_HC.csv: Hydraulic conductivity; fav_SC.csv: Surface coverage; fav_VF.csv: Void fraction
% Unfavorable condition: unfav_CN.csv: Coordination number; unfav_HC.csv: Hydraulic conductivity; unfav_SC.csv: Surface coverage; unfav_VF.csv: Void fraction

p = data(:,1:5);
l = length(data);
rng(70);  % Seed of random splitting
id = randperm(l);
tt_id = id(1:round(l*(10/100))); % The indices of test dataset
tv_id = id(((round(l*(10/100)))+1):end); % The indices of training dataset

%% Normalization of data
    for j = 1:4
    pp(:,j) = (p(:,j)-min(p(:,j)))/(max(p(:,j))-min(p(:,j)));
    end
    
%% Build the neural network (BP)

tt = pp(tt_id,:);   % Test dataset for input parameters
tl = pp(tv_id,:);   % Training dataset for input parameters

xt = tl';            % Training dataset: Input parameters
yt = p(tv_id,5)';    % Training dataset: Output parameter
xtt = tt';           % Test dataset: Input parameters
yo = p(tt_id,5)';    % Test dataset: Output parameter


net = feedforwardnet(4);   % The optimal number of neurons in the single hidden layer is set here.
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.001;
net.trainParam.goal = 0.00001;
net = train(net,xt,yt);
an = sim(net,xtt);

yTest = an;  % Simulated value by the artificial neural network
yTestTrue = yo;  % True value by LBM simulation

mse = (sum((yTest - yTestTrue).^2)) / (length(yo))  % MSE for the test dataset
mae = (sum(abs(yTest - yTestTrue))) / (length(yo))  % MAE for the test dataset
rt = 1 - (sum((yTestTrue - yTest).^2))/(sum((yTestTrue - mean(yTestTrue)).^2))  % R2 for the test dataset

