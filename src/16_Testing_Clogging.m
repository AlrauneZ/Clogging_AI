clear
close all
clc

%% Import data
data = readmatrix('MATLAB_fav_clogging.csv');
% fav_clogging.csv: Dataset of clogging under the favorable condition; unfav_clogging.csv: Dataset of clogging under the unfavorable condition
p = data(:,1:5);
l = length(data);
rng(70);
id = randperm(l);
tt_id = id(1:round(l*(10/100)));
tv_id = id(((round(l*(10/100)))+1):end);

%% Normalization of data
    for j = 1:4
    pp(:,j) = (p(:,j)-min(p(:,j)))/(max(p(:,j))-min(p(:,j)));
    end
    

%% Build the neural network (BP)
e = zeros(1,1);

xt = pp(tv_id,1:4)';
yt = p(tv_id,5)';
xtt = pp(tt_id,1:4)';
ytt = p(tt_id,5)';  
    
net = feedforwardnet(5); % The optimal number of neurons in the single hidden layer is set here.
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.001;
net.trainParam.goal = 0.00001;
net = train(net,xt,yt);
an = sim(net,xtt);
an(an < 0.5) = 0;
an(an >= 0.5) = 1;

yTest = an;  % Simulated value by the artificial neural network
yTestTrue = ytt;  % True value by LBM simulation
rate = sum(yTest == yTestTrue) / length(yTest);  % The rate of accuracy for predicting the event of clogging
e(1,1) = rate
