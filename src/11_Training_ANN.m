clear
close all
clc

%% Import data
data = readmatrix('MATLAB_fav_CN.csv'); % Here you can change the name of dataset for four output parameters.
% Favorable condition: fav_CN.csv: Coordination number; fav_HC.csv: Hydraulic conductivity; fav_SC.csv: Surface coverage; fav_VF.csv: Void fraction
% Unfavorable condition: unfav_CN.csv: Coordination number; unfav_HC.csv: Hydraulic conductivity; unfav_SC.csv: Surface coverage; unfav_VF.csv: Void fraction
p = data(:,1:5);
l = length(data);
rng(70);  % The seed of random splitting
id = randperm(l);
tt_id = id(1:round(l*(10/100)));  % Test dataset: 10 percent of the whole dataset
tv_id = id(((round(l*(10/100)))+1):end);  % Training and validation dataset: 90 percent of the whole dataset
tt = p(tt_id,:);   % Test dataset
q = p(tv_id,:);   % Training and validation dataset

%% Normalization of data
    for j = 1:4
    qq(:,j) = (q(:,j)-min(q(:,j)))/(max(q(:,j))-min(q(:,j)));
    end
    
a = qq(:,1);  % Input parameter: Ionic strength
b = qq(:,2);  % Input parameter: Flow velocity
c = qq(:,3);  % Input parameter: Zeta potential
d = qq(:,4);  % Input parameter: Particle size
t = q(:,5);  % Output parameter: Coordination number, Hydraulic conductivity, Surface coverage, Void fraction. Depends on the input document.

%% Cross validation in the neural network (BP)
e = zeros(1,3);
% Indices for choosing training dataset and validation dataset (Favorable condition: In the MATLAB_fav_Indices.csv; Unfavorable condition: In the MATLAB_unfav_Indices.csv)
indices = [
6
10
8
1
8
4
3
9
4
1
9
6
8
1
10
10
4
8
7
3
10
2
5
7
6
3
6
3
1
2
5
2
9
2
4
5
7
2
5
9
5
8
4
1
6
7
7
3
9
] 

% The k and i are adjusted manually
k = 2   % The number of neurons in the single hidden layer
i = 1   % The number of indices for validation dataset

    att = a(indices == i);
    x = find(indices == i);
    a(x) = [];
    at = a;

    btt = b(indices == i);
    b(x) = [];
    bt = b;
   
    ctt = c(indices == i);
    c(x) = [];
    ct = c;
    
    dtt = d(indices == i);
    d(x) = [];
    dt = d;
    
    xtt = [att btt ctt dtt]';
    xt = [at bt ct dt]';
    ytt = t(indices == i)';  
    t(x) = [];
    yt = t';

    net = feedforwardnet(k);
    net.trainParam.epochs = 1000;
    net.trainParam.lr = 0.001;
    net.trainParam.goal = 0.00001;
    net = train(net,xt,yt);
    an = sim(net,xtt);
    yTest = an;  % Simulated value by the artificial neural network
    yTestTrue = ytt;  % True value by LBM simulation
    mse = (sum((yTest - yTestTrue).^2)) / (length(ytt));  % MSE for the test dataset
    mae = (sum(abs(yTest - yTestTrue))) / (length(ytt));  % MAE for the test dateset
    rt = 1 - (sum((yTestTrue - yTest).^2))/(sum((yTestTrue - mean(yTestTrue)).^2));  % R2 for the test dataset
    
    e(1,1) = mse;
    e(1,2) = mae;
    e(1,3) = rt;
    
    