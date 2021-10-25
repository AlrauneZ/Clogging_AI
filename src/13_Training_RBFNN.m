clear
close all
clc

%% Import data
data = readmatrix('MATLAB_fav_CN.csv');
% Favorable condition: fav_CN.csv: Coordination number; fav_HC.csv: Hydraulic conductivity; fav_SC.csv: Surface coverage; fav_VF.csv: Void fraction
% Unfavorable condition: unfav_CN.csv: Coordination number; unfav_HC.csv: Hydraulic conductivity; unfav_SC.csv: Surface coverage; unfav_VF.csv: Void fraction
p = data(:,1:5);
l = length(data);
rng(70);  % Seed of the random splitting
id = randperm(l);
tt_id = id(1:round(l*(10/100)));
tv_id = id(((round(l*(10/100)))+1):end);
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
t = q(:,5);  % Output parameter: Coordination number
%% Cross validation in the neural network (BP)
e = zeros(10,3);
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

    for i = 1:10
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
    
    k = 15 /10   % The value of hyperparameter spread
    
    net = newrbe(xt,yt,k);
    an = sim(net,xtt);
    yTest = an;  % Simulated value by the radial basis function neural network
    yTestTrue = ytt;  % True value by LBM simulation
    mse = (sum((yTest - yTestTrue).^2)) / (length(ytt));  % MSE for the test dataset
    mae = (sum(abs(yTest - yTestTrue))) / (length(ytt));  % MAE for the test dataset
    rt = 1 - (sum((yTestTrue - yTest).^2))/(sum((yTestTrue - mean(yTestTrue)).^2));  % R2 for the test dataset

    e(i,1) = mse;
    e(i,2) = mae;
    e(i,3) = rt;
        
    a = qq(:,1);  % Input parameter: Ionic strength
    b = qq(:,2);  % Input parameter: Flow velocity
    c = qq(:,3);  % Input parameter: Zeta potential
    d = qq(:,4);  % Input parameter: Particle size
    t = q(:,5);  % Output parameter: Coordination number
       
    end


    
