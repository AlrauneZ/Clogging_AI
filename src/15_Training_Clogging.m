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
t = q(:,5);  % Output parameter: The event of clogging (1 means clogging while 0 means no clogging)

%% Cross validation in the neural network (BP)
e = zeros(1,1);
% Indices for choosing training dataset and validation dataset (Favorable condition: In the MATLAB_fav_Indices.csv; Unfavorable condition: In the MATLAB_unfav_Indices.csv)
indices = [
8
6
3
9
4
6
7
1
1
3
3
1
9
2
6
6
4
5
7
3
2
5
8
1
9
5
8
2
5
2
6
4
4
10
8
4
4
7
9
7
3
1
7
2
7
10
9
10
1
6
10
5
8
8
10
6
10
8
4
1
2
3
9
7
5
10
7
5
9
3
2
5
1
]

    k = 2;% The number of neurons in the single hidden layer
    i = 1; % The number of indices for validation dataset
	
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
    an(an < 0.5) = 0;
    an(an >= 0.5) = 1;

    yTest = an;  % Simulated value by the artificial neural network
    yTestTrue = ytt;  % True value by LBM simulation
    rate = sum(yTest == yTestTrue) / length(yTest);  % The rate of accuracy for predicting the event of clogging
    e(1,1) = rate

 