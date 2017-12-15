function imdb_bcnn_train_dag(imdb, opts, varargin)
% Train a bilinear CNN model on a dataset supplied by imdb

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of BCNN and is made available 
% under the terms of the BSD license (see the COPYING file).
%
% This file modified from IMDB_CNN_TRAIN of MatConvNet

opts.lite = false ;
opts.numFetchThreads = 12 ;

opts = vl_argparse(opts, varargin) ;

opts.train.batchSize = opts.batchSize ;
opts.train.numSubBatches = opts.numSubBatches;
opts.train.numEpochs = opts.numEpochs ;
opts.train.continue = true ;
opts.train.gpus = opts.useGpu ;
opts.train.prefetch = false ;
opts.train.learningRate = opts.learningRate ;
opts.train.expDir = opts.expDir ;
opts.train.sync = true ;
opts.train.cudnn = true ;
opts.train.memoryMapFile = opts.memoryMapFile;


opts.inittrain.weightDecay = 0 ;
opts.inittrain.batchSize = 256 ;
opts.inittrain.numSubBatches = 1;
opts.inittrain.numEpochs = 300 ;
opts.inittrain.continue = true ;
opts.inittrain.gpus = opts.useGpu ;


if(isempty(opts.useGpu))
    opts.inittrain.gpus = opts.useGpu ;
else
    opts.inittrain.gpus = opts.useGpu(1) ;
end

opts.inittrain.prefetch = false ;
opts.inittrain.learningRate = 0.001 ;
opts.inittrain.expDir = fullfile(opts.expDir, 'init') ;



% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

encoderOpts = parseEncoder(opts);

switch encoderOpts.type
    case 'bcnn'
        shareWeights = encoderOpts.shareWeight;

        if shareWeights
            % the case with shared weights    
            initNetFn = @initializeNetworkSharedWeights;
        else
            % the case with two streams
            initNetFn = @initializeNetworkTwoStreams;
        end
        
    case {'impbcnn'}
        shareWeights = true;
        initNetFn = @initializeNonlinearMatrixLayer;
end

net = initNetFn(imdb, encoderOpts, opts);
    
if shareWeights 
    bopts = net.meta.normalization ;
else
    bopts = net.meta.meta1.normalization ;
end
bopts.numThreads = opts.numFetchThreads ;
bopts.averageImage = [];

% compute image statistics (mean, RGB covariances etc)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% can use mean average rgb values: rgbMean
% or use different averages rgb values for different pixel: averageImage 
% or use ImageNet pretrained rgbMean
if shareWeights 
    % net.meta.normalization.averageImage = rgbMean ;
    % net.meta.normalization.averageImage = averageImage;
    if size(net.meta.normalization.averageImage, 3) ~= 3
        net.meta.normalization.averageImage = reshape(net.meta.normalization.averageImage, [1, 1, 3]);
    end
else
    % net.meta.meta1.normalization.averageImage = rgbMean ;
    % net.meta.meta2.normalization.averageImage = rgbMean ;
    % net.meta.meta1.normalization.averageImage = averageImage;
    % net.meta.meta2.normalization.averageImage = averageImage;
    if size(net.meta.meta1.normalization.averageImage, 3) ~= 3
        net.meta.meta1.normalization.averageImage = reshape(net.meta.meta1.normalization.averageImage, [1, 1, 3]);
    end
    if size(net.meta.meta2.normalization.averageImage, 3) ~= 3
        net.meta.meta2.normalization.averageImage = reshape(net.meta.meta2.normalization.averageImage, [1, 1, 3]);
    end
end

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

[v,d] = eig(rgbCovariance) ;

% setting
if shareWeights 
    train_bopts = net.meta.normalization;
    train_bopts.numThreads = opts.numFetchThreads ;
    val_bopts = train_bopts;
    
    train_bopts.transformation = opts.dataAugmentation{1} ;
    if opts.rgbJitter
        train_bopts.rgbVariance = 0.1*sqrt(d)*v' ;
    else
        train_bopts.rgbVariance = [];
    end
    train_bopts.scale = opts.imgScale ;
    
    val_bopts.transformation = opts.dataAugmentation{2};
    val_bopts.rgbVariance = [] ;
    val_bopts.scale = opts.imgScale ;
else
    
    train_bopts(1) = net.meta.meta1.normalization;
    train_bopts(2) = net.meta.meta2.normalization;
    train_bopts(1).numThreads = opts.numFetchThreads ;
    train_bopts(2).numThreads = opts.numFetchThreads ;
    val_bopts = train_bopts;
    
    for i=1:numel(train_bopts)
        train_bopts(i).transformation = opts.dataAugmentation{1} ;
        if opts.rgbJitter
            train_bopts(i).rgbVariance = 0.1*sqrt(d)*v' ;
        else
            train_bopts(i).rgbVariance = [];
        end
        train_bopts(i).scale = opts.imgScale ;
        
        val_bopts(i).transformation = opts.dataAugmentation{2};
        val_bopts(i).rgbVariance = [] ;
        val_bopts(i).scale = opts.imgScale ;
    end

end
useGpu = numel(opts.train.gpus) > 0 ;


if(~exist(fullfile(opts.expDir, 'fine-tuned-model'), 'dir'))
    mkdir(fullfile(opts.expDir, 'fine-tuned-model'))
end

fn_train = getBatchDagNNWrapper(train_bopts, useGpu) ;
fn_val = getBatchDagNNWrapper(val_bopts, useGpu) ;
opts.train = rmfield(opts.train, {'sync', 'cudnn'}) ;
[net, info] = bcnn_train_dag(net, imdb, fn_train, fn_val, opts.train, ...
                            'plotStatistics', opts.plotStatistics) ;
net = net_deploy(net) ;
save(fullfile(opts.expDir, 'fine-tuned-model', 'final-model.mat'), ...
    'net', 'info', '-v7.3');

function encoderOpts = parseEncoder(opts)

% get encoder setting
encoderOpts.type = 'bcnn';
encoderOpts.modela = [];
encoderOpts.layera = 14;
encoderOpts.modelb = [];
encoderOpts.layerb = 14;
encoderOpts.shareWeight = false;
encoderOpts.model = [];
encoderOpts.layer = 14;
encoderOpts.sigma = 0.1;
encoderOpts.pow = 0.5;
encoderOpts.method = {};
encoderOpts.bpMethod = {};
encoderOpts.maxIter = 5;

encoderOpts = vl_argparse(encoderOpts, opts.encoders{1}.opts);


% -------------------------------------------------------------------------
function saveNetwork(fileName, net, info)
% -------------------------------------------------------------------------

% 
% % Replace the last layer with softmax
% layers{end}.type = 'softmax';
% layers{end}.name = 'prob';
net.layers(end-1:end) = [];

% Remove fields corresponding to training parameters
ignoreFields = {'learningRate',...
                'weightDecay',...
                'class'};
for i = 1:length(net.layers),
    net.layers{i} = rmfield(net.layers{i}, ignoreFields(isfield(net.layers{i}, ignoreFields)));
end

save(fileName, 'net', 'info', '-v7.3');


% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 1: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  temp = temp{1};
%   temp = cat(4, temp{1}{:});
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{t} = mean(temp, 4) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;



