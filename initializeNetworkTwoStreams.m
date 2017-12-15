function net = initializeNetworkTwoStreams(imdb, encoderOpts, opts)

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% This code is used for initializing asymmetric B-CNN network 

% -------------------------------------------------------------------------
scal = 1 ;
init_bias = 0.1;
numClass = length(imdb.classes.name);

assert(~isempty(encoderOpts.modela) && ~isempty(encoderOpts.modelb), 'Error: at least one of the network is not specified')

% load the pre-trained models
neta = load(encoderOpts.modela);
netb = load(encoderOpts.modelb);

% load the base network A
if isfield(neta, 'params')
    neta = dagnn.DagNN.loadobj(neta);
end
if ~isa(neta, 'dagnn.DagNN')
    neta = dagnn.DagNN.fromSimpleNN(neta, 'canonicalNames', true);
end

% load the base network B
if isfield(netb, 'params')
    netb = dagnn.DagNN.loadobj(netb);
end
if ~isa(netb, 'dagnn.DagNN')
    netb = dagnn.DagNN.fromSimpleNN(netb, 'canonicalNames', true);
end

% truncate the neta and get the feature dimension
o1_name = neta.layers(encoderOpts.layera).outputs{1};
executeOrder = neta.getLayerExecutionOrder();
maxIndex = find(executeOrder == encoderOpts.layera);
removeIdx = executeOrder(maxIndex+1:end);
removeName = {neta.layers(removeIdx).name};
neta.removeLayer(removeName);
inputName = neta.getInputs();
varSize = neta.getVarSizes({inputName, [neta.meta.normalization.imageSize, 1]});
varIndex1 = neta.getVarIndex(o1_name);
mapSize1 = varSize{varIndex1}(3);

% rename the input to the canonical name
inputName = neta.getInputs();
if ~strcmp(inputName, 'input')
    neta.renameVar(inputName, 'input');
end

% truncate the netb and get the feature dimension
o2_name = netb.layers(encoderOpts.layerb).outputs{1};
executeOrder = netb.getLayerExecutionOrder();
maxIndex = find(executeOrder == encoderOpts.layerb);
removeIdx = executeOrder(maxIndex+1:end);                                                                               removeName = {netb.layers(removeIdx).name};
netb.removeLayer(removeName);
inputName = netb.getInputs();
varSize = netb.getVarSizes({inputName, [netb.meta.normalization.imageSize, 1]});
varIndex2 = netb.getVarIndex(o2_name);
mapSize2 = varSize{varIndex2}(3);

% rename the input to the canonical name
inputName = netb.getInputs();
if ~strcmp(inputName, 'input')
    netb.renameVar(inputName, 'input');
end


% create the network
net = dagnn.DagNN.loadobj(neta);
meta.meta1 = neta.meta;
meta.meta2 = netb.meta;

net.meta = meta;

net.meta.meta1.normalization.keepAspect = opts.keepAspect;
net.meta.meta2.normalization.keepAspect = opts.keepAspect;
net.meta.meta1.normalization.border = opts.border;
net.meta.meta2.normalization.border = opts.border;

for i=1:numel(netb.layers)
    layerName = strcat('netb_', netb.layers(i).name);
    input = strcat('netb_', netb.layers(i).inputs);
    output = strcat('netb_', netb.layers(i).outputs);
    params = strcat('netb_', netb.layers(i).params);
    net.addLayer(layerName, netb.layers(i).block, input, output, params);

    for f = 1:numel(params)
        varId = net.getParamIndex(params{f});
        varIdb = netb.getParamIndex(netb.layers(i).params{f});
        net.params(varId).value = netb.params(varIdb).value;
    end
end

% add bilinear pool layer
input = {o1_name, strcat('netb_', o2_name)};
myBlock = BilinearPooling('normalizeGradients', false);
dim = mapSize1 * mapSize2;
layerName = 'bilr_1';
output = 'b_1';
net.addLayer(layerName, myBlock, input, output);

% Square-root layer
layerName = sprintf('sqrt_1');
input = output;
output = 's_1';
net.addLayer(layerName, SquareRoot(), {input}, output);


% L2 normalization layer
layerName = 'l2_1';
input = output;
bpoutput = 'l_1';
net.addLayer(layerName, L2Norm(), {input}, bpoutput);

% build a linear classifier netc
initialW = 0.001/scal *randn(1,1,dim, numClass,'single');
initialBias = init_bias.*ones(1, numClass, 'single');
netc.layers = {};
netc.layers{end+1} = struct('type', 'conv', 'name', 'classifier', ...
    'weights', {{initialW, initialBias}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1000 1000], ...
    'weightDecay', [0 0]) ;
netc.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
netc = vl_simplenn_tidy(netc) ;


% pretrain the linear classifier with logistic regression
if(opts.bcnnLRinit && ~opts.fromScratch)
    if exist(fullfile(opts.expDir, 'initial_fc.mat'))
        load(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
        
    else
        train = find(ismember(imdb.images.set, [1 2]));
        % compute and cache the bilinear cnn features
        if ~exist(opts.nonftbcnnDir)
            mkdir(opts.nonftbcnnDir)
            
            batchSize = 64;
            
            bopts(1) = net.meta.meta1.normalization;
            bopts(2) = net.meta.meta2.normalization;
            bopts(1).numThreads = opts.numFetchThreads ;
            bopts(2).numThreads = opts.numFetchThreads ;
            
            for i=1:numel(bopts)
                bopts(i).transformation = 'none' ;
                bopts(i).rgbVariance = [];
                bopts(i).scale = opts.imgScale ;
            end
            
            useGpu = numel(opts.train.gpus) > 0 ;
            if useGpu
                gpuDevice(opts.train.gpus(1))
                net.move('gpu') ;
            end
            
            getBatchFn = getBatchDagNNWrapper(bopts, useGpu) ;
            
            for t=1:batchSize:numel(train)
                fprintf('Initialization: extracting bcnn feature of batch %d/%d\n', ceil(t/batchSize), ceil(numel(train)/batchSize));
                batch = train(t:min(numel(train), t+batchSize-1));
                input = getBatchFn(imdb, batch) ;
                if opts.train.prefetch
                    nextBatch = train(t+batchSize:min(t+2*batchSize-1, numel(train))) ;
                    getBatchFn(imdb, nextBatch) ;
                end
                
                input = input(1:4);
                net.mode = 'test' ;
                net.eval(input);
                fIdx = net.getVarIndex('l_1');
                code_b = net.vars(fIdx).value;
                code_b = squeeze(gather(code_b));
                
                for i=1:numel(batch)
                    code = code_b(:,i);
                    savefast(fullfile(opts.nonftbcnnDir, ['bcnn_nonft_', num2str(batch(i), '%05d')]), 'code');
                end
            end
            
            % move back to cpu
            if useGpu
                net.move('cpu') ;
            end
        end
        
        
        bcnndb = imdb;
        tempStr = sprintf('%05d\t', train);
        tempStr = textscan(tempStr, '%s', 'delimiter', '\t');
        bcnndb.images.name = strcat('bcnn_nonft_', tempStr{1}');
        bcnndb.images.id = bcnndb.images.id(train);
        bcnndb.images.label = bcnndb.images.label(train);
        bcnndb.images.set = bcnndb.images.set(train);
        bcnndb.imageDir = opts.nonftbcnnDir;
        
        %train logistic regression
        [netc, info] = cnn_train(netc, bcnndb, @getBatch_bcnn_fromdisk, opts.inittrain, ...
            'conserveMemory', true);
        
        if opts.inittrain.gpus
            vl_simplenn_move(netc, 'cpu') ;
        end
        save(fullfile(opts.expDir, 'initial_fc.mat'), 'netc', '-v7.3') ;
    end
end


% Initial the classifier to the pretrained weights
layerName = 'classifier';
param(1).name = 'convclass_f';
param(1).value = netc.layers{1}.weights{1};
param(2).name = 'convattr_b';
param(2).value = netc.layers{1}.weights{2};
net.addLayer(layerName, dagnn.Conv(), {bpoutput}, 'score', {param(1).name param(2).name});
for f = 1:2,
    varId = net.getParamIndex(param(f).name);
    net.params(varId).value = param(f).value;
    net.params(varId).learningRate = 1000;
    net.params(varId).weightDecay = 0;
end

% add loss functions
net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'score','label'}, 'objective');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'score','label'}, 'top1error');
net.addLayer('top5e', dagnn.Loss('loss', 'topkerror'), {'score','label'}, 'top5error');
clear netc

net.mode = 'normal';

if(opts.fromScratch)
    for i=1:numel(net.layers)
        %if isempty(net.layers{i}.paramIndexes), continue ; end
        if ~isa(net.layers(i).block, 'dagnn.Conv'), continue ; end
        for j=1:numel(net.layers(i).params)
            paramIdx = net.getParamIndex(net.layers(i).params{j});
            if ndims(net.params(paramIdx).value) == 2
                net.params(paramIdx).value = init_bias*ones(size(net.params(paramIdx).value), 'single');
            else
                net.params(paramIdx).value = 0.01/scal * randn(net.params(paramIdx).value, 'single');
            end
        end
    end
end


% Rename classes
net.meta.meta1.classes.name = imdb.classes.name;
net.meta.meta1.classes.description = imdb.classes.name;
net.meta.meta2.classes.name = imdb.classes.name;
net.meta.meta2.classes.description = imdb.classes.name;

% setup the border for translation data jittering
if(~strcmp(opts.dataAugmentation{1}, 'f1') && ~strcmp(opts.dataAugmentation{1}, 'none'))
    net.meta.meta1.normalization.border = 256 - net.meta.meta1.normalization.imageSize(1:2) ;
    net.meta.meta2.normalization.border = 256 - net.meta.meta2.normalization.imageSize(1:2) ;
end

function [im,labels] = getBatch_bcnn_fromdisk(imdb, batch)
% -------------------------------------------------------------------------

im = cell(1, numel(batch));
for i=1:numel(batch)
    load(fullfile(imdb.imageDir, imdb.images.name{batch(i)}));
    im{i} = code;
end
im = cat(2, im{:});
im = reshape(im, 1, 1, size(im,1), size(im, 2));
labels = imdb.images.label(batch) ;




function layers = simpleFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = find(cellfun(@(x)strcmp(x.type, type), net.layers)) ;

% -------------------------------------------------------------------------
function net = simpleRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = simpleFindLayersOfType(net, type) ;
net.layers(layers) = [] ;
