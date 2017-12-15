function net = initializeNonlinearMatrixLayer(imdb, encoderOpts, opts)
% -------------------------------------------------------------------------

scal = 1 ;
init_bias = 0.1;
numClass = length(imdb.classes.name);

% Load the model
net = load(encoderOpts.model);
net.meta.normalization.keepAspect = opts.keepAspect;
net.meta.normalization.border = opts.border;

% check if the network is a dagnn
if isfield(net, 'params')
    net = dagnn.DagNN.loadobj(net);
end
isDag = isa(net, 'dagnn.DagNN');

% truncate the network
maxLayer = encoderOpts.layer;

if isDag
    o1_name = net.layers(encoderOpts.layer).outputs{1};
    
    executeOrder = net.getLayerExecutionOrder();
    maxIndex = find(executeOrder == maxLayer);
    removeIdx = executeOrder(maxIndex+1:end);
    removeName = {net.layers(removeIdx).name};
    net.removeLayer(removeName);
    
    inputName = net.getInputs();
    varSize = net.getVarSizes({inputName, [net.meta.normalization.imageSize, 1]});
    
    varIndex1 = net.getVarIndex(o1_name);
    
    mapSize1 = varSize{varIndex1}(3);
    mapSize2 = mapSize1;    
    
    inputName = net.getInputs();
    if ~strcmp(inputName, 'input')
        net.renameVar(inputName, 'input');
    end
else
    net.layers = net.layers(1:maxLayer);
    % get the feature dimension for both layers
    netInfo = vl_simplenn_display(net);
    mapSize1 = netInfo.dataSize(3, encoderOpts.layer+1);
    mapSize2 = mapSize1;
    
    
    % network setting
    net = vl_simplenn_tidy(net) ;
    for l=numel(net.layers):-1:1
        if strcmp(net.layers{l}.type, 'conv')
            net.layers{l}.opts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
        end
    end
    
    l1_name = net.layers{encoderOpts.layer}.name;
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
    o1_name = net.layers(net.getLayerIndex(l1_name)).outputs{1};
end

inputNames = {o1_name};

paramNames = {};
myBlock = SqrtmPooling('normalizeGradients', false, ...
                       'pow', encoderOpts.pow, ...
                       'sigma', encoderOpts.sigma, ...
                       'method', encoderOpts.method, ...
                       'bpMethod', encoderOpts.bpMethod, ...
                       'maxIter', encoderOpts.maxIter);
dim = mapSize1 * mapSize2;
output = {'b_1', 'svd_u', 'svd_d'};


% Nonlinear matrix operation
net.addLayer('bilr_1', myBlock, inputNames, output, paramNames);

% Square-root layer by silence layer
layerName = sprintf('sqrt_1');
input = output;
output = 's_1';
net.addLayer(layerName, SilenceWrapper('blockType', 'PowerNorm', 'fanIn', 1, 'params', {'pow', 0.5}), input, output);


% L2 normalization layer
layerName = 'l2_1';
input = output;
bpoutput = 'l_1';
net.addLayer(layerName, L2Norm(), {input}, bpoutput);
% ------------------------------------------------------------------------------------


% build a linear classifier netc
initialW = 0.001/scal *randn(1,1,dim, numClass,'single');
initialBias = init_bias.*ones(1, numClass, 'single');
netc.layers = {};
netc.layers{end+1} = struct('type', 'conv', 'name', 'classifier', ...
    'weights', {{initialW, initialBias}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1000 1000], ...
    'weightDecay', [1 1]) ;
netc.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
netc = vl_simplenn_tidy(netc) ;



% pretrain the linear classifier with logistic regression
if(opts.bcnnLRinit && ~opts.fromScratch)
    
    % get bcnn feature for train and val sets
    train = find(imdb.images.set==1|imdb.images.set==2);
    if ~exist(opts.nonftbcnnDir, 'dir')
        mkdir(opts.nonftbcnnDir)
        
        batchSize = 64;
        
        bopts = net.meta.normalization ;
        bopts.numThreads = opts.numFetchThreads ;
        bopts.transformation = 'none' ;
        bopts.rgbVariance = [] ;
        bopts.scale = opts.imgScale;
        
        
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
            
            input = input(1:end-2);
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
            net.move('cpu');
        end
    end
    
    clear code
    
    % get the pretrain linear classifier
    if exist(fullfile(opts.expDir, 'initial_fc.mat'), 'file')
        load(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
    else
        
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
            'conserveMemory', true, 'plotStatistics', opts.plotStatistics);
        
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

% set all parameters to random number if train the model from scratch
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

clear netc

% Rename classes
net.meta.classes.name = imdb.classes.name;
net.meta.classes.description = imdb.classes.name;

% add border for translation data jittering
if(~strcmp(opts.dataAugmentation{1}, 'f1') && ~strcmp(opts.dataAugmentation{1}, 'none'))
    net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
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

