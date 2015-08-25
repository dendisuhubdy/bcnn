function run_experiments_bcnn_train_vdvd_2()
% Fine-tune the CNN model on the facescrub dataset
%maxNumCompThreads(10);
  bcnnmm.name = 'bcnnmm' ;
  bcnnmm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', 'data/models/imagenet-vgg-m.mat', ...
    'layerb', 14,...
    'shareWeight', true,...
    } ;

  bcnnvdm.name = 'bcnnvdm' ;
  bcnnvdm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/models/imagenet-vgg-m.mat', ...
    'layerb', 14,...
    'shareWeight', false,...
    } ;

  bcnnvdvd.name = 'bcnnvdvd' ;
  bcnnvdvd.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layerb', 30,...
    'shareWeight', true,...
    };

%{
  setupNameList = {'bcnnvdm', 'bcnnmm'};
  encoderList = {{bcnnvdm}, {bcnnmm}}; 
  datasetList = {{'cubcrop', 1} , {'cub', 1}};  
%}
    
  setupNameList = {'bcnnvdvd'};
  encoderList = {{bcnnvdvd}}; 
  datasetList = {{'cub', 1}};  

  for ii = 1 : numel(datasetList)
    dataset = datasetList{ii} ;
    if iscell(dataset)
      numSplits = dataset{2} ;
      dataset = dataset{1} ;
    else
      numSplits = 1 ;
    end
    for jj = 1 : numSplits
      for ee = 1: numel(encoderList)    
          [opts, imdb] = model_setup('dataset', dataset, ...
			  'encoders', encoderList{ee}, ...
			  'prefix', 'bcnn-train-dd-f2', ...
			  'batchSize', 8, ...
              'bcnnScale', 2, ...
              'bcnnLRinit', true, ...
              'dataAugmentation', {'f2','none','none'},...
			  'useGpu', 1);
          imdb_bcnn_train(imdb, opts);
      end
    end
  end
end