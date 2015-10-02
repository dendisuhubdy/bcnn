function run_experiments()

% run svm trainin and testing 

  rcnn.name = 'rcnn' ;
  rcnn.opts = {...
    'type', 'rcnn', ...
    'model', 'data/models/imagenet-vgg-m.mat', ...
    'layer', 19} ;

  rcnnvd.name = 'rcnnvd' ;
  rcnnvd.opts = {...
    'type', 'rcnn', ...
    'model', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layer', 35} ;

  dcnn.name = 'dcnn' ;
  dcnn.opts = {...
    'type', 'dcnn', ...
    'model', 'data/models/imagenet-vgg-m.mat', ...
    'layer', 14, ...
    'numWords', 64} ;

  dcnnvd.name = 'dcnnvd' ;
  dcnnvd.opts = {...
    'type', 'dcnn', ...
    'model', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layer', 30, ...
    'numWords', 64} ;

  dsift.name = 'dsift' ;
  dsift.opts = {...
    'type', 'dsift', ...
    'numWords', 256, ...
    'numPcaDimensions', 80} ;

  bcnnmm.name = 'bcnnmm' ;
  bcnnmm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/ft-models/bcnn-cub-mm.mat', ...
    'layera', 14,...
    'modelb', 'data/ft-models/bcnn-cub-mm.mat', ...
    'layerb', 14
    } ;

  bcnnvdm.name = 'bcnnvdm' ;
  bcnnvdm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/ft-models/bcnn-cub-dm-neta.mat', ...
    'layera', 30,...
    'modelb', 'data/ft-models/bcnn-cub-dm-netb.mat', ...
    'layerb', 14
    } ;

  bcnnvdvd.name = 'bcnnvdvd' ;
  bcnnvdvd.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/ft-models/bcnn-cub-dd.mat', ...
    'layera', 30,...
    'modelb', 'data/ft-models/bcnn-cub-dd.mat', ...
    'layerb', 30,...
    };
  

  setupNameList = {'rcnn', 'dcnn', 'bcnnmm'};   % list of diffenet models to train and test
  encoderList = {{rcnn}, {dcnn}, {bcnnmm}};
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
        model_train(...
          'dataset', dataset, ...
          'seed', jj, ...
          'encoders', encoderList{ee}, ...
          'prefix', 'exp', ...
          'suffix', setupNameList{ee}, ...
          'printDatasetInfo', ee == 1, ...
          'useGpu', 1, ...
          'dataAugmentation', 'f2') ;       %flipping for data augmentation. "none" for no augmentation
      end
    end
  end
end
