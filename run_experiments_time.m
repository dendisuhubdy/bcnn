function run_experiments_time()

  rcnn.name = 'rcnn' ;
  rcnn.opts = {...
    'type', 'rcnn', ...
    'model', 'data/models/imagenet-vgg-m.mat', ...
    'layer', 19} ;

  rcnnvd.name = 'rcnnvd' ;
  rcnnvd.opts = {...
    'type', 'rcnn', ...
    'model', 'data/models/imagenet-vgg-verydeep-19.mat', ...
    'layer', 41} ;

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
    'modela', 'data/bcnn-train_mm/cub-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', 'data/bcnn-train_mm/cub-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 14,...
    } ;

  bcnnmmpca.name = 'bcnnmmpca' ;
  bcnnmmpca.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_mm_one_pca_64_norelu/cub-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', 'data/bcnn-train_mm_one_pca_64_norelu/cub-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 15,...
    } ;
    
  bcnnvdmpca.name = 'bcnnvdmpca' ;
  bcnnvdmpca.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_vdm_one_pca_64_norelu/cub-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/bcnn-train_vdm_one_pca_64_norelu/cub-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 15,...
    } ;

  bcnnvdm.name = 'bcnnvdm' ;
  bcnnvdm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_vdm/cub-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/bcnn-train_vdm/cub-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 14,...
    } ;

  bcnnvdvd.name = 'bcnnvdvd' ;
  bcnnvdvd.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layerb', 30,...
    };
 
  dataPath = 'bcnn-train_mm';
  setupNameList = {'bcnnmm'};
  encoderList = {{bcnnmm}};    
  datasetList = {{'cub', 1} };
  N = 100;

  neta = load(encoderList{1}{1}.opts{4});
  netb = load(encoderList{1}{1}.opts{8});
  neta.useGpu = true;
  netb.useGpu = true;
  
  imdb = load(fullfile('data', dataPath, [datasetList{1}{1}, '-seed-', num2str(datasetList{1}{2}, '%02d')], 'imdb', 'imdb-seed-1'));
  
  info = vl_simplenn_display(neta) ;
  borderA = round(info.receptiveField(end)/2+1) ;
  averageColourA = mean(mean(neta.normalization.averageImage,1),2) ;
  imageSizeA = neta.normalization.imageSize;
 
  im = cell(N,1);
  for i=1:N
      i
    im{i} = imread(fullfile(imdb.imageDir, imdb.images.name{i}));
    im{i} = imresize(single(im{i}), imageSizeA([2 1]), 'bilinear');
%     im{i} = imresize(single(im{i}), 2);
%     crop_h = size(im_cropped,1) ;
%     crop_w = size(im_cropped,2) ;
  end
  
  a = tic;
  get_bcnn_features_noresize(neta, netb, im, 'normalization', 'sqrt');
  t = toc(a)
  N/t
  
  %{
  
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
          'prefix', 'bcnn-train-fine-tuned_vdm_check', ...
          'suffix', setupNameList{ee}, ...
          'printDatasetInfo', ee == 1, ...
          'useGpu', true) ;
      end
    end
  end
end

  %}