% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = imdb_get_batch_bcnn(images, opts, ...
                            'prefetch', nargout == 0);
labels = imdb.images.label(batch) ;
numAugments = size(im{1},4)/numel(batch);

labels = reshape(repmat(labels, numAugments, 1), 1, size(im{1},4));

if nargout > 0
  if useGpu
      for i=1:numel(im)
          im{i} = gpuArray(im{i}) ;
      end
  end
  if numel(im) == 1
      inputs = {'input', im{1}} ;
  else
      inputs = {'input', im{1}, 'netb_input', im{2}} ;
  end

  inputs{end+1} = 'label';
  inputs{end+1} = labels;
end

