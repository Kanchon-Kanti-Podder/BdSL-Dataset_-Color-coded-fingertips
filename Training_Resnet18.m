%%Load Data
unzip('MerchData.zip');
imds = imageDatastore('MerchData', 'IncludeSubfolders', true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%%Load pretrained Network
net = resnet18;

%%Replace Final Layers
numClasses = numel(categories(imdsTrain.Labels));
Igraph = layerGraph(net);

newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc', 'WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
Igraph = replaceLayer(Igraph,'fc1000',newFCLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
Igraph = replaceLayer(Igraph, 'ClassificationLayer_predictions', newClassLayer);

%% Train Network 
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm',...
    'MiniBatchSize',10,...
    'MaxEpochs',30,...
    'InitialLearnRate',0.1, ...
    'Shuffle', 'every-epoch',...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
trainedNet = trainNetwork(augimdsTrain,Igraph,options);