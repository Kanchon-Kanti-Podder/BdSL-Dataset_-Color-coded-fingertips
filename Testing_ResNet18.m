%% random data test 

figure
for i = 1:1:100
    a= randperm(4026);
    idx = [a];
    I = readimage(imdsTrain,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label));
    pause(8);
end

%% test data
test = imageDatastore('test',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%% test accuracy
testValidation = augmentedImageSource(inputSize(1:2),test);

%% ypred test
[YPredTest,probs]= classify(trainedNet,testValidation);

%% accu test
accuracyTest = mean(YPredTest == test.Labels);
display(accuracyTest)


%% Plot the confusion matrix.
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(test.Labels,YPredTest);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
