%% Plot ROC 
cgt = double(YpredTest);
cscores = probs;
figure(1)
[X,Y, AUC, OPTROCPT,SUBY,SUBYNAMES] = perfcurve(cgt, cscores(:,1),1);
plot(X,Y);
grid
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for classification CNN')