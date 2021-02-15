
%% image load
net = resnet18;
%% hmmm
%%insert the name/ path of image you want to check activations 
im = imread('test_ho.jpg');
%imshow(im)
imgSize = size(im);
imgSize = imgSize(1:2);
%%Show Activations of First Convolutional Layer
act1 = activations(net,im,'res5b_relu');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 8]);
%imshow(I)

%%Find the Strongest Activation Channel
[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,imgSize);

I = imtile({im,act1chMax});
imshow(I)

%% test
imClosed = imread('test_Dho.jpg');
imshow(imClosed)
act6Closed = activations(net,imClosed,'res5b');
sz = size(act6Closed);
act6Closed = reshape(act6Closed,[sz(1),sz(2),1,sz(3)]);
channelsClosed = repmat(imresize(mat2gray(act6Closed(:,:,:,[14 47])),imgSize),[1 1 3]);
channelsOpen = repmat(imresize(mat2gray(act6relu(:,:,:,[14 47])),imgSize),[1 1 3]);
I = imtile(cat(4,im,channelsOpen*255,imClosed,channelsClosed*255));
imshow(I)
title('Input Image, Channel 14, Channel 47');
