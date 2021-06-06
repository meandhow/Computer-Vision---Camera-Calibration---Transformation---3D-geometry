I1 = imread('IMG_2511.jpeg');
I2 = imread('IMG_2513.jpeg');

I1 = imrotate(I1, 270);
I2 = imrotate(I2, 270);

I1 = rgb2gray(I1);
I2 = rgb2gray(I2);

points1 = detectSURFFeatures(I1, 'MetricThreshold', 5000);
points2 = detectSURFFeatures(I2, 'MetricThreshold', 5000);

[features1,validPoints1] = extractFeatures(I1,points1);
[features2,validPoints2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(features1,features2,'Unique', true, 'MatchThreshold', 0.02) ;
%indexPairs = matchFeatures(features1,features2,'Unique', true, 'MaxRatio', 0.5) ;

matchedPoints1 = validPoints1(indexPairs(:,1));
matchedPoints2 = validPoints2(indexPairs(:,2));

figure; 
showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
legend('matched points 1','matched points 2');

%I = insertMarker(I, points, 'Size', 15);
%imshow(I); hold on;
%plot(points.selectStrongest(1000));




