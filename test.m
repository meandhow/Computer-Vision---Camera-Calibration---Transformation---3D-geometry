clear all;
I1 = imread('rectified_fig.jpeg');
% I2 = imread('h2.jpeg');
% I3 = imread('HG3.jpeg');
figure
h1=imshow(I1);
hp1 = impixelinfo(h1);
readPoints(I1, 4)
function pts = readPoints(image, n)
%readPoints   Read manually-defined points from image
%   POINTS = READPOINTS(IMAGE) displays the image in the current figure,
%   then records the position of each click of button 1 of the mouse in the
%   figure, and stops when another button is clicked. The track of points
%   is drawn as it goes along. The result is a 2 x NPOINTS matrix; each
%   column is [X; Y] for one point.
% 
%   POINTS = READPOINTS(IMAGE, N) reads up to N points only.
if nargin < 2
    n = Inf;
    pts = zeros(2, 0);
else
    pts = zeros(2, n);
end
imshow(image);     % display image
xold = 0;
yold = 0;
k = 0;
hold on;           % and keep it there while we plot
while 1
    [xi, yi, but] = ginput(1);      % get a point
    if ~isequal(but, 1)             % stop if not button 1
        break
    end
    k = k + 1;
    pts(1,k) = xi;
    pts(2,k) = yi;
      if xold
          plot([xold xi], [yold yi], 'go-');  % draw as we go
      else
          plot(xi, yi, 'go');         % first point on its own
      end
      if isequal(k, n)
          break
      end
      xold = xi;
      yold = yi;
  end
hold off;
if k < size(pts,2)
    pts = pts(:, 1:k);
end
end

% figure
% h2=imshow(I2);
% hp2 = impixelinfo(h2);
% figure
% h3=imshow(I3);
% hp3 = impixelinfo(h3);
% imageFileNames = {'/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2551.jpeg',
%     '/Users/meow/Documents/Imperial/Computer Vision/CW1/HG1.jpeg',
%     '/Users/meow/Documents/Imperial/Computer Vision/CW1/HG2.jpeg',
%     '/Users/meow/Documents/Imperial/Computer Vision/CW1/HG3.jpeg'};
% % Detect checkerboards in images
% [imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
% imageFileNames = imageFileNames(imagesUsed);