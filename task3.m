% Auto-generated by cameraCalibrator app on 16-Feb-2021
%-------------------------------------------------------

clear all;
% Define images to process
imageFileNames = {'/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2551.jpeg',...
    '/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2560.jpeg',...
    '/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2561.jpeg',...
    '/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2562.jpeg',...
    '/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2563.jpeg',...
    '/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2564.jpeg',...
    '/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2566.jpeg',...
    '/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2567.jpeg',...
    '/Users/meow/Documents/Imperial/Computer Vision/CW1/calibration_photos/IMG_2565.jpeg',...
    };
% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
squareSize = 25;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')
