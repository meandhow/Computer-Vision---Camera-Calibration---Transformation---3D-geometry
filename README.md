# Computer-Vision---Camera-Calibration---Transformation---3D-geometry

This project represents a full computer vision pipiline from acquisition, thorugh analysis, to manipulation. Crated as submission for Imperial College ELEC97112/ELEC97113 Computer Vision and Pattern Recognition module


Task 1. [10 marks] Collect Data
Build a callibration grid e.g. from a cardbox.
Choose a small textured object, e.g. toy, sculpture, tool.
Collect a sequence of 5-10 pictures (we call it FD) with and without the object in the grid. Change the camera position between pictures.
Collect a similar sequence (we call it HG) by changing the zoom (e.g. factor 1.5) and slightly rotating the camera (e.g. 10-20 degree) but keeping exactly the same location of the camera.
Show the whole data in the appendix.

Task 2. [10 marks ] Keypoint orrespondences between images
1) Compare quality/quantity of correspondences found by two methods
a) Manual (clicking on corresponding points)
b) Automatic (detecting keypoint and matching descriptors)

Task 3. [10 marks] Camera calibration
1) Find and report camera parameters.
2) Can you estimate or illustrate distortions of your camera?

Task 4. [10 marks] Transformation estimation
1) Estimate a homography matrix between a pair of images from HG.
a) Show the keypoints and their correspondences projected from the other image. 2) Estimate fundamental matrix between a pair of images from FD.
a) Show the keypoints and their corresponding epipolar lines in the other image. b) Show epipoles, vanishing points and horizon in your images.

Task 5. [10 marks] 3D geometry
1) Show a stereo rectified pair of your images with epipolar lines.
2) Calculate and display depth map of your object estimated from different views.
