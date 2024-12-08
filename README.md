This code utilizes SIFT to detect keypoints, then uses a matching function to check each detected keypoint 
and determine if they are valid matches between the two images. Once valid matches are identified, it extracts 
the keypoint coordinates from both images and uses them to estimate an affine transformation matrix. 
This matrix captures how the reference image has been transformed.

Affine Transform Info:
https://people.computing.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf (details on how it works)
https://amroamroamro.github.io/mexopencv/matlab/cv.estimateAffine2D.html (cv2 info)

Transform Info:
https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html