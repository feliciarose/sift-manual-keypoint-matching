import cv2
import numpy as np
import sys
#from google.colab.patches import cv2_imshow

def main(): 

    #Read reference image
    #reference_image_path = './reference.png'
    #image1 = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE) / 255.0

    image1 = cv2.imread('/Users/feliciadrysdale/Desktop/Project2_CV/reference.png', cv2.IMREAD_GRAYSCALE) / 255.0

    #Read from standard input
    #image2_image_path = sys.stdin.readline().strip()
    #image2 = cv2.imread(image2_image_path, cv2.IMREAD_GRAYSCALE) / 255.0

    image2 = cv2.imread('/Users/feliciadrysdale/Desktop/Project2_CV/4.png', cv2.IMREAD_GRAYSCALE) / 255.0

    #Resize reference image
    scale_percent = 70
    width = int(image1.shape[1] * scale_percent / 100)
    height = int(image1.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(image1, (width, height), interpolation=cv2.INTER_AREA)

    #Preprocess reference image to detect fewer keypoints to handle (darken)
    image1 = cv2.GaussianBlur(resized_img, (15, 15), 0)
    darkening_factor = 0.37
    image1 = image1 * darkening_factor
    image1 = (image1 * 255).astype(np.uint8) 

    #Sift for keypoint detection from cv2
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)

    #draw keypoints & show (collab)
    #img1_vis = cv2.drawKeypoints(image1_gray, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2_imshow(img1_vis)

    #Preprocess reference image to detect fewer keypoints to handle
    #No resize of image cuz it doesn't make a huge difference
    blurred_img = cv2.GaussianBlur(image2, (5, 5), 1.5)
    darkening_factor = 0.7
    image2 = blurred_img * darkening_factor
    image2 = (image2 * 255).astype(np.uint8)

    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    #call
    matches_1, matches_2 = check_matches(descriptors1, descriptors2)
    center_x, center_y, height, angle = compute_values(matches_1, matches_2, keypoints1, keypoints2, image1)      
            
    print(f"{int(center_x)} {int(center_y)} {int(height)} {int(angle)}") 

def check_matches(descriptors1, descriptors2):

    descriptors1 = np.array(descriptors1)
    descriptors2 = np.array(descriptors2)

    #Initilize list for matches
    matches_1 = []

    #Iterate over reference image descriptor & calculate distance
    for i, descriptor1 in enumerate(descriptors1):

        #normalize
        differences = descriptors2 - descriptor1 
        squared_differences = differences ** 2    
        sum_squared = np.sum(squared_differences, axis=1) 
        distances = np.sqrt(sum_squared)

        min_distance_index = np.argmin(distances)
        matches_1.append((i, min_distance_index, distances[min_distance_index]))

    #Reverse check matches & calculate distance
    matches_2 = []
    for j, descriptor2 in enumerate(descriptors2):

        #normalize
        differences = descriptors1 - descriptor2  
        squared_differences = differences ** 2    
        sum_squared = np.sum(squared_differences, axis=1) 
        distances = np.sqrt(sum_squared)

        min_distance_index = np.argmin(distances)
        matches_2.append((min_distance_index, j, distances[min_distance_index]))

    return matches_1, matches_2

def find_valid_matches(matches_1, matches_2):

    valid_matches = check_matches(matches_1, matches_2)

    valid_matches = []  
    #Check all of the matches
    for m in matches_1:
        if matches_2[m[1]][0] == m[0]: 
            valid_matches.append((m[0], m[1]))
    
    return valid_matches

def compute_values(matches_1, matches_2, keypoints1, keypoints2, image1):

    valid_matches = find_valid_matches(matches_1, matches_2)

    #Check the matchees, if there is enough then calculate
    if len(valid_matches) >= 2:

        #Initialize lists to store image1 and image2 coordinates
        image1_points = []  
        image2_points = [] 

        #Iterate over the matches from valid_matches to extract the keypoint coordinates
        for m in valid_matches:
            image1_points.append(keypoints1[m[0]].pt)  
            image2_points.append(keypoints2[m[1]].pt)

        #Convert the lists to numpy arrays and reshape them
        image1_points = np.float32(image1_points).reshape(-1, 2)  
        image2_points = np.float32(image2_points).reshape(-1, 2) 

        #Estimate affine transformation matrix
        matrix = cv2.estimateAffine2D(image1_points, image2_points)[0] 

        #Define corners
        height, width = image1.shape[:2]
        corners = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 2)

        #transforms points with transformation matrix
        distance = cv2.transform(np.array([corners]), matrix).reshape(-1, 2)

        #Calculate metrics including bounding box and angle relative to vertical
        detected_corners = distance.reshape(-1, 2)
        center_x = np.mean(detected_corners[:, 0])
        center_y = np.mean(detected_corners[:, 1])

        #Calculate height lower and upper
        y_min = np.min(detected_corners[:, 1])
        y_max = np.max(detected_corners[:, 1])
        height = y_max - y_min  

        #Calculate angle relative to vertical (y-axis)
        angle = np.degrees(np.arctan2(detected_corners[0][0] - detected_corners[1][0], detected_corners[1][1] - detected_corners[0][1])) 
        angle = (angle + 360) % 360   

    #If something doesn't work & gradescope can only accept this format
    else:
        center_x = 0
        center_y = 0
        height = 0
        angle = 0

    return center_x, center_y, height, angle

main()