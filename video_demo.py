from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import cv2
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from trackanything import TrackingAnything
from mask_painter import mask_painter as mask_painter2
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from skimage.draw import line

def load_images(start, end, directory):
    images = []
    cnt = 0
    for i in range(start, end + 1):
        file_path = os.path.join(directory, str(i) + ".jpg")
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (960, 540))
        if img is not None:
            cnt += 1
            images.append(img)
        else:
            print(f"Error loading {i}.jpg")
    print(f"Total images loaded: {cnt}")
    return images

def select_frames(tool_name, ground_truth_path):
    df = pd.read_csv(ground_truth_path)
    df_filtered = df[df[tool_name] != 0]
    frames = df_filtered['Frame'].tolist()
    return frames

def save_images(images, start):
    for j in range(len(images)):
        img = cv2.cvtColor(images[j], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'./output/test17/{start + j}.jpg', img)
    print(f"{len(images)} images saved.")
        
def masked_image_painter(images, masks, start):
    colors = [np.array([163, 166, 75]), np.array([191, 78, 78]), np.array([255, 165, 0])]
    for i in range(len(images)):
        image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        mask = masks[i] != 0
        #mask = connect_by_line(mask)
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask] = np.array([100, 149, 237])
        combined_image = cv2.addWeighted(image, 1, colored_mask, 0.8, 0)
        cv2.imwrite(f'./output/{start + i}.jpg', combined_image)

def connect_and_convexify(M):
    # Get the indices of all 1s in the matrix
    points = np.argwhere(M == 1)
    
    if len(points) < 3:
        # If there are less than 3 points, we cannot form a convex hull. Return the matrix as is.
        return M

    # Calculate the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Create a mask for the convex hull
    rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], M.shape)
    
    # Fill the convex hull in the original matrix
    M[rr, cc] = 1
    
    return M

def connect_by_line(mask):
    mask = mask.astype(np.uint8) * 255

    if mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) < 2:
        return mask
    largest_contour = contours[0]
    second_largest_contour = contours[1]

    #leftmost = tuple(second_largest_contour[second_largest_contour[:, :, 0].argmin()][0])
    #rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    
    top_right = tuple(largest_contour[np.argmax(largest_contour[:, :, 0] - largest_contour[:, :, 1])][0])
    bottom_left = tuple(second_largest_contour[np.argmin(second_largest_contour[:, :, 0] - second_largest_contour[:, :, 1])][0])
    
    #top_right = (276,258)
    #bottom_left = (334,104)
    
    line_image = np.zeros_like(mask)
    #cv2.line(line_image, leftmost, rightmost, color=1, thickness=13)
    cv2.line(line_image, top_right, bottom_left, color=1, thickness=10)
    mask[line_image == 1] = 1
    
    return mask

def connect_by_lines(masks, thickness=20):
    for i in range(len(masks)):
        mask = centroid_line(masks[i])
        masks[i] = mask
    return masks

def connect_by_line_2(mask, p1, p2, thickness=10):
    mask = mask.astype(np.uint8) * 255
    if mask.sum() == 0:
        return None
    
    line_image = np.zeros_like(mask)
    cv2.line(line_image, p1, p2, color=1, thickness=thickness)
    mask[line_image == 1] = 1
    
    return mask

def centroid_line(mask, thickness=20):
    mask = mask.astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # If there are fewer than 2 contours, return None or handle as needed
    if len(contours) < 2:
        return mask

    # Get the two largest contours
    largest_contour = contours[0]
    second_largest_contour = contours[1]

    def calculate_centroid(contour):
        # Calculate moments for each contour
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        # Calculate the centroid from the moments
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    # Calculate centroids
    centroid1 = calculate_centroid(largest_contour)
    centroid2 = calculate_centroid(second_largest_contour)
    
    line_image = np.zeros_like(mask)
    cv2.line(line_image, centroid1, centroid2, color=1, thickness=thickness)
    mask[line_image == 1] = 1
    
    return mask

if __name__ == '__main__':
    ground_truth_path = "/home/data/CATARACTS/ground_truth/CATARACTS_2018/images/micro_test_gt/"
    frames_path = "/home/data/CATARACTS/"
    
    # tool = 'phacoemulsifier handpiece'
    #'micromanipulator''Bonn forceps''primary incision knife''secondary incision knife'
    video_num = "test17"
    
    #start, end = select_frames('phacoemulsifier handpiece', ground_truth_path + video_num + ".csv")
    #print('phacoemulsifier handpiece', start, end)
    #frames = select_frames('micromanipulator', ground_truth_path + video_num + ".csv")
    
    start, end = 6667, 6689
    #print('micromanipulator', len(frames), frames)
    images = load_images(start, end, "/home/guests/xi_chen/PTT4CS/output/test17_ph")
    #save_images(images, start)

    ta = TrackingAnything()
    points = np.array([(799,82)])
    labels = np.array([1])
    
    mask, logit, painted_image = ta.first_frame_click(images[0], points, labels)
    #mask = centroid_line(mask, 10)
    #mask = connect_by_line_2(mask, (199,294), (0,345))
    #mask = connect_by_line_2(mask, (311,277), (187,299))

    painted_image = mask_painter2(images[0], mask.astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./output/first_mask_{video_num}.jpg', painted_image)
        
    masks, logits, painted_images = ta.generator(images, mask)
    #masks = connect_by_lines(masks, 25)
    masked_image_painter(images, masks, start)