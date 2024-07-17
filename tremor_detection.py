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

def select_frames(tool_name, ground_truth_path):
    df = pd.read_csv(ground_truth_path)
    df_filtered = df[df[tool_name] == 1]
    frames = df_filtered['Frame'].tolist()
    return frames

def load_images(frames, directory):
    images = []
    cnt = 0
    for f in frames:
        file_path = os.path.join(directory, str(f) + ".jpg")
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            cnt += 1
            images.append(img)
        else:
            print(f"Error loading {f}.jpg")
    print(f"Total images loaded: {cnt}")
    return images

def find_tool_tip(mask, most_point):
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the extreme points
    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
    
    points = [leftmost, rightmost, topmost, bottommost]

    return points[most_point]

def get_all_points(masks, most_point):
    res = []
    for mask in masks:
        mask = mask != 0
        tip = find_tool_tip(mask, most_point)
        res.append(tip)
    return res

def trajectory_plot(tips, video_num):
    x_values, y_values = zip(*tips)
    plt.figure()
    for i in range(len(tips) - 1):
        x1, y1 = tips[i]
        x2, y2 = tips[i + 1]
        plt.plot([x1, x2], [y1, y2], 'bo-', markersize=5)
    plt.plot(x_values[-1], y_values[-1], 'bo', markersize=5)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Trajectory of ' + tool + ' tip of ' + video_num)
    plt.savefig(video_num + '_trajectory.png')
    plt.show()
    
def psd_plot(tips, video_num):
    frame_rate = 29.182879377431906
    time = np.arange(len(tips)) / frame_rate
    coordinates = np.array(tips)
    x, y = coordinates[:, 0], coordinates[:, 1]
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_path_length = np.sum(distances)
    elapsed_time = time[-1] - time[0]
    normalized_path_length = total_path_length / elapsed_time
    freq, psd = welch(x, fs=frame_rate, nperseg=len(x))
    
    # Specify the tremor frequency band (8 to 12 Hz)
    tremor_band_indices = np.where((freq >= 8) & (freq <= 12))[0]
    tremor_bandpower = np.trapz(psd[tremor_band_indices], x=freq[tremor_band_indices])

    # Print results
    print(f"Normalized path length: {normalized_path_length}")
    print(f"Absolute tremor bandpower: {tremor_bandpower}")
    print(f"Mean of PSD: {psd.mean()}")

    # Plot PSD (optional)
    plt.figure(figsize=(8, 4))
    plt.semilogy(freq, psd, label='PSD')
    plt.fill_between(freq[tremor_band_indices], psd[tremor_band_indices], alpha=0.5, label='Tremor Band (8-12 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Power Spectral Density of ' + video_num)
    plt.legend()
    plt.grid(True)
    plt.savefig(video_num + '_PSD.png')
    plt.show()
    
    return psd

if __name__ == '__main__':
    ground_truth_path = "/home/data/CATARACTS/groud_truth/CATARACTS_2017/train_gt/"
    frames_path = "/home/data/CATARACTS/train/"
    tool = 'hydrodissection canula'
    
    prompts = [[(980, 800)],
               [(825, 900)],
               [(890, 900)],
               [(820, 1000)],
               [(840, 1020)]]
    
    psds = []
    
    for i in range(1, 6):
        video_num = "train"+str(i).zfill(2)
        print("Processing video: " + video_num)
        
        frames = select_frames(tool, ground_truth_path + video_num + ".csv")
        images = load_images(frames, frames_path + video_num)
        
        ta = TrackingAnything()
        points = np.array(prompts[i - 1])
        labels = np.array([1])
        mask, logit, painted_image = ta.first_frame_click(images[0], points, labels)
        masks, logits, painted_images = ta.generator(images, mask)
        
        tips = get_all_points(masks, 2)
        trajectory_plot(tips, video_num)