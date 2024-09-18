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
        img = cv2.resize(img, (960, 540))
        if img is not None:
            cnt += 1
            images.append(img)
        else:
            print(f"Error loading {f}.jpg")
    print(f"Total images loaded: {cnt}")
    return images

def find_tool_tip(mask, most_point):
    mask = mask.astype(np.uint8) * 255
    if mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the extreme points
    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
    
    points = [leftmost, rightmost, topmost, bottommost]

    return points[most_point]

def get_all_tips(masks, most_point):
    res = []
    for mask in masks:
        mask = mask != 0
        tip = find_tool_tip(mask, most_point)
        if tip:
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
    plt.savefig('./output/trajectory/' + video_num + '_trajectory.png')
    plt.show()

def calculate_velocity(coordinates, frame_rate):
    velocities = []
    for i in range(1, len(coordinates)):
        dist = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[i - 1]))
        time = 1 / frame_rate
        velocities.append(dist / time)
    return velocities

def calculate_acceleration(velocities, frame_rate):
    accelerations = []
    for i in range(1, len(velocities)):
        acc = (velocities[i] - velocities[i - 1]) * frame_rate
        accelerations.append(acc)
    return accelerations

def psd_plot(tips, video_num):
    frame_rate = 29.182879377431906
    coordinates = np.array(tips)
    
    velocities = calculate_velocity(coordinates, frame_rate)
    accelerations = calculate_acceleration(velocities, frame_rate)
    
    time_vel = np.linspace(0, len(velocities) / frame_rate, len(velocities))
    time_acc = np.linspace(0, len(accelerations) / frame_rate, len(accelerations))
    
    # Plot velocity
    plt.figure(figsize=(12, 6))
    plt.plot(time_vel, velocities)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (units/s)')
    plt.title('Velocity in ' + video_num)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f'./output/velocity/velocity_{video_num}.png')
    plt.close()

    # Plot acceleration
    plt.figure(figsize=(12, 6))
    plt.plot(time_acc, accelerations)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (units/s²)')
    plt.title('Acceleration in ' + video_num)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f'./output/acceleration/acceleration_{video_num}.png')
    plt.close()
    
    freq, psd = welch(accelerations, fs=frame_rate)
    
    # Specify the tremor frequency band (8 to 12 Hz)
    tremor_band_indices = np.where((freq >= 8) & (freq <= 12))[0]
    tremor_bandpower = np.trapezoid(psd[tremor_band_indices], x=freq[tremor_band_indices])
    
    peak_frequency = freq[np.argmax(psd)]
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(freq, psd, label='PSD')
    plt.fill_between(freq[tremor_band_indices], psd[tremor_band_indices], alpha=0.5, label='Tremor Band (8-12 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (Power/Hz)')
    plt.title('Power Spectral Density of ' + video_num)
    plt.legend()
    plt.grid(True)
    plt.savefig('./output/psd/' + video_num + '_PSD.png')
    plt.show()
    plt.close()
    
    return tremor_bandpower, peak_frequency

def compare_psds(peak_frequencies, tremor_bandpowers, video_nums):
    # Ensure the lists are of the same length
    assert len(peak_frequencies) == len(tremor_bandpowers), "Lists must have the same length."

    # Create a DataFrame to tabulate the metrics
    metrics_df = pd.DataFrame({
        'Peak Frequency (Hz)': peak_frequencies,
        'Tremor Band Power': tremor_bandpowers
    }, index=video_nums)

    # Plot the metrics
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(metrics_df.index, metrics_df['Peak Frequency (Hz)'], marker='o', linestyle='-')
    plt.ylabel('Peak Frequency (Hz)')
    plt.title('Comparison of Peak Frequency')

    plt.subplot(2, 1, 2)
    plt.plot(metrics_df.index, metrics_df['Tremor Band Power'], marker='o', linestyle='-', color='red')
    plt.ylabel('Tremor Band Power')
    plt.title('Comparison of Tremor Band Power')
    #plt.set_xticklabels(metrics_df.index, rotation=90)
    plt.xlabel('PSDs Index')

    plt.tight_layout()
    plt.savefig('./output/comparison_1.png')
    plt.close()

    return metrics_df

def get_all_centroids(masks):
    res = []
    for i in range(len(masks)):
        mask = masks[i]
        indices = np.argwhere(mask == 1)
        # Calculate the centroid
        if indices.size:
            centroid = indices.mean(axis=0)
            res.append(centroid)
    return res

def save_images(images, frames, img_name):
    for j in range(len(images)):
        img = cv2.cvtColor(images[j], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{img_name}_{frames[j]}.jpg', img)

def psd_plot_2(tips, centroids, video_num):
    frame_rate = 29.182879377431906
    coordinates = np.array(tips)
    coordinates2 = np.array(centroids)
    
    velocities = calculate_velocity(coordinates, frame_rate)
    accelerations = calculate_acceleration(velocities, frame_rate)
    velocities2 = calculate_velocity(coordinates2, frame_rate)
    accelerations2 = calculate_acceleration(velocities2, frame_rate)
    
    time_vel = np.linspace(0, len(velocities) / frame_rate, len(velocities))
    time_acc = np.linspace(0, len(accelerations) / frame_rate, len(accelerations))
    
    # Plot velocity
    plt.figure(figsize=(12, 6))
    plt.plot(time_vel, velocities, label='tip')
    plt.plot(time_vel, velocities2, label='centroid')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (units/s)')
    plt.title('Velocity in ' + video_num)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f'velocity_{video_num}.png')
    plt.close()

    # Plot acceleration
    plt.figure(figsize=(12, 6))
    plt.plot(time_acc, accelerations, label='tip')
    plt.plot(time_acc, accelerations2, label='centroid')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (units/s²)')
    plt.title('Acceleration in ' + video_num)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f'acceleration_{video_num}.png')
    plt.close()

if __name__ == '__main__':
    ground_truth_path = "/home/data/CATARACTS/ground_truth/CATARACTS_2018/images/micro_train_gt/"
    frames_path = "/home/data/CATARACTS/train/"
    
    tool = 'hydrodissection cannula'
    prompts = [[(980, 800)],
               [(825, 900)],
               [(890, 900)],
               [(820, 1000)],
               [(840, 1020)], #5
               [(780, 1020)], 
               [(1050, 1020)],
               [(1140, 1050)],
               [(1120, 1000)], 
               [(1300, 900)], #10
               [(1020, 900)],
               [(1120, 900)],
               [(850, 900)],
               [(860, 1020)],
               [(1300, 1040)], #15
               [(1100, 1000)],
               [(950, 1000)],
               [(900, 1000)],
               [(1250, 1000)],
               [(1000, 1000)], #20
               [(1060, 900)],
               [(1120, 1000)],
               [(700, 1000)],
               [(900, 1000)],
               [(1100, 1000)] #25
            ]
    prompts = [[(x // 2, y // 2) for (x, y) in prompt] for prompt in prompts]
    
    tremor_bandpowers, peak_frequencies = [], []
    video_nums = []
    
    for i in range(1, 2):
        video_num = "train"+str(i).zfill(2)
        video_nums.append(video_num)
        print("Processing video: " + video_num)
        
        frames = select_frames(tool, ground_truth_path + video_num + ".csv")
        if i in [7, 9, 16]:
            frames = frames[5:]
        images = load_images(frames, frames_path + video_num)
            
        ta = TrackingAnything()
        points = np.array(prompts[i - 1])
        print(f"prompt point: {points}")
        labels = np.array([1])
        
        mask, logit, painted_image = ta.first_frame_click(images[0], points, labels)
        painted_image = mask_painter2(images[0], mask.astype('uint8'), background_alpha=0.8)
        painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'./output/mask/first_mask_{video_num}.jpg', painted_image)
        
        masks, logits, painted_images = ta.generator(images, mask)
        #save_images(images[:20], frames[:20], f"{video_num}")
        #save_images(painted_images[:20], frames[:20], f"painted_images_{video_num}")
        
    '''
        #use centroid
        centroids = get_all_centroids(masks)
        tips = get_all_tips(masks, 2)
        #psd_plot_2(tips, centroids, video_num)
        #print(centroids)
        #trajectory_plot(centroids, video_num)
        tremor_bandpower, peak_frequency = psd_plot(centroids, video_num + "(centroid)")
        tremor_bandpowers.append(tremor_bandpower)
        peak_frequencies.append(peak_frequency)
    
    print(tremor_bandpowers)
    print(peak_frequencies)
    
    metrics_df = compare_psds(peak_frequencies, tremor_bandpowers, video_nums)
    print(metrics_df)
    '''
    
    '''
        #use tip
        if i in [19]:
            tips = get_all_tips(masks, 0)
        else:
            tips = get_all_tips(masks, 2)
        trajectory_plot(tips, video_num)
        tremor_bandpower, peak_frequency = psd_plot(tips, video_num)
        tremor_bandpowers.append(tremor_bandpower)
        peak_frequencies.append(peak_frequency)
    
    metrics_df = compare_psds(peak_frequencies, tremor_bandpowers, video_nums)
    print(metrics_df)
    '''
        
    #sbatch --job-name=td --output=td-%A.out --error=td-%A.err --nodelist=corellia,dagobah --mail-type=ALL --mail-user=ge27vic@mytum.de tremor_detection.sh