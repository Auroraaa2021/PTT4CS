import numpy as np
import cv2
from tqdm import tqdm
from interact_tools import SamControler
from tracker.base_tracker import BaseTracker

class TrackingAnything():
    def __init__(self, device="cuda"):
        self.samcontroler = SamControler("./sam_checkpoint/sam_vit_h_4b8939.pth", "vit_h", device)
        self.xmem = BaseTracker("./xmem_checkpoint/XMem.pth", device=device)
    
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

    def generator(self, images: list, template_mask:np.ndarray):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:           
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
                
            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images
    
    def get_all_tips(self, images: list, template_mask:np.ndarray, most_point):
        tips = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:           
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                tips.append(find_tool_tip(mask, most_point))
            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                tips.append(find_tool_tip(mask, most_point))
        return tips
    

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