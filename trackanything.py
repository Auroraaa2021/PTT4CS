import numpy as np
from tqdm import tqdm
from interact_tools import SamControler
from tracker.base_tracker import BaseTracker

class TrackingAnything():
    def __init__(self, device="cpu"):
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