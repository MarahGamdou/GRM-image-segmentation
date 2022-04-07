import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def compute_score(image_name, mask, path):
    img = cv2.imread(f"data/{image_name}.jpg")
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    if os.path.isfile(f"data/{image_name}.png"):
        gt = cv2.imread(f"data/{image_name}.png")
        gt = (np.ma.filled(gt == 1) * 255).mean(axis=2)
        score = max(np.mean(np.ma.filled(mask > 0) * 255 == gt), np.mean((1 - np.ma.filled(mask > 0)) * 255 == gt))
        print(f"Score: {score:.4f}")
        fig, ax = plt.subplots(ncols=2, figsize=[15, 8])
        ax[0].imshow(img)
        ax[1].imshow(np.ma.filled(mask > 0)[:, :, None] * img)
        fig.savefig(os.path.join(path, f"compare_{image_name}.png"))
    else:
        print("No ground truth file")
