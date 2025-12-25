import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from PIL import Image
import os


class BenchmarkService:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_image(self, anomaly_map, gt_mask_path):
        if not os.path.exists(gt_mask_path):
            return None

        gt_mask = np.array(Image.open(gt_mask_path).convert("L")) > 0
        y_true = gt_mask.flatten()
        y_score = anomaly_map.flatten()

        auc = roc_auc_score(y_true, y_score)
        return auc

    def visualize(self, image_path, anomaly_map, gt_mask_path, save_name="result.png"):
        img = Image.open(image_path)
        gt = np.array(Image.open(gt_mask_path)) if os.path.exists(gt_mask_path) else None

        fig, axes = plt.subplots(1, 3 if gt is not None else 2, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Input")

        im = axes[1].imshow(anomaly_map, cmap='jet')
        axes[1].set_title("Anomaly Map")
        plt.colorbar(im, ax=axes[1])

        if gt is not None:
            axes[2].imshow(gt, cmap='gray')
            axes[2].set_title("Ground Truth")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name))
        plt.close()