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

    def visualize(self, image_path, anomaly_map, save_name="result.png", model_name="CLIP"):
        img = Image.open(image_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        im = axes[1].imshow(anomaly_map, cmap="jet")
        axes[1].set_title("Anomaly Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()