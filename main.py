import numpy as np
import torch
import scipy.ndimage as ndi
from tqdm import tqdm
from PIL import Image

from models.clip_wrapper import CLIPModel
from models.blip_wrapper import BLIP2Model
from services.prompt_service import PromptService
from services.patch_service import PatchService
from factories.model_factory import ModelFactory
from services.benchmark_service import BenchmarkService

class WinCLIPExplainer:
    def __init__(self, class_name="object", clip_name="ViT-B-32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip: CLIPModel = ModelFactory.create_clip(clip_name, self.device)
        self.blip: BLIP2Model = ModelFactory.create_captioner("blip2", self.device)
        self.prompts = PromptService(class_name)
        self.patcher = PatchService(self.clip)
        self.benchmarker = BenchmarkService()
        #TODO test with other classes
        self.class_name = f"glass {class_name}"

        self.defect_labels = [
            "scratch", "dent", "crack", "stain", "hole",
            "misalignment", "missing part", "discoloration"
        ]
        self.normal_embs = self.clip.encode_text(self.prompts.get_normal_prompts())
        self.anomaly_embs = self.clip.encode_text(self.prompts.get_anomaly_prompts())
        self.label_embs = self.clip.encode_text(self.prompts.get_defect_label_prompts(self.defect_labels))


    def run_benchmark(self, image_path, gt_path):
        result = self.explain(image_path)
        auc = self.benchmarker.evaluate_image(result['map'], gt_path)
        self.benchmarker.visualize(image_path, result['map'], gt_path)
        return auc, result

    def _rank_labels(self, patch_emb):
        sims = (patch_emb.to(self.device) @ self.label_embs.T)
        vals, idx = torch.topk(sims, k=3)
        return [(self.defect_labels[i], float(v)) for v, i in zip(vals, idx)]

    def explain(self, image_path):
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        patches = self.patcher.get_patches(img)

        for p in tqdm(patches, desc="Analyzing"):
            emb = p['emb'].to(self.device)
            logits_n = (emb @ self.normal_embs.T)
            logits_a = (emb @ self.anomaly_embs.T)

            sn = logits_n.max().item()
            sa = logits_a.max().item()
            temp = 0.01
            exp_sa = np.exp(sa / temp)
            exp_sn = np.exp(sn / temp)
            p['score'] = exp_sa / (exp_sa + exp_sn)
            p['labels'] = self._rank_labels(emb)
        acc, cnt = np.zeros((H, W)), np.zeros((H, W))
        for p in patches:
            l, t, s = p['left'], p['top'], p['size']
            acc[t:t + s, l:l + s] += p['score']
            cnt[t:t + s, l:l + s] += 1
        amap = acc / np.maximum(cnt, 1)
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        amap = ndi.gaussian_filter(amap, sigma=8)
        top_p = max(patches, key=lambda x: x['score'])
        crop = img.crop((top_p['left'], top_p['top'], top_p['left'] + top_p['size'], top_p['top'] + top_p['size']))

        return {
            'map': amap,
            'explanation': {
                'score': top_p['score'],
                'labels': top_p['labels'],
                'caption': self.blip.generate_caption(crop)
            }
        }


if __name__ == "__main__":
    explainer = WinCLIPExplainer(class_name="bottle", clip_name="ViT-B-16")
    img_p = "bottle/test/broken_large/000.png"
    gt_p = "bottle/ground_truth/broken_large/000_mask.png"

    auc, res = explainer.run_benchmark(img_p, gt_p)
    print(f"AUC: {auc:.4f}")
    print(f"Explanation: {res['explanation']['caption']}")