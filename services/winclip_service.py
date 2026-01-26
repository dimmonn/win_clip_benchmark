import torch
import numpy as np
from collections import Counter

class WinCLIPService:
    def __init__(self, clip_model, patch_service, prompt_service):
        self.clip_model = clip_model
        self.patch_service = patch_service
        self.prompt_service = prompt_service

    def run(self, image):
        normal_prompts = self.prompt_service.get_normal_prompts()
        normal_embs = self.clip_model.encode_text(normal_prompts)
        normal_vec = normal_embs.mean(dim=0)

        anomaly_groups = self.prompt_service.get_anomaly_groups()
        anomaly_data = {
            label: {
                "prompts": prompts,
                "embs": self.clip_model.encode_text(prompts)
            }
            for label, prompts in anomaly_groups.items()
        }

        patches = self.patch_service.get_patches(image)

        anomaly_map = np.zeros((image.height, image.width))
        count_map = np.zeros((image.height, image.width))

        explanations = []
        for p in patches:
            emb = p["emb"].to(self.clip_model.device)

            normal_sim = torch.dot(emb, normal_vec).item()

            best_label = None
            best_text = None
            best_score = -1e9

            for label, data in anomaly_data.items():
                sims = data["embs"] @ emb
                idx = sims.argmax().item()
                score = sims[idx].item() - normal_sim

                if score > best_score:
                    best_score = score
                    best_label = label
                    best_text = data["prompts"][idx]

            if best_score < 0:
                continue

            explanations.append({
                "label": best_label,
                "text": best_text,
                "score": best_score
            })

            l, t, s = p["left"], p["top"], p["size"]
            anomaly_map[t:t+s, l:l+s] += best_score
            count_map[t:t+s, l:l+s] += 1

        count_map[count_map == 0] = 1
        anomaly_map /= count_map

        if not explanations:
            return anomaly_map, [{"label": "good", "text": "intact glass bottle neck"}]

        labels = [e["label"] for e in explanations]
        final_label = Counter(labels).most_common(1)[0][0]

        final_text = Counter(
            [e["text"] for e in explanations if e["label"] == final_label]
        ).most_common(1)[0][0]

        return anomaly_map, [{
            "label": final_label,
            "text": final_text
        }]
