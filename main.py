import time
from factories.model_factory import ModelFactory
from services.patch_service import PatchService
from services.prompt_service import PromptService
from services.winclip_service import WinCLIPService
from services.benchmark_service import BenchmarkService
from PIL import Image
import torch

# Testing WinCLIP model for explainable anomaly detection.
# See if the model can be used to obtain textual explanations of a
# detrected anomalies.
# https://github.com/openai/CLIP.git
# https://github.com/mlfoundations/open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

win_clips = []

clip_model1 = ModelFactory.create_clip("ViT-B-16", device, True)
patch_service1 = PatchService(
    clip_model1,
    window_sizes=(16,),
    strides=(8,)
)
prompt_service1 = PromptService("glass bottle neck")

clip_model2 = ModelFactory.create_clip("ViT-B-16", device, False)
patch_service2 = PatchService(
    clip_model2,
    window_sizes=(16,),
    strides=(8,)
)
prompt_service2 = PromptService("glass bottle neck")


winclip1 = WinCLIPService(
    clip_model1,
    patch_service1,
    prompt_service1
)


winclip2 = WinCLIPService(
    clip_model2,
    patch_service2,
    prompt_service2
)

win_clips.append(winclip1)
win_clips.append(winclip2)

benchmark1 = BenchmarkService()

image_paths = {
    "broken_large": "bottle/broken_large/000.png",
    "broken_small": "bottle/broken_small/000.png",
    "contamination": "bottle/contamination/000.png",
    "good": "bottle/good/000.png",
}

ts = int(time.time())

def print_explanation(label, explanations):
    if explanations and explanations[0] != "no anomaly detected":
        print(f"{label}: {explanations[0]}")
    else:
        print(f"{label}: no anomaly detected")

for label, image_path in image_paths.items():
    image = Image.open(image_path).convert("RGB")

    for winclip in win_clips:
        anomaly_map, explanations = winclip.run(image)

        result_dir = "/".join(image_path.split("/")[:-1])

        benchmark1.visualize(
            image_path,
            anomaly_map,
            save_name=f"{result_dir}/{winclip.clip_model.name}_winclip_{label}_{ts}.png"
        )

        print_explanation(label, explanations)




