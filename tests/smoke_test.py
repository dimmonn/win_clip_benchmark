import torch
import clip

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

model, _ = clip.load("ViT-B/16")
print("CLIP loaded successfully")