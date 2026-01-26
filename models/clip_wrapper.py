import torch
import torch.nn.functional as F
import clip
import open_clip


class CLIPModel:
    def __init__(self, model_name="ViT-B-32", device="cpu", is_open_clip: bool = False):
        self.device = device
        self.model_name = model_name
        if is_open_clip:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.name = "mlfoundations_open_clip"
        else:
            self.model, self.preprocess = clip.load("ViT-B/32", device=device)
            self.tokenizer = clip.tokenize
            self.name = "openai_CLIP"
        self.model.to(device).eval()

    def encode_text(self, texts, batch_size=64):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                tokens = self.tokenizer(batch_texts).to(self.device)
                emb = self.model.encode_text(tokens)
                all_embs.append(F.normalize(emb, dim=-1).cpu())
        return torch.cat(all_embs, dim=0).to(self.device)

    def encode_image(self, images_t):
        with torch.no_grad():
            emb = self.model.encode_image(images_t.to(self.device))
            return F.normalize(emb, dim=-1).cpu()
