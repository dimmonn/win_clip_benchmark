import torch
from PIL import Image

class PatchService:
    def __init__(self, clip_model, window_sizes=(128, 224), strides=(32, 64)):
        self.clip_model = clip_model
        self.window_sizes = window_sizes
        self.strides = strides

    def get_patches(self, image, batch_size=128):
        patches = []
        batch_images, batch_metas = [], []
        
        for size, stride in zip(self.window_sizes, self.strides):
            w, h = image.size
            for top in range(0, h - size + 1, stride):
                for left in range(0, w - size + 1, stride):
                    crop = image.crop((left, top, left + size, top + size))
                    batch_images.append(self.clip_model.preprocess(crop).unsqueeze(0))
                    batch_metas.append((left, top, size))
                    
                    if len(batch_images) >= batch_size:
                        self._process_batch(batch_images, batch_metas, patches)
                        batch_images, batch_metas = [], []
        
        if batch_images:
            self._process_batch(batch_images, batch_metas, patches)
        return patches

    def _process_batch(self, images, metas, patches):
        embs = self.clip_model.encode_image(torch.cat(images, dim=0))
        for m, e in zip(metas, embs):
            patches.append({'left': m[0], 'top': m[1], 'size': m[2], 'emb': e})