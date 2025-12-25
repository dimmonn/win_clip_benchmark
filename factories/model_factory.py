from models.clip_wrapper import CLIPModel
from models.blip_wrapper import BLIP2Model

class ModelFactory:
    @staticmethod
    def create_clip(model_name="ViT-B-32", device="cpu"):
        return CLIPModel(model_name=model_name, device=device)

    @staticmethod
    def create_captioner(model_type="blip2", device="cpu"):
        if model_type == "blip2":
            return BLIP2Model(device=device)
        return None