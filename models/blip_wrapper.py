import torch

try:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    HAS_BLIP2 = True
except ImportError:
    HAS_BLIP2 = False


class BLIP2Model:
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = None
        self.model = None

    def generate_caption(self, pil_image, context_prompt="a photo of a defect"):
        if not HAS_BLIP2:
            return None
        if self.model is None:
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                       torch_dtype=torch.float16).to(self.device)

        inputs = self.processor(images=pil_image, text=context_prompt, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=30,
            min_new_tokens=10,
            repetition_penalty=1.5
        )
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
