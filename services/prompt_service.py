class PromptService:
    STATE_WORDS_NORMAL = ["flawless", "smooth rim", "transparent glass", "solid black background", "empty center"]
    STATE_WORDS_ANOMALY = ["jagged edge", "broken glass rim", "chipped glass neck", "shattered translucent fragments"]
    TEMPLATES = ["a photo of a {}", "a cropped photo of a {}", "a close-up photo of a {}",
                 "a high-resolution photo of a {}", "a picture of {}"]

    def __init__(self, class_name):
        self.class_name = class_name

    def get_normal_prompts(self):
        return [t.format(f"{s} {self.class_name}") for s in self.STATE_WORDS_NORMAL for t in self.TEMPLATES]

    def get_anomaly_prompts(self):
        return [t.format(f"{s} {self.class_name}") for s in self.STATE_WORDS_ANOMALY for t in self.TEMPLATES]
    
    def get_defect_label_prompts(self, labels):
        return [f"a photo of {l}" for l in labels]