class PromptService:
    TEMPLATES = [
        "a photo of {}",
        "a close-up photo of {}",
        "a cropped photo of {}",
        "a high-resolution photo of {}",
    ]

    NORMAL = [
        "an intact glass bottle neck",
        "a smooth circular glass rim",
        "an undamaged bottle opening",
        "a perfectly round glass rim",
    ]

    ANOMALY_GROUPS = {
        "broken_large": [
            "a large missing section of glass",
            "a bottle neck with a big chunk missing",
            "a severely broken glass rim",
            "a bottle opening with a large fracture",
        ],
        "broken_small": [
            "a small chip on the glass rim",
            "a minor crack on the bottle opening",
            "a slightly broken bottle edge",
        ],
        "contamination": [
            "foreign material on the glass surface",
            "dirty residue on the bottle neck",
            "a contaminated glass rim",
        ],
    }

    def __init__(self, class_name: str):
        self.class_name = (class_name or "").strip()

    def _compose_subject(self, phrase: str) -> str:
        phrase = (phrase or "").strip()
        if not self.class_name:
            return phrase

        phrase_l = phrase.lower()
        cls_l = self.class_name.lower()

        if cls_l in phrase_l:
            return phrase

        return f"{phrase} {self.class_name}"

    def get_normal_prompts(self):
        subjects = [self._compose_subject(s) for s in self.NORMAL]
        return [t.format(subj) for subj in subjects for t in self.TEMPLATES]

    def get_anomaly_groups(self):
        out = {}
        for label, phrases in self.ANOMALY_GROUPS.items():
            subjects = [self._compose_subject(s) for s in phrases]
            out[label] = [t.format(subj) for subj in subjects for t in self.TEMPLATES]
        return out
