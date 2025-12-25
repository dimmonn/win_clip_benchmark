# WinCLIP Anomaly Detection & Explanation Benchmark

An advanced, Object-Oriented implementation of the **WinCLIP** (Window-based CLIP) framework for zero-shot industrial anomaly detection. This project extends standard localization with semantic reasoning using **BLIP-2** to provide human-readable explanations of detected defects.

---

## Architecture & Design Patterns

The project is built on **SOLID principles**, prioritizing composition over inheritance and decoupling model logic from application services.

### Components
- **`models/`**: Domain wrappers for Foundation Models.
    - `CLIPModel`: Handles zero-shot localization and patch embedding.
    - `BLIP2Model`: Provides visual-linguistic reasoning for patch captioning.
- **`factories/`**: Implementation of the **Factory Pattern**.
    - `ModelFactory`: Centralizes model instantiation and device management (CPU/CUDA).
- **`services/`**: Core logic layers.
    - `PatchService`: Implements multi-scale sliding window algorithms.
    - `PromptService`: Manages complex prompt ensembling for anomaly states.
    - `BenchmarkService`: Evaluates performance using Pixel-AUC and generates diagnostic plots.

---

## Getting Started

### 1. Installation
Ensure you have [Conda](https://docs.conda.io/en/latest/) installed.
```bash
conda env create -f environment.yml

conda activate win_clip_benchmark
```

### 2. Running the Explainer
The `main.py` script serves as the entry point. It initializes the `WinCLIPExplainer` which composes all necessary services.

```python
from main import WinCLIPExplainer

explainer = WinCLIPExplainer(class_name="bottle")

result = explainer.explain("bottle/test/broken_large/000.png")

print(f"Defect Class: {result['explanation']['labels'][0][0]}")
print(f"AI Description: {result['explanation']['caption']}")
```

---

## Benchmarking & Interpretation

### How it works
1.  **Localization**: `PatchService` extracts overlapping windows at multiple scales (e.g., 128x128, 224x224).
2.  **Zero-Shot Scoring**: Each patch is compared against "normal" and "anomaly" text prompts.
3.  **Explanation**: The most anomalous patch is sent to **BLIP-2**.
    *   **Top Labels**: Cosine similarity rankings against a bank of defect terms (crack, stain, etc.).
    *   **Caption**: A semantic description of the specific crop (e.g., *"a circular object with a red light on top"*).


## Project Structure
```text
 ~/PythonProject/win_clip_benchmark   main ±✚  tree                   
.
├── README.md
├── anomaly_map.png
├── bottle
│             ├── readme.txt
│             └── test
│                 └── broken_large
│                     └── 000.png
├── environment.yml
├── factories
│             ├── __init__.py
│             ├── __pycache__
│             │             ├── __init__.cpython-310.pyc
│             │             └── model_factory.cpython-310.pyc
│             └── model_factory.py
├── main.py
├── models
│             ├── __init__.py
│             ├── __pycache__
│             │             ├── __init__.cpython-310.pyc
│             │             ├── blip_wrapper.cpython-310.pyc
│             │             └── clip_wrapper.cpython-310.pyc
│             ├── blip_wrapper.py
│             └── clip_wrapper.py
├── results
│             └── result.png
├── services
│             ├── __init__.py
│             ├── __pycache__
│             │             ├── __init__.cpython-310.pyc
│             │             ├── benchmark_service.cpython-310.pyc
│             │             ├── patch_service.cpython-310.pyc
│             │             └── prompt_service.cpython-310.pyc
│             ├── benchmark_service.py
│             ├── patch_service.py
│             └── prompt_service.py
└── tests
    └── smoke_test.py
```

## License
This project is intended for research and benchmarking purposes. It utilizes models CLIP and BLIP-2. Please refer to the `readme.txt` in the bottle folder for usage terms.