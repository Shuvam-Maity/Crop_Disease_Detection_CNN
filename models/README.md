# Models

All trained models are too large to store on GitHub directly.

They are hosted on HuggingFace Hub:
🔗 [Shuvam-Maity/Crop_Disease_Detection_CNN](https://huggingface.co/Shuvam-Maity/Crop_Disease_Detection_CNN)

## Download a model programmatically

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Shuvam-Maity/Crop_Disease_Detection_CNN",
    filename="tomato_model.keras"
)
```

## Available Models

| File | Crop | Classes | Test Accuracy |
|------|------|---------|---------------|
| `tomato_model.keras` | Tomato | 10 | 89.80% |
| `grape_model.keras` | Grape | 4 | 98.69% |
| `corn_model.keras` | Corn | 4 | 96.03% |
| `apple_model.keras` | Apple | 4 | 97.27% |
| `peach_model.keras` | Peach | 2 | 99.50% |
| `bell_pepper_model.keras` | Bell Pepper | 2 | 100% |
| `potato_model.keras` | Potato | 3 | 96.30% |
| `cherry_model.keras` | Cherry | 2 | 99.65% |
| `strawberry_model.keras` | Strawberry | 2 | 99.15% |
