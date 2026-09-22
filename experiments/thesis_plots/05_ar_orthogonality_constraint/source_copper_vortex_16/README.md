# x01

## Setup

**Mac:**
```bash
uv sync
```

**UCL (CUDA 13):**
```bash
uv sync
```

**Isambard AI (CUDA 12.7):**
```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)
```
Expected: True 12.6


## Training

Do one of either.
```bash
uv run python -m train_ar
uv run python -m train_ar --smoke
```
