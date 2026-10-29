# CNN Image Classification — CIFAR-10

**Authors:** Malak Rai & Ayoub Outahar

A convolutional neural network (CNN) built with PyTorch to classify the 32 × 32 colour images of the CIFAR-10 dataset into 10 classes.

The work started from a Fashion-MNIST exercise notebook. We kept its structure and instructions, so the file name and some instruction cells still mention Fashion-MNIST. All the code, results, plots and answers are for CIFAR-10.

## Repository contents

| File | Description |
|---|---|
| `Fashion_MNIST_CNN_Exercise.ipynb` | The full notebook: data loading, CNN design, training, evaluation, analysis and experiments (CIFAR-10) |
| `CIFAR_cnn.pth` | Saved weights (`state_dict`) of the trained baseline model (`CifarCNN`) |

The dataset is not stored in the repository. The notebook downloads it into `./data` the first time it runs (about 170 MB).

## Dataset

CIFAR-10 contains 60,000 RGB images of size 32 × 32 (tensor shape `3 × 32 × 32`) in 10 classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

| Split | Images |
|---|---:|
| Training | 40,000 |
| Validation | 10,000 (taken from the official training set, seed 42) |
| Test | 10,000 |

## Model

```
Input 3×32×32
→ Conv2d(3, 16, 3×3, padding=1) → ReLU → MaxPool(2)    # 16×16×16
→ Conv2d(16, 32, 3×3, padding=1) → ReLU → MaxPool(2)   # 32×8×8
→ Flatten (2048) → Linear(2048, 128) → ReLU → Linear(128, 10)   # 10 logits
```

Training setup:

- **Loss:** `CrossEntropyLoss`
- **Optimizer:** Adam, learning rate 0.001
- **Batch size:** 64
- **Epochs:** 10
- **Seed:** 42

## Results

Baseline model on the test set: **67.23 % accuracy** (test loss 0.962).

Accuracy per class on the test set:

| Class | Accuracy | Class | Accuracy |
|---|---:|---|---:|
| airplane | 73.4 % | dog | 55.9 % |
| automobile | 74.5 % | frog | 78.2 % |
| bird | 54.0 % | horse | 67.2 % |
| cat | 38.3 % | ship | 79.9 % |
| deer | 76.0 % | truck | 74.9 % |

### Architecture experiments

We changed one factor at a time and trained each model for 10 epochs:

| Experiment | Change | Train Acc. | Val. Acc. | Test Acc. |
|---|---|---:|---:|---:|
| Baseline | — | 73.77 % | 66.77 % | 67.23 % |
| 1 | Dropout p = 0.5 | 61.91 % | 65.37 % | 65.94 % |
| 2 | Conv2: 32 → 64 filters | 79.48 % | 68.77 % | 68.93 % |
| 3 | Third conv layer (64 filters) | 80.25 % | **69.39 %** | 68.76 % |
| 4 | ReLU → GELU | 83.94 % | 67.90 % | 66.91 % |

We selected models by validation accuracy, so the best architecture is the one with a third convolutional layer. Larger models gain accuracy but overfit more. Dropout removes the overfitting, but after only 10 epochs it does not learn enough to match the baseline.

The numbers come from our last run. Because of random weight initialisation, shuffling and dropout, running the notebook again can give slightly different values.

## What the notebook covers

1. Loading and inspecting CIFAR-10 with `torchvision`
2. Splitting the data into training, validation and test sets, and creating the `DataLoader`s
3. How convolution, pooling and ReLU change tensor shapes
4. Designing the CNN and checking its output shape
5. Choosing the loss function and optimizer, with justification
6. Training and validation loops, with loss and accuracy curves
7. Final test evaluation, sample predictions and per-class accuracy
8. Visualising the feature maps of the first convolutional layer
9. Diagnosing overfitting and underfitting
10. A dropout regularisation experiment
11. Saving and reloading the model
12. Architecture experiments and the written answers to the theory questions

## How to run

Requirements: Python 3.9+ with

```bash
pip install torch torchvision numpy matplotlib pandas jupyter
```

Open the notebook and run all cells from top to bottom:

```bash
jupyter notebook Fashion_MNIST_CNN_Exercise.ipynb
```

The notebook uses a GPU when one is available. Otherwise it runs on the CPU.

### Using the saved model

```python
import torch
import torch.nn as nn

class CifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CifarCNN()
model.load_state_dict(torch.load("CIFAR_cnn.pth", map_location="cpu"))
model.eval()
```

Inputs must be tensors of shape `(N, 3, 32, 32)` with values in `[0, 1]`, as produced by `transforms.ToTensor()`. No normalisation is applied.
