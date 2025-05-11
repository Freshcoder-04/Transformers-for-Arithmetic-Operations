# Transformer Arithmetic Report (+ / –)

## Configurations used:

Config 1:

```TRAIN_FILE = '/kaggle/working/data/train.csv'
VAL_FILE = '/kaggle/working/data/val.csv'
MODEL_DIR = '/kaggle/working/models'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
BATCH_SIZE = 512
EPOCHS = 70
LEARNING_RATE = 0.00005
D_MODEL = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
D_FF = 2048
DROPOUT = 0.2
MAX_LENGTH = 20
SAVE_EVERY_N_EPOCHS = 5

MIN_DIGITS   = 1                # minimum digits per operand
MAX_DIGITS   = 3                # maximum digits per operand
TOTAL        = 600000           # total samples in train+test+val
TRAIN_SIZE   = int(0.8*TOTAL)   # examples in train split
TEST_SIZE    = int(0.1*TOTAL)   # examples in test split
VAL_SIZE     = int(0.1*TOTAL)   # examples in val split
GEN_SIZE     = int(0.1*TOTAL)   # examples in generatlization set
EXTRA_MIN    = 0                # add at least this many digits for generalization
EXTRA_MAX    = 3                # add up to this many digits for generalization
EDGE_FRAC    = 0.7              # fraction of each split reserved for edge cases
```

Config 2:
```
# Hard-coded hyperparameters and paths
TRAIN_FILE = '/kaggle/working/data/train.csv'
VAL_FILE = '/kaggle/working/data/val.csv'
MODEL_DIR = '/kaggle/working/models'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
BATCH_SIZE = 512
EPOCHS = 70
LEARNING_RATE = 0.00005
D_MODEL = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
D_FF = 2048
DROPOUT = 0.2
MAX_LENGTH = 20
SAVE_EVERY_N_EPOCHS = 5

MIN_DIGITS   = 1                # minimum digits per operand
MAX_DIGITS   = 3                # maximum digits per operand
TOTAL        = 1000000           # total samples in train+test+val
TRAIN_SIZE   = int(0.8*TOTAL)   # examples in train split
TEST_SIZE    = int(0.1*TOTAL)   # examples in test split
VAL_SIZE     = int(0.1*TOTAL)   # examples in val split
GEN_SIZE     = int(0.1*TOTAL)   # examples in generatlization set
EXTRA_MIN    = 0                # add at least this many digits for generalization
EXTRA_MAX    = 3                # add up to this many digits for generalization
EDGE_FRAC    = 0.8              # fraction of each split reserved for edge cases
```

Config 3:

```
# Hard-coded hyperparameters and paths
TRAIN_FILE = '/kaggle/working/data/train.csv'
VAL_FILE = '/kaggle/working/data/val.csv'
MODEL_DIR = '/kaggle/working/models'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 0.00001
D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
D_FF = 1024
DROPOUT = 0.0
MAX_LENGTH = 20
SAVE_EVERY_N_EPOCHS = 5

# ------------------ CONFIGURATION ------------------
MIN_DIGITS   = 1                # minimum digits per operand
MAX_DIGITS   = 3                # maximum digits per operand
TOTAL        = 1000000           # total samples in train+test+val
TRAIN_SIZE   = int(0.8*TOTAL)   # examples in train split
TEST_SIZE    = int(0.1*TOTAL)   # examples in test split
VAL_SIZE     = int(0.1*TOTAL)   # examples in val split
GEN_SIZE     = int(0.1*TOTAL)   # examples in generatlization set
EXTRA_MIN    = 0                # add at least this many digits for generalization
EXTRA_MAX    = 3                # add up to this many digits for generalization
EDGE_FRAC    = 0.8              # fraction of each split reserved for edge cases
# ---------------------------------------------------------
```

---

## 1. Quantitative Performance

### Test metrics:
---
| Model   | Exact Match Accuracy | Character-Level Accuracy | Perplexity |
|---------|----------------------|---------------------------|------------|
| Model 1 |      99.66%          |         99.81%            |  1.0032    |
| Model 2 |      99.55%          |         99.78%            |  1.0046    |
| Model 3 |      45.86%          |         61.71%            |  2.0718    |
---

### Generalization metrics:

| Model   | Exact Match Accuracy | Character-Level Accuracy | Perplexity |
|---------|----------------------|---------------------------|------------|
| Model 1 |      39.92%          |         49.14%            |   302.6286 |
| Model 2 |      39.92%          |         47.79%            |   363.3123 |
| Model 3 |      22.46%          |         35.69%            |   827.1450 |

---

## 2. Generalization Analysis

The **generalization split** contains numbers up to three digits **longer** than those seen in training and the same 80 % “edge-case” ratio (forced carry/borrow).

As we can see, all the models perform poorly on the generalization set.

This shows that the models did not learn long term patterns.


---

## 3. Error Analysis

### 3 a. Qualitative Categories

The first 2 models perform very well upto 3 digits.
</br>
The last model makes mistakes as if it hasn't been trained properly.
But that is not the case, the model has been trained to the point that the accuracy becomes almost stagnant and tries to overfit the data.
</br>

### 3 b. Correlation with Input Features
The last model doesn't perform well on longer sequences and those inputs where there is propagation of carry.

---

## 4. Ablation & Sensitivity Study

Changing the dataset size from 600k to 1M did not make a significant difference as the model was already performing well. But in theory, if I had trained the model on more number of digits and then increased the dataset size, the model would've trained better.
</br>
A model with lesser parameters struggles to learn patterns in arithmetic operations and doesn't provide staisfactory result.

---

## 5. Discussion

* **Has the model learned “true” arithmetic?**  
    No

* **Comparison to human computation**  
  Human computation involves the application of rules and computing the answers in a deterministic manner rather than learning some kind of patterns.

---
