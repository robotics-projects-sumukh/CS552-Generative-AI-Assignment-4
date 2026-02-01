# Assignment 4 - Generative Adversarial Networks (GANs)

This repository contains the code, theory solutions, and resources for Assignment 4 of CS552 - Generative AI.

## Course Information

**Course:** CS552 - Generative AI  
**Instructor:** Narahara Chari Dingari, Ph.D.

---

## Repository Structure

```
Assignments/4/
├── code.ipynb              # Main notebook: theory questions + GAN implementation
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores data/ and code_ignore
├── 041---Assignment-4---Geenerative-Adversarial-Networks.pdf
├── Theory_Questions.md     # Standalone theory solutions (optional)
├── data/                   # Dataset downloads (gitignored)
│   ├── MNIST/
│   └── cifar-10-batches-py/
└── generated_images/      # Saved samples (basic_gan, cifar10)
    ├── basic_gan/
    └── cifar10/
```

---

## Assignment 4: Generative Adversarial Networks (GANs)

This assignment implements and trains GANs on MNIST and CIFAR-10, with convolutional generators and discriminators, binary cross-entropy loss, and image saving every 10 epochs.

### Part 1: Theory Questions

Four questions (as in the assignment PDF):

1. **Q1:** Explain the minimax loss function in GANs and how it ensures competitive training between the generator and discriminator.
2. **Q2:** What is mode collapse? Why can it occur during GAN training? How can it be mitigated?
3. **Q3:** Explain the role of the discriminator in adversarial training.
4. **Q4:** How do metrics like IS (Inception Score) and FID evaluate GAN performance?

Answers are written in markdown in `code.ipynb` (Part 1).

### Part 2: Coding Assignment

**Requirements (from assignment PDF):**

1. **Modify the generator** to include additional convolutional layers.
2. **Implement image saving** after every 10 epochs.
3. **Replace MNIST with CIFAR-10** and update the network for the new dimensions (32×32×3).

**Implementation:**

- **Task 1 – GAN on MNIST**
  - Convolutional generator: Dense → Reshape(7×7×256) → ConvTranspose2d layers → 28×28×1 output.
  - Convolutional discriminator: Conv2d layers → Flatten → Dense(1).
  - Loss: binary cross-entropy for both.
  - Training: alternate D and G updates; save generated images every 10 epochs.
  - Loss vs epoch plot and final samples at the end of Task 1.

- **Task 2 – GAN on CIFAR-10**
  - Same GAN design adapted for 32×32×3 (CIFAR-10).
  - Generator: Dense → Reshape(4×4×512) → ConvTranspose2d → 32×32×3.
  - Discriminator: Conv2d down to 2×2, then 2×2 conv to 1 channel, Sigmoid.
  - Save images every 10 epochs.
  - Loss vs epoch plot and final CIFAR-10 samples at the end of Task 2.

### Features

- Convolutional generator and discriminator (no “DCGAN” naming; same GAN used for both datasets).
- Binary cross-entropy loss and Adam optimizers.
- Image saving every 10 epochs to `generated_images/basic_gan` (MNIST) and `generated_images/cifar10` (CIFAR-10).
- Training loss plotted per epoch (not per iteration) for both tasks.
- Helper to visualize grids of real and generated images.

---

## Setup

### Prerequisites

- Python 3.8 or higher  
- CUDA-capable GPU (optional but recommended for training)

### Installation

1. Go to the Assignment 4 directory:

```bash
cd Assignments/4
```

2. (Recommended) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open `code.ipynb` and run cells in order (Part 1 theory, then Part 2 code).

3. **Task 1 (MNIST):** Run data load, model definition, training, then the “Plot training losses” and “Generate final sample images” cells.  
   - Data is downloaded to `./data/MNIST` on first run.  
   - Images are saved under `generated_images/basic_gan/`.

4. **Task 2 (CIFAR-10):** Run CIFAR-10 data load, model definition, training, then the “Plot CIFAR-10 training losses” and “Generate final sample images” cell.  
   - Data is downloaded to `./data` on first run.  
   - Images are saved under `generated_images/cifar10/`.

---

## Dependencies

| Package      | Version  | Purpose                    |
|-------------|----------|----------------------------|
| torch       | ≥ 2.0.0  | PyTorch                    |
| torchvision | ≥ 0.15.0 | Datasets, transforms, utils|
| matplotlib  | ≥ 3.7.0  | Plots and images           |
| numpy       | ≥ 1.24.0 | Arrays and math            |
| jupyter     | ≥ 1.0.0  | Jupyter environment        |
| notebook    | ≥ 6.5.0  | Notebook UI                |

---

## Ignored Files (`.gitignore`)

- **`data/`** – MNIST and CIFAR-10 downloads (large, re-downloadable).
- **`code_ignore`** – Any file or folder named `code_ignore`.

---

## Outputs

- **MNIST:** `generated_images/basic_gan/epoch_0.png`, `epoch_10.png`, …  
- **CIFAR-10:** `generated_images/cifar10/` (if that directory is used in the notebook).  
- Loss vs epoch is shown in the notebook for both tasks.
