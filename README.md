# 🐉 TinyVisionPipeline
A simple, modular deep learning pipeline for image classification using PyTorch and the CIFAR-10 dataset.

This project serves as a template and demonstration for building a clean computer vision pipeline — from data loading and preprocessing to model training, evaluation, and saving.

## 📁 Project Structure
```
TinyVisionPipeline/
├── tiny_vision_pipeline/
│   ├── datasets/        # Data loading and preprocessing
│   ├── model/           # Model architecture
│   ├── prediction/      # Scripts for visualizing model predictions on images
│   ├── evaluate/        # Scripts and utilities for evaluating trained models
│   ├── trained_models/  # Saved trained models
│   ├── Utils/           # Utility functions
│   └── main.py          # Training logic and checkpointing
├── experiments/         # Local experiments (ignored by git)
├── data/                # Datasets (ignored by git)
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

## 🚀 Getting Started
1. Clone the repository

```bash
git clone https://github.com/Anita-P-B/TinyVisionPipeline.git
cd TinyVisionPipeline 
```

2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  
```
3. Install the package locally

First, activate your virtual environment.
Then install the following:
```bash
pip install -e .
pip install -r requirements.txt
```
### ⚡ Optional: Enable GPU Acceleration

By default, this project installs the CPU version of PyTorch.  
If your system supports CUDA and you want faster training, run:

```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
````


3. Start training 
```bash
python tiny_vision_pipeline/main.py --save_path "your_model_name"
```
📝 This will automatically create a new folder named "your_model_name" inside the experiments/ directory.

All training logs and model checkpoints will be saved there by default.

## 🔧 Features

✅ Modular code layout (easy to expand or replace parts)

✅ Clean separation of training, data, and model logic

✅ Supports image augmentation and checkpointing

✅ Model testing & visualization

## 🔍 Evaluation vs. Prediction

This project includes two utility scripts that allow you to work with trained models in different ways:

- eval_summary.py  – For developers and model trainers.
Run this to evaluate a trained model on the test set, with detailed class-wise accuracy and confidence metrics. Useful for inspecting model performance after training.

- predict.py – For general users or demos.
Run this to predict labels on test or custom images using an existing trained model. Ideal for quick visual checks or exploring predictions without digging into performance details.

### 📊 Evaluation Utility (eval_summary.py)
The `eval_summary.py` script runs your trained model on the CIFAR-10 test set and prints:

- Overall accuracy
- Class-wise accuracy
- Class confidence levels

It’s useful for developers who want deeper insight into model performance.

#### 🚀 Usage

```bash
python tiny_vision_pipeline/evaluate/eval_summary.py  --eval_model_path "your/model/path.pt" 
```
Replace "your/model/path.pt" with the path to your model's best checkpoint.

### 🔮 Prediction Utility (predict.py)

The predict.py allows you to test your trained model easily by displaying predictions on images.

🔍 What it does:
- Loads your trained model.

- Runs predictions on 9 random images.

- Displays a grid with:

    - The image
    
    - The predicted label
    
    - The confidence score (probability).

This is useful for quickly checking that your model works as expected after training!

#### ⚙️ Command-line options
You have two testing modes:

| Option          | What it does                                                        |
| --------------- | ------------------------------------------------------------------- |
| `--test`        | Tests your model on 9 **random images from the CIFAR-10 test set.** |
| `--custom_test` | Tests your model on 9 **custom images** from your chosen folder.    |

#### 🖼️ Custom images requirements
Images should be placed in a folder you specify.

Supported formats: .jpg, .jpeg, .png

The script will automatically resize and preprocess your images to match the model input.

#### 🚀 Usage examples

1️⃣ Test on CIFAR-10 test set:

```bash 
python tiny_vision_pipeline/prediction/predict.py --test
```

2️⃣ Test on your own custom images:
```bash
python tiny_vision_pipeline/prediction/predict.py --custom_test "path/to/your/images"
```
(Replace path/to/your/images with the path to your image folder.)

✅ Pro tip:
If your model was saved on a GPU but you're running on CPU now, no worries—predict.py automatically handles the device (CPU/GPU) for smooth running.

## 📜 License

This project is released under the MIT License.
Use, learn, remix — but give credit where due. ⚔️✨