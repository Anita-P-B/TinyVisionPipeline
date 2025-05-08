# 🐉 TinyVisionPipeline
A simple, modular deep learning pipeline for image classification using PyTorch and the CIFAR-10 dataset.

This project serves as a template and demonstration for building a clean computer vision pipeline — from data loading and preprocessing to model training, evaluation, and saving.

## 📁 Project Structure
```
TinyVisionPipeline/
├── tiny_vision_pipeline/
│   ├── datasets/        # Data loading and preprocessing
│   ├── model/           # Model architecture
│   ├── prediction/      # Test and visulalize predictions of trained models
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

```
git clone https://github.com/yourusername/TinyVisionPipeline.git
cd TinyVisionPipeline 
```

2. Install the package locally

First, activate your virtual environment.
Then install the following:
```
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
```commandline
python main.py --save_path "your_model_name"
```
4. To test a trained project:

Update the trained model name and path in CONST file:
```
LOAD_MODEL = "my_trained_model.pt"
LOAD_PATH = "my\trained\model\folder"
```
Then run:
```commandline
python eval.py
```
The evaluation results will be saved at the folder containing the model.


## 🔧 Features

✅ Modular code layout (easy to expand or replace parts)

✅ Clean separation of training, data, and model logic

✅ Supports image augmentation and checkpointing

✅ Model testing & visualization

## 🔮 Model testing and visualization Utility

The predict.py allows you to test your trained model easily by displaying predictions on images.

🔍 What it does:
- Loads your trained model.

- Runs predictions on 9 images.

- Displays a grid with:

    - The image,
    
    - The predicted label,
    
    - The confidence score (probability).

This is useful for quickly checking that your model works as expected after training!

### ⚙️ Command-line options
You have two testing modes:

| Option          | What it does                                                        |
| --------------- | ------------------------------------------------------------------- |
| `--test`        | Tests your model on 9 **random images from the CIFAR-10 test set.** |
| `--custom_test` | Tests your model on 9 **custom images** from your chosen folder.    |

### 🖼️ Custom images requirements
Images should be placed in a folder you specify.

Supported standardize formatting (.jpg, .jpeg, .png).

Make sure your custom images are compatible in size (the script will preprocess them to match the model input).

### 🚀 Usage examples

1️⃣ Test on CIFAR-10 test set:

```python prediction\predict.py --test```

2️⃣ Test on your own custom images:
```
python predict.py --custom_test path/to/your/images
```
(Replace path/to/your/images with your folder path.)

✅ Pro tip:
If your model was saved on a GPU but you're running on CPU now, no worries—predict.py automatically handles the device (CPU/GPU) for smooth running.

## 📜 License

This project is released under the MIT License.
Use, learn, remix — but give credit where due. ⚔️✨