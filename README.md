# 🐉 TinyVisionPipeline
A simple, modular deep learning pipeline for image classification using PyTorch and the CIFAR-10 dataset.

This project serves as a template and demonstration for building a clean computer vision pipeline — from data loading and preprocessing to model training, evaluation, and saving.

⚠️ This is a training-focused version. No pretrained model is included (yet). Future updates may provide model export and prediction tools.

## 📁 Project Structure
```
TinyVisionPipeline/
├── tiny_vision_pipeline/
│   ├── datasets/        # Data loading and preprocessing
│   ├── model/           # Model architecture
│   ├── train/           # Training logic and checkpointing
│   ├── evaluate/        # Evaluation and test performance
│   └── __init__.py
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

First, ctivate your virtual environment.
Then install the follows:
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
python main.py
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
The evaluation results will be saved at the models containing folder.


## 🔧 Features

✅ Modular code layout (easy to expand or replace parts)

✅ Clean separation of training, data, and model logic

✅ Supports image augmentation and checkpointing

✅ Ready to extend with evaluation scripts or pretrained models


📜 License

This project is released under the MIT License.
Use, learn, remix — but give credit where due. ⚔️✨