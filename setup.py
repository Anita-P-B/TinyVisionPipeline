from setuptools import setup, find_packages

setup(
    name='tiny_vision_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'matplotlib',
        # Add other dependencies here
    ],
    author='Anita Best',
    description='A tiny but mighty image classification pipeline using PyTorch.',
)
