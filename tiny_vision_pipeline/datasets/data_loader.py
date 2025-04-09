
import torchvision
import os
import numpy as np
# Load CIFAR-10 from torchvision (uses same underlying data)
def load_datasets():

    data_root = './data'
    cifar_folder = os.path.join(data_root, 'cifar-10-batches-py')

    download = not os.path.exists(cifar_folder)

    train, test = (
        torchvision.datasets.CIFAR10(root=data_root, train=True, download=download),
        torchvision.datasets.CIFAR10(root=data_root, train=False, download=download)
    )

    x_train_np = train.data
    y_train_np = np.array(train.targets)
    x_test_np = test.data
    y_test_np = np.array(test.targets)


    return x_train_np, y_train_np, x_test_np, y_test_np