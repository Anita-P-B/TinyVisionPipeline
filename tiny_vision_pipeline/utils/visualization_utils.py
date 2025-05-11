import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os



def plot_predictions(images, preds, confidences, class_names, model_dir):

    plt.figure(figsize=(12, 8))
    for idx in range(len(preds)):
        img = images[idx]
        pred = preds[idx]

        plt.subplot(3, 3, idx + 1)
        img_disp = img.permute(1, 2, 0).numpy()
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())  # normalized for display

        plt.imshow(img_disp)
        conf = confidences[idx] * 100
        plt.title(f"Pred: {class_names[pred]} (conf:{conf:.1f}%)")
        plt.axis('off')

    plt.suptitle(f"Predictions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
    plt.close()

