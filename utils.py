import zipfile
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def extract_data(_dir):
    for pic_type in ['color', 'grayscale', 'segmented']:
        with zipfile.ZipFile(f'{_dir}/{pic_type}.zip') as zf:
            zf.extractall(f'{_dir}/imgs/')

def preprocess(data, scaler):
    if data is None:  # val/test can be empty
        return None
    # Read image files to pytorch dataset (only temporary).
    transform = transforms.Compose([
        transforms.Resize(64), 
        transforms.CenterCrop(64), 
        transforms.ToTensor()
    ])
    data = datasets.ImageFolder(data, transform=transform)

    # Convert to numpy arrays.
    images_shape = (len(data), *data[0][0].shape)
    images = np.zeros(images_shape)
    labels = np.zeros(len(data))
    for i, (image, label) in enumerate(data):
        images[i] = image
        labels[i] = label

    # Flatten.
    images = images.reshape(len(images), -1)

    # Scale to mean 0 and std 1.
    scaler.fit(images)
    images = scaler.transform(images)

    # Shuffle train set.
    images, labels = sklearn.utils.shuffle(images, labels)

    return [images, labels]

def plot_confusion_matrix(ax, cm, font_size=14, model=''):
    ax.set_title(f'Confusion Matrix for {model}', fontsize=font_size)
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=range(3))
    ax.yaxis.set(ticks=range(3))
    ax.set_ylim(2.5, -0.5)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    ax.set_xlabel('Predicted', fontsize=font_size)
    ax.set_ylabel('True', fontsize=font_size)

def evaluate(model, X_train, y_train, X_test, y_test):
    results = {}
    y_pred = model.predict(X_test)
    results['train_score'] = model.score(X_train, y_train)
    results['test_score'] = model.score(X_test, y_test)
    results['conf_m'] = confusion_matrix(y_test, y_pred)
    results['report'] = classification_report(y_test, y_pred, output_dict=True)
    return results

def table_results(results):
    from matplotlib.font_manager import FontProperties
    plt.rcParams["figure.autolayout"] = True
    columns = ('Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall')
    
    for i, dataset in enumerate(results):
        fig, ax = plt.subplots(figsize=(10, 5))
        rows = []
        for model, res in results[dataset].items():
            macro_avg = res['report']['macro avg']
            rows.append((model, np.round(res['test_score'], 4), np.round(
                macro_avg['f1-score'], 4), np.round(macro_avg['precision'], 4), np.round(macro_avg['recall'], 4)))
        rows = sorted(rows, key=lambda x: np.mean(x[1:]), reverse=True)
        table = ax.table(loc='left', cellLoc='left', colLoc='left', cellText=rows, colLabels=columns)

        ax.set_title(f'{dataset} Dataset', fontsize=14, loc='left', pad=0)
        
        # style
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.auto_set_column_width(col=range(len(rows)))

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontproperties=FontProperties(weight='bold', size=14))

        ax.axis('off')
        fig.tight_layout()
        fig.savefig(f'plots/{dataset}_table_results.png', bbox_inches='tight')

    plt.show()