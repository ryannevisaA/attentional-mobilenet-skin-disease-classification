import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

base_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(base_dir, 'dataset', 'test')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

def attention_block(x):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // 16, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([x, se])

model = tf.keras.models.load_model(
    os.path.join(base_dir, 'best_model.keras'),
    custom_objects={'attention_block': attention_block}
)

class_names = list(test_gen.class_indices.keys())
n_classes = len(class_names)

# Get predictions
print("Getting predictions...")
y_pred_prob = model.predict(test_gen, verbose=1)
y_true = test_gen.classes

# Binarize labels
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

# Plot ROC curve for each class
plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange',
          'purple', 'brown', 'pink']

auc_scores = []
for i, (cls, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{cls} (AUC = {roc_auc:.3f})')

# Macro average AUC
macro_auc = np.mean(auc_scores)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5,
         label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curves - Attentional-MobileNet\nMacro Average AUC = {macro_auc:.3f}',
          fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

print("\nAUC Scores per class:")
for cls, score in zip(class_names, auc_scores):
    print(f"  {cls}: {score:.3f}")
print(f"\nMacro Average AUC: {macro_auc:.3f}")
print("\nROC curve saved!")