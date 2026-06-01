import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(base_dir, 'dataset', 'test')

def attention_block(x):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // 16, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([x, se])

print("Loading model...")
model = tf.keras.models.load_model(
    os.path.join(base_dir, 'best_model.keras'),
    custom_objects={'attention_block': attention_block}
)
print("Model loaded!")

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

# Show 3 rows x 3 cols = 9 samples
sample_classes = ['nv', 'mel', 'bcc', 'bkl', 'akiec', 'vasc', 'df', 'nv', 'mel']
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Attentional-MobileNet — Sample Predictions on Test Images',
             fontsize=13, fontweight='bold', y=1.01)

for idx, cls in enumerate(sample_classes):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    cls_dir = os.path.join(test_dir, cls)
    img_files = os.listdir(cls_dir)
    img_file = img_files[idx % len(img_files)]
    img_path = os.path.join(cls_dir, img_file)

    # Load and predict
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_expanded, verbose=0)[0]
    pred_idx = np.argmax(preds)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = preds[pred_idx] * 100

    # Show image
    ax.imshow(img_array)
    ax.axis('off')

    color = 'green' if pred_class == cls else 'red'
    status = '✓' if pred_class == cls else '✗'

    ax.set_title(
        f'True: {CLASS_FULL[cls]}\n'
        f'Pred: {CLASS_FULL[pred_class]} {status}\n'
        f'Conf: {confidence:.1f}%',
        fontsize=7.5,
        color=color,
        fontweight='bold'
    )

    # Add colored border
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.tight_layout()
plt.savefig('gradcam_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved as gradcam_heatmap.png!")