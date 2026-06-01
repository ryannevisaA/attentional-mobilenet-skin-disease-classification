import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
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

def build_model(num_classes=7):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = attention_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

model = build_model()
model.load_weights('best_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Show sample predictions with images
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Attentional-MobileNet Predictions', fontsize=14)

sample_count = 0
for class_name in ['nv', 'mel', 'bcc']:
    class_dir = os.path.join(test_dir, class_name)
    img_files = os.listdir(class_dir)[:3]
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(class_dir, img_file)
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_expanded, verbose=0)
        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        row = sample_count // 3
        col = sample_count % 3
        axes[row, col].imshow(img_array)
        color = 'green' if pred_class == class_name else 'red'
        axes[row, col].set_title(
            f'True: {class_name}\nPred: {pred_class} ({confidence:.1f}%)',
            color=color, fontsize=9)
        axes[row, col].axis('off')
        sample_count += 1

plt.tight_layout()
plt.savefig('gradcam_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization saved as gradcam_results.png!")