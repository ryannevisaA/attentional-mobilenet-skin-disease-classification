import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(base_dir, 'dataset', 'test')
train_dir = os.path.join(base_dir, 'dataset', 'train')
val_dir = os.path.join(base_dir, 'dataset', 'val')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(test_dir, target_size=IMG_SIZE,
            batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
train_gen = datagen.flow_from_directory(train_dir, target_size=IMG_SIZE,
            batch_size=BATCH_SIZE, class_mode='categorical')
val_gen = datagen.flow_from_directory(val_dir, target_size=IMG_SIZE,
            batch_size=BATCH_SIZE, class_mode='categorical')

# ── Model 1 - Plain MobileNetV2 ──────────────────────────────
def build_plain_mobilenet(num_classes=7):
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input(shape=(224,224,3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# ── Model 2 - VGG16 ──────────────────────────────────────────
def build_vgg16(num_classes=7):
    base = VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input(shape=(224,224,3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# ── Model 3 - ResNet50 ───────────────────────────────────────
def build_resnet50(num_classes=7):
    base = ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input(shape=(224,224,3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# ── Model 4 - Attentional MobileNet ─────────────────────────
def attention_block(x):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // 16, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([x, se])

def build_attentional_mobilenet(num_classes=7):
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input(shape=(224,224,3))
    x = base(inputs, training=False)
    x = attention_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# ── Train and evaluate each model ───────────────────────────
results = {}

models_to_train = [
    ('Plain MobileNetV2', build_plain_mobilenet()),
    ('VGG16', build_vgg16()),
    ('ResNet50', build_resnet50()),
]

for name, model in models_to_train:
    print(f"\nTraining {name}...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_gen, epochs=5,
              validation_data=val_gen, verbose=1)
    # Reset test generator
    test_gen_fresh = datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)
    loss, acc = model.evaluate(test_gen_fresh, verbose=0)
    results[name] = acc * 100
    print(f"{name} Test Accuracy: {acc*100:.2f}%")

# Load your trained model
print("\nLoading Attentional-MobileNet...")
att_model = tf.keras.models.load_model(
    os.path.join(base_dir, 'best_model.keras'),
    custom_objects={'attention_block': attention_block}
)
att_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
test_gen_fresh = datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)
loss, acc = att_model.evaluate(test_gen_fresh, verbose=0)
results['Attentional-MobileNet\n(Proposed)'] = acc * 100
print(f"Attentional-MobileNet Accuracy: {acc*100:.2f}%")

# ── Plot comparison ──────────────────────────────────────────
print("\nAll Results:")
for name, acc in results.items():
    print(f"  {name}: {acc:.2f}%")

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
names = list(results.keys())
accs = list(results.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(names, accs, color=colors,
               width=0.5, edgecolor='black')
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{acc:.2f}%',
             ha='center', fontsize=11,
             fontweight='bold')
plt.title('Model Comparison - Test Accuracy on HAM10000',
          fontsize=14)
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("\nComparison complete!")