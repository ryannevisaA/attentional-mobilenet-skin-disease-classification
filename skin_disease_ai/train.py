import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'dataset', 'train')
val_dir   = os.path.join(base_dir, 'dataset', 'val')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

train_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_data.flow_from_directory(
    train_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = val_data.flow_from_directory(
    val_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)

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

    # Phase 1 — freeze all
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = attention_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs), base_model

model, base_model = build_model()

# Phase 1 — Train with frozen base
print("Phase 1 - Training with frozen MobileNetV2...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase1 = [
    ModelCheckpoint(
        'best_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    EarlyStopping(
        patience=5,
        monitor='val_accuracy',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

history1 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=callbacks_phase1
)

print(f"Phase 1 best val accuracy: {max(history1.history['val_accuracy'])*100:.2f}%")

# Phase 2 — Unfreeze last 30 layers and fine tune
print("\nPhase 2 - Fine tuning last 30 layers...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Lower learning rate for fine tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    ModelCheckpoint(
        'best_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    EarlyStopping(
        patience=7,
        monitor='val_accuracy',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        verbose=1
    )
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_phase2
)

print(f"Phase 2 best val accuracy: {max(history2.history['val_accuracy'])*100:.2f}%")

# Combine histories
combined_acc = history1.history['accuracy'] + history2.history['accuracy']
combined_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
combined_loss = history1.history['loss'] + history2.history['loss']
combined_val_loss = history1.history['val_loss'] + history2.history['val_loss']

# Plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(combined_acc, label='Train Accuracy', color='blue')
plt.plot(combined_val_acc, label='Val Accuracy', color='orange')
plt.axvline(x=len(history1.history['accuracy']),
            color='red', linestyle='--',
            label='Fine tuning starts')
plt.title('Model Accuracy - Two Phase Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(combined_loss, label='Train Loss', color='blue')
plt.plot(combined_val_loss, label='Val Loss', color='orange')
plt.axvline(x=len(history1.history['loss']),
            color='red', linestyle='--',
            label='Fine tuning starts')
plt.title('Model Loss - Two Phase Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
plt.show()

print("\nTwo phase training complete!")
print(f"Phase 1 best: {max(history1.history['val_accuracy'])*100:.2f}%")
print(f"Phase 2 best: {max(history2.history['val_accuracy'])*100:.2f}%")
