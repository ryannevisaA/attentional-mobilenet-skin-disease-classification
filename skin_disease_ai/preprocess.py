import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'dataset', 'train')
val_dir   = os.path.join(base_dir, 'dataset', 'val')
test_dir  = os.path.join(base_dir, 'dataset', 'test')

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load & augment training data
train_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Only rescale for val and test
val_test_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create generators
train_gen = train_data.flow_from_directory(
    train_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)

val_gen = val_test_data.flow_from_directory(
    val_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)

test_gen = val_test_data.flow_from_directory(
    test_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

print("\nClass names:", train_gen.class_indices)
print("Train batches:", len(train_gen))
print("Val batches:", len(val_gen))
print("Test batches:", len(test_gen))

# Show sample images
images, labels = next(train_gen)
plt.figure(figsize=(12, 4))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.suptitle('Sample Training Images')
plt.tight_layout()
plt.savefig('sample_images.png')
plt.show()
print("\nPreprocessing done!")