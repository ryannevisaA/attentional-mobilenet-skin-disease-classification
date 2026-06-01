import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'HAM10000_metadata.csv')
images_dir = os.path.join(base_dir, 'all_images')
dataset_dir = os.path.join(base_dir, 'dataset')

# Load CSV
df = pd.read_csv(csv_path)
print(f"Total images in CSV: {len(df)}")

# Split into train / val / test (70 / 15 / 15)
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['dx'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['dx'], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Copy images to correct folders
def copy_images(dataframe, split_name):
    count = 0
    for _, row in dataframe.iterrows():
        img_name = row['image_id'] + '.jpg'
        label = row['dx']
        src = os.path.join(images_dir, img_name)
        dst_dir = os.path.join(dataset_dir, split_name, label)
        dst = os.path.join(dst_dir, img_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            count += 1
    print(f"{split_name}: {count} images copied")

copy_images(train_df, 'train')
copy_images(val_df, 'val')
copy_images(test_df, 'test')

print("\nDone! Dataset sorted successfully!")