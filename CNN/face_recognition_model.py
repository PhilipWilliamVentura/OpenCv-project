import os
import tarfile
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import kagglehub

# Download dataset
dataset_path = kagglehub.dataset_download("atulanandjha/lfwpeople")
tgz_path = os.path.join(dataset_path, "lfw-funneled.tgz")
extracted_folder = os.path.join(dataset_path, "lfw_funneled")  # Corrected folder name

# Extract if not already extracted
if not os.path.exists(extracted_folder):
    print("Extracting dataset...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=dataset_path, filter="data")  # Fix for Python 3.14
    print("Extraction complete.")

# Verify extraction
print("Extracted contents:", os.listdir(dataset_path))

# Use the extracted dataset as the training directory
train_images = extracted_folder

# Image Augmentation
train_gen = ImageDataGenerator(
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
test_gen = ImageDataGenerator()

# Generating training data
training_data = train_gen.flow_from_directory(
    train_images, 
    target_size=(100, 100),
    batch_size=30,
    class_mode='categorical'
)

# Generating test data
testing_data = test_gen.flow_from_directory(
    train_images, 
    target_size=(100, 100),
    batch_size=30,
    class_mode='categorical'
)

# Printing class labels for each face
print(testing_data.class_indices)
