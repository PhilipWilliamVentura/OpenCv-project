import cv2
import matplotlib.pyplot as plt
import os
import random
import kagglehub #Dataset
import tarfile

#DOWNLOAD DATASET
dataset_path = kagglehub.dataset_download("atulanandjha/lfwpeople")
tgz_path = os.path.join(dataset_path, "lfw-funneled.tgz")
extracted_folder = os.path.join(dataset_path, "lfw_funneled")

if not os.path.exists(extracted_folder):
    print("Extracting dataset...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=dataset_path)
    print("Extraction complete.")

image_folder = extracted_folder

#SELECT RANDOM IMAGE
all_images = []
for root, dirs, files in os.walk(image_folder):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            all_images.append(os.path.join(root, file))

random_image_path = random.choice(all_images)
img = cv2.imread(random_image_path)

#FACE DETECTION
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 4)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#PLOT
plt.figure(figsize = (20, 10))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
