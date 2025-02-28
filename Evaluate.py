import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
import seaborn as sn
from tensorflow.keras import layers

datasets_list = tfds.list_builders()
target_dataset = "food101"
print(f"'{target_dataset}' in TensorFlow Datasets: {target_dataset in datasets_list}")

# Load the test data
(train_data, test_data), ds_info = tfds.load(
    name="food101",
    split=["train[:1%]", "train[1%:]"],
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
)
class_names = ds_info.features["label"].names

# Load trained model
model_path = "Food_Vision_efficientnetb1_feature_extract_model"
try:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model not found at: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Extract and preprocess images
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    return image, label

test_data = test_data.map(preprocess_image)

test_images = test_data.map(lambda image, label: image)
test_images_batch = test_images.batch(32)

pred_probs = model.predict(test_images_batch, verbose=1)
pred_classes = pred_probs.argmax(axis=1)

# Correctly extract and convert y_labels
y_labels = tf.concat([label for image, label in test_data], axis=0).numpy()

from sklearn.metrics import accuracy_score

sklearn_acc = accuracy_score(y_labels, pred_classes)
print(sklearn_acc)

cm = tf.math.confusion_matrix(y_labels, pred_classes)

plt.figure(figsize=(200, 200))
sn.heatmap(cm, annot=False, fmt='', cmap='Blues')
plt.show()
