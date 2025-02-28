import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

from Helper_Function import create_tensorboard_callback, plot_loss_curves

# Get the dataset
dataset_list = tfds.list_builders()

(train_data, test_data), ds_info = tfds.load(name='food101',
                                             split=['train[:80%]', 'train[80%:]'],
                                             shuffle_files=True,  # Changed: Shuffle training files
                                             as_supervised=True,
                                             with_info=True)


# Get class names  <-- FIX: Get class names from ds_info
class_names = ds_info.features["label"].names

# Preprocessing function
def preprocess_img(image, label, img_size=224):
  image = tf.image.resize(image, [img_size, img_size])
  image = tf.cast(image, tf.float16)  # Cast to float16 for mixed precision
  return image, label

# Map, shuffle, batch, and prefetch the training data
train_data = train_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Map, batch, and prefetch the test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)  # Added prefetch

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

lower_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,
                                                monitor='val_accuracy',
                                                min_lr=1e-7,
                                                patience=2,  # Changed: More reasonable patience
                                                verbose=1)

# Set mixed precision policy
mixed_precision.set_global_policy(policy='mixed_float16')

# Create base model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB1(include_top=False)

# Input and Data Augmentation (you could add data augmentation here)
inputs = layers.Input(shape=input_shape, name="input_layer")
x = base_model(inputs)  # , training=False)  <- Important if using BatchNormalization/Dropout during inference

x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = layers.Dropout(.3)(x)

x = layers.Dense(len(class_names))(x)  # Now uses the correct number of classes
outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x) #dtype to address mixed precision
model = tf.keras.Model(inputs, outputs)

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy",  # Correct loss for integer labels
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])

history = model.fit(train_data,
                    epochs=50,
                    validation_data=test_data,
                    callbacks=[create_tensorboard_callback("training-logs", "EfficientNetB1-"),
                               early_stopping,
                               lower_lr])

model.save("Food_Vision_efficientnetb1_feature_extract_model")

plot_loss_curves(history)