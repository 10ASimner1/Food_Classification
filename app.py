from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from Helper_Function import load_and_prep_image
import tensorflow_datasets as tfds

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

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

# Load class names from Food101 dataset
datasets_list = tfds.list_builders()
target_dataset = "food101"
if target_dataset in datasets_list:
    _, ds_info = tfds.load(
        name="food101",
        split=["train[:1%]"],  # Only need info, not actual data
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    class_names = ds_info.features["label"].names
else:
    print(f"Dataset '{target_dataset}' not found.")
    class_names = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path):
    if model is None or class_names is None:
        return "Model or class names not loaded. Cannot predict.", None

    img = load_and_prep_image(image_path, scale=False)
    pred_prob = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[pred_prob.argmax()]
    probability = pred_prob.max()
    return pred_class, probability

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                print(f"Saving file to: {file_path}")
                file.save(file_path)
                print(f"Predicting file from: {file_path}")
                prediction, probability = predict_image(file_path)
                relative_path = os.path.join('uploads', filename).replace('\\', '/') #force forward slash
                print(f"Relative path sent to template: {relative_path}")
                return render_template('result.html', prediction=prediction, probability=probability, image_path=relative_path)
            except Exception as e:
                print(f"An error occurred: {e}")
                return "An error occurred while processing the image."
        else:
            return "Invalid file type. Please upload an image (png, jpg, jpeg)."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)