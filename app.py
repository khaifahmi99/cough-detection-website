from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}
app.config['UPLOAD_FOLDER'] = 'static/images/'

# home page of the server
@app.route('/')
def hello():
    return "Welcome to the Cough Detection Server"

# form to upload the image
@app.route('/upload')
def upload_image():
    return render_template('upload.html', error=None)

# process the uploaded image
# POST only
@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', error="No file uploaded")
        
        file = request.files['file']

        if file.filename == '':
            return render_template('upload.html', error="No selected file")

        # check if the uploaded file in in accepted format
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # save image to server

            score, confidence = get_score(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            confidence = round(confidence * 100, 2)
            if score == 1:
                label = 'Coughing'
            else:
                label = 'Not Coughing'

            return render_template('upload.html', filename=filename, pred={'label': label, 'confidence': confidence})
        else:
            return render_template('upload.html', error="Allowed image types are -> png, jpg, jpeg")

        return render_template('upload.html', error="There is an error processing your uploaded file. Try again or try with another image")

# used for displaying image
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='images/' + filename), code=301)

# used to check the allowed file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_score(file_path):
    model = load_model()
    im = get_image_array(file_path)
    score = model.predict_classes(im)
    percentage = model.predict(im)
    print(score, percentage)
    return score[0][0], percentage[0][0]

# load the ML model from the saved file
def load_model():
    # load json and create model
    file = open('models/customCNN.dms', 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)

    # load weights
    model.load_weights('models/customCNN-weights.hdf5')
    return model

# convert the image to acceptable format
def get_image_array(filename, image_size=64):
    im = image.load_img(filename, target_size=(image_size, image_size, 3))
    im = image.img_to_array(im)
    im = im/255
        
    im = np.array(im)
    im = im.reshape((1, 64, 64, 3))
    print('Image Shape: ', im.shape)
    return im

if __name__ == '__main__':
    app.run(debug=True)