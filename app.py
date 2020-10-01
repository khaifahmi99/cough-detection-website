from flask import Flask, render_template, request, url_for, redirect, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import boto3
import botocore
from boto3.dynamodb.conditions import Key
import librosa
import requests
import pandas as pd
import datetime as dt
import json
from decimal import Decimal
import decimal

import os
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = "523541653"
app.config['UPLOAD_FOLDER'] = 'static/assets/'
CORS(app)

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}
ALLOWED_EXTENSIONS_SOUND = {'wav', 'mp3'}

@app.route('/charts')
def chart():
    return render_template('charts.html')

# home page of the server
@app.route('/data/<id>')
def home(id):
    print(id)
    data=None
    table = boto3.resource('dynamodb', region_name='ap-southeast-2').Table('cough-detection')
    item = table.query(
        KeyConditionExpression=Key('node-id').eq(id)
    )

    item = item['Items'][0]

    df = pd.DataFrame(columns=['date', 'confidence'])
    for o in item['status_history']:
        df = df.append({
            'date': o['ts'],
            'confidence': o['confidence']
        }, ignore_index=True)

    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M:%S')
    df['confidence'] = df['confidence'].astype(int)
    df = df[df['date']>=(dt.datetime.now()-dt.timedelta(hours=12))]
    df = df.groupby(pd.Grouper(key='date',freq='H')).count()

    chart_x = []
    chart_y = []
    for i, row in df[-8:].iterrows():
        chart_x.append(i)
        chart_y.append(str(row['confidence']))

    obj = {
        'node_id': item['node-id'],
        'metadata': {
            'area': item['metadata']['area'],
            'aisle': item['metadata']['aisle'],
            'area_type': item['metadata']['type'],
        },
        'history': item['status_history'],
        'chart_x': chart_x,
        'chart_y': chart_y,
    }

    print(obj)
    return jsonify(obj)

# form to upload the image
@app.route('/upload')
def upload_image():
    return render_template('upload.html', error=None)

# form to upload sound
@app.route('/uploadSound')
def upload_sound():
    return render_template('upload_sound.html', error=None)

# process the uploaded sound
# POST only
@app.route('/processSound', methods=['POST'])
def process_sound():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload_sound.html', error="No file uploaded")
        
        file = request.files['file']

        if file.filename == '':
            return render_template('upload_sound.html', error="No selected file")

        # check if the uploaded file in in accepted format
        if file and allowed_sound_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # save image to server

            score, confidence = get_sound_score(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if score == 1:
                label = 'Coughing'
                confidence = round(confidence * 100, 2)
            else:
                label = 'Not Coughing'
                confidence = 100 - round(confidence * 100, 2)

            return render_template('upload_sound.html', filename=filename, pred={'label': label, 'confidence': confidence})
        else:
            return render_template('upload_sound.html', error="Allowed image types are -> mp3, wav")

        return render_template('upload_sound.html', error="There is an error processing your uploaded file. Try again or try with another sound file")

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
            if score == 1:
                label = 'Coughing'
                confidence = round(confidence * 100, 2)
            else:
                label = 'Not Coughing'
                confidence = 100 - round(confidence * 100, 2)

            return render_template('upload.html', filename=filename, pred={'label': label, 'confidence': confidence})
        else:
            return render_template('upload.html', error="Allowed image types are -> png, jpg, jpeg")

        return render_template('upload.html', error="There is an error processing your uploaded file. Try again or try with another image")

@app.route('/inference')
def inference():
    url = request.args.get('url')
    media = request.args.get('type')

    if url is None or media is None:
        return {'status': 'Error', 'msg': 'request must include url and type parameters'}

    if media == 'image':
        try:
            # download the image from the given url
            response = requests.get(url)

            file = open(os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg'), "wb")
            file.write(response.content)
            file.close()
            
            # do prediction based on stored image
            score, confidence = get_score(os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg'))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg'))
            if score == 1:
                label = 'Coughing'
                confidence = round(confidence * 100, 2)
                status = 'OK'
            else:
                label = 'Not Coughing'
                confidence = 100 - round(confidence * 100, 2)
                status = 'OK'

            response = {
                'label': label,
                'confidence': confidence,
                'status': status
            }
        except Exception as e:
            return {'status': 'Error', 'msg': e}

    elif media == 'sound':
        try:
            # download the image from the given url
            response = requests.get(url)

            file = open(os.path.join(app.config['UPLOAD_FOLDER'], 'test.wav'), "wb")
            file.write(response.content)
            file.close()
            
            # do prediction based on stored image
            score, confidence = get_sound_score(os.path.join(app.config['UPLOAD_FOLDER'], 'test.wav'))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'test.wav'))
            if score == 1:
                label = 'Coughing'
                confidence = round(confidence * 100, 2)
                status = 'OK'
            else:
                label = 'Not Coughing'
                confidence = 100 - round(confidence * 100, 2)
                status = 'OK'

            response = {
                'label': label,
                'confidence': confidence,
                'status': status
            }
        except Exception as e:
            return {'status': 'Error', 'msg': e}

    else:
        return {'status': 'Error', 'msg': 'type value must be either sound or image'}

    
    return response

# used for displaying image
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='images/' + filename), code=301)

# used to check the allowed file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# used to check the allowed file type
def allowed_sound_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_SOUND

# get the model and calculate the score
def get_score(file_path):
    model = load_cnn_model()
    im = get_image_array(file_path)
    score = model.predict_classes(im)
    percentage = model.predict(im)
    print(score, percentage)
    return score[0][0], percentage[0][0]

# load the ML model from the saved file
def load_cnn_model():
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

# get the model and calculate the score
def get_sound_score(file_path):
    model = load_sound_model()
    sound = convert_sound(file_path)
    score = model.predict_classes(sound)
    percentage = model.predict(sound)
    print(score, percentage)
    return score[0][0], percentage[0][0]

# load the ML model from the saved file
def load_sound_model():
    # load json and create model
    file = open('models/sound.dms', 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)

    # load weights
    model.load_weights('models/sound-weights.hdf5')
    return model

# convert the image to acceptable format
def convert_sound(filename):
    y, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T,axis=0)
    mfccs = mfccs.reshape((1, 10, 4, 1))
    print('Sound Shape: ', mfccs.shape)
    return mfccs

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')