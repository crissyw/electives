import os

from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import numpy as np
import tflite_runtime.interpreter as tflite

from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads/'
PREDICTION_THRESHOLD = .4
COMPARISON_ITEM = 'Gojek Driver'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):  #no matter how complicated or malicious the filename is, the secure_filename() function reduces it to a flat filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        prediction = process_file(filepath)
        return render_template('index.html', filename=filename, prediction=prediction)
    else:
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

def process_file(filepath):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    interpreter = tflite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'] # (1, 224, 224, 3)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    input_data = np.ndarray(shape=input_shape, dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(filepath)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = input_shape[1:3] # (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    input_data[0] = normalized_image_array

    # run the inference
    # prediction = model.predict(data) #[[0.0000388  0.99996126]] <class 'numpy.ndarray'>
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data

    # format as percent
    prediction_text = truncate(prediction.item(0)*100,5)

    if prediction.item(0) > PREDICTION_THRESHOLD:
        return "Yay! {}% a {}!".format(prediction_text, COMPARISON_ITEM)
    else:
        return "{}% NOT a {}!".format(prediction_text, COMPARISON_ITEM)

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


app.run(host='0.0.0.0', port=8080)
