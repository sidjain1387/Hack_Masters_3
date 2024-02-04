from flask import Flask, render_template, request, send_from_directory
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import os

app = Flask(__name__)

# Load your trained model
cnn = load_model('model.h5')

# Define class indices
class_indices = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1.html')
def index1():
    return render_template('index1.html')

@app.route('/form.html')
def index2():
    return render_template('form.html')

@app.route('/abc.html')
def index3():
    return render_template('abc.html')

@app.route('/emergency.html')
def index4():
    return render_template('emergency.html')

@app.route('/about_us.html')
def index5():
    return render_template('about_us.html')

@app.route('/indicators.html')
def index6():
    return render_template('indicators.html')

@app.route('/stress.html')
def index7():
    return render_template('stress.html')

@app.route('/scmh.html')
def index8():
    return render_template('scmh.html')

@app.route('/mvf.html')
def index9():
    return render_template('mvf.html')

@app.route('/blogs.html')
def index10():
    return render_template('blogs.html')

@app.route('/dental.html')
def index11():
    return render_template('dental.html')

@app.route('/brusis.html')
def index12():
    return render_template('brusis.html')

@app.route('/scratch.html')
def index13():
    return render_template('scratch.html')

@app.route('/snake.html')
def index14():
    return render_template('snake.html')

@app.route('/diet.html')
def index15():
    return render_template('diet.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    img_bytes = file.read()  # Read the file bytes
    img = Image.open(io.BytesIO(img_bytes))  # Open the image using BytesIO
    
    img = img.resize((64, 64))  # Resize the image to match the input size expected by your model
    
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    
    result = cnn.predict(img)
    predicted_class_index = np.argmax(result)
    predicted_class_name = class_indices[predicted_class_index]
    
    return predicted_class_name

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
