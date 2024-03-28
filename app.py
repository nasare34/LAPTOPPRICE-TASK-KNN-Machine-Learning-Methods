from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

with open('sc.pickle', 'rb') as f:
    sc = pickle.load(f)


# Function to preprocess input data and make prediction
def predict_price(cpu, ghz, gpu, ram, ramtype, screen, storage, ssd, weight):
    # Preprocess input data
    input_data = np.array([[cpu, ghz, gpu, ram, ramtype, screen, storage, ssd, weight]])
    input_data_scaled = sc.transform(input_data)

    # Make prediction
    prediction = classifier.predict(input_data_scaled)

    return prediction


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle form submission and show prediction result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        cpu = float(request.form['CPU'])
        ghz = float(request.form['GHz'])
        gpu = float(request.form['GPU'])
        ram = float(request.form['RAM'])
        ramtype = float(request.form['RAMType'])
        screen = float(request.form['Screen'])
        storage = float(request.form['Storage'])
        ssd = float(request.form['SSD'])
        weight = float(request.form['Weight'])

        # Predict the price
        prediction = predict_price(cpu, ghz, gpu, ram, ramtype, screen, storage, ssd, weight)

        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
