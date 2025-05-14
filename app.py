from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model('model_arus.keras')

# Inisialisasi scaler (sama seperti saat training)
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

# Dummy fit untuk scaler agar bisa transform
# Sesuaikan dengan data kamu saat training
scaler_features.fit(np.random.rand(100, 7))  # 7 fitur: Longitude, Latitude, Velocity, year, month, day, hour
scaler_target.fit(np.random.rand(100, 1))

SEQ_LENGTH = 60

def create_sequence(input_data):
    # Ambil 60 urutan terakhir, atau isi 0 jika kurang
    if len(input_data) < SEQ_LENGTH:
        pad = np.zeros((SEQ_LENGTH - len(input_data), input_data.shape[1]))
        input_data = np.vstack((pad, input_data))
    else:
        input_data = input_data[-SEQ_LENGTH:]
    return np.expand_dims(input_data, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Ambil input dari form
            longitude = float(request.form['longitude'])
            latitude = float(request.form['latitude'])
            velocity = float(request.form['velocity'])
            year = int(request.form['year'])
            month = int(request.form['month'])
            day = int(request.form['day'])
            hour = int(request.form['hour'])

            # Susun data input sebagai array (1 baris saja, bisa dikembangkan jadi sequence)
            input_data = np.array([[longitude, latitude, velocity, year, month, day, hour]])
            scaled_input = scaler_features.transform(input_data)

            # Buat sequence (dummy sequence dari 60 input yang sama)
            sequence = np.repeat(scaled_input, SEQ_LENGTH, axis=0)
            sequence = create_sequence(sequence)

            # Prediksi
            pred = model.predict(sequence)
            prediction = scaler_target.inverse_transform(pred)[0][0]

        except Exception as e:
            prediction = f"Terjadi kesalahan: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)