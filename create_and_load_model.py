import os
import pickle
from sklearn.svm import SVC
import numpy as np

model_path = 'model/gender_model.pkl'

if not os.path.exists(model_path):
    # buat dummy model
    X_dummy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_dummy = ['laki-laki', 'perempuan', 'laki-laki', 'perempuan']
    dummy_model = SVC(probability=True)
    dummy_model.fit(X_dummy, y_dummy)

    os.makedirs('model', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(dummy_model, f)

# load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# model sudah siap digunakan untuk prediksi
print("Model berhasil dimuat dan siap digunakan.")
