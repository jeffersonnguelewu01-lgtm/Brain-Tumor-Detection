import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Modell laden
model = tf.keras.models.load_model('brain_tumor_model.h5')


def predict_tumor(img_path):
    # Bild laden und für die KI vorbereiten
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (150, 150))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Vorhersage treffen
    prediction = model.predict(img_array)[0][0]

    # Ergebnis anzeigen
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if prediction > 0.5:
        plt.title(f"Diagnose: TUMOR ERKANNT ({prediction * 100:.2f}%)")
    else:
        plt.title(f"Diagnose: GESUND ({(1 - prediction) * 100:.2f}%)")
    plt.axis('off')
    plt.show()

# Bild aus dem Test-Ordner testen
predict_tumor('test2.jpg')