import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Pfad zu deinen Daten
base_dir = 'dataset'

# Bilder vorbereiten
# Skalierung der Pixelwerte von 0-255 auf 0-1 (Normalisierung)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Trainings-Daten laden (80%)
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validierungs-Daten laden (20%)
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("--- Status ---")
print("Daten erfolgreich geladen!")

# CNN-Modell definieren
model = tf.keras.models.Sequential([
    # Erste Schicht: Erkennt einfache Kanten
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Zweite Schicht: Erkennt komplexere Formen
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Dritte Schicht
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Daten flach machen für die Klassifikation
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    # Ausgangsschicht: Gibt Wahrscheinlichkeit für Tumor (0 bis 1) aus
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Modell erfolgreich erstellt!")
# Das Training starten
print("Starte Training... Bitte warten, das kann ein paar Minuten dauern.")

history = model.fit(
    train_generator,
    epochs=10,  # Die KI geht 10-mal den kompletten Datensatz durch
    validation_data=validation_generator
)

# Modell speichern
model.save('brain_tumor_model.h5')
print("Training beendet und Modell als 'brain_tumor_model.h5' gespeichert!")

import matplotlib.pyplot as plt

# Erstellen der Grafiken
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# Grafik 1: Genauigkeit
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label=' Validation Accuracy')
plt.title('Training und Validation Accuracy')
plt.xlabel('Epochen')
plt.ylabel('Genauigkeit')
plt.legend(loc='lower right')

# Grafik 2: Fehler
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training und Validation Loss')
plt.xlabel('Epochen')
plt.ylabel('Verlustwert')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()