# Brain Tumor Detection AI
Hausarbeit für das Modul KI Grundlagen und Plattformen.

##  Projektbeschreibung
Dieses Programm nutzt ein Convolutional Neural Network (CNN), um Hirntumore in MRT-Bildern zu klassifizieren. Es wurde mit TensorFlow/Keras entwickelt.

##  Anleitung Test-Modus
Da die trainierte Modell-Datei zu groß für GitHub ist, folgen Sie bitte diesen Schritten, um die KI zu testen:

1.  **Modell herunterladen:** Laden Sie die Datei `brain_tumor_model.h5` hier herunter: 
   https://drive.google.com/file/d/1ycdbO9qZNz8RowQPg0KIrREQGzOCQLnw/view?usp=sharing

2. **Datei platzieren:** Speichern Sie die heruntergeladene Datei `brain_tumor_model.h5` direkt in den Hauptordner dieses Projekts.

3. **Abhängigkeiten installieren:**
   Führen Sie im Terminal aus: `pip install -r requirements.txt`

4. **Vorhersage starten:**
   Führen Sie das Skript `predict.py` aus, um ein Testbild zu klassifizieren.

## Datenquelle
Die Bilder stammen vom Kaggle Datensatz: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/deeppythonist/brain-tumor-mri-dataset?resource=download) (Mohammad rasol esfandiari). 
