import os
import cv2
import tensorflow as tf
import numpy as np
import json
from collections import Counter

MODEL_PATH = "scripts/modelo_reconocimiento_mejorado.h5"
CLASS_INDICES_PATH = "scripts/class_indices.json"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def cargar_modelo():
    """
    Carga el modelo y los índices de clase si existen.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError("El modelo o los índices de clase no existen. Por favor, registre un usuario y entrene el modelo.")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}  # Invertir el diccionario

    return model, class_labels

def reconocer_usuario_streamlit():
    """
    Detección en tiempo real con estabilización de predicciones.
    """
    import streamlit as st
    try:
        model, class_labels = cargar_modelo()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    stframe = st.empty()  # Contenedor para mostrar el video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara. Verifica los permisos.")
        return

    predicciones_recientes = []  # Para realizar votación por mayoría

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error al capturar el frame de la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (150, 150))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            predictions = model.predict(face_input)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            if confidence > 0.8:
                predicciones_recientes.append(class_labels[predicted_class])
                if len(predicciones_recientes) > 10:  # Mantener solo las últimas 10 predicciones
                    predicciones_recientes.pop(0)

                usuario_mas_frecuente = Counter(predicciones_recientes).most_common(1)[0][0]
                label = f"{usuario_mas_frecuente}: {confidence * 100:.2f}%"
            else:
                label = "Usuario no reconocido"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
