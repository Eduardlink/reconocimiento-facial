import streamlit as st
import cv2
from services.usuario_service import buscar_usuario_por_nombre, insertar_usuario
import os
from app.aumento_datos import data_augmentation
from app.preprocesamiento import procesar_imagenes
#from scripts.entrenamiento_modelo import entrenar_modelo
from app.entrenamiento_modelo import entrenar_modelo
from tensorflow.keras.models import load_model
import numpy as np
from app.preprocesamiento import preprocesar_imagen
import json
import h5py
from app.activations import relu, softmax


# Función para cambiar de página
def set_page(page_name):
    st.session_state["current_page"] = page_name

# Función para obtener la página actual
def get_page():
    return st.session_state.get("current_page", "login")

# Página de Login
def login_page():
    st.title("Inicio de Sesión")
    
    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Ingresar"):
            if verificar_credenciales(usuario, contraseña):
                st.success(f"¡Bienvenido, {usuario}!")
                st.session_state["usuario"] = usuario  # Almacenar el nombre del usuario
                set_page("welcome")  # Navegar a la página de bienvenida
            else:
                st.error("Usuario o contraseña incorrectos.")
    with col2:
        if st.button("Registrarse"):
            set_page("register")  # Navegar a la página de registro
    with col3:
        if st.button("Ingresar con Reconocimiento Facial"):
            reconocer_usuario()

def reconocer_usuario():
    """
    Captura una imagen con la cámara, detecta el rostro, lo procesa y lo reconoce manualmente.
    """
    st.info("Abriendo cámara para reconocimiento facial...")

    # Verificar si el modelo y el mapeo de etiquetas están disponibles
    model_path = "model/cnn_model.h5"
    if not os.path.exists(model_path):
        st.error("El modelo no está disponible. Por favor, entrena el modelo primero.")
        return

    # Cargar el modelo manualmente
    with h5py.File(model_path, "r") as f:
        W1 = f["W1"][:]
        b1 = f["b1"][:]
        W2 = f["W2"][:]
        b2 = f["b2"][:]
        class_indices = json.loads(f.attrs["class_indices"])

    label_map = {v: k for k, v in class_indices.items()}  # Invertir el mapeo

    # Cargar el modelo de detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    st.write("Presiona 'c' para capturar el rostro o 'q' para salir.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo abrir la cámara.")
            break

        # Convertir a escala de grises para la detección
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Dibujar rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Reconocimiento Facial", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capturar rostro
            if len(faces) > 0:
                # Recortar el rostro (solo el primero detectado)
                x, y, w, h = faces[0]
                rostro = frame[y:y+h, x:x+w]

                # Preprocesar el rostro
                processed_frame = preprocesar_imagen(rostro)
                processed_frame = processed_frame.flatten().reshape(1, -1)  # Aplanar

                # Forward propagation manual
                Z1 = np.dot(processed_frame, W1) + b1
                A1 = relu(Z1)
                Z2 = np.dot(A1, W2) + b2
                A2 = softmax(Z2)

                # Predicción de la clase
                predicted_label = label_map[np.argmax(A2)]
                st.success(f"Usuario reconocido: {predicted_label}")
                st.session_state["usuario"] = predicted_label
                set_page("welcome")  # Navegar a la página de bienvenida
            else:
                st.error("No se detectó ningún rostro. Intenta de nuevo.")
            break
        elif key == ord('q'):  # Salir sin capturar
            st.info("Cerrando cámara sin capturar rostro.")
            break
    cap.release()
    cv2.destroyAllWindows()


# Verificar credenciales
def verificar_credenciales(usuario, contraseña):
    if not usuario or not contraseña:
        return False
    usuario_encontrado = buscar_usuario_por_nombre(usuario)
    if usuario_encontrado and usuario_encontrado[2] == contraseña:  # Índice 2 es la contraseña
        return True
    return False

# Abrir la cámara con OpenCV
def abrir_camara():
    st.info("Abriendo cámara para reconocimiento facial...")
    
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    st.write("Presiona 'q' para salir.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo abrir la cámara.")
            break
        # Mostrar el video en una ventana
        cv2.imshow("Reconocimiento Facial", frame)
        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()


# Función para capturar imagen

def capturar_imagen(usuario):
    """
    Abre la cámara, captura una imagen, detecta el rostro y lo guarda.
    """
    carpeta_usuario = f"users/{usuario}"
    os.makedirs(carpeta_usuario, exist_ok=True)  # Crear la carpeta si no existe
    ruta_imagen = os.path.join(carpeta_usuario, "rostro.jpg")  # Nombre de la foto
    
    # Cargar el modelo de detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    cap = cv2.VideoCapture(0)
    st.write("Presiona 'c' para capturar el rostro o 'q' para salir.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo abrir la cámara.")
            break
        
        # Convertir a escala de grises para la detección
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Dibujar rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Captura de Rostro", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capturar imagen
            if len(faces) > 0:
                # Extraer el rostro (solo el primero detectado)
                x, y, w, h = faces[0]
                rostro = frame[y:y+h, x:x+w]
                cv2.imwrite(ruta_imagen, rostro)  # Guardar el rostro recortado
                st.success(f"Rostro capturado y guardado en {ruta_imagen}")
            else:
                st.error("No se detectó ningún rostro. Intenta de nuevo.")
            break
        elif key == ord('q'):  # Salir sin capturar
            st.info("Cerrando cámara sin capturar rostro.")
            break
    cap.release()
    cv2.destroyAllWindows()
    return ruta_imagen if os.path.exists(ruta_imagen) else None

# Página de Registro
def register_page():
    st.title("Registro de Usuario")
    
    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")
    confirmar_contraseña = st.text_input("Confirmar Contraseña", type="password")
    
    if "ruta_imagen" not in st.session_state:
        st.session_state["ruta_imagen"] = None
    
    if st.button("Capturar Imagen"):
        if usuario:
            st.session_state["ruta_imagen"] = capturar_imagen(usuario)
        else:
            st.error("Por favor, ingresa un nombre de usuario antes de capturar la imagen.")
    
    if st.button("Registrarse"):
        if contraseña != confirmar_contraseña:
            st.error("Las contraseñas no coinciden.")
        elif not usuario or not contraseña:
            st.error("Todos los campos son obligatorios.")
        elif not st.session_state["ruta_imagen"]:
            st.error("Debes capturar una imagen antes de registrarte.")
        else:
            try:
                # Ruta de la carpeta del usuario
                user_dir = f"users/{usuario}"
                
                # Aumentar datos con Data Augmentation
                data_augmentation(st.session_state["ruta_imagen"], user_dir)
                
                # Procesar imágenes aumentadas
                procesar_imagenes(user_dir, user_dir)
                
                # Entrenar modelo CNN con todas las imágenes
                entrenar_modelo("users", "model/cnn_model.h5")
                
                # Guardar información del usuario en la base de datos
                insertar_usuario(usuario, contraseña, user_dir)
                
                st.success("Usuario registrado, datos aumentados y modelo actualizado con éxito.")
                st.session_state["ruta_imagen"] = None  # Reiniciar la imagen
                set_page("login")  # Volver al login
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Botón para regresar al Login
    if st.button("Volver al Login"):
        set_page("login")
# Página de Bienvenida
def welcome_page():
    usuario = st.session_state.get("usuario", "Usuario")
    st.title(f"Bienvenido, {usuario}")
    
    if st.button("Cerrar Sesión"):
        del st.session_state["usuario"]  # Eliminar el usuario del estado
        set_page("login")  # Volver al login

# Renderizar la Página Actual
def render_page():
    current_page = get_page()
    
    if current_page == "login":
        login_page()
    elif current_page == "register":
        register_page()
    elif current_page == "welcome":
        welcome_page()
    else:
        st.error("Página no encontrada.")

# Configuración inicial de la aplicación
if __name__ == "__main__":
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "login"  # Página inicial
    render_page()
