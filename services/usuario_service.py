import os
import mysql.connector
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Configuración de conexión a MySQL
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "reconocimiento_facial"
}

def buscar_usuario_por_nombre(nombre):
    """
    Busca un usuario en la base de datos por su nombre.
    """
    try:
        conexion = mysql.connector.connect(**DB_CONFIG)
        cursor = conexion.cursor()

        query = "SELECT * FROM usuarios WHERE usuario = %s"
        cursor.execute(query, (nombre,))
        resultado = cursor.fetchone()

        cursor.close()
        conexion.close()
        return resultado
    except mysql.connector.Error as e:
        print(f"Error al conectar con MySQL: {e}")
        return None

def registrar_usuario(nombre, contraseña, url_carpeta):
    """
    Registra un nuevo usuario en la base de datos con la ruta de su carpeta.
    """
    try:
        conexion = mysql.connector.connect(**DB_CONFIG)
        cursor = conexion.cursor()

        # Inserción en la tabla usuarios
        query = """
        INSERT INTO usuarios (usuario, contraseña, url_carpeta) 
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (nombre, contraseña, url_carpeta))
        conexion.commit()

        cursor.close()
        conexion.close()
        return True
    except mysql.connector.Error as e:
        print(f"Error al registrar el usuario: {e}")
        return False

def generar_aumento_datos(user_dir):
    """
    Genera imágenes aumentadas para el usuario registrado.
    """
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for imagen in os.listdir(user_dir):
        imagen_path = os.path.join(user_dir, imagen)
        if os.path.isfile(imagen_path):
            img = Image.open(imagen_path)
            img = img.resize((150, 150))
            img_array = np.array(img).reshape((1, 150, 150, 3))

            i = 0
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=user_dir, save_prefix="aug", save_format="jpg"):
                i += 1
                if i >= 10:  # Generar 10 imágenes aumentadas
                    break

def registro_page():
    st.title("Registro de Usuario")
    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")
    imagen_usuario = st.file_uploader("Sube tu imagen de rostro", type=["jpg", "png"])

    if st.button("Registrar"):
        if not usuario or not contraseña or not imagen_usuario:
            st.error("Por favor, complete todos los campos y suba una imagen.")
            return

        # Crear directorio del usuario
        user_dir = os.path.join("users", usuario)
        os.makedirs(user_dir, exist_ok=True)

        # Guardar la imagen en el directorio del usuario
        ruta_imagen = os.path.join(user_dir, imagen_usuario.name)
        with open(ruta_imagen, "wb") as file:
            file.write(imagen_usuario.read())

        # Definir url_carpeta
        url_carpeta = f"users/{usuario}"  # Ruta relativa

        # Llamar a registrar_usuario pasando los 3 argumentos
        registrado = registrar_usuario(usuario, contraseña, url_carpeta)

        if registrado:
            st.success(f"Usuario {usuario} registrado exitosamente.")

            # Generar aumento de datos
            st.info("Generando aumento de datos...")
            generar_aumento_datos(user_dir)
            st.success("Imágenes aumentadas generadas correctamente.")
        else:
            st.error("Error al registrar el usuario.")

def login_page():
    st.title("Inicio de Sesión")
    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")

    if st.button("Ingresar"):
        user_data = buscar_usuario_por_nombre(usuario)
        if user_data and user_data[2] == contraseña:
            st.success(f"¡Bienvenido, {usuario}!")
        else:
            st.error("Usuario o contraseña incorrectos.")

if __name__ == "__main__":
    st.sidebar.title("Navegación")
    opcion = st.sidebar.radio("Seleccione una opción", ["Login", "Registro"])

    if opcion == "Login":
        login_page()
    elif opcion == "Registro":
        registro_page()
