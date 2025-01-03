import os
import streamlit as st
from app.deteccion import reconocer_usuario_streamlit
from services.usuario_service import buscar_usuario_por_nombre, registrar_usuario
from scripts.entrenamiento_modelo import generar_aumento_datos, entrenar_modelo
from pyngrok import ngrok
public_url = ngrok.connect(8501)
st.sidebar.write(f"URL pública: {public_url}")
def login_page():
    st.title("Inicio de Sesión")

    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")

    # Primera fila: Botones Ingresar y Ir al Registro
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingresar"):
            user_data = buscar_usuario_por_nombre(usuario)
            if user_data and user_data[2] == contraseña:
                st.session_state.page = "Bienvenida"
                st.session_state.usuario = usuario
            else:
                st.error("Usuario o contraseña incorrectos.")
    with col2:
        if st.button("Ir al Registro"):
            st.session_state.page = "Registro"

    # Segunda fila: Botón de Reconocimiento Facial
    if st.button("Ingresar con Reconocimiento Facial"):
        reconocido = reconocer_usuario_streamlit()
        if reconocido:
            st.session_state.page = "Bienvenida"
            st.session_state.usuario = reconocido

def registro_page():
    st.title("Registro de Usuario")
    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")
    imagenes_usuario = st.file_uploader("Sube tus imágenes de rostro", accept_multiple_files=True, type=["jpg", "png"])

    if st.button("Registrar"):
        if not usuario or not contraseña or not imagenes_usuario:
            st.error("Por favor, complete todos los campos y suba imágenes.")
            return

        if registrar_usuario(usuario, contraseña):
            # Crear directorio del usuario
            user_dir = os.path.join("../users", usuario)
            os.makedirs(user_dir, exist_ok=True)

            # Guardar imágenes subidas en el directorio del usuario
            for imagen in imagenes_usuario:
                with open(os.path.join(user_dir, imagen.name), "wb") as f:
                    f.write(imagen.read())

            # Generar aumento de datos
            st.info("Generando imágenes aumentadas...")
            generar_aumento_datos(user_dir, 1000)

            # Entrenar modelo
            st.info("Entrenando el modelo con los nuevos datos...")
            entrenar_modelo()

            st.success("Usuario registrado y modelo actualizado exitosamente.")
        else:
            st.error("Error al registrar el usuario.")

    if st.button("Volver al Login"):
        st.session_state.page = "Login"

def bienvenida_page():
    st.title("Bienvenido a la Aplicación")
    st.write(f"Hola, {st.session_state.usuario}!")
    st.write("Has iniciado sesión correctamente.")
    if st.button("Cerrar Sesión"):
        st.session_state.page = "Login"
        del st.session_state.usuario

if __name__ == "__main__":
    # Configuración inicial del estado
    if "page" not in st.session_state:
        st.session_state.page = "Login"

    # Navegación dinámica basada en el estado
    if st.session_state.page == "Login":
        login_page()
    elif st.session_state.page == "Registro":
        registro_page()
    elif st.session_state.page == "Bienvenida":
        bienvenida_page()
