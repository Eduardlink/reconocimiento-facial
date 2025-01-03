import streamlit as st
from app.deteccion import reconocer_usuario_streamlit
from services.usuario_service import buscar_usuario_por_nombre

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

    if st.button("Ingresar con Reconocimiento Facial"):
        reconocer_usuario_streamlit()

if __name__ == "__main__":
    login_page()
