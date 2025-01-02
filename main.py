import streamlit as st
import cv2
from services.usuario_service import buscar_usuario_por_nombre, insertar_usuario

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
            abrir_camara()

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

# Página de Registro
def register_page():
    st.title("Registro de Usuario")
    
    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")
    confirmar_contraseña = st.text_input("Confirmar Contraseña", type="password")
    
    if st.button("Registrarse"):
        if contraseña != confirmar_contraseña:
            st.error("Las contraseñas no coinciden.")
        elif not usuario or not contraseña:
            st.error("Todos los campos son obligatorios.")
        else:
            try:
                insertar_usuario(usuario, contraseña, f"users/{usuario}")
                st.success("Usuario registrado con éxito.")
                set_page("login")  # Volver al login
            except Exception as e:
                st.error(f"Error al registrar usuario: {e}")
    
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
