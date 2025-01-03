import mysql.connector

# Configuración de conexión a MySQL
DB_CONFIG = {
    "host": "localhost",          # Cambia si usas otro servidor
    "user": "root",               # Usuario de MySQL
    "password": "",               # Contraseña de MySQL
    "database": "reconocimiento_facial"  # Nombre de la base de datos
}

def buscar_usuario_por_nombre(nombre):
    """
    Busca un usuario en la base de datos por su nombre.
    Retorna los datos del usuario si se encuentra, o None si no existe.
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
