from app.base_datos import create_connection, execute_query, close_connection

def insertar_usuario(usuario, contraseña, url_carpeta):
    """
    Inserta un nuevo usuario en la base de datos.
    """
    connection = create_connection()
    if connection:
        query = """
        INSERT INTO usuarios (usuario, contraseña, url_carpeta)
        VALUES (%s, %s, %s);
        """
        params = (usuario, contraseña, url_carpeta)
        execute_query(connection, query, params)
        close_connection(connection)

def consultar_usuarios():
    """
    Consulta todos los usuarios registrados.
    """
    connection = create_connection()
    usuarios = []
    if connection:
        try:
            query = "SELECT * FROM usuarios;"
            cursor = connection.cursor()
            cursor.execute(query)
            usuarios = cursor.fetchall()  # Consume todos los resultados
            cursor.close()  # Asegúrate de cerrar el cursor después de usarlo
        except Exception as e:
            print(f"Error al ejecutar la consulta: {e}")
        finally:
            close_connection(connection)  # Cierra la conexión
    return usuarios

def buscar_usuario_por_nombre(nombre_usuario):
    """
    Busca un usuario por su nombre y devuelve los detalles.
    """
    connection = create_connection()
    usuario = None
    if connection:
        try:
            query = "SELECT * FROM usuarios WHERE usuario = %s;"
            params = (nombre_usuario,)
            cursor = connection.cursor()
            cursor.execute(query, params)
            usuario = cursor.fetchone()
            cursor.close()
        except Exception as e:
            print(f"Error al buscar el usuario: {e}")
        finally:
            close_connection(connection)
    return usuario


def actualizar_carpeta_usuario(nombre_usuario, nueva_url_carpeta):
    """
    Actualiza la carpeta asociada a un usuario.
    """
    connection = create_connection()
    if connection:
        query = """
        UPDATE usuarios
        SET url_carpeta = %s
        WHERE usuario = %s;
        """
        params = (nueva_url_carpeta, nombre_usuario)
        execute_query(connection, query, params)
        close_connection(connection)

def eliminar_usuario(nombre_usuario):
    """
    Elimina un usuario por su nombre.
    """
    connection = create_connection()
    if connection:
        query = "DELETE FROM usuarios WHERE usuario = %s;"
        params = (nombre_usuario,)
        execute_query(connection, query, params)
        close_connection(connection)
