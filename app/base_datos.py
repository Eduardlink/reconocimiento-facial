import mysql.connector
from mysql.connector import Error

def create_connection():
    """
    Crea una conexión con la base de datos MySQL.
    """
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="reconocimiento_facial"
        )
        if connection.is_connected():
            print("Conexión exitosa a la base de datos")
        return connection
    except Error as e:
        print(f"Error al conectar con la base de datos: {e}")
        return None

def execute_query(connection, query, params=None):
    """
    Ejecuta una consulta en la base de datos.
    
    Args:
        connection: Objeto de conexión a la base de datos.
        query: Consulta SQL para ejecutar.
        params: Parámetros para la consulta (por defecto None).
    
    Returns:
        Resultados de la consulta si aplica, None en caso de error.
    """
    try:
        cursor = connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        connection.commit()
        print("Consulta ejecutada con éxito")
        return cursor
    except Error as e:
        print(f"Error al ejecutar la consulta: {e}")
        return None

def close_connection(connection):
    """
    Cierra la conexión con la base de datos.
    """
    if connection.is_connected():
        connection.close()
        print("Conexión cerrada")
