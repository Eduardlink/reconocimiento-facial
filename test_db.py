from app.base_datos import create_connection, close_connection

# Probar la conexión
def test_database_connection():
    connection = create_connection()  # Crear la conexión
    if connection:
        print("¡Conexión exitosa a la base de datos!")
        close_connection(connection)  # Cerrar la conexión
    else:
        print("No se pudo conectar a la base de datos.")

if __name__ == "__main__":
    test_database_connection()
