
# Proyecto de Reconocimiento Facial

Este proyecto implementa un sistema de registro y login utilizando detección facial. Incluye funcionalidades para capturar imágenes, almacenar usuarios en una base de datos y autenticar mediante Streamlit.

---

## Requisitos Previos

1. **Python 3.8 o superior** instalado en tu sistema.
2. **MySQL** instalado y en funcionamiento.

---

## Configuración del Proyecto

### 1. Crear la Base de Datos

1. Abre tu cliente de MySQL (por ejemplo, phpMyAdmin o la línea de comandos).
2. Crea una base de datos llamada `reconocimiento_facial` con el siguiente comando:

   ```sql
   CREATE DATABASE reconocimiento_facial;
   ```

3. Carga el backup de la base de datos ubicado en `services/usuarios.sql`. Puedes hacerlo con el siguiente comando en MySQL:

   ```bash
   mysql -u root -p reconocimiento_facial < services/usuarios.sql
   ```

   Asegúrate de reemplazar `root` con el usuario de tu base de datos y proporcionar la contraseña si es necesario.

---

### 2. Crear el Entorno Virtual

1. En la raíz del proyecto, crea un entorno virtual llamado `reconocimientoEnv`:

   ```bash
   python -m venv reconocimientoEnv
   ```

2. Activa el entorno virtual:
   - **Windows:**
     ```bash
     reconocimientoEnv\Scripts\activate
     ```
   - **MacOS/Linux:**
     ```bash
     source reconocimientoEnv/bin/activate
     ```

3. Instala las dependencias necesarias:

   ```bash
   pip install mysql-connector-python
   pip install streamlit
   pip install opencv-python
   ```

---

### 3. Ejecutar la Aplicación

1. Activa el entorno virtual si no lo has hecho ya (ver instrucciones anteriores).
2. Ejecuta la aplicación con el siguiente comando:

   ```bash
   streamlit run main.py
   ```

3. Abre el enlace que aparece en la terminal (por defecto: `http://localhost:8501`) en tu navegador.

---

## Funcionalidades Principales

1. **Login:**
   - Ingresar con usuario y contraseña.
   - Ingresar con reconocimiento facial.

2. **Registro:**
   - Captura de imágenes con la cámara.
   - Almacenamiento de información del usuario y su imagen en la base de datos.

---

## Estructura del Proyecto

```
reconocimiento-facial/
├── app/                   # Lógica de conexión con la base de datos
├── services/              # Servicios para manejo de usuarios y el archivo SQL
├── templates/             # Plantillas HTML (si se usan)
├── static/                # Recursos estáticos como CSS o JS
├── users/                 # Carpeta para almacenar imágenes de usuarios
├── main.py                # Archivo principal de la aplicación
├── README.md              # Instrucciones del proyecto
└── requirements.txt       # Dependencias del proyecto
```

---

## Notas

1. Asegúrate de que MySQL esté en funcionamiento antes de ejecutar el proyecto.
2. Si necesitas instalar otros paquetes, agrégalos al entorno virtual y luego actualiza el archivo `requirements.txt` con:
   ```bash
   pip freeze > requirements.txt
   ```
