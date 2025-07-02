import os
import subprocess

# Nombre de la aplicación Streamlit (archivo principal)
app = "main.py"

def launch_streamlit_app():
    if not os.path.isfile(app):
        if os.path.isdir("web"):
            os.chdir("web")
            print("Cambiando al directorio 'web'")
        else:
            print(f"No se encontró {app} ni el directorio 'web'.")
            return

    # Ejecuta Streamlit para lanzar main.py
    subprocess.run(["streamlit", "run", app])

if __name__ == "__main__":
    launch_streamlit_app()
