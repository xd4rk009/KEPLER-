import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import io
import sys
import os
import time
from datetime import datetime  # Importar datetime

# Obtener la ruta del directorio padre (la carpeta raíz)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from classes.DuckDB_Helper_v02 import DuckDBHelper
from classes.Tronaduras_File_Reader_v03 import TronadurasFileReader

class TimestampedStream(io.StringIO):
    def write(self, message):
        """Agrega la fecha y hora (con milisegundos) al mensaje antes de imprimirlo"""
        if message != '\n':  # Evitar imprimir saltos de línea vacíos
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # Obtener fecha, hora y milisegundos
            message = f"[{timestamp}] - {message}"
        super().write(message)  # Llamamos al método de StringIO original

def app():
    st.title("Aplicar Imputación de Datos")
    st.markdown("""Esta vista ejecuta el proceso de imputación para el conjunto **Tronaduras**.<br>La consola se muestra al final, de forma similar a la terminal integrada en VSCode.""", unsafe_allow_html=True)
    
    # Contenedor para la consola (simula el terminal)
    console_placeholder = st.empty()

    if st.button("Ejecutar Imputación"):
        # Usamos un StringIO para capturar la salida de la consola
        log_capture = TimestampedStream()  # Usamos nuestro stream con timestamp
        old_stdout = sys.stdout
        sys.stdout = log_capture  # Redirigimos stdout a nuestro stream

        try:
            # Al principio muestra el mensaje de ejecución en progreso
            st.info("Ejecutando imputación... Por favor, espere.")
            
            # Construcción de la ruta absoluta para la base de datos
            DB_NAME = os.path.join(project_root, "data", "Tronaduras_vs_Sismicidad.db")
            SCHEMA_NAME = "Raw_Data"
            CONJUNTO_ESPECIFICO = "Tronaduras"
            
            print("🔗 Conectando a la base de datos...")
            sys.stdout.flush()
            console_placeholder.text_area("Consola", log_capture.getvalue(), height=300)
            time.sleep(0.1)
            
            db_helper = DuckDBHelper(DB_NAME)
            
            print("📂 Leyendo datos de tronaduras...")
            sys.stdout.flush()
            console_placeholder.text_area("Consola", log_capture.getvalue(), height=300)
            time.sleep(0.1)
            
            df_tronaduras = db_helper.select_df(table="Tronaduras", schema="Raw_Data")
            
            print("📊 Leyendo estrategias de imputación...")
            sys.stdout.flush()
            console_placeholder.text_area("Consola", log_capture.getvalue(), height=300)
            time.sleep(0.1)
            
            strategies_df = db_helper.select_df(
                table="Variables_Description",
                columns='"Variable", "Imputacion", "Defecto"',
                where=f"Conjunto = '{CONJUNTO_ESPECIFICO}'",
                schema=SCHEMA_NAME
            )
            
            print("⚙️ Aplicando imputación...")
            sys.stdout.flush()
            console_placeholder.text_area("Consola", log_capture.getvalue(), height=300)
            time.sleep(0.1)
            
            Tronaduras_Reader = TronadurasFileReader()
            df_tronaduras = Tronaduras_Reader.impute_df(df=df_tronaduras, strategies_df=strategies_df)
            
            print("💾 Creando tabla en Processed_Data...")
            sys.stdout.flush()
            console_placeholder.text_area("Consola", log_capture.getvalue(), height=300)
            time.sleep(0.1)
            
            # Por ejemplo, aquí podrías descomentar la siguiente línea si lo necesitas:
            # db_helper.create_table_from_df("Tronaduras", df_tronaduras, schema="Processed_Data")
            
            print("🔒 Cerrando conexión a la base de datos...")
            sys.stdout.flush()
            console_placeholder.text_area("Consola", log_capture.getvalue(), height=300)
            time.sleep(0.1)
            
            db_helper.close_connection()
            
            print("✅ Imputación completada exitosamente.")
            sys.stdout.flush()
            console_placeholder.text_area("Consola", log_capture.getvalue(), height=300)
            
            # Mostrar tablas modificadas
            #st.subheader("Tablas Modificadas:")
            
            # Línea separadora entre la consola y las tablas
            st.markdown('<hr style="margin-top: 5px; margin-bottom: 50px; border: 1px solid #ddd;">', unsafe_allow_html=True)

            # Muestra las primeras filas del DataFrame imputado
            st.subheader("**Tronaduras (imputado):**")
            st.dataframe(df_tronaduras)  # Muestra las primeras 5 filas (ajustar si es necesario)
            
            # Si necesitas mostrar otras tablas modificadas, puedes agregarlas aquí
            # Por ejemplo:
            # st.write("**Otra tabla modificada:**")
            # st.dataframe(df_otra_tabla.head())
            
            # Al finalizar la imputación, cambiamos el mensaje de la información
            st.info("Imputación finalizada.")
            
        except Exception as e:
            print(f"❌ Error durante la imputación: {e}")
            sys.stdout.flush()
            console_placeholder.text_area("Consola", log_capture.getvalue(), height=300)
        finally:
            sys.stdout = old_stdout  # Restauramos stdout al original
