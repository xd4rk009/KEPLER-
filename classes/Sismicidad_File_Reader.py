import pandas as pd
import os

class SismicidadFileReader:
    def __init__(self, directory="../Datos Teniente/", delimiter=";", encoding="utf-8"):
        self.directory = directory
        self.delimiter = delimiter
        self.encoding = encoding

    def get_csv_files(self):
        """
        Retorna una lista de archivos CSV en el directorio especificado.
        """
        try:
            return [f for f in os.listdir(self.directory) if f.endswith('.csv')]
        except Exception as e:
            print(f"Error al listar archivos en el directorio {self.directory}: {e}")
            return []

    def read_csv_file(self, file_path, parse_dates=None):
        """
        Lee un archivo CSV y devuelve un DataFrame de pandas.

        Parámetros:
        - file_path (str): Ruta del archivo CSV.
        - parse_dates (list or None): Lista de nombres de columnas a convertir en fechas (opcional).

        Retorna:
        - pd.DataFrame: DataFrame con los datos del archivo CSV.
        """
        try:
            df = pd.read_csv(file_path, delimiter=self.delimiter, encoding=self.encoding, parse_dates=parse_dates)
            return df
        except Exception as e:
            print(f"Error al leer el archivo CSV: {e}")
            return None

    def read_file(self):
        """
        Procesa un archivo CSV específico y retorna un DataFrame con las transformaciones aplicadas.

        Parámetros:
        - file_name (str): Nombre del archivo CSV a procesar.

        Retorna:
        - pd.DataFrame: DataFrame procesado con la columna 'Date Time'.
        """

        # Leer archivos XLSX del directorio
        csv_files = self.get_csv_files()
        # print('\n'.join(f"Archivo {i}: {file}" for i, file in enumerate(csv_files)) + '\n')

        # Selección de archivo y hojas a leer
        csv_file_path = f"{self.directory}{csv_files[1]}"

        df = self.read_csv_file(file_path=csv_file_path)


        # Asegurar que las columnas sean leídas correctamente como float
        for col in df.columns[2:]:
            df[col] = (df[col].str.replace(',', '.').astype(float))


        if df is not None:
            try:
                # Convertir las columnas a formatos correctos
                df["#EventDate"] = pd.to_datetime(df["#EventDate"], format="%d-%m-%Y")
                df["EventTimeInDay"] = pd.to_datetime(df["EventTimeInDay"], format="%H:%M:%S").dt.time

                # Crear la columna 'Date Time'
                df['Date Time'] = df.apply(
                    lambda row: pd.to_datetime(f"{row['#EventDate'].date()} {row['EventTimeInDay']}") \
                    if pd.notna(row['#EventDate']) else None, axis=1)

                # Insertar 'Date Time' como la primera columna
                df.insert(0, 'Date Time', df.pop('Date Time'))

                # Eliminar las columnas originales
                df = df.drop(['#EventDate', 'EventTimeInDay'], axis=1)

                return df
            except Exception as e:
                print(f"Error al procesar el archivo {csv_file_path}: {e}")
                return None
        else:
            return None