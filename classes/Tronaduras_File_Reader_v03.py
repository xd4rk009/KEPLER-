import pandas as pd
import os
from typing import List
import numpy as np

"""
╭────────────────────────────────────────────────────╮
│ Clase: TronadurasFileReader                        │
│ Lee archivos Excel utilizando la configuración     │
│ definida en un DataFrame (tronaduras_variables_df) │
╰────────────────────────────────────────────────────╯
"""
class TronadurasFileReader:
    """
    ╭────────────────────────────────────────────────────╮
    │ Función: __init__                                  │
    │ Inicializa la clase con un directorio por defecto  │
    │ y una variable para almacenar el DataFrame de      │
    │ variables de configuración.                        │
    ╰────────────────────────────────────────────────────╯
    """
    def __init__(self):

        self.directory = "../Datos Teniente/"  # Directorio por defecto
        self.variables_df = None  # DataFrame con la configuración de variables
        self.df = None


    """
    ╭──────────────────────────────────────────────────╮
    │ Función: get_excel_files                         │
    │ Obtiene archivos XLSX del directorio             │
    ╰──────────────────────────────────────────────────╯
    """
    def get_excel_files(self) -> List[str]:
        return [f for f in os.listdir(self.directory) if f.endswith('.xlsx')]

    """
    ╭──────────────────────────────────────────────────╮
    │ Función: get_sheet_names                         │
    │ Obtiene nombres de hojas de Excel                │
    ╰──────────────────────────────────────────────────╯
    """
    def get_sheet_names(self, file_path: str) -> List[str]:
        return pd.ExcelFile(file_path).sheet_names

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: _apply_data_types                         │
    │ Aplica los tipos de datos a las columnas definidas │
    │ en self.variables_df, considerando sólo las filas  │
    │ con "Lectura" igual a "Si".                        │
    ╰────────────────────────────────────────────────────╯
    """
    def _apply_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        errors = {}
        # Iterar sobre cada variable en el DataFrame de variables (solo si Lectura es "Si")
        for _, row in self.variables_df.iterrows():
            if str(row["Lectura"]).strip().lower() == "si":
                col = row["Variable"]
                dtype = str(row["Tipo"]).strip().lower()
                if col in df.columns:
                    try:
                        if dtype == "integer":
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                        elif dtype == "float":
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                        elif dtype == "datetime":
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif dtype in ["string", "str"]:
                            df[col] = df[col].astype(str)
                        elif dtype == "categorical":
                            df[col] = df[col].astype(str).astype('category')
                        else:
                            # Si el tipo no es reconocido, se deja la columna sin conversión.
                            df[col] = df[col]
                        
                        # Registrar error si (excepto para categoricals) se generan NA tras la conversión.
                        if dtype != "categorical" and df[col].isnull().any():
                            errors[col] = df[df[col].isnull()].index.tolist()
                    except Exception as e:
                        errors[col] = str(e)
        if errors:
            print("\n⚠️ Errores de conversión:")
            for col, error in errors.items():
                print(f"\t⚠ {col}: {error}")
        return df

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: _clean_column_names                       │
    │ Normaliza los nombres de las columnas eliminando   │
    │ saltos de línea y espacios extra                   │
    ╰────────────────────────────────────────────────────╯
    """
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        original_cols = df.columns.tolist()
        df.columns = [col.replace("\n", " ").strip() for col in df.columns]
        renamed = sum(1 for o, n in zip(original_cols, df.columns) if o != n)
        print(f"✅ Columnas renombradas: {renamed}")
        return df

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: read_file                                 │
    │ Lee el archivo Excel utilizando la configuración   │
    │ de variables (variables_df) y filtra las columnas  │
    │ según aquellas con 'Lectura' igual a 'Si'.         │
    ╰────────────────────────────────────────────────────╯
    """
    def read_file(
        self,
        file_name: str = None,
        sheet_name: str = None,
        sheet_index: int = 2,
        directory: str = None,
        variables_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        # Actualizar directorio si se pasa uno nuevo.
        if directory:
            self.directory = directory

        # Asignar el DataFrame de variables (debe incluir al menos 'Variable', 'Lectura' y 'Tipo')
        if variables_df is None:
            raise ValueError("❌ Debe proporcionar un DataFrame de variables (variables_df)")
        self.variables_df = variables_df

        # Seleccionar archivo por defecto si no se especifica ninguno.
        if not file_name:
            archivos = self.get_excel_files()
            if not archivos:
                raise ValueError("❌ No se encontraron archivos XLSX en el directorio")
            file_name = os.path.join(self.directory, archivos[0])
            print(f"⚠️ Usando archivo por defecto: {os.path.basename(file_name)}")
        else:
            file_name = os.path.join(self.directory, file_name)
            
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"❌ Archivo no encontrado: {file_name}")

        # Seleccionar hoja por defecto si no se especifica.
        if not sheet_name:
            sheet_names = self.get_sheet_names(file_name)
            if not sheet_names:
                raise ValueError("❌ No se encontraron hojas en el archivo Excel")
            sheet_name = sheet_names[sheet_index]
            print(f"⚠️ Usando hoja por defecto: {sheet_name}")

        try:
            df = pd.read_excel(file_name, sheet_name=sheet_name, header=0, skiprows=range(1, 4))
        except Exception as e:
            raise ValueError(f"❌ Error leyendo archivo: {str(e)}")
        
        # Limpiar nombres de columnas.
        df = self._clean_column_names(df)
        # Eliminar columnas 'Unnamed' que pueden aparecer al leer el Excel.
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print(f"✨ Columnas encontradas en el archivo: {df.columns.tolist()}")

        # Filtrar solo las columnas definidas en variables_df donde 'Lectura' sea "Si"
        valid_values = {"si", "sí"}  # Conjunto de valores válidos en minúsculas
        valid_variables = self.variables_df.loc[
            self.variables_df["Lectura"].str.strip().str.lower().isin(valid_values), "Variable"
        ].tolist()

        # Limpiar nombres de columnas eliminando saltos de línea y espacios innecesarios
        df.columns = df.columns.str.replace("\n", "", regex=False)
        df.columns = df.columns.str.replace("  ", " ", regex=False)

        columnas_validas = [col for col in df.columns if col in valid_variables]

        if not columnas_validas:
            raise ValueError("❌ Ninguna de las columnas definidas en variables_df se encontró en el archivo.")
        
        df = df[columnas_validas]
        print(f"✅ Columnas seleccionadas para procesar: {columnas_validas}")

        # Aplicar la conversión de tipos de datos según la configuración.
        df = self._apply_data_types(df)
        df.drop(columns=["Hora"], inplace=True)

        return df.reset_index(drop=True)
        

    """
    ╭───────────────────────────────────────────────────────────────────────────╮
    │ Función: read_file                                                        │
    │ Función para calcular el promedio de uno o más valores en formato "X-Y"   │
    │ y devolver el promedio de todos los rangos.                               │
    ╰───────────────────────────────────────────────────────────────────────────╯
    """
    def mean_range(self, valor):
        if pd.isna(valor):  # Manejar valores nulos
            return np.nan
        
        try:
            # Dividir por comas para manejar múltiples rangos
            rangos = valor.split(",")
            promedios = []
            
            for rango in rangos:
                # Eliminar espacios en blanco alrededor del rango
                rango = rango.strip()
                # Dividir el rango en dos partes usando el guion como separador
                partes = rango.split("-")
                if len(partes) == 2:  # Asegurarse de que haya dos partes
                    min_val = float(partes[0])
                    max_val = float(partes[1])
                    promedio = (min_val + max_val) / 2
                    promedios.append(promedio)
            
            if promedios:  # Si se encontraron promedios válidos
                return sum(promedios) / len(promedios)  # Promedio de todos los rangos
            else:
                return np.nan  # Si no se encontraron rangos válidos
        except (ValueError, IndexError):
            return np.nan  # Manejar casos donde el formato no sea válido   
        


    """
    ╭───────────────────────────────────────────────────────────────────────────────────╮
    │ Imputa la columna 'col' del DataFrame 'df' según la estrategia leída de la tabla. │
    │                                                                                   │                  
    │ Métodos disponibles:                                                              │                                 
    │    - "drop": Elimina la columna.                                                  │         
    │    - "median": Imputa con la mediana de la columna.                               │
    │    - "mode": Imputa con la moda (valor más frecuente).                            │
    │    - "constant": Imputa con un valor constante definido en 'Defecto'.             │
    │    - "datetime": Convierte la columna a tipo fecha y, si falla, usa 'Defecto'.    │
    │    - "interpolate": Interpola los valores faltantes (lineal).                     │
    │    - "ffill": Rellena con el valor anterior válido (forward fill).                │
    │    - "bfill": Rellena con el siguiente valor válido (backward fill).              │ 
    ╰───────────────────────────────────────────────────────────────────────────────────╯
    """
    def impute_column(self, df: pd.DataFrame, col: str, strategy: pd.Series):
        """
        Imputa la columna 'col' del DataFrame 'df' según la estrategia definida en el
        objeto pd.Series 'strategy'. Se utilizan directamente las columnas 'Imputacion'
        y 'Defecto' del DataFrame de estrategias.
        """
        method = str(strategy.get("Imputacion", "")).strip().lower()
        
        # ✅ Evitar problemas con columnas categóricas
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)  # Convertir a string antes de reemplazar

        # ✅ Reemplazar "nan" (string) por valores reales NaN
        df.loc[:, col] = df[col].replace(["nan", "NaN", "NAN", ""], np.nan)

         # 🔍 Contar NaN antes de la imputación
        nan_count = df[col].isna().sum()


        if method == "drop":
            df.dropna(subset=[col], inplace=True)  # 🔥 Elimina solo las FILAS donde 'col' sea NaN o vacío
            print(f"🗑️  {nan_count} filas eliminadas donde '{col}' era NaN o vacío.")
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "mode":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                print(f"⚠️  No se pudo imputar la moda para '{col}' porque no se encontró valor dominante.")
        elif method == "constant":
            default_val = strategy.get("Defecto")
            df[col] = df[col].fillna(default_val)
        elif method == "datetime":
            default_val_str = strategy.get("Defecto", "1900-01-01")
            default_val_ts = pd.to_datetime(default_val_str, errors='coerce')
            df[col] = pd.to_datetime(df[col], errors='coerce').fillna(default_val_ts)
        elif method == "interpolate":
            df[col] = df[col].interpolate()
        elif method == "ffill":
            df[col] = df[col].ffill()
        elif method == "bfill":
            df[col] = df[col].bfill()
        else:
            print(f"⚠️  Método '{method}' no reconocido para la columna '{col}'.")
            
        return df
    


    """
    ╭──────────────────────────────────────────────────╮
    │ Función: impute_df                               │
    │ Aplica la imputación a todo el DataFrame usando  │
    │ las estrategias definidas en 'strategies_df'.    │
    │ Retorna el DataFrame imputado.                   │
    ╰──────────────────────────────────────────────────╯
    """
    def impute_df(self, df: pd.DataFrame, strategies_df: pd.DataFrame) -> pd.DataFrame:
        for _, row in strategies_df.iterrows():
            col = row["Variable"]
            if col in df.columns:
                df = self.impute_column(df, col, row)
                print(f"✅ Columna '{col}' procesada correctamente.")
            else:
                print(f"❌ Columna '{col}' no encontrada en el DataFrame, se omite.")

        print("✅ Imputación finalizada.")
        return df  # 🔥 Retorna el DataFrame imputado