import duckdb
import os
import time
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class DuckDBHelper:
    def __init__(self, db_name: Optional[str] = None):
        self.db_name = db_name
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.schemas: Dict[str, List[str]] = {}

        if db_name:
            self.create_db(db_name)

    """
    ╭─────────────────────────────────────────────────────╮
    │ Función: create_db                                  │
    │ Crea/conecta a una base de datos                    │
    ╰─────────────────────────────────────────────────────╯
    """
    def create_db(self, db_name: str):
        self.db_name = db_name
        db_path = ':memory:' if db_name == ':memory:' else os.path.abspath(db_name)
        self.conn = duckdb.connect(db_path)
        logging.info(f"Conectado a {db_name}")
        self.update()

    """
    ╭──────────────────────────────────────────────────────╮
    │ Función: update                                      │
    │ Actualiza metadatos de la base de datos              │
    ╰──────────────────────────────────────────────────────╯
    """
    def update(self):
        if not self.conn:
            return
        schemas = self.conn.execute("SELECT schema_name FROM information_schema.schemata;").fetchall()
        self.schemas = {}
        for (schema,) in schemas:
            tables = self.conn.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}';"
            ).fetchall()
            self.schemas[schema] = [table[0] for table in tables]

    """
    ╭─────────────────────────────────────────────────────╮
    │ Función: delete_db                                  │
    │ Elimina la base de datos actual                     │
    ╰─────────────────────────────────────────────────────╯
    """
    def delete_db(self):
        if self.db_name and self.db_name != ':memory:':
            self.conn.close()
            os.remove(self.db_name)
        self.conn = None
        self.db_name = None
        self.schemas = {}

    """
    ╭────────────────────────────────────────────────────╮
    | Función: close_connection                          |
    | Cierra la conexión a la base de datos actual       |
    ╰────────────────────────────────────────────────────╯
    """
    def close_connection(self):
        if self.conn:
            self.conn.close()
            print("🔌 Conexión cerrada manualmente.")
        # Uso: close_connection()

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: create_schema                             │
    │ Crea un nuevo esquema                              │
    ╰────────────────────────────────────────────────────╯
    """
    def create_schema(self, schema: str):
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        self.update()

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: delete_schema                             │
    │ Elimina un esquema existente                       │
    ╰────────────────────────────────────────────────────╯
    """
    def delete_schema(self, schema: str):
        self.conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
        self.update()

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: create_table                              │
    │ Crea una nueva tabla en el esquema                 │
    ╰────────────────────────────────────────────────────╯
    """
    def create_table(self, table: str, columns: List[str], schema: str = "main"):
        cols = ", ".join(columns)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {schema}.{table} ({cols});")
        self.update()

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: delete_table                              │
    │ Elimina una tabla existente                        │
    ╰────────────────────────────────────────────────────╯
    """
    def delete_table(self, table: str, schema: str = "main"):
        self.conn.execute(f"DROP TABLE IF EXISTS {schema}.{table};")
        self.update()

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: get_schemas                               │
    │ Devuelve todos los esquemas disponibles            │
    ╰────────────────────────────────────────────────────╯
    """
    def get_schemas(self) -> List[str]:
        return list(self.schemas.keys())

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: get_tables                                │
    │ Devuelve tablas de un esquema específico           │
    ╰────────────────────────────────────────────────────╯
    """
    def get_tables(self, schema: str) -> List[str]:
        return self.schemas.get(schema, [])

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: insert_data                               │
    │ Inserta datos en una tabla                         │
    ╰────────────────────────────────────────────────────╯
    """
    def insert_data(self, table: str, data: Dict[str, Any], schema: str = "main"):
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = list(data.values())
        self.conn.execute(f"INSERT INTO {schema}.{table} ({cols}) VALUES ({placeholders})", values)

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: update_data                               │
    │ Actualiza registros en una tabla                   │
    ╰────────────────────────────────────────────────────╯
    """
    def update_data(self, table: str, updates: Dict[str, Any], where: str, schema: str = "main"):
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        self.conn.execute(f"UPDATE {schema}.{table} SET {set_clause} WHERE {where}", values)

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: delete_data                               │
    │ Elimina registros de una tabla                     │
    ╰────────────────────────────────────────────────────╯
    """
    def delete_data(self, table: str, where: str, schema: str = "main"):
        self.conn.execute(f"DELETE FROM {schema}.{table} WHERE {where};")

    """
    ╭────────────────────────────────────────────────────╮
    │ Función: select_data                               │
    │ Consulta datos de una tabla                        │
    ╰────────────────────────────────────────────────────╯
    """
    def select_data(self, table: str, columns: str = "*", where: Optional[str] = None, schema: str = "main"):
        query = f"SELECT {columns} FROM {schema}.{table}"
        if where:
            query += f" WHERE {where}"
        return self.conn.execute(query).fetchall()

    """
    ╭──────────────────────────────────────────────────────────────────────────────────────────╮
    │ Ejecuta un SELECT en la base de datos y devuelve el resultado como DataFrame de pandas.  │
    │ Parámetros:                                                                              │
    │   - table: nombre de la tabla a consultar                                                │
    │   - columns: columnas a seleccionar (default "*")                                        │
    │   - where: condición de selección (default None)                                         │
    │   - schema: esquema de la tabla (default "main")                                         │
    ╰──────────────────────────────────────────────────────────────────────────────────────────╯
    """
    def select_df(self, table: str, columns: str = "*", where: str = None, schema: str = "main") -> pd.DataFrame:
        query = f"SELECT {columns} FROM {schema}.{table}"
        if where:
            query += f" WHERE {where}"
        print(f"✅ Datos leídos desde '{schema}.{table}' y convertidos a DataFrame.")
        return self.conn.execute(query).fetchdf()
        # Uso: select_df("tabla", "nombre, edad", "edad > 30")

    """
    ╭────────────────────────────────────────────────────────────╮
    │ Función: create_table_from_df                              │
    │ Crea tabla desde DataFrame                                 │
    ╰────────────────────────────────────────────────────────────╯
    """
    def create_table_from_df(self, table: str, df: pd.DataFrame, schema: str = "main"):
        if not self.conn:
            raise Exception("La conexión no está abierta.")

        if table in self.get_tables(schema):
            self.conn.execute(f"DROP TABLE {schema}.{table};")
            logging.info(f"Tabla existente '{schema}.{table}' eliminada.")

        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        self.conn.execute(f"CREATE TABLE {schema}.{table} AS SELECT * FROM df", df=df)
        logging.info(f"Tabla '{schema}.{table}' creada desde DataFrame.")
        self.update()

    """
    ╭────────────────────────────────────────────────────────────────╮
    │ Función: create_table_from_csv                                 │
    │ Lee un archivo CSV y crea una tabla en la base de datos        │
    │                                                                │
    │ Parámetros:                                                    │
    │   - table: nombre de la tabla a crear                          │
    │   - csv_file: ruta del archivo CSV                             │
    │   - schema: esquema donde se creará la tabla (default "main")  │
    ╰────────────────────────────────────────────────────────────────╯
    """
    def create_table_from_csv(self, table: str, csv_file: str, separator: str = ';', schema: str = "main"):
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        self.conn.execute(f"CREATE TABLE {schema}.{table} AS SELECT * FROM read_csv_auto('{csv_file}', delim='{separator}');")
        logging.info(f"Tabla '{schema}.{table}' creada desde CSV: {csv_file}")
        self.update()
