# extract.py
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
import requests
import time
import pandas as pd
import polars as pl
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path


class Extract:
    def __init__(self, chunk: int = 50000):
          # Obtener ruta absoluta del .env junto al script
        
        env_path = Path(__file__).parent / ".env"
        print(f"DEBUG: cargando .env desde {env_path}")
        load_dotenv(env_path, override=True)

        # ----------------------------
        # Cargar variables de entorno
        # ----------------------------
        
        self.api_url = os.getenv("API_URL")
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB")
        self.collection_name = os.getenv("MONGO_COLLECTION")
        self.chunk = chunk

        # Limpiar espacios y comillas accidentales
        if self.api_url:
            self.api_url = self.api_url.strip().replace('"', '')
        if self.mongo_uri:
            self.mongo_uri = self.mongo_uri.strip().replace('"', '')
        if self.db_name:
            self.db_name = self.db_name.strip().replace('"', '')
        if self.collection_name:
            self.collection_name = self.collection_name.strip().replace('"', '')

        # Validar que no falte ninguna variable
        if not all([self.api_url, self.mongo_uri, self.db_name, self.collection_name]):
            missing = [
                name for name, val in zip(
                    ["API_URL", "MONGO_URI", "MONGO_DB", "MONGO_COLLECTION"],
                    [self.api_url, self.mongo_uri, self.db_name, self.collection_name]
                ) if not val
            ]
            raise RuntimeError(f"Faltan variables de entorno: {missing}")


        # Conexión a MongoDB

        self.client = MongoClient(self.mongo_uri)
        self.col = self.client[self.db_name][self.collection_name]

        # Crear índice único para evitar duplicados
        self.col.create_index("serialnumber", unique=True)

    def fetch_from_api(self):
        """
        Extrae datos desde la API e inserta en MongoDB.
        Evita duplicados por serialnumber.
        """

        print("-> Comenzando carga incremental desde API a MongoDB...")

        offset = 0
        while True:
            params = {"$limit": self.chunk, "$offset": offset}
            r = requests.get(self.api_url, params=params, timeout=60)
            r.raise_for_status()
            rows = r.json()

            if not rows:
                break

            try:
                self.col.insert_many(rows, ordered=False)
                print(f"   Insertadas {len(rows)} filas (offset {offset})")
            except BulkWriteError:
                print("   ⚠ Algunos registros ya existían y fueron omitidos.")

            offset += self.chunk
            time.sleep(0.2)

        print("✅ Carga completa")

    def load_from_mongo(self):
        """
        Carga toda la colección de MongoDB a un DataFrame de Polars.
        Esto se ejecuta siempre, ya haya o no datos nuevos.
        """
        df = pd.DataFrame(list(self.col.find()))
        if df.empty:
            print("⚠ La colección está vacía. No hay datos para cargar.")
            return pl.DataFrame()  # devuelve un Polars vacío

        df['_id'] = df['_id'].astype(str)  # convertir ObjectId a string
        pl_df = pl.from_pandas(df)
        print(f"-> Datos cargados a Polars: {pl_df.shape[0]} filas, {pl_df.shape[1]} columnas")
        return pl_df
