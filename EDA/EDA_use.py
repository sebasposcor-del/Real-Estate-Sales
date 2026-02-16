
from EDA_class import EDA


import polars as pl
import os
from dotenv import load_dotenv, find_dotenv
import psycopg2
from urllib.parse import urlparse


import os
from urllib.parse import urlparse

import polars as pl
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine

SQL_DEFAULT = """
SELECT *
FROM ml_table;
"""

def load_df_from_sql(sql: str = SQL_DEFAULT) -> pl.DataFrame:
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path, override=True)

    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Falta DATABASE_URL en .env")

    p = urlparse(url)
    host = p.hostname or "?"
    port = p.port or "?"
    print(f"Conectando a: {host}:{port}")
    engine = create_engine(url)

    df = pl.read_database(sql, engine)

    engine.dispose()
    return df

def load_df_from_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)

def main():
    print("INICIANDO EDA")

    # === EXTRACT ===
    # Opción A) desde SQL:
    try:
        print(f"iniciando carga de datos")
        df_ml = load_df_from_sql(SQL_DEFAULT)
        print(f"Finalizada carga desde: {df_ml.shape}")
    except Exception as e:
        print(f"[Aviso] SQL falló ({e}). Intento Parquet...")
        # Opción B) desde archivo (ajusta la ruta):
        df_ml = load_df_from_parquet("data/ml_table.parquet")
        print(f"Datos cargados desde Parquet: {df_ml.shape}")

    # === EDA ===
    eda = (
        EDA(df_ml)
        .explorar(head_n=5)
        .limpieza_inicial(nonusecode_fill="UKNOWN", num_fill=0.0)
        .add_date_features(date_col="daterecorded",
                           year_name="sale_year",
                           month_num_name="sale_month_num",
                           month_name="sale_month")
        .build_schema_lists()
    )

    # Objetivo
    print("\n[Objetivo] saleamount")
    eda.target_summary("saleamount")
    eda.target_plots("saleamount")  # hist/box log, qq, por 'town'

    # Numéricas
    print("\n[Numéricas] describe / std-var / plots log")
    print(eda.nums_describe(exclude=["sale_id", "saleamount"]))
    print(eda.nums_std_var(exclude=["sale_id", "saleamount"]))
    eda.nums_plots_log(exclude=["sale_id", "saleamount"], max_plots=8)

    # Categóricas
    print("\n[Categóricas] tablas de frecuencia")
    eda.freq_tables(eda.cat_cols)
    eda.cat_vs_target_plots(target="saleamount", cat="residentialtype", top=10)
    eda.cat_vs_target_plots(target="saleamount", cat="sale_month", top=12)

    # Correlación
    print("\n[Correlación] heatmap")
    num_cols = eda.nums_select(exclude=["sale_id", "saleamount"])
    eda.corr_heatmap(num_cols, method="pearson")

    print("\nEDA COMPLETO ✅")

if __name__ == "__main__":
    main()