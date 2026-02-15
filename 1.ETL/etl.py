from pathlib import Path
from pprint import pprint

# Importar clases
from extract import Extract
from transform import Transform
from load import Load


def main():

    print("INICIANDO ETL")

    # EXTRACT

    #print("EXTRACT")

    extractor = Extract()  

    #print("-> Extrayendo datos desde API a MongoDB...")
    #extractor.fetch_from_api()

    #print("-> Cargando datos desde MongoDB a Polars...")
    pl_df_raw = extractor.load_from_mongo()

    print(f"Datos cargados: {pl_df_raw.shape[0]} filas")

    # TRANSFORM
    print("TRANSFORM")

    columns_to_drop = [
    ':@computed_region_dam5_q64j',
    ':@computed_region_nhmp_cq6b',
    ':@computed_region_m4y2_whse',
    ':@computed_region_snd5_k6zv'
]

    transformer = Transform(pl_df_raw)

    pl_clean = (
        transformer
        .drop_columns(columns_to_drop)
        .convert_types()
        .convert_dates()
        .extract_coordinates()
        .clean_categoricals()
        .get_clean()
    )

    print("Transformación completada")

    print("-> Generando tablas dimensionales y fact...")
    tables = transformer.create_tables()

    for name in tables:
        print(f"   ✔ {name}: {tables[name].shape[0]} filas")


    # LOAD
    print("LOAD")

    loader = Load()  

    loader.run_load(
        non_use_df=tables["non_use_code_dim"],
        property_df=tables["property_dim"],
        sales_df=tables["sales_fact"]
    )

    print("\n ETL COMPLETO. Datos cargados.")


# ENTRYPOINT

if __name__ == "__main__":
    main()







