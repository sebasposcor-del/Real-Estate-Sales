import polars as pl
import pandas as pd
from pprint import pprint

class Transform:
    def __init__(self, pl_df: pl.DataFrame):
        self.pl_df = pl_df
        self.pl_clean = pl_df.clone()
        
    def drop_columns(self, columns_to_drop: list):
        self.pl_clean = self.pl_clean.drop(columns_to_drop)
        print(f"Columnas eliminadas: {columns_to_drop}")
        return self

    def convert_types(self):
        self.pl_clean = self.pl_clean.with_columns([
            pl.col("listyear").cast(pl.Int64, strict=False),
            pl.col("assessedvalue").cast(pl.Float64, strict=False),
            pl.col("saleamount").cast(pl.Float64, strict=False),
            pl.col("salesratio").cast(pl.Float64, strict=False),
        ])
        print("Conversiones completadas")
        pprint(self.pl_clean.schema)
        return self

    def convert_dates(self):
        self.pl_clean = self.pl_clean.with_columns(
            pl.col("daterecorded")
              .str.strptime(pl.Datetime, strict=False)
              .dt.date()
              .alias("daterecorded")
        )
        print("Fechas convertidas correctamente")
        return self

    def extract_coordinates(self):
        self.pl_clean = (
            self.pl_clean
            .with_columns([
                pl.col("geo_coordinates")
                  .struct.field("coordinates")
                  .list.get(0)
                  .alias("longitude"),

                pl.col("geo_coordinates")
                  .struct.field("coordinates")
                  .list.get(1)
                  .alias("latitude"),
            ])
            .drop("geo_coordinates")
        )
        print("Coordenadas extraídas")
        return self

    def clean_categoricals(self):
        self.pl_clean = self.pl_clean.with_columns([
            pl.col("town").str.strip_chars().str.to_uppercase(),
            pl.col("address").str.strip_chars().str.to_uppercase(),
            pl.col("propertytype").str.strip_chars(),
            pl.col("residentialtype").str.strip_chars(),
        ])
        print("Categorías limpiadas")
        return self

    def get_clean(self):
        return self.pl_clean
    
    def create_tables(self):
        pl_tablas = self.pl_clean.clone()

        # town_dim
        town_dim = (
            pl_tablas.select(pl.col("town").fill_null("UNKNOWN"))
                     .unique()
                     .sort("town")
        )

        # property_dim
        property_dim = (
            pl_tablas.select([
                pl.col("town"),
                pl.col("address").fill_null("UNKNOWN"),
                pl.col("propertytype").fill_null("UNKNOWN"),
                pl.col("residentialtype").fill_null("UNKNOWN"),
                pl.col("latitude"),
                pl.col("longitude"),
            ])
            .unique()
            .sort(["town", "address"])
        )

        # non_use_code_dim
        non_use_code_dim = (
            pl_tablas.select(pl.col("nonusecode").fill_null("UNKNOWN"))
                     .unique()
                     .sort("nonusecode")
        )

        # sale_notes
        sale_notes = (
            pl_tablas.select([
                pl.col("serialnumber"),
                pl.col("remarks").fill_null("UNKNOWN"),
                pl.col("opm_remarks").fill_null("UNKNOWN"),
            ])
        )

        # sales_fact
        sales_fact = (
            pl_tablas.select([
                pl.col("serialnumber"),
                pl.col("listyear"),
                pl.col("daterecorded"),
                pl.col("assessedvalue"),
                pl.col("saleamount"),
                pl.col("salesratio"),
                pl.col("town"),
                pl.col("address"),
                pl.col("propertytype"),
                pl.col("residentialtype"),
                pl.col("latitude"),
                pl.col("longitude"),
                pl.col("nonusecode"),
            ])
        )

        return {
            "town_dim": town_dim,
            "property_dim": property_dim,
            "non_use_code_dim": non_use_code_dim,
            "sale_notes": sale_notes,
            "sales_fact": sales_fact
        }