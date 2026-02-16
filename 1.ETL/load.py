import os
import io
from urllib.parse import urlparse
from dotenv import load_dotenv, find_dotenv
import psycopg2
import polars as pl

class Load:

    def __init__(self):
        self.conn = self.get_connection()
        print("Conexión establecida con Postgres")

    # Conexión
    def get_connection(self):
        env_path = find_dotenv(usecwd=True)
        load_dotenv(env_path, override=True)

        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("Falta DATABASE_URL en .env")

        p = urlparse(url)
        print(f"Conectando a: {p.hostname}:{p.port}")

        conn = psycopg2.connect(url)
        return conn
# ==========================================================
# Dimensión Non Use Code


    def load_non_use_code(self, df: pl.DataFrame):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS d_non_use_code (
                    nonusecode_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    nonusecode TEXT UNIQUE NOT NULL
                );
            """)

            buf = io.StringIO()
            df.select("nonusecode").unique().write_csv(buf, include_header=False)
            buf.seek(0)

            cur.execute("DROP TABLE IF EXISTS tmp_non_use_code;")
            cur.execute("""
                CREATE TEMP TABLE tmp_non_use_code (
                    nonusecode TEXT
                ) ON COMMIT DROP;
            """)

            cur.copy_from(buf, 'tmp_non_use_code', columns=('nonusecode',), sep=',')

            cur.execute("""
                INSERT INTO d_non_use_code (nonusecode)
                SELECT DISTINCT nonusecode
                FROM tmp_non_use_code
                WHERE nonusecode IS NOT NULL AND nonusecode <> ''
                ON CONFLICT (nonusecode) DO NOTHING;
            """)

        self.conn.commit()
        print("✔ d_non_use_code cargada")


    # ==========================================================
    # Dimensión Property

    def load_property(self, df: pl.DataFrame):
        with self.conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS d_property(
                    property_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    address TEXT NOT NULL,
                    town_id BIGINT NOT NULL,
                    propertytype TEXT,
                    residentialtype TEXT,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    CONSTRAINT uq_property UNIQUE (address, town_id)
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS d_town(
                    town_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    town TEXT UNIQUE NOT NULL
                );
            """)

            buf = io.StringIO()
            df.unique().write_csv(buf, include_header=False)
            buf.seek(0)

            cur.execute("DROP TABLE IF EXISTS tmp_property;")
            cur.execute("""
                CREATE TEMP TABLE tmp_property (
                    address TEXT,
                    town TEXT,
                    propertytype TEXT,
                    residentialtype TEXT,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION
                ) ON COMMIT DROP;
            """)

            cur.copy_expert("""
                COPY tmp_property (
                    address, town, propertytype, residentialtype, latitude, longitude
                ) FROM STDIN WITH (FORMAT CSV)
            """, buf)

            # Insert towns
            cur.execute("""
                INSERT INTO d_town (town)
                SELECT DISTINCT town
                FROM tmp_property
                WHERE town IS NOT NULL AND town <> ''
                ON CONFLICT (town) DO NOTHING;
            """)

            # Insert properties
            cur.execute("""
                INSERT INTO d_property (
                    address, town_id, propertytype, residentialtype, latitude, longitude
                )
                SELECT 
                    p.address,
                    t.town_id,
                    p.propertytype,
                    p.residentialtype,
                    p.latitude,
                    p.longitude
                FROM tmp_property p
                JOIN d_town t ON t.town = p.town
                ON CONFLICT (address, town_id) DO NOTHING;
            """)

        self.conn.commit()
        print("✔ d_property cargada")

    # ===================================================================
    # Town
    def load_town(self, df: pl.DataFrame):
        """Carga la dimensión d_town con un proceso robusto para texto y caracteres especiales."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS d_town(
                    town_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    town TEXT UNIQUE NOT NULL
                );
            """)

            buf = io.StringIO()
            df.select("town").unique().write_csv(buf, include_header=False)
            buf.seek(0)

            cur.execute("DROP TABLE IF EXISTS tmp_town;")
            cur.execute("""
                CREATE TEMP TABLE tmp_town (
                    town TEXT
                ) ON COMMIT DROP;
            """)

            cur.copy_from(buf, 'tmp_town', columns=('town',), sep=',')

            cur.execute("""
                INSERT INTO d_town (town)
                SELECT DISTINCT town
                FROM tmp_town
                WHERE town IS NOT NULL AND town <> ''
                ON CONFLICT (town) DO NOTHING;
            """)

        self.conn.commit()
        print("d_town cargada")

    # ====================================================
    # Sales Notes
    def load_sale_notes(self, sale_notes: pl.DataFrame):
        """Carga la dimensión d_sale_notes con un proceso robusto para texto largo y caracteres especiales."""
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS d_sale_notes (
                note_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                serialnumber TEXT,
                remarks TEXT,
                opm_remarks TEXT
                );
            """)

            df = sale_notes.select([
                "serialnumber",
                "remarks",
                "opm_remarks"
            ]).unique()

            buf = io.StringIO()
            df.write_csv(buf, include_header=False)
            buf.seek(0)

            #  Tabla temporal
            cur.execute("""
                CREATE TEMP TABLE tmp_sale_notes (
                    serialnumber TEXT,
                    remarks TEXT,
                    opm_remarks TEXT
                ) ON COMMIT DROP;
            """)

            cur.copy_expert(
                "COPY tmp_sale_notes (serialnumber, remarks, opm_remarks) "
                "FROM STDIN WITH (FORMAT CSV)",
                buf
            )

            # Insert final
            cur.execute("""
                INSERT INTO d_sale_notes (serialnumber, remarks, opm_remarks)
                SELECT DISTINCT
                    serialnumber,
                    remarks,
                    opm_remarks
                FROM tmp_sale_notes
                WHERE serialnumber IS NOT NULL AND serialnumber <> '';
            """)

            self.conn.commit()

            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM d_sale_notes;")
                print("✔ d_sale_notes cargada. Filas:", cur.fetchone()[0])

    # ==========================================================
    # Fact Sales

    def load_sales(self, df: pl.DataFrame):

        with self.conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS f_sales (
                    sale_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    serialnumber TEXT,
                    listyear INT,
                    daterecorded DATE,
                    assessedvalue DOUBLE PRECISION,
                    saleamount DOUBLE PRECISION,
                    salesratio DOUBLE PRECISION,
                    property_id BIGINT,
                    nonusecode_id BIGINT
                );
            """)

            buf = io.StringIO()
            df.unique().write_csv(buf, include_header=False)
            buf.seek(0)

            cur.execute("DROP TABLE IF EXISTS tmp_f_sales;")
            cur.execute("""
                CREATE TEMP TABLE tmp_f_sales (
                    serialnumber TEXT,
                    listyear INT,
                    daterecorded TEXT,
                    assessedvalue TEXT,
                    saleamount TEXT,
                    salesratio TEXT,
                    town TEXT,
                    address TEXT,
                    propertytype TEXT,
                    residentialtype TEXT,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    nonusecode TEXT
                ) ON COMMIT DROP;
            """)

            cur.copy_expert("""
                COPY tmp_f_sales (
                    serialnumber, listyear, daterecorded,
                    assessedvalue, saleamount, salesratio,
                    town, address, propertytype,
                    residentialtype, latitude, longitude, nonusecode
                )
                FROM STDIN WITH (FORMAT CSV)
            """, buf)

            cur.execute("""
                INSERT INTO f_sales (
                    serialnumber, listyear, daterecorded,
                    assessedvalue, saleamount, salesratio,
                    property_id, nonusecode_id
                )
                SELECT
                    s.serialnumber,
                    s.listyear,
                    s.daterecorded::date,
                    NULLIF(s.assessedvalue, '')::double precision,
                    NULLIF(s.saleamount, '')::double precision,
                    NULLIF(s.salesratio, '')::double precision,
                    dp.property_id,
                    nuc.nonusecode_id
                FROM tmp_f_sales s
                JOIN d_town t ON t.town = s.town
                JOIN d_property dp 
                    ON dp.address = s.address
                AND dp.town_id = t.town_id
                LEFT JOIN d_non_use_code nuc
                    ON nuc.nonusecode = s.nonusecode;
            """)

        self.conn.commit()
        print("✔ f_sales cargada")
    # ======================================================
    #ML Table
    def load_ml_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS ML_Table (
                sale_id         BIGINT PRIMARY KEY,
                serialnumber    TEXT,
                listyear        INT,
                daterecorded    DATE,
                assessedvalue   DOUBLE PRECISION,
                saleamount      DOUBLE PRECISION,
                salesratio      DOUBLE PRECISION,
                address         TEXT,
                propertytype    TEXT,
                residentialtype TEXT,
                latitude        DOUBLE PRECISION,
                longitude       DOUBLE PRECISION,
                town            TEXT,
                nonusecode      TEXT,
                remarks_all     TEXT,
                opm_remarks_all TEXT
                );
            """)

            cur.execute("""
                WITH notes AS (
                    SELECT
                        serialnumber,
                        STRING_AGG(remarks, ' | ' ORDER BY note_id)     AS remarks_all,
                        STRING_AGG(opm_remarks, ' | ' ORDER BY note_id) AS opm_remarks_all
                    FROM d_sale_notes
                    GROUP BY serialnumber
                )
                INSERT INTO ML_Table (
                    sale_id,
                    serialnumber,
                    listyear,
                    daterecorded,
                    assessedvalue,
                    saleamount,
                    salesratio,
                    address,
                    propertytype,
                    residentialtype,
                    latitude,
                    longitude,
                    town,
                    nonusecode,
                    remarks_all,
                    opm_remarks_all
                )
                SELECT
                    f.sale_id,
                    f.serialnumber,
                    f.listyear,
                    f.daterecorded,
                    f.assessedvalue,
                    f.saleamount,
                    f.salesratio,
                    p.address,
                    p.propertytype,
                    p.residentialtype,
                    p.latitude,
                    p.longitude,
                    t.town,
                    n.nonusecode,
                    notes.remarks_all,
                    notes.opm_remarks_all
                FROM f_sales AS f
                JOIN d_property AS p
                    ON p.property_id = f.property_id
                JOIN d_town AS t
                    ON t.town_id = p.town_id
                LEFT JOIN d_non_use_code AS n
                    ON n.nonusecode_id = f.nonusecode_id
                LEFT JOIN notes
                    ON notes.serialnumber = f.serialnumber
                ON CONFLICT (sale_id) DO NOTHING;
            """)

        self.conn.commit()

        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ML_Table;")
            print(" ML_Table cargada. Filas:", cur.fetchone()[0])


    # ==========================================================
    # definir orquestador

    def run_load(self, non_use_df, property_df, sales_df):

        conn = self.get_connection()

        try:
            self.load_non_use_code(non_use_df)
            self.load_property(property_df)
            self.load_sales(sales_df)

        finally:
            self.conn.close()
            print("🔒 Conexión cerrada")
