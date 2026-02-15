
import os
from dotenv import load_dotenv, find_dotenv
import psycopg
import pandas as pd

class SalesQueryService:
    
    def __init__(self):
        # Ajusta la ruta al .env real
        env_path = find_dotenv("1.ETL/.env")  # ruta relativa a tu proyecto
        if not env_path:
            raise FileNotFoundError("No se encontró .env en 1.ETL/")
        load_dotenv(env_path, override=True)

        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("Falta DATABASE_URL en .env")

        print(f"Conectando a: {url}")
        import psycopg
        self.conn = psycopg.connect(url)
        self.conn.autocommit = True
        
       
    def search_sales(
        self,
        year=None,
        month=None,
        min_price=None,
        max_price=None,
        town=None,
        property_type=None,
        residential_type=None,
        order_by="saleamount",
        top_n=None
    ):
        query = """
            SELECT 
                s.sale_id,
                s.saleamount,
                s.assessedvalue,
                s.salesratio,
                s.listyear,
                s.daterecorded,
                t.town,
                p.propertytype,
                p.residentialtype
            FROM f_sales s
            JOIN d_property p ON s.property_id = p.property_id
            JOIN d_town t ON p.town_id = t.town_id
            WHERE 1=1
        """

        conditions = []
        params = []

        if year is not None:
            conditions.append("AND s.listyear = %s")
            params.append(year)

        if min_price is not None:
            conditions.append("AND s.saleamount >= %s")
            params.append(min_price)

        if max_price is not None:
            conditions.append("AND s.saleamount <= %s")
            params.append(max_price)

        if town:
            conditions.append("AND TRIM(t.town) ILIKE %s")
            params.append(f"%{town.strip()}%")

        if property_type:
            conditions.append("AND TRIM(p.propertytype) ILIKE %s")
            params.append(f"%{property_type.strip()}%")

        if residential_type:
            conditions.append("AND TRIM(p.residentialtype) ILIKE %s")
            params.append(f"%{residential_type.strip()}%")

        query += " ".join(conditions)
        query += f" ORDER BY {order_by} DESC"

        if top_n is not None:
            query += f" LIMIT {top_n}"

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            results = cur.fetchall()

        df = pd.DataFrame(results, columns=columns)
        return df

    def close(self):
        self.conn.close()
