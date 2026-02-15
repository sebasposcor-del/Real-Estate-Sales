
from sales_query import SalesQueryService
from tabulate import tabulate

def cli_search():
    # Inicializar servicio de consultas (toma DATABASE_URL del .env)
    service = SalesQueryService()
    
    print("📊 Bienvenido al Buscador de Real Estate Sales")
    print("Deja un filtro vacío si no quieres usarlo.\n")

    year = input("Año (list_year): ")
    year = int(year) if year.strip() else None

    min_price = input("Precio mínimo: ")
    min_price = float(min_price) if min_price.strip() else None

    max_price = input("Precio máximo: ")
    max_price = float(max_price) if max_price.strip() else None

    town = input("Ciudad: ").strip() or None
    property_type = input("Tipo de propiedad: ").strip() or None
    residential_type = input("Tipo residencial: ").strip() or None

    print("\nOrden:")
    print("1 - Precio")
    print("2 - Año")
    print("3 - Fecha venta")
    order_choice = input("Opción: ").strip()

    if order_choice == "1":
        order_by = "saleamount"
    elif order_choice == "2":
        order_by = "listyear"
    else:
        order_by = "daterecorded"

    top_n = input("Top N resultados: ")
    top_n = int(top_n) if top_n.strip() else None

    df = service.search_sales(
        year=year,
        min_price=min_price,
        max_price=max_price,
        town=town,
        property_type=property_type,
        residential_type=residential_type,
        order_by=order_by,
        top_n=top_n
    )

    if df.empty:
        print("\n❌ No se encontraron resultados.")
    else:
        print("\n✅ Resultados:\n")
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

    service.close()

