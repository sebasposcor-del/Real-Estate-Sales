# EDA/EDA_class.py
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EDA:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []

    def explorar(self, head_n: int = 5):
        """Exploración inicial: shape, head y describe del df."""
        print("shape:", self.df.shape)
        print(self.df.head(head_n))
        try:
            print(self.df.describe())
        except Exception as e:
            print("describe() falló:", e)
        return self

    def resumen_nulls(self) -> pl.DataFrame:
        """Conteo de nulos por columna."""
        return self.df.select(pl.all().null_count())

    def limpieza_inicial(self, nonusecode_fill: str = "UKNOWN", num_fill: float = 0.0):
        """Rellena nulos en nonusecode/lat/long y filtra filas con salesratio y daterecorded no nulos."""
        self.df = self.df.with_columns([
            pl.col("nonusecode").fill_null(nonusecode_fill),
            pl.col("latitude").fill_null(num_fill),
            pl.col("longitude").fill_null(num_fill),
        ])
        self.df = self.df.filter(pl.col("salesratio").is_not_null() & pl.col("daterecorded").is_not_null())
        return self

    def add_date_features(
        self,
        date_col: str = "daterecorded",
        year_name: str = "sale_year",
        month_num_name: str = "sale_month_num",
        month_name: str = "sale_month",
        drop_prev_month_cols: list[str] = ["sale_month", "sale_month_cat"],
    ):
        """Crea year y month_num; crea month abreviado categórico y elimina columnas previas de mes si existieran."""
        self.df = self.df.with_columns([
            pl.col(date_col).dt.year().alias(year_name),
            pl.col(date_col).dt.month().alias(month_num_name),
        ])
        self.df = (
            self.df
            .with_columns(pl.col(date_col).dt.strftime("%b").alias("sale_month_name"))
            .with_columns(pl.col("sale_month_name").cast(pl.Categorical))
        )
        self.df = self.df.drop(drop_prev_month_cols, strict=False).rename({"sale_month_name": month_name})
        return self

    def build_schema_lists(self):
        """Construye listas de columnas según tipo."""
        schema = self.df.schema
        self.num_cols = [c for c, dt in schema.items() if dt in (pl.Int64, pl.Float64, pl.Int32)]
        self.cat_cols = [c for c, dt in schema.items() if dt in (pl.Utf8, pl.Categorical)]
        return self 

    def head(self, n=5) -> pl.DataFrame:
        return self.df.head(n)

    def get(self) -> pl.DataFrame:
        return self.df

    # ===============   ANÁLISIS DESCRIPTIVO   =============
    def qqplot(self, col: str, log: bool = False):
        """QQ-plot de una columna; opción log1p."""
        x = self.df[col].drop_nulls().to_numpy()
        if x.size == 0:
            print(f"[{col}] sin datos."); return
        if log:
            x = np.log1p(x)
            title = f"QQ-plot {col} (log1p)"
        else:
            title = f"QQ-plot {col}"
        stats.probplot(x, dist="norm", plot=plt)
        plt.title(title); plt.tight_layout(); plt.show()
    
    def nums_plots_log(self, exclude: list[str] = None, max_plots: int | None = None):
        """Hist+box (log1p) para cada columna numérica seleccionada."""
        if exclude is None:
            exclude = ["sale_id", "saleamount"]
        cols = self.nums_select(exclude=exclude)
        if max_plots is not None:
            cols = cols[:max_plots]
        for c in cols:
            print(f"[Plot] {c} (log1p)")
            self.hist_box_log(c)

    def hist_box_log(self, col: str):
        """Hist + boxplot del log1p de una columna."""
        x = self.df[col].drop_nulls().to_numpy()
        x = np.log1p(x)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,5),
                                       gridspec_kw={"height_ratios":[3,1]},
                                       sharex=True)
        ax1.hist(x, bins="auto", color="#2a9d8f"); ax1.set_title(f"{col} (log1p)")
        ax2.boxplot(x, vert=False)
        plt.tight_layout(); plt.show()

    def hist_box_by_cat(self, y: str, cat: str, top: int = 10):
        """Hist y boxplot (log1p) de y separados por categoría."""
        d = (self.df
             .filter(pl.col(y).is_not_null() & pl.col(cat).is_not_null())
             .select([y, cat]))
        cats = (d.group_by(cat).len().sort("len", descending=True).head(top)[cat].to_list())
        d = d.filter(pl.col(cat).is_in(cats)).to_pandas()
        d[y] = np.log1p(d[y])

        plt.figure(figsize=(10,4))
        sns.histplot(data=d, x=y, hue=cat, element="step", stat="density", common_norm=False)
        plt.title(f"{y} (log1p) - hist por {cat}")
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(10,4))
        sns.boxplot(data=d, x=cat, y=y, hue=cat, dodge=False, fliersize=1)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{y} (log1p) - boxplot por {cat}")
        plt.tight_layout(); plt.show()

    def bar_by_cat(self, y: str, cat: str, top: int = 20):
        """Barplot de la media de y por categoría (ordenado, y-scale log)."""
        d = (self.df
             .filter(pl.col(y).is_not_null() & pl.col(cat).is_not_null())
             .select([y, cat]))
        cats = (d.group_by(cat).len().sort("len", descending=True).head(top)[cat].to_list())
        d = d.filter(pl.col(cat).is_in(cats)).to_pandas()
        order = d.groupby(cat)[y].mean().sort_values(ascending=False).index
        plt.figure(figsize=(10,4))
        sns.barplot(data=d, x=cat, y=y, order=order, color="#2a9d8f")
        plt.yscale("log"); plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.show()

    def target_summary(self, col: str):
        """Describe, std y var de la variable objetivo."""
        print(self.df.select(pl.col(col)).describe())
        print("\n=== STD ==="); print(self.df.select(pl.col(col)).std())
        print("\n=== VAR ==="); print(self.df.select(pl.col(col)).var())

    def target_plots(self, col: str):
        """Hist+box (log) y QQ-plot (log) para objetivo; además por 'town'."""
        self.hist_box_log(col)
        self.qqplot(col, log=True)
        self.hist_box_by_cat(col, "town")
        self.bar_by_cat(col, "town")

    # ===============   NUMÉRICAS   =============
    def nums_select(self, exclude: list[str] = None) -> list[str]:
        if exclude is None:
            exclude = []
        if getattr(self, "num_cols", None):
            cols = [c for c in self.num_cols if c not in exclude]
        else:
            schema = self.df.schema
            cols = [c for c, dt in schema.items() if dt in (pl.Int64, pl.Float64, pl.Int32) and c not in exclude]
        return cols

    def nums_describe(self, exclude: list[str] = None) -> pl.DataFrame:
        if exclude is None:
            exclude = ["sale_id"]
        cols = self.nums_select(exclude=exclude)
        if not cols:
            print("No hay columnas numéricas tras exclusión."); return pl.DataFrame()
        return self.df.select([pl.col(c) for c in cols]).describe()

    def nums_std_var(self, exclude: list[str] = None) -> pl.DataFrame:
        if exclude is None:
            exclude = ["sale_id"]
        cols = self.nums_select(exclude=exclude)
        if not cols:
            return pl.DataFrame()
        exprs_std = [(pl.col(c).cast(pl.Float64).std()).alias(f"{c}__std") for c in cols]
        exprs_var = [(pl.col(c).cast(pl.Float64).var()).alias(f"{c}__var") for c in cols]
        stats_row = self.df.select(exprs_std + exprs_var)
        long_rows = []
        for c in cols:
            long_rows.append((c, stats_row[f"{c}__std"][0], stats_row[f"{c}__var"][0]))
        return pl.DataFrame(long_rows, schema=["col", "std", "var"]).sort("std", descending=True)

    def corr_vs_target(self, target: str, num_cols: list[str] | None = None) -> pl.DataFrame:
        if num_cols is None:
            num_cols = self.nums_select(exclude=[])
        cols = [c for c in num_cols if c != target]
        if not cols:
            return pl.DataFrame(schema=["feature", "corr"], data=[])
        exprs = [pl.corr(pl.col(c).cast(pl.Float64), pl.col(target).cast(pl.Float64)).alias(c) for c in cols]
        corr_row = self.df.select(exprs)
        melted = corr_row.melt(variable_name="feature", value_name="corr")
        return melted.sort("corr", descending=True)

    # ===============   CATEGÓRICAS   =============
    def freq_tables(self, cat_cols):
        """Imprime tabla de frecuencias (obs y %) para cada categórica; salta binarias."""
        for c in cat_cols:
            n = self.df[c].n_unique()
            if n <= 2:
                continue
            tab = (
                self.df.group_by(c).len().rename({"len": "obs"})
                      .sort("obs", descending=True)
                      .with_columns((pl.col("obs") * 100 / pl.col("obs").sum()).alias("perc"))
            )
            print(f"\n {c} ({n} categorías)")
            print(tab)

    def cat_vs_target_plots(self, target: str, cat: str, top: int = 10):
        """Hist+box (log1p) por categoría y barplot (media target, y log)."""
        print(f"\n=== {cat} vs {target} ===")
        self.hist_box_by_cat(y=target, cat=cat, top=top)
        self.bar_by_cat(y=target, cat=cat, top=top)

    # ===============   CORRELACIÓN   =============
    def corr_matrix(self, num_cols, method: str = "pearson"):
        """Matriz de correlación vía pandas.corr()."""
        return self.df.select(num_cols).to_pandas().corr(method=method)

    def corr_heatmap(self, num_cols, method: str = "pearson"):
        """Heatmap triángulo inferior y devuelve la matriz."""
        C = self.corr_matrix(num_cols, method)
        mask = np.triu(np.ones_like(C, dtype=bool))
        plt.figure(figsize=(0.6 * len(C), 0.6 * len(C)))
        sns.heatmap(C, mask=mask, cmap="vlag", center=0, linewidths=.5)
        plt.title(f"Correlación ({method})")
        plt.tight_layout(); plt.show()
        return C