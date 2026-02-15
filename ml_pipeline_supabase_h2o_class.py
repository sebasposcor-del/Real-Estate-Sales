# ml_pipeline_supabase_h2o_class.py
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import psycopg2

import h2o
from h2o.automl import H2OAutoML



# Config

@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuración central del pipeline (Data Engineering + Data Science).

    Esta clase agrupa todos los parámetros para que el pipeline sea:
    - Reproducible (seed, splits, límites de AutoML).
    - Fácil de ajustar (chunk_size, columnas, filtros SQL) sin tocar la lógica.
    - Seguro en credenciales: DATABASE_URL se lee desde un .env y no se hardcodea.

    Componentes:
    - Conexión: ruta del .env y clave de entorno.
    - Fuente: tabla en Supabase/Postgres y PK para paginar.
    - Ingesta por chunks: tamaño del batch y punto inicial.
    - Modelado: target, features y configuración de H2O AutoML.
    - SQL: columnas a extraer y filtros de calidad mínimos.
    """
    # env
    env_path: str = ".env"
    db_env_key: str = "DATABASE_URL"

    # Supabase table
    source_table: str = "public.ml_table"
    pk_col: str = "sale_id"

    # chunking
    chunk_size: int = 100_000
    start_sale_id: int = -1  # pon 0 o 1 si tus sale_id empiezan ahí

    # H2O / AutoML
    h2o_max_mem: str = "24G"
    seed: int = 42
    split_train_valid: Tuple[float, float] = (0.7, 0.15)
    max_models: int = 15
    max_runtime_secs: int = 1800
    sort_metric: str = "RMSE"

    # columns
    target: str = "saleamount"
    target_log: str = "saleamount_log"

    # Importante: sale_id NO entra al modelo; town tampoco (como tu versión original)
    features: Tuple[str, ...] = ("assessedvalue", "propertytype", "residentialtype", "listyear")
    categorical: Tuple[str, ...] = ("propertytype", "residentialtype")

    # solo columnas que pediste traer
    select_cols: Tuple[str, ...] = (
        "sale_id",
        "saleamount",
        "assessedvalue",
        "propertytype",
        "residentialtype",
        "town",
        "listyear",
    )

    # filtros 
    where_filters: str = """
        saleamount IS NOT NULL
        AND assessedvalue IS NOT NULL
        AND saleamount > 2000
        AND assessedvalue > 0
        AND listyear IS NOT NULL
    """



# Pipeline

class MLPipeline:
    """
    Pipeline end-to-end: Supabase (Postgres) -> ingesta incremental -> H2O AutoML -> evaluación.

    Qué hace (visión de ingeniería):
    1) Lee DATABASE_URL desde .env (credenciales fuera del código).
    2) Verifica conectividad a Supabase (consulta liviana).
    3) Extrae datos desde public.ml_table en batches (chunks) usando keyset pagination:
       - sale_id > last_sale_id
       - ORDER BY sale_id
       - LIMIT chunk_size
       Esto evita la degradación típica de OFFSET en tablas grandes.
    4) Importa cada chunk a H2O y lo acumula en un H2OFrame único (self.hf).
       Los CSV se generan como temporales y se eliminan al terminar.
    5) Feature engineering:
       - Crea el target log: saleamount_log = log(saleamount + 1).
       - Castea categorías (propertytype, residentialtype) a factor.
    6) Split train/valid/test y entrenamiento con H2OAutoML.
    7) Reporte de métricas en test (RMSE/MAE/R2 en escala log) + leaderboard.
    8) Diagnóstico básico: scatter pred vs real (log) y variable importance del mejor GBM.


    """

    def __init__(self, cfg: PipelineConfig):
        """
        Inicializa el pipeline y reserva atributos que se llenan durante run().

        Args:
            cfg: instancia de PipelineConfig con parámetros de conexión, ingesta y modelado.
        """
        self.cfg = cfg
        self.db_url: Optional[str] = None

        # H2OFrame completo (acumulado) y splits
        self.hf = None
        self.train = None
        self.valid = None
        self.test = None

        # AutoML y modelo ganador
        self.aml: Optional[H2OAutoML] = None
        self.leader = None

    # Supabase connection helpers 
    def load_env(self) -> str:
        """
        Carga el archivo .env y obtiene DATABASE_URL.

        Motivación (DE/ops):
        - Mantener credenciales fuera del repo/código.
        - Facilitar ejecuciones en distintos entornos (local, docker, CI).

        Returns:
            El DATABASE_URL leído del entorno.

        Raises:
            RuntimeError: si no existe la variable DATABASE_URL en el .env.
        """
        load_dotenv(self.cfg.env_path)
        url = os.getenv(self.cfg.db_env_key)
        if not url:
            raise RuntimeError(f"No encuentro {self.cfg.db_env_key} en {self.cfg.env_path}")
        self.db_url = url
        return url

    def _connect(self):
        """
        Abre una conexión psycopg2 a Supabase usando DATABASE_URL.

        Returns:
            Conexión psycopg2 lista para ejecutar queries y COPY.

        Raises:
            RuntimeError: si db_url no fue inicializado.
        """
        if not self.db_url:
            raise RuntimeError("db_url no inicializado. Llama primero a load_env().")
        return psycopg2.connect(self.db_url)

    def test_connection(self) -> None:
        """
        Verifica la conectividad a la base en Supabase con una consulta liviana.

        Qué valida:
        - DNS/host/puerto correctos
        - credenciales válidas
        - sesión estable para ejecutar operaciones posteriores

        Imprime:
        - host y puerto
        - usuario actual, puerto servidor y timestamp (now())

        Raises:
            RuntimeError: si db_url no fue inicializado.
        """
        if not self.db_url:
            raise RuntimeError("db_url no inicializado. Llama primero a load_env().")

        p = urlparse(self.db_url)
        print("Conectando a:", p.hostname, p.port)

        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT current_user, inet_server_port(), now();")
        print("OK:", cur.fetchone())
        cur.close()
        conn.close()

    # H2O 
    def h2o_start(self) -> None:
        """
        Inicia el cluster local de H2O.

        Notas:
        - H2O gestiona su propio backend (memoria + posible spill a disco).
        - max_mem_size controla el presupuesto de memoria del cluster.
        - nthreads=-1 usa todos los hilos disponibles.

        Esto es clave para trabajar con datasets grandes sin depender
        de que todo quepa en memoria del proceso Python.
        """
        h2o.init(nthreads=-1, max_mem_size=self.cfg.h2o_max_mem)

    #  Chunk query (keyset by sale_id) 
    def _build_chunk_query(self, last_sale_id: int) -> str:
        """
        Construye un SELECT paginado por keyset usando sale_id (PK).

        Estrategia:
        - Keyset pagination (recomendada en grandes volúmenes)
        - Condición: sale_id > last_sale_id
        - Orden: ORDER BY sale_id
        - Límite: LIMIT chunk_size

        Ventaja sobre OFFSET:
        - OFFSET se vuelve progresivamente más lento a offsets grandes.
        - Keyset aprovecha el índice de la PK y mantiene rendimiento estable.

        Args:
            last_sale_id: último sale_id procesado; el siguiente chunk empieza después de este valor.

        Returns:
            Query SQL (string) lista para envolver en COPY(...).

        Seguridad:
        - forzamos int(last_sale_id) para asegurar que sea numérico y evitar inyección.
        """
        last_sale_id = int(last_sale_id)

        cols = ", ".join(self.cfg.select_cols)
        pk = self.cfg.pk_col
        where = self.cfg.where_filters.strip()

        return f"""
        SELECT {cols}
        FROM {self.cfg.source_table}
        WHERE {where}
          AND {pk} > {last_sale_id}
        ORDER BY {pk}
        LIMIT {self.cfg.chunk_size}
        """.strip()

    def export_chunk_to_csv(self, conn, last_sale_id: int, csv_path: str) -> int:
        """
        Exporta un chunk desde Supabase a CSV usando COPY TO STDOUT.

        Por qué COPY:
        - Es el método más eficiente en Postgres para exportar grandes volúmenes.
        - Evita traer fila por fila vía cursor normal.

        Flujo:
        1) Genera el SELECT paginado (keyset).
        2) Ejecuta: COPY (SELECT...) TO STDOUT WITH CSV HEADER.
        3) Cuenta filas del CSV (restando header) para saber si hay más datos.

        Args:
            conn: conexión psycopg2 abierta.
            last_sale_id: cursor de paginación.
            csv_path: ruta del archivo CSV temporal.

        Returns:
            Número de filas exportadas (sin incluir el header).
        """
        query = self._build_chunk_query(last_sale_id)
        copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER"

        with conn.cursor() as cur, open(csv_path, "w", encoding="utf-8", newline="") as f:
            cur.copy_expert(copy_sql, f)

        # contar filas (sin cargar dataset completo)
        with open(csv_path, "r", encoding="utf-8") as f:
            nrows = sum(1 for _ in f) - 1
        return max(nrows, 0)

    def get_last_sale_id_from_csv(self, csv_path: str) -> int:
        """
        Obtiene el mayor sale_id exportado en el chunk actual.

        Idea:
        - Como el SELECT se exporta con ORDER BY sale_id,
          la última fila del CSV tiene el mayor sale_id.
        - Esto permite avanzar el cursor de paginación sin hacer consultas extra.

        Args:
            csv_path: CSV temporal exportado del chunk.

        Returns:
            El sale_id máximo del chunk; si el chunk está vacío, retorna -1.

        Raises:
            RuntimeError: si la columna PK no existe en el header del CSV.
        """
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            if self.cfg.pk_col not in header:
                raise RuntimeError(f"No encuentro columna {self.cfg.pk_col} en el CSV.")
            pk_idx = header.index(self.cfg.pk_col)

            last_line = None
            for line in f:
                last_line = line

        if not last_line:
            return -1

        return int(last_line.strip().split(",")[pk_idx])

    def append_chunk_to_h2o(self, csv_path: str) -> None:
        """
        Importa un CSV chunk a H2O y lo concatena al H2OFrame acumulado (self.hf).

        Implementación:
        - h2o.import_file(csv_path) crea un H2OFrame del chunk.
        - rbind concatena filas (append vertical).

        Nota:
        - self.hf vive dentro del cluster H2O y puede gestionar datasets grandes.
        """
        hf_chunk = h2o.import_file(csv_path)
        self.hf = hf_chunk if self.hf is None else self.hf.rbind(hf_chunk)

    def load_frame_in_chunks(self) -> None:
        """
        Construye self.hf trayendo datos desde Supabase por chunks (batch ingestion).

        Enfoque:
        - Genera CSVs temporales por chunk (en un directorio temporal).
        - Cada CSV se importa a H2O inmediatamente y se concatena.
        - No se crea un CSV final gigante en disco local.

        Beneficios (optimización de proceso):
        - Reduce el uso de almacenamiento local (archivos temporales que se eliminan).
        - Evita el overhead de manejar un artefacto grande persistente.
        - Mantiene extracción eficiente gracias a keyset pagination por sale_id.

        Estado:
        - last_sale_id actúa como cursor y se actualiza con el máximo sale_id del chunk.
        """
        conn = self._connect()
        last_sale_id = self.cfg.start_sale_id

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                total_rows = 0

                while True:
                    csv_path = os.path.join(tmpdir, f"chunk_after_{last_sale_id}.csv")
                    nrows = self.export_chunk_to_csv(conn, last_sale_id, csv_path)

                    if nrows <= 0:
                        print("No hay más filas. Fin de ingest por chunks.")
                        break

                    self.append_chunk_to_h2o(csv_path)
                    total_rows += nrows

                    new_last = self.get_last_sale_id_from_csv(csv_path)
                    print(f"Chunk ok | rows={nrows} | total={total_rows} | last_sale_id={new_last}")

                    last_sale_id = new_last

                    if nrows < self.cfg.chunk_size:
                        print("Último chunk incompleto. Fin.")
                        break
        finally:
            conn.close()

    # Feature engineering 
    def add_log_target(self) -> None:
        """
        Crea el target en escala logarítmica: saleamount_log = log(saleamount + 1).

        Motivación (optimización de modelo):
        - Los precios inmobiliarios suelen ser altamente asimétricos (colas largas).
        - El log reduce la influencia de outliers y estabiliza la varianza.
        - Mejora la estabilidad de métricas tipo RMSE/MAE y la generalización.

        Nota:
        - Se usa +1 para evitar log(0).
        - El entrenamiento/evaluación se realiza en escala log.

        Raises:
            RuntimeError: si self.hf aún no fue construido.
        """
        if self.hf is None:
            raise RuntimeError("No hay H2OFrame. Llama primero a load_frame_in_chunks().")
        y = self.cfg.target
        self.hf[self.cfg.target_log] = (self.hf[y] + 1).log()

    def cast_categoricals(self) -> None:
        """
        Convierte columnas categóricas a factor dentro de H2O.

        Motivación:
        - Evita que H2O interprete categorías como numéricas.
        - Permite que AutoML aplique estrategias adecuadas para categóricas.

        Columnas a convertir:
        - propertytype, residentialtype
        """
        if self.hf is None:
            raise RuntimeError("No hay H2OFrame.")
        for c in self.cfg.categorical:
            if c in self.hf.columns:
                self.hf[c] = self.hf[c].asfactor()

    # Split / Train / Eval 
    def split(self) -> None:
        """
        Divide el H2OFrame en train/valid/test con proporciones fijas y seed.

        Implementación:
        - split_frame([train_ratio, valid_ratio], seed=seed)
        - test queda como el residual (1 - train - valid)

        Motivación:
        - Validación separada para AutoML (selección de modelos).
        - Test independiente para reportar performance final.
        """
        if self.hf is None:
            raise RuntimeError("No hay H2OFrame.")
        tr, va = self.cfg.split_train_valid
        self.train, self.valid, self.test = self.hf.split_frame([tr, va], seed=self.cfg.seed)

    def train_automl(self) -> None:
        """
        Entrena H2O AutoML sobre el target log usando train/valid.

        Configuración:
        - max_models: limita cantidad de modelos (control de tiempo y complejidad).
        - max_runtime_secs: límite duro de tiempo.
        - sort_metric: RMSE (en el target log), para ordenar/seleccionar.

        Importante:
        - x = features definidas en config (NO incluye sale_id).
        - y = saleamount_log (target transformado).

        Resultado:
        - self.aml guarda el objeto AutoML (leaderboard + modelos).
        - self.leader guarda el mejor modelo según la métrica.
        """
        if self.train is None or self.valid is None:
            raise RuntimeError("Splits no creados. Llama primero a split().")

        self.aml = H2OAutoML(
            max_models=self.cfg.max_models,
            max_runtime_secs=self.cfg.max_runtime_secs,
            seed=self.cfg.seed,
            sort_metric=self.cfg.sort_metric,
        )
        self.aml.train(
            x=list(self.cfg.features),      # NO incluye sale_id
            y=self.cfg.target_log,          # target log
            training_frame=self.train,
            validation_frame=self.valid,
        )
        self.leader = self.aml.leader

    def evaluate(self) -> Dict[str, float]:
        """
        Evalúa el modelo líder en el set de test (en escala log).

        Métricas:
        - RMSE: penaliza errores grandes; útil para comparar modelos.
        - MAE: más robusto; error medio absoluto.
        - R2: proporción de varianza explicada (en escala log).

        Returns:
            Dict con rmse/mae/r2.

        Raises:
            RuntimeError: si leader o test no están listos.
        """
        if self.leader is None or self.test is None:
            raise RuntimeError("Leader/Test no listos.")
        perf = self.leader.model_performance(self.test)
        metrics = {"rmse": perf.rmse(), "mae": perf.mae(), "r2": perf.r2()}

        print("Leader:", self.leader.model_id)
        print("Test RMSE (log):", metrics["rmse"])
        print("Test MAE  (log):", metrics["mae"])
        print("Test R2   (log):", metrics["r2"])
        return metrics

    def show_leaderboard(self, top_n: int = 10) -> None:
        """
        Imprime el leaderboard de AutoML (top_n modelos).

        Útil para:
        - comparar familias (GBM, ensembles, etc.)
        - justificar por qué el leader es el mejor bajo la métrica elegida
        """
        if self.aml is None:
            raise RuntimeError("AutoML no entrenado.")
        print(self.aml.leaderboard.head(rows=top_n))

    def plot_pred_vs_true_log(self, alpha: float = 0.15) -> None:
        """
        Scatter plot de Predicción vs Real en escala log, con línea y=x.

        Objetivo:
        - Diagnóstico visual rápido de calibración:
          si el modelo está bien calibrado, los puntos se agrupan alrededor de la diagonal.
        - Detectar sesgos sistemáticos (sobre/infra-predicción).

        Args:
            alpha: transparencia de los puntos (útil con muchos registros).

        Raises:
            RuntimeError: si leader o test no están listos.
        """
        if self.leader is None or self.test is None:
            raise RuntimeError("Leader/Test no listos.")

        pred = self.leader.predict(self.test)
        y_true = self.test[self.cfg.target_log].as_data_frame(use_pandas=True).iloc[:, 0].to_numpy()
        y_pred = pred["predict"].as_data_frame(use_pandas=True).iloc[:, 0].to_numpy()

        plt.figure()
        plt.scatter(y_true, y_pred, alpha=alpha)
        mn = np.nanmin([y_true.min(), y_pred.min()])
        mx = np.nanmax([y_true.max(), y_pred.max()])
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("Real (log)")
        plt.ylabel("Predicción (log)")
        plt.title("Predicción vs Real (escala log)")
        plt.show()

    def best_gbm_varimp(self) -> None:
        """
        Encuentra el mejor modelo GBM dentro del leaderboard y muestra su importancia de variables.

        Por qué GBM y no el leader:
        - El leader puede ser un StackedEnsemble (menos interpretable).
        - Un GBM top suele ser casi igual de performante y mucho más interpretable.

        Flujo:
        1) Convierte leaderboard a dataframe.
        2) Filtra model_id que contienen "GBM".
        3) Toma el primero (mejor GBM según sort_metric).
        4) Imprime varimp y grafica varimp_plot().

        Nota:
        - varimp depende del tipo de modelo; GBM lo soporta bien.
        """
        if self.aml is None:
            raise RuntimeError("AutoML no entrenado.")

        lb = self.aml.leaderboard.as_data_frame()
        gbm_rows = lb[lb["model_id"].str.contains("GBM", na=False)]
        if gbm_rows.empty:
            print("No hay GBM en el leaderboard.")
            return

        best_gbm_id = gbm_rows.iloc[0]["model_id"]
        best_gbm = h2o.get_model(best_gbm_id)

        print("Best GBM:", best_gbm_id)
        vi = best_gbm.varimp(use_pandas=True)
        print(vi)
        best_gbm.varimp_plot()

    # Orchestrator 
    def run(self) -> Dict[str, float]:
        """
        Ejecuta el pipeline completo de manera determinista (end-to-end).

        Orden:
        1) load_env()          -> carga DATABASE_URL desde .env
        2) test_connection()   -> sanity check de conectividad a Supabase
        3) h2o_start()         -> inicia cluster local de H2O
        4) load_frame_in_chunks() -> ingesta incremental desde Postgres en batches (keyset)
        5) add_log_target()    -> crea target log para estabilizar distribución
        6) cast_categoricals() -> convierte categóricas a factor
        7) split()             -> train/valid/test
        8) train_automl()      -> entrena AutoML
        9) evaluate()          -> métricas en test
        10) show_leaderboard() -> top modelos
        11) plot_pred_vs_true_log() y best_gbm_varimp() como diagnóstico

        Returns:
            Diccionario con métricas de test (rmse/mae/r2).
        """
        self.load_env()
        self.test_connection()

        self.h2o_start()
        self.load_frame_in_chunks()

        self.add_log_target()
        self.cast_categoricals()

        self.split()
        self.train_automl()

        metrics = self.evaluate()
        self.show_leaderboard(top_n=10)

        # extras (comenta si querés)
        # self.leader.explain(self.test)
        self.plot_pred_vs_true_log()
        self.best_gbm_varimp()

        return metrics


def main() -> None:
    """
    Entry point para ejecución como script.

    Permite correr:
        python ml_pipeline_supabase_h2o_class.py

    Nota:
    - Mantener chunk_size en config para controlar tiempo de extracción.
    - Si querés pruebas rápidas, podés reducir chunk_size o cortar ingest en load_frame_in_chunks().
    """
    cfg = PipelineConfig(chunk_size=100_000)
    pipe = MLPipeline(cfg)
    pipe.run()


if __name__ == "__main__":
    main()
