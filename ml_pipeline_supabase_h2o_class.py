# ml_pipeline_supabase_h2o_class.py
from __future__ import annotations

import os
import json
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import psycopg2

import h2o
from h2o.automl import H2OAutoML


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuración central del pipeline (Data Engineering + Data Science).

    Esta clase agrupa parámetros para que el pipeline sea reproducible, configurable y
    seguro (sin credenciales hardcodeadas). Se usa como “single source of truth” para:

    - Conexión a Supabase/Postgres mediante DATABASE_URL en un .env.
    - Tabla fuente (ml_table) y columna PK (sale_id) para paginar por keyset.
    - Ingesta incremental por chunks (tamaño y punto de inicio).
    - Parámetros de H2O AutoML (memoria, seeds, tiempo, número de modelos).
    - Definición de target, features y columnas categóricas.
    - Filtros mínimos de calidad (WHERE) para construir el dataset de entrenamiento.
    - Carpeta local para guardar artefactos (métricas, leaderboard, figuras, modelo).
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
    split_train_valid: Tuple[float, float] = (0.7, 0.15)  # test = resto
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

    # filtros mínimos de calidad
    where_filters: str = """
        saleamount IS NOT NULL
        AND assessedvalue IS NOT NULL
        AND saleamount > 2000
        AND assessedvalue > 0
        AND listyear IS NOT NULL
    """

    # carpeta para resultados/artefactos
    output_dir: str = "artifacts"


# -----------------------------
# Pipeline
# -----------------------------
class MLPipeline:
    """
    Pipeline end-to-end para entrenamiento con H2O AutoML consultando Supabase/Postgres.

    Resumen de etapas:
    1) Carga DATABASE_URL desde .env (credenciales fuera del código).
    2) Valida conexión a la base (sanity check).
    3) Extrae datos en chunks desde la tabla fuente con keyset pagination por sale_id.
    4) Importa cada chunk a H2O y lo concatena en un H2OFrame acumulado.
    5) Aplica feature engineering mínimo:
       - target log: log(saleamount + 1)
       - casteo de categóricas a factor
    6) Split train/valid/test.
    7) Entrena AutoML y evalúa en test.
    8) Genera y guarda artefactos (leaderboard, métricas, figuras, varimp, modelo líder).
    """

    def __init__(self, cfg: PipelineConfig):
        """
        Inicializa el pipeline y reserva atributos que se llenan durante la ejecución.

        Args:
            cfg: configuración del pipeline (conexión, columnas, chunks, AutoML y outputs).
        """
        self.cfg = cfg
        self.db_url: Optional[str] = None

        # Dataset acumulado y splits (H2OFrames)
        self.hf = None
        self.train = None
        self.valid = None
        self.test = None

        # AutoML y modelo ganador
        self.aml: Optional[H2OAutoML] = None
        self.leader = None

    # -----------------------------
    # Helpers de outputs (artefactos)
    # -----------------------------
    def _ensure_output_dir(self) -> str:
        """
        Crea (si no existe) la carpeta donde se guardan los artefactos del pipeline.

        Returns:
            Ruta a la carpeta de salida (cfg.output_dir).
        """
        out = self.cfg.output_dir
        os.makedirs(out, exist_ok=True)
        return out

    def _safe_filename(self, name: str) -> str:
        """
        Normaliza nombres de archivo para que sean seguros en Windows y sistemas de archivos.

        H2O genera model_id con caracteres como ':' que pueden causar problemas en paths.

        Args:
            name: nombre original (ej. model_id o filename).

        Returns:
            String con caracteres problemáticos reemplazados.
        """
        return (
            str(name)
            .replace(":", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

    def _artifact_path(self, filename: str) -> str:
        """
        Construye una ruta absoluta/relativa a un archivo dentro de cfg.output_dir.

        Args:
            filename: nombre del archivo a guardar.

        Returns:
            Path completo (output_dir/filename normalizado).
        """
        out = self._ensure_output_dir()
        return os.path.join(out, self._safe_filename(filename))

    # -----------------------------
    # Supabase/Postgres: conexión
    # -----------------------------
    def load_env(self) -> str:
        """
        Carga el archivo .env y obtiene DATABASE_URL.

        Motivo:
        - Mantener credenciales fuera del código y del repositorio.

        Returns:
            DATABASE_URL como string.

        Raises:
            RuntimeError: si DATABASE_URL no existe en el entorno.
        """
        load_dotenv(self.cfg.env_path)
        url = os.getenv(self.cfg.db_env_key)
        if not url:
            raise RuntimeError(f"No encuentro {self.cfg.db_env_key} en {self.cfg.env_path}")
        self.db_url = url
        return url

    def _connect(self):
        """
        Abre una conexión psycopg2 usando self.db_url.

        Returns:
            Conexión psycopg2.

        Raises:
            RuntimeError: si load_env() no se llamó antes.
        """
        if not self.db_url:
            raise RuntimeError("db_url no inicializado. Llama primero a load_env().")
        return psycopg2.connect(self.db_url)

    def test_connection(self) -> None:
        """
        Verifica la conectividad a Supabase/Postgres ejecutando una query liviana.

        Imprime:
        - Host y puerto del servidor
        - Usuario actual
        - Puerto del servidor
        - Timestamp actual

        Raises:
            RuntimeError: si db_url no está inicializado.
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

    # -----------------------------
    # H2O
    # -----------------------------
    def h2o_start(self) -> None:
        """
        Inicia el cluster local de H2O.

        Configuración:
        - nthreads=-1 usa todos los hilos disponibles.
        - max_mem_size limita el uso de memoria por el cluster (cfg.h2o_max_mem).
        """
        h2o.init(nthreads=-1, max_mem_size=self.cfg.h2o_max_mem)

    # -----------------------------
    # Extracción por chunks (keyset pagination)
    # -----------------------------
    def _build_chunk_query(self, last_sale_id: int) -> str:
        """
        Construye un SELECT paginado por keyset usando la PK (sale_id).

        Estrategia (keyset pagination):
        - WHERE sale_id > last_sale_id
        - ORDER BY sale_id
        - LIMIT chunk_size

        Ventaja:
        - Escala mucho mejor que OFFSET para millones de filas.

        Args:
            last_sale_id: último sale_id procesado.

        Returns:
            Query SQL lista para envolver en COPY(...).
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
        Exporta un chunk a CSV usando COPY TO STDOUT (rápido y eficiente en Postgres).

        Flujo:
        - Genera el SELECT paginado.
        - Ejecuta COPY (SELECT ...) TO STDOUT WITH CSV HEADER.
        - Cuenta filas del archivo resultante (sin cargar todo en memoria).

        Args:
            conn: conexión psycopg2 abierta.
            last_sale_id: cursor keyset.
            csv_path: ruta del CSV temporal.

        Returns:
            Cantidad de filas exportadas (sin header).
        """
        query = self._build_chunk_query(last_sale_id)
        copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER"

        with conn.cursor() as cur, open(csv_path, "w", encoding="utf-8", newline="") as f:
            cur.copy_expert(copy_sql, f)

        with open(csv_path, "r", encoding="utf-8") as f:
            nrows = sum(1 for _ in f) - 1
        return max(nrows, 0)

    def get_last_sale_id_from_csv(self, csv_path: str) -> int:
        """
        Obtiene el mayor sale_id exportado en el CSV del chunk.

        Como el SELECT se exporta ordenado (ORDER BY sale_id),
        la última fila del CSV contiene el sale_id máximo del chunk.

        Args:
            csv_path: ruta al CSV temporal.

        Returns:
            sale_id máximo del chunk (int). Si el archivo está vacío, retorna -1.

        Raises:
            RuntimeError: si pk_col no está en el header.
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
        Importa un CSV temporal a H2O y lo concatena al H2OFrame acumulado.

        Args:
            csv_path: ruta al CSV temporal exportado desde Postgres.
        """
        hf_chunk = h2o.import_file(csv_path)
        self.hf = hf_chunk if self.hf is None else self.hf.rbind(hf_chunk)

    def load_frame_in_chunks(self) -> None:
        """
        Construye self.hf trayendo datos por chunks desde Supabase/Postgres.

        Implementación:
        - Crea un directorio temporal.
        - Exporta chunks a CSV temporales (COPY).
        - Importa cada CSV a H2O y concatena.
        - El directorio temporal se borra al finalizar.

        Efecto:
        - Evita generar un CSV único gigante en disco local.
        - Mantiene extracción estable para grandes volúmenes usando keyset pagination.
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

    # -----------------------------
    # Feature engineering
    # -----------------------------
    def add_log_target(self) -> None:
        """
        Crea el target logarítmico: saleamount_log = log(saleamount + 1).

        Motivo:
        - Reduce asimetría y el impacto de outliers en precios inmobiliarios.
        - Suele estabilizar el entrenamiento y mejorar generalización.

        Raises:
            RuntimeError: si self.hf no existe.
        """
        if self.hf is None:
            raise RuntimeError("No hay H2OFrame. Llama primero a load_frame_in_chunks().")
        y = self.cfg.target
        self.hf[self.cfg.target_log] = (self.hf[y] + 1).log()

    def cast_categoricals(self) -> None:
        """
        Convierte columnas categóricas a factor (tipo categoría) en H2O.

        Motivo:
        - Asegura que H2O trate estas columnas como categorías y no como numéricas.

        Raises:
            RuntimeError: si self.hf no existe.
        """
        if self.hf is None:
            raise RuntimeError("No hay H2OFrame.")
        for c in self.cfg.categorical:
            if c in self.hf.columns:
                self.hf[c] = self.hf[c].asfactor()

    # -----------------------------
    # Split / Train / Eval
    # -----------------------------
    def split(self) -> None:
        """
        Divide el dataset en train/valid/test según cfg.split_train_valid y cfg.seed.

        - train_ratio = split_train_valid[0]
        - valid_ratio = split_train_valid[1]
        - test_ratio  = residual

        Raises:
            RuntimeError: si self.hf no existe.
        """
        if self.hf is None:
            raise RuntimeError("No hay H2OFrame.")
        tr, va = self.cfg.split_train_valid
        self.train, self.valid, self.test = self.hf.split_frame([tr, va], seed=self.cfg.seed)

    def train_automl(self) -> None:
        """
        Entrena H2OAutoML sobre train y usa valid para selección/early-stopping.

        Resultado:
        - self.aml: objeto AutoML con leaderboard y modelos entrenados.
        - self.leader: mejor modelo (según cfg.sort_metric) en valid.

        Raises:
            RuntimeError: si train/valid no existen.
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
            x=list(self.cfg.features),  # NO incluye sale_id
            y=self.cfg.target_log,      # target log
            training_frame=self.train,
            validation_frame=self.valid,
        )
        self.leader = self.aml.leader

    def evaluate(self) -> Dict[str, float]:
        """
        Evalúa el modelo líder en test y devuelve métricas en escala log.

        Métricas:
        - RMSE
        - MAE
        - R2

        Returns:
            dict con keys: rmse, mae, r2

        Raises:
            RuntimeError: si leader o test no existen.
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
        Imprime por consola el leaderboard (top_n modelos) de AutoML.

        Args:
            top_n: número de filas a mostrar.

        Raises:
            RuntimeError: si AutoML no fue entrenado.
        """
        if self.aml is None:
            raise RuntimeError("AutoML no entrenado.")
        print(self.aml.leaderboard.head(rows=top_n))

    # -----------------------------
    # Artefactos: plots + varimp
    # -----------------------------
    def plot_pred_vs_true_log(self, alpha: float = 0.15, save: bool = True) -> None:
        """
        Genera un scatter plot de predicción vs real (escala log) y opcionalmente lo guarda.

        Objetivo:
        - Diagnóstico visual de calibración del modelo (ideal: puntos cerca de y=x).
        - Detectar sesgos de sobre/infra-predicción.

        Args:
            alpha: transparencia de los puntos (útil con muchos registros).
            save: si True, guarda PNG en cfg.output_dir.

        Raises:
            RuntimeError: si leader o test no existen.
        """
        if self.leader is None or self.test is None:
            raise RuntimeError("Leader/Test no listos.")

        pred = self.leader.predict(self.test)
        y_true_df = self.test[self.cfg.target_log].as_data_frame(use_pandas=True)
        y_pred_df = pred["predict"].as_data_frame(use_pandas=True)

        y_true = np.asarray(y_true_df[self.cfg.target_log]).astype(float)
        y_pred = np.asarray(y_pred_df["predict"]).astype(float)

        plt.figure()
        plt.scatter(y_true, y_pred, alpha=alpha)
        mn = np.nanmin([y_true.min(), y_pred.min()])
        mx = np.nanmax([y_true.max(), y_pred.max()])
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("Real (log)")
        plt.ylabel("Predicción (log)")
        plt.title("Predicción vs Real (escala log)")

        if save:
            fig_path = self._artifact_path("pred_vs_real_log.png")
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            print("Guardado:", fig_path)

        plt.show()
        plt.close()

    def best_gbm_varimp(self, save: bool = True) -> None:
        """
        Busca el mejor GBM en el leaderboard y extrae su importancia de variables.

        Motivo:
        - El leader puede ser un StackedEnsemble (menos interpretable).
        - Un GBM top suele ser similar en performance y más interpretable.

        Guardado:
        - CSV con tabla de importancias (si save=True).
        - PNG con el varimp_plot (si save=True).

        Args:
            save: si True, guarda CSV y PNG en cfg.output_dir.

        Raises:
            RuntimeError: si AutoML no fue entrenado.
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

        if save:
            csv_path = self._artifact_path(f"varimp_{best_gbm_id}.csv")
            vi.to_csv(csv_path, index=False)
            print("Guardado:", csv_path)

        best_gbm.varimp_plot()

        if save:
            png_path = self._artifact_path(f"varimp_{best_gbm_id}.png")
            plt.savefig(png_path, dpi=200, bbox_inches="tight")
            print("Guardado:", png_path)

        plt.show()
        plt.close()

    # -----------------------------
    # Guardado de resultados
    # -----------------------------
    def save_results(self, metrics: Dict[str, float]) -> None:
        """
        Guarda resultados del pipeline en cfg.output_dir para ejecución desde terminal.

        Artefactos:
        - metrics.json: métricas finales en test.
        - leaderboard.csv: ranking completo de modelos de AutoML.
        - leader model: serialización del modelo ganador con h2o.save_model().

        Args:
            metrics: diccionario con métricas (rmse/mae/r2).
        """
        self._ensure_output_dir()

        # metrics.json
        metrics_path = self._artifact_path("metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Guardado:", metrics_path)

        # leaderboard.csv
        if self.aml is not None:
            lb_df = self.aml.leaderboard.as_data_frame()
            lb_path = self._artifact_path("leaderboard.csv")
            lb_df.to_csv(lb_path, index=False)
            print("Guardado:", lb_path)

        # leader model (H2O)
        if self.leader is not None:
            model_path = h2o.save_model(
                model=self.leader,
                path=self.cfg.output_dir,
                force=True,
            )
            print("Guardado leader model en:", model_path)

    # -----------------------------
    # Orchestrator
    # -----------------------------
    def run(self) -> Dict[str, float]:
        """
        Ejecuta el pipeline completo (end-to-end) en orden determinista.

        Flujo:
        1) load_env()            -> lee DATABASE_URL desde .env
        2) test_connection()     -> valida conexión a Supabase/Postgres
        3) h2o_start()           -> inicia el cluster H2O
        4) load_frame_in_chunks()-> extrae dataset por chunks y lo carga a H2OFrame
        5) add_log_target()      -> crea variable objetivo logarítmica
        6) cast_categoricals()   -> castea columnas categóricas a factor
        7) split()               -> genera train/valid/test
        8) train_automl()        -> entrena AutoML y define leader
        9) evaluate()            -> evalúa en test y devuelve métricas
        10) show_leaderboard()   -> imprime top modelos
        11) plot_pred_vs_true_log() + best_gbm_varimp() -> diagnósticos y guardado de figuras
        12) save_results()       -> persistencia de métricas/leaderboard/modelo a disco

        Returns:
            Diccionario con métricas finales (rmse/mae/r2).
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

        self.plot_pred_vs_true_log(alpha=0.15, save=True)
        self.best_gbm_varimp(save=True)

        self.save_results(metrics)
        return metrics


def main() -> None:
    """
    Entry point para ejecución como script desde terminal.

    Uso:
        python ml_pipeline_supabase_h2o_class.py

    Ajustes comunes:
    - Reducir chunk_size para pruebas rápidas.
    - Aumentar max_runtime_secs / max_models para búsquedas más amplias.
    - Cambiar output_dir para separar ejecuciones.
    """
    cfg = PipelineConfig(chunk_size=100_000)
    pipe = MLPipeline(cfg)
    pipe.run()


if __name__ == "__main__":
    main()
