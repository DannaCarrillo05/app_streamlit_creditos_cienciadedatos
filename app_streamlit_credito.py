import io
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Prediccion de Credit Score", page_icon="💳", layout="wide")

st.title("Prediccion de Credit Score")
st.caption("Interfaz para inferencia del modelo entrenado en Taller 2")


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent


DEFAULTS = {
    "model": PROJECT_DIR / "modelo_credito.keras",
    "scaler": PROJECT_DIR / "minmax_scaler.joblib",
    "encoders": PROJECT_DIR / "label_encoders.joblib",
    "pca": PROJECT_DIR / "pca_8_componentes.joblib",
}

DEFAULT_SEARCH_DIRS = [
    PROJECT_DIR,
    APP_DIR,
    PROJECT_DIR / "notebook",
    Path.cwd(),
]


@st.cache_resource
def load_model(model_bytes: bytes):
    def strip_key_recursive(obj, key_to_remove: str):
        if isinstance(obj, dict):
            return {
                k: strip_key_recursive(v, key_to_remove)
                for k, v in obj.items()
                if k != key_to_remove
            }
        if isinstance(obj, list):
            return [strip_key_recursive(x, key_to_remove) for x in obj]
        return obj

    def sanitize_keras_archive(src_path: Path, dst_path: Path):
        with zipfile.ZipFile(src_path, "r") as zin:
            with zipfile.ZipFile(dst_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    raw = zin.read(info.filename)
                    if info.filename == "config.json":
                        data = json.loads(raw.decode("utf-8"))
                        data = strip_key_recursive(data, "quantization_config")
                        raw = json.dumps(data, ensure_ascii=True).encode("utf-8")
                    zout.writestr(info, raw)

    buffer = io.BytesIO(model_bytes)
    temp_path = Path("_tmp_modelo_credito.keras")
    temp_path.write_bytes(buffer.getvalue())
    sanitized_path = Path("_tmp_modelo_credito_sanitized.keras")

    load_errors = []

    # Intento 1: tensorflow.keras con capa Dense compatible.
    try:
        from tensorflow.keras.layers import Dense as TFDense
        from tensorflow.keras.models import load_model as tf_load_model

        class DenseCompatTF(TFDense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)

        return tf_load_model(
            temp_path,
            compile=False,
            custom_objects={"Dense": DenseCompatTF},
        )
    except Exception as exc:
        load_errors.append(f"tensorflow.keras: {exc}")

    # Intento 2: keras standalone con safe_mode=False y capa compatible.
    try:
        import keras
        from keras.layers import Dense as KDense

        class DenseCompatKeras(KDense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)

        return keras.models.load_model(
            temp_path,
            compile=False,
            safe_mode=False,
            custom_objects={"Dense": DenseCompatKeras},
        )
    except Exception as exc:
        load_errors.append(f"keras: {exc}")

    # Intento 3: sanea config.json dentro del .keras y vuelve a intentar.
    try:
        sanitize_keras_archive(temp_path, sanitized_path)
    except Exception as exc:
        load_errors.append(f"sanitize: {exc}")
    else:
        try:
            from tensorflow.keras.models import load_model as tf_load_model

            return tf_load_model(sanitized_path, compile=False)
        except Exception as exc:
            load_errors.append(f"tensorflow.keras (sanitized): {exc}")

        try:
            import keras

            return keras.models.load_model(sanitized_path, compile=False, safe_mode=False)
        except Exception as exc:
            load_errors.append(f"keras (sanitized): {exc}")

    msg = " | ".join(load_errors)
    raise ValueError(
        "No fue posible deserializar el modelo .keras con las librerias disponibles. "
        "Esto suele ocurrir por incompatibilidad de versiones entre Colab y local. "
        f"Detalle: {msg}"
    )


@st.cache_resource
def load_pickle(bytes_blob: bytes):
    return joblib.load(io.BytesIO(bytes_blob))


def file_to_bytes(uploaded_file, fallback_path: Path):
    if uploaded_file is not None:
        return uploaded_file.getvalue(), f"subido ({uploaded_file.name})"
    if fallback_path.exists():
        return fallback_path.read_bytes(), f"local ({fallback_path})"
    return None, "no encontrado"


def search_artifact(filename: str):
    for base in DEFAULT_SEARCH_DIRS:
        if not base.exists() or not base.is_dir():
            continue
        try:
            candidates = sorted(base.rglob(filename))
        except Exception:
            candidates = []
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def list_candidate_files():
    rows = []
    patterns = ["*.keras", "*.joblib"]
    for base in DEFAULT_SEARCH_DIRS:
        if not base.exists() or not base.is_dir():
            continue
        for pattern in patterns:
            try:
                matches = sorted(base.rglob(pattern))
            except Exception:
                matches = []
            for m in matches[:50]:
                if m.is_file():
                    rows.append({"archivo": m.name, "ruta": str(m)})
    if not rows:
        return pd.DataFrame(columns=["archivo", "ruta"])
    # Evita duplicados cuando una ruta aparece desde distintos base dirs.
    dedup = pd.DataFrame(rows).drop_duplicates(subset=["ruta"]).reset_index(drop=True)
    return dedup


def resolve_artifact_bytes(uploaded_file, fallback_path: Path, explicit_path: str):
    if uploaded_file is not None:
        return uploaded_file.getvalue(), f"subido ({uploaded_file.name})"

    if explicit_path.strip():
        p = Path(explicit_path.strip())
        if p.exists() and p.is_file():
            return p.read_bytes(), f"ruta manual ({p})"
        return None, f"ruta manual invalida ({p})"

    if fallback_path.exists() and fallback_path.is_file():
        return fallback_path.read_bytes(), f"local ({fallback_path})"

    found = search_artifact(fallback_path.name)
    if found is not None:
        return found.read_bytes(), f"autodetectado ({found})"

    return None, "no encontrado"


def validate_pipeline_dimensions(model, scaler, pca=None):
    n_scaler_features = len(getattr(scaler, "feature_names_in_", []))
    if n_scaler_features == 0:
        return False, "No pude inferir columnas del scaler (feature_names_in_ ausente o vacio).", None

    expected_model_input = model.input_shape[-1]
    transformed_dim = pca.n_components_ if pca is not None else n_scaler_features

    if expected_model_input != transformed_dim:
        msg = (
            f"Dimension incompatible: modelo espera {expected_model_input} features, "
            f"pero el pipeline produce {transformed_dim}. "
            "Revisa si debes activar/desactivar PCA o cargar artefactos del mismo entrenamiento."
        )
        return False, msg, transformed_dim
    return True, "OK", transformed_dim


def infer_feature_schema(scaler, label_encoders):
    if not hasattr(scaler, "feature_names_in_"):
        raise ValueError(
            "El scaler no tiene feature_names_in_. Vuelve a entrenar con un DataFrame para conservar nombres de columnas."
        )

    feature_cols = list(scaler.feature_names_in_)
    encoder_cols = set(label_encoders.keys()) if isinstance(label_encoders, dict) else set()
    categorical_cols = [c for c in feature_cols if c in encoder_cols]
    numeric_cols = [c for c in feature_cols if c not in encoder_cols]
    return feature_cols, numeric_cols, categorical_cols


def encode_categoricals(df_in: pd.DataFrame, categorical_cols, label_encoders):
    df = df_in.copy()
    for col in categorical_cols:
        encoder = label_encoders[col]

        if pd.api.types.is_numeric_dtype(df[col]):
            # Permite enviar columnas ya codificadas como enteros.
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                raise ValueError(f"La columna {col} tiene valores no numericos invalidos.")
            continue

        values = df[col].astype(str).str.strip()
        known = set(map(str, encoder.classes_))
        unknown = sorted(set(values) - known)
        if unknown:
            sample = ", ".join(unknown[:5])
            raise ValueError(
                f"La columna {col} tiene categorias no vistas en entrenamiento: {sample}."
            )

        df[col] = encoder.transform(values)

    return df


def preprocess_input(df_raw: pd.DataFrame, scaler, label_encoders, pca=None):
    feature_cols, numeric_cols, categorical_cols = infer_feature_schema(scaler, label_encoders)

    missing = [c for c in feature_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df_raw[feature_cols].copy()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        na_cols = df[numeric_cols].columns[df[numeric_cols].isna().any()].tolist()
        raise ValueError(f"Hay valores no numericos o nulos en columnas numericas: {na_cols}")

    df = encode_categoricals(df, categorical_cols, label_encoders)

    x_scaled = scaler.transform(df)

    if pca is not None:
        x_final = pca.transform(x_scaled)
    else:
        x_final = x_scaled

    return x_final, feature_cols, numeric_cols, categorical_cols


def predict_df(df_raw: pd.DataFrame, model, scaler, label_encoders, pca=None):
    x_final, feature_cols, numeric_cols, categorical_cols = preprocess_input(
        df_raw=df_raw,
        scaler=scaler,
        label_encoders=label_encoders,
        pca=pca,
    )

    probs = model.predict(x_final, verbose=0)
    pred_idx = probs.argmax(axis=1)

    target_encoder = label_encoders.get("Credit_Score", None) if isinstance(label_encoders, dict) else None
    if target_encoder is not None:
        pred_label = target_encoder.inverse_transform(pred_idx)
    else:
        pred_label = pred_idx.astype(str)

    result = df_raw.copy()
    result["pred_class_index"] = pred_idx
    result["pred_credit_score"] = pred_label

    for i in range(probs.shape[1]):
        result[f"prob_class_{i}"] = np.round(probs[:, i], 6)

    return result, feature_cols, numeric_cols, categorical_cols


with st.sidebar:
    st.header("Archivos del modelo")
    st.write("Puedes subir archivos o dejar que la app busque rutas locales.")

    model_file = st.file_uploader("Modelo Keras (.keras)", type=["keras"])
    scaler_file = st.file_uploader("Scaler (.joblib)", type=["joblib"])
    encoders_file = st.file_uploader("Label encoders (.joblib)", type=["joblib"])
    pca_file = st.file_uploader("PCA opcional (.joblib)", type=["joblib"])

    st.markdown("Ruta local manual (opcional)")
    model_path_txt = st.text_input("Ruta modelo", value="")
    scaler_path_txt = st.text_input("Ruta scaler", value="")
    encoders_path_txt = st.text_input("Ruta encoders", value="")
    pca_path_txt = st.text_input("Ruta PCA", value="")

    use_pca = st.checkbox("Aplicar PCA si esta disponible", value=True)

model_bytes, model_source = resolve_artifact_bytes(model_file, DEFAULTS["model"], model_path_txt)
scaler_bytes, scaler_source = resolve_artifact_bytes(scaler_file, DEFAULTS["scaler"], scaler_path_txt)
encoders_bytes, encoders_source = resolve_artifact_bytes(encoders_file, DEFAULTS["encoders"], encoders_path_txt)
pca_bytes, pca_source = resolve_artifact_bytes(pca_file, DEFAULTS["pca"], pca_path_txt)

status_df = pd.DataFrame(
    {
        "artefacto": ["modelo", "scaler", "encoders", "pca"],
        "origen": [model_source, scaler_source, encoders_source, pca_source],
    }
)
st.subheader("Estado de artefactos")
st.dataframe(status_df, width="stretch")

with st.expander("Diagnostico de archivos locales", expanded=False):
    st.write("Si no aparecen artefactos arriba, revisa esta lista para copiar/pegar rutas manuales.")
    candidate_df = list_candidate_files()
    if candidate_df.empty:
        st.warning("No se encontraron .keras/.joblib en las rutas de busqueda locales.")
    else:
        st.dataframe(candidate_df, width="stretch")

if model_bytes is None or scaler_bytes is None or encoders_bytes is None:
    st.warning(
        "Carga modelo, scaler y encoders para continuar. Si no existen localmente, subelos desde tu Google Drive o carpeta de entrenamiento."
    )
    st.stop()

try:
    with st.spinner("Cargando artefactos..."):
        model = load_model(model_bytes)
        scaler = load_pickle(scaler_bytes)
        label_encoders = load_pickle(encoders_bytes)
        pca = load_pickle(pca_bytes) if use_pca and pca_bytes is not None else None
except Exception as exc:
    st.error(f"No fue posible cargar artefactos: {exc}")
    st.stop()

st.success("Artefactos cargados correctamente")

n_scaler_features = len(getattr(scaler, "feature_names_in_", []))
model_input_dim = model.input_shape[-1]
pca_dim = pca.n_components_ if pca is not None else None

# Autoajuste de PCA para evitar bloquear la app por incompatibilidad frecuente.
if pca is not None and model_input_dim == n_scaler_features and model_input_dim != pca_dim:
    pca = None
    st.warning(
        "Se detecto incompatibilidad de dimensiones con PCA activo. "
        "La app desactivo PCA automaticamente porque el modelo fue entrenado con features sin PCA."
    )
elif pca is None and pca_bytes is not None and model_input_dim != n_scaler_features:
    try:
        pca_candidate = load_pickle(pca_bytes)
        if getattr(pca_candidate, "n_components_", None) == model_input_dim:
            pca = pca_candidate
            st.info("La app activo PCA automaticamente para coincidir con las dimensiones del modelo.")
    except Exception:
        pass

is_compatible, compatibility_msg, _ = validate_pipeline_dimensions(model, scaler, pca)
if not is_compatible:
    st.error(compatibility_msg)
    st.stop()

st.info("Compatibilidad de dimensiones validada entre modelo y preprocesamiento.")

try:
    expected_features, numeric_features, categorical_features = infer_feature_schema(scaler, label_encoders)
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Esquema esperado de entrada", expanded=False):
    st.write("Features esperadas por el scaler y el modelo:")
    st.write(expected_features)
    st.write("Columnas numericas:")
    st.write(numeric_features)
    st.write("Columnas categoricas:")
    st.write(categorical_features)

mode = st.radio("Modo de prediccion", ["Registro individual", "Lote CSV"], horizontal=True)

if mode == "Registro individual":
    st.subheader("Prediccion individual")

    col1, col2 = st.columns(2)
    row_dict = {}

    for idx, col in enumerate(expected_features):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if col in categorical_features:
                opts = list(map(str, label_encoders[col].classes_))
                row_dict[col] = st.selectbox(f"{col}", opts, key=f"single_{col}")
            else:
                row_dict[col] = st.number_input(f"{col}", value=0.0, format="%.6f", key=f"single_{col}")

    if st.button("Predecir", type="primary"):
        input_df = pd.DataFrame([row_dict])
        try:
            pred_df, _, _, _ = predict_df(input_df, model, scaler, label_encoders, pca)
        except Exception as exc:
            st.error(f"Error en prediccion: {exc}")
            st.stop()

        st.success("Prediccion realizada")
        st.write(pred_df[["pred_credit_score", "pred_class_index"]])

        prob_cols = [c for c in pred_df.columns if c.startswith("prob_class_")]
        probs = pred_df.loc[0, prob_cols]
        st.bar_chart(probs)

else:
    st.subheader("Prediccion por lote (CSV)")
    st.write(
        "Sube un CSV con las columnas esperadas. Puede tener columnas extra; solo se usaran las necesarias."
    )

    uploaded_csv = st.file_uploader("CSV de entrada", type=["csv"], key="csv_batch")

    if uploaded_csv is not None:
        try:
            df_batch = pd.read_csv(uploaded_csv)
            st.write("Vista previa del CSV de entrada:")
            st.dataframe(df_batch.head(20), width="stretch")
        except Exception as exc:
            st.error(f"No pude leer el CSV: {exc}")
            st.stop()

        if st.button("Ejecutar prediccion por lote", type="primary"):
            try:
                result_df, _, _, _ = predict_df(df_batch, model, scaler, label_encoders, pca)
            except Exception as exc:
                st.error(f"Error en prediccion por lote: {exc}")
                st.stop()

            st.success(f"Predicciones completadas: {len(result_df)} registros")
            st.dataframe(result_df.head(50), width="stretch")

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar resultados CSV",
                data=csv_out,
                file_name="predicciones_credit_score.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption(
    "Ejecuta con: streamlit run cienciadedatos/app_streamlit_credito.py"
)
