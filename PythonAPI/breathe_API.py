import earthaccess as ea
from pathlib import Path
from datetime import datetime, timedelta, timezone
import xarray as xr
import numpy as np
from flask import Flask, request, jsonify
import math
import os
import json
from threading import Lock

# =========================
# CONFIG
# =========================
#Hardware simple DB
HARDWARE_TABLE = {
    "001": {"lat": 19.4326, "lon": -99.1332},   # CDMX
    "002": {"lat": 20.6736, "lon": -103.3440},  # Guadalajara
    "003": {"lat": 25.6866, "lon": -100.3161},  # Monterrey
    "004": {"lat": 19.5296, "lon": -96.9236},   # Xalapa
}
STORE_PATH = "/tmp/hardware_store.json"
STORE_LOCK = Lock()
SENSOR_STORE = {}
def load_from_disk():
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    HARDWARE_TABLE.update(data)
        except Exception:
            pass  # si falla, seguimos con el diccionario en memoria

def save_to_disk():
    try:
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(HARDWARE_TABLE, f, ensure_ascii=False)
    except Exception:
        pass
def _load_store():
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    SENSOR_STORE.update(data)
        except Exception:
            pass

def _save_store():
    try:
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(SENSOR_STORE, f, ensure_ascii=False)
    except Exception:
        pass

_load_store()
# inicial load (hardware table)
load_from_disk()



# Bounding box: (min_lon, min_lat, max_lon, max_lat)
bbox = (-96.999273, 19.455968, -96.822608, 19.622459) #Xalapa
#bbox = (-5.22, 41.595021, -5.20, 41.696425) # Valladolid, España
#bbox = (-74.122824, 40.645596, -73.848943, 40.853080)  # Nueva York, EE.UU.

AVOG_N = 6.022e23
N_AIR_1ATM_298K = 2.5e25  # moléculas/m^3 (aprox)

# Last 6 hours
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=3)
temporal_iso = (start_time.isoformat(), end_time.isoformat())

outdir = Path("data")
outdir.mkdir(exist_ok=True)

# =========================
# DATASETS per BLOCK
# =========================
blocks = {
    # 1) AQI — todo NRT global salvo AOD (L2 ~1 día)
    "AQI": [
        "S5P_L2__NO2____HiR_NRT",   # NO2 NRT (TROPOMI) :contentReference[oaicite:2]{index=2}
        "S5P_L2__O3_TCL_NRT",       # O3 troposférico NRT :contentReference[oaicite:3]{index=3}
        "S5P_L2__SO2____HiR_NRT",   # SO2 NRT (alta resolución) :contentReference[oaicite:4]{index=4}
        "S5P_L2__CO_____HiR_NRT",   # CO NRT :contentReference[oaicite:5]{index=5}
        "S5P_L2__AER_AI__NRTI",     # AOD Terra L2 (útil ~NRT/día)
        # Alternativa/backup: "MYD04_L2"  # AOD Aqua L2
    ],

    # 2) FIRE_DUST — NRT global
    "FIRE_DUST": [
        "S5P_L2__AER_AI__NRTI",     # Aerosol Index NRT (absorbing) :contentReference[oaicite:6]{index=6}
        "S5P_L2__CO_____HiR_NRT",   # CO NRT (humo/incendios) :contentReference[oaicite:7]{index=7}
        "S5P_L2__HCHO___NRTI",      # Formaldehído NRT (piroquímica)
    ],

    # 3) MET_EFFECT — NO2 + Meteorología (preferir GEOS-FP NRT; si no, dejar MERRA-2)
    "MET_EFFECT": [
        "S5P_L2__NO2____HiR_NRT",   # NO2 NRT :contentReference[oaicite:8]{index=8}
        # Preferido (NRT horario). Si no está accesible en tu cuenta, usa MERRA-2:
        # "GEOS_FP_tavg1_2d_slv_Nx",
        "MERRA2_400.tavg1_2d_slv_Nx"  # Temp 2m, vientos, PBLH (reanálisis)
    ],

    # 4) HEALTH_RISK — NO2 + O3 + PM + Temp (PM en MERRA-2; O3/NO2 NRT)
    "HEALTH_RISK": [
        "S5P_L2__NO2____HiR_NRT",   # NO2 NRT :contentReference[oaicite:9]{index=9}
        "S5P_L2__O3_TCL_NRT",       # O3 troposférico NRT :contentReference[oaicite:10]{index=10}
        # PM2.5 estimado (reanálisis horario):
        "M2T1NXAER",                # MERRA-2 Aerosol Diagnostics (PM2.5, etc.) :contentReference[oaicite:11]{index=11}
        # Meteo (NRT ideal = GEOS-FP; si no, MERRA-2):
        # "GEOS_FP_tavg1_2d_slv_Nx",
        "MERRA2_400.tavg1_2d_slv_Nx"
    ],
}
#######################################################################################################################
#################################FUNCTIONS FOR ANALYZING DATA FROM SATELITES###########################################
#######################################################################################################################
# =========================
# AUTENTICATION
# =========================
ea.login(strategy="environment")

# =========================
# HELPERS
# =========================
def point_to_bbox(lat_deg, lon_deg, radius_m):
    """
    Convierte un punto (lat, lon) y un radio (m) a un bounding box
    en formato [min_lon, min_lat, max_lon, max_lat].
    """
    # Aproximaciones (válidas para radios <50 km)
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat_deg)))

    dlat = radius_m * lat_per_m
    dlon = radius_m * lon_per_m

    min_lat = lat_deg - dlat
    max_lat = lat_deg + dlat
    min_lon = lon_deg - dlon
    max_lon = lon_deg + dlon

    return [min_lon, min_lat, max_lon, max_lat]

def fetch_one(short_name: str):
    """Serch and download 1. Return route or None."""
    try:
        res = ea.search_data(
            short_name=short_name,
            temporal=temporal_iso,
            bounding_box=bbox,
            count=1
        )
        if not res:
            print(f"[{short_name}] sin resultados en {temporal_iso} y bbox={bbox}")
            return None
        files = ea.download(res, outdir)
        if not files:
            print(f"[{short_name}] encontrado pero no se descargó.")
            return None
        return Path(files[0])
    except Exception as e:
        print(f"[{short_name}] error: {e}")
        return None

def open_nc_any(path: Path):
    """Abre un NetCDF con xarray probando motores comunes."""
    for engine in ("netcdf4", "h5netcdf", None):  # None => default backend
        try:
            ds = xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)
            return ds
        except Exception:
            pass
    raise RuntimeError(f"No se pudo abrir {path} con xarray")

# =========================
# PIPELINE
# =========================
all_data = {}

def obtain_data(bbox, blocks):
    """
    Descarga y abre los datasets definidos en 'blocks' para las coordenadas 'bbox',
    guardando todo en la variable global 'all_data' sin imprimir información detallada.
    Solo muestra un mensaje general al inicio.
    """
    global all_data
    print(f"\nImprimiendo información de estas coordenadas: {bbox}\n")

    all_data = {}

    for block, products in blocks.items():
        all_data[block] = {}

        for sn in products:
            p = fetch_one(sn)
            if not p:
                continue

            try:
                ds = open_nc_any(p)
            except Exception:
                continue

            # Caso TEMPO L3: tiene subgrupo 'product'
            if sn.startswith("TEMPO_"):
                try:
                    ds_prod = xr.open_dataset(p, engine="netcdf4", group="product")
                    all_data[block][sn] = {"root": ds, "product": ds_prod}
                except Exception:
                    all_data[block][sn] = {"root": ds}
            else:
                all_data[block][sn] = {"root": ds}
# =========================
# AQI helpers (columna→ppb, selección de variable, bbox & QA)
# =========================

def column_to_ppb(col_molec_cm2, pbl_m=1000.0):
    """
    Convierte columna (molecules/cm^2) a mezcla superficial aproximada (ppb),
    asumiendo que todo el contaminante está dentro de la PBL (pbl_m).
    """
    # ppb ≈ (col[molec/cm2] * 1e4 [cm2->m2]) / (n_air * PBL[m]) * 1e9
    return (col_molec_cm2 * 1e4) / (N_AIR_1ATM_298K * pbl_m) * 1e9

def classify_three_levels_gas(value_ppb, gas):
    """
    Devuelve nivel 0,1,2 (0=🟢, 1=🟡, 2=🔴) según umbrales por gas.
    CO se clasifica en ppm (convertimos de ppb a ppm dentro).
    """
    if value_ppb is None or np.isnan(value_ppb):
        return None  # <- si no hay dato útil, no clasifica
    thresholds = {
        "NO2":  (50.0, 100.0),   # ppb
        "O3":   (70.0, 120.0),   # ppb
        "SO2":  (35.0, 75.0),    # ppb
        "COppm":(4.4, 9.0),      # ppm
    }
    if gas == "CO":
        co_ppm = value_ppb / 1000.0  # ppb -> ppm
        low, high = thresholds["COppm"]
        if co_ppm < low:  return 0
        if co_ppm < high: return 1
        return 2
    else:
        low, high = thresholds[gas]
        if value_ppb < low:  return 0
        if value_ppb < high: return 1
        return 2

def label_from_level(level):
    return ["🟢 Saludable", "🟡 Moderado", "🔴 No saludable"][level]

def pick_s5p_var(ds_product, hints=("column","tropos","nitrogen","ozone","sulfur","monoxide","density")):
    """Heurística para elegir una variable científica en group='PRODUCT'."""
    keys = list(ds_product.data_vars)
    if not keys:
        return None
    # Busca por palabras clave en attrs y nombre
    for k in keys:
        blob = (k + " " + str(ds_product[k].attrs)).lower()
        if any(h in blob for h in hints):
            return k
    # Si no encontramos por hints, devuelve la primera
    return keys[0]

def read_s5p_mean_ppb(nc_path: Path, gas: str, pbl_m=1000.0):
    """
    Abre S5P L2 NRT (NetCDF), toma group='PRODUCT', elige variable,
    hace promedio simple y devuelve valor en ppb (o ppb-equivalente para CO).
    """
    ds_prod = xr.open_dataset(nc_path, engine="netcdf4", group="PRODUCT")
    varname = pick_s5p_var(ds_prod)
    if varname is None:
        raise ValueError(f"No se encontró variable científica en {nc_path.name}")
    # media ignorando NaN/máscaras
    col = ds_prod[varname]
    mean_col = float(col.where(np.isfinite(col)).mean().item())  # molecules/cm^2
    # Convertir columna a ppb aprox
    value_ppb = column_to_ppb(mean_col, pbl_m=pbl_m)
    return value_ppb, varname, ds_prod[varname].attrs

def pick_latest_path(pattern: str):
    cand = sorted(Path("data").glob(pattern))
    return max(cand, key=lambda p: p.stat().st_mtime) if cand else None

def open_s5p_with_geo(nc_path):
    ds_p = xr.open_dataset(nc_path, engine="netcdf4", group="PRODUCT")
    # lat/lon
    try:
        ds_g = xr.open_dataset(nc_path, engine="netcdf4", group="PRODUCT/SUPPORT_DATA/GEOLOCATIONS")
        for c in ("latitude","longitude"):
            if c in ds_g:
                ds_p = ds_p.assign_coords({c: ds_g[c]})
    except Exception:
        pass
    return ds_p

VAR_HINT = {
    "NO2": r"tropos.*no2|nitrogen.*tropos.*column",
    "O3":  r"tropo.*ozone.*column|o3.*tropo",
    "SO2": r"so2|sulphur|sulfur.*dioxide.*column",
    "CO":  r"carbon.*monoxide.*column|co.*total"
}

def pick_var_regex(ds, regex):
    import re
    pat = re.compile(regex, re.I)
    for k, v in ds.data_vars.items():
        blob = f"{k} {v.attrs}".lower()
        if pat.search(blob):
            return k
    return None

def bbox_mask(ds, bbox):
    if ("latitude" not in ds.coords) or ("longitude" not in ds.coords):
        return None
    lat, lon = ds["latitude"], ds["longitude"]
    return ((lat >= bbox[1]) & (lat <= bbox[3]) &
            (lon >= bbox[0]) & (lon <= bbox[2]))

def quality_mask(ds, thr=0.5):
    # S5P típico:
    if "qa_value" in ds:
        return ds["qa_value"] >= thr
    for k in ds.data_vars:
        if k.endswith("_validity"):
            return ds[k] > 0
    return None

def to_molec_cm2(mean_val, units: str | None):
    """
    Convierte columna a molecules/cm^2 según 'units'.
    - 'mol m-2' o 'mol/m^2'  -> * AVOG_N / 1e4
    - 'molec cm-2'           -> tal cual
    Si no reconoce, asume mol/m^2 (S5P típico) por seguridad.
    """
    if units is None:
        units = ""
    u = units.lower().replace(" ", "").replace("^", "")
    # variantes comunes
    if ("molec" in u and ("cm-2" in u or "/cm2" in u)) or ("molecules/cm2" in u):
        return float(mean_val)  # ya está en molec/cm^2
    if ("mol/m2" in u) or ("molm-2" in u) or ("molm^-2" in u) or ("molm−2" in u) or ("molm–2" in u) or ("molm^−2" in u) or ("molm⁻²" in u) or ("molm-²" in u) or ("molm-2" in u) or ("molm-²" in u) or ("molm−²" in u) or ("molm−2" in u) or ("molm-2" in u) or ("molm-2" in u):
        return float(mean_val) * AVOG_N / 1e4
    if ("molm-2" in u) or ("molm-2" in u) or ("molm^-2" in u) or ("molm-2" in u) or ("molm-2" in u):
        return float(mean_val) * AVOG_N / 1e4
    if ("molm-2" in u) or ("mol m-2" in units.lower()):
        return float(mean_val) * AVOG_N / 1e4
    # por defecto: trata como mol/m2 (caso S5P)
    return float(mean_val) * AVOG_N / 1e4


def read_s5p_mean_ppb_bbox(nc_path, gas, bbox, pbl_m=1000.0):
    ds = open_s5p_with_geo(nc_path)
    var = pick_var_regex(ds, VAR_HINT[gas])
    if var is None:
        print(f"    [WARN] No se encontró var para {gas} en {nc_path.name}")
        return np.nan, None, {}

    base = ds[var]

    tries = [
        ("QA>=0.5 + bbox", 0.5, True),
        ("QA>=0.2 + bbox", 0.2, True),
        ("sin QA + bbox",  None, True),
        ("sin QA + sin bbox", None, False),
    ]

    for label, qa_thr, use_bbox in tries:
        field = base
        if qa_thr is not None:
            qm = quality_mask(ds, thr=qa_thr)
            if qm is not None:
                field = field.where(qm)
        if use_bbox:
            bm = bbox_mask(ds, bbox)
            if bm is not None:
                field = field.where(bm)

        field = field.where(field > 0)  # anti-ceros
        _debug_stats(label, field)

        # --- aquí: media y conversión de unidades ---
        mean_val = field.where(np.isfinite(field)).mean().item()
        if mean_val is None or not np.isfinite(mean_val):
            continue

        units = ds[var].attrs.get("units", None)
        col_molec_cm2 = to_molec_cm2(float(mean_val), units)
        val_ppb = column_to_ppb(col_molec_cm2, pbl_m=pbl_m)
        return val_ppb, var, dict(ds[var].attrs)

    print("    [INFO] Sin datos válidos tras todos los intentos.")
    return np.nan, var, dict(ds[var].attrs)

def aqi_simple_from_files(files_dict, bbox, pbl_m=1000.0):
    """
    files_dict: dict con rutas por gas (ya descargadas), ej:
        {
          "NO2": Path("...NO2....nc"),
          "O3":  Path("...O3....nc"),
          "SO2": Path("...SO2...nc"),
          "CO":  Path("...CO....nc"),
        }
    Devuelve dict con valores, niveles por gas y nivel final (peor caso).
    """
    out = {"inputs": {}, "niveles": {}, "detalles": {}}
    niveles_validos = []

    for gas, path in files_dict.items():
        if not path or not path.exists():
            continue
        # usa lectura con bbox y QA
        ppb, var, attrs = read_s5p_mean_ppb_bbox(path, gas, bbox, pbl_m=pbl_m)
        lvl = classify_three_levels_gas(ppb, gas)

        out["inputs"][gas]  = ppb
        out["niveles"][gas] = lvl
        out["detalles"][gas]= {"var": var, "units_in": "molecules/cm^2", "attrs": dict(attrs), "file": path.name}
        if lvl is not None:
            niveles_validos.append(lvl)

    worst = max(niveles_validos) if niveles_validos else None
    out["AQ_simple_nivel"]   = worst
    out["AQ_simple_etiqueta"]= (label_from_level(worst) if worst is not None else "Sin datos suficientes")
    return out

def _debug_stats(name, arr):
    import numpy as np
    a = arr.where(np.isfinite(arr))
    n = int(a.count().item())
    vmin = float(a.min().item()) if n else float("nan")
    vmax = float(a.max().item()) if n else float("nan")
    print(f"    [{name}] n_valid={n}, min={vmin:.3e}, max={vmax:.3e}")
    
# =========================
# Archivos por variable (usando nombres descargados en variables)
# =========================
def evaluar_calidad_aire(files, bbox, pbl_m=1000.0):
    """
    Retorna un dict con campos para tu API:
      - aqi_index: int {0,1,2}
      - aqi_label_es: str {"buena","regular","mala"} (o "sin_datos")
      - exercise_index: int {0,1,2} (0=evitar,1=precaución,2=ok)
      - dust_index: int {0,1,2} o -1 si no hay AER_AI
      - health_risk_index: int {0,1,2}
      - gases_ppb: dict con ppb por gas (NaN si faltó)
    """

    # 1) AQI con tus funciones
    res = aqi_simple_from_files(files, bbox=bbox, pbl_m=pbl_m)
    worst = res.get("AQ_simple_nivel", None)  # 0/1/2 o None
    if worst is None:
        aqi_index = None
        aqi_label_es = "sin_datos"
    else:
        aqi_index = int(worst)
        aqi_label_es = {0: "buena", 1: "regular", 2: "mala"}[aqi_index]

    # 2) Health risk = peor de los gases
    health_risk_index = aqi_index if aqi_index is not None else -1

    # 3) Exercise index (simple y útil): inverso del riesgo
    #    2=ok si AQI 0; 1=precaución si AQI 1; 0=evitar si AQI 2; -1 si sin datos
    exercise_index = {0: 2, 1: 1, 2: 0}.get(aqi_index, -1)

    # 4) Dust index (AER_AI opcional)
    #    Umbrales típicos: <0.5 limpio, 0.5–1 moderado, >=1 alto (humo/polvo).
    #    Intentamos descargar/usar AER_AI si no lo tienes; si falla => -1.
    def _leer_aer_ai(bbox_local):
        try:
            # Reutiliza tu fetch_one; si ya lo incluyes en blocks, llegará rápido.
            aer_file = fetch_one("S5P_L2__AER_AI__NRTI")
            if not aer_file:
                return None
            # Abrimos y buscamos la variable típica
            ds = open_s5p_with_geo(aer_file)
            # Nombre usual:
            if "absorbing_aerosol_index" in ds.data_vars:
                v = ds["absorbing_aerosol_index"]
            else:
                # fallback por regex
                vname = pick_var_regex(ds, r"aer.*index|absorbing.*index|aerosol.*index")
                if not vname:
                    return None
                v = ds[vname]
            # QA si existe
            qm = quality_mask(ds, thr=0.5)
            if qm is not None:
                v = v.where(qm)
            # bbox
            bm = bbox_mask(ds, bbox_local)
            if bm is not None:
                v = v.where(bm)
            mean_ai = v.where(np.isfinite(v)).mean().item()
            return float(mean_ai) if (mean_ai is not None and np.isfinite(mean_ai)) else None
        except Exception:
            return None

    aer_ai = _leer_aer_ai(bbox)
    if aer_ai is None:
        dust_index = -1
    else:
        if aer_ai < 0.5:
            dust_index = 0
        elif aer_ai < 1.0:
            dust_index = 1
        else:
            dust_index = 2

    # 5) Salida compacta para API
    gases_ppb = {}
    for g, v in res.get("inputs", {}).items():
        # deja NaN si faltó; Flask/JSON lo puede serializar como null si conviertes
        gases_ppb[g] = float(v) if (v is not None and not np.isnan(v)) else float("nan")

    return {
        "aqi_index": aqi_index if aqi_index is not None else -1,
        "aqi_label_es": aqi_label_es,
        "exercise_index": exercise_index,
        "dust_index": dust_index,
        "health_risk_index": health_risk_index if health_risk_index is not None else -1,
        "gases_ppb": gases_ppb,
    }

#######################################################################################################################
#################################################Flask api functions###################################################
#######################################################################################################################
app = Flask(__name__)

@app.route('/')
def home():
    return "¡Hola Mundo desde Flask!"


@app.route('/saludo/<nombre>', methods=['GET'])
def saludo(nombre):
    return jsonify({"mensaje": f"Hola {nombre}, bienvenido a Flask!"})

@app.route('/app/<user>', methods=['POST'])
def app_request(user):
    
    data = request.get_json()
    if not data or "lat" not in data or "lon" not in data or "age" not in data:
        return jsonify({"error": "Faltan parámetros: age, lat, lon"}), 400

    age = data["age"]
    lat = float(data["lat"])
    lon = float(data["lon"])
    
    bbox = point_to_bbox(lat, lon, 50000)
    obtain_data(bbox, blocks)

    no2_file = fetch_one("S5P_L2__NO2____HiR_NRT")
    o3_file  = fetch_one("S5P_L2__O3_TCL_NRT")
    so2_file = fetch_one("S5P_L2__SO2____HiR_NRT")
    co_file  = fetch_one("S5P_L2__CO_____HiR_NRT")

    files = {
        "NO2": no2_file,
        "O3":  o3_file,
        "SO2": so2_file,
        "CO":  co_file,
    }
    PBL_METROS = 1000.0  # cámbialo por PBL real cuando integres GEOS-FP/MERRA-2
    resultado_api = evaluar_calidad_aire(files, bbox=bbox, pbl_m=PBL_METROS)
    respuesta = {
        "mensaje": f"{user},{age} information",
        "ubicacion": {
            "lat": lat,
            "lon": lon,
            "age": age,
        },
        "resultados": resultado_api
    }

    return jsonify(respuesta)

@app.route('/cookie_hardware/<ID>', methods=['POST'])
def get_hardware_data(ID):
    data = request.get_json(silent=True)

    # Validar que se haya enviado JSON
    if not data:
        return jsonify({"error": "no_json_received"}), 400

    # Validar sensores esperados
    expected_keys = {"PMS5003", "MQ131", "MQ4"}
    missing = expected_keys - data.keys()

    if missing:
        return jsonify({
            "err": "missing_fields",
            "missing": list(missing),
        }), 400

    # Buscar ID de hardware
    coords = HARDWARE_TABLE.get(ID)
    if not coords:
        return jsonify({"err": "hardware_id_not_found", "id": ID}), 404
    
    record = {
        "coords": coords,
        "sensors": {
            "PMS5003": data["PMS5003"],
            "MQ131":   data["MQ131"],
            "MQ4":     data["MQ4"],
        },
        "received_at": datetime.now(timezone.utc).isoformat()
    }
    with STORE_LOCK:
        SENSOR_STORE[ID] = record
        _save_store()
    # Estructurar la respuesta
    #response = {
    #    "hardware_id": ID,
    #    "ubicacion": coords,
    #    "lecturas_recibidas": {
    #        "sensor1": data["sensor1"],
    #        "sensor2": data["sensor2"],
    #        "sensor3": data["sensor3"]
    #    },
    #    "estado": "OK",
    #    "mensaje": f"Datos de {ID} recibidos correctamente."
    #}

    return jsonify({"status": "ok"}), 200
    
    
@app.route("/app/user", methods=["POST"])
def user():
    if not request.is_json:
        return jsonify(error="Expected application/json",
                       got=request.headers.get("Content-Type")), 415
    data = request.get_json()
    # valida campos
    for k in ("latitud","longitud","edad","condicion"):
        if k not in data:
            return jsonify(error=f"Falta '{k}'"), 400
    return jsonify(ok=True), 201


if __name__ == '__main__':
    app.run(debug=True)
    
    
    