import earthaccess as ea
from pathlib import Path
from datetime import datetime, timedelta, timezone
import xarray as xr
import numpy as np
from flask import Flask, request, jsonify
import math
import os
import json
from threading import Lock, Timer

# =========================
# CONFIG
# =========================
# Hardware simple DB
HARDWARE_TABLE = {
    "001": {"lat": 19.4326, "lon": -99.1332},   # CDMX_Zocalo
    "002": {"lat": 19.4363, "lon": -99.0719},   # CDMX_Int.Airport
    "003": {"lat": 19.4204, "lon": -99.1819},   # CDMX_Chapultepec Castle
    "004": {"lat": 40.4169, "lon": -3.7038},    # Madrid_Puerta del Sol
    "005": {"lat": 31.8591, "lon": -116.6243},  # Ensenada_Downtown
    "006": {"lat": 19.5296, "lon": -96.9236},   # Xalapa
    "007": {"lat": 40.7580, "lon": -73.9855},   # NYC_TimesSquare
    "008": {"lat": 51.5079, "lon": -0.1283},    # London_TrafalgarSq
    "009": {"lat": 48.8584, "lon": 2.2945},     # Paris_EiffelTower
    "010": {"lat": 35.6595, "lon": 139.7005},   # Tokyo_ShibuyaCrossing
    "011": {"lat": -33.8568, "lon": 151.2153},  # Sydney_OperaHouse
    "012": {"lat": 30.0444, "lon": 31.2357},    # Cairo_TahrirSquare
    "013": {"lat": -23.5614, "lon": -46.6566},  # SaoPaulo_PaulistaAve
    "014": {"lat": -1.2864, "lon": 36.8172},    # Nairobi_CBD
    "015": {"lat": 43.6426, "lon": -79.3871},   # Toronto_CNTower
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

BBOX = None  # (min_lon, min_lat, max_lon, max_lat) o None para sin filtro espacial
BBOX_LOCK = Lock()

AVOG_N = 6.022e23
N_AIR_1ATM_298K = 2.5e25  # moléculas/m^3 (aprox)

# =========================
# RANGO TEMPORAL (últimos 7 dias) con actualización diaria
# =========================
def update_temporal_iso():
    global temporal_iso
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)
    temporal_iso = (
        start_time.strftime("%Y-%m-%d"),
        end_time.strftime("%Y-%m-%d"),
    )
    print(f"[INFO] temporal_iso actualizado: {temporal_iso}")
    # vuelve a ejecutar en 24h
    Timer(86400, update_temporal_iso).start()

update_temporal_iso()

outdir = Path("data")
outdir.mkdir(exist_ok=True)

# =========================
# DATASETS per BLOCK
# =========================
blocks = {
    # 1) AQI — todo NRT global
    "AQI": [
        "S5P_L2__NO2____HiR_NRT",   # NO2 NRT (TROPOMI)
        "S5P_L2__O3_TCL_NRT",       # O3 troposférico NRT
        "S5P_L2__SO2____HiR_NRT",   # SO2 NRT (alta resolución)
        "S5P_L2__CO_____HiR_NRT",   # CO NRT
        "S5P_L2__AER_AI__NRTI",     # Aerosol Index NRT
    ],

    # 2) FIRE_DUST — NRT global
    "FIRE_DUST": [
        "S5P_L2__AER_AI__NRTI",     # Absorbing Aerosol Index
        "S5P_L2__CO_____HiR_NRT",   # CO NRT (humo/incendios)
        "S5P_L2__HCHO___NRTI",      # Formaldehído NRT (piroquímica)
    ],

    # 3) MET_EFFECT — NO2 + Meteorología (preferir GEOS-FP NRT; si no, MERRA-2)
    "MET_EFFECT": [
        "S5P_L2__NO2____HiR_NRT",
        "MERRA2_400.tavg1_2d_slv_Nx"  # Temp 2m, vientos, PBLH (reanálisis)
    ],

    # 4) HEALTH_RISK — NO2 + O3 + PM + Temp (PM en MERRA-2; O3/NO2 NRT)
    "HEALTH_RISK": [
        "S5P_L2__NO2____HiR_NRT",
        "S5P_L2__O3_TCL_NRT",
        "M2T1NXAER",                # MERRA-2 Aerosol Diagnostics (PM2.5, etc.)
        "MERRA2_400.tavg1_2d_slv_Nx"
    ],
}

#######################################################################################################################
################################# HELPERS DE APERTURA UNIVERSAL (evita engine fijo) ###################################
#######################################################################################################################
def _open_ds_try_engines(path, group=None):
    """
    Abre NetCDF/HDF5 probando engines ('netcdf4', 'h5netcdf', y default).
    Evita errores cuando el entorno no tiene un engine específico instalado.
    """
    engines = ("netcdf4", "h5netcdf", None)  # None => xarray elige
    last_err = None
    for eng in engines:
        try:
            if group is None:
                return xr.open_dataset(path, engine=eng) if eng else xr.open_dataset(path)
            else:
                return xr.open_dataset(path, engine=eng, group=group) if eng else xr.open_dataset(path, group=group)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No se pudo abrir {path} (group={group}): {last_err}")

def open_nc_any(path: Path):
    """Compatibilidad general: abre NetCDF con fallback automático de engine."""
    return _open_ds_try_engines(path, group=None)

def open_s5p_with_geo(nc_path):
    """
    Abre un archivo Sentinel-5P, combinando datos científicos (PRODUCT)
    con coordenadas geográficas (lat/lon) si están en GEOLOCATIONS.
    """
    ds_p = _open_ds_try_engines(nc_path, group="PRODUCT")
    try:
        ds_g = _open_ds_try_engines(nc_path, group="PRODUCT/SUPPORT_DATA/GEOLOCATIONS")
        for c in ("latitude", "longitude"):
            if c in ds_g:
                ds_p = ds_p.assign_coords({c: ds_g[c]})
    except Exception:
        pass
    return ds_p

#######################################################################################################################
################################# FUNCTIONS FOR ANALYZING DATA FROM SATELLITES ########################################
#######################################################################################################################
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # radio medio de la Tierra en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def compute_nearby_sensor_avgs(lat, lon, radio_km=5.0):
    """
    Busca sensores en SENSOR_STORE dentro de 'radio_km' km y
    devuelve (promedios, lista_sensores_cercanos).
    'promedios' tiene claves: PM1_0, PM2_5, PM10, O3_ppm, CH4_ppm (o None).
    """
    nearby = []
    pm1_vals, pm25_vals, pm10_vals, o3_vals, ch4_vals = [], [], [], [], []

    with STORE_LOCK:
        for sensor_id, sensor_data in SENSOR_STORE.items():
            coords = sensor_data.get("coords")
            if not coords:
                continue
            dist = haversine_km(lat, lon, coords["lat"], coords["lon"])
            if dist <= radio_km:
                lect = sensor_data.get("sensors", {})
                nearby.append({
                    "id": sensor_id,
                    "distancia_km": round(dist, 2),
                    "lecturas": lect,
                    "hora": sensor_data.get("received_at", "")
                })
                # PMS5003
                pms = lect.get("PMS5003", {})
                if "PM1_0" in pms: pm1_vals.append(float(pms["PM1_0"]))
                if "PM2_5" in pms: pm25_vals.append(float(pms["PM2_5"]))
                if "PM10"  in pms: pm10_vals.append(float(pms["PM10"]))
                # MQ131 (O3)
                mq131 = lect.get("MQ131", {})
                if "O3_ppm" in mq131: o3_vals.append(float(mq131["O3_ppm"]))
                # MQ4 (CH4)
                mq4 = lect.get("MQ4", {})
                if "CH4_ppm" in mq4: ch4_vals.append(float(mq4["CH4_ppm"]))

    avgs = {
        "PM1_0":  round(np.mean(pm1_vals), 2) if pm1_vals else None,
        "PM2_5":  round(np.mean(pm25_vals), 2) if pm25_vals else None,
        "PM10":   round(np.mean(pm10_vals), 2) if pm10_vals else None,
        "O3_ppm": round(np.mean(o3_vals), 4)   if o3_vals else None,
        "CH4_ppm":round(np.mean(ch4_vals), 4)  if ch4_vals else None,
    }
    return avgs, nearby

# =========================
# AUTENTICATION
# =========================
# Usa .netrc en el HOME (o cambia a "environment" si prefieres variables de entorno)
ea.login(strategy="netrc", persist=True)

# =========================
# HELPERS
# =========================
def fetch_one(short_name: str, bbox_local=None):
    """
    Busca y descarga 1 granule. Si la versión de earthaccess falla con bounding_box,
    reintenta sin bbox (fallback).
    """
    # fuerza tupla (west, south, east, north)
    bbox_tuple = None
    if bbox_local and len(bbox_local) == 4:
        bbox_tuple = tuple(float(x) for x in bbox_local)

    try:
        res = ea.search_data(
            short_name=short_name,
            temporal=temporal_iso,
            bounding_box=bbox_tuple,   # <- tupla de 4 floats
            count=3
        )
    except TypeError as e:
        # Caso típico: "DataGranules.bounding_box() missing 3 required positional arguments..."
        print(f"[{short_name}] WARN bbox en search_data: {e}. Reintentando sin bbox…")
        res = ea.search_data(
            short_name=short_name,
            temporal=temporal_iso,
            count=1
        )
    except Exception as e:
        print(f"[{short_name}] error en search_data: {e}")
        return None

    if not res:
        print(f"[{short_name}] sin resultados en {temporal_iso} (bbox={bbox_tuple})")
        return None

    try:
        files = ea.download(res, outdir)
    except Exception as e:
        print(f"[{short_name}] error en download: {e}")
        return None

    if not files:
        print(f"[{short_name}] encontrado pero no se descargó.")
        return None

    return Path(files[0])

# =========================
# PIPELINE
# =========================
all_data = {}
bbox = None
def obtain_data(bbox_local, blocks_dict):
    """
    Descarga y abre los datasets definidos en 'blocks_dict' para 'bbox_local',
    guardando todo en la variable global 'all_data' sin imprimir información detallada.
    Solo imprime una línea de estado.
    """
    global all_data, bbox
    bbox = bbox_local  # para que otras funciones lo vean si lo usan
    print(f"\nImprimiendo información de estas coordenadas: {bbox}\n")

    all_data = {}
    for block, products in blocks_dict.items():
        all_data[block] = {}
        for sn in products:
            p = fetch_one(sn, bbox_local=bbox_local)  # <- pasa bbox_local aquí
            if not p:
                continue
            try:
                ds = open_nc_any(p)
            except Exception:
                continue
            if sn.startswith("TEMPO_"):
                try:
                    ds_prod = _open_ds_try_engines(p, group="product")
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
    Devuelve nivel 0,1,2 (0=good, 1=medium, 2=bad) según umbrales por gas.
    CO se clasifica en ppm (convertimos de ppb a ppm dentro).
    """
    if value_ppb is None or np.isnan(value_ppb):
        return None
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
    return ["Saludable", "Moderado", "No saludable"][level]

def pick_s5p_var(ds_product, hints=("column","tropos","nitrogen","ozone","sulfur","monoxide","density")):
    """Heurística para elegir una variable científica en group='PRODUCT'."""
    keys = list(ds_product.data_vars)
    if not keys:
        return None
    for k in keys:
        blob = (k + " " + str(ds_product[k].attrs)).lower()
        if any(h in blob for h in hints):
            return k
    return keys[0]

def read_s5p_mean_ppb(nc_path: Path, gas: str, pbl_m=1000.0):
    """
    Abre S5P L2 NRT (NetCDF), toma group='PRODUCT', elige variable,
    hace promedio simple y devuelve valor en ppb (o ppb-equivalente para CO).
    """
    ds_prod = _open_ds_try_engines(nc_path, group="PRODUCT")
    varname = pick_s5p_var(ds_prod)
    if varname is None:
        raise ValueError(f"No se encontró variable científica en {nc_path.name}")
    col = ds_prod[varname]
    mean_col = float(col.where(np.isfinite(col)).mean().item())  # molecules/cm^2
    value_ppb = column_to_ppb(mean_col, pbl_m=pbl_m)
    return value_ppb, varname, ds_prod[varname].attrs

def pick_latest_path(pattern: str):
    cand = sorted(Path("data").glob(pattern))
    return max(cand, key=lambda p: p.stat().st_mtime) if cand else None

def VAR_HINT_init():
    return {
        "NO2": r"tropos.*no2|nitrogen.*tropos.*column",
        "O3":  r"tropo.*ozone.*column|o3.*tropo",
        "SO2": r"so2|sulphur|sulfur.*dioxide.*column",
        "CO":  r"carbon.*monoxide.*column|co.*total"
    }

VAR_HINT = VAR_HINT_init()

def pick_var_regex(ds, regex):
    import re
    pat = re.compile(regex, re.I)
    for k, v in ds.data_vars.items():
        blob = f"{k} {v.attrs}".lower()
        if pat.search(blob):
            return k
    return None
def point_to_bbox(lat_deg, lon_deg, radius_m):
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat_deg)))
    dlat = radius_m * lat_per_m
    dlon = radius_m * lon_per_m
    return [lon_deg - dlon, lat_deg - dlat, lon_deg + dlon, lat_deg + dlat]

def bbox_mask(ds, bbox):
    # Si no hay bbox (None), no se aplica máscara espacial
    if bbox is None:
        return None
    if ("latitude" not in ds.coords) or ("longitude" not in ds.coords):
        return None
    lat, lon = ds["latitude"], ds["longitude"]
    return ((lat >= bbox[1]) & (lat <= bbox[3]) &
            (lon >= bbox[0]) & (lon <= bbox[2]))

def quality_mask(ds, thr=0.5):
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
    if ("molec" in u and ("cm-2" in u or "/cm2" in u)) or ("molecules/cm2" in u):
        return float(mean_val)
    if ("mol/m2" in u) or ("molm-2" in u) or ("mol m-2" in units.lower()):
        return float(mean_val) * AVOG_N / 1e4
    return float(mean_val) * AVOG_N / 1e4

def read_s5p_mean_ppb_bbox(nc_path, gas, bbox, pbl_m=1000.0):
    """
    Lee un NetCDF S5P y calcula el promedio en ppb para un gas.
    Si cualquier paso falla, regresa NaN y no corta el flujo.
    """
    try:
        ds = open_s5p_with_geo(nc_path)
    except Exception as e:
        print(f"[{gas}] no se pudo abrir {getattr(nc_path, 'name', nc_path)}: {e}")
        return np.nan, None, {}

    var = pick_var_regex(ds, VAR_HINT[gas])
    if var is None:
        print(f"[{gas}] WARN: variable no encontrada en {getattr(nc_path, 'name', nc_path)}")
        return np.nan, None, {}

    base = ds[var]
    tries = [
        ("QA>=0.5 + bbox", 0.5, True),
        ("QA>=0.2 + bbox", 0.2, True),
        ("sin QA + bbox",  None, True),
        ("sin QA + sin bbox", None, False),
    ]

    for _, qa_thr, use_bbox in tries:
        field = base
        try:
            if qa_thr is not None:
                qm = quality_mask(ds, thr=qa_thr)
                if qm is not None:
                    field = field.where(qm)
            if use_bbox:
                bm = bbox_mask(ds, bbox)  # si bbox es None, no se aplica
                if bm is not None:
                    field = field.where(bm)

            field = field.where(field > 0)
            mean_val = field.where(np.isfinite(field)).mean().item()
            if mean_val is None or not np.isfinite(mean_val):
                continue

            units = ds[var].attrs.get("units", None)
            col_molec_cm2 = to_molec_cm2(float(mean_val), units)
            val_ppb = column_to_ppb(col_molec_cm2, pbl_m=pbl_m)
            return val_ppb, var, dict(ds[var].attrs)
        except Exception:
            continue

    return np.nan, var, dict(ds[var].attrs) if var else {}

def aqi_simple_from_files(files_dict, bbox, pbl_m=1000.0):
    """
    Itera por gases y calcula ppb. Si un archivo falta o falla, lo salta.
    """
    out = {"inputs": {}, "niveles": {}, "detalles": {}}
    niveles_validos = []

    for gas, path in files_dict.items():
        if not path or not Path(path).exists():
            continue
        try:
            ppb, var, attrs = read_s5p_mean_ppb_bbox(path, gas, bbox, pbl_m=pbl_m)
        except Exception:
            continue

        out["inputs"][gas] = float(ppb) if (ppb is not None and np.isfinite(ppb)) else float("nan")
        out["detalles"][gas] = {
            "var": var, "units_in": "molecules/cm^2", "attrs": dict(attrs) if attrs else {}, "file": Path(path).name
        }

        if ppb is not None and np.isfinite(ppb):
            lvl = classify_three_levels_gas(ppb, gas)
            out["niveles"][gas] = lvl
            if lvl is not None:
                niveles_validos.append(lvl)
        else:
            out["niveles"][gas] = None

    worst = max(niveles_validos) if niveles_validos else None
    out["AQ_simple_nivel"] = worst
    out["AQ_simple_etiqueta"] = (label_from_level(worst) if worst is not None else "Sin datos suficientes")
    return out

# =========================
# Fusión y salida API
# =========================
def evaluar_calidad_aire(files, bbox, pbl_m=1000.0, sensor_avgs=None):
    """
    Retorna un dict con:
      - aqi_index: int {0,1,2,-1}
      - aqi_label_es: {"buena","regular","mala","sin_datos"}
      - exercise_index: {0,1,2,-1}
      - dust_index: {0,1,2,-1}
      - health_risk_index: {0,1,2,-1}
      - gases_ppb: dict (ppb satelital + PM/CH4 informativos)
      - sensor_merge: info fusión sensores
    """
    res = aqi_simple_from_files(files, bbox=bbox, pbl_m=pbl_m)
    worst = res.get("AQ_simple_nivel", None)
    if worst is None:
        aqi_index = None
        aqi_label_es = "sin_datos"
    else:
        aqi_index = int(worst)
        aqi_label_es = {0: "buena", 1: "regular", 2: "mala"}[aqi_index]

    health_risk_index = aqi_index if aqi_index is not None else -1
    exercise_index = {0: 2, 1: 1, 2: 0}.get(aqi_index, -1)

    # Dust index (AER_AI opcional)
    def _leer_aer_ai(bbox_local):
        try:
            aer_file = fetch_one("S5P_L2__AER_AI__NRTI")
            if not aer_file:
                return None
            ds = open_s5p_with_geo(aer_file)
            v = None
            if "absorbing_aerosol_index" in ds.data_vars:
                v = ds["absorbing_aerosol_index"]
            else:
                vname = pick_var_regex(ds, r"aer.*index|absorbing.*index|aerosol.*index")
                if vname:
                    v = ds[vname]
            if v is None:
                return None
            qm = quality_mask(ds, thr=0.5)
            if qm is not None:
                v = v.where(qm)
            bm = bbox_mask(ds, bbox_local)  # si bbox_local es None, no recorta
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
        dust_index = 0 if aer_ai < 0.5 else (1 if aer_ai < 1.0 else 2)

    gases_ppb = {}
    for g, v in res.get("inputs", {}).items():
        gases_ppb[g] = float(v) if (v is not None and not np.isnan(v)) else float("nan")

    sensor_fusion_info = {"aplicada": False, "regla": "Sin datos de sensores cercanos."}

    # Fusión con sensores cercanos
    if sensor_avgs and isinstance(sensor_avgs, dict):
        def pm25_to_level(pm25):
            if pm25 is None or not np.isfinite(pm25):
                return None
            if pm25 < 12.0: return 0
            if pm25 < 35.5: return 1
            return 2

        o3_ppm = sensor_avgs.get("O3_ppm", None)
        if o3_ppm is not None and np.isfinite(o3_ppm):
            o3_ppb_sensor = float(o3_ppm) * 1000.0
            gases_ppb["O3"] = o3_ppb_sensor
            lvl_o3 = classify_three_levels_gas(o3_ppb_sensor, "O3")
        else:
            lvl_o3 = classify_three_levels_gas(gases_ppb.get("O3", np.nan), "O3")

        pm25 = sensor_avgs.get("PM2_5", None)
        pm_level = pm25_to_level(pm25)
        ch4_ppm = sensor_avgs.get("CH4_ppm", None)

        niveles = []
        if aqi_index is not None: niveles.append(aqi_index)
        if pm_level is not None:  niveles.append(pm_level)
        if lvl_o3 is not None:    niveles.append(lvl_o3)

        aqi_fusion = max(niveles) if niveles else None
        if aqi_fusion is None:
            aqi_label_fusion = "sin_datos"
            exercise_fusion = -1
            health_fusion = -1
        else:
            aqi_label_fusion = {0: "buena", 1: "regular", 2: "mala"}[aqi_fusion]
            exercise_fusion = {0: 2, 1: 1, 2: 0}[aqi_fusion]
            health_fusion = aqi_fusion

        gases_ppb["PM2_5_ugm3"] = float(pm25) if (pm25 is not None and np.isfinite(pm25)) else float("nan")
        gases_ppb["PM1_0_ugm3"] = float(sensor_avgs.get("PM1_0")) if sensor_avgs.get("PM1_0") is not None else float("nan")
        gases_ppb["PM10_ugm3"]  = float(sensor_avgs.get("PM10"))  if sensor_avgs.get("PM10")  is not None else float("nan")
        gases_ppb["CH4_ppm"]    = float(ch4_ppm) if (ch4_ppm is not None and np.isfinite(ch4_ppm)) else float("nan")

        aqi_index = aqi_fusion if aqi_fusion is not None else aqi_index
        aqi_label_es = aqi_label_fusion
        exercise_index = exercise_fusion
        health_risk_index = health_fusion

        sensor_fusion_info = {
            "aplicada": True,
            "o3_ppm_entrada": o3_ppm,
            "o3_ppb_usado": (float(o3_ppm)*1000.0) if o3_ppm is not None else None,
            "pm25_ugm3": pm25,
            "pm_index": pm_level,
            "ch4_ppm": ch4_ppm,
            "regla": "Peor caso entre AQI satelital y PM2.5 de sensor; O3 de sensor sustituye al satelital si está disponible."
        }

    return {
        "aqi_index": aqi_index if aqi_index is not None else -1,
        "aqi_label_es": aqi_label_es,
        "exercise_index": exercise_index,
        "dust_index": dust_index,
        "health_risk_index": health_risk_index if health_risk_index is not None else -1,
        "gases_ppb": gases_ppb,
        "sensor_merge": sensor_fusion_info,
    }

#######################################################################################################################
################################################# Flask API functions #################################################
#######################################################################################################################
app = Flask(__name__)

@app.route('/')
def home():
    return "¡Hola Mundo desde Flask!"

@app.route('/saludo/<nombre>', methods=['GET'])
def saludo(nombre):
    return jsonify({"mensaje": f"Hola {nombre}, bienvenido a Flask!"})

def try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_payload(req):
    ct = (req.headers.get("Content-Type") or "").lower()
    raw = req.get_data(as_text=True)

    # 1) JSON normal
    if "application/json" in ct:
        data = request.get_json(silent=True)
        if isinstance(data, dict):
            return data, "json", ct, raw

    # 2) FORM ─ varias estrategias
    if "application/x-www-form-urlencoded" in ct:
        f = request.form.to_dict()

        # 2.a) Pares campo=valor (incluyendo nombre)
        if all(k in f for k in ("nombre", "latitud", "longitud", "edad", "condicion")):
            return f, "form-fields", ct, raw

        # 2.b) Form con UNA clave que en realidad es un JSON (casos MIT AI)
        if len(f) == 1:
            only_key, only_val = next(iter(f.items()))
            d = try_parse_json(only_key)
            if isinstance(d, dict):
                return d, "form-key-json", ct, raw
            d = try_parse_json(only_val)
            if isinstance(d, dict):
                return d, "form-value-json", ct, raw

        # 2.c) payload=data con JSON dentro
        for k in ("payload", "data"):
            if k in f:
                d = try_parse_json(f[k])
                if isinstance(d, dict):
                    return d, "form-payload-json", ct, raw

        # Último recurso: devolver el form crudo
        return f, "form-raw", ct, raw

    # 3) Intento forzado desde el cuerpo crudo (por si viene text/plain)
    d = try_parse_json(raw)
    if isinstance(d, dict):
        return d, "raw-json", ct, raw

    return None, None, ct, raw

def _nan_to_none(obj):
    """Convierte NaN/inf a None recursivamente para JSON válido."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


@app.route("/app/user", methods=["POST"])
def app_user():
    """
    Endpoint combinado:
    - Compatible con MIT App Inventor.
    - Requiere 'nombre' en el payload.
    - Devuelve SOLO 'resultados' con los índices calculados.
    """
    data, source, ct, raw = parse_payload(request)

    if not isinstance(data, dict):
        return jsonify(ok=False, error="No pude leer el cuerpo",
                       content_type=ct, raw=raw), 400

    # Validación mínima
    missing = [k for k in ("nombre", "latitud", "longitud", "edad", "condicion")
               if k not in data or str(data[k]).strip() == ""]
    if missing:
        return jsonify(ok=False, error="Faltan campos",
                       missing=missing, received=data, source=source,
                       content_type=ct), 400

    # Normaliza tipos / coma decimal
    try:
        nombre = str(data["nombre"]).strip()
        lat = float(str(data["latitud"]).replace(",", "."))
        lon = float(str(data["longitud"]).replace(",", "."))
        age = int(float(str(data["edad"]).replace(",", ".")))
        condicion = str(data["condicion"])
    except Exception as e:
        return jsonify(ok=False, error="Tipos inválidos",
                       detail=str(e), received=data), 400

    # radio opcional (m)
    try:
        radio_m = float(str(data.get("radio_m", "10000")).replace(",", "."))
    except Exception:
        radio_m = 10000.0

    # bbox centrado en el punto
    bbox_local = point_to_bbox(lat, lon, radio_m)

    # (opcional) actualizar un BBOX global si lo usas en otras partes
    try:
        global BBOX
    except NameError:
        pass
    else:
        try:
            BBOX_LOCK.acquire()
            BBOX = bbox_local
        finally:
            BBOX_LOCK.release()

    # Descarga/selección de archivos recientes
    try:
        obtain_data(bbox_local, blocks)

        no2_file = fetch_one("S5P_L2__NO2____HiR_NRT", bbox_local=bbox_local)
        o3_file  = fetch_one("S5P_L2__O3_TCL_NRT",     bbox_local=bbox_local)
        so2_file = fetch_one("S5P_L2__SO2____HiR_NRT", bbox_local=bbox_local)
        co_file  = fetch_one("S5P_L2__CO_____HiR_NRT", bbox_local=bbox_local)

        files = {"NO2": no2_file, "O3": o3_file, "SO2": so2_file, "CO": co_file}
    except Exception as e:
        return jsonify(ok=False, error="Error preparando datos satelitales", detail=str(e)), 500

    # Sensores cercanos → promedios
    sensor_avgs, sensores_cercanos = compute_nearby_sensor_avgs(lat, lon, radio_km=50.0)

    # Evaluación final (fusión)
    try:
        PBL_METROS = 1000.0
        resultado_api = evaluar_calidad_aire(
            files, bbox=bbox_local, pbl_m=PBL_METROS, sensor_avgs=sensor_avgs
        )
        # (opcional) enriquecer internamente; no se enviará en la respuesta final
        resultado_api.setdefault("sensor_merge", {})
        resultado_api["sensor_merge"]["sensores_cercanos"] = sensores_cercanos
        resultado_api["sensor_merge"]["radio_km"] = 50.0
        resultado_api["bbox_usado"] = bbox_local
    except Exception as e:
        return jsonify(ok=False, error="Error en evaluación de calidad del aire", detail=str(e)), 500

    # Solo el bloque 'resultados' y con NaN saneado
    resultados_puros = {
        "aqi_index": resultado_api.get("aqi_index"),
        "aqi_label_es": resultado_api.get("aqi_label_es"),
        "dust_index": resultado_api.get("dust_index"),
        "exercise_index": resultado_api.get("exercise_index"),
        "gases_ppb": resultado_api.get("gases_ppb", {}),
        "health_risk_index": resultado_api.get("health_risk_index"),
    }
    resultados_puros = _nan_to_none(resultados_puros)

    return jsonify({"resultados": resultados_puros}), 201


@app.route('/cookie_hardware/<ID>', methods=['POST'])
def get_hardware_data(ID):
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "no_json_received"}), 400

    expected_keys = {"PMS5003", "MQ131", "MQ4"}
    missing = expected_keys - data.keys()

    if missing:
        return jsonify({
            "err": "missing_fields",
            "missing": list(missing),
        }), 400

    coords = HARDWARE_TABLE.get(ID)
    if not coords:
        return jsonify({"err": "hardware_id_not_found", "id": ID}), 404

    record = {
        "coords": coords,
        "sensors": {
            "PMS5003": data["PMS5003"],  # {'PM1_0':..,'PM2_5':..,'PM10':..}
            "MQ131":   data["MQ131"],    # {'O3_ppm':..}
            "MQ4":     data["MQ4"],      # {'CH4_ppm':..}
        },
        "received_at": datetime.now(timezone.utc).isoformat()
    }
    print(record)
    with STORE_LOCK:
        SENSOR_STORE[ID] = record
        _save_store()

    return jsonify({"status": "ok"}), 200

@app.get("/healthz")
def healthz():
    return "ok", 200

if __name__ == '__main__':
    app.run(debug=True)
