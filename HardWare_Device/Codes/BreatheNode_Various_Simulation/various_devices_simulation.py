"""
VariousDevice
This code simulates the function or varios devices
doing the the actions of the microcontroler ESP32 (sending the the data of the 
sensors: PM5003, MQ5003, MQ-4, to the server)
"""
import time
import random
from datetime import datetime, timezone
import requests
import re

# URL of your local server endpoint

base = "https://breathe-nasa-space-apps-2025.onrender.com/cookie_hardware/000"

def make_urls(base_url: str, start: int, end: int, width: int = 3):
    """
    Build a list of URLs by replacing the trailing number in base_url
    with zero-padded integers from start to end (inclusive).
    """
    prefix = re.sub(r'\d+$', '', base_url)  # strip trailing digits
    return [f"{prefix}{i:0{width}d}" for i in range(start, end + 1)]

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def gen_pms5003():
    """Simulate particulate matter in µg/m³."""
    # Typical urban background 0-50; spikes up to 150+ possible
    pm1_0 = max(0, int(random.gauss(12, 8)))
    pm2_5 = max(pm1_0, int(pm1_0 + random.gauss(8, 6)))
    pm10  = max(pm2_5, int(pm2_5 + random.gauss(10, 8)))
    return {"PM1_0": pm1_0, "PM2_5": pm2_5, "PM10": pm10}

def gen_mq131():
    """Simulate ozone (O3) in ppm (very small)."""
    # 0.000–0.100 ppm typical; spikes can occur near sources
    o3 = max(0.0, round(random.uniform(0.000, 0.080) + random.uniform(0, 0.020), 3))
    return {"O3_ppm": o3}

def gen_mq4():
    """Simulate methane (CH4) in ppm."""
    # Background ~1.8 ppm; industrial areas can be higher
    base = random.uniform(1.6, 2.4)
    spike = random.choice([0, 0, 0, random.uniform(0.5, 5.0)])  # occasional spike
    ch4 = round(base + spike, 3)
    return {"CH4_ppm": ch4}

def payload_devices():
  # Data of the sensors
    payload = {
        "PMS5003": gen_pms5003(),
        "MQ131": gen_mq131(),
        "MQ4": gen_mq4(),
    } 
    return payload

devices=14
urls = make_urls(base, start=1, end=devices, width=3)


counter_time=1200 #time in seconds to wait until the next post 10min
counter_frec=100 #frecuency to iterate the loop until it ends
sent = 0

#Data posting
while True:
    for i in range(devices):
        payload=payload_devices()
        #print(urls[i])
        try:
            try:
                now_utc = datetime.now(timezone.utc) #Get time 
                response = requests.post(urls[i], json=payload)
                print(now_utc)  #Print time 
                print("Status code:", response.status_code) 
                print("Response body:", response.text)
            except requests.exceptions.RequestException as e:
                print(f"[ERR] Failed to POST: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting…") 
        time.sleep(10)
    time.sleep(counter_time)
 
##--------------------------------------------------------------------------------------------------
"""
AIRR-QUALITY SENSOR INTERPRETATION

Notes:
• PMS5003 returns instantaneous particulate concentration in µg/m³.
  Use averaging (e.g., moving window). AQI categories are based on 24-h
  averages (PM2.5/PM10) or NowCast methods.
• MQ131 (O₃) and MQ-4 (CH₄) output analog voltage → derive Rs and estimate ppm
  via calibration curves. Treat as indicative (not a regulatory analyzer).

PMS5003 — PARTICULATE MATTER (µg/m³, atmospheric block recommended)
PM2.5 (24-hour average)
┌──────────────┬──────────────────────────────┬───────────────────────────────────────────────┐
│ Range        │ Category                     │ Quick meaning / action                        │
├──────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ 0 – 12.0     │ Good                         │ Clean air; normal activity                    │
│ 12.1 – 35.4  │ Moderate                     │ Mild impact; sensitive groups watch symptoms  │
│ 35.5 – 55.4  │ Unhealthy for sensitive grp. │ Asthma/CVD risk ↑; reduce exposure            │
│ 55.5 – 150.4 │ Unhealthy                    │ General population impacted; avoid exertion   │
│ 150.5 – 250.4│ Very Unhealthy               │ High risk; stay indoors / purify air          │
│ > 250.4      │ Hazardous                    │ Severe risk; alerts / minimize exposure       │
└──────────────┴──────────────────────────────┴───────────────────────────────────────────────┘

PM10 (24-hour average)
┌──────────────┬──────────────────────────────┬───────────────────────────────────────────────┐
│ Range        │ Category                     │ Quick meaning / action                        │
├──────────────┼──────────────────────────────┼───────────────────────────────────────────────┤
│ 0 – 54       │ Good                         │ Clean air; normal activity                    │
│ 55 – 154     │ Moderate                     │ Mild impact; sensitive groups caution         │
│ 155 – 254    │ Unhealthy for sensitive grp. │ Reduce exposure                               │
│ 255 – 354    │ Unhealthy                    │ Avoid strenuous outdoor activity              │
│ 355 – 424    │ Very Unhealthy               │ Stay indoors                                  │
│ 425 – 604    │ Hazardous                    │ Severe risk                                   │
└──────────────┴──────────────────────────────┴───────────────────────────────────────────────┘
PM1.0: no official AQI categories; use as indicator of very fine aerosol.

MQ131 — OZONE (O₃)
Output: analog voltage → compute sensor resistance (Rs), normalize with R0,
then map Rs/R0 to ppm using the datasheet's log-log curve.
  Rs = RL * (Vcc − Vout) / Vout
  log10(ppm) = a * log10(Rs/R0) + b   # (fit ‘a’, ‘b’ from the curve)

Typical interpretation (8-hour average):
┌───────────────────┬──────────────┬──────────────────────────────┬────────────────────────────┐
│ O₃ (ppm, 8h avg)  │ ~µg/m³ (*)   │ Category                     │ Action                     │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────────┤
│ 0.000 – 0.054     │ 0 – 106      │ Good                         │ Normal activity            │
│ 0.055 – 0.070     │ 108 – 137    │ Moderate                     │ Sensitive: reduce exposure │
│ 0.071 – 0.085     │ 139 – 167    │ Unhealthy for sens. groups   │ Limit outdoor activity     │
│ 0.086 – 0.105     │ 169 – 206    │ Unhealthy                    │ Avoid strenuous outdoors   │
│ 0.106 – 0.200     │ 208 – 393    │ Very Unhealthy               │ Stay indoors               │
└───────────────────┴──────────────┴──────────────────────────────┴────────────────────────────┘
(*) Approx. conversion at 25°C, 1 atm:  µg/m³ ≈ ppm × 1963  (for O₃)

MQ-4 — METHANE (CH₄)
Output: analog voltage → compute Rs, normalize with R0, then use datasheet curve
to estimate ppm. For safety context, compare to %LEL (Lower Explosive Limit).
  %LEL ≈ (ppm / 50,000) × 100   # LEL of CH₄ ≈ 5% v/v = 50,000 ppm

┌──────────────┬───────────────┬──────────────────────────────────┬────────────────────────────┐
│ CH₄ (ppm)    │ %LEL (approx) │ Interpretation                   │ Recommended action         │
├──────────────┼───────────────┼──────────────────────────────────┼────────────────────────────┤
│ 1.6 – 3.0    │ < 0.01%       │ Ambient background               │ Normal                     │
│ 3 – 50       │ < 0.1%        │ Slightly elevated                │ Ventilate / inspect        │
│ 50 – 1,000   │ 0.1 – 2%      │ High (possible local leak)       │ Investigate & ventilate    │
│ 1,000 – 5,000│ 2 – 10%       │ Very high                        │ Evacuate / cut sources     │
│ 5,000 – 10,000│10 – 20%      │ Alarm zone (industrial)          │ Safety protocols           │
│ > 10,000     │ > 20%         │ Severe hazard / explosivity      │ Emergency                  │
└──────────────┴───────────────┴──────────────────────────────────┴────────────────────────────┘

"""

