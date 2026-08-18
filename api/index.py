import os
import re
import requests
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template

app = Flask(__name__, template_folder="../api/templates")

# ==========================================
# 🛑 AWC SECURITY HEADERS
# ==========================================
# This User-Agent string prevents AWC from blocking the Vercel server.
HEADERS = {
    "User-Agent": "ZUA-CE-PIREP-Dashboard/1.0 (contact: connorengland9@gmail.com)",
    "Accept": "application/json"
}

# A station's most recent METAR older than this is flagged offline/stale
# rather than shown as current, even though AWC still returns it within our
# fetch window (see STALE_MINUTES usage in build_airport_entry).
STALE_MINUTES = 120

MAIN_AIRPORTS = {
    "PGUM": "Agana Airport",
    "PGUA": "Andersen AFB",
    "PGSN": "Saipan Airport"
}

AUX_AIRPORTS = {
    "PGRO": "Rota",
    "PGWT": "Tinian"
}


def get_cloud_base(layer):
    base = layer.get('base')
    try:
        if base is not None:
            return int(base)
    except (ValueError, TypeError):
        pass
    return None


def metar_body(raw_text):
    """Everything before RMK -- avoids matching TS/GR inside remarks."""
    if not raw_text:
        return ""
    return re.split(r'\bRMK\b', raw_text.upper())[0]


# A weather-phenomena group is built entirely from these 2-letter codes,
# optionally prefixed with intensity (+/-) or VC, and one descriptor --
# e.g. "+TSGR" (heavy thunderstorm hail). Matching the whole token against
# this grammar (rather than scanning raw text for bare substrings) is what
# keeps a station ID like "PGRO" from being mistaken for hail ("GR").
_WX_DESCRIPTORS = "MI|PR|BC|DR|BL|SH|TS|FZ"
_WX_PHENOMENA = "DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS"
WX_TOKEN_RE = re.compile(rf'^(?:[-+]|VC)?(?:{_WX_DESCRIPTORS})?(?:{_WX_PHENOMENA})+$')


def extract_wx_tokens(raw_text):
    """Pull only genuine weather-phenomena groups out of a raw METAR body,
    ignoring the station ID / wind / cloud groups that can coincidentally
    contain the same 2-letter codes (e.g. "PGRO" contains "GR")."""
    return " ".join(t for t in metar_body(raw_text).split() if WX_TOKEN_RE.match(t))


def check_pirep_condition(obs):
    """Structured-field hazard check (ceiling/visibility/wx codes from AWC's
    clouds/visib/wxString fields) -- far more reliable than scanning the raw
    METAR text for bare substrings, which false-positives on station IDs."""
    conditions = []

    ceiling_layers = []
    for layer in obs.get('clouds') or []:
        cover = layer.get('cover') or ''
        base = get_cloud_base(layer)
        if cover in ('BKN', 'OVC', 'VV') and base is not None and base <= 5000:
            ceiling_layers.append(base)
    if ceiling_layers:
        conditions.append(f"CIG {min(ceiling_layers)}FT")

    vis = obs.get('visib')
    if vis is not None:
        try:
            val = float(vis.replace('+', '')) if isinstance(vis, str) and '+' in vis else float(vis)
            if val <= 5.0:
                v_str = vis if isinstance(vis, str) else str(val)
                conditions.append(f"VIS {v_str}SM")
        except (ValueError, TypeError):
            pass

    wx = (obs.get('wxString') or "").upper()
    if not wx:
        wx = extract_wx_tokens(obs.get('rawOb') or "")

    if 'TS' in wx: conditions.append("THUNDERSTORM")
    if 'VA' in wx: conditions.append("VOLCANIC ASH")
    if 'FC' in wx: conditions.append("FUNNEL CLOUD")
    if 'GR' in wx: conditions.append("HAIL")
    if 'WS' in wx: conditions.append("WIND SHEAR")
    if '+RA' in wx: conditions.append("HEAVY RAIN")

    if conditions:
        return True, " / ".join(sorted(set(conditions)))
    return False, "PIREP NOT REQUIRED"


def check_ifr_status(obs):
    for layer in obs.get('clouds') or []:
        cover = layer.get('cover') or ''
        base = get_cloud_base(layer)
        if cover in ('BKN', 'OVC', 'VV') and base is not None and base < 1000:
            return True

    vis = obs.get('visib')
    if vis is not None:
        try:
            val = float(vis.replace('+', '')) if isinstance(vis, str) and '+' in vis else float(vis)
            if val < 3.0:
                return True
        except (ValueError, TypeError):
            pass
    return False


def obs_age_minutes(obs):
    """Minutes since this METAR's observation time; None if unknown."""
    obs_time = obs.get('obsTime')
    if not obs_time:
        return None
    try:
        obs_dt = datetime.fromtimestamp(obs_time, timezone.utc)
    except (ValueError, OSError, TypeError):
        return None
    return (datetime.now(timezone.utc) - obs_dt).total_seconds() / 60


def build_airport_entry(apt_id, name, obs):
    if not obs:
        return {
            "id": apt_id, "name": name, "raw": "OFFLINE", "time": "",
            "pirep_needed": False, "reason": "OFFLINE", "status": "offline", "is_ifr": False
        }

    age = obs_age_minutes(obs)
    if age is not None and age > STALE_MINUTES:
        return {
            "id": apt_id, "name": name,
            "raw": obs.get("rawOb", ""), "time": obs.get("reportTime", ""),
            "pirep_needed": False, "reason": f"STALE DATA ({int(age)}MIN OLD)",
            "status": "offline", "is_ifr": False
        }

    pirep_needed, reason = check_pirep_condition(obs)
    is_ifr = check_ifr_status(obs)
    if is_ifr:
        pirep_needed = True
        if "IFR CONDITIONS" not in reason:
            reason = "IFR CONDITIONS" if reason == "PIREP NOT REQUIRED" else f"IFR CONDITIONS • {reason}"

    return {
        "id": apt_id, "name": name,
        "raw": obs.get("rawOb", ""), "time": obs.get("reportTime", ""),
        "pirep_needed": pirep_needed, "reason": reason,
        "status": "online", "is_ifr": is_ifr
    }


@app.route("/")
def index():
    # Serves the HTML file located in api/templates/CE_PIREP_INDEX.html
    return render_template("CE_PIREP_INDEX.html")


@app.route("/api/data")
def get_data():
    try:
        # 1. FETCH METARS
        # hours=3 gives us enough lookback to bridge brief reporting gaps,
        # but it also means AWC can return more than one historical record
        # per station -- the freshest-record selection below (by obsTime) is
        # what actually decides what's "current", not this window alone.
        metar_res = requests.get(
            "https://aviationweather.gov/api/data/metar",
            params={"ids": "PGUM,PGUA,PGSN,PGRO,PGWT", "format": "json", "taf": "false", "hours": 3},
            headers=HEADERS, timeout=10
        )
        metar_data = metar_res.json() if metar_res.status_code == 200 else []

        # 2. FETCH PIREPS
        # Bounding box covering the Marianas (Lat 10 to 20, Lon 140 to 150)
        pirep_res = requests.get(
            "https://aviationweather.gov/api/data/pirep",
            params={"bbox": "10,140,20,150", "age": 2, "format": "json"},
            headers=HEADERS, timeout=10
        )
        pirep_data = pirep_res.json() if pirep_res.status_code == 200 else []

        # Convert METAR list into a dictionary for easy lookup by ICAO ID,
        # keeping only the freshest record per station by obsTime -- a plain
        # dict comprehension would keep whichever record AWC happened to
        # list last, which is not guaranteed to be the most recent one.
        metars_dict = {}
        if isinstance(metar_data, list):
            for obs in metar_data:
                icao = obs.get("icaoId")
                if not icao:
                    continue
                existing = metars_dict.get(icao)
                if existing is None or (obs.get("obsTime") or 0) > (existing.get("obsTime") or 0):
                    metars_dict[icao] = obs

        # 3. BUILD MAIN AIRPORTS DATA
        main_metars_list = [
            build_airport_entry(apt_id, name, metars_dict.get(apt_id))
            for apt_id, name in MAIN_AIRPORTS.items()
        ]

        # 4. BUILD AUX AIRPORTS DATA
        aux_metars_list = []
        for apt_id, name in AUX_AIRPORTS.items():
            entry = build_airport_entry(apt_id, name, metars_dict.get(apt_id))
            if entry["status"] == "offline":
                entry["reason"] = "NOT PIREP-ABLE"
            aux_metars_list.append(entry)

        # 5. BUILD PIREPS DATA
        # Real /api/data/pirep fields: rawOb, receiptTime, acType, fltLvl,
        # fltLvlType, pirepType. There is no "acft" or "urgent" field -- the
        # old code guessed those and they never matched anything, so every
        # PIREP silently showed acft="UNKN" and type="UA" regardless of
        # what AWC actually reported.
        pireps_list = []
        if isinstance(pirep_data, list):
            for p in pirep_data:
                raw = p.get("rawOb") or ""
                if not raw:
                    continue

                fl_val = p.get("fltLvl")
                fl_type = (p.get("fltLvlType") or "").upper()
                if fl_val:
                    fl = f"FL{int(fl_val):03d}"
                elif fl_type == "DURC":
                    fl = "DURING CLIMB"
                elif fl_type == "DURD":
                    fl = "DURING DESCENT"
                else:
                    fl = "UNK"

                p_type = "UUA" if "urgent" in (p.get("pirepType") or "").lower() else "UA"

                pireps_list.append({
                    "type": p_type,
                    "time": p.get("receiptTime") or "",
                    "raw": raw,
                    "acft": p.get("acType") or "UNK",
                    "fl": fl
                })

        return jsonify({
            "metars": main_metars_list,
            "aux_metars": aux_metars_list,
            "pireps": pireps_list
        })

    except Exception as e:
        print(f"Server Error fetching data: {e}")
        # Safe fallback so the frontend doesn't crash, triggers "RETRYING CONNECTION..."
        return jsonify({
            "metars": [
                { "id": "PGUM", "name": "Agana Airport", "raw": "RETRYING CONNECTION...", "time": "", "pirep_needed": False, "reason": "OFFLINE", "status": "offline", "is_ifr": False },
                { "id": "PGUA", "name": "Andersen AFB", "raw": "RETRYING CONNECTION...", "time": "", "pirep_needed": False, "reason": "OFFLINE", "status": "offline", "is_ifr": False },
                { "id": "PGSN", "name": "Saipan Airport", "raw": "RETRYING CONNECTION...", "time": "", "pirep_needed": False, "reason": "OFFLINE", "status": "offline", "is_ifr": False }
            ],
            "aux_metars": [
                { "id": "PGRO", "name": "Rota", "raw": "RETRYING CONNECTION...", "time": "", "pirep_needed": False, "reason": "OFFLINE", "status": "offline", "is_ifr": False },
                { "id": "PGWT", "name": "Tinian", "raw": "RETRYING CONNECTION...", "time": "", "pirep_needed": False, "reason": "OFFLINE", "status": "offline", "is_ifr": False }
            ],
            "pireps": []
        })

# SIGMETs relevant to a ZUA (Guam CERAP) controller: Guam's own FIR plus the
# adjoining Oakland Oceanic FIR. Everything else on the international feed
# (Auckland, Tahiti, Vancouver, etc.) is too far away to be operationally
# useful here, per the actual scope of this tool.
RELEVANT_FIR_IDS = {"PGZU", "KZAK"}


@app.route("/api/sigmets")
def get_sigmets():
    try:
        res = requests.get(
            "https://aviationweather.gov/api/data/isigmet",
            params={"format": "json"},
            headers=HEADERS, timeout=10
        )
        data = res.json() if res.status_code == 200 else []
        if not isinstance(data, list):
            data = []

        features = []
        for s in data:
            if s.get("firId") not in RELEVANT_FIR_IDS:
                continue
            coords = s.get("coords") or []
            if len(coords) < 3:
                continue

            ring = [[c["lon"], c["lat"]] for c in coords]
            if ring[0] != ring[-1]:
                ring.append(ring[0])

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "firId": s.get("firId"),
                    "firName": s.get("firName"),
                    "hazard": s.get("hazard"),
                    "qualifier": s.get("qualifier"),
                    "base": s.get("base"),
                    "top": s.get("top"),
                    "validTimeFrom": s.get("validTimeFrom"),
                    "validTimeTo": s.get("validTimeTo"),
                    "rawSigmet": s.get("rawSigmet"),
                }
            })

        return jsonify({"type": "FeatureCollection", "features": features})

    except Exception as e:
        print(f"Server Error fetching sigmets: {e}")
        return jsonify({"type": "FeatureCollection", "features": []})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=True)
