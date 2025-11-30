# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# Try to import optional autorefresh helper (not required)
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ---------------------------
# Config (edit if needed)
# ---------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/class/session5/sensor"
TOPIC_OUTPUT = "iot/class/session5/output"
MODEL_PATH = "iot_temp_model.pkl"   # put your model file in the repo root

# Timezone GMT+7 helper
TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

# UI constants
MAX_POINTS = 200

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="IoT ML Realtime Dashboard — ML Enhanced", layout="wide")
st.title("🔥 IoT ML Realtime Dashboard — ML Enhanced Version")

# ---------------------------
# Session state init
# ---------------------------
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = queue.Queue()      # where MQTT thread pushes raw messages
if "logs" not in st.session_state:
    st.session_state.logs = []                      # list of dict rows
if "last" not in st.session_state:
    st.session_state.last = None
if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False
if "ml_model" not in st.session_state:
    st.session_state.ml_model = None
if "anomaly_model" not in st.session_state:
    st.session_state.anomaly_model = None

# ---------------------------
# Load ML model (safe)
# ---------------------------
@st.cache_resource
def load_ml_model(path):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        # don't crash the app if model missing
        st.warning(f"Could not load ML model from {path}: {e}")
        return None

st.session_state.ml_model = load_ml_model(MODEL_PATH)

# optional anomaly model
@st.cache_resource
def try_load_anomaly():
    try:
        m = joblib.load("anomaly_model.pkl")
        return m
    except Exception:
        return None

st.session_state.anomaly_model = try_load_anomaly()

# ---------------------------
# MQTT callbacks (push to queue only)
# ---------------------------
def _on_connect(client, userdata, flags, rc):
    # subscribe when connected
    try:
        client.subscribe(TOPIC_SENSOR)
    except Exception:
        pass
    # post status to queue
    st.session_state.msg_queue.put({"_type": "status", "connected": rc == 0})

def _on_message(client, userdata, msg):
    payload = None
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        st.session_state.msg_queue.put({"_type": "sensor", "data": data, "ts": time.time()})
    except Exception:
        # push raw string if json parse fails
        st.session_state.msg_queue.put({"_type": "raw", "payload": payload, "ts": time.time()})

# ---------------------------
# Start MQTT background thread (safe)
# ---------------------------
def start_mqtt_thread():
    def worker():
        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                st.session_state.msg_queue.put({"_type": "error", "msg": f"MQTT connect error: {e}"})
                time.sleep(5)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    st.session_state.mqtt_thread_started = True

if not st.session_state.mqtt_thread_started:
    start_mqtt_thread()
    time.sleep(0.1)  # small pause so first status may arrive

# ---------------------------
# ML helpers
# ---------------------------
def model_predict(row):
    """Return (pred_label, prob_max_or_None)"""
    model = st.session_state.ml_model
    if model is None:
        return ("N/A", None)
    try:
        X = [[row["temp"], row["hum"]]]
        y_pred = model.predict(X)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(np.max(model.predict_proba(X)))
            except Exception:
                prob = None
        return (y_pred, prob)
    except Exception:
        return ("ERR", None)

def detect_anomaly_isolation(df_window, row):
    mdl = st.session_state.anomaly_model
    if mdl is not None:
        try:
            X = np.array([[row["temp"], row["hum"]]])
            is_anom = mdl.predict(X)[0] == -1
            score = float(mdl.decision_function(X)[0])
            return (bool(is_anom), score)
        except Exception:
            pass
    # fallback: simple z-score on temp
    try:
        temps = df_window["temp"].values
        if temps.size < 10:
            return (False, 0.0)
        mean = np.mean(temps)
        std = np.std(temps)
        if std == 0:
            return (False, 0.0)
        z = (row["temp"] - mean) / std
        return (abs(z) > 2.5, float(z))
    except Exception:
        return (False, 0.0)

# ---------------------------
# Process queue (drain and update logs)
# ---------------------------
def process_queue_once():
    processed = False
    while not st.session_state.msg_queue.empty():
        msg = st.session_state.msg_queue.get()
        processed = True
        try:
            if msg.get("_type") == "status":
                # use status to show connected or not (we update in UI below)
                st.session_state.last = st.session_state.last or {}
                st.session_state.last["_mqtt_connected"] = msg.get("connected", False)
            elif msg.get("_type") == "sensor":
                d = msg.get("data", {})
                try:
                    temp = float(d.get("temp"))
                    hum = float(d.get("hum"))
                except Exception:
                    continue
                row = {
                    "ts": datetime.fromtimestamp(msg.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S"),
                    "temp": temp,
                    "hum": hum
                }
                # prediction
                pred_label, conf = model_predict(row)
                row["pred"] = pred_label
                row["pred_conf"] = conf
                # anomaly detection using last window
                df_window = pd.DataFrame(st.session_state.logs[-50:]) if st.session_state.logs else pd.DataFrame(columns=["temp","hum"])
                is_anom, score = detect_anomaly_isolation(df_window, row)
                row["anomaly"] = is_anom
                row["anomaly_score"] = score
                st.session_state.last = row
                st.session_state.logs.append(row)
                # bound logs size
                if len(st.session_state.logs) > 5000:
                    st.session_state.logs = st.session_state.logs[-5000:]
                # auto publish back to device
                if row["pred"] == "Panas":
                    try:
                        pubc = mqtt.Client()
                        pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
                        pubc.publish(TOPIC_OUTPUT, "ALERT_ON")
                        pubc.disconnect()
                    except Exception:
                        pass
                else:
                    try:
                        pubc = mqtt.Client()
                        pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
                        pubc.publish(TOPIC_OUTPUT, "ALERT_OFF")
                        pubc.disconnect()
                    except Exception:
                        pass
            elif msg.get("_type") == "raw":
                st.session_state.logs.append({"ts": now_str(), "raw": msg.get("payload")})
            elif msg.get("_type") == "error":
                st.error(msg.get("msg"))
        except Exception as e:
            print("Error processing queue item:", e)
    return processed

# If autorefresh package present, use it to update UI every 2s
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, limit=None, key="auto_refresh")

# ---------------------------
# UI layout
# ---------------------------
left, right = st.columns([1, 2])

with left:
    st.header("Connection Status")
    st.write("Broker:", f"{MQTT_BROKER}:{MQTT_PORT}")
    connected_label = "Yes" if (st.session_state.last and st.session_state.last.get("_mqtt_connected")) or len(st.session_state.logs) > 0 else "No"
    st.metric("MQTT Connected", connected_label)
    st.write("Topic:", TOPIC_SENSOR)
    st.markdown("---")

    st.markdown("### Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"**Time (local GMT+7):** {last.get('ts', now_str())}")
        st.write(f"**Temp:** {last.get('temp')}")
        st.write(f"**Hum:** {last.get('hum')}")
        st.write(f"**Prediction:** {last.get('pred')}")
        st.write(f"**Confidence:** {last.get('pred_conf')}")
        st.write(f"**Anomaly:** {last.get('anomaly')} (score: {last.get('anomaly_score')})")
    else:
        st.info("Waiting for data...")

    st.markdown("### Manual Output Control")
    col1, col2 = st.columns(2)
    if col1.button("Send ALERT_ON"):
        try:
            pubc = mqtt.Client()
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_OUTPUT, "ALERT_ON")
            pubc.disconnect()
            st.success("Published ALERT_ON")
        except Exception as e:
            st.error(f"Publish failed: {e}")
    if col2.button("Send ALERT_OFF"):
        try:
            pubc = mqtt.Client()
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_OUTPUT, "ALERT_OFF")
            pubc.disconnect()
            st.success("Published ALERT_OFF")
        except Exception as e:
            st.error(f"Publish failed: {e}")

    st.markdown("---")
    st.markdown("### Download Logs")
    if st.button("Download CSV"):
        if st.session_state.logs:
            df_dl = pd.DataFrame(st.session_state.logs)
            csv = df_dl.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV file", data=csv, file_name=f"iot_logs_{int(time.time())}.csv")
        else:
            st.info("No logs to download")

with right:
    st.header(f"Live Chart (last {MAX_POINTS} points)")
    # process queue first so UI shows fresh
    process_queue_once()

    df_plot = pd.DataFrame(st.session_state.logs[-MAX_POINTS:])
    if not df_plot.empty and {"temp", "hum"}.issubset(df_plot.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], mode="lines+markers", name="Temp (°C)"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="Hum (%)", yaxis="y2"))
        # secondary axis
        fig.update_layout(
            yaxis=dict(title="Temp (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right", showgrid=False),
            height=500
        )
        # color markers by anomaly / label
        colors = []
        for _, r in df_plot.iterrows():
            if r.get("anomaly"):
                colors.append("magenta")
            else:
                lab = r.get("pred", "")
                if lab == "Panas":
                    colors.append("red")
                elif lab == "Normal":
                    colors.append("green")
                elif lab == "Dingin":
                    colors.append("blue")
                else:
                    colors.append("gray")
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to the topic:", TOPIC_SENSOR)

    st.markdown("### Recent Logs")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs)[::-1].head(50))
    else:
        st.write("—")

# always process queue at least once at the end (if autorefresh isn't used)
process_queue_once()
