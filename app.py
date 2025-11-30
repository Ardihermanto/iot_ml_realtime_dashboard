# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import queue
import threading
import plotly.graph_objs as go
from datetime import datetime, timezone, timedelta
import paho.mqtt.client as mqtt
from streamlit_autorefresh import st_autorefresh


# ==============================
# CONFIG
# ==============================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/class/session5/sensor"
TOPIC_OUTPUT = "iot/class/session5/output"
MODEL_PATH = "iot_temp_model.pkl"

TZ = timezone(timedelta(hours=7))   # GMT+7


# ==============================
# Session State Init
# ==============================
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = queue.Queue()

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_started" not in st.session_state:
    st.session_state.mqtt_started = False


# ==============================
# Load ML Model Safely
# ==============================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Model could not be loaded: {e}")
        return None

model = load_model(MODEL_PATH)


# ==============================
# MQTT Thread (Safe for Streamlit Cloud)
# ==============================
def start_mqtt():

    def _on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(TOPIC_SENSOR)
        st.session_state.msg_queue.put({"type": "status", "connected": rc == 0})

    def _on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
        except:
            return

        st.session_state.msg_queue.put({
            "type": "sensor",
            "temp": float(data.get("temp")),
            "hum": float(data.get("hum")),
            "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
        })

    def worker():
        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message

        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            st.session_state.msg_queue.put({"type": "error", "msg": str(e)})
            time.sleep(3)

    threading.Thread(target=worker, daemon=True).start()


if not st.session_state.mqtt_started:
    start_mqtt()
    st.session_state.mqtt_started = True
    time.sleep(0.2)


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")
st.title("🔥 IoT ML Realtime Dashboard — Streamlit Cloud Version")

st_autorefresh(interval=2000, key="refresh")  # refresh every 2 sec


left, right = st.columns([1, 2])

# ==============================
# Process Incoming Messages
# ==============================
while not st.session_state.msg_queue.empty():
    msg = st.session_state.msg_queue.get()

    if msg["type"] == "sensor":
        row = {
            "ts": msg["ts"],
            "temp": msg["temp"],
            "hum": msg["hum"],
        }

        # ML Prediction
        if model is not None:
            X = [[row["temp"], row["hum"]]]
            try:
                row["pred"] = model.predict(X)[0]
            except:
                row["pred"] = "ERR"

            try:
                row["conf"] = float(np.max(model.predict_proba(X)))
            except:
                row["conf"] = None
        else:
            row["pred"] = "N/A"
            row["conf"] = None

        st.session_state.last = row
        st.session_state.logs.append(row)

    elif msg["type"] == "status":
        st.session_state.mqtt_status = msg["connected"]


# ==============================
# LEFT PANEL
# ==============================
with left:
    st.subheader("Connection Status")

    connected = st.session_state.get("mqtt_status", False)
    st.metric("MQTT Connected", "Yes" if connected else "No")
    st.write("Broker:", MQTT_BROKER)

    st.markdown("### Last Reading")
    if st.session_state.last:
        st.write(st.session_state.last)
    else:
        st.info("Waiting for data...")

    st.markdown("### Manual Output")
    if st.button("Send ALERT_ON"):
        pub = mqtt.Client()
        pub.connect(MQTT_BROKER, MQTT_PORT)
        pub.publish(TOPIC_OUTPUT, "ALERT_ON")
        pub.disconnect()
        st.success("Sent ALERT_ON")

    if st.button("Send ALERT_OFF"):
        pub = mqtt.Client()
        pub.connect(MQTT_BROKER, MQTT_PORT)
        pub.publish(TOPIC_OUTPUT, "ALERT_OFF")
        pub.disconnect()
        st.success("Sent ALERT_OFF")

    st.markdown("### Download Logs")
    if st.button("Download CSV"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button("Download file", df.to_csv(index=False), file_name="logs.csv")


# ==============================
# RIGHT PANEL — CHART
# ==============================
with right:
    st.subheader("Live Chart (last 200 points)")

    df = pd.DataFrame(st.session_state.logs[-200:])

    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines+markers", name="Temperature"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines+markers", name="Humidity"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

    st.subheader("Recent Logs")
    st.dataframe(df[::-1].head(50))
