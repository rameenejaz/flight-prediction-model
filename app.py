"""Flight fare estimator: Streamlit front end for the model trained in the Colab notebook.
Author: Rameen Muzammil Ejaz (FA24-BSE-160)

Files needed next to this script (all produced by the notebook):
    flight_price_model.joblib   model_meta.json   requirements.txt
"""
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="Flight Fare Estimator", page_icon="✈️", layout="centered")

# ─────────────────────────────── Load model & metadata ───────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(HERE / "flight_price_model.joblib")

@st.cache_data
def load_meta():
    return json.loads((HERE / "model_meta.json").read_text())

try:
    model, meta = load_model(), load_meta()
except FileNotFoundError as e:
    st.error(f"Missing file: {Path(e.filename).name}. Upload flight_price_model.joblib and "
             "model_meta.json from the notebook's deployment_files.zip to the same folder as app.py.")
    st.stop()

CITY_CODES = {"Delhi": "DEL", "Mumbai": "BOM", "Bangalore": "BLR",
              "Kolkata": "CCU", "Hyderabad": "HYD", "Chennai": "MAA"}
STOP_LABELS = {"zero": "Non-stop", "one": "1 stop", "two_or_more": "2 or more stops"}

def nice(label: str) -> str:
    """'Air_India' -> 'Air India', 'Early_Morning' -> 'Early morning'."""
    words = label.split("_")
    if words[0].isupper() or len(words[0]) <= 2:      # keep 'GO FIRST', 'Air India' readable
        return " ".join(words)
    return " ".join([words[0]] + [w.lower() for w in words[1:]])

def inr(amount: float) -> str:
    """Indian digit grouping: 123456 -> 1,23,456."""
    s = f"{int(round(amount))}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail

def hours_text(h: float) -> str:
    return f"{int(h)}h {int(round((h % 1) * 60)):02d}m"

# ─────────────────────────────── Styles ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');

:root {
  --paper: #EEF2F5; --ink: #14213D; --muted: #5B6B7B; --rule: #C9D3DC;
  --signal: #FFB000; --tile: #1F2933; --tile-edge: #0E151C; --card: #FFFFFF;
}
html, body, [class*="css"], .stMarkdown, .stSelectbox, .stSlider, .stNumberInput, button {
  font-family: 'Barlow', 'Segoe UI', system-ui, sans-serif;
}
.block-container { padding-top: 2.5rem; max-width: 780px; }
.stApp { background-color: var(--paper); }

.masthead h1 {
  font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700;
  font-size: clamp(2.4rem, 7vw, 3.4rem); line-height: 1; letter-spacing: -0.01em;
  color: var(--ink); margin: 0 0 .5rem 0; padding: 0;
}
.masthead p { color: var(--muted); font-size: 1.05rem; max-width: 60ch; margin: 0 0 1.5rem 0; }

h3.section {
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.35rem;
  color: var(--ink); margin: .25rem 0 .25rem 0; padding: 0;
}

div[data-testid="stFormSubmitButton"] button {
  background: var(--signal); color: var(--ink); border: 0; font-weight: 600;
  font-size: 1.05rem; padding: .6rem 1.6rem; border-radius: 6px;
}
div[data-testid="stFormSubmitButton"] button:hover { background: #E89F00; color: var(--ink); }
div[data-testid="stFormSubmitButton"] button:focus-visible { outline: 3px solid var(--ink); outline-offset: 2px; }

/* ── Boarding pass ── */
.pass {
  display: grid; grid-template-columns: 1fr auto; margin: 1.75rem 0 1rem 0;
  background: var(--card); border-radius: 14px; overflow: hidden;
  border: 1px solid var(--rule); color: var(--ink);
}
.pass-main { padding: 1.4rem 1.6rem 1.3rem 1.6rem; }
.pass-airline { color: var(--muted); font-weight: 500; margin-bottom: .35rem; }
.route { display: flex; align-items: center; gap: 1rem; }
.code { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 3.6rem; line-height: .95; }
.city { color: var(--muted); font-size: .95rem; }
.path { flex: 1; border-top: 2px dashed var(--rule); position: relative; min-width: 2.5rem; }
.path span { position: absolute; top: -0.95rem; left: 50%; transform: translateX(-50%);
             background: var(--card); padding: 0 .2rem; font-size: 1.1rem; }
.details { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem 1.2rem;
           margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--rule); }
.details div { font-size: .82rem; color: var(--muted); }
.details b { display: block; font-size: 1.02rem; color: var(--ink); font-weight: 600; }

.stub {
  background: var(--ink); color: #DCE3EA; padding: 1.4rem 1.5rem; min-width: 15rem;
  display: flex; flex-direction: column; justify-content: center; position: relative;
  border-left: 2px dashed #46536A;
}
.stub::before, .stub::after { content: ""; position: absolute; left: -11px; width: 20px; height: 20px;
  background: var(--paper); border-radius: 50%; }
.stub::before { top: -10px; } .stub::after { bottom: -10px; }
.stub-label { font-size: .9rem; margin-bottom: .55rem; }
.flaps { display: flex; gap: 3px; flex-wrap: nowrap; }
.flap {
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 2.2rem; line-height: 1;
  color: var(--signal); background: var(--tile); border: 1px solid var(--tile-edge);
  border-radius: 4px; padding: .3rem .32rem .26rem .32rem; min-width: 1.45rem; text-align: center;
  position: relative; box-shadow: inset 0 -1px 0 rgba(255,255,255,.05);
}
.flap::after { content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: var(--tile-edge); }
.flap.sep { min-width: .7rem; padding-left: .12rem; padding-right: .12rem; color: #8A97A6; }
.stub-note { font-size: .82rem; margin-top: .7rem; color: #9FB0C2; }

@media (prefers-reduced-motion: no-preference) {
  .flap { animation: flip .5s cubic-bezier(.3,.7,.4,1) both; transform-origin: 50% 50%; }
  @keyframes flip { from { transform: rotateX(90deg); opacity: .2; } to { transform: rotateX(0); opacity: 1; } }
}
@media (max-width: 640px) {
  .pass { grid-template-columns: 1fr; }
  .stub { border-left: 0; border-top: 2px dashed #46536A; min-width: 0; }
  .stub::before, .stub::after { display: none; }
  .code { font-size: 2.8rem; }
  .details { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.foot { color: var(--muted); font-size: .85rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--rule); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────── Header ───────────────────────────────
best = next(r for r in meta["cv_results"] if r["model"] == meta["model_name"])
st.markdown(f"""
<div class="masthead">
  <h1>What should this flight cost?</h1>
  <p>Enter the details of a domestic flight between India's six biggest cities to get an estimated
  ticket price. The estimate comes from a {meta['model_name']} model trained on {meta['trained_on_rows']:,} real fares.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────── Form ───────────────────────────────
opts = meta["options"]
with st.form("flight"):
    st.markdown('<h3 class="section">Route and airline</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    source = c1.selectbox("From", opts["source_city"], index=opts["source_city"].index("Delhi")
                          if "Delhi" in opts["source_city"] else 0)
    destination = c2.selectbox("To", opts["destination_city"], index=opts["destination_city"].index("Mumbai")
                               if "Mumbai" in opts["destination_city"] else 1)
    c1, c2, c3 = st.columns(3)
    airline = c1.selectbox("Airline", opts["airline"], format_func=nice)
    travel_class = c2.selectbox("Class", meta["class_options"])
    stops = c3.selectbox("Stops", meta["stops_options"], index=1, format_func=STOP_LABELS.get)

    st.markdown('<h3 class="section">Timing</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    departure = c1.selectbox("Departure time", opts["departure_time"], format_func=nice)
    arrival = c2.selectbox("Arrival time", opts["arrival_time"], format_func=nice)
    d_min, d_max, d_med = meta["duration_range"]
    duration = st.slider("Total journey time (hours)", float(round(d_min, 1)), float(round(d_max)),
                         float(round(d_med * 4) / 4), step=0.25)
    k_min, k_max = meta["days_left_range"]
    days_left = st.slider("Days between booking and travel", k_min, k_max, 15)

    submitted = st.form_submit_button("Estimate fare")

# ─────────────────────────────── Prediction ───────────────────────────────
if submitted:
    if source == destination:
        st.error("Departure and arrival cities are the same. Choose a different destination.")
        st.stop()
    if travel_class not in meta["airline_classes"].get(airline, [travel_class]):
        st.warning(f"{nice(airline)} only sells {', '.join(meta['airline_classes'][airline])} class in the "
                   f"training data, so this {travel_class.lower()} estimate is less reliable.")

    row = pd.DataFrame([{
        "stops": stops, "class": travel_class, "airline": airline, "source_city": source,
        "departure_time": departure, "arrival_time": arrival, "destination_city": destination,
        "duration": duration, "days_left": days_left,
    }])[meta["features"]]
    price = max(0.0, float(model.predict(row)[0]))

    fare = inr(price)
    tiles = '<div class="flap sep">₹</div>' + "".join(
        f'<div class="flap{" sep" if ch == "," else ""}" style="animation-delay:{i*45}ms">{ch}</div>'
        for i, ch in enumerate(fare))

    st.markdown(f"""
<div class="pass" role="region" aria-label="Estimated fare: {fare} rupees">
  <div class="pass-main">
    <div class="pass-airline">{nice(airline)}, {travel_class.lower()} class</div>
    <div class="route">
      <div><div class="code">{CITY_CODES.get(source, source[:3].upper())}</div><div class="city">{source}</div></div>
      <div class="path" aria-hidden="true"><span>✈</span></div>
      <div><div class="code">{CITY_CODES.get(destination, destination[:3].upper())}</div><div class="city">{destination}</div></div>
    </div>
    <div class="details">
      <div>Departs<b>{nice(departure)}</b></div>
      <div>Arrives<b>{nice(arrival)}</b></div>
      <div>Stops<b>{STOP_LABELS[stops]}</b></div>
      <div>Journey<b>{hours_text(duration)}</b></div>
      <div>Booked<b>{days_left} day{'s' if days_left != 1 else ''} ahead</b></div>
      <div>Model<b>{meta['model_name']}</b></div>
    </div>
  </div>
  <div class="stub">
    <div class="stub-label">Estimated fare</div>
    <div class="flaps" aria-hidden="true">{tiles}</div>
    <div class="stub-note">Typical error ± ₹{inr(best['mae'])}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────── Model comparison ───────────────────────────────
with st.expander("How this model was chosen"):
    st.write(f"Five regression algorithms were compared with {meta['n_folds']}-fold cross-validation on the "
             f"full dataset of {meta['trained_on_rows']:,} flights. Lower RMSE and MAE are better.")
    table = pd.DataFrame(meta["cv_results"]).rename(columns={
        "rank": "Rank", "model": "Model", "rmse": "RMSE (₹)", "rmse_std": "RMSE std (₹)",
        "mae": "MAE (₹)", "r2": "R²", "train_s": "Training time, 10 folds (s)"})
    table = table[["Rank", "Model", "RMSE (₹)", "RMSE std (₹)", "MAE (₹)", "R²", "Training time, 10 folds (s)"]]
    st.dataframe(
        table.style.format({"RMSE (₹)": "{:,.0f}", "RMSE std (₹)": "{:,.0f}", "MAE (₹)": "{:,.0f}",
                            "R²": "{:.4f}", "Training time, 10 folds (s)": "{:,.1f}"}),
        hide_index=True, width="stretch")
    chart = (alt.Chart(table).mark_bar(color="#14213D")
             .encode(x=alt.X("RMSE (₹):Q", title="Mean RMSE (₹), lower is better"),
                     y=alt.Y("Model:N", sort=alt.EncodingSortField("Rank"), title=None),
                     tooltip=["Model", alt.Tooltip("RMSE (₹):Q", format=",.0f")])
             .properties(height=200))
    st.altair_chart(chart, width="stretch")
    if meta["model_name"] != meta["best_cv_model"]:
        st.caption(f"{meta['best_cv_model']} scored best but its model file is too large for free hosting, "
                   f"so the next-best model, {meta['model_name']}, is deployed.")

st.markdown(f"""
<div class="foot">Built by <b>Rameen Muzammil Ejaz</b> (FA24-BSE-160) for Machine Learning Fundamentals (AIC354),
COMSATS University Islamabad, Lahore Campus. Instructor: Dr. Rao Muhammad Adeel Nawab.<br>
Data: Kaggle Flight Price Prediction (EaseMyTrip). Model trained {meta['trained_at']}.
Estimates are for learning purposes, not real quotes.</div>
""", unsafe_allow_html=True)
