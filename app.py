
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Battery Intelligence",
    page_icon="??",
    layout="wide"
)

data = pd.read_csv("results/all_battery_cycles.csv")
predictions = pd.read_csv("results/model_predictions.csv")
metrics = pd.read_csv("results/model_metrics.csv")

st.title("AI-Driven Battery Intelligence")
st.caption(
    "Predictive Health | Remaining Useful Life | "
    "Laboratory Data Monitoring"
)

st.info(
    "Research prototype using NASA battery cycling data. "
    "Predictions are experimental and are not battery safety guarantees."
)

st.sidebar.header("Battery Selection")
cell = st.sidebar.selectbox(
    "Choose battery",
    sorted(data["cell_id"].unique())
)

cell_data = data[data["cell_id"] == cell].sort_values("cycle")
latest = cell_data.iloc[-1]

soh_pred = predictions[
    (predictions["cell_id"] == cell) &
    (predictions["target"] == "soh_percent")
].sort_values("cycle")

rul_pred = predictions[
    (predictions["cell_id"] == cell) &
    (predictions["target"] == "rul_cycles")
].sort_values("cycle")

st.header("Battery Health Overview")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Latest Measured SOH",
    f"{latest['soh_percent']:.2f}%"
)

c2.metric(
    "Latest Capacity",
    f"{latest['capacity_ah']:.3f} Ah"
)

if not rul_pred.empty:
    c3.metric(
        "Predicted RUL at Last Cycle",
        f"{rul_pred.iloc[-1]['predicted']:.1f} cycles"
    )
else:
    c3.metric("Predicted RUL", "Unavailable")

st.divider()
st.header("Battery Health Trend")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    cell_data["cycle"],
    cell_data["soh_percent"],
    label="Measured SOH"
)

if not soh_pred.empty:
    ax.plot(
        soh_pred["cycle"],
        soh_pred["predicted"],
        label="Held-out Model Prediction"
    )

ax.axhline(
    80,
    linestyle="--",
    label="Project EOL Threshold (80%)"
)

ax.set_xlabel("Discharge Cycle")
ax.set_ylabel("SOH (%)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

st.header("Remaining Useful Life")

if not rul_pred.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        rul_pred["cycle"],
        rul_pred["actual"],
        label="Reference RUL"
    )
    ax.plot(
        rul_pred["cycle"],
        rul_pred["predicted"],
        label="Model Prediction"
    )
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Remaining Cycles")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.header("Compare All Battery Cells")

summary = (
    data.sort_values("cycle")
    .groupby("cell_id")
    .tail(1)[
        ["cell_id", "cycle", "capacity_ah", "soh_percent"]
    ]
    .sort_values("cell_id")
    .reset_index(drop=True)
)

st.dataframe(summary, use_container_width=True)

st.bar_chart(
    summary.set_index("cell_id")["soh_percent"],
    y_label="SOH (%)"
)

st.divider()
st.header("Model Evaluation")

selected_metrics = metrics[metrics["test_cell"] == cell]

st.dataframe(
    selected_metrics[
        ["target", "MAE", "RMSE", "R2"]
    ].round(3),
    use_container_width=True
)

st.divider()
st.header("Measurement Anomaly Screening")

st.write(
    "This section flags statistical outliers for investigation. "
    "It is not a thermal-runaway predictor or safety alarm."
)

for column in [
    "max_temperature",
    "mean_voltage",
    "mean_current"
]:
    values = cell_data[column]
    mean_value = values.mean()
    std_value = values.std()

    if pd.notna(std_value) and std_value > 0:
        unusual = cell_data[
            (cell_data[column] - mean_value).abs()
            > 3 * std_value
        ]

        st.write(
            f"**{column}:** {len(unusual)} unusual records "
            "using the three-standard-deviation rule"
        )

st.header("Battery Cycle Records")
st.dataframe(cell_data, use_container_width=True)

st.caption(
    "SOH uses the 2.0 Ah rated capacity. "
    "EOL uses the project's 80% SOH threshold. "
    "Laboratory results may not generalize to other batteries."
)
