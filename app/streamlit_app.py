
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="OceanEmbed Explorer",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 OceanEmbed Explorer")
st.caption("Prototype — Subsurface Ocean Temperature Reconstruction")

# Load model and scalers
model = tf.keras.models.load_model(
    "OceanEmbed_model/oceanembed_cnn.keras"
)

X_scaler = joblib.load(
    "OceanEmbed_model/X_patch_scaler.pkl"
)

Y_scaler = joblib.load(
    "OceanEmbed_model/Y_cnn_scaler.pkl"
)

depths = np.array([
    0, 5, 10, 20, 30, 50, 75,
    100, 125, 150, 200, 300,
    500, 700, 1000
])

st.sidebar.header("Ocean Location")

lat = st.sidebar.number_input(
    "Latitude (°N)",
    min_value=5.0,
    max_value=30.0,
    value=12.0
)

lon = st.sidebar.number_input(
    "Longitude (°E)",
    min_value=45.0,
    max_value=105.0,
    value=90.0
)

st.sidebar.info(
    "Prototype currently uses a 3×3 surface patch "
    "with 7 ocean surface variables."
)

st.subheader("Input Surface Variables")

cols = st.columns(4)

sst = cols[0].number_input("SST (°C)", value=28.0)
sss = cols[1].number_input("SSS (PSU)", value=34.5)
sla = cols[2].number_input("SLA (m)", value=0.0)
u_current = cols[3].number_input("U Current (m/s)", value=0.0)

cols2 = st.columns(3)

v_current = cols2[0].number_input("V Current (m/s)", value=0.0)
u_wind = cols2[1].number_input("U Wind (m/s)", value=0.0)
v_wind = cols2[2].number_input("V Wind (m/s)", value=0.0)

if st.button("🔍 Reconstruct Temperature Profile"):

    # Create center pixel
    center = np.array([
        sst,
        sss,
        sla,
        u_current,
        v_current,
        u_wind,
        v_wind
    ], dtype=np.float32)

    # Create 3×3 patch
    patch = np.tile(center, (3, 3, 1))

    # Scale
    patch_scaled = X_scaler.transform(
        patch.reshape(-1, 7)
    ).reshape(1, 3, 3, 7)

    # Prediction
    prediction_scaled = model.predict(
        patch_scaled,
        verbose=0
    )

    prediction = Y_scaler.inverse_transform(
        prediction_scaled
    )[0]

    st.success("Temperature profile reconstructed!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Surface Temperature",
            f"{prediction[0]:.2f} °C"
        )

        st.metric(
            "1000 m Temperature",
            f"{prediction[-1]:.2f} °C"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(5, 7))

        ax.plot(
            prediction,
            depths,
            marker="o"
        )

        ax.invert_yaxis()

        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(
            f"Temperature Profile\n"
            f"{lat:.2f}°N, {lon:.2f}°E"
        )

        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

    st.subheader("Reconstructed Temperature Profile")

    st.dataframe(
        {
            "Depth (m)": depths,
            "Temperature (°C)": np.round(
                prediction, 2
            )
        },
        use_container_width=True
    )

st.divider()

st.caption(
    "OceanEmbed PoC | GLORYS used as reference training target. "
    "Independent ARGO validation is planned for the full system."
)
