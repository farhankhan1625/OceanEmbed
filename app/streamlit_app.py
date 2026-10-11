import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import xarray as xr

st.set_page_config(page_title="OceanEmbed Explorer", page_icon="🌊", layout="wide")

DEPTHS_FALLBACK = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)
CHANNELS = ["sst", "sss", "sla", "uo", "vo", "uwind", "vwind"]


class OceanEmbedResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return self.relu(z + identity)


class OceanEmbedModelExact(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 32, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = OceanEmbedResidualBlock(128)
        self.res2 = OceanEmbedResidualBlock(128)
        self.temporal_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.temporal_norm = nn.LayerNorm(128)
        self.temporal_ffn = nn.Sequential(nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Linear(256, 128))
        self.embedding_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128))
        self.embedding_to_spatial = nn.Conv2d(128, 128, kernel_size=1)
        self.decoder1 = nn.Conv2d(256, 128, 3, padding=1)
        self.decoder2 = nn.Conv2d(128, 64, 3, padding=1)
        self.output_layer = nn.Conv2d(64, 15, kernel_size=1)

    def forward(self, x):
        _, time_steps, _, height, width = x.shape
        spatial_frames, pooled_frames = [], []
        for t in range(time_steps):
            z = x[:, t]
            z = self.relu(self.bn1(self.conv1(z)))
            z = self.relu(self.bn2(self.conv2(z)))
            z = self.relu(self.bn3(self.conv3(z)))
            z = self.res1(z)
            z = self.res2(z)
            spatial_frames.append(z)
            pooled_frames.append(z.mean(dim=(-2, -1)))
        temporal = torch.stack(pooled_frames, dim=1)
        attention_output, _ = self.temporal_attention(temporal, temporal, temporal, need_weights=False)
        temporal = self.temporal_norm(temporal + attention_output)
        temporal = temporal + self.temporal_ffn(temporal)
        embedding = self.embedding_head(temporal.mean(dim=1))
        embedding_map = self.embedding_to_spatial(embedding[:, :, None, None].expand(-1, -1, height, width))
        z = torch.cat([spatial_frames[-1], embedding_map], dim=1)
        z = self.relu(self.decoder1(z))
        z = self.relu(self.decoder2(z))
        return self.output_layer(z), embedding


def get_secret(key):
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def get_bytes(upload, secret_name):
    if upload is not None:
        return upload.getvalue()
    url = get_secret(secret_name)
    if not url:
        return None
    try:
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as exc:
        st.error(f"Could not retrieve {secret_name}: {exc}")
        return None


@st.cache_resource(show_spinner="Loading OceanEmbed model...")
def load_model(checkpoint_bytes):
    buffer = io.BytesIO(checkpoint_bytes)
    try:
        obj = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        obj = torch.load(buffer, map_location="cpu")
    state = obj.get("model_state_dict", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
    model = OceanEmbedModelExact()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_npz(raw, label):
    try:
        with np.load(io.BytesIO(raw), allow_pickle=True) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    except Exception as exc:
        st.error(f"Could not load {label}: {exc}")
        return None


st.title("🌊 OceanEmbed Explorer")
st.caption("PyTorch-based subsurface ocean temperature reconstruction")
st.info("The model uses seven consecutive daily frames of seven surface variables to estimate temperatures at 15 depth levels. Predictions are estimates, not direct observations.")

with st.sidebar:
    st.header("Model files")
    checkpoint_upload = st.file_uploader("Upload oceanembed_best.pth", type=["pth", "pt"])
    surface_upload = st.file_uploader("Upload surface_normalization.npz", type=["npz"])
    target_upload = st.file_uploader("Upload target_temperature_stats.npz", type=["npz"])
    st.header("Input data")
    nc_upload = st.file_uploader("Upload monthly OceanEmbed NetCDF", type=["nc", "netcdf"])

checkpoint_bytes = get_bytes(checkpoint_upload, "MODEL_URL")
surface_bytes = get_bytes(surface_upload, "SURFACE_STATS_URL")
target_bytes = get_bytes(target_upload, "TARGET_STATS_URL")

if not all([checkpoint_bytes, surface_bytes, target_bytes]):
    st.warning("Upload the checkpoint and both normalization files, or configure their direct-download URLs in Streamlit Secrets.")
    st.code("MODEL_URL = 'https://.../oceanembed_best.pth'\nSURFACE_STATS_URL = 'https://.../surface_normalization.npz'\nTARGET_STATS_URL = 'https://.../target_temperature_stats.npz'", language="toml")
    st.stop()

surface_stats = load_npz(surface_bytes, "surface normalization")
target_stats = load_npz(target_bytes, "target statistics")
if surface_stats is None or target_stats is None:
    st.stop()

try:
    surface_mean = surface_stats["mean"].astype(np.float32)
    surface_std = surface_stats["std"].astype(np.float32)
    target_mean = target_stats["mean"].astype(np.float32)
    target_std = target_stats["std"].astype(np.float32)
    channel_names = [str(v) for v in surface_stats["channels"].tolist()]
    assert surface_mean.shape == (7,) and surface_std.shape == (7,)
    assert target_mean.shape == (15,) and target_std.shape == (15,)
    assert np.all(np.isfinite(surface_mean)) and np.all(np.isfinite(surface_std)) and np.all(surface_std > 0)
    assert np.all(np.isfinite(target_mean)) and np.all(np.isfinite(target_std)) and np.all(target_std > 0)
    assert len(channel_names) == 7 and all(c in channel_names for c in CHANNELS)
    model = load_model(checkpoint_bytes)
except Exception as exc:
    st.error(f"Model setup failed: {exc}")
    st.stop()

st.success("PyTorch checkpoint loaded successfully.")

if nc_upload is None:
    st.warning("Upload a monthly NetCDF file to choose a date and location.")
    st.stop()

try:
    with xr.open_dataset(io.BytesIO(nc_upload.getvalue())) as ds_open:
        ds = ds_open.load()
except Exception as exc:
    st.error(f"Could not open NetCDF file: {exc}")
    st.stop()

if not {"surface", "thetao"}.issubset(ds.variables) or not {"time", "channel", "latitude", "longitude", "depth"}.issubset(ds.coords):
    st.error("NetCDF must contain surface, thetao, time, channel, latitude, longitude and depth.")
    st.stop()
if ds.sizes["time"] < 7:
    st.error("At least seven daily frames are required.")
    st.stop()

times = pd.to_datetime(ds.time.values)
valid_idx = list(range(6, len(times)))
st.header("Select prediction date and location")
c1, c2, c3 = st.columns(3)
with c1:
    target_idx = st.selectbox("Target date (uses previous six days)", valid_idx, format_func=lambda i: pd.Timestamp(times[i]).strftime("%Y-%m-%d"))
with c2:
    lat_vals = np.asarray(ds.latitude.values)
    lat = st.selectbox("Latitude", lat_vals.tolist(), index=len(lat_vals)//2)
with c3:
    lon_vals = np.asarray(ds.longitude.values)
    lon = st.selectbox("Longitude", lon_vals.tolist(), index=len(lon_vals)//2)

map_depth = st.selectbox("Depth for spatial map", list(range(15)), format_func=lambda i: f"{float(ds.depth.values[i]):g} m")

if st.button("🌡️ Reconstruct temperature", type="primary"):
    try:
        surf = ds.surface.transpose("time", "channel", "latitude", "longitude")
        labels = [str(v) for v in ds.channel.values.tolist()]
        order = [labels.index(c) for c in channel_names]
        x = np.asarray(surf.isel(time=slice(target_idx-6, target_idx+1)).values, dtype=np.float32)[:, order]
        x = (x - surface_mean[None, :, None, None]) / surface_std[None, :, None, None]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xt = torch.from_numpy(np.ascontiguousarray(x))[None].float()
        with torch.inference_mode():
            pred_norm, embedding = model(xt)
        pred = pred_norm[0].cpu().numpy() * target_std[:, None, None] + target_mean[:, None, None]
        lat_vals = np.asarray(ds.latitude.values)
        lon_vals = np.asarray(ds.longitude.values)
        iy = int(np.abs(lat_vals-float(lat)).argmin())
        ix = int(np.abs(lon_vals-float(lon)).argmin())
        profile = pred[:, iy, ix]
        depths = np.asarray(ds.depth.values, dtype=np.float32)
        st.success(f"Prediction generated for {pd.Timestamp(times[target_idx]).strftime('%Y-%m-%d')}.")
        a, b, c = st.columns(3)
        a.metric("Surface estimate", f"{profile[0]:.2f} °C")
        b.metric("1000 m estimate", f"{profile[-1]:.2f} °C")
        c.metric("Embedding size", str(embedding.shape[-1]))
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(5, 6))
            ax.plot(profile, depths, marker="o")
            ax.invert_yaxis()
            ax.set_xlabel("Predicted temperature (°C)")
            ax.set_ylabel("Depth (m)")
            ax.set_title(f"Profile · {float(lat):.2f}°N, {float(lon):.2f}°E")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(pred[map_depth], origin="lower", aspect="auto",
                           extent=[float(lon_vals.min()), float(lon_vals.max()), float(lat_vals.min()), float(lat_vals.max())])
            ax.scatter([float(lon)], [float(lat)], marker="x")
            fig.colorbar(im, ax=ax, label="Predicted temperature (°C)")
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(f"Spatial prediction · {float(depths[map_depth]):g} m")
            st.pyplot(fig)
            plt.close(fig)
        result = pd.DataFrame({"Depth_m": depths, "Predicted_temperature_C": np.round(profile, 4)})
        st.subheader("Predicted temperature profile")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("Download profile CSV", result.to_csv(index=False).encode(), f"oceanembed_profile_{pd.Timestamp(times[target_idx]).strftime('%Y%m%d')}.csv", "text/csv")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.exception(exc)

st.divider()
st.subheader("About OceanEmbed")
c1, c2, c3 = st.columns(3)
c1.metric("Surface variables", "7")
c2.metric("Input window", "7 days")
c3.metric("Output depths", "15")
st.markdown("Inputs: SST, SSS, SLA, U/V currents and U/V winds. The model combines convolutional residual features, temporal attention and a spatial decoder. Independent Argo validation provenance has not been verified; do not interpret predictions as direct observations.")
