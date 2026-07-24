import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from io import BytesIO


def generate_synthetic_climate(start_date, end_date, freq, n_locations=3, seed=42, anomaly_rate=0.0):
	np.random.seed(seed)
	date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
	rows = []
	for loc in range(1, n_locations + 1):
		lat = np.random.uniform(-60, 60)
		lon = np.random.uniform(-140, 140)
		base_temp = np.random.uniform(-5, 25)
		amp = np.random.uniform(5, 15)
		noise_scale = np.random.uniform(0.3, 3.0)
		for t in date_range:
			day_of_year = t.timetuple().tm_yday
			seasonal = amp * np.sin(2 * np.pi * (day_of_year / 365.0))
			diurnal = 2.0 * np.sin(2 * np.pi * (t.hour / 24.0)) if freq.upper().startswith("H") else 0.0
			temp = base_temp + seasonal + diurnal + np.random.normal(scale=noise_scale)
			humidity = float(np.clip(55 + 20 * np.sin(2 * np.pi * (day_of_year / 365.0) + 1.0) + np.random.normal(scale=8), 0, 100))
			wind_speed = float(np.abs(np.random.normal(loc=5.0, scale=2.5)))
			precip = float(max(0.0, np.random.exponential(scale=1.0) - 0.7 * np.sin(2 * np.pi * (day_of_year / 365.0))))
			rows.append({
				"timestamp": t,
				"location": f"loc_{loc}",
				"latitude": lat,
				"longitude": lon,
				"temperature": round(temp, 2),
				"humidity": round(humidity, 1),
				"wind_speed": round(wind_speed, 2),
				"precipitation": round(precip, 3),
			})

	df = pd.DataFrame(rows)

	# add random anomalies
	if anomaly_rate and anomaly_rate > 0:
		n_anom = int(len(df) * anomaly_rate)
		idx = np.random.choice(df.index, size=n_anom, replace=False)
		df.loc[idx, "temperature"] *= np.random.choice([1.5, 2.0, -1.2], size=n_anom)

	return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
	return df.to_csv(index=False).encode("utf-8")


def main():
	st.set_page_config(layout="wide", page_title="Generador y EDA Climático")
	st.title("Generador de datos sintéticos del clima — EDA interactivo")

	with st.sidebar:
		st.header("Parámetros de generación")
		start_date = st.date_input("Fecha inicio", value=dt.date.today() - dt.timedelta(days=30))
		end_date = st.date_input("Fecha fin", value=dt.date.today())
		freq = st.selectbox("Frecuencia", options=["D", "H"], format_func=lambda x: "Diaria (D)" if x == "D" else "Horaria (H)")
		n_locations = st.slider("Número de ubicaciones", 1, 10, 3)
		seed = st.number_input("Semilla aleatoria", value=42, step=1)
		anomaly_rate = st.slider("Tasa de anomalías (fracción)", 0.0, 0.2, 0.0, step=0.01)
		generate = st.button("Generar datos")

	if generate:
		with st.spinner("Generando datos..."):
			df = generate_synthetic_climate(start_date, end_date, freq, n_locations=n_locations, seed=int(seed), anomaly_rate=float(anomaly_rate))

		st.success(f"Datos generados: {len(df)} filas — {n_locations} ubicaciones")

		# Data preview and download
		st.subheader("Vista previa de datos")
		st.dataframe(df.head(200))
		st.download_button("Descargar CSV", data=to_csv_bytes(df), file_name="synthetic_climate.csv", mime="text/csv")

		# Summary stats
		st.subheader("Estadísticas resumidas")
		st.write(df.describe())

		# Correlation heatmap
		st.subheader("Mapa de correlación")
		numeric_cols = ["temperature", "humidity", "wind_speed", "precipitation"]
		corr = df[numeric_cols].corr()
		fig, ax = plt.subplots(figsize=(6, 4))
		sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
		st.pyplot(fig)

		# Time series plotting
		st.subheader("Series temporales — Interactivo")
		col1, col2 = st.columns([3, 1])
		with col2:
			var = st.selectbox("Variable", options=numeric_cols)
			loc = st.selectbox("Ubicación", options=sorted(df["location"].unique()))
		df_loc = df[df["location"] == loc]
		fig_px = px.line(df_loc, x="timestamp", y=var, title=f"{var} en {loc}")
		fig_px.update_layout(hovermode="x unified")
		with col1:
			st.plotly_chart(fig_px, use_container_width=True)

		# Distribution plots
		st.subheader("Distribuciones")
		fig2 = px.histogram(df, x=var, nbins=50, color="location", marginal="box", title=f"Histograma de {var}")
		st.plotly_chart(fig2, use_container_width=True)

		# Altair example: temperature heatmap over time and location
		st.subheader("Mapa temporal por ubicación (Altair)")
		pivot = df.pivot_table(index="timestamp", columns="location", values="temperature")
		pivot_reset = pivot.reset_index().melt(id_vars=["timestamp"], var_name="location", value_name="temperature")
		chart = alt.Chart(pivot_reset).mark_rect().encode(
			x=alt.X("timestamp:T", title="Fecha"),
			y=alt.Y("location:N", title="Ubicación"),
			color=alt.Color("temperature:Q", scale=alt.Scale(scheme="turbo")),
			tooltip=[alt.Tooltip("timestamp:T"), alt.Tooltip("location:N"), alt.Tooltip("temperature:Q")],
		).properties(height=300)
		st.altair_chart(chart, use_container_width=True)

		# Profile report (ydata-profiling)
		with st.expander("Reporte de perfil (ydata-profiling)"):
			st.info("El reporte puede tardar unos segundos dependiendo del tamaño del dataset.")
			profile = ProfileReport(df, minimal=True)
			html = profile.to_html()
			components.html(html, height=700, scrolling=True)


if __name__ == "__main__":
	main()

