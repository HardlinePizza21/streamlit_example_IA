import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def generate_climate_data(start_date: str = "2023-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    days = np.arange(len(dates))

    # Seasonal temperature pattern + random noise
    temp_base = 15 + 10 * np.sin(2 * np.pi * days / 365)
    temp_noise = np.random.normal(scale=3.5, size=len(dates))
    temperature_c = temp_base + temp_noise

    # Precipitation with seasonal tendency and random events
    precip_chance = 0.35 + 0.15 * np.cos(2 * np.pi * days / 365)
    precipitation_mm = np.where(
        np.random.rand(len(dates)) < precip_chance,
        np.random.gamma(shape=2.0, scale=3.0, size=len(dates)),
        0.0,
    )
    precipitation_mm = np.round(precipitation_mm, 1)

    # Humidity depends on precipitation and season
    humidity = (
        65
        + 15 * np.cos(2 * np.pi * days / 365)
        + 0.25 * precipitation_mm
        + np.random.normal(scale=6.0, size=len(dates))
    )
    humidity = np.clip(humidity, 20, 100)

    # Wind speed and solar radiation patterns
    wind_speed_kmh = np.clip(5 + 2 * np.sin(2 * np.pi * days / 14) + np.random.normal(scale=1.5, size=len(dates)), 0, 25)
    solar_radiation_kwh = np.clip(5 + 4 * np.sin(2 * np.pi * days / 365 - 0.4) + np.random.normal(scale=1.1, size=len(dates)), 0, 12)

    # Cloud cover based on precipitation and humidity
    cloud_cover = np.clip(40 + 0.7 * humidity - 0.9 * temperature_c + np.random.normal(scale=10, size=len(dates)), 0, 100)

    df = pd.DataFrame(
        {
            "date": dates,
            "temperature_C": np.round(temperature_c, 1),
            "precipitation_mm": precipitation_mm,
            "humidity_%": np.round(humidity, 1),
            "wind_speed_kmh": np.round(wind_speed_kmh, 1),
            "solar_radiation_kwh": np.round(solar_radiation_kwh, 1),
            "cloud_cover_%": np.round(cloud_cover, 1),
        }
    )

    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear

    def season_of_date(date: pd.Timestamp) -> str:
        month = date.month
        if month in (12, 1, 2):
            return "Invierno"
        if month in (3, 4, 5):
            return "Primavera"
        if month in (6, 7, 8):
            return "Verano"
        return "Otoño"

    df["season"] = df["date"].apply(season_of_date)
    df["weekday"] = df["date"].dt.day_name()
    return df


def add_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")
    years = sorted(df["year"].unique())
    seasons = ["Todas"] + ["Invierno", "Primavera", "Verano", "Otoño"]
    selected_years = st.sidebar.multiselect("Año", years, default=years)
    selected_season = st.sidebar.selectbox("Temporada", seasons, index=0)
    selected_variables = st.sidebar.multiselect(
        "Variables para gráfico de serie",
        ["temperature_C", "precipitation_mm", "humidity_%", "wind_speed_kmh", "solar_radiation_kwh", "cloud_cover_%"],
        default=["temperature_C", "precipitation_mm"],
    )
    st.sidebar.markdown("---")
    show_correlations = st.sidebar.checkbox("Mostrar matriz de correlación", value=True)
    show_monthly_heatmap = st.sidebar.checkbox("Mostrar heatmap mensual", value=True)

    filtered = df[df["year"].isin(selected_years)]
    if selected_season != "Todas":
        filtered = filtered[filtered["season"] == selected_season]

    return filtered, selected_variables, show_correlations, show_monthly_heatmap


def display_metrics(df: pd.DataFrame) -> None:
    st.subheader("Indicadores clave")
    cols = st.columns(4)
    cols[0].metric("Temp. media (°C)", f"{df['temperature_C'].mean():.1f}")
    cols[1].metric("Precipitación total (mm)", f"{df['precipitation_mm'].sum():.0f}")
    cols[2].metric("Humedad media (%)", f"{df['humidity_%'].mean():.1f}")
    cols[3].metric("Vel. viento media (km/h)", f"{df['wind_speed_kmh'].mean():.1f}")


def timeline_plot(df: pd.DataFrame, variables: list[str]) -> None:
    st.subheader("Serie temporal de clima")
    if not variables:
        st.info("Selecciona al menos una variable para el gráfico de serie.")
        return

    timeline = df.melt(id_vars=["date"], value_vars=variables, var_name="variable", value_name="valor")
    fig = px.line(
        timeline,
        x="date",
        y="valor",
        color="variable",
        labels={"date": "Fecha", "valor": "Valor", "variable": "Variable"},
        title="Evolución diaria de variables climáticas",
        height=450,
    )
    fig.update_layout(legend_title_text="Variable")
    st.plotly_chart(fig, use_container_width=True)


def distribution_plots(df: pd.DataFrame) -> None:
    st.subheader("Distribuciones por variable")
    variables = ["temperature_C", "precipitation_mm", "humidity_%", "wind_speed_kmh", "solar_radiation_kwh", "cloud_cover_%"]
    for variable in variables:
        fig = px.histogram(
            df,
            x=variable,
            nbins=30,
            marginal="box",
            opacity=0.8,
            title=f"Distribución de {variable.replace('_', ' ')}",
            labels={variable: variable.replace('_', ' '), "count": "Frecuencia"},
        )
        st.plotly_chart(fig, use_container_width=True)


def correlation_section(df: pd.DataFrame) -> None:
    st.subheader("Correlación entre variables")
    numeric_cols = ["temperature_C", "precipitation_mm", "humidity_%", "wind_speed_kmh", "solar_radiation_kwh", "cloud_cover_%"]
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels={"x": "Variable", "y": "Variable", "color": "Correlación"},
        title="Matriz de correlación de variables climáticas",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)


def monthly_aggregation_section(df: pd.DataFrame) -> None:
    st.subheader("Resumen mensual")
    monthly = (
        df.groupby(["year", "month", "month_name"])
        .agg(
            temperature_C_mean=("temperature_C", "mean"),
            precipitation_mm_sum=("precipitation_mm", "sum"),
            humidity_mean=("humidity_%", "mean"),
            wind_speed_mean=("wind_speed_kmh", "mean"),
            solar_radiation_mean=("solar_radiation_kwh", "mean"),
        )
        .reset_index()
    )
    monthly["month_name"] = pd.Categorical(
        monthly["month_name"],
        categories=list(calendar.month_name[1:]),
        ordered=True,
    )
    monthly = monthly.sort_values(["year", "month"])

    st.dataframe(monthly.style.format({
        "temperature_C_mean": "{:.1f}",
        "precipitation_mm_sum": "{:.0f}",
        "humidity_mean": "{:.1f}",
        "wind_speed_mean": "{:.1f}",
        "solar_radiation_mean": "{:.1f}",
    }), use_container_width=True)

    fig = px.line(
        monthly,
        x="month_name",
        y="temperature_C_mean",
        color="year",
        markers=True,
        labels={"month_name": "Mes", "temperature_C_mean": "Temp. media (°C)", "year": "Año"},
        title="Temperatura media mensual por año",
    )
    st.plotly_chart(fig, use_container_width=True)


def seasonal_boxplot_section(df: pd.DataFrame) -> None:
    st.subheader("Análisis por temporada")
    fig = px.box(
        df,
        x="season",
        y="temperature_C",
        color="season",
        labels={"season": "Temporada", "temperature_C": "Temperatura (°C)"},
        title="Distribución de temperatura por temporada",
    )
    st.plotly_chart(fig, use_container_width=True)


def monthly_heatmap(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby(["month", "season"]) ["precipitation_mm"].mean().reset_index()
    )
    monthly["month_name"] = pd.Categorical(
        monthly["month"].apply(lambda x: calendar.month_name[x]),
        categories=list(calendar.month_name[1:]),
        ordered=True,
    )
    monthly = monthly.sort_values("month")
    heatmap = monthly.pivot(index="season", columns="month_name", values="precipitation_mm")
    fig = px.imshow(
        heatmap,
        labels={"x": "Mes", "y": "Temporada", "color": "Precipitación media (mm)"},
        title="Precipitación media mensual por temporada",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="EDA Climático - Streamlit",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Análisis EDA de datos climáticos simulados")
    st.markdown(
        "Este dashboard simula datos meteorológicos diarios y ofrece un análisis exploratorio de variables como temperatura, precipitación, humedad, viento y radiación solar."
    )

    df = generate_climate_data()
    filtered_df, selected_vars, show_corr, show_heatmap = add_sidebar_filters(df)

    display_metrics(filtered_df)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Muestra de datos")
        st.dataframe(filtered_df.head(10), use_container_width=True)
    with col2:
        st.markdown("### Estadísticas generales")
        st.write(filtered_df.describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    st.markdown("---")
    timeline_plot(filtered_df, selected_vars)
    distribution_plots(filtered_df)
    if show_corr:
        correlation_section(filtered_df)
    monthly_aggregation_section(filtered_df)
    seasonal_boxplot_section(filtered_df)
    if show_heatmap:
        monthly_heatmap(filtered_df)

    st.markdown("---")
    st.caption("Datos generados de forma sintética para demostración de EDA y visualización en Streamlit.")


if __name__ == "__main__":
    main()
