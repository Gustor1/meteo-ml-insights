from pathlib import Path

import streamlit as st

from run_analysis import INDICATOR_LABELS, run_full_analysis
from src.local_llm import ask_ollama_local

st.set_page_config(page_title="MeteoInsight", page_icon="🌦️", layout="wide")


def format_trends_for_llm(results: dict) -> str:
    annual = results.get("indicator_trends")
    seasonal = results.get("seasonal_trends")
    if annual is None or annual.empty:
        return "Tendances climatiques détaillées non calculées."

    annual_text = "\n".join(
        f"- {INDICATOR_LABELS.get(row['indicateur'], (row['indicateur'], ''))[0]} : "
        f"{row['pente_par_decennie']:+.2f} par décennie ; "
        f"p-value {row['p_value']:.4g} ; R² {row['r_squared']:.3f} ; "
        f"significative : {'oui' if row['significatif'] else 'non'}."
        for _, row in annual.iterrows()
    )
    seasonal_text = "\n".join(
        f"- {row['saison']} : {row['pente_par_decennie']:+.2f} °C par décennie ; "
        f"p-value {row['p_value']:.4g} ; significative : {'oui' if row['significatif'] else 'non'}."
        for _, row in seasonal.iterrows()
    )
    return f"Tendances annuelles :\n{annual_text}\n\nTendances par saison :\n{seasonal_text}"


def build_llm_context(results: dict) -> str:
    """Construit le contexte agrégé envoyé à Ollama, jamais le CSV brut."""
    quality = results["quality_report"]
    period = quality["periode"]
    trend = results["trend"]
    metrics = results["model_metrics"]
    extremes = results["daily_extremes"]
    importance = results["importance_df"]
    walk_forward = results.get("walk_forward_summary")

    best_name = min(metrics, key=lambda name: metrics[name]["mae"])
    best = metrics[best_name]
    baseline = metrics["Baseline naïve"]
    models_text = "\n".join(
        f"- {name} : MAE {values['mae']:.3f} °C ; RMSE {values['rmse']:.3f} °C ; biais {values['bias']:+.3f} °C."
        for name, values in metrics.items()
    )
    features_text = "\n".join(
        f"- {row['variable']} : +{row['augmentation_mae_moyenne']:.3f} °C de dégradation de MAE si mélangée."
        for _, row in importance.head(5).iterrows()
    )
    if walk_forward is not None and not walk_forward.empty:
        walk_text = "\n".join(
            f"- {row['modele']} : MAE moyenne {row['mae_moyenne']:.3f} °C ; "
            f"écart-type {row['mae_ecart_type']:.3f} °C ; biais moyen {row['biais_moyen']:+.3f} °C."
            for _, row in walk_forward.iterrows()
        )
    else:
        walk_text = "Non exécutée."

    return f"""
Résultats locaux pour {results['source_filename']}.
Période : {period['date_min'].date()} à {period['date_max'].date()}.
Tendance globale : {trend['pente_par_decennie']:+.3f} °C par décennie ; significative : {'oui' if trend['significatif'] else 'non'} ; p-value {trend['p_value']:.6g} ; R² {trend['r_squared']:.4f}.

Extrêmes :
- Jours chauds : {extremes['jours_chauds']} avec tmax ≥ {extremes['seuil_jour_chaud_c']:.1f} °C.
- Jours de gel : {extremes['jours_de_gel']} avec tmin < {extremes['seuil_gel_c']:.1f} °C.
- Jours de forte pluie : {extremes['jours_forte_pluie']} avec précipitations ≥ {extremes['seuil_forte_pluie_mm']:.1f} mm.
- Mois avec anomalie saisonnière : {len(results['monthly_anomalies'])}.

{format_trends_for_llm(results)}

Résultats de prévision J+1 :
{models_text}
Meilleur modèle du test final : {best_name}, MAE {best['mae']:.3f} °C. Gain sur baseline : {baseline['mae'] - best['mae']:.3f} °C.

Validation temporelle glissante :
{walk_text}

Variables importantes :
{features_text}

Règles : réponds seulement avec les résultats ci-dessus. N'invente pas de cause. Une tendance ou une p-value ne démontre pas une cause scientifique. Distingue record brut, anomalie saisonnière, test final et validation glissante.
""".strip()


def show_climate_trends(results: dict) -> None:
    annual_coverage = results["annual_coverage"]
    indicators = results["indicator_trends"].copy()
    seasons = results["seasonal_trends"].copy()
    complete_count = int((annual_coverage["taux_couverture"] >= 0.99).sum())

    st.markdown("### Évolution climatique annuelle et saisonnière")
    col1, col2 = st.columns(2)
    col1.metric("Années complètes retenues", f"{complete_count} / {len(annual_coverage)}", "couverture ≥ 99 %")
    col2.metric("Saisons analysées", len(seasons), "hiver, printemps, été, automne")
    st.caption("Les tendances annuelles excluent les années incomplètes. Elles décrivent cette série sans prouver une cause scientifique.")

    indicators = indicators[["indicateur", "pente_par_decennie", "p_value", "r_squared", "significatif"]]
    indicators["indicateur"] = indicators["indicateur"].map(lambda value: INDICATOR_LABELS.get(value, (value, ""))[0])
    indicators = indicators.rename(columns={
        "indicateur": "Indicateur", "pente_par_decennie": "Tendance/décennie",
        "p_value": "p-value", "r_squared": "R²", "significatif": "Significative",
    })
    indicators["Tendance/décennie"] = indicators["Tendance/décennie"].round(2)
    indicators["p-value"] = indicators["p-value"].round(4)
    indicators["R²"] = indicators["R²"].round(3)
    indicators["Significative"] = indicators["Significative"].map({True: "Oui", False: "Non"})

    seasons = seasons[["saison", "pente_par_decennie", "p_value", "r_squared", "significatif"]]
    seasons = seasons.rename(columns={
        "saison": "Saison", "pente_par_decennie": "Tendance °C/décennie",
        "p_value": "p-value", "r_squared": "R²", "significatif": "Significative",
    })
    seasons["Tendance °C/décennie"] = seasons["Tendance °C/décennie"].round(2)
    seasons["p-value"] = seasons["p-value"].round(4)
    seasons["R²"] = seasons["R²"].round(3)
    seasons["Significative"] = seasons["Significative"].map({True: "Oui", False: "Non"})

    left, right = st.columns(2)
    with left:
        st.markdown("#### Indicateurs annuels")
        st.dataframe(indicators, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Température par saison")
        st.dataframe(seasons, use_container_width=True, hide_index=True)

    with st.expander("Couverture annuelle détaillée"):
        st.dataframe(annual_coverage, use_container_width=True, hide_index=True)


def show_images(title: str, images: list[tuple[str, str]], columns: int) -> None:
    st.markdown(f"### {title}")
    for row_start in range(0, len(images), columns):
        row = images[row_start:row_start + columns]
        for column, (caption, path) in zip(st.columns(columns), row):
            if Path(path).exists():
                column.image(path, caption=caption, use_container_width=True)
            else:
                column.warning(f"Graphique introuvable : {path}")


def show_results(results: dict) -> None:
    quality = results["quality_report"]
    period = quality["periode"]
    metrics = results["model_metrics"]
    extremes = results["daily_extremes"]
    best_name = min(metrics, key=lambda name: metrics[name]["mae"])
    best = metrics[best_name]

    st.subheader("Résultats principaux")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Meilleur modèle", best_name)
    col2.metric("MAE du meilleur modèle", f"{best['mae']:.3f} °C")
    col3.metric("Tendance par décennie", f"{results['trend']['pente_par_decennie']:+.3f} °C")
    col4.metric("Période couverte", f"{period['date_min'].year}–{period['date_max'].year}")

    st.markdown("### Extrêmes climatiques")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jours chauds", extremes["jours_chauds"], f"tmax ≥ {extremes['seuil_jour_chaud_c']:.0f} °C")
    col2.metric("Jours de gel", extremes["jours_de_gel"], f"tmin < {extremes['seuil_gel_c']:.0f} °C")
    col3.metric("Jours de forte pluie", extremes["jours_forte_pluie"], f"pluie ≥ {extremes['seuil_forte_pluie_mm']:.0f} mm")
    col4.metric("Mois anormaux", len(results["monthly_anomalies"]), "seuil |z| ≥ 2,0")

    st.markdown("### Comparaison des modèles J+1")
    st.dataframe({
        "Modèle": list(metrics.keys()),
        "MAE (°C)": [item["mae"] for item in metrics.values()],
        "RMSE (°C)": [item["rmse"] for item in metrics.values()],
        "Biais (°C)": [item["bias"] for item in metrics.values()],
    }, use_container_width=True, hide_index=True)

    st.markdown("### Validation temporelle glissante")
    summary = results.get("walk_forward_summary")
    details = results.get("walk_forward_results")
    if summary is not None and not summary.empty:
        st.caption("Chaque fenêtre teste environ une année future, avec un entraînement effectué seulement sur les données antérieures.")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if details is not None and not details.empty:
            with st.expander("Résultats détaillés par fenêtre temporelle"):
                st.dataframe(details, use_container_width=True, hide_index=True)

    st.markdown("### Tendance climatique globale")
    st.info(results["trend_text"])
    show_climate_trends(results)

    st.markdown("### Mois et années remarquables")
    tab_hot, tab_cold, tab_wet, tab_years, tab_anomalies = st.tabs(["Plus chauds", "Plus froids", "Plus humides", "Années chaudes", "Anomalies"])
    with tab_hot:
        st.dataframe(results["hottest_months"], use_container_width=True, hide_index=True)
    with tab_cold:
        st.dataframe(results["coldest_months"], use_container_width=True, hide_index=True)
    with tab_wet:
        st.dataframe(results["wettest_months"], use_container_width=True, hide_index=True)
    with tab_years:
        st.dataframe(results["hottest_years"], use_container_width=True, hide_index=True)
    with tab_anomalies:
        st.dataframe(results["monthly_anomalies"], use_container_width=True, hide_index=True)

    show_images("Graphiques de prédiction", [
        ("Baseline naïve", "outputs/predictions_baseline_naive.png"),
        ("Régression linéaire", "outputs/predictions_regression_lineaire.png"),
        ("Random Forest", "outputs/predictions_random_forest.png"),
    ], columns=3)
    show_images("Graphiques climatiques", [
        ("Synthèse climatique annuelle", "outputs/climat_annuel.png"),
        ("Climatologie mensuelle", "outputs/climatologie_mensuelle.png"),
    ], columns=2)
    show_images("Graphiques d'évolution climatique", [
        ("Température par saison", "outputs/tendances_temperature_saisons.png"),
        ("Extrêmes thermiques", "outputs/tendances_extremes_thermiques.png"),
        ("Précipitations et fortes pluies", "outputs/tendances_precipitations.png"),
    ], columns=2)

    with st.expander("Importance par permutation"):
        st.dataframe(results["importance_df"], use_container_width=True, hide_index=True)
    with st.expander("Rapport Markdown local"):
        report_path = Path("outputs/rapport_meteo.md")
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.warning("Le rapport Markdown est introuvable.")


def main() -> None:
    st.title("🌦️ MeteoInsight")
    st.write("Analyse météo et assistant conversationnel local. Le fichier CSV est traité uniquement sur cet ordinateur.")
    uploaded_file = st.file_uploader("Choisis un fichier météo CSV local", type=["csv"])

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if uploaded_file is not None:
        st.success(f"Fichier sélectionné : {uploaded_file.name}")
    if st.button("Lancer l'analyse complète", type="primary", disabled=uploaded_file is None):
        upload_path = Path("data/raw/fichier_charge_dans_interface.csv")
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(uploaded_file.getvalue())
        with st.spinner("Analyse locale : qualité, climat, anomalies, modèles et graphiques..."):
            try:
                results = run_full_analysis(str(upload_path), test_size=0.20, create_plots=True)
            except (ValueError, FileNotFoundError) as error:
                st.error(f"Analyse impossible : {error}")
                return
        st.session_state.analysis_results = results
        st.session_state.chat_messages = []
        st.success("Analyse terminée. Les résultats restent locaux.")

    results = st.session_state.analysis_results
    if results is None:
        return
    show_results(results)

    st.divider()
    st.subheader("Assistant météo local")
    st.caption("Qwen répond à partir d'un résumé calculé localement. Le CSV brut n'est jamais transmis au modèle de langage.")
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Exemple : Les jours très chauds augmentent-ils dans la série ?")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("MeteoInsight réfléchit localement..."):
                try:
                    answer = ask_ollama_local(question=question, context=build_llm_context(results))
                except RuntimeError as error:
                    answer = f"Erreur lors de la communication avec Ollama : {error}"
            st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
