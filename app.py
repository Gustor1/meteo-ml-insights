from pathlib import Path

import streamlit as st

from run_analysis import run_full_analysis
from src.local_llm import ask_ollama_local

st.set_page_config(page_title="MeteoInsight", page_icon="🌦️", layout="wide")


def build_llm_context(results: dict) -> str:
    """Crée le résumé agrégé transmis au LLM local, jamais le CSV brut."""
    quality = results["quality_report"]
    period = quality["periode"]
    trend = results["trend"]
    metrics = results["model_metrics"]
    importance_df = results["importance_df"]
    extremes = results["daily_extremes"]
    anomalies = results["monthly_anomalies"]
    hottest_month = results["hottest_months"].iloc[0]
    wettest_month = results["wettest_months"].iloc[0]
    hottest_year = results["hottest_years"].iloc[0]

    best_model_name = min(metrics, key=lambda name: metrics[name]["mae"])
    best_model = metrics[best_model_name]
    baseline = metrics["Baseline naïve"]
    mae_gain = baseline["mae"] - best_model["mae"]
    relative_gain = (mae_gain / baseline["mae"]) * 100
    total_missing = sum(quality["valeurs_manquantes"].values())
    total_physical_issues = sum(quality["incoherences_physiques"].values())

    models_text = "\n".join(
        f"- {name} : MAE {values['mae']:.3f} °C ; RMSE {values['rmse']:.3f} °C ; biais {values['bias']:+.3f} °C."
        for name, values in metrics.items()
    )
    top_features_text = "\n".join(
        f"- {row['variable']} : +{row['augmentation_mae_moyenne']:.3f} °C de dégradation de MAE si mélangée."
        for _, row in importance_df.head(5).iterrows()
            if "walk_forward_summary" in results
    )
    walk_forward_summary = results.get(
        "walk_forward_summary"
    )

    return f"""
Fichier analysé : {results['source_filename']}.

Qualité et période :
- Du {period['date_min'].date()} au {period['date_max'].date()}.
- {quality['nb_lignes']} lignes et {quality['nb_colonnes']} colonnes.
- {period['dates_manquantes']} date(s) manquante(s) ; {quality['doublons_date']} doublon(s) ; {total_missing} valeur(s) manquante(s) ; {total_physical_issues} incohérence(s) physique(s).

Tendance climatique :
- Température moyenne : {trend['pente_par_decennie']:+.3f} °C par décennie.
- Significative : {'oui' if trend['significatif'] else 'non'} ; p-value {trend['p_value']:.6g} ; R² {trend['r_squared']:.4f}.

Extrêmes calculés sur l'ensemble de la période :
- {extremes['jours_chauds']} jours chauds : tmax ≥ {extremes['seuil_jour_chaud_c']:.1f} °C.
- {extremes['jours_de_gel']} jours de gel : tmin < {extremes['seuil_gel_c']:.1f} °C.
- {extremes['jours_forte_pluie']} jours de forte pluie : précipitations ≥ {extremes['seuil_forte_pluie_mm']:.1f} mm.
- {len(anomalies)} mois avec au moins une anomalie saisonnière, définie par |z| ≥ 2,0.
- Mois le plus chaud en température moyenne brute : {hottest_month['date'].strftime('%Y-%m')}, {hottest_month['temperature_moyenne_mensuelle']:.2f} °C, z-score saisonnier {hottest_month['zscore_temperature_saisonnier']:+.2f}.
- Mois le plus humide : {wettest_month['date'].strftime('%Y-%m')}, {wettest_month['precipitation_totale_mensuelle']:.1f} mm, z-score saisonnier {wettest_month['zscore_precipitation_saisonnier']:+.2f}.
- Année la plus chaude : {int(hottest_year['annee'])}, {hottest_year['temperature_moyenne_annuelle']:.2f} °C.

Résultats des modèles :
{models_text}

Meilleur modèle :
- {best_model_name}, MAE {best_model['mae']:.3f} °C, RMSE {best_model['rmse']:.3f} °C, biais {best_model['bias']:+.3f} °C.
- Gain sur la baseline naïve : {mae_gain:.3f} °C, soit {relative_gain:.1f} % de réduction de MAE.

Variables les plus importantes pour la Random Forest :
{top_features_text}

Règles de réponse :
- Utilise uniquement les chiffres du contexte.
- N'invente pas de cause météorologique ou climatique.
- Distingue un record brut d'une anomalie saisonnière.
- Indique que les anomalies comparent un mois aux mêmes mois calendaires des autres années.
- Rappelle les limites si une conclusion de causalité est demandée.
""".strip()


def show_results(results: dict) -> None:
    """Affiche les résultats calculés localement."""
    quality = results["quality_report"]
    period = quality["periode"]
    metrics = results["model_metrics"]
    extremes = results["daily_extremes"]
    best_model_name = min(metrics, key=lambda name: metrics[name]["mae"])
    best_model = metrics[best_model_name]

    st.subheader("Résultats principaux")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Meilleur modèle", best_model_name)
    col2.metric("MAE du meilleur modèle", f"{best_model['mae']:.3f} °C")
    col3.metric("Tendance par décennie", f"{results['trend']['pente_par_decennie']:+.3f} °C")
    col4.metric("Période couverte", f"{period['date_min'].year}–{period['date_max'].year}")

    st.markdown("### Extrêmes climatiques")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jours chauds", extremes["jours_chauds"], f"tmax ≥ {extremes['seuil_jour_chaud_c']:.0f} °C")
    col2.metric("Jours de gel", extremes["jours_de_gel"], f"tmin < {extremes['seuil_gel_c']:.0f} °C")
    col3.metric("Jours de forte pluie", extremes["jours_forte_pluie"], f"pluie ≥ {extremes['seuil_forte_pluie_mm']:.0f} mm")
    col4.metric("Mois anormaux", len(results["monthly_anomalies"]), "seuil |z| ≥ 2,0")

    st.caption("Les anomalies saisonnières comparent chaque mois aux mêmes mois calendaires des autres années.")

    st.markdown("### Comparaison des modèles")
    st.dataframe(
        {
            "Modèle": list(metrics.keys()),
            "MAE (°C)": [value["mae"] for value in metrics.values()],
            "RMSE (°C)": [value["rmse"] for value in metrics.values()],
            "Biais (°C)": [value["bias"] for value in metrics.values()],
        },
        use_container_width=True,
        hide_index=True,
    )
    monthly_forecasting = results.get("monthly_forecasting")
    if monthly_forecasting:
        monthly_metrics = monthly_forecasting["comparison_metrics"]
        future_forecast = monthly_forecasting["future_forecast"]
        best_monthly_model = min(monthly_metrics, key=lambda name: monthly_metrics[name]["mae"])

        st.markdown("### Prévision mensuelle Holt-Winters")
        st.caption(
            "Analyse mensuelle distincte de la prévision quotidienne à J+1 : "
            "elle prolonge la tendance et la saisonnalité observées."
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Meilleur modèle mensuel", best_monthly_model)
        col2.metric("MAE mensuelle", f"{monthly_metrics[best_monthly_model]['mae']:.3f} °C")
        col3.metric("Horizon projeté", f"{len(future_forecast)} mois")

        st.dataframe(
            {
                "Modèle": list(monthly_metrics.keys()),
                "MAE (°C)": [value["mae"] for value in monthly_metrics.values()],
                "RMSE (°C)": [value["rmse"] for value in monthly_metrics.values()],
                "Biais (°C)": [value["bias"] for value in monthly_metrics.values()],
            },
            use_container_width=True,
            hide_index=True,
        )
        forecast_table = future_forecast.rename("Température moyenne projetée (°C)").to_frame()
        forecast_table.index = forecast_table.index.strftime("%Y-%m")
        forecast_table.index.name = "Mois"
        st.dataframe(forecast_table, use_container_width=True)

        monthly_chart = Path("outputs/prevision_mensuelle_holt_winters.png")
        if monthly_chart.exists():
            st.image(
                str(monthly_chart),
                caption="Prévision mensuelle : test, Holt-Winters et projection",
                use_container_width=True,
            )
        else:
            st.warning(
                "Le graphique Holt-Winters est introuvable. Relance l'analyse avec les graphiques activés."
            )
        st.info(
            "La projection est statistique et mensuelle : elle ne constitue pas une prévision météo quotidienne."
        )

    st.markdown("### Validation temporelle glissante")

    walk_forward_summary = results.get(
        "walk_forward_summary"
    )

    walk_forward_results = results.get(
        "walk_forward_results"
    )

    if (
        walk_forward_summary is not None
        and not walk_forward_summary.empty
    ):
        st.caption(
            "Chaque fenêtre teste environ une année future. "
            "Le modèle est entraîné uniquement sur les données "
            "antérieures à cette période."
        )

        st.dataframe(
            walk_forward_summary,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Résultats détaillés par fenêtre temporelle"
        ):
            st.dataframe(
                walk_forward_results,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info(
            "La validation temporelle glissante n'a pas été exécutée."
        )

    st.markdown("### Tendance climatique")
    st.info(results["trend_text"])

    st.markdown("### Mois et années remarquables")
    tab_hot, tab_cold, tab_wet, tab_years, tab_anomalies = st.tabs([
        "Plus chauds", "Plus froids", "Plus humides", "Années chaudes", "Anomalies"
    ])
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

    st.markdown("### Graphiques générés localement")
    graph_columns = st.columns(3)
    graphs = [
        ("Baseline naïve", "outputs/predictions_baseline_naive.png"),
        ("Régression linéaire", "outputs/predictions_regression_lineaire.png"),
        ("Random Forest", "outputs/predictions_random_forest.png"),
    ]
    st.markdown("### Graphiques climatiques")

    climate_columns = st.columns(2)

    climate_graphs = [
        (
            "Synthèse climatique annuelle",
            "outputs/climat_annuel.png",
        ),
        (
            "Climatologie mensuelle",
            "outputs/climatologie_mensuelle.png",
        ),
    ]

    for column, (title, image_path) in zip(
        climate_columns,
        climate_graphs,
    ):
        if Path(image_path).exists():
            column.image(
                image_path,
                caption=title,
                use_container_width=True,
            )
        else:
            column.warning(
                f"Graphique introuvable : {image_path}"
            )
    for column, (title, image_path) in zip(graph_columns, graphs):
        if Path(image_path).exists():
            column.image(image_path, caption=title, use_container_width=True)

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
        with st.spinner("Analyse locale : qualité, anomalies, modèles et graphiques..."):
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

    question = st.chat_input("Exemple : Quel est le mois le plus anormalement humide ?")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        context = build_llm_context(results)
        with st.chat_message("assistant"):
            with st.spinner("MeteoInsight réfléchit localement..."):
                try:
                    answer = ask_ollama_local(question=question, context=context)
                except RuntimeError as error:
                    answer = f"Erreur lors de la communication avec Ollama : {error}"
            st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
