# MeteoInsight

Application locale d'analyse météo et de machine learning. MeteoInsight charge un CSV météo sur l'ordinateur, contrôle sa qualité, analyse les tendances et extrêmes, entraîne des modèles de prévision de température à J+1, puis génère un rapport et une interface Streamlit.

Le projet est conçu pour fonctionner sur CPU et pour conserver les données météo localement.

## Fonctionnalités

- Chargement local de fichiers météo CSV.
- Détection et mapping des colonnes météo.
- Contrôles de qualité : période, dates manquantes, doublons, valeurs manquantes et incohérences physiques.
- Tendance de température par régression linéaire, avec pente par décennie, p-value et R².
- Statistiques mensuelles et annuelles.
- Comptage de jours chauds, de jours de gel et de jours de forte pluie.
- Détection d'anomalies saisonnières avec z-score : chaque janvier est comparé aux autres janvier, chaque juillet aux autres juillet, etc.
- Classement des mois les plus chauds, froids ou humides et des années les plus chaudes.
- Prévision de la température moyenne du lendemain.
- Comparaison entre baseline naïve, régression linéaire et Random Forest.
- Évaluation par MAE, RMSE et biais.
- Importance des variables par permutation.
- Validation temporelle glissante sur plusieurs fenêtres futures.
- Graphiques de prédiction, synthèse climatique annuelle et climatologie mensuelle.
- Rapport Markdown créé localement.
- Interface Streamlit et assistant Ollama local alimenté uniquement par des résultats agrégés.

## Confidentialité

Les fichiers météo bruts, calculs, rapports et modèles restent sur l'ordinateur.

- Le CSV brut n'est pas transmis au modèle de langage local.
- L'assistant reçoit uniquement un résumé calculé, tel que les métriques, anomalies et tendances.
- Ne versionnez pas vos données personnelles ou confidentielles dans Git.
- Le fichier `.gitignore` exclut les données, sorties et environnements locaux.

## Prérequis

- Windows, macOS ou Linux.
- Python 3.10 ou version plus récente.
- Un environnement virtuel Python recommandé.
- Ollama uniquement si vous souhaitez utiliser le chat local.

## Installation Windows

Dans PowerShell, placez-vous dans le dossier du projet :

```powershell
cd C:\chemin\vers\meteo-ml-insights
```

Créez puis activez un environnement virtuel :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installez les dépendances :

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Structure du projet

```text
meteo-ml-insights/
├── app.py
├── run_analysis.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/                 # CSV météo local, ignoré par Git
│   └── processed/           # Données transformées, ignorées par Git
├── models/                  # Modèles locaux, ignorés par Git
├── outputs/                 # Rapports et graphiques, ignorés par Git
├── src/
│   ├── loader.py
│   ├── validator.py
│   ├── features.py
│   ├── forecasting.py
│   ├── evaluation.py
│   ├── anomalies.py
│   ├── insights.py
│   ├── reporting.py
│   └── local_llm.py
└── tests/
    └── test_analysis_core.py
```

## Lancer l'analyse

Placez votre fichier CSV dans `data/raw/`, puis lancez :

```powershell
python run_analysis.py "data/raw/mon_fichier_meteo.csv"
```

L'analyse effectue automatiquement :

```text
Qualité des données
→ Tendance climatique
→ Extrêmes et anomalies saisonnières
→ Prévision J+1 et comparaison des modèles
→ Importance par permutation
→ Validation temporelle glissante
→ Graphiques et rapport Markdown
```

Les résultats sont créés dans `outputs/`.

### Options utiles

Pour ne pas générer les graphiques :

```powershell
python run_analysis.py "data/raw/mon_fichier_meteo.csv" --no-plots
```

Pour lancer plus vite en ignorant la validation temporelle glissante :

```powershell
python run_analysis.py "data/raw/mon_fichier_meteo.csv" --skip-walk-forward
```

## Lancer l'interface

Activez l'environnement virtuel, puis lancez :

```powershell
streamlit run app.py
```

Dans le navigateur :

1. Sélectionnez un CSV météo local.
2. Cliquez sur **Lancer l'analyse complète**.
3. Consultez les résultats, tableaux, graphiques et rapport.
4. Utilisez le chat local si Ollama et le modèle configuré sont disponibles.

La validation glissante peut rallonger l'analyse dans Streamlit, car la Random Forest est réentraînée sur plusieurs fenêtres temporelles.

## Lancer les tests

Les tests utilisent des données synthétiques : ils ne lisent pas le CSV personnel.

```powershell
python -m pytest -q
```

Résultat attendu :

```text
4 passed
```

Pour le détail :

```powershell
python -m pytest -v
```

## Méthode et limites

- La cible prédite est la température moyenne à J+1.
- Les données de train et de test sont séparées chronologiquement.
- La validation glissante vérifie la stabilité des performances dans plusieurs périodes futures.
- La MAE est l'erreur absolue moyenne en degrés Celsius.
- Le RMSE pénalise davantage les grosses erreurs.
- Le biais est la moyenne de `prédiction - réalité` : un biais négatif indique une sous-estimation moyenne.
- L'importance par permutation mesure une utilité prédictive dans le modèle ; elle ne prouve pas une relation causale.
- Une anomalie saisonnière est détectée lorsque le z-score d'un mois atteint au moins 2 en valeur absolue.
- Les tendances, anomalies et prédictions décrivent les données analysées ; elles ne permettent pas d'attribuer une cause scientifique aux phénomènes observés.

## GitHub

Avant le premier envoi sur GitHub, vérifiez ce qui sera ajouté :

```powershell
git status
```

Le statut ne doit pas contenir de CSV météo personnel, de contenu dans `data/raw/`, de modèles, de rapports dans `outputs/` ou de dossier `.venv/`.

Initialisez ensuite le dépôt et créez un premier commit :

```powershell
git init
git add .
git status
git commit -m "Initial MeteoInsight project"
```

Pour une démonstration publique, utilisez un fichier de données synthétique ou public, distinct des données réelles.

## Évolutions possibles

- Ajouter le chargement XLSX/XLSB.
- Rendre les seuils d'extrêmes configurables dans Streamlit.
- Ajouter un graphique de l'erreur de prévision par fenêtre ou par année.
- Ajouter une validation temporelle glissante paramétrable.
- Comparer ultérieurement un modèle supplémentaire, tel que HistGradientBoostingRegressor.
- Exporter le rapport en HTML ou PDF.
