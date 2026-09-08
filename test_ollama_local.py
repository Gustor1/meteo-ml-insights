import json
import urllib.error
import urllib.request


def ask_ollama(question: str, context: str) -> str:
    """
    Envoie uniquement un contexte agrégé à Ollama local.
    Aucun fichier CSV brut n'est envoyé.
    """
    system_prompt = """
Tu es MeteoInsight, un assistant local spécialisé en analyse météo
et machine learning.

Règles impératives :
- Réponds uniquement en français.
- Réponds en trois phrases maximum, sauf si une liste est indispensable.
- Utilise exclusivement les chiffres et informations présents dans le contexte.
- N'invente jamais une valeur, une période, une métrique ou une cause scientifique.
- Si le contexte ne permet pas de répondre, dis clairement :
  "Cette information n'a pas encore été calculée par MeteoInsight."
- Ne prétends pas avoir accès au fichier CSV brut.
- Une importance de variable est une utilité prédictive, pas une causalité.
- Un biais négatif signifie que le modèle sous-estime en moyenne.
- Un biais positif signifie que le modèle surestime en moyenne.
- La MAE signifie Mean Absolute Error, ou erreur absolue moyenne.
"""

    payload = {
        "model": "qwen2.5:7b",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": (
                    "Voici les résultats calculés localement :\n\n"
                    f"{context}\n\n"
                    "Question de l'utilisateur :\n"
                    f"{question}"
                ),
            },
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 250,
        },
    }

    request = urllib.request.Request(
        url="http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Impossible de joindre Ollama local. Vérifie qu'Ollama est installé, "
            "que qwen2.5:7b est téléchargé, puis relance Ollama si nécessaire."
        ) from error

    return response_data["message"]["content"]


context = """
Période des données : 1992-09-01 à 2025-12-31.
Nombre de lignes : 12175.
Dates manquantes : 0.
Doublons de date : 0.

Tendance de température moyenne : +0,659 °C par décennie.
La tendance est statistiquement significative : p < 0,001.

Modèles évalués sur une période de test allant du 2019-05-12 au 2025-12-30 :
- Baseline naïve : MAE 1,819 °C ; RMSE 2,378 °C ; biais +0,006 °C.
- Régression linéaire : MAE 1,513 °C ; RMSE 1,980 °C ; biais -0,056 °C.
- Random Forest : MAE 1,459 °C ; RMSE 1,920 °C ; biais -0,153 °C.

Le meilleur modèle selon la MAE est la Random Forest.
La Random Forest réduit la MAE de 0,360 °C par rapport à la baseline naïve.

Variables les plus importantes pour la Random Forest :
- tmean : +3,638 °C de dégradation de MAE si mélangée.
- tmax : +1,766 °C.
- jour_annee_cos : +0,278 °C.
- insolation_min : +0,176 °C.

rayonnement_am est exclu du ML car il contient beaucoup de valeurs
manquantes durant la période récente.
""".strip()


question = (
    "Quel modèle est le meilleur et que signifie son biais ?"
)

answer = ask_ollama(
    question=question,
    context=context,
)

print("\n=== RÉPONSE D'OLLAMA LOCAL ===\n")
print(answer)