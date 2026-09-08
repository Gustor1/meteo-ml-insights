import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"


SYSTEM_PROMPT = """
Tu es MeteoInsight, un assistant local spécialisé en analyse météo
et machine learning.

Tu réponds à l'utilisateur à partir d'un contexte calculé localement
par un programme Python.

Règles impératives :
- Réponds uniquement en français.
- Réponds clairement et de façon concise : trois phrases maximum,
  sauf si une courte liste est indispensable.
- Utilise exclusivement les chiffres et informations présents dans
  le contexte transmis.
- N'invente jamais une valeur, une date, une période, une métrique,
  une variable ou une cause scientifique.
- Si le contexte ne permet pas de répondre, réponds exactement :
  "Cette information n'a pas encore été calculée par MeteoInsight."
- Ne prétends jamais avoir lu le fichier CSV brut.
- Ne propose jamais de causalité scientifique à partir d'une corrélation,
  d'une tendance ou d'une importance de variable.
- Une importance de permutation décrit une utilité prédictive dans un
  modèle ; elle ne démontre pas une causalité.
- Un biais négatif signifie que le modèle sous-estime en moyenne.
- Un biais positif signifie que le modèle surestime en moyenne.
- MAE signifie Mean Absolute Error, ou erreur absolue moyenne.
- RMSE signifie Root Mean Squared Error, ou racine de l'erreur quadratique moyenne.
- Si la question ne concerne pas l'analyse météo fournie, indique
  poliment que MeteoInsight répond uniquement aux résultats locaux
  de l'analyse en cours.
""".strip()


def ask_ollama_local(
    question: str,
    context: str,
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Interroge Ollama localement avec un contexte agrégé.

    Aucun CSV brut ni aucune donnée quotidienne détaillée
    n'est transmis au modèle.
    """
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Contexte météo calculé localement :\n\n"
                    f"{context}\n\n"
                    "Question de l'utilisateur :\n"
                    f"{question}"
                ),
            },
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 300,
        },
    }

    request = urllib.request.Request(
        url=OLLAMA_URL,
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
            "Impossible de joindre Ollama local. Vérifie qu'Ollama est ouvert "
            "et que le modèle qwen2.5:7b est installé."
        ) from error

    return response_data["message"]["content"]