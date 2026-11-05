import numpy as np
from sentence_transformers import SentenceTransformer

from first_search import documents, search as search_bm25


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Chargement du modèle…")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

passage_embeddings = model.encode(
    [document["text"] for document in documents],
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print("Dimensions des embeddings :", passage_embeddings.shape)


def search_semantic(question, top_k=2):
    if not question.strip() or top_k <= 0:
        return []

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = passage_embeddings @ question_embedding

    ranked_indices = np.argsort(-scores)[:top_k]

    return [
        {
            "id": documents[index]["id"],
            "source": documents[index]["source"],
            "text": documents[index]["text"],
            "score": float(scores[index]),
        }
        for index in ranked_indices
    ]


def display_results(title, results):
    print(f"\n--- {title} ---")

    if not results:
        print("Aucun passage retourné.")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. [{result['id']}] "
            f"Score : {result['score']:.3f}"
        )
        print(result["text"])


if __name__ == "__main__":
    question = input("\nTa question : ")

    display_results(
        "Recherche BM25",
        search_bm25(question, top_k=2),
    )

    display_results(
        "Recherche par embeddings",
        search_semantic(question, top_k=2),
    )