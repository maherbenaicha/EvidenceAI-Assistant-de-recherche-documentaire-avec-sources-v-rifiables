from time import perf_counter

from hybrid_search import fuse_rankings
from semantic_search import search_bm25, search_semantic


QUESTIONS = [
    ("pression huile", "circuit_huile.txt::p001"),
    ("graphique température", "temperature.txt::p002"),
    ("indicateur du réservoir", "refroidissement.txt::p002"),
    ("À quelle fréquence la chaleur du moteur est-elle relevée ?",
     "temperature.txt::p001"),
]


def search_hybrid(question):
    bm25 = search_bm25(question, top_k=5)
    semantic = search_semantic(question, top_k=5)
    return fuse_rankings(bm25, semantic, top_k=3)


engines = {
    "BM25": lambda question: search_bm25(question, top_k=3),
    "Embeddings": lambda question: search_semantic(question, top_k=3),
    "Hybride RRF": search_hybrid,
}


if __name__ == "__main__":
    for name, search in engines.items():
        hits_at_1 = hits_at_3 = 0
        total_seconds = 0.0

        for question, expected_id in QUESTIONS:
            start = perf_counter()
            results = search(question)
            total_seconds += perf_counter() - start

            ids = [result["id"] for result in results]
            hits_at_1 += expected_id in ids[:1]
            hits_at_3 += expected_id in ids[:3]

            if expected_id not in ids[:1]:
                print(f"{name} | {question} | premier résultat = {ids[:1]}")

        count = len(QUESTIONS)
        print(
            f"{name} | Top 1 = {hits_at_1}/{count}"
            f" | Top 3 = {hits_at_3}/{count}"
            f" | Moyenne = {1000 * total_seconds / count:.1f} ms"
        )