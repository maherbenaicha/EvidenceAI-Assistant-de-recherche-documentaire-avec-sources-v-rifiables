from semantic_search import display_results, search_bm25, search_semantic


def fuse_rankings(*rankings, top_k=3):
    passages = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            passage_id = result["id"]
            passage = passages.setdefault(
                passage_id, {**result, "score": 0.0}
            )
            passage["score"] += 1 / (60 + rank)

    return sorted(
        passages.values(),
        key=lambda passage: passage["score"],
        reverse=True,
    )[:top_k]


if __name__ == "__main__":
    question = input("\nTa question : ")
    bm25 = search_bm25(question, top_k=5)
    semantic = search_semantic(question, top_k=5)

    display_results("BM25", bm25)
    display_results("Embeddings", semantic)
    display_results("Hybride RRF", fuse_rankings(bm25, semantic))