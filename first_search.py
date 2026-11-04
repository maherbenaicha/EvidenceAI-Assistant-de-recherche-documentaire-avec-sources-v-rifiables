import re
from pathlib import Path
from rank_bm25 import BM25Okapi

def load_documents(folder):
    documents = []

    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8-sig").strip()

        if not text:
            continue

        documents.append({
            "id": path.name,
            "source": path.name,
            "text": text,
        })

    return documents


def split_into_passages(documents):
    passages = []

    for document in documents:
        paragraphs = re.split(r"\n\s*\n", document["text"])

        for number, paragraph in enumerate(paragraphs, start=1):
            # Réunir les lignes d’un même paragraphe.
            text = " ".join(paragraph.split())

            if not text:
                continue

            passages.append({
                "id": f"{document['id']}::p{number:03d}",
                "document_id": document["id"],
                "source": document["source"],
                "text": text,
            })

    return passages
def tokenize(text):
    """Passer en minuscules et extraire les mots."""
    return re.findall(r"\b\w+\b", text.lower())


PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "data" / "documents"

source_documents = load_documents(DOCUMENTS_DIR)
documents = split_into_passages(source_documents)

if not documents:
    raise ValueError(
        f"Aucun passage trouvé. Ajoute des fichiers .txt dans {DOCUMENTS_DIR}"
    )

tokenized_documents = [
    tokenize(document["text"])
    for document in documents
]

search_engine = BM25Okapi(tokenized_documents)

print(
    f"{len(source_documents)} documents chargés, "
    f"{len(documents)} passages indexés."
)

def search(question, top_k=3):
    tokens = tokenize(question)

    if not tokens:
        return []

    # Garder uniquement les documents partageant au moins
    # un mot avec la question.
    query_words = set(tokens)

    candidate_indices = [
        index
        for index, document_tokens in enumerate(tokenized_documents)
        if query_words.intersection(document_tokens)
    ]

    if not candidate_indices:
        return []

    scores = search_engine.get_scores(tokens)

    ranked_indices = sorted(
        candidate_indices,
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices[:top_k]:
        document = documents[index]

        results.append({
            "id": document["id"],
            "source": document["source"],
            "text": document["text"],
            "score": float(scores[index]),
        })

    return results


if __name__ == "__main__":
    question = input("Ta question : ")
    results = search(question, top_k=2)

    if not results:
        print("Aucun passage ne partage de mots avec la question.")
    else:
        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. [{result['id']}] {result['source']} "
                f"— score : {result['score']:.3f}"
            )
            print(result["text"])