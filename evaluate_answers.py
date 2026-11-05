from time import perf_counter

from answer_question import ABSTENTION, answer_question


cases = [
    (
        "À quelle fréquence la température du moteur est-elle relevée ?",
        "chaque minute",
        "temperature.txt::p001",
    ),
    (
        "Quelle est la vitesse du réseau Ethernet dans le moteur ?",
        None,
        None,
    ),
]

for question, expected_text, expected_source in cases:
    start = perf_counter()
    text, citations = answer_question(question)
    elapsed = perf_counter() - start

    if expected_text is None:
        passed = text == ABSTENTION and citations == []
    else:
        passed = (
            expected_text in text.casefold()
            and any(source == expected_source for source, _ in citations)
        )

    print(f"\n{'OK' if passed else 'À examiner'} | {elapsed:.2f} s")
    print(question)
    print(text)
    for source, excerpt in citations:
        print(f"[{source}] {excerpt}")
