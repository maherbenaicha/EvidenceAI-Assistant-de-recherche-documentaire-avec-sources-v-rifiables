import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import APIError

st.set_page_config(page_title="EvidenceAI", page_icon="🔎")
st.title("EvidenceAI")
st.write("Pose une question sur les documents du projet.")

load_dotenv(Path(__file__).with_name(".env"))

if not os.getenv("GROQ_API_KEY"):
    st.error("Ajoute GROQ_API_KEY dans le fichier .env du projet.")
    st.stop()

from answer_question import answer_question

with st.form("recherche"):
    question = st.text_input("Ta question")
    submitted = st.form_submit_button("Chercher une réponse")

if submitted:
    if not question.strip():
        st.warning("Saisis une question.")
    else:
        try:
            with st.spinner("Recherche des passages et génération…"):
                texte, citations = answer_question(question)
        except APIError:
            st.error("L’appel à Groq a échoué. Vérifie ta clé et ton accès au modèle.")
        else:
            st.subheader("Réponse")
            st.text(texte)

            if citations:
                st.subheader("Sources")
                for passage_id, extrait in citations:
                    with st.expander(passage_id):
                        st.text(extrait)