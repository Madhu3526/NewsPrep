# ===================== Updated Imports =====================
import os
import pickle
import random
import string
import traceback
from typing import List, Tuple

import numpy as np
import textdistance
import torch
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sense2vec import Sense2Vec
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from newsapi import NewsApiClient
from keybert import KeyBERT  # ✅ NEW

# ===================== Constants & Setup =====================
MODEL_PATHS = {
    'summary_model': 't5_summary_model.pkl',
    'summary_tokenizer': 't5_summary_tokenizer.pkl',
    'question_model': 't5_question_model.pkl',
    'question_tokenizer': 't5_question_tokenizer.pkl',
    'sentence_transformer': 'sentence_transformer_model.pkl'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== NewsAPI Integration =====================
NEWS_API_KEY = "f138d5525fb141199062ea1b105558a5"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def fetch_news_articles(query="India", language="en", page_size=10):
    articles = newsapi.get_everything(q=query, language=language, page_size=page_size)
    return [(a["title"], a["content"] or a["description"]) for a in articles["articles"] if a["content"]]

# ===================== MCQ Generator Class =====================
class MCAGenerator:
    def __init__(self):
        self.s2v = self._load_sense2vec()
        self.summary_model, self.summary_tokenizer = self._load_model('t5-base', 'summary')
        self.question_model, self.question_tokenizer = self._load_model('ramsrigouthamg/t5_squad_v1', 'question')
        self.sentence_transformer = self._load_sentence_transformer()
        self.keyword_model = KeyBERT(model="all-MiniLM-L6-v2")  # ✅ KeyBERT Init

        self.summary_model = self.summary_model.to(device)
        self.question_model = self.question_model.to(device)

    def _load_sense2vec(self) -> Sense2Vec:
        return Sense2Vec().from_disk('s2v_old')

    def _load_model(self, model_name: str, model_type: str) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
        model_path = MODEL_PATHS[f'{model_type}_model']
        tokenizer_path = MODEL_PATHS[f'{model_type}_tokenizer']

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
        else:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)

        return model, tokenizer

    def _load_sentence_transformer(self) -> SentenceTransformer:
        if os.path.exists(MODEL_PATHS['sentence_transformer']):
            with open(MODEL_PATHS['sentence_transformer'], 'rb') as f:
                return pickle.load(f)
        model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")
        with open(MODEL_PATHS['sentence_transformer'], 'wb') as f:
            pickle.dump(model, f)
        return model

    def summarize(self, text: str) -> str:
        text = "summarize: " + text.strip().replace("\n", " ")
        encoding = self.summary_tokenizer.encode_plus(
            text, max_length=512, truncation=True, return_tensors="pt"
        ).to(device)

        outs = self.summary_model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            early_stopping=True,
            num_beams=3,
            max_length=500
        )
        return self.summary_tokenizer.decode(outs[0], skip_special_tokens=True).strip()

    def extract_keyphrases(self, text: str) -> List[str]:
        keywords = self.keyword_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
        return [kw[0] for kw in keywords]

    def generate_question(self, context: str, answer: str) -> str:
        text = f"context: {context} answer: {answer}"
        encoding = self.question_tokenizer.encode_plus(
            text, max_length=384, truncation=True, return_tensors="pt"
        ).to(device)

        outs = self.question_model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            num_beams=5,
            max_length=72
        )
        return self.question_tokenizer.decode(outs[0], skip_special_tokens=True).replace("question:", "").strip()

    def get_distractors(self, word: str, keywords: List[str]) -> List[str]:
        distractors = []

        try:
            sense = self.s2v.get_best_sense(word)
            similar = self.s2v.most_similar(sense, n=20)
            distractors = [s[0].split("|")[0].replace("_", " ").lower() for s in similar]
        except:
            pass

        if len(distractors) < 3:
            synsets = wn.synsets(word.lower(), 'n')
            if synsets:
                for hypernym in synsets[0].hypernyms():
                    for hyponym in hypernym.hyponyms():
                        name = hyponym.lemmas()[0].name().lower()
                        if name != word.lower():
                            distractors.append(name)

        if len(distractors) < 3:
            candidates = [kw.lower() for kw in keywords if kw.lower() != word.lower()]
            random.shuffle(candidates)
            distractors.extend(candidates[:3])

        distractors = list(set([d.capitalize() if random.random() < 0.5 else d.lower() for d in distractors
                                if d.lower() != word.lower()
                                and textdistance.levenshtein.normalized_similarity(d, word) < 0.8]))[:3]

        return distractors

    def generate_mca_questions(self, context: str) -> List[Tuple[str, List[str], str]]:
        summarized = self.summarize(context)
        keywords = self.extract_keyphrases(context)
        outputs = []

        for answer in keywords:
            try:
                question = self.generate_question(context, answer)
                if not question.endswith("?"):
                    question += "?"

                distractors = self.get_distractors(answer, keywords)
                answer_formatted = answer.lower()

                if len(distractors) < 3:
                    more = [kw.lower() for kw in keywords if kw.lower() != answer_formatted]
                    distractors += more[:3 - len(distractors)]

                distractors = distractors[:3]
                options = distractors + [answer_formatted]
                random.shuffle(options)

                outputs.append((question, options, answer_formatted))
            except Exception:
                continue

        return outputs



# ===================== Visualization Functions =====================

def plot_word_cloud(text: str):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

def plot_quiz_performance(user_answers, correct_answers):
    
    correct = sum([1 for ans, correct_ans in zip(user_answers, correct_answers) if ans == correct_ans])
    incorrect = len(user_answers) - correct

    labels = ['Correct', 'Incorrect']
    counts = [correct, incorrect]

    fig, ax = plt.subplots()
    ax.bar(labels, counts, color=['green', 'red'])

    ax.set_ylabel('Number of Questions')
    ax.set_title('Quiz Performance')
    st.pyplot(fig)

# ===================== Streamlit App UI =====================

st.set_page_config(page_title="NewsPrep Quiz Generator", layout="wide")

st.title("🧠 NewsPrep - UPSC Quiz Generator (Live News Edition)")

news_options = fetch_news_articles(query="UPSC OR India OR Economy OR Science", page_size=10)

if news_options:
    titles = [title for title, _ in news_options]
    selected_title = st.selectbox("🗞️ Choose a news article:", titles)
    selected_text = dict(news_options)[selected_title]
else:
    st.warning("No news articles available right now.")
    selected_text = ""

if "questions" not in st.session_state:
    st.session_state.questions = []
    st.session_state.summary = ""

if st.button("Generate Summary and MCQs") and selected_text.strip():
    with st.spinner("Generating summary and quiz..."):
        generator = MCAGenerator()
        st.session_state.summary = generator.summarize(selected_text)
        st.session_state.questions = generator.generate_mca_questions(selected_text)
        plot_word_cloud(st.session_state.summary)

if st.session_state.summary:
    st.subheader("📝 Summary")
    st.write(st.session_state.summary)

if st.session_state.questions:
    st.subheader("🧠 Multiple-Choice Questions")
    score = 0
    user_answers = []

    for i, (question, options, correct_answer) in enumerate(st.session_state.questions):
        st.write(f"**Q{i+1}:** {question}")
        selected = st.radio("Choose your answer:", [f"(a) {options[0]}", f"(b) {options[1]}", f"(c) {options[2]}", f"(d) {options[3]}"], key=f"q{i}")
        user_answers.append((selected, options, correct_answer))

    if st.button("Submit Answers"):
        for i, (selected, options, correct_answer) in enumerate(user_answers):
            selected_option = selected.split(") ")[1]
            if selected_option == correct_answer:
                score += 1

        plot_quiz_performance([ans[0].split(") ")[1] for ans in user_answers], [correct_answer for _, _, correct_answer in st.session_state.questions])
        st.success(f"✅ Your Score: {score}/{len(user_answers)}")

        for i, (question, options, correct_answer) in enumerate(st.session_state.questions):
            correct_label = chr(97 + options.index(correct_answer))
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"✅ Correct Answer: ({correct_label}) {correct_answer}")
