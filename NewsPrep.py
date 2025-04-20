# ===================== Imports =====================
import os
import random
import traceback
from typing import List, Tuple

import numpy as np
import torch
import textdistance
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartTokenizer, BartForConditionalGeneration
from sense2vec import Sense2Vec
from keybert import KeyBERT
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from newsapi import NewsApiClient

# ===================== Setup =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEWS_API_KEY = "f138d5525fb141199062ea1b105558a5"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# ===================== News Fetcher =====================
def fetch_news_articles(query="India", language="en", page_size=10):
    articles = newsapi.get_everything(q=query, language=language, page_size=page_size)
    return [(a["title"], a["content"] or a["description"]) for a in articles["articles"] if a["content"]]

# ===================== MCQ Generator Class =====================
class MCAGenerator:
    def __init__(self):
        # Initialize Sense2Vec
        self.s2v = Sense2Vec().from_disk('s2v_old')
        
        # Initialize BART model and tokenizer
        self.bart_tokenizer = BartTokenizer.from_pretrained('bart-finetuned-newss')
        self.bart_model = BartForConditionalGeneration.from_pretrained('bart-finetuned-newss').to(device)
        
        # Initialize Local Fine-Tuned T5 Model for Question Generation
        model_path = r"C:\Users\HP\Desktop\NewsPrep\t5_qg_finetuned"
        
        # Load the tokenizer and model from the local directory and save as instance variables
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        
        # Initialize other models
        self.sentence_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v2')
        self.keyword_model = KeyBERT(model="all-MiniLM-L6-v2")

    def summarize(self, text: str) -> str:
        inputs = self.bart_tokenizer([text], return_tensors='pt', max_length=1024, truncation=True).to(device)
        summary_ids = self.bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150)
        return self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def extract_keyphrases(self, text: str) -> List[str]:
        keywords = self.keyword_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
        return [kw[0] for kw in keywords]

    def generate_question(self, context: str, answer: str) -> str:
        input_text = f"context: {context} answer: {answer}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=384, truncation=True).to(device)
        output = self.model.generate(inputs["input_ids"], num_beams=5, max_length=72)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).replace("question:", "").strip()

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
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().lower()
                    if name != word.lower():
                        distractors.append(name)

        if len(distractors) < 3:
            candidates = [kw.lower() for kw in keywords if kw.lower() != word.lower()]
            random.shuffle(candidates)
            distractors.extend(candidates[:3])

        distractors = list(set([d for d in distractors if d != word.lower()
                                and textdistance.levenshtein.normalized_similarity(d, word) < 0.8]))[:3]
        return distractors

    def generate_mca_questions(self, context: str) -> List[Tuple[str, List[str], str]]:
        summary = self.summarize(context)
        keywords = self.extract_keyphrases(context)
        outputs = []

        for answer in keywords:
            try:
                question = self.generate_question(context, answer)
                if not question.endswith("?"):
                    question += "?"
                distractors = self.get_distractors(answer, keywords)

                options = distractors + [answer.lower()]
                random.shuffle(options)

                outputs.append((question, options, answer.lower()))
            except Exception as e:
                print("Error generating question:", e)
                traceback.print_exc()
                continue

        return outputs


# ===================== Visualization =====================
def plot_word_cloud(text: str):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

def plot_quiz_performance(user_answers, correct_answers):
    correct = sum([1 for u, c in zip(user_answers, correct_answers) if u.lower() == c.lower()])
    incorrect = len(user_answers) - correct

    fig, ax = plt.subplots()
    ax.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
    ax.set_ylabel('Number of Questions')
    ax.set_title('Quiz Performance')
    st.pyplot(fig)

# ===================== Streamlit UI =====================
st.set_page_config(page_title="NewsPrep Quiz Generator", layout="wide")
st.title("🧠 NewsPrep - UPSC Quiz Generator (Live News Edition)")

news_options = fetch_news_articles(query="UPSC OR India OR Economy OR Science", page_size=10)

if news_options:
    titles = [title for title, _ in news_options]
    selected_title = st.selectbox("🗞️ Choose a news article:", titles)
    selected_text = dict(news_options)[selected_title]
else:
    st.warning("No news articles available.")
    selected_text = ""

if "questions" not in st.session_state:
    st.session_state.questions = []
    st.session_state.summary = ""

if st.button("Generate Summary and MCQs") and selected_text.strip():
    with st.spinner("Generating..."):
        generator = MCAGenerator()
        st.session_state.summary = generator.summarize(selected_text)
        st.session_state.questions = generator.generate_mca_questions(selected_text)
        plot_word_cloud(st.session_state.summary)

if st.session_state.summary:
    st.subheader("📝 Summary")
    st.write(st.session_state.summary)

if st.session_state.questions:
    st.subheader("🧠 Quiz")
    user_selected = []

    for i, (question, options, correct) in enumerate(st.session_state.questions):
        st.write(f"**Q{i+1}:** {question}")
        choice = st.radio("Choose your answer:", [f"(a) {options[0]}", f"(b) {options[1]}", f"(c) {options[2]}", f"(d) {options[3]}"], key=f"q{i}")
        user_selected.append(choice.split(") ")[1].strip())

    if st.button("Submit Answers"):
        correct_answers = [correct for _, _, correct in st.session_state.questions]
        plot_quiz_performance(user_selected, correct_answers)
        score = sum([1 for u, c in zip(user_selected, correct_answers) if u.lower() == c.lower()])
        st.success(f"✅ Your Score: {score}/{len(user_selected)}")

        st.subheader("📌 Correct Answers")
        for i, (q, opts, correct) in enumerate(st.session_state.questions):
            correct_label = chr(97 + opts.index(correct))
            st.write(f"**Q{i+1}:** {q}")
            st.write(f"✅ Correct Answer: ({correct_label}) {correct}")
