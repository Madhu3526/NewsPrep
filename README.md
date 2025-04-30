NewsPrep: UPSC Quiz Generator with Live News & Fine-Tuned Transformers
NewsPrep is an intelligent UPSC-prep assistant that fetches live news articles, summarizes them, and generates context-aware multiple-choice questions (MCQs). Built with fine-tuned transformer models (BART for summarization and T5 for question generation), it helps aspirants reinforce current affairs knowledge with quizzes curated from real-time news.

🚀 Features
    🔍 Fetches latest news using NewsAPI
    📝 Summarizes articles using fine-tuned BART
    ❓ Generates MCQs using fine-tuned T5 model
    🧠 Creates meaningful distractors using Sense2Vec, WordNet, and semantic filtering
    📊 Visualizes quiz performance and word cloud
    ⚙️ Fully interactive Streamlit UI

🔧 Tech Stack
    Frontend/UI: Streamlit
    Backend: Python (Transformers, NLTK, KeyBERT, Sense2Vec)
    Models: bart-finetuned-newss (local) – Summarization 
            t5_qg_finetuned (local) – Question Generation
    News Source: NewsAPI

