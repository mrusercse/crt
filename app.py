import streamlit as st
import pdfplumber
import docx2txt
import tempfile
import json
import re
import csv
import os
import time
import io
import pickle
from transformers import pipeline
import torch
import gdown
import numpy as np
import nltk
import re
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from streamlit.components.v1 import html


nltk.download('wordnet')
# ---- Page Setup ----
st.set_page_config(page_title="Contract Analyzer", layout="wide")
st.title("Contract Analyzer")

output_dir = "contract-analyzer/outputs"
os.makedirs(output_dir, exist_ok=True)



CLAUSE_KEYWORDS = {
    "Confidentiality": [
        "confidential", "non-disclosure", "nda", "keep secret", "confidential information",
        "maintain confidentiality", "confidentiality obligations", "privileged information"
    ],
    "Termination": [
        "termination", "terminate", "end of agreement", "may be terminated", "breach", 
        "notice of termination", "termination clause", "cancel", "terminate this contract"
    ],
    "Liability": [
        "liable", "liability", "responsibility", "shall be responsible", "held responsible",
        "limited liability", "liability cap", "excluded liability", "damages", "losses"
    ],
    "Indemnification": [
        "indemnify", "indemnification", "hold harmless", "indemnity", "indemnified against",
        "defend and indemnify", "indemnification obligations", "liabilities and expenses"
    ],
    "Governing Law": [
        "jurisdiction", "governing law", "court of", "laws of", "subject to the laws",
        "under the laws", "venue shall be", "exclusive jurisdiction", "applicable law",
        "legal jurisdiction", "governed by", "governed under"
    ],
    "Force Majeure": [
        "force majeure", "act of god", "natural disasters", "beyond reasonable control",
        "unforeseeable circumstances", "war", "riot", "strike", "earthquake", "epidemic"
    ],
    "Dispute Resolution": [
        "dispute resolution", "settlement", "arbitration", "mediator", "mediation", 
        "resolve disputes", "legal dispute", "arbitration panel", "binding arbitration"
    ],
    "Payment Terms": [
        "payment", "remuneration", "invoice", "fee", "compensation", "payable", "due date",
        "late fee", "payment schedule", "billing"
    ],
    "Intellectual Property": [
        "intellectual property", "ip rights", "copyright", "patent", "trademark", 
        "ownership of ideas", "retain all rights", "proprietary", "license"
    ],
    "Non-Compete": [
        "non-compete", "restrictive covenant", "competition", "not compete", "exclusive dealing",
        "not engage", "prohibited from engaging", "no competing products"
    ]
}



def expand_keywords_with_synonyms(keywords):
    expanded = set(keywords)
    for word in keywords:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " ").lower())
    return list(expanded)

# Expand keywords with synonyms (run once)
EXPANDED_CLAUSE_KEYWORDS = {
    label: expand_keywords_with_synonyms(keywords)
    for label, keywords in CLAUSE_KEYWORDS.items()
}

def detect_clause_type_advanced(clause, threshold=1):
    clause = clause.lower()
    words = word_tokenize(clause)
    scores = defaultdict(int)

    for label, keywords in EXPANDED_CLAUSE_KEYWORDS.items():
        for kw in keywords:
            pattern = re.escape(kw)
            matches = re.findall(rf'\b{pattern}\b', clause)
            scores[label] += len(matches)

    if not scores:
        return "Other"

    best_match = max(scores.items(), key=lambda x: x[1])
    
    # Return best label only if it has enough weight
    return best_match[0] if best_match[1] >= threshold else "Other"

def analyze_clauses_inline(clauses):
    return [
        {
            "clause": clause.strip(),
            "clause_type": detect_clause_type_advanced(clause)
        }
        for clause in clauses if clause.strip()
    ]


def download_model_from_gdrive():
    os.makedirs("outputs", exist_ok=True)

    files = {
        "outputs/clause_classifier.pkl": "1MLx7XvNmYsbIgsg51kFW52SewJ2GLu9N",
        "outputs/vectorizer.pkl": "1XLJyNjnr6N-9b-FKNygOianwqph0syuV"
    }

    for path, file_id in files.items():
        if not os.path.exists(path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)

def load_classifier_model():
    download_model_from_gdrive()
    with open("outputs/clause_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("outputs/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer



def classify_clauses_ml(clauses):
    model, vectorizer = load_classifier_model()
    X_vec = vectorizer.transform(clauses)
    return model.predict(X_vec)





nltk.download("punkt")

# ---- Summarization ----
@st.cache_resource
def load_summarizer():
   
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")

summarizer = load_summarizer()

# ---- Sentence-aware Truncation ----
def truncate_text_by_sentences(text, max_words=800):
    sentences = sent_tokenize(text)
    result = []
    word_count = 0
    for sent in sentences:
        words = sent.split()
        if word_count + len(words) > max_words:
            break
        result.append(sent)
        word_count += len(words)
    return " ".join(result)

# ---- Punctuation Fix ----
def fix_summary_punctuation(text):
    if text and not re.search(r'[.!?]$', text.strip()):
        return text.strip() + "."
    return text.strip()

# ---- Summarization Wrapper ----
def summarize_text_bart(text, max_length=130):
    if not text or len(text.split()) < 40:
        return text.strip()

    text = truncate_text_by_sentences(text)

    dynamic_min_length = min(30, max(10, len(text.split()) // 2))  # adaptive min_length

    try:
        result = summarizer(
            text,
            max_length=max_length,
            min_length=dynamic_min_length,
            do_sample=False
        )
        if result and isinstance(result, list) and "summary_text" in result[0]:
            return fix_summary_punctuation(result[0]["summary_text"])
        else:
            print(f"[!] Empty or invalid summarization result for text:\n{text[:100]}...")
            return text
    except Exception as e:
        print(f"[!] Summarization failed:\n{text[:100]}...\nError: {e}")
        return text

# ---- Summarize All Clauses ----
def summarize_clauses(analyzed):
    all_text = " ".join([c["clause"] for c in analyzed])
    full_summary = summarize_text_bart(all_text, max_length=200)

    clause_summaries = []
    for item in analyzed:
        summary = summarize_text_bart(item["clause"], max_length=80)
        clause_summaries.append({
            "clause": item["clause"],
            "clause_type": item["clause_type"],
            "risk_label": item.get("risk_label", ""),
            "summary": summary
        })

    return full_summary, clause_summaries




# ---- Text Extraction ----
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Clean up extra spaces, headers/footers
                lines = page_text.strip().split('\n')
                cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
                text += "\n".join(cleaned_lines) + "\n"
    return text.strip()

def extract_text_from_docx(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    text = docx2txt.process(tmp_path).strip()
    os.remove(tmp_path)
    # Remove multiple blank lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text

# ---- Clause Splitting ----

def split_pdf_clauses(text):
    """
    Improved clause splitter: Supports numbered sections, "Section", "Article", etc.
    """
    # Match: 1., 1.1, Section 2, Article III, Clause 4, etc.
    pattern = re.compile(
        r'(?=(?:\n|^)(?:\d+(\.\d+)*|Section\s+\d+|Article\s+[IVXLC]+|Clause\s+\d+)[\.\):\s])',
        re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    clauses = []
    if not matches:
        return [text.strip()]

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause = text[start:end].strip()
        if len(clause) > 30:
            clauses.append(clause)
    return clauses

def split_docx_clauses(text):
    """
    Advanced DOCX clause splitter: uses section titles, and filters meaningful paragraphs.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    clauses = []
    buffer = []

    section_pattern = re.compile(r'^(?:\d+(\.\d+)*|Section \d+|Article [IVXLC]+|Clause \d+)[\.\):\s]', re.IGNORECASE)

    for line in lines:
        if section_pattern.match(line):
            if buffer:
                full_clause = " ".join(buffer).strip()
                if len(full_clause) > 30:
                    clauses.append(full_clause)
                buffer = []
        buffer.append(line)

    if buffer:
        full_clause = " ".join(buffer).strip()
        if len(full_clause) > 30:
            clauses.append(full_clause)

    return clauses


def parse_uploaded_document(file, filetype):
    if filetype == "pdf":
        text = extract_text_from_pdf(file)
        clauses = split_pdf_clauses(text)
    elif filetype == "docx":
        text = extract_text_from_docx(file)
        clauses = split_docx_clauses(text)
    else:
        raise ValueError("Unsupported file format.")
    return text, clauses



# ---- File Upload ----
uploaded_file = st.file_uploader("Upload a contract (.pdf or .docx)", type=["pdf", "docx"])




def split_txt_clauses(text):
    return [para.strip() for para in text.split('\n') if len(para.strip()) > 30]


# ---- Save Outputs ----
def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



if uploaded_file:
    name_without_ext = os.path.splitext(uploaded_file.name)[0].lower()
    filetype = uploaded_file.name.split(".")[-1].lower()

    # Clear session_state if file changes
    if "last_filename" not in st.session_state or st.session_state["last_filename"] != uploaded_file.name:
        st.session_state.clear()
        st.session_state["last_filename"] = uploaded_file.name


    if "analyzed" not in st.session_state:
        with st.spinner("Parsing and analyzing document..."):
            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                """
                <div style="text-align:center;">
                    <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiwNv2lENjUn4-lBWVVFT1zohQCWu_S9HvYQoztSNZytFd2h00HXv3r7Dm0Np4H1CkVR25Z_uBM3YUDkT_gjQIMhvksc-jzhX5lPDY80Oo-b-R5K3y_jITyEDgiOompkogEtqqWigAUyZbM06sQROmagVmXX6E0uata1yO_5rnEe4NHe_-wGjSEQJ8xhyphenhyphenwF/s1600/Contract.gif" width="800"/>
                    <p style="font-size:18px;">Analyzing your contract... Please wait!</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            full_text, clauses = parse_uploaded_document(uploaded_file, filetype)
            analyzed = analyze_clauses_inline(clauses)

            ml_preds = classify_clauses_ml([c["clause"] for c in analyzed])
            for i in range(len(analyzed)):
                analyzed[i]["risk_label"] = ml_preds[i]

            full_summary, clause_summaries = summarize_clauses(analyzed)

            # Cleanup loader
            loading_placeholder.empty()

            st.session_state["analyzed"] = analyzed
            st.session_state["full_summary"] = full_summary
            st.session_state["clause_summaries"] = clause_summaries

            # Save outputs
            save_json(analyzed, os.path.join(output_dir, f"classified_{name_without_ext}.json"))
            

    analyzed = st.session_state["analyzed"]
    full_summary = st.session_state["full_summary"]
    clause_summaries = st.session_state["clause_summaries"]

    # ---- Display Summary ----
    st.subheader("Contract Summary")
    st.markdown("Full Summary")

    st.info(full_summary)

    html(f"""
        <div style="margin-top:-5px;color:white;background-color:black;">
            <button onclick="speakSummary(`{full_summary}`)" style="margin-right:10px;color:white;background-color:black;border:2px solid white">🔊 Speak</button>
            <button onclick="stopSpeech()" style="color:white;background-color:black;border:2px solid white">⏹ Stop</button>
        </div>

        <script>
            function speakSummary(text) {{
                var utterance = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            }}
            function stopSpeech() {{
                window.speechSynthesis.cancel();
            }}
        </script>
    """, height=50)

    st.markdown("Clause Summaries")

    # ---- Risk Filter ----
    risk_levels = list(set(item['risk_label'] for item in clause_summaries))
    risk_levels.sort()
    selected_risk_levels = st.multiselect("Filter by Risk Level", options=risk_levels, default=risk_levels)

    filtered_clauses = [item for item in clause_summaries if item['risk_label'] in selected_risk_levels]

    if not filtered_clauses:
        st.warning("No clauses match the selected risk level(s).")
    else:


        for i, item in enumerate(filtered_clauses, 1):
            with st.expander(f"Clause Type {i}: {item['clause_type']} | Risk Level: {item['risk_label']}"):
                st.markdown(f"**Original Clause:**\n{item['clause']}")
                st.markdown(f"**Summary:**\n{item['summary']}")

                html(f"""
                    <div style="margin-top:-5px;color:white;background-color:black;">
                        <button onclick="speakClause{i}()" style="color:white;background-color:black;border:2px solid white">🔊 Speak Summary</button>
                        <button onclick="stopSpeech()" style="color:white;background-color:black;border:2px solid white">⏹ Stop</button>
                    </div>

                    <script>
                        function speakClause{i}() {{
                            var utterance = new SpeechSynthesisUtterance(`{item['summary']}`);
                            window.speechSynthesis.cancel();
                            window.speechSynthesis.speak(utterance);
                        }}
                        function stopSpeech() {{
                            window.speechSynthesis.cancel();
                        }}
                    </script>
                """, height=50)

    # ---- Download Buttons ----
    st.download_button(
        label="Download Results as JSON",
        data=json.dumps(analyzed, indent=4, ensure_ascii=False),
        file_name=f"classified_{name_without_ext}.json",
        mime="application/json"
    )

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["clause_number", "clause_text", "clause_type"])
    for i, item in enumerate(analyzed, start=1):
        writer.writerow([i, item["clause"], item["clause_type"]])

    

    summary_json = {
        "full_summary": full_summary,
        "clause_summaries": clause_summaries
    }

    st.download_button(
        label="Download Summary as JSON",
        data=json.dumps(summary_json, indent=4, ensure_ascii=False),
        file_name=f"summarized_{name_without_ext}.json",
        mime="application/json"
    )
    

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# ---- Download Labeled Clauses JSON if Not Exists ----
def download_labeled_clauses():
    os.makedirs("data", exist_ok=True)
    file_path = "data/labeled_clauses.json"
    if not os.path.exists(file_path):
        file_id = "1gmfFmTQW8UgDn0Bjnl31Rhli3JVXiKAt"  # Replace with your actual file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)
    return file_path

with st.expander("📊 Model Performance Metrics (click to expand)", expanded=False):
    try:
        json_path = download_labeled_clauses()
        with open(json_path, "r", encoding="utf-8") as f:
            labeled_data = json.load(f)

        df = pd.DataFrame(labeled_data)
        y_true = df["label"]

        model, vectorizer = load_classifier_model()
        X_test = vectorizer.transform(df["clause"])
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        st.markdown(f"**Accuracy:** `{accuracy:.2f}`")

        # Classification Report as Table
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().round(2)
        st.markdown("**Classification Report:**")
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=model.classes_,
                    yticklabels=model.classes_,
                    ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error showing model performance: {e}")
