import streamlit as st
import pdfplumber
import docx2txt
import tempfile
import json
import re
import os
import pickle
import gdown
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from collections import defaultdict
from streamlit.components.v1 import html
from nltk.data import find

# ---- Downloads ----
@st.cache_resource
def download_nltk_resources():
    for resource in ["punkt", "wordnet"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# ---- Page Setup ----
st.set_page_config(page_title="Contract Analyzer", layout="wide")
st.title("Contract Analyzer")

# ---- Clause Keywords ----
CLAUSE_KEYWORDS = {
    "Confidentiality": ["confidential", "non-disclosure", "nda", "keep secret"],
    "Termination": ["termination", "terminate", "breach", "notice of termination"],
    "Liability": ["liable", "liability", "responsibility", "damages"],
    "Indemnification": ["indemnify", "hold harmless", "indemnity"],
    "Governing Law": ["jurisdiction", "governing law", "laws of"],
    "Force Majeure": ["force majeure", "act of god", "natural disasters"],
    "Dispute Resolution": ["dispute", "arbitration", "mediator", "settlement"],
    "Payment Terms": ["payment", "invoice", "fee", "billing"],
    "Intellectual Property": ["intellectual property", "copyright", "patent"],
    "Non-Compete": ["non-compete", "exclusive dealing", "not compete"]
}

def expand_keywords_with_synonyms(keywords):
    expanded = set(keywords)
    for word in keywords:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " ").lower())
    return list(expanded)

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
    return best_match[0] if best_match[1] >= threshold else "Other"

def analyze_clauses_inline(clauses):
    return [
        {"clause": clause.strip(), "clause_type": detect_clause_type_advanced(clause)}
        for clause in clauses if clause.strip()
    ]

@st.cache_resource
def load_summarizer():
    # return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
    return pipeline("summarization", model="knkarthick/MEETING_SUMMARY", framework="pt")



def truncate_text_by_sentences(text, max_words=800):
    sentences = sent_tokenize(text)
    result, word_count = [], 0
    for sent in sentences:
        words = sent.split()
        if word_count + len(words) > max_words:
            break
        result.append(sent)
        word_count += len(words)
    return " ".join(result)

def fix_summary_punctuation(text):
    return text.strip() + "." if text and not re.search(r'[.!?]$', text.strip()) else text.strip()

def summarize_text_bart(text, max_length=130):
    if not text or len(text.split()) < 40:
        return text.strip()
    text = truncate_text_by_sentences(text)
    try:
        result = summarizer(text, max_length=max_length, min_length=min(30, max(10, len(text.split()) // 2)), do_sample=False)
        return fix_summary_punctuation(result[0]["summary_text"]) if result else text
    except Exception:
        return text

def summarize_clauses(analyzed):
    all_text = " ".join([c["clause"] for c in analyzed])
    full_summary = summarize_text_bart(all_text, max_length=200)
    clause_summaries = [
        {
            "clause": item["clause"],
            "clause_type": item["clause_type"],
            "risk_label": item.get("risk_label", ""),
            "summary": summarize_text_bart(item["clause"], max_length=80)
        } for item in analyzed
    ]
    return full_summary, clause_summaries

# ---- Model Loader ----
@st.cache_resource
def load_classifier_model():
    file_ids = {
        "clause_classifier": "1MLx7XvNmYsbIgsg51kFW52SewJ2GLu9N",
        "vectorizer": "1XLJyNjnr6N-9b-FKNygOianwqph0syuV"
    }
    models = {}
    for key, fid in file_ids.items():
        url = f"https://drive.google.com/uc?id={fid}"
        output = gdown.download(url, quiet=True, fuzzy=True)
        with open(output, "rb") as f:
            models[key] = pickle.load(f)
        os.remove(output)
    return models["clause_classifier"], models["vectorizer"]

def classify_clauses_ml(clauses):
    model, vectorizer = load_classifier_model()
    return model.predict(vectorizer.transform(clauses))

# ---- Upload ----
uploaded_file = st.file_uploader("Upload a contract (.pdf or .docx)", type=["pdf", "docx"])

# ---- Text Extraction ----
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                lines = content.strip().split("\n")
                text += "\n".join([line.strip() for line in lines if len(line.strip()) > 10]) + "\n"
    return text.strip()

def extract_text_from_docx(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    text = docx2txt.process(tmp_path).strip()
    os.remove(tmp_path)
    return re.sub(r'\n\s*\n', '\n\n', text)

def split_pdf_clauses(text):
    pattern = re.compile(r'(?=(?:\n|^)(?:\d+(\.\d+)*|Section\s+\d+|Article\s+[IVXLC]+|Clause\s+\d+)[\.\):\s])', re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [text.strip()]
    return [text[matches[i].start():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
            for i in range(len(matches)) if len(text[matches[i].start():].strip()) > 30]

def split_docx_clauses(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    buffer, clauses = [], []
    section_pattern = re.compile(r'^(?:\d+(\.\d+)*|Section \d+|Article [IVXLC]+|Clause \d+)[\.\):\s]', re.IGNORECASE)
    for line in lines:
        if section_pattern.match(line) and buffer:
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
        raise ValueError("Unsupported format")
    return text, clauses

# ---- Process Uploaded File ----
if uploaded_file:
    name_without_ext = os.path.splitext(uploaded_file.name)[0].lower()
    filetype = uploaded_file.name.split(".")[-1].lower()

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
            <button onclick="speakSummary(`{full_summary}`)" style="color:white;background:black;border:2px solid white">🔊 Speak</button>
            <button onclick="stopSpeech()" style="color:white;background:black;border:2px solid white">⏹ Stop</button>
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
    selected_risk_levels = st.multiselect("Filter by Risk Level", options=sorted(set(i["risk_label"] for i in clause_summaries)), default=None)

    for i, item in enumerate(clause_summaries):
        if selected_risk_levels and item["risk_label"] not in selected_risk_levels:
            continue
        with st.expander(f"Clause Type {i+1}: {item['clause_type']} | Risk Level: {item['risk_label']}"):
            st.markdown(f"**Original Clause:**\n{item['clause']}")
            st.markdown(f"**Summary:**\n{item['summary']}")
            html(f"""
                <div>
                    <button onclick="speakClause{i}()" style="color:white;background:black;border:2px solid white">🔊 Speak Summary</button>
                    <button onclick="stopSpeech()" style="color:white;background:black;border:2px solid white">⏹ Stop</button>
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

    # ---- Downloads ----
    st.download_button("Download Results (JSON)", json.dumps(analyzed, indent=4, ensure_ascii=False), file_name=f"classified_{name_without_ext}.json", mime="application/json")
    st.download_button("Download Summary (JSON)", json.dumps({"full_summary": full_summary, "clause_summaries": clause_summaries}, indent=4, ensure_ascii=False), file_name=f"summarized_{name_without_ext}.json", mime="application/json")


