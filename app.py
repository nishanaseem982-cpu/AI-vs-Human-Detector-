import html
import json
import re
import statistics
from collections import Counter

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# CONSTANTS
# ============================================================

MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"
MAX_TOKENS = 512
BATCH_SIZE = 16

AI_BUZZWORDS = [
    "delve", "tapestry", "moreover", "testament", "foster", "transformative",
    "multi-faceted", "landscape", "pivotal", "in conclusion", "furthermore",
]
# Longest first so multi-word phrases win over single words.
BUZZWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(AI_BUZZWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

EXAMPLES = {
    "Formal sample": (
        "In today's rapidly evolving digital landscape, artificial intelligence stands as a "
        "testament to human ingenuity. Moreover, it plays a pivotal role in shaping how "
        "organizations foster innovation. Furthermore, its transformative potential continues "
        "to expand across a multi-faceted range of industries. In conclusion, it is essential "
        "to delve deeper into both its benefits and its challenges."
    ),
    "Casual sample": (
        "So I finally fixed the leak under the kitchen sink. Took three trips to the hardware "
        "shop. Three! The first washer was the wrong size, the second one I dropped down the "
        "drain like an idiot, and by the third trip the guy at the counter just laughed at me. "
        "Anyway, it's dry now. My back hurts and I smell like old pipe, but it's dry."
    ),
}

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1100px; padding-top: 2.5rem;}
      .hero h1 {margin-bottom: 0.2rem;}
      .hero p {opacity: 0.75; font-size: 1.05rem; margin-top: 0;}
      .verdict {
          border-left: 6px solid var(--c); background: var(--bg);
          border-radius: 10px; padding: 1.1rem 1.4rem; margin: 0.5rem 0 1rem 0;
      }
      .verdict .label {font-size: 0.85rem; text-transform: uppercase;
          letter-spacing: 0.08em; opacity: 0.7;}
      .verdict .value {font-size: 1.7rem; font-weight: 700; color: var(--c);}
      .verdict .note {opacity: 0.8; margin-top: 0.2rem;}
      .highlight-box {line-height: 1.8; padding: 1rem 1.2rem; border-radius: 10px;
          border: 1px solid rgba(128,128,128,0.3); max-height: 420px; overflow-y: auto;}
      .highlight-box mark {background: rgba(255, 193, 7, 0.45); color: inherit;
          padding: 0.1rem 0.25rem; border-radius: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL
# ============================================================


@st.cache_resource(show_spinner="Loading detection model (first run only)...")
def load_classification_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Find the human label; the other class is the AI class.
    labels = {int(k): str(v) for k, v in model.config.id2label.items()}
    human_id = next((k for k, v in labels.items() if "human" in v.lower()), 0)
    ai_id = next((k for k in labels if k != human_id), 1)
    return tokenizer, model, device, ai_id


try:
    tokenizer, model, device, AI_ID = load_classification_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Model loading error: {e}")
    st.stop()

# ============================================================
# ANALYSIS
# ============================================================


def analyze(text: str, chunk_size: int) -> dict:
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    ai_probs = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_TOKENS
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1)
        ai_probs.extend((probs[:, AI_ID] * 100).tolist())

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]
    buzz_counts = Counter(m.group(0).lower() for m in BUZZWORD_PATTERN.finditer(text))

    return {
        "text": text,
        "word_count": len(words),
        "chunks": chunks,
        "chunk_ai_probs": ai_probs,
        "ai_percentage": sum(ai_probs) / len(ai_probs),
        "sentence_lengths": sentence_lengths,
        "buzzwords": dict(buzz_counts),
    }


def verdict_for(ai_pct: float, threshold: int) -> tuple[str, str, str, str]:
    """Returns (title, note, colour, background)."""
    if ai_pct >= threshold + 10:
        return ("Likely AI-generated", "The model sees strong AI patterns in this text.",
                "#e5484d", "rgba(229,72,77,0.10)")
    if ai_pct <= threshold - 10:
        return ("Likely human-written", "The model sees mostly human patterns in this text.",
                "#30a46c", "rgba(48,164,108,0.10)")
    return ("Uncertain / mixed", "The score is close to the threshold, so treat it with caution.",
            "#f5a524", "rgba(245,165,36,0.12)")


def highlight_buzzwords(text: str) -> str:
    safe = html.escape(text)
    safe = BUZZWORD_PATTERN.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
    return safe.replace("\n", "<br>")


# ============================================================
# STATE AND CALLBACKS
# ============================================================

st.session_state.setdefault("text_input", "")
st.session_state.setdefault("result", None)


def load_example(name: str):
    st.session_state.text_input = EXAMPLES[name]
    st.session_state.result = None


def clear_all():
    st.session_state.text_input = ""
    st.session_state.result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "AI decision threshold (%)", 30, 70, 50, 5,
        help="Scores well above this are flagged as AI, well below as human. "
             "Within 10 points either side is reported as uncertain.",
    )
    chunk_size = st.slider(
        "Chunk size (words)", 100, 300, 150, 25,
        help="Long text is split into chunks of this size and each chunk is scored separately.",
    )

    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        f"""
        **Model:** `{MODEL_ID}` (RoBERTa)

        **Running on:** {"GPU" if device.type == "cuda" else "CPU"}

        **How it works**
        1. Your text is split into chunks.
        2. Each chunk gets an AI probability.
        3. The chunk scores are averaged into one result.
        """
    )
    st.caption(
        "Results are an estimate from a statistical model, not proof of authorship. "
        "Short texts are especially unreliable."
    )

# ============================================================
# HEADER AND INPUT
# ============================================================

st.markdown(
    """
    <div class="hero">
      <h1>🤖 AI vs 👤 Human Text Detector</h1>
      <p>Paste a paragraph or a full document to estimate whether it was written by AI or a human.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.text_area(
        "Your text",
        key="text_input",
        placeholder="Type or paste your text here. Any length works; long text is analysed in chunks.",
        height=220,
    )

    current_words = len(st.session_state.text_input.split())
    left, mid, right = st.columns([2, 3, 2])
    with left:
        analyze_clicked = st.button("🔍 Analyze text", type="primary")
    with mid:
        ex_cols = st.columns(len(EXAMPLES))
        for col, name in zip(ex_cols, EXAMPLES):
            col.button(name, on_click=load_example, args=(name,))
    with right:
        st.button("🗑️ Clear", on_click=clear_all)

    st.caption(f"{current_words} words · {len(st.session_state.text_input)} characters")

if analyze_clicked:
    text = st.session_state.text_input.strip()
    if not text:
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing your text..."):
            st.session_state.result = analyze(text, chunk_size)
        st.toast("Analysis complete", icon="✅")

# ============================================================
# RESULTS
# ============================================================

result = st.session_state.result

if result is None:
    st.info("Paste some text above and press **Analyze text**, or try one of the samples.")
    st.stop()

ai_pct = result["ai_percentage"]
chunk_probs = result["chunk_ai_probs"]
n_chunks = len(chunk_probs)
ai_chunks = sum(p > threshold for p in chunk_probs)
title, note, colour, bg = verdict_for(ai_pct, threshold)

st.subheader("📊 Result")

st.markdown(
    f"""
    <div class="verdict" style="--c:{colour}; --bg:{bg};">
      <div class="label">Verdict</div>
      <div class="value">{title}</div>
      <div class="note">{note}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("AI probability", f"{ai_pct:.1f}%")
m2.metric("Human probability", f"{100 - ai_pct:.1f}%")
m3.metric("AI chunks", f"{ai_chunks} / {n_chunks}")
m4.metric("Words analysed", f"{result['word_count']:,}")

st.progress(min(max(int(round(ai_pct)), 0), 100), text="← Human  ·  AI →")

if result["word_count"] < 50:
    st.warning("This text is under 50 words. Detection on short text is unreliable.")

tab_chunks, tab_buzz, tab_flow, tab_export = st.tabs(
    ["🧩 Chunk breakdown", "🔍 Buzzwords", "📈 Sentence flow", "⬇️ Export"]
)

# ---- Chunks ----
with tab_chunks:
    chunk_df = pd.DataFrame(
        {
            "Chunk": [f"#{i + 1}" for i in range(n_chunks)],
            "AI probability": chunk_probs,
            "Verdict": ["AI" if p > threshold else "Human" for p in chunk_probs],
            "Preview": [c[:120] + ("..." if len(c) > 120 else "") for c in result["chunks"]],
        }
    )
    if n_chunks > 1:
        st.write("AI probability for each part of the text. Spikes show which sections look most AI-like.")
        st.bar_chart(chunk_df, x="Chunk", y="AI probability", color=colour)
    st.dataframe(
        chunk_df,
        hide_index=True,
        column_config={
            "AI probability": st.column_config.ProgressColumn(
                "AI probability", format="%.1f%%", min_value=0, max_value=100
            ),
            "Preview": st.column_config.TextColumn("Preview", width="large"),
        },
    )

# ---- Buzzwords ----
with tab_buzz:
    buzz = result["buzzwords"]
    if buzz:
        st.write(
            f"Found **{sum(buzz.values())}** occurrence(s) of **{len(buzz)}** common AI-style word(s). "
            "They are highlighted in the text below."
        )
        b1, b2 = st.columns([1, 2])
        with b1:
            st.dataframe(
                pd.DataFrame(
                    sorted(buzz.items(), key=lambda kv: -kv[1]), columns=["Word / phrase", "Count"]
                ),
                hide_index=True,
            )
        with b2:
            st.markdown(
                f'<div class="highlight-box">{highlight_buzzwords(result["text"])}</div>',
                unsafe_allow_html=True,
            )
        st.caption("These words are only a hint. Plenty of human writers use them too.")
    else:
        st.success("None of the common AI-style buzzwords were found in this text.")

# ---- Sentence flow ----
with tab_flow:
    lengths = result["sentence_lengths"]
    if len(lengths) > 1:
        s1, s2, s3 = st.columns(3)
        s1.metric("Sentences", len(lengths))
        s2.metric("Average length", f"{statistics.mean(lengths):.1f} words")
        s3.metric(
            "Variation (std. dev.)", f"{statistics.pstdev(lengths):.1f}",
            help="Higher variation means more uneven pacing, which is more typical of human writing.",
        )
        st.write(
            "Human writing tends to mix short and long sentences. "
            "AI text often keeps sentence lengths more uniform."
        )
        flow_df = pd.DataFrame(
            {"Sentence number": range(1, len(lengths) + 1), "Words in sentence": lengths}
        )
        st.bar_chart(flow_df, x="Sentence number", y="Words in sentence")
    else:
        st.info("Provide at least 2 sentences to see the sentence length analysis.")

# ---- Export ----
with tab_export:
    report = {
        "model": MODEL_ID,
        "threshold": threshold,
        "verdict": title,
        "ai_probability": round(ai_pct, 2),
        "human_probability": round(100 - ai_pct, 2),
        "word_count": result["word_count"],
        "chunk_ai_probabilities": [round(p, 2) for p in chunk_probs],
        "buzzwords": result["buzzwords"],
        "sentence_lengths": result["sentence_lengths"],
    }
    st.write("Download the full analysis for your records.")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download report (JSON)", json.dumps(report, indent=2),
        file_name="ai_detection_report.json", mime="application/json",
    )
    d2.download_button(
        "Download chunk table (CSV)", chunk_df.to_csv(index=False),
        file_name="chunk_breakdown.csv", mime="text/csv",
    )

st.divider()
st.caption(
    "This prediction is an estimate from the trained model, not a guaranteed determination of authorship."
)
