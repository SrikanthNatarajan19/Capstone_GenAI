import json
import time
import streamlit as st

from utils import extract_text_from_pdf, clean_text, chunk_text
from retriever import SemanticRetriever
from generator import AnswerGenerator
from evaluator import (
    get_memory_usage_mb,
    token_f1_score,
    rouge_scores,
    answer_grounded_in_context,
    retrieval_hit_at_k,
)


st.set_page_config(page_title="Academic QA Assistant", layout="wide")
st.info("""
How to use this app:
1. Upload a PDF or paste text
2. Click 'Process Document'
3. Ask a question or generate a summary
4. Review retrieved chunks and evaluation metrics
""")


@st.cache_resource
def load_retriever():
    return SemanticRetriever()


@st.cache_resource
def load_generator():
    return AnswerGenerator()


def load_sample_questions():
    try:
        with open("sample_questions.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def main():
    st.title("Academic Question Answering using RAG + FLAN-T5")
    st.write("Upload a PDF or paste academic text, then ask questions grounded in the document.")

    retriever = load_retriever()
    generator = load_generator()

    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "index_built" not in st.session_state:
        st.session_state.index_built = False

    st.sidebar.header("Settings")
    chunk_size = st.sidebar.slider("Chunk Size", 300, 1200, 700, 50)
    overlap = st.sidebar.slider("Chunk Overlap", 50, 300, 120, 10)
    top_k = st.sidebar.slider("Top-K Retrieval", 1, 8, 5, 1)
    min_score_threshold = st.sidebar.slider("Minimum Retrieval Score", 0.0, 1.0, 0.25, 0.01)

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    pasted_text = st.text_area("Or paste academic text here", height=200)

    if st.button("Process Document"):
        raw_text = ""

        if uploaded_pdf is not None:
            try:
                raw_text += extract_text_from_pdf(uploaded_pdf)
            except Exception as e:
                st.error(str(e))
                return

        if pasted_text.strip():
            raw_text += "\n" + pasted_text.strip()

        cleaned = clean_text(raw_text)

        if not cleaned:
            st.warning("No readable text found. Please upload a valid PDF or paste text.")
            return

        chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)

        if not chunks:
            st.warning("Chunking failed. No chunks were created.")
            return

        start_index_time = time.perf_counter()
        retriever.build_index(chunks)
        index_time = time.perf_counter() - start_index_time

        st.session_state.document_text = cleaned
        st.session_state.chunks = chunks
        st.session_state.index_built = True

        st.success("Document processed successfully.")
        st.write(f"**Characters extracted:** {len(cleaned)}")
        st.write(f"**Number of chunks:** {len(chunks)}")
        st.write(f"**Index build time:** {index_time:.3f} seconds")
        st.write(f"**Memory usage:** {get_memory_usage_mb():.2f} MB")

    if st.session_state.index_built:
        st.subheader("Ask a Question")
        question = st.text_input("Enter your question")

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
                return

            retrieval_start = time.perf_counter()
            results = retriever.retrieve(question, top_k=top_k)
            retrieval_time = time.perf_counter() - retrieval_start

            st.subheader("Retrieved Context")
            for i, r in enumerate(results):
                st.write(f"Chunk {i + 1} (score={r['score']:.3f}):")
                st.write(r["text"])

            filtered_results = [r for r in results if r["score"] >= min_score_threshold]

            if not filtered_results:
                st.subheader("Generated Answer")
                st.write("Answer not found in the provided document.")
                return

            contexts = [r["text"] for r in filtered_results]

            generation_start = time.perf_counter()
            answer = generator.answer_question(question, contexts)
            generation_time = time.perf_counter() - generation_start

            total_time = retrieval_time + generation_time
            grounded = answer_grounded_in_context(answer, contexts)

            st.subheader("Generated Answer")
            st.write(answer)

            st.subheader("Performance Metrics")
            st.write(f"**Retrieval time:** {retrieval_time:.3f} seconds")
            st.write(f"**Generation time:** {generation_time:.3f} seconds")
            st.write(f"**Total time:** {total_time:.3f} seconds")
            st.write(f"**Memory usage:** {get_memory_usage_mb():.2f} MB")
            st.write(f"**Used chunks after filtering:** {len(filtered_results)}")
            st.write(f"**Grounded in retrieved context:** {'Yes' if grounded else 'No'}")

            with st.expander("Retrieved Chunks"):
                for i, result in enumerate(results, start=1):
                    used_tag = "Used" if result["score"] >= min_score_threshold else "Filtered Out"
                    st.markdown(f"**Chunk {i}** | Score: {result['score']:.4f} | {used_tag}")
                    st.write(result["text"])
                    st.markdown("---")

        st.subheader("Generate Summary")
        if st.button("Generate Document Summary"):
            # Summarize more than before, but avoid going too long for model input
            doc_preview = st.session_state.document_text[:4000]

            summary_start = time.perf_counter()
            summary = generator.summarize_text(doc_preview)
            summary_time = time.perf_counter() - summary_start

            st.write(summary)
            st.write(f"**Summary generation time:** {summary_time:.3f} seconds")

        st.subheader("Mini Evaluation")
        sample_qas = load_sample_questions()

        if st.button("Run Sample Evaluation"):
            if not sample_qas:
                st.warning("No sample questions found.")
                return

            all_f1_scores = []
            all_rouge1 = []
            all_rouge2 = []
            all_rougel = []
            retrieval_hits = []
            grounded_flags = []

            for item in sample_qas:
                q = item["question"]
                expected = item["expected_answer"]

                results = retriever.retrieve(q, top_k=top_k)
                filtered_results = [r for r in results if r["score"] >= min_score_threshold]
                contexts = [r["text"] for r in filtered_results]

                if contexts:
                    generated = generator.answer_question(q, contexts)
                else:
                    generated = "Answer not found in the provided document."

                f1 = token_f1_score(generated, expected)
                rouge = rouge_scores(expected, generated)
                hit = retrieval_hit_at_k(results, expected)
                grounded = answer_grounded_in_context(generated, contexts) if contexts else False

                all_f1_scores.append(f1)
                all_rouge1.append(rouge["rouge1"])
                all_rouge2.append(rouge["rouge2"])
                all_rougel.append(rouge["rougeL"])
                retrieval_hits.append(hit)
                grounded_flags.append(1 if grounded else 0)

                with st.expander(f"Question: {q}"):
                    st.write(f"**Expected:** {expected}")
                    st.write(f"**Generated:** {generated}")
                    st.write(f"**Token F1:** {f1:.4f}")
                    st.write(f"**ROUGE-1:** {rouge['rouge1']:.4f}")
                    st.write(f"**ROUGE-2:** {rouge['rouge2']:.4f}")
                    st.write(f"**ROUGE-L:** {rouge['rougeL']:.4f}")
                    st.write(f"**Retrieval Hit@K:** {hit}")
                    st.write(f"**Grounded:** {'Yes' if grounded else 'No'}")

            avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
            avg_rouge1 = sum(all_rouge1) / len(all_rouge1) if all_rouge1 else 0.0
            avg_rouge2 = sum(all_rouge2) / len(all_rouge2) if all_rouge2 else 0.0
            avg_rougel = sum(all_rougel) / len(all_rougel) if all_rougel else 0.0
            avg_hit = sum(retrieval_hits) / len(retrieval_hits) if retrieval_hits else 0.0
            avg_grounded = sum(grounded_flags) / len(grounded_flags) if grounded_flags else 0.0

            st.write(f"**Average Token F1 Score:** {avg_f1:.4f}")
            st.write(f"**Average ROUGE-1:** {avg_rouge1:.4f}")
            st.write(f"**Average ROUGE-2:** {avg_rouge2:.4f}")
            st.write(f"**Average ROUGE-L:** {avg_rougel:.4f}")
            st.write(f"**Average Retrieval Hit@K:** {avg_hit:.4f}")
            st.write(f"**Average Groundedness:** {avg_grounded:.4f}")

        with st.expander("View Extracted Document Text"):
            st.write(st.session_state.document_text[:5000])


if __name__ == "__main__":
    main()