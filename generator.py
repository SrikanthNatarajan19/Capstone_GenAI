from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


LLM_NAME = "google/flan-t5-large"


class AnswerGenerator:
    def __init__(self, model_name: str = LLM_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def build_qa_prompt(self, question: str, contexts):
        joined_context = "\n\n".join(
            [f"Context {i + 1}: {ctx}" for i, ctx in enumerate(contexts)]
        )

        prompt = f"""Answer the question in a complete, grammatically correct sentence based only on the provided context. If the answer is not in the context, say "Answer not found in the provided document."

        Context:
        {joined_context}

        Question: {question}
        Answer:"""
        return prompt.strip()

    def generate_text(self, prompt: str, max_new_tokens: int = 180):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # num_beams=5,
            # early_stopping=True,
            # no_repeat_ngram_size=3,
            # min_length=20,
            # length_penalty=1.2
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer_question(self, question: str, contexts):
        prompt = self.build_qa_prompt(question, contexts)
        return self.generate_text(prompt, max_new_tokens=180)

    def summarize_text(self, text: str):
        prompt = self.build_summary_prompt(text)
        return self.generate_text(prompt, max_new_tokens=200)