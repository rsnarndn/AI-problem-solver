import os
import warnings
import logging
import re
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sympy as sp
from rank_bm25 import BM25Okapi
import ollama  # Import the ollama package

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)

# -------------------------
# Physics Knowledge Base
# -------------------------
class PhysicsKnowledgeBase:
    def __init__(self, knowledge_dir=r"C:\Users\DELL\Documents\AI problem solver\physics_knowledge"):
        self.knowledge = []
        self.tokenized_knowledge = []
        self.bm25 = None

        # Create directory if it doesn't exist
        Path(knowledge_dir).mkdir(exist_ok=True)

        # Load files from all subdirectories (only text files)
        for root, _, files in os.walk(knowledge_dir):
            for file in files:
                if file.endswith(".txt"):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            content = f.read()
                            sentences = sent_tokenize(content)
                            self.knowledge.extend(sentences)
                    except Exception as e:
                        logging.warning(f"Could not load {file}: {str(e)}")
                        continue

        if not self.knowledge:
            raise ValueError("No valid physics knowledge found in the directory.")

        self.tokenized_knowledge = [word_tokenize(doc.lower()) for doc in self.knowledge]
        self.bm25 = BM25Okapi(self.tokenized_knowledge)
        logging.info("Physics knowledge base loaded with %d sentences.", len(self.knowledge))

    def retrieve_context(self, question, top_k=3):
        tokenized_question = word_tokenize(question.lower())
        scores = self.bm25.get_scores(tokenized_question)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        context = ' '.join([self.knowledge[i] for i in top_indices])
        logging.info("Retrieved context: %s", context[:150] + "..." if len(context) > 150 else context)
        return context

# -------------------------
# Deepseek QA Function using Ollama
# -------------------------
def deepseek_qa_ollama(question, context):
    """
    Uses the ollama package to interact with your local Deepseek model.
    Combines the question and context into a prompt and sends it to the model.
    """
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    try:
        response = ollama.chat(
            model="deepseek-r1:1.5b-qwen-distill-q8_0",
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        return response['message']['content']
    except Exception as e:
        logging.error("Ollama Deepseek API call failed: %s", e)
        return "Error: Unable to connect to Deepseek model via Ollama."

# -------------------------
# AI Physics Solver
# -------------------------
class PhysicsSolver:
    def __init__(self, use_deepseek=True):
        self.use_deepseek = use_deepseek
        if not self.use_deepseek:
            # Fallback: Use a Hugging Face QA pipeline
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
            self.model_name = "deepset/roberta-base-squad2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1
            )
            logging.info("QA pipeline initialized with model: %s", self.model_name)
        else:
            logging.info("Using local Deepseek model via Ollama API.")

        self.knowledge_base = PhysicsKnowledgeBase()

    def symbolic_solver(self, text):
        """
        A simple symbolic solver that extracts the first two numerical values
        and attempts to perform an arithmetic operation.
        (This is a placeholder; you can expand it to handle more complex equations.)
        """
        try:
            numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            if len(numbers) >= 2:
                expr = f"{numbers[0]} * {numbers[1]}"
                result = sp.sympify(expr).evalf()
                return f"{expr} = {result}"
        except Exception as e:
            logging.error("Symbolic solver error: %s", e)
        return None

    def solve_problem(self, problem):
        # Retrieve relevant context from the knowledge base
        context = self.knowledge_base.retrieve_context(problem)
        if self.use_deepseek:
            answer = deepseek_qa_ollama(problem, context)
        else:
            try:
                result = self.qa_pipeline(question=problem, context=context)
                answer = result.get('answer', "No answer found.")
            except Exception as e:
                logging.error("QA pipeline error: %s", e)
                answer = "Error in processing your query."

        sym_solution = self.symbolic_solver(answer)
        if sym_solution:
            return f"{answer}\nSymbolic Computation: {sym_solution}"
        return answer

# -------------------------
# Main Application
# -------------------------
def main():
    ascii_art = """
    █▀█ █▀▀ █░█ █▀█ █▀▄▀█ ▄▀█ █▀   █▀█ ▄▀█ █▀▀ █▀▀ █ █▀█ █▀▀
    █▀▄ ██▄ ▀▄▀ █▄█ █░▀░█ █▀█ ▄█   █▀▀ █▀█ █▄█ ██▄ █ █▄█ █▄█

    Advanced Physics Problem Solver for IIT-JEE/JAM/CSIR-NET/GATE
    """
    print(ascii_art)
    logging.info("Device set to use CPU")
    
    solver = PhysicsSolver(use_deepseek=True)
    
    while True:
        problem = input("\nEnter your physics problem (or 'exit'): ")
        if problem.lower() in ['exit', 'quit']:
            print("\nThank you for using the solver. Good luck with your exams!")
            break
        
        solution = solver.solve_problem(problem)
        print(f"\nSOLUTION:\n{solution}\n")

if __name__ == "__main__":
    main()
