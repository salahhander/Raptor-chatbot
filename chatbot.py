import cProfile
import pstats
import io
import logging
import torch
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel
from raptor.EmbeddingModels import BaseEmbeddingModel
from raptor.RetrievalAugmentation import RetrievalAugmentationConfig
from raptor.RetrievalAugmentation import RetrievalAugmentation

import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authenticate with HuggingFace if needed
login()

class GEMMASummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="google/gemma-2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarization_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    def summarize(self, context, max_tokens=150):
        messages = [{"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.summarization_pipeline(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        summary = outputs[0]["generated_text"].strip()
        return summary

class GEMMAQAModel(BaseQAModel):
    def __init__(self, model_name="google/gemma-2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    def answer_question(self, context, question):
        messages = [{"role": "user", "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.qa_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        answer = outputs[0]["generated_text"][len(prompt):]
        return answer

class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

def main():
    logger.info("Initializing models and configurations")
    RAC = RetrievalAugmentationConfig(
        summarization_model=GEMMASummarizationModel(),
        qa_model=GEMMAQAModel(),
        embedding_model=SBertEmbeddingModel()
    )

    RA = RetrievalAugmentation(config=RAC)

    input_dir = r'C:\Users\salah\final chabot\documents\cleaned_splited_doc'  # Directory containing the split files
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
                RA.add_documents(file.read())

    question = "En cas de collision avec un autre véhicule, de choc contre un corps fixe ou mobile, ou de renversement sans collision préalable, du véhicule assuré, l’assureur garantit quoi?"
    logger.info("Answering question: %s", question)
    answer = RA.answer_question(question=question)
    logger.info("Answer: %s", answer)
    print("Answer: ", answer)

if __name__ == "__main__":
    main()
