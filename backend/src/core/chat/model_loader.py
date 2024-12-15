from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from unsloth import FastLanguageModel


def load_model(model_name: str = "cyberagent/Mistral-Nemo-Japanese-Instruct-2408"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, return_full_text=False
    )
    return ChatHuggingFace(
        llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tokenizer
    )