from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from unsloth import FastLanguageModel


class ChatModelLoader:
    def __init__(
        self,
        model_name: str = "cyberagent/Mistral-Nemo-Japanese-Instruct-2408",
    ) -> ChatHuggingFace:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, return_full_text=False
        )
        self.llm = ChatHuggingFace(
            llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tokenizer
        )
        
        self.runnable = self.llm
        
    def from_prompt_template(self, settings: dict):
        system_prompt = f"あなたは「assistant」として、以下のキャラクター設定と世界観の情報に基づいて「user」のメッセージに自然な返事をしてください。\n\nassistantのキャラクター設定：{settings['character_setting']}。\n世界観の情報：{settings['world_view']}"
        messages = [
            (
                "system",
                "会話の記録:\n{recall_memories}\n\n" f"{system_prompt}",
            ),
            ("placeholder", "{messages}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt
    
    # def set_prompt(self, prompt: str):
        
