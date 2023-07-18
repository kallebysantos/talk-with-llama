import hashlib
from tempfile import template
from typing import Dict, Optional

from langchain import ConversationChain, LLMChain, LlamaCpp, PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate


from .models.chat_instruction import ChatInstruction

from .streaming_web import StreamingWebCallbackHandler

class Assistant():
    model: BaseLanguageModel
    chains: Dict[str, LLMChain] = {}
    handler = StreamingWebCallbackHandler()

    def __init__(self, model_path : str):
        self.model = LlamaCpp(
            verbose=True,
            model_path=model_path, 
            callbacks=[self.handler],
            temperature=0.4,
            n_ctx=1024,
            max_tokens=2048,
            last_n_tokens_size = 16,
            repeat_penalty=1.1,
            #use_mlock=True,
            #top_k=0,
            #top_p=0,
            #suffix = "END\n\n",
            stop=["User:"]
        )
        
    def new_chain(self):
        template = """
This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information.
Continue the chat dialogue below. Write a single reply for the character "Assistant".

User: What's the capital of France?\n\n
Assistant: Paris is the city you're looking for.\n\n
User: What means "Bom dia" ?\n\n
Assistant: Good morning in Portuguese.\n\n
User: {question}\n\n
Assistant:"""

        prompt = PromptTemplate(
            template=template, 
            input_variables=["question"], 
        )

        return LLMChain(
            memory=None,
            prompt=prompt, 
            llm=self.model, 
            callbacks=[self.handler]
        )
    
    def add_chain(self, key: str):
        hashed_key = hashlib.sha256(str.encode(key)).hexdigest()

        if(hashed_key not in self.chains):
            self.chains[hashed_key] = self.new_chain()

        return self.chains[hashed_key], hashed_key

    def get_chain(self, hashed_key: str) -> Optional[LLMChain]:
        return self.chains[hashed_key] if hashed_key in self.chains else None