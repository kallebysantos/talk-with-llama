from langchain import LlamaCpp, ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

from .._assistant import Assistant

class Chat(Assistant):
    """
    Chat data structure.
    """
    def __init__(self, model_path: str):
        self.model = LlamaCpp(
            verbose=True,
            model_path=model_path, 
            callbacks=[self.handler],
            n_gpu_layers=25,
            n_batch=512,
            n_ctx=2048,
            top_k=100,
            top_p=0.37,
            temperature=0.98,
            max_tokens=200,
        )
            
    def new_chain(self):
        return ConversationChain(
            llm=self.model,
            prompt=self.get_prompt_template(), 
            callbacks=[self.handler],
            memory=ConversationBufferMemory(),
        )

    def get_prompt_template(self):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        instruction = "Chat History:\n\n{history} \n\nUser: {input}"
        system_prompt = B_SYS +"You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"+ E_SYS
        
        template =  B_INST + system_prompt + instruction + E_INST

        return PromptTemplate(
            template=template, 
            input_variables=["history", "input"], 
        )
