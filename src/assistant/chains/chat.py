from typing import Any
from langchain import LlamaCpp, ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

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
            n_batch=256,
            n_ctx=1024,
            top_k=100,
            top_p=0.37,
            temperature=0.7,
            max_tokens=200,
        )
            
    def new_chain(self, **kwargs: Any):
        human_prefix=kwargs.get("human_prefix", "Human")

        return ConversationChain(
            llm=self.model,
            prompt=self.get_prompt_template(human_prefix), 
            callbacks=[self.handler],
            memory=ConversationBufferWindowMemory(
                k=3,
                human_prefix=human_prefix
            ),
        )

    def get_prompt_template(self, human_prefix: str = "User"):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        instruction = "Chat History:\n\n{history} \n\{human_prefix}: {input}"
        system_prompt = B_SYS +"You are a helpful assistant that always greeting users by they name, you always only answer for the assistant then you stop. read the chat history to get context"+ E_SYS
        
        template =  B_INST + system_prompt + instruction + E_INST + "\nAI:"

        return PromptTemplate(
            template=template, 
            input_variables=["history", "input"],
            partial_variables={"human_prefix": human_prefix}
        )
