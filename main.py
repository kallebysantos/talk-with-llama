import hashlib
import sys
import datetime
from typing import Dict, List

from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.schema.language_model import BaseLanguageModel

from src.streaming_web import StreamingWebCallbackHandler

ml_model: BaseLanguageModel
ml_chains: Dict[str, LLMChain] = {}
stream_handler = StreamingWebCallbackHandler()  

@asynccontextmanager
async def lifespan(_: FastAPI):
    global ml_model, ml_chains
    ml_model = get_model()
        
    yield

    ml_chains.clear()

templates = Jinja2Templates(directory="templates")
app = FastAPI(lifespan=lifespan)

def get_model():
    #Make sure the model path is correct for your system!
    return LlamaCpp(
        model_path="./models/OpenLLaMA_3B.ggmlv1.q4_0.bin", 
        callbacks=[stream_handler], 
        verbose=True
    )


def get_new_chain():
    template = """Question: {question}

    Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    return LLMChain(prompt=prompt, llm=ml_model, callbacks=[stream_handler])

@app.get('/response/{user_id}')
async def streamed_response(user_id: str):
    print(ml_chains, user_id)

    user_chain = ml_chains.get(user_id)
    if user_chain is None or len(user_chain.callbacks) <= 0:
        return Response(status_code=422)

    handler = user_chain.callbacks[0]

    def generate():
        while True:
            while handler.is_responding & len(handler.tokens) > 0:
                token = handler.tokens.pop(0)
                sys.stdout.write(token)
                sys.stdout.flush()

                yield {
                    "event": "assistant-responding",
                    "id": handler.response_id,
                    "data": token
                }
            
            if handler.is_responding == False:
                yield {
                    "event": "assistant-waiting",
                    "id": handler.response_id,
                    "data": 'waiting'
                }

    return EventSourceResponse(generate())

class Message(BaseModel):
    username: str
    data: str
    
@app.post('/message')
async def handle_message(message: Message, tasks: BackgroundTasks):
    global ml_chains

    user_id = hashlib.sha256(str.encode(message.username)).hexdigest()
    ml_chains[user_id] = get_new_chain()

    tasks.add_task(ml_chains[user_id].run, message.data)

    return {
        'id': str(user_id),
        'name': message.username,
        'message': message.data,
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

@app.get('/', response_class=HTMLResponse)
async def chat_ui(req: Request):
    return templates.TemplateResponse('chat_ui.html', { "request": req })