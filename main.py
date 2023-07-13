import sys
import datetime
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

from src.streaming_web import StreamingWebCallbackHandler

global stream_handler 
stream_handler = StreamingWebCallbackHandler()

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Make sure the model path is correct for your system!
llm = LlamaCpp(model_path="./models/OpenLLaMA_3B.ggmlv1.q4_0.bin", callbacks=[stream_handler], verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get('/response')
async def streamed_response():
    def generate():
        while True:
            while stream_handler.is_responding & len(stream_handler.tokens) > 0:
                token = stream_handler.tokens.pop(0)
                sys.stdout.write(token)
                sys.stdout.flush()

                yield {
                    "event": "assistant-response",
                    "id": stream_handler.response_id,
                    "data": token
                }

    return EventSourceResponse(generate())

class Message(BaseModel):
    data: str
    
@app.post('/message')
async def handle_message(message: Message, tasks: BackgroundTasks):

    tasks.add_task(llm_chain.run, message.data)

    return {
        'name': 'Guest',
        'message': message.data,
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

@app.get('/', response_class=HTMLResponse)
async def chat_ui(req: Request):
    return templates.TemplateResponse('chat_ui.html', { "request": req })