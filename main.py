import sys
import datetime

from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.assistant import Assistant, StreamingWebCallbackHandler
from src.assistant.chains.chat_instruction import ChatInstruction

assistant: Assistant

@asynccontextmanager
async def lifespan(_: FastAPI):
    global assistant
    assistant = ChatInstruction("./models/OpenLLaMA_3B.ggmlv1.q4_0.bin")
        
    yield

    assistant.chains.clear()

templates = Jinja2Templates(directory="templates")
app = FastAPI(lifespan=lifespan)

@app.get('/response/{user_id}')
async def streamed_response(user_id: str):
    chain = assistant.get_chain(user_id)
    if chain is None or len(chain.callbacks) <= 0:
        return Response(status_code=422)

    handler: StreamingWebCallbackHandler = chain.callbacks[0]

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
    global assistant

    chain, chain_hash = assistant.add_chain(message.username)

    tasks.add_task(chain.run, message.data)

    return {
        'id': str(chain_hash),
        'name': message.username,
        'message': message.data,
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

@app.get('/', response_class=HTMLResponse)
async def chat_ui(req: Request):
    return templates.TemplateResponse('chat_ui.html', { "request": req })