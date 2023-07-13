import sys
import datetime
import time

from celery import Celery, Task
from flask import Flask, Response, stream_with_context, render_template, request, session

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.streaming_web import StreamingWebCallbackHandler

global stream_handler 
stream_handler = StreamingWebCallbackHandler()

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Make sure the model path is correct for your system!
llm = LlamaCpp(model_path="./models/OpenLLaMA_3B.ggmlv1.q4_0.bin", callbacks=[stream_handler], verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

app = Flask(__name__)

@app.get('/response')
def streamed_response():
    @stream_with_context
    def generate():
        while True:
            while stream_handler.is_responding & len(stream_handler.tokens) > 0:
                token = stream_handler.tokens.pop(0)
                sys.stdout.write(token)
                sys.stdout.flush()

                yield f'event: assistant-response\nid: 0\ndata: {token}\n\n'


    return Response(generate(), mimetype='text/event-stream')


@app.post('/message')
def handle_message():
    message = request.json["data"]

    llm_chain.run(message)

    # socket_io.start_background_task(llm_chain.run, message)

    return {
        'name': session["user"],
        'message': message,
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

@app.route('/')
def sessions():
    return render_template('session.html')

if __name__ == '__main__':
    app.run(
        port=7860,
        threaded=True,
        debug=True
    )