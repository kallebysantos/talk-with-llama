import datetime
from flask import Flask, render_template, session
from flask_socketio import SocketIO

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./models/OpenLLaMA_3B.ggmlv1.q4_0.bin", callback_manager=callback_manager, verbose=True
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'

socket_io = SocketIO(app)

def processPrompt(prompt: str):
    socket_io.emit('response', {
        'name': 'Assistant',
        'message': llm_chain.run(prompt),
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    })

@socket_io.on('connected')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    session["user"] = json["data"]["username"]

    print('User connected: ' + session["user"])

@socket_io.on('message')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    message = json["data"]

    processPrompt(message)

    return {
        'name': session["user"],
        'message': message,
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

@app.route('/')
def sessions():
    print('message was received!!!')
    return render_template('session.html')

if __name__ == '__main__':
    socket_io.run(app, port=7860, debug=True)