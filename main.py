import datetime
from time import time
from flask import Flask, render_template, session
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'

socket_io = SocketIO(app)

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socket_io.on('connected')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    session["user"] = json["data"]["username"]

    print('User connected: ' + session["user"])

@socket_io.on('message')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    socket_io.emit('my response', json, callback=messageReceived)

    return {
        'name': session["user"],
        'message': json["data"],
        'timestamp': datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    }

@app.route('/')
def sessions():
    print('message was received!!!')
    return render_template('session.html')

if __name__ == '__main__':
    socket_io.run(app, port=7860, debug=True)