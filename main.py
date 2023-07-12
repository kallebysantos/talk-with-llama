from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'

socket_io = SocketIO(app)


def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socket_io.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    socket_io.emit('my response', json, callback=messageReceived)

@app.route('/')
def sessions():
    print('message was received!!!')
    return render_template('session.html')

if __name__ == '__main__':
    socket_io.run(app, port=7860, debug=True)