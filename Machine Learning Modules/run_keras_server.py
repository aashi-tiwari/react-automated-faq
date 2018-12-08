import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('/Users/aashitiwari/Documents/Courses Data/Spring 2018/CMPE-295A/Project/Notebooks/question_pairs_weights.h5')
    graph = tf.get_default_graph()

def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('test_id'))
    parameters.append(flask.request.args.get('question1'))
    parameters.append(flask.request.args.get('question2'))
    return parameters

    # Cross origin support
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

    # API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    testId = flask.request.args.get('test_id')
    parameters = getParameters()
    inputFeature = np.asarray(parameters).reshape(1, 4)
    with graph.as_default():
        raw_prediction = model.predict(inputFeature)[0][0]
    if raw_prediction > 0.7:
        prediction = 'Duplicate'
    else:
        prediction = 'not duplicate'
    return sendResponse({testId: prediction})

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(threaded=True)