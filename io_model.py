# based on https://machinelearningmastery.com/save-load-keras-deep-learning-models/

from keras.models import model_from_json


def save(model, model_filepath):
    '''Saves model to disk
    :param model: Keras model
    :param model_filepath: path to module (without file extension)
    '''
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_filepath+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_filepath + ".h5")
    print("Saved model to disk: " + model_filepath)


def load(model_filepath):
    '''Loads model from disk
    :param model_filepath: path to module (without file extension)
    :return: model loaded from disk
    '''
    json_file = open(model_filepath + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_filepath + ".h5")
    print("Loaded model from disk: " + model_filepath)
    return loaded_model