from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from config import obp_model_params, abp_model_params, obp_model_path, abp_model_path


def build_nn(input_dim=10, hidden_layer_dims=(32, 64, 32), optimizer='adam',
             loss='binary_crossentropy', learning_rate=0.001, dropout=0.2, metrics=None):
    """
    Builds a neural network with specified parameters.

    Parameters:
    - input_dim: Number of input features
    - hidden_layer_dims: Tuple containing the sizes of the hidden layers
    - optimizer: Name of the optimizer
    - loss: Loss function for the model training
    - learning_rate: Learning rate for the optimizer
    - dropout: Dropout rate for regularization
    - metrics: List of metrics to be evaluated by the model during training and testing

    Returns:
    - Compiled model.
    """

    if metrics is None:
        metrics = ['accuracy', 'AUC']

    optimizers = {
        'adam': Adam(learning_rate=learning_rate),
        'sgd': SGD(learning_rate=learning_rate),
        'rmsprop': RMSprop(learning_rate=learning_rate)
    }

    optimizer_fun = optimizers.get(optimizer)
    if not optimizer_fun:
        raise ValueError(f"Unsupported optimizer '{optimizer}'. Choose from {list(optimizers.keys())}.")

    model = Sequential()
    model.add(Dense(hidden_layer_dims[0], input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())

    for dim in hidden_layer_dims[1:]:
        model.add(Dense(dim, activation='relu'))
        model.add(BatchNormalization())

    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer_fun, loss=loss, metrics=metrics)

    return model


def load_models():
    obp_model = build_nn(input_dim=10, hidden_layer_dims=obp_model_params['hidden_layer_dim'],
                         optimizer=obp_model_params['optimizer'], loss=obp_model_params['loss'],
                         learning_rate=obp_model_params['learning_rate'])

    abp_model = build_nn(input_dim=10, hidden_layer_dims=abp_model_params['hidden_layer_dim'],
                         optimizer=abp_model_params['optimizer'], loss=abp_model_params['loss'],
                         learning_rate=abp_model_params['learning_rate'])

    # load weights
    obp_model.load_weights(obp_model_path)
    abp_model.load_weights(abp_model_path)

    return obp_model, abp_model
