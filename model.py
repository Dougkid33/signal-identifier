from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

def build_cnn_model(input_shape, num_classes):
    """
    Constrói um modelo CNN compatível com entradas bidimensionais.
    Args:
        input_shape: Formato da entrada no formato (window_size, 2).
        num_classes: Número de classes de saída.
    Returns:
        Modelo Keras compilado.
    """
    model = Sequential([
        Input(shape=input_shape),  # Agora espera (window_size, 2)
        Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model