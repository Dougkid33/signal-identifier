import logging
import numpy as np
from keras._tf_keras.keras.models import load_model

# Configuração de logging
logging.basicConfig(
    filename="../train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_signal(file_path):
    """
    Carrega o sinal Wow pré-processado.
    """
    try:
        signal = np.load(file_path)  # Arquivo wow_signal.npy
        logging.info(f"Sinal carregado com sucesso de {file_path}.")
        return signal
    except Exception as e:
        logging.error(f"Erro ao carregar o sinal Wow: {e}")
        raise

def preprocess_signal(signal):
    """
    Pré-processa o sinal para classificação.
    """
    signal = (signal - np.mean(signal)) / np.std(signal)  # Normalização
    signal = np.expand_dims(signal, axis=0)  # Adiciona dimensão do batch
    signal = np.expand_dims(signal, axis=-1)  # Adiciona dimensão do canal
    logging.info(f"Sinal pré-processado. Formato: {signal.shape}")
    return signal

def classify_signal(signal, model_path):
    """
    Classifica o sinal usando o modelo treinado.
    """
    try:
        model = load_model(model_path)
        predictions = model.predict(signal)
        predicted_class = np.argmax(predictions, axis=1)
        logging.info(f"Predição: {predictions}, Classe mais provável: {predicted_class[0]}")
        return predictions, predicted_class[0]
    except Exception as e:
        logging.error(f"Erro ao classificar o sinal Wow: {e}")
        raise

if __name__ == "__main__":
    # Caminhos de entrada e saída
    signal_path = "../data/wow_signal.npy"  # Arquivo com o sinal Wow
    model_path = "../models/signal_identifier_cnn.h5"

    # Fluxo principal
    try:
        signal = load_signal(signal_path)
        signal = preprocess_signal(signal)
        predictions, predicted_class = classify_signal(signal, model_path)
        print(f"Predições: {predictions}")
        print(f"Classe mais provável: {predicted_class}")
    except Exception as e:
        print(f"Erro durante a classificação: {e}")
