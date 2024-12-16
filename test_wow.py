import numpy as np
from keras._tf_keras.keras.models import load_model
from preprocess import process_wow_signal
from vizualize import plot_signal, plot_frequency_spectrum, plot_predictions

def test_wow_signal(wow_signal_path, model_path, sampling_rate):
    """Testa o sinal Wow usando o modelo treinado e exibe gráficos."""
    # Carrega o sinal Wow processado
    wow_signal = np.load(wow_signal_path)

    # Visualiza o sinal no domínio do tempo
    plot_signal(wow_signal, title="Sinal Wow no Domínio do Tempo")

    # Visualiza o espectro de frequência
    plot_frequency_spectrum(wow_signal, sampling_rate, title="Espectro de Frequência do Sinal Wow")

    # Carrega o modelo treinado e faz a predição
    model = load_model(model_path)
    input_data = np.expand_dims(wow_signal, axis=(0, -1))  # Ajuste de dimensões
    predictions = model.predict(input_data)

    # Visualiza as predições
    classes = ["BPSK", "QPSK", "8PSK", "16QAM"]  # Ajuste conforme o modelo
    plot_predictions(predictions, classes)

if __name__ == "__main__":
    wow_signal_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/data/wow_signal.npy"
    model_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/models/signal_identifier_cnn.h5"
    sampling_rate = 1000000  # Taxa de amostragem de 1 MHz

    test_wow_signal(wow_signal_path, model_path, sampling_rate)
