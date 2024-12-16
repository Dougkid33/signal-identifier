import matplotlib.pyplot as plt
import numpy as np

def plot_signal_time(signal, title="Sinal no Domínio do Tempo"):
    """
    Plota o sinal no domínio do tempo.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def plot_signal_frequency(signal, sample_rate, title="Espectro de Frequência"):
    """
    Plota o espectro de frequência do sinal.
    """
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/sample_rate)
    magnitude = np.abs(fft)

    plt.figure(figsize=(10, 4))
    plt.plot(freq[:len(freq)//2], magnitude[:len(freq)//2])  # Apenas frequências positivas
    plt.title(title)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Carrega o sinal Wow pré-processado
    signal_path = "../data/wow_signal.npy"
    signal = np.load(signal_path)

    # Configurações
    sample_rate = 1000000  # Exemplo de taxa de amostragem (1 MHz)

    # Visualizações
    plot_signal_time(signal, title="Sinal Wow no Domínio do Tempo")
    plot_signal_frequency(signal, sample_rate, title="Espectro de Frequência do Sinal Wow")
