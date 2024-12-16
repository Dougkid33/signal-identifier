import numpy as np
import h5py
import librosa
import os

def inspect_h5_file(filepath):
    """Inspeciona e lista as chaves do arquivo HDF5."""
    with h5py.File(filepath, 'r') as f:
        print("Estrutura do arquivo HDF5:")
        def print_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name} (dataset) - shape: {obj.shape}")
            else:
                print(f"{name} (group)")
        f.visititems(print_keys)

def load_h5_dataset(filepath):
    """
    Carrega todos os datasets `iq_signal` dentro dos grupos `sample_X`.
    Args:
        filepath: Caminho para o arquivo HDF5.
    Returns:
        Tuple contendo os sinais concatenados e um array de rótulos simulados.
    """
    signals = []
    labels = []
    with h5py.File(filepath, 'r') as f:
        for group_name in f.keys():  # Itera sobre os grupos
            group = f[group_name]
            if 'iq_signal' in group:
                iq_data = np.array(group['iq_signal'])
                signals.append(iq_data)
                labels.append(int(group_name.split('_')[1]) % 4)  # Rótulos fictícios para exemplo
    return np.vstack(signals), np.array(labels)

def process_wow_signal(filepath, target_sr=1000000):
    """
    Carrega o sinal Wow em formato MP3 e converte para numpy array.
    Args:
        filepath: Caminho para o arquivo MP3.
        target_sr: Taxa de amostragem desejada (default: 1 MHz).
    Returns:
        Sinal normalizado no domínio do tempo.
    """
    signal, sr = librosa.load(filepath, sr=None)  # Carrega o sinal com a taxa original
    signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
    return (signal_resampled - np.mean(signal_resampled)) / np.std(signal_resampled)

def save_preprocessed_signal(signal_data, output_path):
    """Salva um sinal processado como .npy."""
    np.save(output_path, signal_data)

if __name__ == "__main__":
    # Caminho para o arquivo HDF5
    dataset_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/data/rf_dataset.h5"
    wow_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/data/wow_signal.mp3"

    # Inspecione o arquivo HDF5
    inspect_h5_file(dataset_path)

    # Carregue os dados do HDF5
    signals, labels = load_h5_dataset(dataset_path)
    print(f"Dataset carregado: {signals.shape} sinais, {labels.shape} rótulos")

    # Processando o sinal Wow
    wow_signal = process_wow_signal(wow_path)
    save_preprocessed_signal(wow_signal, "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/data/wow_signal.npy")
    print("Sinal Wow pré-processado e salvo.")
