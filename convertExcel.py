import h5py
import pandas as pd
from tqdm import tqdm
import logging
import gc

# Configuração de logging
logging.basicConfig(
    filename="/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/h5_to_excel.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Função para processar um único chunk
def process_chunk_to_dataframe(chunk_keys, h5_path):
    """
    Lê um conjunto de chaves do HDF5 e converte em um DataFrame.
    """
    data = []
    with h5py.File(h5_path, 'r') as f:
        for key in chunk_keys:
            try:
                if 'iq_signal' in f[key]:
                    dataset = f[key]['iq_signal'][:]
                    data.append({'key': key, 'iq_signal': dataset.tolist()})
            except Exception as e:
                logging.error(f"Erro ao processar a chave {key}: {e}")
    return pd.DataFrame(data)

# Função principal para conversão com otimização de memória
def h5_to_excel_with_chunks_optimized(h5_path, excel_path, chunk_size=2):
    """
    Converte um arquivo HDF5 em Excel usando chunks e otimizando o uso de memória.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            keys = list(f.keys())  # Obtem todas as chaves do HDF5
            total_keys = len(keys)
            logging.info(f"Arquivo HDF5 contém {total_keys} chaves.")

        # Dividir as chaves em chunks
        chunks = [keys[i:i + chunk_size] for i in range(0, total_keys, chunk_size)]

        # Processar chunks sequencialmente
        with pd.ExcelWriter(excel_path, mode='w') as writer:
            for i, chunk_keys in enumerate(tqdm(chunks, desc="Processando chunks")):
                logging.info(f"Processando chunk {i + 1}/{len(chunks)}")
                chunk_df = process_chunk_to_dataframe(chunk_keys, h5_path)

                # Escreve o chunk diretamente no Excel
                chunk_df.to_excel(writer, sheet_name=f"chunk_{i + 1}", index=False)

                # Libera memória
                del chunk_df
                gc.collect()
                logging.info(f"Chunk {i + 1} salvo e memória liberada.")

        logging.info(f"Arquivo Excel salvo em: {excel_path}")
        print(f"Arquivo Excel salvo em: {excel_path}")

    except Exception as e:
        logging.error(f"Erro durante o processo: {e}")
        raise

# Caminhos e configuração
h5_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/data/rf_dataset.h5"
excel_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/dataconverted/rf_dataset.xlsx"
chunk_size = 2  # Reduzido para otimizar memória

# Executa a conversão
h5_to_excel_with_chunks_optimized(h5_path, excel_path, chunk_size)
