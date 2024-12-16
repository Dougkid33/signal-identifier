import logging
import numpy as np
import tensorflow as tf
import pandas as pd
from keras._tf_keras.keras.utils import to_categorical
from model import build_cnn_model
import os
import ast  # Para verificar se uma string é uma lista válida

# Configuração de logging
logging.basicConfig(
    filename="/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/train.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()  # Para exibir no console também
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


# Função para configurar múltiplos threads
def configure_threads():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    logging.info("Configuração de threads ajustada para treinamento eficiente.")


# Validação e correção dos dados do Excel
def validate_and_clean_excel(df):
    valid_rows = []
    for index, row in df.iterrows():
        try:
            iq_signal = ast.literal_eval(row['iq_signal'])
            if isinstance(iq_signal, list) and all(isinstance(sub, list) and len(sub) == 2 for sub in iq_signal):
                valid_rows.append(row)
            else:
                logging.warning(f"Dados inválidos na linha {index}: {row['iq_signal']}. Ignorando.")
        except Exception as e:
            logging.warning(f"Erro ao validar linha {index}: {e}. Ignorando.")
    return pd.DataFrame(valid_rows)


# Generator para processar dados do Excel
def excel_generator(df, num_classes, window_size):
    try:
        for index, row in df.iterrows():
            try:
                iq_signal = np.array(ast.literal_eval(row['iq_signal']))

                if iq_signal.ndim != 2 or iq_signal.shape[1] != 2:
                    logging.warning(f"Formato inesperado do sinal na linha {index}. Ignorando.")
                    continue

                # Normalização
                iq_signal = (iq_signal - np.mean(iq_signal, axis=0)) / np.std(iq_signal, axis=0)

                # Criar janelas
                for i in range(0, len(iq_signal) - window_size + 1, window_size):
                    window = iq_signal[i:i + window_size]
                    label = row['key'] % num_classes  # Mapear chave para um rótulo
                    yield window, to_categorical(label, num_classes=num_classes)

            except Exception as e:
                logging.warning(f"Erro ao processar sinal na linha {index}: {e}. Ignorando.")
                continue

    except Exception as e:
        logging.error(f"Erro ao processar dados do Excel: {e}")
        raise


# Função para criar datasets
def create_excel_dataset(df, batch_size, num_classes, window_size):
    try:
        dataset = tf.data.Dataset.from_generator(
            lambda: excel_generator(df, num_classes, window_size),
            output_signature=(
                tf.TensorSpec(shape=(window_size, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
            )
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        logging.info(f"Dataset criado com sucesso para os dados fornecidos.")
        return dataset
    except Exception as e:
        logging.error(f"Erro ao criar o dataset: {e}")
        return None


if __name__ == "__main__":
    logging.info("Início do treinamento")

    try:
        configure_threads()

        # Configurações
        excel_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/dataconverted/rf_dataset.xlsx"
        train_split = 0.8  # 80% treino, 20% validação
        batch_size = 32
        num_classes = 4  # Ajuste conforme o dataset
        window_size = 2  # Tamanho das janelas para o modelo

        # Lendo o arquivo Excel
        logging.info(f"Lendo arquivo Excel: {excel_path}")
        df = pd.read_excel(excel_path)

        # Validar e limpar dados
        logging.info("Validando e limpando os dados do Excel.")
        df = validate_and_clean_excel(df)

        if df.empty:
            raise ValueError("Nenhum dado válido encontrado no arquivo Excel.")

        # Divisão de treino e validação
        train_length = int(len(df) * train_split)
        train_df = df.iloc[:train_length]
        val_df = df.iloc[train_length:]

        logging.info(f"Total de linhas no Excel: {len(df)}")
        logging.info(f"Linhas de treino: {len(train_df)}")
        logging.info(f"Linhas de validação: {len(val_df)}")

        # Criar datasets de treino e validação
        logging.info("Criando dataset de treino")
        train_dataset = create_excel_dataset(train_df, batch_size, num_classes, window_size)
        if train_dataset is None:
            raise ValueError("O dataset de treino não foi criado corretamente.")

        logging.info("Criando dataset de validação")
        val_dataset = create_excel_dataset(val_df, batch_size, num_classes, window_size)
        if val_dataset is None:
            raise ValueError("O dataset de validação não foi criado corretamente.")

        # Validação do formato do dataset
        for batch_signals, batch_labels in train_dataset.take(1):
            logging.info(f"Formato do lote de sinais: {batch_signals.shape}")
            logging.info(f"Formato do lote de rótulos: {batch_labels.shape}")

        # Construção do modelo
        input_shape = (window_size, 2)  # Formato ajustado para compatibilidade
        logging.info(f"Formato esperado pelo modelo: {input_shape}")
        model = build_cnn_model(input_shape, num_classes)

        # Configuração do callback para salvar checkpoints
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/models/checkpoint.weights.h5",
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1
        )

        # Início do treinamento
        logging.info("Iniciando treinamento")
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=[checkpoint_callback]
        )

        # Salvar o modelo final
        final_model_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/models/signal_identifier_cnn.h5"
        model.save(final_model_path)
        logging.info(f"Modelo treinado e salvo com sucesso em {final_model_path}")
    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        raise
