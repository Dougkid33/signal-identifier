import pandas as pd
import ast
import logging

# Configuração de logging
logging.basicConfig(
    filename="conversion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_excel_data(input_path, output_path):
    try:
        logging.info(f"Iniciando a conversão do arquivo: {input_path}")

        # Carregar o arquivo Excel
        excel_data = pd.ExcelFile(input_path)
        corrected_sheets = {}

        # Processar cada aba do Excel
        for sheet_name in excel_data.sheet_names:
            logging.info(f"Processando aba: {sheet_name}")
            try:
                df = excel_data.parse(sheet_name)

                # Verificar se a coluna 'iq_signal' existe
                if 'iq_signal' not in df.columns:
                    logging.warning(f"Aba {sheet_name} não contém a coluna 'iq_signal'. Ignorando.")
                    continue

                # Converter as células de 'iq_signal' de strings para listas reais
                def parse_signal(signal):
                    try:
                        if isinstance(signal, str):
                            return ast.literal_eval(signal)
                        return signal
                    except Exception as e:
                        logging.warning(f"Erro ao interpretar o sinal: {signal}. Erro: {e}")
                        return None  # Retornar None para sinais inválidos

                df['iq_signal'] = df['iq_signal'].apply(parse_signal)

                # Remover linhas com sinais inválidos (None)
                before_drop = len(df)
                df = df.dropna(subset=['iq_signal']).reset_index(drop=True)
                after_drop = len(df)

                logging.info(
                    f"Linhas removidas devido a sinais inválidos na aba {sheet_name}: {before_drop - after_drop}")

                corrected_sheets[sheet_name] = df

            except Exception as e:
                logging.error(f"Erro ao processar a aba {sheet_name}: {e}")

        # Salvar os dados corrigidos em um novo arquivo Excel
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, corrected_df in corrected_sheets.items():
                corrected_df.to_excel(writer, index=False, sheet_name=sheet_name)

        logging.info(f"Arquivo corrigido salvo em: {output_path}")

    except Exception as e:
        logging.error(f"Erro geral ao processar o arquivo: {e}")


# Caminhos dos arquivos
input_excel_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/dataconverted/rf_dataset.xlsx"
output_excel_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/dataconverted/rf_dataset_corrected.xlsx"

# Executar a conversão
convert_excel_data(input_excel_path, output_excel_path)
