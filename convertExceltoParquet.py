import pandas as pd
import logging

# Configuração de logging
logging.basicConfig(
    filename="/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/convert_to_parquet.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

def excel_to_parquet(excel_path, parquet_path):
    """
    Converte um arquivo Excel em formato Parquet.
    """
    try:
        logging.info(f"Lendo arquivo Excel: {excel_path}")
        df = pd.read_excel(excel_path, sheet_name=None)  # Lê todas as abas do Excel

        logging.info("Convertendo para Parquet...")
        for sheet_name, data in df.items():
            sheet_parquet_path = parquet_path.replace(".parquet", f"_{sheet_name}.parquet")
            data.to_parquet(sheet_parquet_path, index=False)
            logging.info(f"Aba {sheet_name} salva como Parquet em: {sheet_parquet_path}")

        logging.info("Conversão para Parquet concluída com sucesso!")
        print(f"Arquivos Parquet salvos em: {parquet_path}")
    except Exception as e:
        logging.error(f"Erro durante a conversão: {e}")
        raise

# Caminhos de entrada e saída
excel_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/dataconverted/rf_dataset.xlsx"
parquet_path = "/home/user/Área de trabalho/IFTM/IA/Exercicios Jeffão/SignalIdentifier/dataconverted/rf_dataset.parquet"

# Executa a conversão
excel_to_parquet(excel_path, parquet_path)
