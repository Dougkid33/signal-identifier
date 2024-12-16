# Signal Identifier 📡✨
> Um projeto para explorar e classificar o misterioso Sinal Wow com Machine Learning.

## 🚀 Visão Geral
Este repositório contém o projeto **Signal Identifier**, uma aplicação de **machine learning** para identificar padrões no famoso **Sinal Wow**, capturado em 1977 pelo programa SETI. O foco principal deste projeto é aprendizado, análise de dados, e experimentação com técnicas de IA.

## 🛠️ Tecnologias Utilizadas
- **Python** 🐍
- **TensorFlow/Keras** para o modelo CNN
- **NumPy** para manipulação de dados
- **Librosa** para processamento de sinais
- **Matplotlib** para visualização de sinais
- **Pandas** para manipulação de arquivos Excel e HDF5

## 📂 Estrutura do Repositório
- `correct_formact.py` 📑: Script para correção e formatação de arquivos Excel.
- `train.py` 🤖: Treinamento do modelo CNN para classificação de sinais.
- `convertExceltoParquet.py` 🔄: Conversão de arquivos Excel para formato Parquet.
- `model.py` 🧠: Construção do modelo CNN.
- `vizualize.py` 📊: Visualização do sinal no domínio do tempo e frequência.
- `classify_wow.py` 🛸: Script para classificação do Sinal Wow com o modelo treinado.
- `preprocess.py` 🛠️: Pré-processamento e normalização de sinais.

## 🚦 Estado Atual
- **Funcionalidade**: A pipeline de dados está funcional, mas limitada pela capacidade computacional disponível.
- **Próximos Passos**:
  - Refinar o pipeline de dados.
  - Otimizar o modelo CNN para maior precisão.
  - Expandir a análise para sinais adicionais.

## 🛠️ Como Usar
1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/signal-identifier.git
2. Instale as dependências:
```bash
pip install -r requirements.txt
```
3. Pré-processe o sinal Wow:
```bash
python preprocess.py
```
4. Treine o modelo:
```bash
python train.py
```
5. Classifique o sinal:
```bash
python classify_wow.py
```
🤝 Contribuições
Contribuições são bem-vindas! Caso tenha ideias ou melhorias, abra uma issue ou envie um pull request.

