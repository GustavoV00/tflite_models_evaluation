# Treinamento e Otimização de Modelos MNIST

Este projeto demonstra o processo de treinamento de um modelo CNN no dataset MNIST, convertendo-o para o formato TensorFlow Lite e avaliando seu desempenho. O projeto é estruturado para otimizar modelos para implantação em dispositivos com recursos limitados, como o Raspberry Pi.

## Estrutura do Projeto

```
inference_in_raspberry/
├── requirements.txt
└── src/
    ├── prepare_data.py
    ├── train_model.py
    ├── convert_to_tflite.py
    ├── evaluate_tflite_models.py
    ├── data/           
    ├── models/        
    │   ├── keras_models/
    │   └── tflite_models/
    └── results/       
```

## Configuração

1. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate
```

2. Instale os pacotes necessários:
```bash
pip install -r requirements.txt
```

## Etapas do Pipeline

### 1. Preparação dos Dados (`prepare_data.py`)

Este script lida com a preparação do dataset MNIST:
- Faz o download do dataset MNIST usando TensorFlow
- Normaliza os valores dos pixels para o intervalo [0,1]
- Redimensiona as imagens para incluir a dimensão do canal
- Salva os dados processados como arrays NumPy em `src/data/`

```bash
python src/prepare_data.py
```

Arquivos gerados:
- `x_train.npy`: Imagens de treino
- `y_train.npy`: Rótulos de treino
- `x_test.npy`: Imagens de teste
- `y_test.npy`: Rótulos de teste

### 2. Treinamento dos Modelos (`train_model.py`)

Treina três diferentes arquiteturas CNN no dataset MNIST:
- `cnn_simple`: CNN básica com uma única camada convolucional
- `cnn_deeper`: Arquitetura mais profunda com múltiplas camadas convolucionais
- `cnn_dropout`: CNN com camadas de dropout para regularização

```bash
python src/train_model.py
```

Os modelos são salvos no diretório `src/models/keras_models/`.

Parâmetros de treinamento:
- `cnn_simple`: 10 épocas
- `cnn_deeper`: 15 épocas
- `cnn_dropout`: 20 épocas

### 3. Conversão para TFLite (`convert_to_tflite.py`)

Converte os modelos Keras treinados para o formato TensorFlow Lite com diferentes níveis de otimização:
- Float32 (sem quantização)
- Quantização Float16
- Quantização INT8 (usando dados de calibração)

```bash
python src/convert_to_tflite.py
```

Os modelos convertidos são salvos em `src/models/tflite_models/` com diferentes sufixos indicando o tipo de quantização.

### 4. Avaliação dos Modelos (`evaluate_tflite_models.py`)

Avalia o desempenho de todos os modelos TFLite:
- Mede a acurácia no conjunto de teste
- Calcula o tempo de inferência
- Gera relatórios detalhados de classificação
- Calcula precisão, recall e F1-score

```bash
python src/evaluate_tflite_models.py
```

Os resultados são salvos no diretório `src/results/`:
- `tflite_evaluation_summary.csv`: Métricas gerais de desempenho
- Relatórios de classificação individuais para cada modelo

## Análise dos Resultados

Os resultados da avaliação incluem:
- Comparação da acurácia dos modelos
- Tempo médio de inferência
- Tamanho do modelo após quantização
- Métricas detalhadas de classificação por classe

Essas informações ajudam na seleção da melhor variante do modelo para implantação em dispositivos com recursos limitados, equilibrando acurácia, tamanho e velocidade de inferência.

## Nota sobre Otimização dos Modelos

O projeto implementa três estratégias de quantização:
1. Sem quantização (float32) - Desempenho base
2. Quantização Float16 - Reduz o tamanho do modelo com impacto mínimo na acurácia
3. Quantização INT8 - Máxima redução de tamanho, pode afetar a acurácia

Escolha o modelo apropriado com base nas suas restrições de implantação e requisitos de acurácia.

## Fluxo de Trabalho Recomendado

1. Execute `prepare_data.py` primeiro para configurar o dataset
2. Treine os modelos usando `train_model.py`
3. Converta os modelos para TFLite usando `convert_to_tflite.py`
4. Avalie todos os modelos com `evaluate_tflite_models.py`
5. Revise os resultados no diretório `src/results/` para selecionar o melhor modelo para seu caso de uso
