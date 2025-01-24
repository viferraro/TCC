import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from thop import profile
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from carbontracker.tracker import CarbonTracker
from carbontracker import parser
import pynvml

# Constante para inicialização do gerador de números aleatórios (usada para reprodutibilidade,
# embora não esteja sendo usada diretamente aqui)
SEED = 10

# Verifica se a GPU está disponível e define o dispositivo de treinamento
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dispositivo)

# Inicializa o NVML (NVIDIA Management Library) para monitoramento do consumo de energia da GPU
pynvml.nvmlInit()

# Função para criar um diretório com incremento numérico no nome, caso já exista um diretório com o mesmo nome base
def criar_diretorio_incrementado(diretorio_base, nome_subpasta):
    contador = 1
    diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    while os.path.exists(diretorio_pai):
        contador += 1
        diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    os.makedirs(diretorio_pai) # Cria o diretório
    return diretorio_pai

# Cria o diretório pai 'resultadosAlexNet/alexNetMNIST_X' (X é um número que incrementa se o diretório já existir)
diretorio_pai = criar_diretorio_incrementado('resultadosAlexNet', 'alexNetMNIST')
print(f'Diretório criado: {diretorio_pai}')

# Cria o subdiretório 'alexNet_carbono_X' dentro do diretório pai para os logs do Carbon Tracker
diretorio_carbon = criar_diretorio_incrementado(diretorio_pai, 'alexNet_carbono')
print(f'Diretório Carbono criado: {diretorio_carbon}')

# Definições iniciais para o treinamento
maximo_epocas = 50
tempos_treino = []
potencias_treino = [] # Lista para armazenar o consumo de energia a cada iteração de treino
# Inicializa o Carbon Tracker para monitorar o consumo de carbono durante o treinamento
tracker = CarbonTracker(epochs=maximo_epocas, monitor_epochs=-1, interpretable=True,
                           log_dir=f"./{diretorio_carbon}/",
                           log_file_prefix="cbt")

# Carrega o dataset MNIST e aplica transformações (normalização)
transformacao = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Normalização com média e desvio padrão do MNIST
])

conjunto_treino_completo = datasets.MNIST(root='./data', train=True, download=True, transform=transformacao)
conjunto_teste = datasets.MNIST(root='./data', train=False, download=True, transform=transformacao)

# Divide o conjunto de treino em treino e validação (80% para treino, 20% para validação)
tamanho_treino = int(0.8 * len(conjunto_treino_completo))
tamanho_validacao = len(conjunto_treino_completo) - tamanho_treino
conjunto_treino, conjunto_validacao = random_split(conjunto_treino_completo, [tamanho_treino, tamanho_validacao])

# Cria os DataLoaders para treino, validação e teste
carregador_treino = DataLoader(conjunto_treino, batch_size=32, shuffle=True) # shuffle=True embaralha os dados a cada época
carregador_validacao = DataLoader(conjunto_validacao, batch_size=32, shuffle=False)
carregador_teste = DataLoader(conjunto_teste, batch_size=32, shuffle=False)

# Definição da arquitetura da rede neural AlexNet (adaptada para o MNIST)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # Camada convolucional 1 (entrada: 1 canal (grayscale), saída: 64 canais)
            nn.ReLU(inplace=True), # Função de ativação ReLU
            nn.MaxPool2d(kernel_size=2), # Max Pooling
            nn.Conv2d(64, 192, kernel_size=3, padding=1), # Camada convolucional 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # Camada convolucional 3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # Camada convolucional 4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Camada convolucional 5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2)) # Adaptive Average Pooling para garantir tamanho de entrada consistente para o classificador
        self.classificador = nn.Sequential(
            nn.Dropout(), # Dropout para regularização
            nn.Linear(256 * 2 * 2, 4096), # Camada linear totalmente conectada
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), # Camada linear de saída (10 classes para o MNIST)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Achatamento da saída para entrada no classificador
        x = self.classificador(x)
        return x

# Função para inicializar os pesos da rede usando inicialização Kaiming Normal
def inicializar_pesos(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

# Classe para implementar Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

modelo = AlexNet().to(dispositivo) # Instancia o modelo e o move para o dispositivo (GPU ou CPU)
criterio = nn.CrossEntropyLoss() # Define a função de perda Cross-Entropy
otimizador = optim.Adam(modelo.parameters(), lr=0.0001, weight_decay=1e-4) # Define o otimizador Adam com taxa de aprendizado e weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, 'min', patience=3, factor=0.1) # Scheduler para reduzir a taxa de aprendizado quando o loss de validação estagna
#early_stopping = EarlyStopping(patience=5, min_delta=0.001)

print(modelo) # Imprime a arquitetura do modelo

modelo.apply(inicializar_pesos) # Aplica a inicialização de pesos ao modelo

# Captura o resumo do modelo usando torchsummary e o salva em um arquivo
saida_padrao_original = sys.stdout
sys.stdout = buffer = io.StringIO()
summary(modelo, (1, 28, 28))
resumo_str = buffer.getvalue()
sys.stdout = saida_padrao_original
with open(f'{diretorio_pai}/resumo_modelo.txt', 'w') as f:
    f.write(resumo_str)

# Redireciona a saída padrão para um arquivo para salvar os logs do treinamento
saida_padrao_original = sys.stdout
sys.stdout = open(f'{diretorio_pai}/saida.txt', 'w')

# Função para treinar e validar um modelo
def treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, epocas, i):
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    modelo.train()
    tempo_inicio = datetime.now()
    tracker.epoch_start()
    for epoca in range(epocas):
        tracker.epoch_start()
        perda_acumulada = 0.0
        corretos = 0
        total = 0
        for dados in carregador_treino: # Remove o índice 'i' desnecessário aqui
            entradas, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
            otimizador.zero_grad()
            saidas = modelo(entradas)
            perda = criterio(saidas, rotulos)
            perda.backward()

            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            perda_acumulada += perda.item()
            _, previstos = torch.max(saidas.data, 1)
            total += rotulos.size(0)
            corretos += (previstos == rotulos).sum().item()

            # Medir o consumo de energia a cada *batch*
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetPowerUsage(handle)
            consumo_energia = info / 1000.0
            potencias_treino.append(consumo_energia)

        perda_treino = perda_acumulada / len(carregador_treino)
        acuracia_treino = corretos / total
        print(f'Época {epoca + 1}, Perda Treino: {perda_treino:.4f}, Acurácia Treino: {acuracia_treino:.4f}')

        # Validação
        modelo.eval()
        perda_validacao = 0.0
        corretos = 0
        total = 0
        with torch.no_grad():
            for dados in carregador_validacao:
                imagens, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
                saidas = modelo(imagens)
                perda = criterio(saidas, rotulos)
                perda_validacao += perda.item()
                _, previstos = torch.max(saidas.data, 1)
                total += rotulos.size(0)
                corretos += (previstos == rotulos).sum().item()
        tracker.epoch_end()
        perda_validacao /= len(carregador_validacao)
        acuracia_validacao = corretos / total
        print(f'Época {epoca + 1}, Perda Validação: {perda_validacao:.4f}, Acurácia Validação: {acuracia_validacao:.4f}')
        scheduler.step(perda_validacao)

        # Verifique o Early Stopping
        early_stopping(perda_validacao)
        if early_stopping.early_stop:
            print(f"Early stopping na época {epoca + 1}")
            break

        # Salva o melhor modelo *desta execução* dentro da função treinar_e_validar
        if early_stopping.best_loss is None or perda_validacao < early_stopping.best_loss:
            early_stopping.best_loss = perda_validacao
            torch.save(modelo.state_dict(), f'{diretorio_pai}/melhor_modelo_{i}.pth')

    tempo_fim = datetime.now()
    tempo_treino = (tempo_fim - tempo_inicio)
    tempos_treino.append(tempo_treino.total_seconds())
    tracker.epoch_end()
    return perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia

# Treinamento e seleção do melhor modelo entre 'numero_modelos' candidatos
numero_modelos = 10
medias_acuracia_validacao = []
modelos = []
metricas = []
media_metricas = []

for i in range(numero_modelos):
    print("______________________________________________________________________________________________________")
    print(f'Treinando modelo {i + 1}/{numero_modelos}')
    entrada = torch.randn(1, 1, 28, 28).to(dispositivo) # Cria uma entrada dummy para o profile
    modelo = AlexNet().to(dispositivo)
    otimizador = optim.Adam(modelo.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, 'min', patience=3, factor=0.1)
    flops, parametros = profile(modelo, inputs=(entrada,), verbose=False)
    criterio = nn.CrossEntropyLoss()
    perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia = (
        treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, maximo_epocas, i))
    metricas.append((perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino.total_seconds(), consumo_energia))

    # Calcula a média das métricas *até o modelo atual*
    media_perda_treino = np.mean([m[0] for m in metricas])
    media_acuracia_treino = np.mean([m[1] for m in metricas])
    media_perda_validacao = np.mean([m[2] for m in metricas])
    media_acuracia_validacao = np.mean([m[3] for m in metricas])
    print(f'Modelo {i + 1}: Média Perda Treino: {media_perda_treino:.4f}, Média Acurácia Treino: {media_acuracia_treino:.4f}, '
          f'Média Perda Validação: {media_perda_validacao:.4f}, Média Acurácia Validação: {media_acuracia_validacao:.4f}')
    print(f'Tempo de treino: {tempo_treino}')
    print(f'FLOPs: {flops}')
    print(f'Parâmetros: {parametros}')
    print(f'Consumo de energia: {consumo_energia} W')

    medias_acuracia_validacao.append(acuracia_validacao) # Armazena *a acurácia de validação do modelo atual*
    media_metricas.append((media_perda_treino, media_acuracia_treino, media_perda_validacao, media_acuracia_validacao, tempo_treino.total_seconds(), consumo_energia))
    modelos.append(modelo)

# Cria um DataFrame com as métricas médias e salva em um arquivo Excel
df_metricas = pd.DataFrame(media_metricas, columns=['Média Perda Treino', 'Média Acurácia Treino', 'Média Perda Validação',
                                                    'Média Acurácia Validação', 'TempoTreino', 'ConsumoEnergia'])

# Adiciona uma coluna 'Modelo_x' ao DataFrame
nomes_modelos = ['Modelo_' + str(i + 1) for i in range(numero_modelos)]
df_metricas.insert(0, 'Modelo', nomes_modelos)

# Salva as métricas de todos os modelos em um único arquivo no diretório pai
df_metricas.to_excel(f'{diretorio_pai}/metricas_modelos.xlsx', index=False)

# Seleciona o melhor modelo com base na *maior acurácia de validação obtida individualmente* (não a média das médias)
indice_melhor_modelo = medias_acuracia_validacao.index(max(medias_acuracia_validacao))
melhor_modelo = modelos[indice_melhor_modelo]

print('************************************************************************************************')
print(f'O melhor modelo é o {nomes_modelos[indice_melhor_modelo]} com a maior acurácia de validação: {medias_acuracia_validacao[indice_melhor_modelo]:.4f}')
print('************************************************************************************************')

# Calcular a média dos tempos de treino e consumo de energia *de todos os modelos treinados*
media_tempo_treino = np.mean(tempos_treino)
media_consumo_energia = np.mean(potencias_treino)
print(f'Tempo Médio de Treino (todos os modelos): {media_tempo_treino:.2f} segundos') # Formatação para melhor visualização
print(f'Consumo Médio de Energia (durante o treino de todos os modelos): {media_consumo_energia:.2f} W') # Formatação para melhor visualização

# Carrega o estado do melhor modelo (salvo durante o treinamento)
melhor_modelo.load_state_dict(torch.load(f'{diretorio_pai}/melhor_modelo_{indice_melhor_modelo}.pth'))
melhor_modelo.to(dispositivo) # Garante que o modelo está no dispositivo correto para inferência

# Inicializa listas para armazenar métricas de todas as inferências
acuracias = []
precisoes = []
revocacoes = []
pontuacoes_f1 = []
tempos_teste = []
matrizes_confusao = [] # Lista para armazenar as matrizes de confusão de cada inferência

# Realiza 10 inferências e armazena as métricas
for _ in range(10): # Usando _ para indicar que o índice não é usado
    y_verdadeiros = []
    y_previstos = []
    inicio_tempo_teste = datetime.now()
    melhor_modelo.eval() # Importante: coloca o modelo em modo de avaliação
    with torch.no_grad(): # Desativa o cálculo de gradientes durante a inferência para economizar memória
        for dados in carregador_teste:
            imagens, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
            saidas = melhor_modelo(imagens)
            _, previstos = torch.max(saidas.data, 1) # Obtém as classes previstas
            y_verdadeiros.extend(rotulos.cpu().numpy()) # Move os rótulos para a CPU e os converte para numpy
            y_previstos.extend(previstos.cpu().numpy()) # Move as previsões para a CPU e as converte para numpy
    fim_tempo_teste = datetime.now()

    # Calcula as métricas para a inferência atual
    acuracias.append(accuracy_score(y_verdadeiros, y_previstos))
    precisoes.append(precision_score(y_verdadeiros, y_previstos, average='macro', zero_division=0))
    revocacoes.append(recall_score(y_verdadeiros, y_previstos, average='macro'))
    pontuacoes_f1.append(f1_score(y_verdadeiros, y_previstos, average='macro'))
    tempos_teste.append((fim_tempo_teste - inicio_tempo_teste).total_seconds())
    matrizes_confusao.append(confusion_matrix(y_verdadeiros, y_previstos)) # Armazena a matriz de confusão

# Calcula a média das métricas
media_acuracia = np.mean(acuracias)
media_precisao = np.mean(precisoes)
media_revocacao = np.mean(revocacoes)
media_f1 = np.mean(pontuacoes_f1)
media_tempo_teste = np.mean(tempos_teste)

# Imprime as médias das métricas com formatação
print(f'Média da Acurácia: {media_acuracia:.4f}')
print(f'Média da Precisão: {media_precisao:.4f}')
print(f'Média do Recall: {media_revocacao:.4f}')
print(f'Média do F1 Score: {media_f1:.4f}')
print(f'Média do Tempo de Teste: {media_tempo_teste:.4f} segundos')

sys.stdout.close()
sys.stdout = saida_padrao_original

# Calcular a matriz de confusão *média*
matriz_confusao_media = np.mean(matrizes_confusao, axis=0).astype(int) # Calcula a média das matrizes e converte para inteiro

# Plotar a matriz de confusão *média*
plt.figure(figsize=(10, 7))
sns.heatmap(matriz_confusao_media, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.xlabel('Previstos')
plt.ylabel('Verdadeiros')
plt.title('Matriz de Confusão Média') # Adiciona um título
plt.savefig(f'{diretorio_pai}/matriz_confusao_media.png') # Salva a matriz de confusão média
plt.close()

# Salva as médias das métricas em um arquivo
with open(f'{diretorio_pai}/metricas_medias_modelo.txt', 'w') as f:
    f.write(f'Média da Acurácia: {media_acuracia:.4f}\n')
    f.write(f'Média da Precisão: {media_precisao:.4f}\n')
    f.write(f'Média do Recall: {media_revocacao:.4f}\n')
    f.write(f'Média do F1 Score: {media_f1:.4f}\n')
    f.write(f'Média do Tempo de Teste: {media_tempo_teste:.4f} segundos\n')
    f.write(f'Tempo Médio de Treino (todos os modelos): {media_tempo_treino:.2f} segundos\n')
    f.write(f'Consumo Médio de Energia (durante o treino de todos os modelos): {media_consumo_energia:.2f} W\n')

pynvml.nvmlShutdown()
tracker.stop()
print('Treinamento concluído. Os resultados foram salvos nos arquivos especificados.')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')