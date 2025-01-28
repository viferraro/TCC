import torch
import torch.nn as nn
import graphviz
import os

# Definição da sua AlexNet (já corrigida com AdaptiveAvgPool2d((1, 1)))
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Função para criar um diretório com incremento, se necessário
def criar_diretorio_incrementado(diretorio_base, nome_subpasta):
    contador = 1
    diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    while os.path.exists(diretorio_pai):
        contador += 1
        diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    os.makedirs(diretorio_pai)
    return diretorio_pai

# Cria o diretório pai 'LeNetMNIST_' com incremento, se necessário
diretorio_pai = criar_diretorio_incrementado('Arch_leNet', 'leNetMNIST')
print(f'Diretório criado: {diretorio_pai}')

# Cria o modelo
modelo = AlexNet()

# Cria o grafo do Graphviz
dot = graphviz.Digraph(comment='AlexNet Architecture (Mais Simplificado)', graph_attr={'rankdir': 'TB'})

# Função auxiliar para adicionar nós e arestas
def adicionar_bloco(nome, label, anterior=None):
    dot.node(nome, label, shape='box')
    if anterior:
        dot.edge(anterior, nome)
    return nome

# Adiciona os blocos conceituais (COM JUNÇÃO DOS BLOCOS)
entrada = (28, 28, 1)
anterior = adicionar_bloco('input', f'Entrada ({entrada[0]}x{entrada[1]}x{entrada[2]})')

anterior = adicionar_bloco('conv1_2_block', 'Blocos Convolucionais 1 e 2\n(Conv + ReLU + MaxPool)\n(Conv + ReLU + MaxPool)', anterior) #Blocos 1 e 2 JUNTOS
anterior = adicionar_bloco('conv3_4_block', 'Blocos Convolucionais 3 e 4\n(Conv + ReLU)\n(Conv + ReLU)', anterior) #Blocos 3 e 4 JUNTOS
anterior = adicionar_bloco('conv5_block', 'Bloco Convolucional 5\n(Conv + ReLU + MaxPool)', anterior)
anterior = adicionar_bloco('avgpool', 'Pooling\n(AdaptiveAvgPool2d)', anterior)
anterior = adicionar_bloco('flatten', 'Flatten', anterior)

# Blocos Totalmente Conectados (com Dropout) - JUNTANDO OS DROPOUTS TAMBÉM
anterior = adicionar_bloco('fc1_block', 'Bloco Totalmente Conectado 1\n(Linear + ReLU + Dropout)', anterior)
anterior = adicionar_bloco('fc2_block', 'Bloco Totalmente Conectado 2\n(Linear + ReLU + Dropout)', anterior)
adicionar_bloco('fc3', 'Camada Linear Final\n(Softmax)', anterior)

# ... (Resto do código igual ao anterior - salvar o diagrama)

# Salva o diagrama
diretorio_pai = criar_diretorio_incrementado('Arch_alexNet', 'alexNetDiagrama')
nome_arquivo = os.path.join(diretorio_pai, "alexnet_diagram_blocos")
dot.render(nome_arquivo, view=True)

print(f"Diagrama salvo em: {nome_arquivo}.pdf")