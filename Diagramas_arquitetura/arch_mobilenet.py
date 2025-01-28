import torch.nn as nn
import graphviz
import os

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=32, bias=False), # Depthwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False), # Pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=64, bias=False), # Downsampling Depthwise
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False), # Pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


modelo = MobileNet()

# Cria o grafo do Graphviz
dot = graphviz.Digraph(comment='MobileNet Architecture (Simplificado)', graph_attr={'rankdir': 'TB'})

# Função auxiliar para adicionar blocos
def adicionar_bloco(nome, label, anterior=None):
    dot.node(nome, label, shape='box')
    if anterior:
        dot.edge(anterior, nome)
    return nome

# Adiciona os blocos conceituais (correspondendo à descrição textual)
anterior = adicionar_bloco('input', 'Entrada (28x28x1)')

anterior = adicionar_bloco('camada_inicial', 'Camada Inicial\n(Conv + BN + ReLU)', anterior)

anterior = adicionar_bloco('bloco_ds1', 'Bloco Depthwise Separable 1\n(Depthwise Conv + Pointwise Conv)', anterior)

anterior = adicionar_bloco('bloco_ds2', 'Bloco Depthwise Separable 2\n(Depthwise Conv + Pointwise Conv)', anterior)

anterior = adicionar_bloco('pooling_classificacao', 'Subamostragem e Classificação\n(AdaptiveAvgPool2d + Flatten + Softmax)', anterior)

def criar_diretorio_incrementado(diretorio_base, nome_subpasta):
    contador = 1
    diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    while os.path.exists(diretorio_pai):
        contador += 1
        diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    os.makedirs(diretorio_pai)
    return diretorio_pai

# Salva o diagrama
diretorio_pai = criar_diretorio_incrementado('Arch_mobilenet', 'mobilenetDiagrama')
nome_arquivo = os.path.join(diretorio_pai, "mobilenet_diagram_texto")
dot.render(nome_arquivo, view=True)

print(f"Diagrama salvo em: {nome_arquivo}.pdf")
