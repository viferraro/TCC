import torch.nn as nn
import graphviz
import os

## Defina um BasicBlock genérico (para simplificar o diagrama)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        # ... (forward pass - não precisa para o diagrama)
        pass

# Cria um modelo dummy para extrair a estrutura (substitua pelo seu modelo real)
class ModeloResNetSimplificado(nn.Module):
    def __init__(self):
        super(ModeloResNetSimplificado, self).__init__()  # Inicializa a classe pai, nn.Module
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)  # Primeira camada convolucional
        self.bn1 = nn.BatchNorm2d(64)  # Normalização em lote para acelerar o treinamento
        self.relu = nn.ReLU(inplace=True)  # Função de ativação ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Camada de agrupamento máximo
        self.layer1 = self._make_layer(64, 64, 3)  # Primeiro bloco residual
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # Segundo bloco residual
        self.layer3 = self._make_layer(128, 256, 6, stride=2)  # Terceiro bloco residual
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # Quarto bloco residual
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Camada de agrupamento médio adaptativo
        self.fc = nn.Linear(512, 10)  # Camada totalmente conectada para classificação

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        # Verifica se é necessário uma camada de downsample
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                # Camada convolucional para downsample
                nn.BatchNorm2d(planes),  # Normalização em lote
            )

        layers = []
        layers.append(BasicBlock(inplanes, planes, stride,
                                 downsample))  # Adiciona o primeiro bloco com downsample se necessário
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(inplanes, planes))  # Adiciona os blocos restantes

        return nn.Sequential(*layers)  # Retorna os blocos como uma sequência

    def forward(self, x):
        # ... (forward pass - não precisa para o diagrama)
        pass


modelo = ModeloResNetSimplificado()

# Cria o grafo do Graphviz
dot = graphviz.Digraph(comment='ResNet Architecture (Com Informações nos Blocos)', graph_attr={'rankdir': 'TB'})

# Função auxiliar para adicionar blocos
def adicionar_bloco(nome, label, anterior=None):
    dot.node(nome, label, shape='box')
    if anterior:
        dot.edge(anterior, nome)
    return nome

# Adiciona os blocos conceituais (COM INFORMAÇÕES NOS BLOCOS)
anterior = adicionar_bloco('input', 'Entrada (28x28x1)')

# Camada Inicial (com detalhes)
anterior = adicionar_bloco('camada_inicial', 'Camada Inicial\n(Conv2d + BN + ReLU + MaxPool2d)', anterior)

# Blocos Residuais (com informações importantes)
anterior = adicionar_bloco('bloco1', 'Bloco 1 (x3)\n(BasicBlock:\nConv2d + BN + ReLU\nConv2d + BN + ReLU)', anterior)
anterior = adicionar_bloco('bloco2', 'Bloco 2 (x4)\n(BasicBlock:\nConv2d + BN + ReLU\nConv2d + BN + ReLU)', anterior)
anterior = adicionar_bloco('bloco3', 'Bloco 3 (x6)\n(BasicBlock:\nConv2d + BN + ReLU\nConv2d + BN + ReLU)', anterior)
anterior = adicionar_bloco('bloco4', 'Bloco 4 (x3)\n(BasicBlock:\nConv2d + BN + ReLU\nConv2d + BN + ReLU)', anterior)

# Pooling e Classificação
adicionar_bloco('pooling_classificacao', 'Pooling e Classificação\n(AdaptiveAvgPool2d + Softmax)', anterior)

# ... (Função criar_diretorio_incrementado e salvamento do diagrama - iguais ao código anterior)
def criar_diretorio_incrementado(diretorio_base, nome_subpasta):
    contador = 1
    diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    while os.path.exists(diretorio_pai):
        contador += 1
        diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    os.makedirs(diretorio_pai)
    return diretorio_pai

# Salva o diagrama
diretorio_pai = criar_diretorio_incrementado('Arch_resnet', 'resnetDiagrama')
nome_arquivo = os.path.join(diretorio_pai, "resnet_diagram_resumido")
dot.render(nome_arquivo, view=True)

print(f"Diagrama salvo em: {nome_arquivo}.pdf")