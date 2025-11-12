from torchvision.datasets import CIFAR10
from torchvision import transforms
import os
from PIL import Image

saida = "./imagens_cifar"
os.makedirs(saida, exist_ok=True)

# Baixar conjunto de treino
trainset = CIFAR10(root="./cifar_data", train=True, download=True, transform=transforms.ToTensor())
for i, (img, label) in enumerate(trainset):
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(f"{saida}/train_{i}.png")

# Baixar conjunto de teste
testset = CIFAR10(root="./cifar_data", train=False, download=True, transform=transforms.ToTensor())
for i, (img, label) in enumerate(testset):
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(f"{saida}/test_{i}.png")
