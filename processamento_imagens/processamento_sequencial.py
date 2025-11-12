import cv2
import numpy as np
import os
import time
from tqdm import tqdm

def aplicar_filtro(img_path, filtro):
    img = cv2.imread(img_path)
    if img is None:
        return
    resultado = cv2.filter2D(img, -1, filtro)
    return resultado

blur = np.ones((3,3), np.float32) / 9
edge_detection = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])

input_dir = "./imagens_cifar"
output_dir = "./saida_sequencial"
os.makedirs(output_dir, exist_ok=True)

inicio = time.time()
for nome in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, nome)
    result = aplicar_filtro(img_path, blur)
    if result is not None:
        cv2.imwrite(os.path.join(output_dir, nome), result)
fim = time.time()

total = fim - inicio
throughput = len(os.listdir(input_dir)) / total

print(f"Tempo total (sequencial): {total:.2f}s")
print(f"Throughput: {throughput:.2f} imagens/s")


