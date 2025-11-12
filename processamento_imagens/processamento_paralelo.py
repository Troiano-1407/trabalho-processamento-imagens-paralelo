import cv2
import numpy as np
import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def aplicar_filtro(img_path):
    filtro = np.ones((3,3), np.float32) / 9  
    imagem = cv2.imread(img_path)
    if imagem is None:
        return None
    resultado = cv2.filter2D(imagem, -1, filtro)
    return img_path, resultado


if __name__ == "__main__":  
    input_dir = "./imagens_cifar"
    output_dir = "./saida_paralelo"
    os.makedirs(output_dir, exist_ok=True)

    imagens = [os.path.join(input_dir, nome) for nome in os.listdir(input_dir)]

    inicio = time.time()

    with Pool(cpu_count()) as p:
        for result in tqdm(p.imap(aplicar_filtro, imagens), total=len(imagens)):
            if result is not None:
                path, img = result
                nome = os.path.basename(path)
                cv2.imwrite(os.path.join(output_dir, nome), img)

    fim = time.time()
    total = fim - inicio
    throughput = len(imagens) / total

    print(f"Tempo total (paralelo): {total:.2f}s")
    print(f"Throughput: {throughput:.2f} imagens/s")
