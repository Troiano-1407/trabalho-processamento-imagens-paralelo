import argparse
import cv2
import numpy as np
from pathlib import Path
import time
from typing import Optional
from tqdm import tqdm

from filtros import obter_filtro, listar_filtros


def aplicar_filtro(img_path: Path, filtro: np.ndarray):
    """Aplica o filtro de convolucao em uma imagem."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return cv2.filter2D(img, -1, filtro)


def processar_imagens(
    input_dir: str,
    output_dir: str,
    filtro_nome: str = "blur",
    limite: Optional[int] = None,
):
    entrada = Path(input_dir)
    saida = Path(output_dir)
    saida.mkdir(parents=True, exist_ok=True)

    if not entrada.exists():
        raise FileNotFoundError(f"Diretorio de entrada '{entrada}' nao encontrado. Execute baixar_cifar.py primeiro.")
    if not entrada.is_dir():
        raise NotADirectoryError(f"'{entrada}' nao e um diretorio valido.")

    imagens = sorted(p for p in entrada.iterdir() if p.is_file())
    if limite:
        imagens = imagens[:limite]

    filtro = obter_filtro(filtro_nome)
    inicio = time.perf_counter()
    processadas = 0

    for img_path in tqdm(imagens, desc=f"Sequencial ({filtro_nome})", unit="img"):
        resultado = aplicar_filtro(img_path, filtro)
        if resultado is None:
            continue
        cv2.imwrite(str(saida / img_path.name), resultado)
        processadas += 1

    total = time.perf_counter() - inicio
    throughput = processadas / total if total else 0

    return {
        "tempo_total": total,
        "throughput": throughput,
        "imagens_processadas": processadas,
        "filtro": filtro_nome,
    }


def main():
    parser = argparse.ArgumentParser(description="Processamento sequencial de imagens CIFAR-10.")
    parser.add_argument("--input-dir", default="./imagens_cifar", help="Diretorio com as imagens de entrada.")
    parser.add_argument("--output-dir", default="./saida_sequencial", help="Diretorio para salvar as imagens processadas.")
    parser.add_argument(
        "--filter", choices=listar_filtros(), default="blur", help="Filtro de convolucao a ser aplicado."
    )
    parser.add_argument("--limit", type=int, default=None, help="Processa apenas N imagens (util para testes).")
    args = parser.parse_args()

    metricas = processar_imagens(args.input_dir, args.output_dir, args.filter, args.limit)
    print(f"Tempo total (sequencial): {metricas['tempo_total']:.2f}s")
    print(f"Throughput: {metricas['throughput']:.2f} imagens/s")
    print(f"Imagens processadas: {metricas['imagens_processadas']}")


if __name__ == "__main__":
    main()
