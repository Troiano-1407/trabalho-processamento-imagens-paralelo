import argparse
import cv2
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
from typing import Optional
from tqdm import tqdm

from filtros import obter_filtro, listar_filtros

FILTER_KERNEL = None


def _init_worker(kernel):
    global FILTER_KERNEL
    FILTER_KERNEL = kernel


def aplicar_filtro(img_path: str):
    imagem = cv2.imread(img_path)
    if imagem is None or FILTER_KERNEL is None:
        return None
    resultado = cv2.filter2D(imagem, -1, FILTER_KERNEL)
    return Path(img_path).name, resultado


def processar_imagens(
    input_dir: str,
    output_dir: str,
    filtro_nome: str = "blur",
    limite: Optional[int] = None,
    workers: Optional[int] = None,
):
    entrada = Path(input_dir)
    saida = Path(output_dir)
    saida.mkdir(parents=True, exist_ok=True)

    if not entrada.exists():
        raise FileNotFoundError(f"Diretorio de entrada '{entrada}' nao encontrado. Execute baixar_cifar.py primeiro.")
    if not entrada.is_dir():
        raise NotADirectoryError(f"'{entrada}' nao e um diretorio valido.")

    imagens = sorted(str(p) for p in entrada.iterdir() if p.is_file())
    if limite:
        imagens = imagens[:limite]

    if not imagens:
        raise ValueError(f"Nenhuma imagem encontrada em {entrada}")

    kernel = obter_filtro(filtro_nome)
    processos = workers or cpu_count()
    inicio = time.perf_counter()
    processadas = 0

    with Pool(processes=processos, initializer=_init_worker, initargs=(kernel,)) as pool:
        for resultado in tqdm(
            pool.imap(aplicar_filtro, imagens),
            total=len(imagens),
            desc=f"Paralelo ({filtro_nome})",
            unit="img",
        ):
            if resultado is None:
                continue
            nome, img = resultado
            cv2.imwrite(str(saida / nome), img)
            processadas += 1

    total = time.perf_counter() - inicio
    throughput = processadas / total if total else 0

    return {
        "tempo_total": total,
        "throughput": throughput,
        "imagens_processadas": processadas,
        "filtro": filtro_nome,
        "workers": processos,
    }


def main():
    parser = argparse.ArgumentParser(description="Processamento paralelo de imagens CIFAR-10.")
    parser.add_argument("--input-dir", default="./imagens_cifar", help="Diretorio com as imagens de entrada.")
    parser.add_argument("--output-dir", default="./saida_paralelo", help="Diretorio para salvar as imagens processadas.")
    parser.add_argument(
        "--filter", choices=listar_filtros(), default="blur", help="Filtro de convolucao a ser aplicado."
    )
    parser.add_argument("--limit", type=int, default=None, help="Processa apenas N imagens (util para testes).")
    parser.add_argument("--workers", type=int, default=None, help="Quantidade de processos do Pool. Padrao = cpu_count().")
    args = parser.parse_args()

    metricas = processar_imagens(args.input_dir, args.output_dir, args.filter, args.limit, args.workers)
    print(f"Tempo total (paralelo): {metricas['tempo_total']:.2f}s")
    print(f"Throughput: {metricas['throughput']:.2f} imagens/s")
    print(f"Imagens processadas: {metricas['imagens_processadas']}")
    print(f"Processos utilizados: {metricas['workers']}")


if __name__ == "__main__":
    main()
