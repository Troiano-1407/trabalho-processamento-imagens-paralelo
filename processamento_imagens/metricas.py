import argparse
from multiprocessing import cpu_count

from filtros import listar_filtros
from processamento_sequencial import processar_imagens as executar_sequencial
from processamento_paralelo import processar_imagens as executar_paralelo


def main():
    parser = argparse.ArgumentParser(description="Calcula metricas de processamento para CIFAR-10.")
    parser.add_argument("--input-dir", default="./imagens_cifar", help="Diretorio base com as imagens originais.")
    parser.add_argument("--output-seq", default="./saida_sequencial", help="Diretorio de saida do modo sequencial.")
    parser.add_argument("--output-par", default="./saida_paralelo", help="Diretorio de saida do modo paralelo.")
    parser.add_argument("--filter", default="blur", choices=listar_filtros(), help="Filtro utilizado nos testes.")
    parser.add_argument("--limit", type=int, default=None, help="Processa apenas N imagens (util para testes).")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Quantidade de processos para o modo paralelo (padrao = cpu_count()).",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=cpu_count(),
        help="Numero de nucleos disponiveis para calcular eficiencia (padrao = cpu_count()).",
    )
    args = parser.parse_args()

    print("=== Executando processamento sequencial ===")
    metricas_seq = executar_sequencial(args.input_dir, args.output_seq, args.filter, args.limit)

    print("\n=== Executando processamento paralelo ===")
    metricas_par = executar_paralelo(args.input_dir, args.output_par, args.filter, args.limit, args.workers)

    if metricas_par["tempo_total"] == 0:
        raise RuntimeError("Tempo total do processamento paralelo foi zero; nao e possivel calcular speedup.")

    speedup = metricas_seq["tempo_total"] / metricas_par["tempo_total"]
    eficiencia = speedup / args.cores if args.cores else 0

    print("\n=== Resumo das metricas ===")
    print(
        f"Sequencial -> tempo: {metricas_seq['tempo_total']:.2f}s | throughput: {metricas_seq['throughput']:.2f} img/s | imagens: {metricas_seq['imagens_processadas']}"
    )
    print(
        f"Paralelo    -> tempo: {metricas_par['tempo_total']:.2f}s | throughput: {metricas_par['throughput']:.2f} img/s | imagens: {metricas_par['imagens_processadas']} | workers: {metricas_par['workers']}"
    )
    print(f"Speedup: {speedup:.2f}x")
    print(f"Eficiencia: {eficiencia * 100:.2f}% (cores = {args.cores})")


if __name__ == "__main__":
    main()
