## Processamento de imagens em lote (CIFAR-10)

Este projeto implementa um pipeline completo para processar milhares de imagens da base CIFAR-10 aplicando filtros de convolucao (blur e edge detection), comparando execucao sequencial e paralela, e calculando metricas de desempenho (tempo total, throughput, speedup e eficiencia).

### Estrutura
- `processamento_imagens/baixar_cifar.py`: baixa a base CIFAR-10 via `torchvision` e exporta para PNG.
- `processamento_imagens/processamento_sequencial.py`: aplica filtros de convolucao de forma sequencial, medindo tempo e throughput.
- `processamento_imagens/processamento_paralelo.py`: mesma logica, mas usando multiprocessing para paralelizar o tratamento.
- `processamento_imagens/metricas.py`: executa ambos os modos e gera speedup/eficiencia agregados.
- `processamento_imagens/filtros.py`: dicionario centralizado com os kernels disponiveis.

### Dependencias
Python 3.9+ e os pacotes: `torchvision`, `torch`, `numpy`, `opencv-python`, `tqdm`, `pillow`.

Instale com:
```
pip install -r requirements.txt
```
ou instale manualmente os pacotes listados acima.

### Como executar
1. **Baixar imagens**  
   ```
   python processamento_imagens/baixar_cifar.py
   ```
   As imagens sao salvas em `./imagens_cifar/` (60k arquivos).

2. **Processar sequencialmente**  
   ```
   python processamento_imagens/processamento_sequencial.py --filter blur
   ```
   Parametros uteis:
   - `--filter {blur,edge}`
   - `--input-dir ./imagens_cifar`
   - `--output-dir ./saida_sequencial`
   - `--limit 1000` (limita o numero de imagens, util em testes)

3. **Processar em paralelo**  
   ```
   python processamento_imagens/processamento_paralelo.py --filter edge --workers 8
   ```
   Parametros adicionais:
   - `--workers`: define o numero de processos (padrao = `cpu_count()`).

4. **Gerar metricas consolidadas**  
   ```
   python processamento_imagens/metricas.py --filter blur --workers 8 --cores 8
   ```
   O script executa os modos sequencial e paralelo (respeitando `--limit`, quando informado) e imprime:
   - tempo total por modo
   - throughput (imagens/s)
   - speedup (seq/par)
   - eficiencia (speedup / cores)

### Observacoes
- Todos os caminhos sao relativos ao diretorio raiz do repositorio.
- Os filtros podem ser extendidos adicionando kernels em `processamento_imagens/filtros.py`.
- Para experimentos rapidos, use `--limit` para processar um subconjunto das 60k imagens.
