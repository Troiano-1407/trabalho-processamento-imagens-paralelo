# Substitua pelos seus valores reais
tempo_seq = 47.99     # tempo do processamento_sequencial.py
tempo_par = 40.64     # tempo do processamento_paralelo.py
n_cores = 6          # número de núcleos do seu processador

# Cálculos
speedup = tempo_seq / tempo_par
eficiencia = speedup / n_cores
throughput_seq = 60000 / tempo_seq   # se usou todas as imagens do CIFAR-10
throughput_par = 60000 / tempo_par

# Exibir resultados
print(f"Speedup: {speedup:.2f}x")
print(f"Eficiência: {eficiencia*100:.2f}%")
print(f"Throughput Sequencial: {throughput_seq:.2f} imagens/s")
print(f"Throughput Paralelo: {throughput_par:.2f} imagens/s")
