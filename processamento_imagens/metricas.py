tempo_seq = 545.92     
tempo_par = 42.11     
n_cores = 6          

speedup = tempo_seq / tempo_par
eficiencia = speedup / n_cores
throughput_seq = 60000 / tempo_seq   
throughput_par = 60000 / tempo_par

print(f"Speedup: {speedup:.2f}x")
print(f"Eficiência: {eficiencia*100:.2f}%")
print(f"Throughput Sequencial: {throughput_seq:.2f} imagens/s")
print(f"Throughput Paralelo: {throughput_par:.2f} imagens/s")
