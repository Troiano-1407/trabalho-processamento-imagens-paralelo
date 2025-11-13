import numpy as np

FILTROS = {
    "blur": np.ones((3, 3), dtype=np.float32) / 9,
    "edge": np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    ),
}


def obter_filtro(nome: str):
    if nome not in FILTROS:
        raise ValueError(f"Filtro '{nome}' nao suportado. Opcoes: {list(FILTROS)}")
    return FILTROS[nome]


def listar_filtros():
    return list(FILTROS.keys())
