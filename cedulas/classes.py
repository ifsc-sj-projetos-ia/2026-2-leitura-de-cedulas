"""Classes do problema e normalização dos rótulos vindos de datasets diferentes."""

import re

VALORES = [2, 5, 10, 20, 50, 100, 200]
CLASSES = [f"{v}_reais" for v in VALORES]

# Abaixo deste nível de confiança o app não fala um valor: pede para tentar de novo.
# Calibrado com a curva cobertura x acurácia gerada pelo avaliar.py.
LIMIAR_CONFIANCA = 0.80

_POR_EXTENSO = {
    "dois": 2,
    "cinco": 5,
    "dez": 10,
    "vinte": 20,
    "cinquenta": 50,
    "cem": 100,
    "duzentos": 200,
}


def normalizar_rotulo(nome: str) -> str | None:
    """Converte rótulos como "R$ 20", "20reais", "vinte", "Real_200" na classe canônica.

    Retorna None para o que não é cédula válida (moedas, centavos, fundo...).
    """
    texto = nome.lower().strip()
    if any(p in texto for p in ("centavo", "moeda", "coin", "cent")):
        return None
    numeros = re.findall(r"\d+", texto)
    if numeros:
        valor = int(numeros[0])
    else:
        palavras = re.findall(r"[a-z]+", texto)
        valor = next((_POR_EXTENSO[p] for p in palavras if p in _POR_EXTENSO), None)
    if valor not in VALORES:
        return None
    return f"{valor}_reais"


def valor_da_classe(classe: str) -> int:
    return int(classe.split("_")[0])


def frase_para_falar(classe: str | None) -> str:
    if classe is None:
        return "Não consegui identificar. Aproxime a nota da câmera e tente de novo."
    return f"Nota de {valor_da_classe(classe)} reais"
