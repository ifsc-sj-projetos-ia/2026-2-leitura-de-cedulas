"""Carrega o modelo treinado e transforma uma imagem em (classe, confiança)."""

from dataclasses import dataclass

from PIL import Image

from cedulas.classes import LIMIAR_CONFIANCA


@dataclass
class Leitura:
    classe: str | None  # None quando a confiança fica abaixo do limiar
    confianca: float
    probabilidades: dict[str, float]


class Leitor:
    def __init__(self, pesos: str, limiar: float = LIMIAR_CONFIANCA):
        from ultralytics import YOLO

        self.modelo = YOLO(pesos)
        self.limiar = limiar

    def ler(self, imagem: Image.Image | str) -> Leitura:
        resultado = self.modelo.predict(imagem, verbose=False)[0]
        probs = resultado.probs.data.tolist()
        nomes = resultado.names  # índice -> nome da pasta (ordem alfabética do Ultralytics)
        probabilidades = {nomes[i]: p for i, p in enumerate(probs)}
        melhor = max(probabilidades, key=probabilidades.get)
        confianca = probabilidades[melhor]
        return Leitura(melhor if confianca >= self.limiar else None, confianca, probabilidades)
