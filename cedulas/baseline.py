"""Baseline simples: histograma de cor (HSV) + Regressão Logística.

Cada cédula do Real tem uma cor dominante (R$2 azul, R$5 lilás, R$10 vermelho,
R$20 amarelo, R$50 laranja/marrom, R$100 azul, R$200 cinza/sépia). A pergunta é:
só a cor já basta? Se não, o ganho da rede neural mostra o que ela aprende além disso
(o 2 e o 100, ambos azulados, são o caso interessante).

Uso:
    python -m cedulas.baseline
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cedulas.classes import CLASSES

BINS = (12, 4, 4)  # matiz, saturação, valor


def histograma_hsv(caminho: str) -> np.ndarray:
    img = Image.open(caminho).convert("RGB").resize((128, 64)).convert("HSV")
    hsv = np.asarray(img).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=BINS, range=((0, 256), (0, 256), (0, 256)))
    return (hist / hist.sum()).ravel()


def carregar_split(manifest: Path, split: str):
    with open(manifest, encoding="utf-8") as f:
        linhas = [l for l in csv.DictReader(f) if l["split"] == split]
    X = np.stack([histograma_hsv(l["arquivo"]) for l in linhas])
    y = np.array([CLASSES.index(l["classe"]) for l in linhas])
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--saida", default="results/baseline")
    args = parser.parse_args()

    X_train, y_train = carregar_split(Path(args.manifest), "train")
    X_test, y_test = carregar_split(Path(args.manifest), "test")

    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    rotulos = list(range(len(CLASSES)))
    print(classification_report(y_test, pred, labels=rotulos, target_names=CLASSES, zero_division=0))
    metricas = {
        "modelo": "histograma HSV + regressão logística",
        "acuracia": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, labels=rotulos, average="macro", zero_division=0),
        "n_teste": int(len(y_test)),
        "matriz_confusao": confusion_matrix(y_test, pred, labels=rotulos).tolist(),
        "classes": CLASSES,
    }
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"acurácia={metricas['acuracia']:.3f}  f1_macro={metricas['f1_macro']:.3f}  -> {saida}/metricas.json")


if __name__ == "__main__":
    main()
