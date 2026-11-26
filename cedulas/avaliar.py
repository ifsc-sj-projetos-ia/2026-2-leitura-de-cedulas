"""Avalia um modelo no split de teste e gera os artefatos para o README.

Gera em results/<nome>/:
    metricas.json        acurácia, F1 macro, por classe e por fonte (Roboflow x fotos próprias)
    matriz_confusao.png
    limiar.png           cobertura x acurácia conforme o limiar de confiança
    erros.csv            imagens erradas, para análise de erros

Uso:
    python -m cedulas.avaliar --pesos runs/classify/yolo11n/weights/best.pt --nome yolo11n
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from cedulas.classes import CLASSES
from cedulas.inferencia import Leitor


def curva_limiar(confs: np.ndarray, acertos: np.ndarray) -> list[dict]:
    """Para cada limiar: fração de leituras faladas (cobertura) e acurácia entre elas."""
    pontos = []
    for limiar in np.round(np.arange(0.0, 1.0, 0.05), 2):
        falou = confs >= limiar
        pontos.append(
            {
                "limiar": float(limiar),
                "cobertura": float(falou.mean()),
                "acuracia_quando_fala": float(acertos[falou].mean()) if falou.any() else None,
            }
        )
    return pontos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pesos", required=True)
    parser.add_argument("--nome", required=True)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        linhas = [l for l in csv.DictReader(f) if l["split"] == args.split]

    leitor = Leitor(args.pesos, limiar=0.0)
    y_true, y_pred, confs, fontes, erros = [], [], [], [], []
    for l in linhas:
        leitura = leitor.ler(l["arquivo"])
        y_true.append(l["classe"])
        y_pred.append(leitura.classe)
        confs.append(leitura.confianca)
        fontes.append(l["fonte"])
        if leitura.classe != l["classe"]:
            erros.append([l["arquivo"], l["classe"], leitura.classe, f"{leitura.confianca:.3f}", l["fonte"]])

    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    acertos = np.array([t == p for t, p in zip(y_true, y_pred)])

    por_fonte = defaultdict(lambda: ([], []))
    for t, p, fonte in zip(y_true, y_pred, fontes):
        por_fonte[fonte][0].append(t)
        por_fonte[fonte][1].append(p)

    metricas = {
        "pesos": args.pesos,
        "split": args.split,
        "n": len(y_true),
        "acuracia": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "f1_por_classe": dict(
            zip(CLASSES, f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0).tolist())
        ),
        "por_fonte": {
            fonte: {"n": len(t), "acuracia": accuracy_score(t, p)} for fonte, (t, p) in sorted(por_fonte.items())
        },
        "curva_limiar": curva_limiar(np.array(confs), acertos),
    }

    saida = Path("results") / args.nome
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))

    with open(saida / "erros.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "verdadeiro", "previsto", "confianca", "fonte"])
        w.writerows(erros)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=[c.replace("_reais", "") for c in CLASSES]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_xlabel("previsto (R$)")
    ax.set_ylabel("verdadeiro (R$)")
    ax.set_title(f"{args.nome} — acurácia {metricas['acuracia']:.1%}")
    fig.tight_layout()
    fig.savefig(saida / "matriz_confusao.png", dpi=150)

    curva = metricas["curva_limiar"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p["limiar"] for p in curva], [p["cobertura"] for p in curva], label="cobertura (fala um valor)")
    ax.plot(
        [p["limiar"] for p in curva],
        [p["acuracia_quando_fala"] or np.nan for p in curva],
        label="acurácia quando fala",
    )
    ax.set_xlabel("limiar de confiança")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(saida / "limiar.png", dpi=150)

    print(f"acurácia={metricas['acuracia']:.3f}  f1_macro={metricas['f1_macro']:.3f}")
    for fonte, m in metricas["por_fonte"].items():
        print(f"  {fonte:<30} n={m['n']:<5} acurácia={m['acuracia']:.3f}")
    print(f"Resultados em {saida}/")


if __name__ == "__main__":
    main()
