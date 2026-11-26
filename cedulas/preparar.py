"""Junta Roboflow + fotos próprias num único dataset de classificação.

Entradas:
    data/raw/<dataset>/      datasets baixados pelo baixar_roboflow.py
    data/proprias/<valor>/   suas fotos (veja data/proprias/README.md)

Saída:
    data/processed/{train,val,test}/<classe>/*.jpg   formato do Ultralytics (classificação)
    data/processed/manifest.csv                     origem, grupo e split de cada imagem

Uso:
    python -m cedulas.preparar
    python -m cedulas.preparar --proprias-so-teste   # experimento de mudança de domínio
"""

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageOps

from cedulas.classes import CLASSES, normalizar_rotulo

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LADO_MAXIMO = 640  # px; reduz espaço em disco sem perder nada para imgsz=224
LADO_MINIMO_RECORTE = 32  # caixas menores que isso são ruído
MARGEM_RECORTE = 0.05


@dataclass
class Amostra:
    origem: Path  # arquivo original (rastreabilidade)
    classe: str
    fonte: str  # nome do dataset ou "proprias"
    grupo: str  # imagens do mesmo grupo nunca vão para splits diferentes
    caixa: tuple[int, int, int, int] | None = None  # recorte da nota, se veio de detecção

    def carregar(self) -> Image.Image:
        """Abre sob demanda: manter milhares de imagens em memória não cabe num PC comum."""
        img = abrir(self.origem)
        if self.caixa:
            img = img.crop(self.caixa)
        img.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
        return img


def grupo_roboflow(caminho: Path) -> str:
    """Roboflow nomeia cópias aumentadas como "foto_jpg.rf.<hash>.jpg": agrupa pela foto original."""
    return caminho.name.split(".rf.")[0]


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES)


def abrir(caminho: Path) -> Image.Image:
    img = Image.open(caminho)
    return ImageOps.exif_transpose(img).convert("RGB")


def ler_deteccao(raiz: Path) -> list[Amostra]:
    """Dataset Roboflow em formato YOLO (detecção): recorta cada nota anotada."""
    nomes = yaml.safe_load((raiz / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(nomes, dict):
        nomes = [nomes[i] for i in sorted(nomes)]
    amostras = []
    ignorados = Counter()
    for caminho in listar_imagens(raiz):
        if caminho.parent.name != "images":
            continue
        rotulos = caminho.parent.parent / "labels" / f"{caminho.stem}.txt"
        if not rotulos.exists():
            continue
        with Image.open(caminho) as img:
            w, h = ImageOps.exif_transpose(img).size
        for linha in rotulos.read_text().splitlines():
            partes = linha.split()
            if len(partes) < 5:
                continue
            classe = normalizar_rotulo(str(nomes[int(partes[0])]))
            if classe is None:
                ignorados[nomes[int(partes[0])]] += 1
                continue
            coords = [float(v) for v in partes[1:]]
            if len(coords) == 4:
                cx, cy, bw, bh = coords
            else:  # polígono (segmentação): usa a caixa envolvente
                xs, ys = coords[0::2], coords[1::2]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            bw, bh = bw * (1 + 2 * MARGEM_RECORTE), bh * (1 + 2 * MARGEM_RECORTE)
            caixa = (
                max(0, int((cx - bw / 2) * w)),
                max(0, int((cy - bh / 2) * h)),
                min(w, int((cx + bw / 2) * w)),
                min(h, int((cy + bh / 2) * h)),
            )
            if min(caixa[2] - caixa[0], caixa[3] - caixa[1]) < LADO_MINIMO_RECORTE:
                continue
            amostras.append(Amostra(caminho, classe, raiz.name, grupo_roboflow(caminho), caixa))
    if ignorados:
        print(f"  rótulos ignorados em {raiz.name}: {dict(ignorados)}")
    return amostras


def ler_pastas(raiz: Path, fonte: str, agrupar) -> list[Amostra]:
    """Qualquer estrutura .../<rotulo>/<imagem>: usada para Roboflow 'folder' e fotos próprias."""
    amostras = []
    ignorados = Counter()
    for caminho in listar_imagens(raiz):
        rotulo = caminho.relative_to(raiz).parts
        # procura, de dentro para fora, a primeira pasta que seja um rótulo válido
        classe = next((c for c in map(normalizar_rotulo, reversed(rotulo[:-1])) if c), None)
        if classe is None:
            ignorados[caminho.parent.name] += 1
            continue
        amostras.append(Amostra(caminho, classe, fonte, agrupar(caminho)))
    if ignorados:
        print(f"  pastas ignoradas em {fonte}: {dict(ignorados)}")
    return amostras


def grupo_propria(caminho: Path) -> str:
    """Fotos próprias: "n03_frente_02.jpg" -> grupo "proprias/20_reais/n03" (a mesma cédula física)."""
    cedula = re.split(r"[_\-. ]", caminho.stem)[0]
    return f"proprias/{caminho.parent.name}/{cedula}"


def dhash(img: Image.Image, tamanho: int = 8) -> int:
    """Hash perceptual simples para remover imagens duplicadas entre datasets."""
    cinza = img.convert("L").resize((tamanho + 1, tamanho), Image.Resampling.BILINEAR)
    px = list(cinza.getdata())
    bits = 0
    for y in range(tamanho):
        for x in range(tamanho):
            i = y * (tamanho + 1) + x
            bits = (bits << 1) | (px[i] > px[i + 1])
    return bits


def remover_duplicadas(amostras: list[Amostra]) -> list[Amostra]:
    vistos = {}
    unicas = []
    for a in amostras:
        h = (dhash(a.carregar()), a.classe)
        if h in vistos:
            continue
        vistos[h] = a
        unicas.append(a)
    print(f"Duplicadas removidas: {len(amostras) - len(unicas)}")
    return unicas


def dividir_por_grupo(
    amostras: list[Amostra], proporcoes=(0.7, 0.15, 0.15), semente: int = 42
) -> dict[str, list[Amostra]]:
    """Split estratificado por classe e agrupado (nenhum grupo aparece em dois splits)."""
    rng = random.Random(semente)
    splits = {"train": [], "val": [], "test": []}
    por_classe = defaultdict(lambda: defaultdict(list))
    for a in amostras:
        por_classe[a.classe][a.grupo].append(a)
    for classe in sorted(por_classe):
        grupos = sorted(por_classe[classe])
        rng.shuffle(grupos)
        n = len(grupos)
        n_train = round(n * proporcoes[0])
        n_val = round(n * proporcoes[1])
        if n >= 3:  # garante pelo menos um grupo em val e test
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))
        for i, g in enumerate(grupos):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            splits[split].extend(por_classe[classe][g])
    return splits


def salvar(splits: dict[str, list[Amostra]], saida: Path) -> None:
    if saida.exists():
        shutil.rmtree(saida)
    linhas = []
    for split, amostras in splits.items():
        for c in CLASSES:  # cria todas as pastas, mesmo vazias: o Ultralytics usa a ordem delas
            (saida / split / c).mkdir(parents=True, exist_ok=True)
        for i, a in enumerate(amostras):
            img = a.carregar()
            destino = saida / split / a.classe / f"{a.fonte}_{i:06d}.jpg"
            img.save(destino, quality=92)
            linhas.append([str(destino), a.classe, a.fonte, a.grupo, split, str(a.origem)])
    with open(saida / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "classe", "fonte", "grupo", "split", "origem"])
        w.writerows(linhas)


def resumo(splits: dict[str, list[Amostra]]) -> None:
    print(f"\n{'classe':<12}" + "".join(f"{s:>8}" for s in splits))
    for c in CLASSES:
        print(f"{c:<12}" + "".join(f"{sum(a.classe == c for a in splits[s]):>8}" for s in splits))
    fontes = Counter((a.fonte, s) for s, lista in splits.items() for a in lista)
    print("\npor fonte:", dict(sorted(fontes.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--proprias", default="data/proprias")
    parser.add_argument("--saida", default="data/processed")
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument(
        "--proprias-so-teste",
        action="store_true",
        help="usa as fotos próprias só no teste (mede generalização para fotos reais)",
    )
    args = parser.parse_args()

    externas = []
    for raiz in sorted(p for p in Path(args.raw).glob("*") if p.is_dir()):
        print(f"Lendo {raiz}")
        if (raiz / "data.yaml").exists():
            externas += ler_deteccao(raiz)
        else:
            externas += ler_pastas(raiz, raiz.name, grupo_roboflow)

    proprias = []
    if Path(args.proprias).exists():
        print(f"Lendo {args.proprias}")
        proprias = ler_pastas(Path(args.proprias), "proprias", grupo_propria)

    externas = remover_duplicadas(externas)
    if not externas and not proprias:
        raise SystemExit("Nenhuma imagem encontrada. Rode o baixar_roboflow.py ou adicione fotos em data/proprias/.")

    if args.proprias_so_teste:
        splits = dividir_por_grupo(externas, semente=args.semente)
        splits["test"] += proprias
    else:
        splits = dividir_por_grupo(externas + proprias, semente=args.semente)

    salvar(splits, Path(args.saida))
    resumo(splits)
    print(f"\nDataset salvo em {args.saida}")


if __name__ == "__main__":
    main()
