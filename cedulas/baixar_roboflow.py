"""Baixa os datasets listados em configs/datasets.yaml para data/raw/.

Uso:
    export ROBOFLOW_API_KEY=...   # https://app.roboflow.com/settings/api
    python -m cedulas.baixar_roboflow
"""

import argparse
import os
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--saida", default="data/raw")
    args = parser.parse_args()

    chave = os.environ.get("ROBOFLOW_API_KEY")
    if not chave:
        raise SystemExit("Defina a variável de ambiente ROBOFLOW_API_KEY.")

    from roboflow import Roboflow

    rf = Roboflow(api_key=chave)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    for ds in config["datasets"]:
        destino = Path(args.saida) / f"{ds['project']}-v{ds['version']}"
        if destino.exists():
            print(f"[pulando] {destino} já existe")
            continue
        print(f"[baixando] {ds['workspace']}/{ds['project']} v{ds['version']} ({ds['formato']})")
        versao = rf.workspace(ds["workspace"]).project(ds["project"]).version(ds["version"])
        versao.download(ds["formato"], location=str(destino))


if __name__ == "__main__":
    main()
