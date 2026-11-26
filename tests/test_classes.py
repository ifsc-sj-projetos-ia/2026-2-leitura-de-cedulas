from pathlib import Path

import pytest

from cedulas.classes import CLASSES, frase_para_falar, normalizar_rotulo
from cedulas.preparar import Amostra, dividir_por_grupo, grupo_propria, grupo_roboflow


@pytest.mark.parametrize(
    "rotulo, esperado",
    [
        ("2", "2_reais"),
        ("R$ 20", "20_reais"),
        ("20reais", "20_reais"),
        ("Real_200", "200_reais"),
        ("100-reais-verso", "100_reais"),
        ("vinte", "20_reais"),
        ("cem_reais", "100_reais"),
        ("50 centavos", None),
        ("moeda_1", None),
        ("1 real", None),
        ("background", None),
    ],
)
def test_normalizar_rotulo(rotulo, esperado):
    assert normalizar_rotulo(rotulo) == esperado


def test_frase():
    assert frase_para_falar("50_reais") == "Nota de 50 reais"
    assert "Não consegui" in frase_para_falar(None)


def test_grupos():
    assert grupo_roboflow(Path("IMG_01_jpg.rf.abc123.jpg")) == "IMG_01_jpg"
    assert grupo_propria(Path("data/proprias/20/n03_frente_02.jpg")) == "proprias/20/n03"


def test_split_nao_vaza_grupos():
    amostras = [
        Amostra(Path(f"{c}_{g}_{i}.jpg"), c, "x", f"{c}/{g}") for c in CLASSES for g in range(10) for i in range(3)
    ]
    splits = dividir_por_grupo(amostras)
    grupos = {s: {a.grupo for a in lista} for s, lista in splits.items()}
    assert not grupos["train"] & grupos["val"]
    assert not grupos["train"] & grupos["test"]
    assert not grupos["val"] & grupos["test"]
    for c in CLASSES:  # toda classe aparece em todos os splits
        for lista in splits.values():
            assert any(a.classe == c for a in lista)
