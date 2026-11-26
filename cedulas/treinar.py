"""Fine-tuning de um classificador YOLO (Ultralytics) pré-treinado no ImageNet.

Uso (no Colab, com GPU):
    python -m cedulas.treinar --modelo yolo11n-cls.pt --epocas 30 --nome yolo11n
    python -m cedulas.treinar --modelo yolo11s-cls.pt --epocas 30 --nome yolo11s

O melhor peso fica em runs/classify/<nome>/weights/best.pt.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dados", default="data/processed")
    parser.add_argument("--modelo", default="yolo11n-cls.pt")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--nome", default="yolo11n")
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--dispositivo", default=None, help="ex.: 0 (GPU), cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    modelo = YOLO(args.modelo)
    modelo.train(
        data=args.dados,
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.nome,
        project="runs/classify",
        exist_ok=True,
        seed=args.semente,
        deterministic=True,
        patience=10,
        device=args.dispositivo,
        # Augmentation pensada para o problema:
        hsv_h=0.0,  # NÃO mudar a matiz: a cor é uma das pistas principais do valor
        hsv_s=0.4,  # saturação/brilho variam com a iluminação real
        hsv_v=0.4,
        fliplr=0.0,  # nota espelhada não existe no mundo real
        auto_augment="randaugment",  # inclui rotação/perspectiva: a nota nunca está perfeitamente reta
        erasing=0.3,  # dedos cobrindo parte da nota
    )


if __name__ == "__main__":
    main()
