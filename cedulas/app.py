"""Demo: aponte a câmera do celular para a nota e ouça o valor.

A voz é gerada pelo próprio navegador (Web Speech API, pt-BR): funciona no
celular sem instalar nada e sem enviar o áudio para nenhum servidor.

Uso:
    python -m cedulas.app --pesos runs/classify/yolo11n/weights/best.pt --compartilhar

--compartilhar cria um link HTTPS público temporário (gradio.live). O celular só
libera a câmera em páginas HTTPS, então é assim que se abre o app no telefone.
"""

import argparse

import gradio as gr

from cedulas.classes import LIMIAR_CONFIANCA, frase_para_falar
from cedulas.inferencia import Leitor

FALAR_JS = """
(frase) => {
    if (!frase || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const fala = new SpeechSynthesisUtterance(frase);
    fala.lang = "pt-BR";
    fala.rate = 0.95;
    window.speechSynthesis.speak(fala);
}
"""

CSS = """
#resultado textarea { font-size: 2.2rem !important; font-weight: 700; text-align: center; }
#identificar { font-size: 1.6rem; min-height: 4.5rem; }
"""


def criar_app(leitor: Leitor) -> gr.Blocks:
    def identificar(imagem):
        if imagem is None:
            return "Nenhuma imagem. Tire uma foto da nota.", {}
        leitura = leitor.ler(imagem)
        probs = {k.replace("_reais", " reais"): v for k, v in leitura.probabilidades.items()}
        return frase_para_falar(leitura.classe), probs

    with gr.Blocks(title="Leitor de cédulas", css=CSS) as app:
        gr.Markdown("# Leitor de cédulas\nFotografe a nota inteira, de perto e com boa luz. O valor é falado em voz alta.")
        imagem = gr.Image(
            sources=["upload", "webcam"],  # no celular, "upload" abre a câmera traseira nativa
            type="pil",
            label="Foto da nota",
        )
        botao = gr.Button("Identificar nota", variant="primary", elem_id="identificar")
        resultado = gr.Textbox(label="Resultado", elem_id="resultado", interactive=False)
        with gr.Accordion("Detalhes (probabilidades)", open=False):
            probs = gr.Label(num_top_classes=3, label="Probabilidades")

        for evento in (botao.click, imagem.upload):
            evento(identificar, inputs=imagem, outputs=[resultado, probs]).then(
                fn=None, inputs=resultado, js=FALAR_JS
            )
        botao_repetir = gr.Button("Repetir áudio")
        botao_repetir.click(fn=None, inputs=resultado, js=FALAR_JS)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pesos", required=True)
    parser.add_argument("--limiar", type=float, default=LIMIAR_CONFIANCA)
    parser.add_argument("--compartilhar", action="store_true", help="gera link HTTPS para abrir no celular")
    args = parser.parse_args()

    criar_app(Leitor(args.pesos, args.limiar)).launch(share=args.compartilhar)


if __name__ == "__main__":
    main()
