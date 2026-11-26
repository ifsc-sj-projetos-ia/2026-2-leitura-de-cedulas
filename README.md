# Leitor de cédulas para pessoas cegas

A câmera do celular identifica a cédula do Real (R$2 a R$200) e **fala o valor em voz alta**.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ifsc-sj-projetos-ia/2026-2-leitura-de-cedulas/blob/main/notebooks/cedulas_colab.ipynb)
![Licença MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-green)

Projeto Prático de IA — ICD / ADS, IFSC São José, 2026/2.

---

## 1. Problema e motivação

O Brasil tem cerca de 6,5 milhões de pessoas com deficiência visual severa (Censo IBGE 2010). A
segunda família do Real tem marcas táteis e tamanhos diferentes por valor, mas
na prática elas se gastam com o uso, e as diferenças de tamanho são de poucos
milímetros: distinguir uma nota de R$20 de uma de R$50 pelo tato é difícil, e
depender de outra pessoa para conferir o troco expõe a pessoa a erros e golpes.

**O que o sistema faz:** a pessoa aponta o celular para a nota, toca em um botão
grande e ouve "Nota de 50 reais". Se o modelo não estiver confiante, ele diz
que não conseguiu identificar em vez de chutar — **falar um valor errado é pior
do que não falar nada**.

## 2. Dados

| Fonte | Conteúdo | Uso |
|---|---|---|
| [Reconhecedor de Cédulas - Real](https://universe.roboflow.com/anagirelli-alunos-utfpr-edu-br/reconhecedor-de-cedulas-real) (v1, CC BY 4.0) | 2.478 imagens, 7 valores × frente/verso, sem augmentation | treino/val/teste |
| [Reconhecimento de Real](https://universe.roboflow.com/gabriel-pereira-aakjt/reconhecimento-de-real) (v2, CC BY 4.0) | 622 imagens, 7 valores × frente/verso + moedas | treino/val/teste |
| [Brazilian currency](https://universe.roboflow.com/elainesilva/brazilian-currency) (v12, CC BY 4.0) | 1.694 imagens (~565 originais com augmentation ×3), 7 valores + moedas | treino/val/teste |
| Fotos próprias (`data/proprias/`) | fotos de celular de notas reais, várias condições de luz/fundo | treino/val/teste ou só teste |

Os três datasets são de detecção (caixas) e exportados redimensionados para 640×640 com
"stretch", o que distorce a proporção da nota; o recorte herda essa distorção. As moedas são descartadas.
Licença CC BY 4.0: uso livre com atribuição aos autores (links acima).

**Variável alvo:** 7 classes — `2_reais`, `5_reais`, `10_reais`, `20_reais`, `50_reais`, `100_reais`, `200_reais`.

**Tratamento** (`cedulas/preparar.py`):

1. **Unificação de rótulos** — cada dataset nomeia as classes de um jeito (`R$ 20`, `vinte`, `Real_200`...).
   `normalizar_rotulo()` converte tudo para a classe canônica e descarta moedas/centavos.
2. **Recorte** — datasets de detecção viram classificação recortando cada nota anotada (com 5% de margem;
   caixas < 32 px são descartadas).
3. **Deduplicação** — hash perceptual (dHash) remove a mesma foto repetida em datasets diferentes.
4. **Split agrupado e estratificado** (70/15/15) — cópias aumentadas do Roboflow
   (`foto_jpg.rf.<hash>.jpg`) e fotos da mesma cédula física (`n03_...`) ficam sempre
   no mesmo split. Sem isso, o modelo vê "a mesma foto" no treino e no teste e a métrica fica inflada.
5. **Manifest** — `data/processed/manifest.csv` registra origem, fonte, grupo e split de cada imagem.

> _Preencher após rodar:_ tabela com a contagem por classe e por split (o `preparar.py` imprime).

## 3. Modelo e resultados

| Modelo | Ideia | Acurácia | F1 macro | Acurácia nas fotos próprias |
|---|---|---|---|---|
| Baseline: histograma HSV + regressão logística | só a cor da nota | — | — | — |
| YOLO11n-cls (fine-tuning, ImageNet) | CNN leve, roda em celular | — | — | — |
| YOLO11s-cls (fine-tuning, ImageNet) | CNN maior | — | — | — |
| YOLO11n-cls treinado só com Roboflow | mede mudança de domínio | — | — | — |

_Preencher com `results/*/metricas.json`._

**Por que classificação e não detecção?** O usuário enquadra uma nota por vez;
classificar a imagem inteira é mais simples, mais rápido e precisa de menos dados.

**Escolhas de augmentation** (`cedulas/treinar.py`): a matiz (`hsv_h`) **não** é alterada,
porque a cor é uma pista forte do valor; espelhamento desligado (nota espelhada não existe);
apagamento aleatório (`erasing`) simula dedos cobrindo a nota; brilho/saturação variam
para simular iluminação.

**Limiar de confiança:** `results/<modelo>/limiar.png` mostra, para cada limiar, quantas
leituras o app fala (cobertura) e a acurácia entre elas. O limiar do app
(`LIMIAR_CONFIANCA` em `cedulas/classes.py`) é escolhido nessa curva.

**Análise de erros:** `results/<modelo>/erros.csv` e a matriz de confusão. _Preencher: quais pares
de notas mais se confundem e por quê._

## 4. Como executar

### Opção A — Google Colab (recomendado, tem GPU)

Clique no botão "Abrir no Colab" acima e execute as células em ordem. O notebook faz tudo:
download, preparação, baseline, treino, avaliação e abre a demo com link para o celular.

### Opção B — Local

```bash
git clone https://github.com/ifsc-sj-projetos-ia/2026-2-leitura-de-cedulas.git
cd 2026-2-leitura-de-cedulas
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export ROBOFLOW_API_KEY=...                 # https://app.roboflow.com/settings/api
python -m cedulas.baixar_roboflow           # -> data/raw/
python -m cedulas.preparar                  # -> data/processed/
python -m cedulas.baseline                  # -> results/baseline/
python -m cedulas.treinar --nome yolo11n    # -> runs/classify/yolo11n/  (lento sem GPU)
python -m cedulas.avaliar --pesos runs/classify/yolo11n/weights/best.pt --nome yolo11n
python -m cedulas.app --pesos runs/classify/yolo11n/weights/best.pt --compartilhar
pytest                                      # testes de rótulos e do split
```

### Demo no celular

`--compartilhar` gera um link `https://….gradio.live`. Abra no celular, toque em
**Foto da nota** (abre a câmera traseira), fotografe e toque em **Identificar nota**.
A voz é sintetizada pelo próprio navegador (pt-BR). Se o iPhone não falar na primeira
vez, toque em **Repetir áudio** (o Safari exige um toque para liberar o som).

## 5. Estrutura

```
cedulas/
  classes.py          classes, normalização de rótulos, frase falada, limiar
  baixar_roboflow.py  download dos datasets (configs/datasets.yaml)
  preparar.py         recorte, deduplicação, split agrupado, manifest
  baseline.py         histograma HSV + regressão logística
  treinar.py          fine-tuning YOLO11-cls
  avaliar.py          métricas, matriz de confusão, curva de limiar, erros
  inferencia.py       carrega o modelo e lê uma imagem
  app.py              demo Gradio com voz
configs/datasets.yaml
data/proprias/        suas fotos (não versionadas) + protocolo de coleta
docs/relatorio.md     escopo, requisitos e análise ética
notebooks/            notebook do Colab
results/              métricas e gráficos de cada experimento
tests/
```

## 6. Ética e limitações

Resumo — detalhes em [docs/relatorio.md](docs/relatorio.md):

- **Não detecta notas falsas.** Só diz o valor.
- Dados públicos têm viés para notas novas, bem iluminadas e fundo limpo; notas gastas,
  rasgadas e a primeira família do Real são sub-representadas.
- Erro silencioso é o pior caso: por isso o limiar de confiança e a resposta "não consegui identificar".
- Não foi testado com usuários cegos — isso é o próximo passo mais importante.

## 7. Próximos passos

- Testes com pessoas cegas (ex.: associações locais) para validar o fluxo de uso.
- Modo contínuo (lê sozinho enquanto a câmera está aberta, sem precisar tocar em botões).
- Exportar para TFLite/ONNX e rodar 100% offline no celular.
- Classe "não é nota" para evitar leituras de outros objetos.

## Licença

Código sob [MIT](LICENSE). Os datasets do Roboflow seguem as licenças dos respectivos autores (ver tabela de dados).
