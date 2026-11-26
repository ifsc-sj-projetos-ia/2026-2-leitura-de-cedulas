# Relatório — Leitor de cédulas para pessoas cegas

## 1. Escopo

**Problema:** pessoas cegas ou com baixa visão têm dificuldade em identificar o
valor das cédulas do Real sem ajuda de terceiros.

**Solução proposta:** protótipo que recebe uma foto tirada com o celular,
classifica a cédula em um de 7 valores (R$2, 5, 10, 20, 50, 100, 200) e fala
o resultado em voz alta, em português.

**Dentro do escopo**
- Cédulas da 2ª família do Real (em circulação desde 2010) e R$200 (2020).
- Uma nota por foto, frente ou verso.
- Uso pelo navegador do celular, com conexão à internet.

**Fora do escopo**
- Moedas.
- Detecção de notas falsas.
- Várias notas na mesma foto / soma de valores.
- Funcionamento offline (está nos próximos passos).

## 2. Requisitos funcionais

| ID | Requisito |
|---|---|
| RF1 | O sistema deve receber uma imagem da câmera do celular ou de um arquivo. |
| RF2 | O sistema deve classificar a imagem em um dos 7 valores de cédula. |
| RF3 | O sistema deve falar o valor reconhecido em voz alta, em português ("Nota de 20 reais"). |
| RF4 | Se a confiança for menor que o limiar, o sistema deve dizer que não conseguiu identificar e orientar a tentar de novo, **sem falar um valor**. |
| RF5 | O usuário deve poder repetir o último áudio. |
| RF6 | A interface deve ter botões grandes e poucos elementos, navegáveis com leitor de tela. |

**Requisitos não funcionais:** resposta em até ~2 s; acurácia ≥ 95% nas fotos
próprias de teste entre as leituras faladas; código e notebook reproduzíveis
(semente fixa, split salvo em `manifest.csv`).

## 3. Análise ética

### Viés dos dados
- **Condição das notas:** datasets públicos têm muitas notas novas, planas e bem
  iluminadas. Notas velhas, dobradas, rasgadas ou rabiscadas — comuns no troco
  — aparecem pouco. Pessoas de baixa renda, que lidam mais com dinheiro físico
  e notas gastas, podem ter pior desempenho.
- **Desbalanceamento:** R$200 é rara (circula pouco) e tem menos exemplos.
- **Câmeras:** fotos dos datasets e as fotos próprias vêm de poucos modelos de
  celular; celulares mais simples (câmera pior, menos luz) podem errar mais.
- **Família antiga do Real:** ainda tem validade legal e circula, mas quase não
  está nos dados.
- **Mitigação:** fotos próprias em condições variadas, métricas separadas por
  fonte (Roboflow × fotos próprias) e análise dos erros por classe.

### Limitações
- O modelo **não sabe dizer "isto não é uma nota"**: qualquer imagem vira um dos
  7 valores. O limiar de confiança reduz, mas não elimina, esse risco.
- **Não verifica autenticidade.** Uma nota falsa ou uma foto de nota será lida
  como verdadeira.
- A demo depende de internet (servidor Gradio) e a qualidade da voz depende do
  celular.
- **Não foi avaliado com usuários cegos.** A usabilidade real (enquadrar a nota
  sem ver a tela) é a maior incerteza do projeto.

### Cuidados no uso
- **Erro silencioso é o pior caso:** confundir R$50 com R$5 causa prejuízo
  direto. Por isso o sistema prefere dizer "não consegui identificar" a chutar,
  e o limiar foi escolhido priorizando acurácia sobre cobertura.
- Deixar claro ao usuário que é um **auxílio**, não uma garantia; em transações
  de valor alto, confirmar por outro meio.
- **Privacidade:** a imagem é processada no servidor do app e não é armazenada;
  a voz é gerada no próprio navegador. Fotos próprias do dataset não contêm
  rostos nem dados pessoais. O link `gradio.live` é público e temporário: não
  deve ser usado como serviço permanente.
- **Acessibilidade de verdade exige participação:** o desenho final deveria ser
  feito com pessoas cegas, não apenas para elas.

## 4. Cronograma

| Semana | Atividades | Entregas |
|---|---|---|
| 1 — Escopo e dados | escolher datasets no Roboflow; coletar fotos próprias (protocolo em `data/proprias/README.md`); inspecionar rótulos | `configs/datasets.yaml`, fotos no Drive |
| 2 — Preparação e baseline | `preparar.py` (recorte, deduplicação, split agrupado); baseline HSV + regressão logística | `results/baseline/` |
| 3 — Modelagem | YOLO11n × YOLO11s; experimento de domínio (treino só Roboflow → teste fotos próprias); análise de erros; escolha do limiar | `results/*/`, tabela do README |
| 4 — Finalização | README com resultados; demo no celular; slides e ensaio | repositório final, apresentação |
