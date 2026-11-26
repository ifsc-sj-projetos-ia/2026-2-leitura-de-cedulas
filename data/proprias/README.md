# Fotos próprias

Fotos tiradas por você com as suas notas. Elas são o teste mais honesto do
modelo: celular real, iluminação real, mão real — diferente das imagens
"limpas" dos datasets públicos.

As imagens **não são versionadas** (estão no `.gitignore`). Guarde-as no Google
Drive e copie para cá antes de rodar `python -m cedulas.preparar`.

## Estrutura

```
data/proprias/
├── 2/   n01_frente_01.jpg  n01_verso_01.jpg  n02_frente_01.jpg ...
├── 5/
├── 10/
├── 20/
├── 50/
├── 100/
└── 200/
```

- A pasta é o valor da nota (também aceita `2_reais`, `R$2` etc.).
- O arquivo começa com o **identificador da cédula física** (`n01`, `n02`...)
  seguido de `_`. Todas as fotos da mesma nota ficam no mesmo split
  (treino, validação ou teste) — senão o modelo "decora" aquela nota e a
  métrica fica otimista.

## Protocolo de coleta (sugestão)

Para cada cédula física, tirar ~10 fotos variando:

| Variável     | Variações                                               |
|--------------|---------------------------------------------------------|
| Lado         | frente, verso                                           |
| Luz          | luz do dia, lâmpada amarela, pouca luz                   |
| Fundo        | mesa clara, mesa escura, na mão                         |
| Enquadramento| nota inteira, nota parcialmente cortada, dobrada ao meio |
| Estado       | nota nova, nota velha/amassada (se tiver)               |

Meta mínima: 3 cédulas diferentes por valor (pedir emprestado R$100/R$200 se
precisar) → ~30 fotos por valor. A família antiga do Real (pré-2010) também
circula; se tiver, inclua e anote no README.

**Privacidade:** não fotografe rostos, documentos ou outras informações
pessoais junto com a nota.
