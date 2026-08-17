# Notas de estudo — decisões técnicas do pipeline

Este arquivo documenta o porquê de cada decisão não óbvia do projeto, no
nível de detalhe que eu preciso pra defender numa entrevista técnica. Cada
seção linka pro código correspondente.

## 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)

**O que é.** Equalização de histograma comum (`cv2.equalizeHist`) redistribui
os níveis de cinza da imagem inteira pra que o histograma fique mais
uniforme, aumentando contraste global. O problema é que ela usa um único
histograma pra imagem inteira: se a chapa tem uma região mais iluminada que
outra (reflexo de lâmpada, sombra de suporte de fixação — comum em célula de
inspeção industrial), a equalização global amplifica ruído na região que já
estava clara e faz pouco pela região escura.

CLAHE resolve isso de duas formas:
1. **Adaptativo**: divide a imagem numa grade de blocos (`tileGridSize`, uso
   8x8) e equaliza o histograma de cada bloco separadamente, depois
   interpola nas bordas entre blocos pra não criar descontinuidade visível.
2. **Contrast limited**: antes de equalizar cada bloco, corta (clipa) o
   histograma num limite (`clipLimit`, uso 2.0) e redistribui o excedente.
   Sem isso, uma região quase uniforme (fundo homogêneo da chapa) teria um
   pico enorme no histograma, e equalizar sem limite amplificaria ruído de
   sensor nessa região pra um contraste artificial enorme.

**Por que entra antes do threshold e não depois.** A ordem no pipeline é
cinza → denoise (mediana) → CLAHE → segmentação (`preproc.py`). CLAHE
amplifica contraste local, incluindo o contraste de ruído — por isso a
mediana vem antes, pra tirar ruído de sal-e-pimenta do sensor primeiro. E
CLAHE precisa vir antes do threshold adaptativo (não depois) porque o
próprio threshold adaptativo já lida com iluminação não-uniforme calculando
um limiar local — as duas técnicas atacam o mesmo problema em estágios
diferentes: CLAHE normaliza o *contraste* pra que a diferença entre defeito
e fundo fique mais parecida em toda a imagem, o threshold adaptativo decide
o *limiar* local de forma independente da média global. Uma sem a outra
funciona pior: só threshold adaptativo sem CLAHE ainda perde defeito sutil
numa região de baixo contraste local; só CLAHE sem threshold adaptativo
ainda sofre com iluminação inclinada numa escala maior que o tile.

## 2. Threshold global vs. threshold adaptativo

Threshold global (`cv2.threshold` com um valor fixo) compara cada pixel
contra um único limiar pra imagem inteira. Funciona bem quando a iluminação
é uniforme e há separação clara entre o histograma do fundo e do objeto de
interesse. Quebra quando a iluminação varia espacialmente: um limiar que
separa bem defeito de fundo numa região da chapa pode classificar a região
mais escura inteira como "defeito" (falsos positivos maciços) ou a região
mais clara inteira como "fundo" (defeito sumindo).

Threshold adaptativo (`cv2.adaptiveThreshold`, uso o método
`ADAPTIVE_THRESH_GAUSSIAN_C`) resolve isso calculando um limiar *por pixel*,
baseado na média (ponderada por gaussiana, no meu caso) de uma vizinhança
local de tamanho `blockSize`, subtraída de uma constante `C`. Cada pixel é
comparado contra o "fundo local" ao redor dele, não contra uma média global.
O trade-off é que fica mais sensível a dois parâmetros que precisam ser
calibrados pro tamanho típico do defeito que se quer detectar:

- `blockSize` muito pequeno (perto do tamanho do próprio defeito): o defeito
  entra na própria vizinhança usada pra calcular o limiar local, e a
  comparação perde sentido — vi isso na prática (`segmentacao.py`): com
  `blockSize=25` e um defeito redondo de raio ~13-19px, a máscara saía como
  um **anel oco** em vez de disco cheio, porque só a borda do defeito (onde
  o gradiente é grande o bastante pra vencer o `C`) passava no limiar.
  Subi pra `blockSize=35` e o problema diminuiu bastante.
- `C` (a constante subtraída da média local) muito baixo deixa o threshold
  sensível demais a variação de textura fina — testei `C=5` e a máscara
  saía com mais de 20% dos pixels da imagem marcados como "defeito", pura
  textura de fundo. Subir pra `C=15` derrubou isso pra menos de 1%.

## 3. Abertura e fechamento morfológico

Ambos operam sobre uma máscara binária usando um elemento estruturante
(aqui, elipse) que "varre" a imagem.

- **Erosão**: um pixel só continua branco se toda a vizinhança sob o
  elemento estruturante também for branca. Encolhe regiões brancas, apaga
  ruído pequeno isolado.
- **Dilatação**: o oposto — um pixel vira branco se *qualquer* pixel da
  vizinhança for branco. Cresce regiões brancas, fecha buracos pequenos.
- **Abertura** = erosão seguida de dilatação. Remove ruído pequeno isolado
  (pontinhos que sobraram do threshold) sem encolher permanentemente as
  regiões grandes, porque a dilatação que vem depois "devolve" o tamanho
  original das regiões que sobreviveram à erosão.
- **Fechamento** = dilatação seguida de erosão. Fecha pequenas lacunas e
  reconecta fragmentos próximos, sem inchar permanentemente o tamanho das
  regiões, pelo mesmo motivo inverso.

No pipeline (`segmentacao.py`), aplico abertura com kernel pequeno (3x3)
primeiro, pra tirar o ruído de textura que sobra do threshold sem apagar um
risco fino de poucos pixels de espessura — testei 5x5 e ele já começava a
cortar riscos finos ao meio. Depois aplico fechamento com kernel maior (9x9)
pra reconectar a borda do Canny com a máscara do threshold e, no caso dos
defeitos redondos que saem como anel (ver seção 2), fechar esse anel numa
forma única e conectada — sem isso, o `findContours` via enxergar o anel
como múltiplos arcos desconectados, cada um pequeno demais pra passar no
filtro de área mínima.

## 4. Circularidade e solidez

Ambas vêm do contorno externo do candidato a defeito (`features.py`).

**Circularidade** = `4π × área / perímetro²`. Um círculo perfeito tem
circularidade exatamente 1 (é a forma que minimiza perímetro pra uma dada
área). Qualquer desvio de um círculo — alongar, criar reentrâncias, deixar a
borda irregular — aumenta o perímetro em relação à área e derruba esse
valor pra baixo de 1. Na prática, um risco fino e longo tem circularidade
bem baixa (~0.1-0.3, medido nos meus dados sintéticos) porque o perímetro
cresce quase linear com o comprimento enquanto a área cresce devagar (é
"quase uma linha"); um furo de rebite ausente, que é aproximadamente
circular, fica na faixa de 0.7-0.9.

**Solidez** = `área do contorno / área do fecho convexo (convex hull) do
mesmo contorno`. O fecho convexo é o menor polígono convexo que envolve
todos os pontos do contorno — imagine esticar um elástico ao redor da forma.
Solidez próxima de 1 significa que a forma já é quase convexa (poucas
reentrâncias). Solidez baixa indica uma forma com "baías" — reentrâncias que
ficam de fora da forma mas dentro do fecho convexo. Uma mancha de corrosão
com borda irregular (que no meu gerador sintético vem de vários lóbulos
circulares sobrepostos, ver `dataset.py`) tem solidez mais baixa que um
furo, que é aproximadamente convexo por natureza.

Uso as duas juntas (e não uma sozinha) porque elas capturam tipos diferentes
de "desvio de circularidade": um octógono regular teria solidez alta
(quase convexo) mas circularidade menor que 1; uma estrela de cinco pontas
teria as duas baixas. Combinadas, dão uma discriminação melhor entre "forma
compacta e convexa" (furo) e "forma irregular" (mancha) do que qualquer uma
sozinha.

## 5. Como calculo o IoU (Intersection over Union)

IoU mede sobreposição entre duas caixas (aqui, bounding boxes axis-aligned):

```
IoU = área da interseção / área da união
    = área da interseção / (área(A) + área(B) - área da interseção)
```

Implementação em `avaliacao.py:calcular_iou`: acho o retângulo de
interseção pegando o máximo dos cantos superiores-esquerdos e o mínimo dos
cantos inferiores-direitos das duas caixas; se a largura ou altura resultante
for negativa (caixas não se sobrepõem), a interseção é zero. A união eu
calculo por inclusão-exclusão (soma das áreas menos a interseção, pra não
contar a região sobreposta duas vezes).

Uso IoU ≥ 0.3 como limiar mínimo pra considerar que uma detecção "achou" um
defeito do ground truth (`casar_deteccoes_com_gt`). O valor mais comum na
literatura de detecção de objetos é 0.5 (padrão do Pascal VOC), mas usei um
limiar mais permissivo porque meu objetivo aqui é avaliar *classificação*
sobre candidatos já segmentados, não a qualidade da segmentação em si — um
contorno de risco fino que capturou 80% do comprimento real já é uma
detecção útil pra triagem, mesmo com IoU abaixo de 0.5 por causa da
diferença de espessura entre o contorno detectado e o ground truth sintético.

## 6. Por que precisão e recall contam histórias diferentes

- **Precisão** = TP / (TP + FP): das vezes que o sistema disse "achei um
  defeito", quantas vezes ele realmente tinha achado um. Precisão baixa
  significa alarme falso — inspetor perde tempo verificando peça boa.
- **Recall** = TP / (TP + FN): dos defeitos que realmente existem, quantos o
  sistema encontrou. Recall baixo significa defeito passando despercebido.

Os dois competem: forçar mais candidatos a virar detecção positiva
(threshold de segmentação mais sensível, classificador menos conservador)
tende a subir recall e derrubar precisão, e vice-versa. Qual métrica
importa mais depende inteiramente do custo de cada tipo de erro — que é o
assunto da próxima seção.

## 7. Por que falso negativo custa mais caro que falso positivo aqui

Num cenário de inspeção de segurança (estrutural, aeronáutico), as duas
classes de erro têm custo assimétrico:

- **Falso positivo** (marcar uma peça boa como defeituosa): custo é
  operacional — inspetor humano gasta alguns minutos verificando e libera a
  peça. Chato, mas reversível e barato.
- **Falso negativo** (deixar passar um defeito real): custo é a peça
  seguir na linha de produção com um defeito não detectado. Num componente
  estrutural, isso pode virar propagação de trinca por fadiga, falha em
  serviço, e no limite um problema de segurança de voo. O custo não é
  simétrico com o do falso positivo — não dá pra comparar "minutos de
  verificação extra" com "risco de falha estrutural em serviço".

Por isso, num pipeline de triagem automática pra inspeção de segurança, a
escolha de operação (limiar de decisão, parâmetros do classificador) deve
favorecer recall alto mesmo à custa de precisão mais baixa — é aceitável
(e esperado) que o sistema gere mais alarmes falsos que um humano
revisaria, desde que a taxa de defeito real não detectado fique o mais
baixa possível. Esse sistema automático nunca deveria ser a última linha de
defesa sozinho: ele é uma pré-triagem que reduz a carga do inspetor humano,
não um substituto pra ele. A tabela de métricas no README reflete isso: o
classificador de ML tem recall mais alto que o de regras em duas das três
classes de defeito, o que pesa a favor dele mesmo nos casos onde a precisão
das duas abordagens é parecida.
