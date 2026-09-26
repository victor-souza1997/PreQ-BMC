# Como provamos que uma rede neural embarcada é robusta: a metodologia explicada passo a passo

Este texto explica, sem pressupor conhecimento prévio, o que o nosso trabalho
faz, por que faz e como faz. Termos técnicos aparecem em **negrito** na primeira
vez e estão reunidos no glossário no fim.

---

## 1. O problema em uma frase

Um carro com câmera precisa reconhecer placas de trânsito ("Pare",
"Velocidade máxima 60", "Proibido ultrapassar"...). Quem faz esse
reconhecimento é um programa chamado **rede neural**. Queremos **provar
matematicamente** que o programa que de fato roda no carro não muda de resposta
quando a imagem sofre uma alteração minúscula.

Três palavras dessa frase importam muito:

- **provar**, e não apenas testar (seção 5);
- **o programa que de fato roda no carro**, e não uma versão idealizada dele
  (seção 4);
- **alteração minúscula**: a garantia vale para um tipo muito específico de
  perturbação, e não para qualquer coisa (seção 9).

---

## 2. O que é uma rede neural, de forma simplificada

Pense na rede neural como uma **receita gigante de contas**.

1. A imagem entra como uma lista de números. Cada pixel tem três cores
   (vermelho, verde e azul), e cada cor é um número de 0 a 255.
2. A rede multiplica esses números por outros números fixos, os **pesos**,
   soma tudo, faz alguns ajustes e repete isso várias vezes, em **camadas**.
3. No fim saem 43 números, um para cada tipo de placa que a rede conhece. É a
   "nota" que a rede dá para cada tipo.
4. A resposta é o tipo com a maior nota.

Os pesos não são escolhidos à mão. Eles são aprendidos mostrando à rede
dezenas de milhares de fotos de placas com a resposta certa: é o
**treinamento**.

Um dos ajustes entre as camadas se chama **ReLU**, e é bem simples: se o
número for negativo, vira zero; se for positivo, fica como está. Ele parece
bobo, mas é o que dá à rede sua capacidade e, como veremos, também o que torna
a verificação difícil.

No nosso caso:

- **Conjunto de dados:** GTSRB, um banco público alemão com mais de 50 mil
  fotos de placas de trânsito e 43 classes.
- **Rede:** uma rede convolucional de 5 camadas. **Convolucional** quer dizer
  que ela olha a imagem por pequenas janelas, como uma lupa que desliza sobre
  a foto.
- **Tamanho da entrada:** cada foto é reduzida para 32 × 32 pixels antes de
  entrar, ou seja, 3.072 números (32 × 32 × 3 cores).

---

## 3. Por que "quantizar": o computador do carro

Durante o treinamento, a rede usa **números com vírgula** (ponto flutuante,
como 0,7324518). Isso é confortável num computador potente, mas num carro há
motivos para preferir **números inteiros**:

- **Custo e energia:** as centrais eletrônicas de um carro são baratas e
  gastam pouca energia, e os aceleradores embarcados trabalham melhor com
  inteiros.
- **Previsibilidade:** contas com números com vírgula podem dar resultados
  ligeiramente diferentes em computadores ou compiladores diferentes. Contas
  com inteiros dão **exatamente o mesmo resultado em qualquer máquina**. Esse
  segundo motivo é o mais importante para nós: uma prova feita no nosso
  computador continua valendo na placa do carro, bit por bit.

**Quantizar** é converter a rede de números com vírgula para números inteiros.

### Analogia: centavos

Imagine que, em vez de guardar "R$ 1,50", você guarda "150 centavos". Você
trabalha só com inteiros, e basta lembrar que a unidade é o centavo.

Nós fazemos o mesmo, mas a "unidade" é 1/256 em vez de 1/100:

- o número 1,5 é guardado como 1,5 × 256 = **384**;
- o número 0,7324518 vira 0,7324518 × 256 ≈ 187,5, arredondado para **188**.

Esse formato se chama **Q7.8**: 16 bits no total, 8 deles para a "parte dos
centavos" (fração) e 7 para a parte inteira (mais o sinal). Ele representa
números de cerca de −128 até +127,996, com passos de 1/256.

### O que acontece em cada camada, em inteiros

1. **Multiplicar e somar:** a camada multiplica cada entrada pelo seu peso e
   soma tudo. Para não haver risco de a soma "estourar", ela é feita numa
   variável enorme, de 128 bits.
2. **Voltar à escala:** como multiplicamos "centavos por centavos", o resultado
   fica 256 vezes grande demais. Dividimos por 256 e **arredondamos** (meio
   para longe do zero: 2,5 vira 3; −2,5 vira −3).
3. **Somar o viés:** um número fixo por neurônio, chamado **viés**.
4. **Saturar:** se o resultado passar do maior valor que cabe em 16 bits, ele é
   "cortado" no limite, como um velocímetro que para no máximo do mostrador.
5. **ReLU:** negativos viram zero.

### Isso estraga a rede?

Não, e medimos isso:

| Versão | Acurácia no conjunto de teste (12.630 fotos) |
|---|---|
| Rede original, com números com vírgula | 91,32% |
| **Programa em C com inteiros, que vai para o carro** | **91,35%** |

A diferença é de 3 fotos a mais acertadas, o que é ruído de arredondamento, e
não uma melhora real. Na prática, a quantização **não custou acurácia**.

---

## 4. O programa que roda no carro

A rede quantizada é transformada automaticamente num arquivo de código em
**linguagem C**, `qnn_conv_native.c`. É esse arquivo, e não o modelo de
treinamento, que é compilado e executado no carro.

Esse detalhe é central. Muitas ferramentas de verificação analisam o modelo
"de laboratório", com números com vírgula, e não o programa que realmente roda.
Trabalhos anteriores mostraram que essa diferença **pode ser explorada**: é
possível construir imagens adversariais dentro de regiões que uma ferramenta
tinha "verificado", aproveitando pequenos erros numéricos que a ferramenta
ignorava.

Por isso, **tudo o que provamos é sobre esse arquivo C exato**. Para garantir
que ninguém troque o arquivo depois:

- calculamos a **impressão digital** dele (um código **SHA-256**, que muda
  completamente se um único caractere do arquivo mudar);
- toda prova registra essa impressão digital;
- a verificação recusa qualquer prova feita para outro arquivo.

---

## 5. O que queremos provar e por que testar não basta

### A propriedade

Escolhemos uma foto de placa que a rede acerta. Imagine alguém (um atacante,
um ruído no sensor, um defeito na transmissão) que pode alterar **cada byte
de cor de cada pixel em até 1 unidade**, para cima ou para baixo. Um vermelho
120 pode virar 119, 120 ou 121, e isso em todos os pixels ao mesmo tempo, em
qualquer combinação.

Queremos provar que, **em todas essas combinações, o programa em C continua
dando a mesma resposta**. Chamamos isso de **robustez local** com raio
**ε = 1** (épsilon igual a 1).

### Por que testar não resolve

Cada um dos 3.072 números da entrada pode assumir até 3 valores. O total de
combinações é de até 3 elevado a 3.072, um número com **cerca de 1.466
dígitos**. Para comparar: o número de átomos no universo observável tem cerca
de 80 dígitos.

Testar mil, um milhão ou um bilhão de combinações não diz nada sobre as
restantes. Um teste só mostra que **não encontramos** problema, nunca que
**não existe** problema. Precisamos de uma **prova**, que cubra todas as
combinações de uma vez.

### A ferramenta de prova: o ESBMC

O **ESBMC** é um **verificador de modelos**: um programa que lê código C e
responde, com rigor matemático, se uma afirmação sobre esse código vale para
**todas** as entradas possíveis. Ele não executa o código caso a caso. Ele
transforma o código e a afirmação num grande problema lógico e usa um
**resolvedor SMT** (um "resolvedor de equações lógicas") para decidir se existe
alguma entrada que viole a afirmação.

As respostas possíveis são:

- **VERIFICADO:** nenhuma entrada viola a afirmação. É uma prova.
- **FALHOU:** existe uma entrada que viola, e o ESBMC mostra qual.
- **TEMPO ESGOTADO / MEMÓRIA ESGOTADA:** não conseguiu decidir. Isso **não** é
  prova de nada, nem para um lado nem para o outro.

O ESBMC entende o C **bit a bit**: o arredondamento, a saturação, os 128 bits
do acumulador, tudo exatamente como o processador executa. Por isso dizemos que
a verificação é **bit-precisa**.

Neste trabalho, **o ESBMC é o único juiz**. Nada é declarado "verificado" sem
que o ESBMC tenha dito VERIFICADO.

---

## 6. O obstáculo: a rede inteira é grande demais para o ESBMC de uma vez

Se pedirmos ao ESBMC "prove que a rede inteira não muda de resposta", o
problema lógico fica tão grande que ele não termina: medimos isso, o tempo e a
memória se esgotam. Somente a primeira parte da conta já tem cerca de 2,7
milhões de multiplicações, e cada ReLU dobra as possibilidades que o
resolvedor precisa considerar.

A saída foi **dividir a prova em milhares de pedaços pequenos**, cada um fácil
para o ESBMC, de modo que os pedaços juntos impliquem a propriedade completa.

---

## 7. A ideia central: quem propõe não precisa ser confiável; quem confere, sim

### Analogia: o Sudoku

Resolver um Sudoku difícil dá trabalho. **Conferir** um Sudoku já resolvido é
fácil: basta olhar cada linha, coluna e quadrado. E, se a conferência passa, a
solução está certa, **não importa quem a fez nem como**.

Usamos a mesma divisão de trabalho:

1. Um **buscador** (um programa em Python, rápido e esperto, mas que **não
   precisa ser confiável**) faz o trabalho difícil. Ele calcula limites para os
   valores dentro da rede e monta um **certificado**: uma lista enorme de
   afirmações pequenas, cada uma com a sua justificativa.
2. O **ESBMC confere cada afirmação** contra o código C exato.

Se o buscador tiver um erro de programação, a consequência é que alguma
conferência **falha** e a imagem fica sem prova. **Um erro no buscador nunca
produz um "VERIFICADO" falso.** Essa ideia é conhecida em segurança de software
como **código portador de prova** (*proof-carrying code*).

### O que o buscador calcula

**Etapa A: caixas (intervalos).** Para cada neurônio de cada camada, o
buscador calcula um intervalo, "o valor deste neurônio fica sempre entre 12 e
340", válido para todas as perturbações permitidas. Ele faz isso camada por
camada, usando apenas o pior caso de cada soma. O ESBMC confere essas caixas em
blocos de 24 neurônios (589 blocos, no caso da imagem de validação usada no
piloto).

O problema das caixas é que elas **incham** a cada camada, porque ignoram que
os valores estão relacionados entre si. Na última camada, ficam largas demais
para decidir a resposta.

**Etapa B: cadeias (relaxações lineares, no estilo CROWN).** Para apertar os
limites, o buscador escreve cada limite importante como uma **fórmula** que
volta camada por camada até a imagem de entrada. É como dizer "a nota da
classe 12 menos a nota da classe 14 é pelo menos (uma soma de termos da
entrada)". A ReLU, que é o trecho difícil, é substituída por retas que a
limitam por cima ou por baixo, e cada escolha de reta só vale sob certas
condições, que precisam ser provadas também.

Cada passo de cada cadeia vira um pequeno programa C que o ESBMC confere.
Alguns exemplos:

- que as contas de um passo foram feitas corretamente;
- que o arredondamento do código real nunca erra mais do que meio passo (1/512),
  provado sobre a mesma função de arredondamento que está no código do carro;
- que a reta usada para substituir a ReLU realmente fica do lado certo, em todo
  o intervalo em que é usada;
- que cada fato citado (uma caixa, ou o resultado de outra cadeia) já foi
  provado antes.

**Etapa C: a conclusão.** No fim, para a classe correta e **cada uma das
outras 42 classes**, o certificado mostra que a nota da classe correta é
**sempre maior**, em todas as combinações de perturbação. Se as 42
comparações fecham e **todos** os pedaços foram VERIFICADOS pelo ESBMC, a
imagem é declarada **VERIFICADA**.

### Quanto trabalho é isso

Na imagem de validação usada no piloto (a de número 3652):

| Item | Quantidade |
|---|---|
| Cadeias no certificado | 4.054 |
| Verificações separadas do ESBMC | 58.695 |
| Resultado | **todas VERIFICADAS**, nenhuma falha |
| Menor vantagem da classe correta | 2.622 unidades inteiras, contra a classe 14 |
| Tempo | cerca de 4 horas numa única máquina, com 4 verificações em paralelo |

### E se o conferente estiver sendo "bonzinho"?

Uma pergunta justa: e se as nossas verificações fossem frouxas demais e
aceitassem qualquer coisa? Para descartar isso, **sabotamos o certificado de
propósito**. Alteramos um número de cada vez, por exemplo uma constante, um
arredondamento ou um limite citado, e rodamos o ESBMC de novo. Em **todos os
14 tipos de sabotagem**, o ESBMC respondeu FALHOU, apontando exatamente a
afirmação adulterada. Um teste anterior, que corrompia as próprias afirmações,
também passou em 15 de 15.

---

## 8. Como os experimentos foram organizados (para não "roubar sem querer")

Em aprendizado de máquina e em verificação, é fácil enganar a si mesmo
escolhendo, sem perceber, os casos que dão certo. Tomamos três cuidados.

**1. Separação dos dados.** As fotos foram divididas em três grupos:

- **treino:** a rede aprende com elas;
- **validação:** usamos para escolher a rede e ajustar o método;
- **teste:** fica guardado e é usado **uma única vez**, no fim, para medir o
  resultado.

A imagem 3652 é de **validação**. Ela serviu para desenvolver o método e
**não conta como resultado do artigo**.

**2. Pré-registro.** Antes de olhar qualquer resultado no conjunto de teste,
escrevemos e registramos no git (um sistema que guarda o histórico de cada
alteração, com data) um **protocolo**, contendo:

- quais 80 fotos de teste serão usadas, sorteadas com uma semente fixa
  (20260925);
- o raio ε = 1;
- a impressão digital de **todo** o código e de todos os dados usados;
- a regra de orçamento: nenhuma prova nova começa depois de segunda-feira,
  28/09, às 04:00;
- o que conta como cada resultado.

Se alguém alterar qualquer arquivo de código depois disso, o programa se
**recusa** a rodar.

**3. Todos os resultados são reportados.** Cada uma das 80 fotos termina em
exatamente uma categoria:

| Resultado | Significado |
|---|---|
| **VERIFICADA** | O ESBMC provou todos os pedaços. É a única categoria que conta como "robusta". |
| CLASSIFICADA ERRADO | A rede já erra a foto original; não há o que proteger. |
| BUSCADOR INCONCLUSIVO | O buscador não conseguiu montar limites apertados o suficiente. **Não** significa que a rede não seja robusta, só que não conseguimos provar. |
| FALHA NA VERIFICAÇÃO | O ESBMC rejeitou algum pedaço do certificado. |
| NÃO EXECUTADA | O tempo acabou antes de chegar a vez da foto. |

A **taxa de certificação** é o número de VERIFICADAS dividido por **todas as
80**. As que ficaram sem prova contam como "não certificadas", e nunca são
descartadas. E **nenhuma** foto é declarada "não robusta" por falta de prova.

---

## 9. O que a prova garante, e o que não garante

### Garante (provado pelo ESBMC)

Para cada foto VERIFICADA: **qualquer** imagem cujos bytes de cor difiram da
original em no máximo 1 unidade recebe do programa C implantado a **mesma
resposta**. Isso inclui a etapa que reduz a foto para 32 × 32, que também faz
parte do código verificado.

### Medimos (evidência forte, mas não é prova)

- O programa C e uma implementação independente em Python dão **exatamente os
  mesmos 43 números** em todas as 12.630 fotos de teste.
- O mesmo arquivo C compilado para a placa embarcada deve dar os mesmos
  números, bit por bit. Esse teste está sendo feito agora.

### Supomos (e declaramos abertamente)

- Que o próprio ESBMC e o resolvedor lógico estão corretos.
- Que o compilador C traduz o código corretamente para o processador. O ESBMC
  confere o código-fonte, não o binário.
- Que a "cola" em Python que junta os milhares de pedaços num único veredito
  está correta. Ela é pequena, foi revisada e é testada, mas não foi provada
  pelo ESBMC.

### Não garante

- **Perturbações físicas:** adesivos na placa, reflexo do sol, chuva, borrão de
  movimento. Essas mudanças são muito maiores do que 1 unidade por byte.
- **Fotos não certificadas:** para elas não há garantia alguma.
- **O resto do sistema:** a câmera, o recorte da placa na imagem, a decisão
  que o carro toma depois.
- **Segurança do veículo como um todo:** normas automotivas (como a ISO 26262)
  exigem muito mais. Uma rede verificada é uma peça de evidência, não o
  argumento completo.

---

## 10. E o DeepPoly, a pré-imagem e o MILP?

Nas versões anteriores deste trabalho, com redes pequenas (Iris, Seeds, MNIST e
uma primeira rede GTSRB com 41,75% de acurácia), os limites vinham de duas
técnicas:

- **DeepPoly:** outro método de relaxação linear, da mesma família do CROWN;
- **síntese de pré-imagem por MILP:** um otimizador matemático que calcula
  exatamente quais entradas levam a cada região.

Os limites que elas produziam também eram conferidos pelo ESBMC. Com a rede
atual, de 43 classes e 91% de acurácia, essas abordagens não escalaram:

- a rede foi convertida em matrizes densas enormes;
- as caixas perderam informação por volta da quarta camada;
- a última camada esgotou a memória.

Isso motivou o formato de certificado descrito na seção 7, em que o CROWN
ocupa o lugar do DeepPoly e as cadeias ocupam o lugar da pré-imagem. O
princípio não mudou: **buscadores não confiáveis propõem, o ESBMC decide**.

O MILP continua útil como ferramenta de diagnóstico. Nas fotos em que o
buscador fica inconclusivo, ele pode procurar um contraexemplo real: uma
imagem perturbada que de fato muda a resposta. Se encontrar, a imagem é
executada no C para confirmar.

---

## 11. Limitações, ditas com franqueza

- **Custo:** cerca de 4 horas por foto, numa única máquina. Isso serve para
  uma amostra, não para milhares de fotos. A maior parte do tempo é gasta
  iniciando o ESBMC dezenas de milhares de vezes; agrupar verificações é o
  próximo passo natural.
- **Cobertura:** o buscador só consegue montar certificados para parte das
  fotos (nos testes de validação, cerca de 1 em cada 7). Isso é limitação do
  buscador, não necessariamente da rede.
- **Raio pequeno:** ε = 1 é a perturbação digital mínima possível. Raios
  maiores são trabalho futuro.
- **Uma rede, um conjunto de dados:** os resultados valem para esta rede e para
  a GTSRB.

---

## 12. Resumo em cinco passos

1. **Treinamos** uma rede neural para reconhecer 43 tipos de placa de trânsito
   (91,32% de acurácia).
2. **Quantizamos** a rede para inteiros de 16 bits e **geramos o código C** que
   roda no carro (91,35% de acurácia, sem perda).
3. Para cada foto, um **buscador não confiável** monta um certificado com
   dezenas de milhares de afirmações pequenas.
4. O **ESBMC confere cada afirmação** sobre o código C exato, bit por bit. Só se
   todas passarem a foto é declarada robusta.
5. Os resultados vêm de uma **amostra sorteada e pré-registrada** do conjunto
   de teste, com **todos** os desfechos reportados, e o mesmo código roda na
   placa embarcada com saídas idênticas.

---

## Glossário

| Termo | Explicação curta |
|---|---|
| Rede neural | Programa que aprende, a partir de exemplos, a transformar uma entrada (imagem) numa resposta (classe). |
| Peso, viés | Números fixos da rede, aprendidos no treinamento. |
| Camada | Uma etapa da receita de contas: multiplica, soma, ajusta. |
| Convolucional | Rede que analisa a imagem por pequenas janelas deslizantes. |
| ReLU | Ajuste "negativo vira zero, positivo fica". |
| Classe | Cada tipo de placa (43 no total). |
| Acurácia | Porcentagem de fotos classificadas corretamente. |
| Ponto flutuante | Número com vírgula, como o computador costuma guardar. |
| Quantização | Conversão para números inteiros com uma escala fixa. |
| Q7.8 | Formato de 16 bits: 7 bits inteiros, 8 de fração, e sinal; passo de 1/256. |
| Saturação | Corte do valor no limite máximo ou mínimo que cabe no formato. |
| Robustez local | A resposta não muda para nenhuma perturbação pequena **ao redor de uma foto específica**. |
| ε (épsilon) | Tamanho máximo da perturbação por byte; aqui, 1. |
| Verificação formal | Prova matemática de que uma propriedade vale para **todas** as entradas. |
| ESBMC | Verificador de modelos para programas C; o juiz final deste trabalho. |
| Resolvedor SMT | Motor lógico que o ESBMC usa para decidir se uma afirmação pode ser violada. |
| Bit-preciso | Que considera o comportamento exato do código, bit por bit, incluindo arredondamentos. |
| Buscador | Programa não confiável que propõe limites e certificados. |
| Certificado | Lista de afirmações pequenas que, juntas, implicam a robustez. |
| Intervalo / caixa | Limite inferior e superior para o valor de um neurônio. |
| CROWN, DeepPoly | Métodos que limitam a rede com retas (relaxações lineares). |
| MILP | Otimizador matemático exato; útil para buscar contraexemplos. |
| Contraexemplo | Uma entrada concreta que viola a propriedade. |
| SHA-256 | "Impressão digital" de um arquivo; muda se qualquer caractere mudar. |
| Pré-registro | Fixar e registrar o plano do experimento antes de ver os resultados. |
| Validação / teste | Dados usados para desenvolver o método / dados usados uma única vez para medir o resultado. |
