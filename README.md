### Descrição do problema

O nosso problema consiste em  **Minimizar o Tempo Total de Viagem** dado uma rota feita por um transporte público. É um desafio de otimização de roteamento, tipicamente modelado como um Vehicle Routing Problem (VRP).

Para isso, são utilizadas instâncias construídas a partir do *dataset* SUNT (Salvador Urban Network Transportation), que representa a rede de transporte urbano como um grafo de vértices (paradas/estações) e arestas (conexões). A função objetivo central é a **minimização do tempo total da viagem**. 

Uma solução para este problema é definida como uma **rota**, que é uma **permutação** dos nós ativos que o agente deve percorrer, onde cada permutação é uma posição no espaço de soluções.

- **O que é uma rota?**
    
    > A rota é formalmente representada como uma **permutação dos** n **vértices** (paradas/estações). Isto é, uma **sequência ordenada de paradas/estações** que um veículo deve percorrer para satisfazer as demandas do problema, minimizando o custo total associado à `matriz_tempo_viagem`.
    
    A rota de um veículo deve ser um ciclo que começa e termina no depósito
    > 

### Base de Dados: SUNT

O **Salvador Urban Network Transportation (SUNT)** é um conjunto de dados espaciotemporal coletado em Salvador entre março de 2024 e março de 2025, com o objetivo de apoiar a otimização da mobilidade urbana. O *dataset* abrange uma área de 694 km² e inclui informações detalhadas de três sistemas de transporte público (ônibus regulares, metrô e BRT), integrando dados de cerca de 700.000 passageiros e aproximadamente 2.000 veículos distribuídos em 400 linhas, conectando cerca de 3.000 paradas e estações. O SUNT é notável pela inclusão inovadora de dados de passageiros ****(embarque e desembarque) e por sua granularidade temporal inferior a um minuto, o que o torna um recurso robusto para o desenvolvimento e avaliação de métodos orientados a dados em Sistemas de Transporte Inteligentes (ITS).

### Função objetivo

<aside>
🎯

Minimizar Tempo total de viagem.

A Função Objetivo é calculada somando os custos temporais (`trip_time`) de todos os trechos percorridos na rota (permutação).

</aside>

$$
\min f(\pi) = \sum_{i=1}^{n-1} \text{tempo}(\pi_i, \pi_{i+1}) + \text{tempo}(\pi_n, depósito)
$$

- **Exemplo de cálculo do custo de uma rota**
    - **Matriz de tempo de viagem**
        
        A **matriz_tempo_viagem** é lida como uma matriz de custos, onde a **linha** representa o **Nó de Origem** e a **coluna** representa o **Nó de Destino**. O valor na intersecção é o tempo necessário para percorrer aquele trecho, simulando o atributo `trip_time` do SUNT.
        
        ![Captura de tela de 2025-12-15 22-38-37.png](attachment:e60d3483-722c-4c95-8766-ecf9eae57c45:Captura_de_tela_de_2025-12-15_22-38-37.png)
        
    
    A arquitetura multiagente proposta busca a **permutação ótima** (a rota) utilizando uma matriz que contém o tempo necessário para o veículo sair de uma origem i e ir para um destino j. 
    
    A função objetivo é **avaliar cada rota (**π**)** que o agente encontra ou gera.
    
    Se o agente gerar uma rota π=(0→1→3→4→2→0), o cálculo do **Tempo Total de Viagem** é feito somando os tempos de cada trecho na matriz:
    
    Tempo Total=T0,1+T1,3+T3,4+T4,2+T2,0 Tempo Total=5+9+5+7+8=34 minutos
    
    Esta avaliação numérica é o que define o **valor da função objetivo** que o agente busca minimizar.
    
    > O algoritmo MAHM/BDI tentará encontrar uma permutação diferente que resulte em um custo total menor. Se essa solução fosse a melhor encontrada pelo agente, ela se tornaria o `pbest`
    > 

### Ambiente = **Espaço de soluções**

Ambiente composto pelo conjunto das soluções viáveis*.

$$
\mathcal{S} = \{ \pi \mid \pi \text{ é uma permutação dos nós ativos} \}
$$

<aside>
🎯

Solução:

A solução é uma rota. Isto é, uma permutação* (ou sequência ordenada) das paradas que o veículo deve visitar.

A permutação ótima é a sequência específica de visita aos nós ativos (paradas) que resulta no menor Tempo Total de Viagem quando somados os custos (tempos) correspondentes na `matriz_tempo_viagem`. É essa permutação/sequência que o algoritmo, através do movimento cooperativo dos agentes (intensificação e diversificação), está constantemente buscando.

</aside>

- **Restrições para a viabilidade de uma solução**
    
    Formalmente, uma solução $π$ (gerada por permutação) é viável se, e somente se, todos os arcos selecionados na rota obedecerem às restrições:
    
    $$
    π=(0,v_1,v_2,…,v_n,0)
    $$
    
    onde 0 é o depósito e:
    
    - **Restrição de Capacidade:** Ao percorrer o arco (i,j) (a aresta), a carga de passageiros (`loading`) no veículo naquele momento deve ser igual ou inferior à capacidade máxima do veículo (`max_capacity`). Se a permutação levar a uma sobrecarga em qualquer arco, a solução π é considerada **inviável**, mesmo que a troca dos nós seja estruturalmente simples
    - Cada nó ativo aparece exatamente uma vez (isto é, são visitados e atendidos exatamente uma vez)
    - A rota começa e termina no depósito
    - Todos os arcos usados existem na matriz_tempo_viagem 💡
    
    <aside>
    💡
    
    O problema é modelado como um VRP em um **grafo completo dirigido***, onde todas as paradas são mutuamente alcançáveis. As arestas representam apenas custos temporais, não restrições de conectividade. Dessa forma, a viabilidade de uma solução depende exclusivamente das restrições de capacidade, e não da existência de caminhos.
    
    $G=(V,E),E=V×V$
    
    </aside>
    

### Aquitetura BDI

Essa arquitetura é baseada na arquitetura descrita no artigo 4, o **MAHM (Multiagent Architecture for Hybridization of Metaheuristics).** A descrição, porém, segue a arquitetura BDI (Beliefs, Desire and Intentions). ****

Arquitetura baseada na Otimização por Enxame de Partículas (PSO), como agentes paralelos e colaborativos.

**Crenças (Beliefs)**

> O conhecimento ou percepção de um agente sobre o ambinete.
> 

No nosso caso, cada agente teria em sua base de conhecimento:

1 - Uma solução possível dentro do espaço de soluções viáveis, que é a melhor encontrada pelo agente até então (`pbest`)

2 - O valor da função objetivo associado a essa solução. 

3 - A melhor solução encontrada pelo enxame, o `gbest`.  

**Desejo (Desire)**

> Os objetivos ou aspirações do agente que servem como a principal motivação para a ação. São os objetivos a serem alcançados.
> 

No nosso caso, cada a gente tem como desejo minimizar a função objetivo. 

**Intenção (Intention)**

> São os desejos específicos que o agente se compromete a alcançar, impulsionando suas ações e planos. Basicamente, é quando o agente tranforma um objetivo em um plano de ação. É uma lista de ações para atingir objetivos.
> 

A intenção (o plano) do agente para fazer com que ele “alcance” ou se aproxime do seu desejo (minimizar a função objetivo) é dividida em duas etapas: uma de diversificação e outra de intensificação. 

<aside>
🎯

As intenções do agente são executadas com o propósito de encontrar uma nova posição cujo custo, determinado pela soma dos elementos da matriz, seja menor que o custo das posições atuais (`pbest` e `gbest`)

</aside>

Eis como o agente se movimenta:

1. **Primeira Fase: A Metaheurística inicial (O Método de Decisão)**

Antes de o agente tentar se aproximar do grupo (pbest ou gbest), ele tenta melhorar a sua solução atual por conta própria.

- O **Método de Decisão** escolhe uma metaheurística baseada em solução única . Inicialmente, essa escolha pode ser aleatória.
- Essa metaheurística escolhida é executada na posição atual do agente
- O resultado dessa execução gera uma nova solução. Esta nova solução torna-se a "Posição de Origem" para o operador de velocidade.
1. **Segunda Fase: O Operador de Velocidade (Path-Relinking) → DIVERSIFICAÇÃO**

Agora que o agente já aplicou a metaheurística e tem uma "Posição de Origem" (esperançosamente melhorada), ele precisa se mover em direção à memória/crença do enxame.

- O Path-Relinking atua como o operador de velocidade. Ele constrói uma trajetória (uma sequência de soluções intermediárias) conectando a **Posição de Origem** (onde o agente está após a metaheurística) à **Posição de Destino** (que pode ser o *pbest* ou *gbest*)
- O objetivo aqui é a "**diversificação**": ao transformar a solução de origem na solução de destino, o agente explora novas áreas do espaço de busca que estão entre essas duas soluções.
1. **Terceira Fase: Intensificação (Metaheurística Aninhada) → INTENSIFICAÇÃO**
- Durante o trajeto do *path-relinking*, o agente passa por várias soluções intermediárias.
- Se, durante essa travessia, for encontrada uma posição intermediária que seja melhor que a atual, **a trajetória é interrompida.**.
- Nesse momento, a metaheurística mais eficiente (conforme o método de aprendizado) é **executada novamente** a partir dessa posição intermediária para tentar explorar (intensificar) aquela região promissora.

> Resumo do Fluxo:

Para visualizar, o ciclo de uma iteração no agente ocorre assim:

1. **Escolha e Execução inicial:** O agente escolhe uma estratégia e a executa na sua solução atual.

2. **Definição do Movimento:** O resultado vira a *Origem*. O alvo (*pbest*/*gbest*) vira o *Destino*.

3. **Movimentação (Velocity Operator):** Inicia-se o *path-relinking* da Origem para o Destino.

4. **Interrupção Oportunista:** Se no meio do caminho surgir algo bom, para-se o movimento e **roda-se a metaheurística**. 🧠
> 

🧠 Aqui que entra a “inteligência” do agente. Precisamos definir uma **Metodologia de Seleção de Heurísticas,** isto é, dado as percepções do agente do ambiente e de sua própria base de crença, qual seria a melhor heurística para ser aplicada neste momento?

<aside>
💡

Eis o que precisamos:

1 - Definir uma forma inteligente de escolher essas intenções/metaheurísticas. O agente olhará para um pool de metaheurísticas e vai escolher a que probalisticamente pode retornar o melhor resultado. Isso será feito pelo componente chamado DM →MÉTODO DE DECISÃO
2 - Uma vez escolhida a metaheurística, precisamos avaliar essa escolha para decisões futuras. Isso será feito pelo componente chamado LM →MÉTODO DE APRENDIZAGEM

</aside>

**Método de Decisão e Método de Aprendizagem**

**1. Método de Decisão** 

O método de decisão é o responsável por determinar **qual estratégia** o agente utilizará para tentar melhorar sua posição atual no espaço de busca.

- **Seleção de Estratégia:** Em vez de usar um único algoritmo fixo, o agente possui acesso a um repositório de estratégias (pool de algoritmos), que são metaheurísticas baseadas em solução única.
- **Critério de Escolha:** A escolha não é necessariamente aleatória. O método decide qual estratégia aplicar com base em critérios predefinidos ou estatísticos fornecidos pelo Método de Aprendizado.
- **Adaptação:** Se a estratégia atual começar a falhar em encontrar boas soluções, o método de decisão pode trocá-la por outra, baseando-se nas informações fornecidas pelo método de aprendizado.

**2. Método de Aprendizado** 

O método de aprendizado é responsável por **atualizar a memória e o histórico** (suas crenças) do agente após a execução das ações, permitindo que experiências passadas influenciem ações futuras.

- **Registro de Histórico:** Ele armazena informações cruciais sobre a trajetória de busca, como as soluções visitadas, os parâmetros utilizados e, vitalmente, as estatísticas de **sucessos e falhas** das estratégias aplicadas.
- **Feedback para o Sistema:** O objetivo principal deste método é processar os resultados obtidos para "ensinar" ao agente o que funciona e o que não funciona. Nos experimentos relatados no artigo 4, o método de aprendizado usou as estatísticas de sucesso para aumentar a probabilidade de que estratégias eficientes fossem escolhidas novamente no futuro.
- **Repositório Centralizado:** Assim como os métodos de decisão, os métodos de aprendizado podem ser armazenados em um repositório central, permitindo que diferentes agentes usem diferentes lógicas de aprendizado ou compartilhem o mesmo método.

**A Interação entre os Dois Métodos:**

A relação entre esses dois métodos cria um ciclo de melhoria contínua, descrito no artigo 4

1. O **Método de Decisão** escolhe uma metaheurística e a executa.

2. O agente se move (via *path-relinking*).

3. O **Método de Aprendizado** avalia o resultado: "Essa estratégia melhorou a solução?" e atualiza a memória do agente.

4. Na próxima iteração, o Método de Decisão consulta essa memória atualizada para fazer uma escolha mais informada.