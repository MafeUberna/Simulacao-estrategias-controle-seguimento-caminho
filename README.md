# Simulacao-estrategias-controle-seguimento-caminho
Repositório contendo os arquivos de simulação do documento Comparação de Estratégias de Seguimento de Caminho Para Veículos com Direção Ackermann  e Tração Traseira Independente.
A simulação foi desenvolvida na linguagem de programação Python, com o suporte de
bibliotecas como Matplotlib, Pandas e NumPy. O projeto possui uma estrutura modular para
simular e analisar o comportamento dos controladores do veículo. Essa estrutura é organizada
em classes: uma classe dedicada para cada um dos controladores (Pure Pursuit, Stanley e MPC), 
além do controlador de velocidade PID. Outra classe é o simulador principal, que recebe um controlador,
o modelo do veículo e atuadores,e um caminho pré-definido, para então executar a simulação e registrar todos os dados. 
Uma última classe de visualização é responsável por processar os dados registrados e gerar os gráficos
e tabelas.

A lógica de execução inicia-se pela definição de uma variável principal que determina
qual tipo de análise será feita. As análises possíveis são: análise dos erros pela velocidade, análise
comparativa dos modelos do MPC e análise comparativa do desempenho entre os controladores
laterais. Com base nessa escolha, o script executa as simulações necessárias por meio da classe
do simulador. Quando o veículo chega ao final do caminho, a classe de visualização é acionada
para criar os resultados gráficos solicitados.
No cenário em que se simula o veículo sem diferencial mecânico, aborda-se o comporta-
mento de deslizamento pela análise da diferença entre as velocidades reais das rodas traseiras e
as velocidades ideais determinadas pelo modelo de direção Ackermann. Esta divergência gera o
deslizamento individual de cada roda. A diferença resultante
entre os deslizamentos da roda direita e esquerda é, então, empregada para ajustar o ângulo de
esterço efetivo do veículo.

Além disso, a implementação opera como um sistema de controle multi-taxa (multi-rate),
no qual cada malha de controle e atualização do modelo possui uma frequência de execução
específica, refletindo as diferentes escalas de tempo da dinâmica do sistema.
