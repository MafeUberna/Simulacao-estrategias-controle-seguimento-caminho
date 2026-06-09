import random
import numpy as np
import pandas as pd
from ackermann_model import AckermannSlipModel
from mpc_controller import MPCController
from simulator import Simulator

class GeneticAlgorithmNSGA2:
    def __init__(self, path_x, path_y, path_theta, pop_size=30, generations=20):
        self.path_x = path_x
        self.path_y = path_y
        self.path_theta = path_theta
        self.pop_size = pop_size
        self.generations = generations
        
        # Limites dos genes: [q_pos(exp), q_ori(exp), r_motor(exp), r_esterco(exp), v_ref]
        # Usamos expoentes (10^x) para as matrizes Q e R varrerem ordens de grandeza
        self.bounds = [
            (-2, 2.5), # [0] q_pos
            (-2, 2.5), # [1] q_theta 
            (-2, 2.5), # [2] q_delta
            (-2, 2.5), # [3] q_v
            (-2, 3.5), # [4] r_motores (R geralmente pode ser maior, deixamos até ~3000)
            (-2, 3.5), # [5] r_esterco
            (0.5, 3.5) # [6] v_ref (velocidade máxima)
        ]
        self.taxa_cruzamento = 0.8
        self.taxa_mutacao = 0.2

    # =========================================================================
    # 1. FUNÇÃO DE SIMULAÇÃO E AVALIAÇÃO (7 PARÂMETROS - NORMALIZADA)
    # =========================================================================
    def _simular_e_avaliar(self, individuo):
        """Roda a simulação curta e retorna os 3 objetivos normalizados para o NSGA-II."""
        # Decodifica os 7 parâmetros contínuos (usando base 10 para escalas logarítmicas)
        q_pos = 10 ** individuo[0]
        q_theta = 10 ** individuo[1]
        q_delta = 10 ** individuo[2]
        q_v = 10 ** individuo[3]
        
        # Aplica o MESMO peso q_pos para x e y (Garante simetria nos eixos cartesianos)
        q_diag = [q_pos, q_pos, q_theta, q_delta, q_v]
        
        r_motores = 10 ** individuo[4]
        r_esterco = 10 ** individuo[5]
        
        # Aplica o MESMO peso r_motores para a roda esquerda e direita (Controle Simétrico)
        r_diag = [r_motores, r_motores, r_esterco]
        
        v_ref = individuo[6]
        
        # Instancia o modelo cinemático e o controlador preditivo (MPC)
        model = AckermannSlipModel(use_mechanical_differential=False, slip_gain=1.0)
        
        controller = MPCController(
            model=model, path_x=self.path_x, path_y=self.path_y, path_theta=self.path_theta,
            ref_v=v_ref, dt=0.1, horizon=10, control_horizon_m=5,
            use_differential=True, q_diag=q_diag, r_diag=r_diag,
            v_max=3.5, delta_max_deg=30
        )
        
        # Configuração do simulador com Passo e Tempo real do projeto
        sim = Simulator(
            model=model, controller=controller,
            path_x=self.path_x, path_y=self.path_y,
            end_of_path_threshold=0.3, use_velocity_controller=False,
            T=40.0, dt=0.001
        )
        sim.run()
        
        # --- Extração das Métricas Logadas ---
        log = sim.system_log
        error_data = Simulator.calculate_tracking_errors(log, self.path_x, self.path_y, self.path_theta)
        rms_xy, rms_theta = Simulator.calculate_rms_error(error_data)
        
        # Cálculo do Índice de Esforço de Controle (Variação dos Atuadores)
        df_motor = pd.DataFrame(log.get("motor_cmd", []), columns=['time', 'left', 'right'])
        if not df_motor.empty:
            esforco = np.sum(np.abs(np.diff(df_motor['left']))) + np.sum(np.abs(np.diff(df_motor['right'])))
        else:
            esforco = 999.0
            
        # --- Critério de Falha Dinâmica (Death Penalty) ---
        # 1. Se o desvio lateral for intolerável (> 1.2 metros)
        # 2. Se a simulação esgotar o tempo total (40s) e o carro ainda estiver longe da meta
        dist_final = np.hypot(sim.x - self.path_x[-1], sim.y - self.path_y[-1])
        falhou = (rms_xy > 1.2) or (dist_final > 1.0 and sim.time >= 39.9)
        
        if falhou:
            return [999.0, 999.0, 999.0] # Punição severa para eliminar o indivíduo do ranking
            
        # =====================================================================
        # AJUSTE MATEMÁTICO: NORMALIZAÇÃO COERENTE PARA O NSGA-II
        # =====================================================================
        # O NSGA-II minimiza estritamente todas as posições do vetor retornado.
        
        # Objetivo 1 (Precisão): Minimiza o erro geométrico e angular combinado
        obj1_erro = rms_xy + np.deg2rad(rms_theta)
        
        # Objetivo 2 (Suavidade): Esforço escalonado (fator /100) para equilibrar ordens de grandeza
        obj2_esforco = esforco / 100.0
        
        # Objetivo 3 (Performance/Rapidez): Transforma a MAXIMIZAÇÃO da velocidade em MINIMIZAÇÃO.
        # Quanto maior for a v_ref (perto do limite de 3.5), menor e melhor será o custo enviado à IA.
        obj3_velocidade = 3.5 - v_ref
        
        return [obj1_erro, obj2_esforco, obj3_velocidade]
    # =========================================================================
    # 2. CORE DO NSGA-II: ORDENAÇÃO E DISTÂNCIA
    # =========================================================================
    def _fast_non_dominated_sort(self, objetivos):
        pop_size = len(objetivos)
        S = [[] for _ in range(pop_size)]
        n = [0] * pop_size
        ranks = [[] for _ in range(pop_size + 1)]
        rank_do_ind = [0] * pop_size

        for p in range(pop_size):
            for q in range(pop_size):
                # p domina q se for menor/igual em todos e estritamente menor em pelo menos um
                domina = all(objetivos[p][i] <= objetivos[q][i] for i in range(3)) and \
                         any(objetivos[p][i] < objetivos[q][i] for i in range(3))
                dominado = all(objetivos[q][i] <= objetivos[p][i] for i in range(3)) and \
                           any(objetivos[q][i] < objetivos[p][i] for i in range(3))
                if domina:
                    S[p].append(q)
                elif dominado:
                    n[p] += 1
                    
            if n[p] == 0:
                rank_do_ind[p] = 1
                ranks[1].append(p)

        i = 1
        while len(ranks[i]) > 0:
            proximo_front = []
            for p in ranks[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank_do_ind[q] = i + 1
                        proximo_front.append(q)
            i += 1
            ranks[i] = proximo_front

        return [f for f in ranks if len(f) > 0], rank_do_ind

    def _calcular_crowding_distance(self, objetivos, front):
        tamanho = len(front)
        if tamanho == 0: return {}
        distancias = {ind: 0.0 for ind in front}
        
        for m in range(3): # 3 objetivos
            front_ordenado = sorted(front, key=lambda x: objetivos[x][m])
            distancias[front_ordenado[0]] = float('inf')
            distancias[front_ordenado[-1]] = float('inf')
            
            min_obj = objetivos[front_ordenado[0]][m]
            max_obj = objetivos[front_ordenado[-1]][m]
            extensoes = max_obj - min_obj if (max_obj - min_obj) > 0 else 1e-6
                
            for i in range(1, tamanho - 1):
                distancias[front_ordenado[i]] += (objetivos[front_ordenado[i+1]][m] - objetivos[front_ordenado[i-1]][m]) / extensoes
        return distancias

    # =========================================================================
    # 3. OPERADORES GENÉTICOS
    # =========================================================================
    def _selecao_torneio_nsga2(self, populacao, ranks, distancias, k=2):
        candidatos = random.sample(range(len(populacao)), k)
        melhor = candidatos[0]
        for c in candidatos[1:]:
            if ranks[c] < ranks[melhor]: # Prefere Rank menor (melhor)
                melhor = c
            elif ranks[c] == ranks[melhor]:
                if distancias[c] > distancias[melhor]: # Desempata pelo mais isolado
                    melhor = c
        return populacao[melhor]

    def _cruzamento_blx(self, p1, p2, alpha=0.5):
        c1, c2 = [], []
        for g1, g2, b in zip(p1, p2, self.bounds):
            d = abs(g1 - g2)
            min_v, max_v = max(min(g1, g2) - alpha*d, b[0]), min(max(g1, g2) + alpha*d, b[1])
            c1.append(random.uniform(min_v, max_v))
            c2.append(random.uniform(min_v, max_v))
        return c1, c2

    def _mutacao(self, ind):
        for i in range(len(ind)):
            if random.random() < self.taxa_mutacao:
                ind[i] = np.clip(ind[i] + random.gauss(0, 0.5), self.bounds[i][0], self.bounds[i][1])
        return ind

    # =========================================================================
    # 4. LOOP PRINCIPAL DO ALGORITMO
    # =========================================================================
    def solve(self):
        print("-> Inicializando NSGA-II: Gerando população aleatória...")
        pop_pai = [[random.uniform(b[0], b[1]) for b in self.bounds] for _ in range(self.pop_size)]
        
        for geracao in range(self.generations):
            # 1. Avalia Pais
            objs_pai = [self._simular_e_avaliar(ind) for ind in pop_pai]
            ranks_pai, rank_vetor_pai = self._fast_non_dominated_sort(objs_pai)
            
            dist_pai = {}
            for front in ranks_pai:
                dist_pai.update(self._calcular_crowding_distance(objs_pai, front))
                
            # 2. Gera Filhos
            pop_filho = []
            while len(pop_filho) < self.pop_size:
                p1 = self._selecao_torneio_nsga2(pop_pai, rank_vetor_pai, dist_pai)
                p2 = self._selecao_torneio_nsga2(pop_pai, rank_vetor_pai, dist_pai)
                c1, c2 = self._cruzamento_blx(p1, p2) if random.random() < self.taxa_cruzamento else (p1.copy(), p2.copy())
                pop_filho.extend([self._mutacao(c1), self._mutacao(c2)])
                
            # 3. Une Pais e Filhos (2N)
            pop_mista = pop_pai + pop_filho[:self.pop_size]
            objs_mista = [self._simular_e_avaliar(ind) for ind in pop_mista]
            
            # 4. Re-ordena e Seleciona os melhores para a próxima geração
            fronts_mista, rank_vetor_mista = self._fast_non_dominated_sort(objs_mista)
            nova_pop = []
            i = 0
            while len(nova_pop) + len(fronts_mista[i]) <= self.pop_size:
                nova_pop.extend([pop_mista[idx] for idx in fronts_mista[i]])
                i += 1
                if i >= len(fronts_mista): break
                
            if len(nova_pop) < self.pop_size and i < len(fronts_mista):
                dist_ultimo = self._calcular_crowding_distance(objs_mista, fronts_mista[i])
                ind_ordenados = sorted(fronts_mista[i], key=lambda x: dist_ultimo[x], reverse=True)
                nova_pop.extend([pop_mista[idx] for idx in ind_ordenados[:self.pop_size - len(nova_pop)]])
                
            pop_pai = nova_pop
            print(f"Geração {geracao+1:02d} concluída | Soluções de Rank 1 (Pareto): {len(ranks_pai[0])}")
            
        # Retorna os indivíduos que compõem a melhor Fronteira de Pareto
        objs_finais = [self._simular_e_avaliar(ind) for ind in pop_pai]
        ranks_finais, _ = self._fast_non_dominated_sort(objs_finais)
        fronteira_pareto_indices = ranks_finais[0]
        
        melhores_solucoes = [pop_pai[idx] for idx in fronteira_pareto_indices]
        melhores_objetivos = [objs_finais[idx] for idx in fronteira_pareto_indices]
        
        return melhores_solucoes, melhores_objetivos