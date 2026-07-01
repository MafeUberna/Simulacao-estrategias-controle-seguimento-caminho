import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from ackermann_model import AckermannSlipModel
from mpc_controller import MPCController
from simulator import Simulator

class GeneticAlgorithmNSGA2:
    def __init__(self, path_x, path_y, path_theta, pop_size=40, generations=24):
        self.path_x = path_x
        self.path_y = path_y
        self.path_theta = path_theta
        self.pop_size = pop_size
        self.generations = generations
        
        # Limites dos genes: [q_pos(exp), q_ori(exp), r_motor(exp), r_esterco(exp), v_ref]
        self.bounds = [
            (-2, 2.5),  # [0] q_pos
            (-2, 2.5),  # [1] q_theta 
            (-2, 2.5),  # [2] q_delta
            (-2, 2.5),  # [3] q_v
            (-1, 1),    # [4] r_motores
            (-1, 1),    # [5] r_esterco
            (0.5, 3.5)  # [6] v_ref (linear)
        ]
        self.taxa_cruzamento = 0.8
        self.taxa_mutacao = 0.2

    def _inicializar_populacao(self):
        pop = []
        for _ in range(self.pop_size):
            ind = [random.uniform(b[0], b[1]) for b in self.bounds]
            pop.append(ind)
        return pop

    @staticmethod
    def decodificar_genes(gene):
        """Converte os expoentes logarítmicos e lineares em ganhos físicos do MPC"""
        q_pos = 10**gene[0]
        q_theta = 10**gene[1]
        q_delta = 10**gene[2]
        q_v = 10**gene[3]
        r_motor = 10**gene[4]
        r_est = 10**gene[5]
        v_ref = gene[6]
        return q_pos, q_theta, q_delta, q_v, r_motor, r_est, v_ref

    def _simular_e_avaliar(self, ind):
        """Executa a simulação física transiente em malha fechada para o indivíduo"""
        q_pos, q_theta, q_delta, q_v, r_motor, r_est, v_ref = self.decodificar_genes(ind)
        
        model = AckermannSlipModel(use_mechanical_differential=False, slip_gain=1.0)
        controller = MPCController(
            model=model, path_x=self.path_x, path_y=self.path_y, path_theta=self.path_theta,
            ref_v=v_ref, dt=0.1, horizon=10, control_horizon_m=5, use_differential=True,
            q_diag=[q_pos, q_pos, q_theta, q_delta, q_v],
            r_diag=[r_motor, r_motor, r_est],
            v_max=3.5, delta_max_deg=30
        )
        sim = Simulator(
            model=model, controller=controller, path_x=self.path_x, path_y=self.path_y,
            end_of_path_threshold=0.3, use_velocity_controller=False, T=40, dt=0.001
        )
        sim.run()
        log = sim.system_log
        
        error_data = Simulator.calculate_tracking_errors(log, self.path_x, self.path_y, self.path_theta)
        rms_xy, rms_theta = self.simulator.calculate_rms_error(error_data) if hasattr(self, 'simulator') else Simulator.calculate_rms_error(error_data)
        
        f1_erro = rms_xy + np.deg2rad(rms_theta)
        
        df_motor = pd.DataFrame(log.get("motor_cmd", []), columns=['time', 'left', 'right'])
        f2_esforco = np.sum(np.abs(np.diff(df_motor['left']))) + np.sum(np.abs(np.diff(df_motor['right']))) if not df_motor.empty else 999.0
        
        # NSGA-II minimiza estritamente: invertemos a velocidade máxima linearmente
        f3_agilidade = 3.5 - v_ref
        
        # Aplicação de Death Penalty caso o carro saia da pista ou capote por instabilidade
        poses = log.get('vehicle_pose', [])
        
        # Tratamento robusto para extrair o erro lateral independente do tipo de objeto retornado
        if isinstance(error_data, dict):
            tracking_ref = error_data.get('cross_track_error', [])
            max_lateral_error = np.max(np.abs(tracking_ref)) if len(tracking_ref) > 0 else 0.0
        elif isinstance(error_data, pd.DataFrame):
            max_lateral_error = error_data['cross_track_error'].abs().max() if 'cross_track_error' in error_data.columns else 0.0
        else:
            # Se for uma lista pura ou array de erros, assume o cálculo direto sobre ela
            try:
                max_lateral_error = np.max(np.abs(error_data)) if len(error_data) > 0 else 0.0
            except:
                max_lateral_error = 0.0

        # Aplicação da penalidade absoluta se o veículo estourar o limite de 1.2 metros da pista
        if len(poses) < 5 or max_lateral_error > 1.2:
            return [999.0, 999.0, 999.0]

    def _fast_non_dominated_sort(self, objs):
        num_ind = len(objs)
        S = [[] for _ in range(num_ind)]
        n = [0] * num_ind
        rank = [0] * num_ind
        fronts = [[]]

        for p in range(num_ind):
            for q in range(num_ind):
                p_dominates = False
                q_dominates = False
                
                # Checa dominância multi-objetivo estrita
                if (objs[p][0] <= objs[q][0] and objs[p][1] <= objs[q][1] and objs[p][2] <= objs[q][2]) and \
                   (objs[p][0] < objs[q][0] or objs[p][1] < objs[q][1] or objs[p][2] < objs[q][2]):
                    p_dominates = True
                elif (objs[q][0] <= objs[p][0] and objs[q][1] <= objs[p][1] and objs[q][2] <= objs[p][2]) and \
                     (objs[q][0] < objs[p][0] or objs[q][1] < objs[p][1] or objs[q][2] < objs[p][2]):
                    q_dominates = True

                if p_dominates:
                    S[p].append(q)
                elif q_dominates:
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1], rank

    def _calcular_crowding_distance(self, objs, front):
        dist = {idx: 0.0 for idx in front}
        for m in range(3):  # 3 Objetivos de aptidão
            front_ordenado = sorted(front, key=lambda x: objs[x][m])
            min_obj = objs[front_ordenado[0]][m]
            max_obj = objs[front_ordenado[-1]][m]
            if max_obj == min_obj: continue
            dist[front_ordenado[0]] = float('inf')
            dist[front_ordenado[-1]] = float('inf')
            for i in range(1, len(front_ordenado) - 1):
                dist[front_ordenado[i]] += (objs[front_ordenado[i+1]][m] - objs[front_ordenado[i-1]][m]) / (max_obj - min_obj)
        return dist

    def _cruzamento_blx(self, p1, p2, alpha=0.5):
        c1, c2 = [], []
        for i in range(len(self.bounds)):
            c_min = min(p1[i], p2[i])
            c_max = max(p1[i], p2[i])
            I = c_max - c_min
            val1 = random.uniform(c_min - alpha*I, c_max + alpha*I)
            val2 = random.uniform(c_min - alpha*I, c_max + alpha*I)
            c1.append(np.clip(val1, self.bounds[i][0], self.bounds[i][1]))
            c2.append(np.clip(val2, self.bounds[i][0], self.bounds[i][1]))
        return c1, c2

    def _mutacao(self, ind):
        for i in range(len(self.bounds)):
            if random.random() < self.taxa_mutacao:
                escala = (self.bounds[i][1] - self.bounds[i][0]) * 0.1
                ind[i] = np.clip(ind[i] + random.gauss(0, escala), self.bounds[i][0], self.bounds[i][1])
        return ind

    def solve(self):
        """Orquestra as gerações evolucionárias do Rank de Pareto"""
        pop_pai = self._inicializar_populacao()
        
        for geracao in range(self.generations):
            objs_pai = [self._simular_e_avaliar(ind) for ind in pop_pai]
            fronts_pai, rank_vetor_pai = self._fast_non_dominated_sort(objs_pai)
            
            pop_filho = []
            while len(pop_filho) < self.pop_size:
                # Sorteio via Torneio Binário Estrito
                idx1, idx2 = random.sample(range(self.pop_size), 2)
                p1 = pop_pai[idx1] if rank_vetor_pai[idx1] < rank_vetor_pai[idx2] else pop_pai[idx2]
                idx3, idx4 = random.sample(range(self.pop_size), 2)
                p2 = pop_pai[idx3] if rank_vetor_pai[idx3] < rank_vetor_pai[idx4] else pop_pai[idx4]
                
                c1, c2 = self._cruzamento_blx(p1, p2)
                pop_filho.extend([self._mutacao(c1), self._mutacao(c2)])
            
            # Estratégia Elitista: Fusão 2N e Descarte das 40 piores soluções
            pop_mista = pop_pai + pop_filho[:self.pop_size]
            objs_mista = [self._simular_e_avaliar(ind) for ind in pop_mista]
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
            print(f"Geração {geracao+1:02d}/{self.generations} concluída | Rank 1 Elite: {len(fronts_mista[0])}")
            
        objs_finais = [self._simular_e_avaliar(ind) for ind in pop_pai]
        return pop_pai, objs_finais

    @staticmethod
    def plotar_resultados_otimizacao(solucoes, objetivos):
        """Gera e salva todos os artefatos de dados e convergência tridimensionais"""
        solucoes = np.array(solucoes)
        objetivos = np.array(objetivos)
        v_reais = solucoes[:, 6]
        
        # Filtra os 5 melhores indivíduos por ordem crescente de erro (f1)
        indices_5_melhores = np.argsort(objetivos[:, 0])[:5]
        cores_5 = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        
        # Gráfico 1: Barreira de Pareto (Erro x Velocidade)
        plt.figure(figsize=(7, 5))
        plt.scatter(objetivos[:, 0], v_reais, color='gray', alpha=0.4, label='Outras Soluções Rank 1')
        for idx_cor, idx in enumerate(indices_5_melhores):
            plt.scatter(objetivos[idx, 0], v_reais[idx], color=cores_5[idx_cor], edgecolor='black', s=150, zorder=5, label=f'Melhor {idx_cor+1}')
        plt.xlabel('Precisão de Rastreamento (Erro f1)')
        plt.ylabel('Velocidade de Referência (v_ref)')
        plt.title('Barreira de Pareto: Erro de Rastreamento vs Velocidade')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend()
        plt.savefig('pareto_erro_velocidade.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Gráfico 2: Barreira de Pareto (Erro x Esforço de Controle)
        plt.figure(figsize=(7, 5))
        plt.scatter(objetivos[:, 0], objetivos[:, 1], color='gray', alpha=0.4, label='Outras Soluções Rank 1')
        for idx_cor, idx in enumerate(indices_5_melhores):
            plt.scatter(objetivos[idx, 0], objetivos[idx, 1], color=cores_5[idx_cor], edgecolor='black', s=150, zorder=5, label=f'Melhor {idx_cor+1}')
        plt.xlabel('Precisão de Rastreamento (Erro f1)')
        plt.ylabel('Esforço de Controle (f2)')
        plt.title('Barreira de Pareto: Erro de Rastreamento vs Esforço de Controle')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend()
        plt.savefig('pareto_erro_esforco.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Gráfico 3: Coordenadas Paralelas (Convergência Global do Espaço de Busca)
        plt.figure(figsize=(11, 5))
        df_busca = pd.DataFrame(solucoes, columns=['q_pos', 'q_theta', 'q_delta', 'q_v', 'r_motor', 'r_est', 'v_ref'])
        df_busca['Elite'] = 'Outros Indivíduos'
        for idx_cor, idx in enumerate(indices_5_melhores):
            df_busca.loc[idx, 'Elite'] = f'Melhor {idx_cor+1}'
        df_busca = df_busca.sort_values(by='Elite', ascending=False)
        
        paleta = {f'Melhor {i+1}': cores_5[i] for i in range(5)}
        paleta['Outros Indivíduos'] = '#e0e0e0'
        
        parallel_coordinates(df_busca, 'Elite', color=[paleta[c] for c in df_busca['Elite'].unique()], alpha=0.8, linewidth=2)
        plt.title('Convergência do Espaço de Busca (Verificação de Ótimo Global)')
        plt.ylabel('Valor Numérico do Gene')
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.xticks(rotation=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig('convergencia_espaco_busca.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[SUCESSO] Os 3 gráficos analíticos de Pareto e Convergência Global foram exportados!")