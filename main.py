import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Controladores e Modelos
from path_generator import PathGenerator
from pure_pursuit_controller import PurePursuitController 
from stanley_controller import StanleyController         
from mpc_controller import MPCController
from ackermann_model import AckermannSlipModel
from simulator import Simulator
from GA_opt import GeneticAlgorithmNSGA2
# Funções de Plotagem (incluindo as customizadas)
from plot_data import (SimulationPlotter, plot_trajectories_custom, plot_rpms_custom, plot_motor_commands_custom,
                       plot_mpc_du_custom, plot_steering_and_delta_steering_custom, plot_constraints_in_action,
                       plot_error_analysis_detailed, plot_vehicle_speeds_custom, plot_comparative_constraints)
from visualize_sim import VisualizerSim


def run_simulation(control_method,
                   use_mechanical_differential,
                   use_mpc_slip_constraints,
                   path_x,
                   path_y,
                   path_theta,
                   v_car_ref_sim,
                   T_sim,
                   end_path_threshold=0.4,
                   v_limit=2.0,
                   angle_limit_deg=30):
    """
    Função generalizada que instancia e executa uma simulação com base nos parâmetros.
    """
    model = AckermannSlipModel(use_mechanical_differential=use_mechanical_differential, slip_gain=1)

    # --- Bloco de seleção do controlador ---
    if control_method == "pp":
        controller = PurePursuitController(model=model,
                                path_x=path_x,
                                path_y=path_y,
                                lookahead_distance=0.7) # Parâmetro do Pure Pursuit
    elif control_method == "stanley":
        controller = StanleyController(model=model,
                                path_x=path_x,
                                path_y=path_y,
                                path_theta=path_theta,
                                k=0.5) # Parâmetro do Stanley

    elif control_method == "mpc":
        controller = MPCController(model=model,
                                path_x=path_x,
                                path_y=path_y,
                                path_theta=path_theta,
                                ref_v=v_car_ref_sim,
                                dt=0.1, horizon=10, control_horizon_m=5,
                                use_differential=use_mpc_slip_constraints,
                                q_diag = [3.73, 3.73, 597.11, 103.14, 4043.00],
                                r_diag = [104.41, 104.41, 1.77],
                                v_max=v_limit,
                                delta_max_deg=angle_limit_deg)
    else:
        raise ValueError(f"Controlador desconhecido ou não implementado: {control_method}")

    # --- Instanciação e execução do Simulador ---
    sim = Simulator(model=model,
                    controller=controller,
                    path_x=path_x,
                    path_y=path_y,
                    end_of_path_threshold=end_path_threshold,
                    use_velocity_controller=(control_method != "mpc"),
                    x0=0.0,
                    y0=0.0,
                    theta0=0.0,
                    v_car_ref=v_car_ref_sim,
                    T=T_sim,
                    dt=0.001)

    sim.run()
    return sim.system_log


if __name__ == "__main__":
    # ==============================================================================
    # SELECIONE A ANÁLISE QUE DESEJA EXECUTAR
    # Opções: "COMPARACAO_MPC", "ANALISE_VELOCIDADE", "COMPARACAO_CONTROLADORES", "OTIMIZACAO_NSGA2", "VALIDACAO_GA"
    # ==============================================================================
    ANALISE_A_FAZER = "OTIMIZACAO_NSGA2"

    # --- Geração do Caminho de Referência (comum a todas as análises) ---
    path_gen = PathGenerator(start_pos=(0, 0), start_theta=0)
    path_gen.add_straight(length=3)
    path_gen.add_curve(radius=4, angle_deg=60)   # Curva Suave (Esquerda)
    path_gen.add_straight(length=2)
    path_gen.add_curve(radius=1.5, angle_deg=-90) # Curva Fechada (Direita - Início da Chicane)
    path_gen.add_curve(radius=1.5, angle_deg=90)  # Curva Fechada (Esquerda - Fim da Chicane)
    path_gen.add_straight(length=4)
    path_x, path_y, path_theta = path_gen.get_path()

    # ==============================================================================
    # Bloco de Lógica para a Análise de Velocidade
    # ==============================================================================
    if ANALISE_A_FAZER == "ANALISE_VELOCIDADE":
        print("--- INICIANDO ANÁLISE DE ERRO POR VELOCIDADE ---")
        velocities_to_test = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75, 2.0]
        velocity_analysis_results = []
        rms_results = [] # Lista para armazenar os resultados do RMS para a tabela

        config = {
            'control_method': 'mpc',
            'use_mechanical_differential': False,
            'use_mpc_slip_constraints': True,
        }

        for v in velocities_to_test:
            print(f"\n--- Executando simulação para V = {v:.2f} m/s ---")
            log = run_simulation(
                control_method=config['control_method'],
                use_mechanical_differential=config['use_mechanical_differential'],
                use_mpc_slip_constraints=config['use_mpc_slip_constraints'],
                path_x=path_x, path_y=path_y, path_theta=path_theta,
                v_car_ref_sim=v, T_sim=116, end_path_threshold=0.1
            )
            error_data = Simulator.calculate_tracking_errors(log, path_x, path_y, path_theta)
            rms_xy, rms_theta = Simulator.calculate_rms_error(error_data)
            rms_results.append({
                'velocity': v, 'rms_xy_m': rms_xy, 'rms_theta_deg': rms_theta
            })
            velocity_analysis_results.append({
                'velocity': v, 'log': pd.DataFrame(log.get('vehicle_pose', []), columns=['time', 'x', 'y', 'theta', 'beta']), 'error_data': error_data
            })

        print("\n" + "="*50)
        print(" " * 10 + "Tabela de Resultados - Erro RMS")
        print("="*50)
        print(f"{'Velocidade (m/s)':<20} | {'RMS Posição (m)':<20} | {'RMS Ângulo (graus)':<20}")
        print("-"*65)
        for result in rms_results:
            print(f"{result['velocity']:<20.2f} | {result['rms_xy_m']:<20.4f} | {result['rms_theta_deg']:<20.4f}")
        print("="*50)
        plot_error_analysis_detailed(velocity_analysis_results, path_x, path_y)
    
    # ==============================================================================
    # Bloco de Lógica para a Comparação de Configurações MPC
    # ==============================================================================
    elif ANALISE_A_FAZER == "COMPARACAO_MPC":
        print("--- INICIANDO COMPARAÇÃO DE CONFIGURAÇÕES MPC ---")
        experiments = [
            {'label': 'MPC (Com Dif Mec)', 'control_method': 'mpc', 'use_mechanical_differential': True, 'use_mpc_slip_constraints': False},
            {'label': 'MPC (Sem Dif Mec, CR)', 'control_method': 'mpc', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': True},
            {'label': 'MPC (Sem Dif Mec, SR)', 'control_method': 'mpc', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': False},
        ]

        all_logs = []
        labels = []
        for exp in experiments:
            log = run_simulation(
                control_method=exp['control_method'], use_mechanical_differential=exp['use_mechanical_differential'],
                use_mpc_slip_constraints=exp['use_mpc_slip_constraints'], path_x=path_x, path_y=path_y, path_theta=path_theta,
                v_car_ref_sim=0.75, T_sim=48
            )
            all_logs.append(log)
            labels.append(exp['label'])

        plotter = SimulationPlotter(log_data_list=all_logs, labels=labels)
        plot_trajectories_custom(plotter, path_x, path_y, heading_step=100)
        plot_rpms_custom(plotter, experiments)
        plot_motor_commands_custom(plotter, experiments)
        plot_vehicle_speeds_custom(plotter)
        plot_mpc_du_custom(plotter)
        plot_steering_and_delta_steering_custom(plotter)

        v_limit_baixo = 0.6
        angle_limit_baixo = 6
        label_base = 'MPC (Sem Dif Mec, CR)'
        try:
            indice_log_real = labels.index(label_base)
            log_limite_real = all_logs[indice_log_real]
            log_limite_baixo = run_simulation(
                control_method='mpc', use_mechanical_differential=False, use_mpc_slip_constraints=True,
                path_x=path_x, path_y=path_y, path_theta=path_theta, v_car_ref_sim=1.0, T_sim=50,
                v_limit=v_limit_baixo, angle_limit_deg=angle_limit_baixo
            )
            plot_comparative_constraints(
                log_real=log_limite_real, label_real=f'{label_base} (Limite Real)', log_baixo=log_limite_baixo,
                label_baixo=f'{label_base} (Limite Baixo)', v_limit_real=2, v_limit_baixo=v_limit_baixo,
                angle_limit_real=30, angle_limit_baixo=angle_limit_baixo
            )
        except ValueError:
            print("Aviso: Experimento base não encontrado para restrições.")

        visualizer = VisualizerSim(path_x=path_x, path_y=path_y)
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]
        for i, log in enumerate(all_logs):
            vehicle_poses = log.get("vehicle_pose", [])
            if vehicle_poses:
                visualizer.add_car_trajectory([p[1] for p in vehicle_poses], [p[2] for p in vehicle_poses], [p[3] for p in vehicle_poses], color=colors[i % len(colors)])
        visualizer.render(sleep_time=0.01, record_video=False, video_path="simulacao_comparativa_mpc.mp4", follow_car_index=0)

    # ==============================================================================
    # MODO 3: Comparação dos Controladores PP, Stanley e MPC
    # ==============================================================================
    elif ANALISE_A_FAZER == "COMPARACAO_CONTROLADORES":
        print("--- INICIANDO COMPARAÇÃO GERAL DE CONTROLADORES ---")
        experiments = [
            {'label': 'MPC (Com Dif Mec)', 'control_method': 'mpc', 'use_mechanical_differential': True, 'use_mpc_slip_constraints': False},
            {'label': 'MPC (Sem Dif, C/ Restr. Slip)', 'control_method': 'mpc', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': True},
            {'label': 'MPC (Sem Dif, Sem Restr.)', 'control_method': 'mpc', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': False},
            {'label': 'Stanley (Com Dif Mec)', 'control_method': 'stanley', 'use_mechanical_differential': True, 'use_mpc_slip_constraints': False},
            {'label': 'Stanley (Sem Dif Mec)', 'control_method': 'stanley', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': False},
            {'label': 'Pure Pursuit (Com Dif Mec)', 'control_method': 'pp', 'use_mechanical_differential': True, 'use_mpc_slip_constraints': False},
            {'label': 'Pure Pursuit (Sem Dif Mec)', 'control_method': 'pp', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': False},
        ]

        all_logs = []
        labels = []
        rms_results = [] 
        v_ref = 0.75 
        T_sim = 60   

        for exp in experiments:
            log = run_simulation(
                control_method=exp['control_method'], use_mechanical_differential=exp['use_mechanical_differential'],
                use_mpc_slip_constraints=exp['use_mpc_slip_constraints'], path_x=path_x, path_y=path_y, path_theta=path_theta,
                v_car_ref_sim=v_ref, T_sim=T_sim
            )
            all_logs.append(log)
            labels.append(exp['label'])
            error_data = Simulator.calculate_tracking_errors(log, path_x, path_y, path_theta)
            rms_xy, rms_theta = Simulator.calculate_rms_error(error_data)
            rms_results.append({'label': exp['label'], 'rms_xy_m': rms_xy, 'rms_theta_deg': rms_theta})

        print("\n" + "="*80)
        print(f"{'Configuração':<35} | {'RMS Posição (m)':<20} | {'RMS Ângulo (graus)':<20}")
        print("-"*80)
        for result in sorted(rms_results, key=lambda x: x['label']):
            print(f"{result['label']:<35} | {result['rms_xy_m']:<20.4f} | {result['rms_theta_deg']:<20.4f}")
        print("="*80)

        labels_for_filtered_plot = ['Pure Pursuit (Sem Dif Mec)', 'Stanley (Sem Dif Mec)', 'MPC (Sem Dif, C/ Restr. Slip)']
        filtered_logs = [all_logs[labels.index(lbl)] for lbl in labels_for_filtered_plot if lbl in labels]
        filtered_labels = [lbl for lbl in labels_for_filtered_plot if lbl in labels]

        plotter_filtered = SimulationPlotter(log_data_list=filtered_logs, labels=filtered_labels)
        plot_trajectories_custom(plotter_filtered, path_x, path_y, heading_step=150)
        plotter_filtered.plot_rpms() 
        plotter_filtered.plot_velocity_commands()
        plotter_filtered.plot_steering_angles()
        plotter_filtered.plot_vehicle_speeds(target_linear_speed=v_ref)
        
        visualizer = VisualizerSim(path_x=path_x, path_y=path_y)
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]
        for i, log in enumerate(all_logs):
            vehicle_poses = log.get("vehicle_pose", [])
            if vehicle_poses:
                visualizer.add_car_trajectory([p[1] for p in vehicle_poses], [p[2] for p in vehicle_poses], [p[3] for p in vehicle_poses], color=colors[i % len(colors)])
        visualizer.render(sleep_time=0.01, record_video=False, video_path="simulacao_comparativa_controladores.mp4", follow_car_index=1)
    
    # =========================================================================
    # MODIFICAÇÃO EXCLUSIVA: OTIMIZAÇÃO (NSGA2) COM PERSISTÊNCIA E 5 ELITES
    # =========================================================================
    elif ANALISE_A_FAZER == "OTIMIZACAO_NSGA2":
        print("\n" + "="*60)
        print(" INICIANDO TREINAMENTO OFFLINE DO MPC COM NSGA-II")
        print("="*60)
        
        # Instanciação clássica e execução concentrada dentro do GA_opt
        ga = GeneticAlgorithmNSGA2(path_x, path_y, path_theta, pop_size=40, generations=24)
        solucoes_pareto, objetivos_pareto = ga.solve()
        
        # Persistência em disco rígido: Salvamento binário das matrizes (.npy)
        np.save('ga_cached_pop.npy', solucoes_pareto)
        np.save('ga_cached_objs.npy', objetivos_pareto)
        print("\n[SUCESSO] População final e custos salvos em disco ('ga_cached_pop.npy' e 'ga_cached_objs.npy')!")
        
        # Delegação estrita: O main solicita a plotagem das barreiras e convergência global
        print("[PROCESSO] Gerando gráficos de Pareto e Coordenadas Paralelas de Busca...")
        GeneticAlgorithmNSGA2.plotar_resultados_otimizacao(solucoes_pareto, objetivos_pareto)

    # =========================================================================
    # MODIFICAÇÃO EXCLUSIVA: VALIDAÇÃO MULTICORES NAS 3 PISTAS INÉDITAS
    # =========================================================================
    elif ANALISE_A_FAZER == "VALIDACAO_GA":
        print("\n" + "="*80)
        print(" INICIANDO LAÇO DE VALIDAÇÃO CRUZADA MULTICRITÉRIO (SEM RETREINAR)")
        print("="*80)

        if not os.path.exists('ga_cached_pop.npy') or not os.path.exists('ga_cached_objs.npy'):
            print("[ERRO CRÍTICO] Cache não encontrado! Execute a opção 'OTIMIZACAO_NSGA2' pelo menos uma vez.")
        else:
            pop_final = np.load('ga_cached_pop.npy')
            objs_final = np.load('ga_cached_objs.npy')
            print(f"[CACHE] Carregados {len(pop_final)} indivíduos da fronteira de Pareto ótima em disco.")

            # Geração das 3 pistas fechadas inéditas para validação cruzada de robustez
            pista_A = PathGenerator(start_pos=(0, 0), start_theta=0)
            pista_A.add_straight(length=2.0); pista_A.add_curve(radius=1.5, angle_deg=90)
            pista_A.add_straight(length=3.0); pista_A.add_curve(radius=1.5, angle_deg=90)
            pista_A.add_straight(length=1.5); pista_A.add_curve(radius=1.0, angle_deg=-90)
            pista_A.add_curve(radius=1.0, angle_deg=90); pista_A.add_straight(length=2.0)
            pista_A.add_curve(radius=1.5, angle_deg=90); pista_A.add_straight(length=4.0)
            ax_ref, ay_ref, ath_ref = pista_A.get_path()
            
            pista_B = PathGenerator(start_pos=(0, 0), start_theta=0)
            pista_B.add_straight(length=3.0); pista_B.add_curve(radius=2.0, angle_deg=-45)
            pista_B.add_curve(radius=2.0, angle_deg=90); pista_B.add_curve(radius=2.0, angle_deg=-45)
            pista_B.add_straight(length=4.0)
            bx_ref, by_ref, bth_ref = pista_B.get_path()

            pista_C = PathGenerator(start_pos=(0, 0), start_theta=0)
            pista_C.add_straight(length=4.0); pista_C.add_curve(radius=0.8, angle_deg=90)
            pista_C.add_straight(length=2.0); pista_C.add_curve(radius=0.8, angle_deg=90)
            pista_C.add_straight(length=3.0); pista_C.add_curve(radius=1.0, angle_deg=-90)
            pista_C.add_straight(length=2.0); pista_C.add_curve(radius=0.8, angle_deg=90)
            pista_C.add_straight(length=4.0)
            cx_ref, cy_ref, cth_ref = pista_C.get_path()

            cenarios_pistas = [
                {"nome": "Pista A (Sinuosa Fechada)", "x": ax_ref, "y": ay_ref, "th": ath_ref},
                {"nome": "Pista B (Onda Fechada)", "x": bx_ref, "y": by_ref, "th": bth_ref},
                {"nome": "Pista C (Labirinto Fechado)", "x": cx_ref, "y": cy_ref, "th": cth_ref}
            ]

            # Injeção e casamento do controlador MPC com a nova rotina multicor
            # Extraímos os 5 melhores por f1 para simulação de validação
            idx_precisao = np.argsort(objs_final[:, 0])[:5]
            cores_5 = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

            for cenario in cenarios_pistas:
                plt.figure(figsize=(8, 6))
                plt.plot(cenario['x'], cenario['y'], 'k--', linewidth=1.5, label='Trajetória de Referência')
                
                for idx_cor, idx in enumerate(idx_precisao):
                    cromossomo = pop_final[idx]
                    q_pos, q_theta, q_delta, q_v, r_motor, r_est, v_ref = GeneticAlgorithmNSGA2.decodificar_genes(cromossomo)
                    
                    model = AckermannSlipModel(use_mechanical_differential=False, slip_gain=1)
                    controller = MPCController(
                        model=model, path_x=cenario['x'], path_y=cenario['y'], path_theta=cenario['th'],
                        ref_v=v_ref, dt=0.1, horizon=10, control_horizon_m=5, use_differential=True,
                        q_diag=[q_pos, q_pos, q_theta, q_delta, q_v], r_diag=[r_motor, r_motor, r_est],
                        v_max=3.5, delta_max_deg=30
                    )
                    sim = Simulator(
                        model=model, controller=controller, path_x=cenario['x'], path_y=cenario['y'],
                        end_of_path_threshold=0.3, use_velocity_controller=False, T=50, dt=0.001
                    )
                    sim.run()
                    log_dados = sim.system_log
                    plt.plot(log_dados['vehicle_pose'][:, 1], log_dados['vehicle_pose'][:, 2], color=cores_5[idx_cor], linewidth=2, label=f'Melhor {idx_cor+1}')
                    
                plt.xlabel('X (m)'); plt.ylabel('Y (m)')
                plt.title(f"Validação Cruzada Estendida: {cenario['nome']}")
                plt.grid(True, linestyle=':', alpha=0.5); plt.axis('equal'); plt.legend(loc='best')
                plt.show()