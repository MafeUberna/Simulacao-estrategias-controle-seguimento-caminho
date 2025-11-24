import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Controladores e Modelos
from path_generator import PathGenerator
from pure_pursuit_controller import PurePursuitController 
from stanley_controller import StanleyController         
from mpc_controller import MPCController
from ackermann_model import AckermannSlipModel
from simulator import Simulator

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
                                q_diag = [2.0, 2.0, 1.0, 0.5, 1.0],
                                r_diag = [2.0, 2.0, 20.0],
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
    # Opções: "COMPARACAO_MPC", "ANALISE_VELOCIDADE", "COMPARACAO_CONTROLADORES"
    ANALISE_A_FAZER = "COMPARACAO_CONTROLADORES"
    # ==============================================================================

    # --- Geração do Caminho de Referência (comum a todas as análises) ---
    path_gen = PathGenerator(start_pos=(0, 0), start_theta=0)
    path_gen.add_straight(length=5)
    path_gen.add_curve(radius=3, angle_deg=60)
    path_gen.add_straight(length=10)
    path_gen.add_curve(radius=3, angle_deg=-60)
    path_gen.add_straight(length=10)
    path_x, path_y, path_theta = path_gen.get_path()

    # ==============================================================================
    # Bloco de Lógica para a Análise de Velocidade
    # ==============================================================================
    if ANALISE_A_FAZER == "ANALISE_VELOCIDADE":
        print("--- INICIANDO ANÁLISE DE ERRO POR VELOCIDADE ---")
        velocities_to_test = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75, 2.0]
        velocity_analysis_results = []
        rms_results = [] # Lista para armazenar os resultados do RMS para a tabela

        # Define uma configuração base de controlador para o teste
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
                path_x=path_x,
                path_y=path_y,
                path_theta=path_theta,
                v_car_ref_sim=v,
                T_sim=116,
                end_path_threshold=0.1
            )

            print("Calculando erros de trajetória...")
            error_data = Simulator.calculate_tracking_errors(log, path_x, path_y, path_theta)
            
            # --- Cálculo e armazenamento do erro RMS ---
            rms_xy, rms_theta = Simulator.calculate_rms_error(error_data)
            rms_results.append({
                'velocity': v,
                'rms_xy_m': rms_xy,
                'rms_theta_deg': rms_theta
            })
            # -----------------------------------------------

            velocity_analysis_results.append({
                'velocity': v,
                'log': pd.DataFrame(log.get('vehicle_pose', []), columns=['time', 'x', 'y', 'theta', 'beta']),
                'error_data': error_data
            })

        # --- Impressão da tabela de resultados ---
        print("\n" + "="*50)
        print(" " * 10 + "Tabela de Resultados - Erro RMS")
        print("="*50)
        print(f"{'Velocidade (m/s)':<20} | {'RMS Posição (m)':<20} | {'RMS Ângulo (graus)':<20}")
        print("-"*65)
        for result in rms_results:
            print(f"{result['velocity']:<20.2f} | {result['rms_xy_m']:<20.4f} | {result['rms_theta_deg']:<20.4f}")
        print("="*50)
        # -----------------------------------------------

        print("\nSimulações concluídas. Gerando gráfico de análise detalhada de erro...")
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
            print(f"\n--- Executando Experimento: {exp['label']} ---")
            log = run_simulation(
                control_method=exp['control_method'],
                use_mechanical_differential=exp['use_mechanical_differential'],
                use_mpc_slip_constraints=exp['use_mpc_slip_constraints'],
                path_x=path_x, path_y=path_y, path_theta=path_theta,
                v_car_ref_sim=0.75 , T_sim=48
            )
            all_logs.append(log)
            labels.append(exp['label'])

        print("\nSimulações concluídas. Gerando gráficos comparativos...")
        plotter = SimulationPlotter(log_data_list=all_logs, labels=labels)

        is_mpc_in_experiments = any(exp['control_method'] == 'mpc' for exp in experiments)
        if is_mpc_in_experiments:
            plot_trajectories_custom(plotter, path_x, path_y, heading_step=100)
            plot_rpms_custom(plotter, experiments)
            plot_motor_commands_custom(plotter, experiments)
            plot_vehicle_speeds_custom(plotter)
            plot_mpc_du_custom(plotter)
            plot_steering_and_delta_steering_custom(plotter)
            # --- Bloco para o Gráfico Comparativo da Figura 7 ---
            print("\n--- Gerando Gráfico Comparativo de Restrições (Figura 7) ---")

            # 1. Definir os limites baixos para o teste
            v_limit_baixo = 0.6
            angle_limit_baixo = 6

            # 2. Encontrar o log da simulação com os limites reais que já rodamos
            # Vamos usar o 'MPC (Sem Dif Mec, CR)' como base para a comparação
            label_base = 'MPC (Sem Dif Mec, CR)'
            try:
                indice_log_real = labels.index(label_base)
                log_limite_real = all_logs[indice_log_real]

                # 3. Rodar uma NOVA simulação para o mesmo controlador, mas com os LIMITES BAIXOS
                print(f"Executando simulação extra para '{label_base}' com limites baixos...")
                log_limite_baixo = run_simulation(
                    control_method='mpc',
                    use_mechanical_differential=False,
                    use_mpc_slip_constraints=True,
                    path_x=path_x, path_y=path_y, path_theta=path_theta,
                    v_car_ref_sim=1.0, T_sim=50,
                    v_limit=v_limit_baixo,          # <-- Passa o limite baixo de velocidade
                    angle_limit_deg=angle_limit_baixo  # <-- Passa o limite baixo de ângulo
                )

                # 4. Chamar a nova função de plotagem com os dois resultados
                plot_comparative_constraints(
                    log_real=log_limite_real,
                    label_real=f'{label_base} (Limite Real)',
                    log_baixo=log_limite_baixo,
                    label_baixo=f'{label_base} (Limite Baixo)',
                    v_limit_real=2,
                    v_limit_baixo=v_limit_baixo,
                    angle_limit_real=30,
                    angle_limit_baixo=angle_limit_baixo
                )
            except ValueError as e:
                print(f"AVISO: O experimento '{label_base}' não foi encontrado. Pulando o gráfico da Figura 7.")
        # ======================================================================
        # NOVO BLOCO: Visualização 3D com PyBullet e Geração de Vídeo para COMPARACAO_MPC
        # ======================================================================
        print("\n--- INICIANDO VISUALIZAÇÃO 3D E GERAÇÃO DE VÍDEO PARA COMPARACAO_MPC ---")

        # 1. Instanciar o VisualizerSim
        visualizer = VisualizerSim(path_x=path_x, path_y=path_y)

        # 2. Adicionar as trajetórias dos carros
        # A lista 'all_logs' já contém os dados para os experimentos MPC
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0.5, 0.5, 0.5, 1]]

        for i, log in enumerate(all_logs):
            vehicle_poses = log.get("vehicle_pose", [])
            if vehicle_poses:
                traj_x = [p[1] for p in vehicle_poses]
                traj_y = [p[2] for p in vehicle_poses]
                traj_theta = [p[3] for p in vehicle_poses]

                car_color = colors[i % len(colors)]
                visualizer.add_car_trajectory(traj_x, traj_y, traj_theta, color=car_color)
                print(f"Adicionada trajetória para: {labels[i]} (Cor: {car_color[:3]})")

        # 3. Renderizar a simulação e gravar o vídeo
        video_output_path = "simulacao_comparativa_mpc.mp4"
        print(f"Renderizando visualização 3D e gravando vídeo para: {video_output_path}")
        # Ajuste sleep_time e follow_car_index conforme sua preferência
        visualizer.render(sleep_time=0.01, record_video=False, video_path=video_output_path, follow_car_index=0) # Exemplo: seguir o primeiro carro
        # ======================================================================

    # ==============================================================================
    # MODO 3: Comparação dos Controladores PP, Stanley e MPC
    # ==============================================================================
    elif ANALISE_A_FAZER == "COMPARACAO_CONTROLADORES":
        print("--- INICIANDO COMPARAÇÃO GERAL DE CONTROLADORES E CENÁRIOS ---")

        # 1° MUDANÇA: Adicionado 'MPC (Sem Dif, Sem Restr.)' à lista de experimentos
        experiments = [
            # MPC
            {'label': 'MPC (Com Dif Mec)', 'control_method': 'mpc', 'use_mechanical_differential': True, 'use_mpc_slip_constraints': False},
            {'label': 'MPC (Sem Dif, C/ Restr. Slip)', 'control_method': 'mpc', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': True},
            {'label': 'MPC (Sem Dif, Sem Restr.)', 'control_method': 'mpc', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': False},

            # Stanley
            {'label': 'Stanley (Com Dif Mec)', 'control_method': 'stanley', 'use_mechanical_differential': True, 'use_mpc_slip_constraints': False},
            {'label': 'Stanley (Sem Dif Mec)', 'control_method': 'stanley', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': False},

            # Pure Pursuit
            {'label': 'Pure Pursuit (Com Dif Mec)', 'control_method': 'pp', 'use_mechanical_differential': True, 'use_mpc_slip_constraints': False},
            {'label': 'Pure Pursuit (Sem Dif Mec)', 'control_method': 'pp', 'use_mechanical_differential': False, 'use_mpc_slip_constraints': False},
        ]

        all_logs = []
        labels = []
        rms_results = [] 
        v_ref = 0.75 
        T_sim = 60   

        for exp in experiments:
            print(f"\n--- Executando Experimento: {exp['label']} ---")
            log = run_simulation(
                control_method=exp['control_method'],
                use_mechanical_differential=exp['use_mechanical_differential'],
                use_mpc_slip_constraints=exp['use_mpc_slip_constraints'],
                path_x=path_x, 
                path_y=path_y, 
                path_theta=path_theta,
                v_car_ref_sim=v_ref,
                T_sim=T_sim,
            )
            all_logs.append(log)
            labels.append(exp['label'])

            print("Calculando erros de trajetória...")
            error_data = Simulator.calculate_tracking_errors(log, path_x, path_y, path_theta)
            rms_xy, rms_theta = Simulator.calculate_rms_error(error_data)
            rms_results.append({
                'label': exp['label'],
                'rms_xy_m': rms_xy,
                'rms_theta_deg': rms_theta
            })

        print("\n" + "="*80)
        print(" " * 20 + "Tabela Comparativa - Erro RMS de Rastreamento")
        print("="*80)
        print(f"{'Configuração':<35} | {'RMS Posição (m)':<20} | {'RMS Ângulo (graus)':<20}")
        print("-"*80)
        rms_results.sort(key=lambda x: x['label'])
        for result in rms_results:
            print(f"{result['label']:<35} | {result['rms_xy_m']:<20.4f} | {result['rms_theta_deg']:<20.4f}")
        print("="*80)

        print("\nSimulações concluídas. Gerando gráficos comparativos para o cenário selecionado...")
        
        # 2° MUDANÇA: Ordem dos gráficos alterada para Pure Pursuit -> Stanley -> MPC
        labels_for_filtered_plot = [
            'Pure Pursuit (Sem Dif Mec)',
            'Stanley (Sem Dif Mec)',
            'MPC (Sem Dif, C/ Restr. Slip)'
        ]
        
        filtered_logs = []
        filtered_labels = []
        
        for label_to_find in labels_for_filtered_plot:
            try:
                idx = labels.index(label_to_find)
                filtered_logs.append(all_logs[idx])
                filtered_labels.append(labels[idx])
            except ValueError:
                print(f"AVISO: O cenário '{label_to_find}' não foi encontrado e será ignorado nos gráficos.")

        plotter_filtered = SimulationPlotter(log_data_list=filtered_logs, labels=filtered_labels)

        print("\nGerando Gráfico 1: Trajetórias")
        plot_trajectories_custom(plotter_filtered, path_x, path_y, heading_step=150)

        # 3° MUDANÇA: Os gráficos de RPM e Comando de Velocidade usarão a nova lógica implementada em `plot_data.py`
        print("Gerando Gráfico 2: Perfil de Velocidade das Rodas (RPM)")
        plotter_filtered.plot_rpms() 

        print("Gerando Gráfico 3: Comando de Velocidade (Linear)")
        plotter_filtered.plot_velocity_commands()

        print("Gerando Gráfico 4: Ângulo de Esterçamento")
        plotter_filtered.plot_steering_angles()

        print("Gerando Gráfico 5: Perfil de Velocidade do Veículo")
        plotter_filtered.plot_vehicle_speeds(target_linear_speed=v_ref)
        
        # ======================================================================
        # NOVO BLOCO: Visualização 3D com PyBullet e Geração de Vídeo
        # ======================================================================
        print("\n--- INICIANDO VISUALIZAÇÃO 3D E GERAÇÃO DE VÍDEO ---")

        # 1. Instanciar o VisualizerSim
        # Passe o caminho de referência (road) para ele.
        visualizer = VisualizerSim(path_x=path_x, path_y=path_y)

        # 2. Adicionar as trajetórias dos carros
        # Itere sobre todos os logs coletados e adicione cada trajetória.
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0.5, 0.5, 0.5, 1]] # Cores para diferentes carros

        for i, log in enumerate(all_logs): # Use all_logs para ter todas as simulações no vídeo
            vehicle_poses = log.get("vehicle_pose", [])
            if vehicle_poses:
                # Extrair x, y, theta do log de pose do veículo
                traj_x = [p[1] for p in vehicle_poses]
                traj_y = [p[2] for p in vehicle_poses]
                traj_theta = [p[3] for p in vehicle_poses] # O PyBullet usa radianos, que já é o que você tem

                # Atribuir uma cor baseada no índice ou usar uma padrão
                car_color = colors[i % len(colors)] # Pega uma cor da lista, ciclando se necessário
                visualizer.add_car_trajectory(traj_x, traj_y, traj_theta, color=car_color)
                print(f"Adicionada trajetória para: {labels[i]} (Cor: {car_color[:3]})") # Adiciona um print para feedback

        # 3. Renderizar a simulação e gravar o vídeo
        # Você pode ajustar sleep_time para controlar a velocidade da animação.
        # Defina record_video=True para gerar o arquivo MP4.
        # O follow_car_index (opcional) fará a câmera seguir um carro específico (0 para o primeiro, 1 para o segundo, etc.)
        video_output_path = "simulacao_comparativa_controladores.mp4"
        print(f"Renderizando visualização 3D e gravando vídeo para: {video_output_path}")
        visualizer.render(sleep_time=0.01, record_video=False, video_path=video_output_path, follow_car_index=1) # Exemplo: seguir o segundo carro
        # ======================================================================

    else:
        print(f"ERRO: Análise '{ANALISE_A_FAZER}' desconhecida. Verifique a variável no início do script.")

    print("\nProcesso finalizado.")