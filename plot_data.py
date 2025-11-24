import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec


class SimulationPlotter:

    def __init__(self, log_data_list: list, labels: list):
        """
        Initializes the SimulationPlotter with a list of log data dictionaries
        and corresponding labels for an easy comparison.
        """
        if len(log_data_list) != len(labels):
            raise ValueError("The number of log_data sets must match the number of labels.")

        self.log_data_list = log_data_list
        self.labels = labels
        self.processed_logs = self._process_logs()

    def _process_logs(self):
        processed_logs = []
        for log_data in self.log_data_list:
            processed = {}
            for key, data_entries in log_data.items():
                if not data_entries:
                    processed[key] = pd.DataFrame()
                    continue

                columns_map = {
                    "rpm_ref": ['time', 'left', 'right'],
                    "rpms": ['time', 'left', 'right'],
                    "motor_cmd": ['time', 'left', 'right'],
                    "vehicle_pose": ['time', 'x', 'y', 'theta', 'beta'],
                    "vehicle_speeds": ['time', 'linear_v', 'angular_omega_rads', 'v_left_actual', 'v_right_actual'],
                    "control_delta": ['time', 'delta_rad'],
                    "vel_cmd": ['time', 'v_left_cmd', 'v_right_cmd'],
                    "mpc_du": ['time', 'dvl', 'dvr', 'd_delta'],
                }
                columns = columns_map.get(key)

                if columns:
                    valid_entries = [entry for entry in data_entries if len(entry) == len(columns)]
                    processed[key] = pd.DataFrame(valid_entries, columns=columns) if valid_entries else pd.DataFrame(
                        columns=columns)
                else:
                    try:
                        processed[key] = pd.DataFrame(data_entries)
                    except Exception:
                        processed[key] = pd.DataFrame()
            processed_logs.append(processed)
        return processed_logs

    def plot_trajectories(self, reference_path_x=None, reference_path_y=None, title_suffix="", **kwargs):
        # 1ª MUDANÇA: Passa para a função customizada que agora terá estilos diferentes
        plot_trajectories_custom(self, reference_path_x, reference_path_y, title_suffix=title_suffix, **kwargs)

    def plot_rpms(self, title_suffix=""):
        # Esta função permanece como na última alteração (layout 1x3)
        fig, axs = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)
        # fig.suptitle(f"Perfil de Velocidade das Rodas (RPM) {title_suffix}", fontsize=16)

        stanley_linestyle = (0, (5, 2)) 

        logs_by_controller = {}
        for i, label in enumerate(self.labels):
            if "Pure Pursuit" in label:
                logs_by_controller['pp'] = self.processed_logs[i]
            elif "Stanley" in label:
                logs_by_controller['st'] = self.processed_logs[i]
            elif "MPC" in label:
                logs_by_controller['mpc'] = self.processed_logs[i]
        
        ax_pp_st = axs[0]
        ax_pp_st.set_title("Pure Pursuit & Stanley")
        ax_pp_st.set_xlabel("Tempo (s)")
        ax_pp_st.set_ylabel("RPM")
        
        if 'pp' in logs_by_controller:
            df_actual = logs_by_controller['pp'].get("rpms")
            if df_actual is not None and not df_actual.empty:
                ax_pp_st.plot(df_actual['time'], df_actual['left'], linestyle='-', label='Pure Pursuit')
        
        if 'st' in logs_by_controller:
            df_actual = logs_by_controller['st'].get("rpms")
            if df_actual is not None and not df_actual.empty:
                ax_pp_st.plot(df_actual['time'], df_actual['left'], linestyle=stanley_linestyle, label='Stanley')
        
        ax_pp_st.legend(loc='lower right')
        ax_pp_st.grid(True)

        if 'mpc' in logs_by_controller:
            df_ref = logs_by_controller['mpc'].get("rpm_ref")
            df_actual = logs_by_controller['mpc'].get("rpms")
            
            ax_mpc_l = axs[1]
            ax_mpc_l.set_title("MPC - Roda Esquerda")
            ax_mpc_l.set_xlabel("Tempo (s)")
            if df_actual is not None and not df_actual.empty:
                ax_mpc_l.plot(df_actual['time'], df_actual['left'], label='Real')
            if df_ref is not None and not df_ref.empty:
                ax_mpc_l.plot(df_ref['time'], df_ref['left'], 'r--', label='Ref.')
            ax_mpc_l.legend(loc='lower right')
            ax_mpc_l.grid(True)

            ax_mpc_r = axs[2]
            ax_mpc_r.set_title("MPC - Roda Direita")
            ax_mpc_r.set_xlabel("Tempo (s)")
            if df_actual is not None and not df_actual.empty:
                ax_mpc_r.plot(df_actual['time'], df_actual['right'], label='Real')
            if df_ref is not None and not df_ref.empty:
                ax_mpc_r.plot(df_ref['time'], df_ref['right'], 'r--', label='Ref.')
            ax_mpc_r.legend(loc='lower right')
            ax_mpc_r.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_motor_commands(self, title_suffix=""):
        # Sem alterações aqui
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # fig.suptitle(f"Sinal de Comando do Motor {title_suffix}", fontsize=16)
        
        stanley_linestyle = (0, (5, 2))

        for i, log in enumerate(self.processed_logs):
            label = self.labels[i]
            style = stanley_linestyle if 'Stanley' in label else '-'
            df_motor_cmd = log.get("motor_cmd")
            if not df_motor_cmd.empty:
                if 'left' in df_motor_cmd.columns:
                    axs[0].plot(df_motor_cmd['time'], df_motor_cmd['left'], linestyle=style, label=f"{label} Esquerda")
                if 'right' in df_motor_cmd.columns:
                    axs[1].plot(df_motor_cmd['time'], df_motor_cmd['right'], linestyle=style, label=f"{label} Direita")
        axs[0].set(ylabel="Comando Motor Esq")
        axs[0].legend(loc='upper right')
        axs[0].grid(True)
        axs[1].set(ylabel="Comando Motor Dir", xlabel="Tempo(s)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_velocity_commands(self):
        """
        2ª MUDANÇA: Plota todos os comandos de velocidade em um único gráfico.
        - MPC mostra os comandos da esquerda e da direita.
        """
        plt.figure(figsize=(14, 7))
        ax = plt.gca() # Pega o eixo atual

        # Define estilos para distinguir os comandos do MPC
        mpc_left_style = '-'
        mpc_right_style = (0, (2, 2)) # Estilo pontilhado para a direita

        for i, log in enumerate(self.processed_logs):
            label = self.labels[i]
            df_vel_cmd = log.get("vel_cmd")
            if df_vel_cmd is None or df_vel_cmd.empty:
                continue

            if "Pure Pursuit" in label:
                ax.plot(df_vel_cmd['time'], df_vel_cmd['v_left_cmd'], linestyle='-', label=label)
            
            elif "Stanley" in label:
                ax.plot(df_vel_cmd['time'], df_vel_cmd['v_left_cmd'], linestyle=(0, (3, 5)), label=label)

            elif "MPC" in label:
                # Plota as duas linhas para o MPC
                ax.plot(df_vel_cmd['time'], df_vel_cmd['v_left_cmd'], linestyle=mpc_left_style, label=f'{label} (Esq)')
                ax.plot(df_vel_cmd['time'], df_vel_cmd['v_right_cmd'], linestyle=mpc_right_style, label=f'{label} (Dir)')
        
        # ax.set_title("Comparativo de Comando de Velocidade (Linear)")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Velocidade (m/s)")
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_steering_angles(self, title_suffix=""):
        # Sem alterações aqui
        plt.figure(figsize=(12, 6))
        
        stanley_linestyle = '-'

        for i, log in enumerate(self.processed_logs):
            label = self.labels[i]
            style = stanley_linestyle if 'Stanley' in label else '-'
            df_delta = log.get("control_delta")
            if df_delta is not None and not df_delta.empty:
                plt.plot(df_delta['time'], np.rad2deg(df_delta['delta_rad']), linestyle=style, label=label)

        plt.xlabel("Tempo (s)")
        plt.ylabel("Ângulo de Esterçamento (°)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_vehicle_speeds(self, target_linear_speed=None, title_suffix=""):
        # MUDANÇA APLICADA AQUI
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        styles = {
            'pp': '--',                
            'st': (0, (5, 2, 5, 2)),   
            'mpc': ':',                
        }

        for i, log in enumerate(self.processed_logs):
            label = self.labels[i]
            
            style_key = 'st' if 'Stanley' in label else 'mpc' if 'MPC' in label else 'pp'
            current_style_for_linear = styles[style_key]

            df_speeds = log.get("vehicle_speeds")
            if not df_speeds.empty:
                # Velocidade linear usa o estilo customizado
                if 'linear_v' in df_speeds.columns:
                    axs[0].plot(df_speeds['time'], df_speeds['linear_v'], linestyle=current_style_for_linear, label=label)
                
                # Taxa de variação angular usa linha contínua ('-') para todos
                if 'angular_omega_rads' in df_speeds.columns:
                    axs[1].plot(df_speeds['time'], np.rad2deg(df_speeds['angular_omega_rads']), linestyle='-', label=label)

        if target_linear_speed is not None:
            axs[0].axhline(target_linear_speed, color='r', linestyle='--', label=f'Vel. Referencial ({target_linear_speed} m/s)')
        
        axs[0].set(ylabel="Vel. Linear V (m/s)")
        axs[0].legend(loc='lower right')
        axs[0].grid(True)
        axs[1].set(ylabel="Taxa de Variação do Ângulo δ(°/s)", xlabel="Time (s)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    def plot_heading_over_time(self, title_suffix=""):
        plt.figure(figsize=(12, 6))
        for i, log in enumerate(self.processed_logs):
            df_pose = log.get("vehicle_pose")
            if not df_pose.empty and 'theta' in df_pose.columns:
                plt.plot(df_pose['time'], np.rad2deg(df_pose['theta']), label=f"{self.labels[i]} Actual Heading")
        plt.title(f"Vehicle Heading Over Time {title_suffix}")
        plt.xlabel("Time (s)")
        plt.ylabel("Heading (°)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_mpc_du(self, title_suffix=""):
        plot_mpc_du_custom(self, title_suffix=title_suffix)


# ==============================================================================
# Funções de Plotagem Customizadas
# ==============================================================================

def plot_trajectories_custom(plotter, ref_x, ref_y, title_suffix="", heading_step=200, heading_scale=5, heading_width=0.005):
    # 1ª MUDANÇA: Adiciona estilos de linha diferentes para cada controlador
    plt.figure(figsize=(12, 9))
    if ref_x is not None and ref_y is not None:
        plt.plot(ref_x, ref_y, "k-", label="Referência", linewidth=2)
    
    # Define os estilos de linha únicos
    styles = {
        'pp': '--',                # Pure Pursuit: Tracejado normal
        'st': (0, (5, 2, 1, 2)),   # Stanley: Traço-ponto-ponto
        'mpc': ':',                # MPC: Pontilhado
    }

    for i, log in enumerate(plotter.processed_logs):
        label = plotter.labels[i]
        
        style_key = 'st' if 'Stanley' in label else 'mpc' if 'MPC' in label else 'pp'
        current_style = styles[style_key]
        
        df_pose = log.get("vehicle_pose")
        if not df_pose.empty and all(col in df_pose.columns for col in ['x', 'y', 'theta']):
            line, = plt.plot(df_pose['x'], df_pose['y'], linestyle=current_style, label=f"{label}")
            trajectory_color = line.get_color()
            
            if len(df_pose['x']) >= heading_step:
                q_x, q_y, q_theta = df_pose[['x', 'y', 'theta']].iloc[::heading_step].values.T
            elif len(df_pose['x']) > 0:
                q_x, q_y, q_theta = df_pose[['x', 'y', 'theta']].iloc[0].values
            else:
                continue
            
            plt.quiver(q_x, q_y, np.cos(q_theta), np.sin(q_theta), scale=heading_scale, width=heading_width,
                       color=trajectory_color, angles='xy', scale_units='xy', headwidth=4, headlength=5,
                       minshaft=1, minlength=1)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc='best')
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_log_by_properties(plotter, experiments, use_mech, use_slip):
    """Função auxiliar para encontrar a etiqueta e o log corretos."""
    for exp in experiments:
        if exp.get('use_mechanical_differential') is use_mech and exp.get('use_mpc_slip_constraints') is use_slip:
            label = exp['label']
            try:
                log_index = plotter.labels.index(label)
                return plotter.processed_logs[log_index], label
            except ValueError:
                continue
    return None, None

def plot_rpms_custom(plotter, experiments):
    """
    Imagem 2: Comparação de RPMs em uma grade 3x2 para os 3 cenários do MPC.
    """
    # Cria uma grade 3x2 para acomodar todos os cenários
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True)
    # fig.suptitle('Comparativo de Perfis de RPM (MPC)', fontsize=16)

    # Encontra os logs para cada um dos 3 cenários
    log_mec, label_mec = find_log_by_properties(plotter, experiments, use_mech=True, use_slip=False)
    log_constr, label_constr = find_log_by_properties(plotter, experiments, use_mech=False, use_slip=True)
    log_unconstr, label_unconstr = find_log_by_properties(plotter, experiments, use_mech=False, use_slip=False)

    logs_info = [
        (log_mec, label_mec),
        (log_constr, label_constr),
        (log_unconstr, label_unconstr)
    ]

    for i, (log, label) in enumerate(logs_info):
        if log:
            # Coluna 0: Roda Esquerda
            axs[i, 0].plot(log['rpm_ref']['time'], log['rpm_ref']['left'], 'r--', label='Ref. RPM')
            axs[i, 0].plot(log['rpms']['time'], log['rpms']['left'], 'b-', label='Real RPM')
            axs[i, 0].set_ylabel(f"{label}\nRPM")
            axs[i, 0].legend(loc='lower right')
            axs[i, 0].grid(True)
            
            # Coluna 1: Roda Direita
            axs[i, 1].plot(log['rpm_ref']['time'], log['rpm_ref']['right'], 'r--', label='Ref. RPM')
            axs[i, 1].plot(log['rpms']['time'], log['rpms']['right'], 'b-', label='Real RPM')
            axs[i, 1].legend(loc='lower right')
            axs[i, 1].grid(True)

    # Adiciona títulos apenas para as colunas
    axs[0, 0].set_title("Roda Esquerda")
    axs[0, 1].set_title("Roda Direita")

    # Adiciona o rótulo do eixo X apenas na última linha
    axs[2, 0].set_xlabel("Tempo (s)")
    axs[2, 1].set_xlabel("Tempo (s)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_motor_commands_custom(plotter, experiments):
    """
    Imagem 3: Compara os comandos dos motores Esq/Dir para os 3 cenários MPC.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # fig.suptitle('Comparativo de Comando do Motor (MPC)', fontsize=16)

    # Itera sobre todos os logs (experimentos) disponíveis no plotter
    for i, log in enumerate(plotter.processed_logs):
        label = plotter.labels[i]
        df_motor_cmd = log.get("motor_cmd")
        
        if df_motor_cmd is not None and not df_motor_cmd.empty:
            # Plota o comando do motor esquerdo no gráfico de cima
            axs[0].plot(df_motor_cmd['time'], df_motor_cmd['left'], label=label)
            # Plota o comando do motor direito no gráfico de baixo
            axs[1].plot(df_motor_cmd['time'], df_motor_cmd['right'], label=label)

    axs[0].set_ylabel("Comando Motor Esquerdo")
    axs[0].legend(loc='best')
    axs[0].grid(True)
    
    axs[1].set_ylabel("Comando Motor Direito")
    axs[1].set_xlabel("Tempo (s)")
    axs[1].legend(loc='best')
    axs[1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_vehicle_speeds_custom(plotter, target_linear_speed=None, title_suffix=""):
    """
    Imagem 4: Plota os perfis de velocidade linear e angular do veículo 
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # fig.suptitle(f"Imagem 4: Perfis de Velocidade do Veículo {title_suffix}", fontsize=16)

    # Itera sobre cada um dos logs de simulação
    for i, log in enumerate(plotter.processed_logs):
        label_prefix = plotter.labels[i]
        df_speeds = log.get("vehicle_speeds")

        if df_speeds is not None and not df_speeds.empty:
            # Plota a velocidade linear no gráfico de cima
            if 'linear_v' in df_speeds.columns:
                axs[0].plot(df_speeds['time'], df_speeds['linear_v'],
                            label=f"{label_prefix}")

            # Plota a velocidade angular no gráfico de baixo
            if 'angular_omega_rads' in df_speeds.columns:
                axs[1].plot(df_speeds['time'], np.rad2deg(df_speeds['angular_omega_rads']),
                            label=f"{label_prefix} ")

    # Adiciona a linha de velocidade de referência
    if target_linear_speed is not None:
        axs[0].axhline(target_linear_speed, color='r', linestyle='--',
                       label=f'Velocidade Referencial ({target_linear_speed} m/s)')

    # Configura os eixos
    axs[0].set_ylabel("Velocidade Linear (m/s)")
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    axs[1].set_ylabel("Taxa de Variação Angular de δ (°/s)")
    axs[1].set_xlabel("Tempo (s)")
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_mpc_du_custom(plotter, title_suffix=""):
    """Imagem 5: MPC Control Inputs (du) Comparison, em subplots."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # fig.suptitle(f"Imagem 5: Comparativo de Entradas de Controle MPC (ΔV) {title_suffix}", fontsize=16)
    for i, log in enumerate(plotter.processed_logs):
        df_mpc_du = log.get("mpc_du")
        if not df_mpc_du.empty:
            if 'dvl' in df_mpc_du.columns:
                axs[0].plot(df_mpc_du['time'], df_mpc_du['dvl'], label=f'{plotter.labels[i]}')
            if 'dvr' in df_mpc_du.columns:
                axs[1].plot(df_mpc_du['time'], df_mpc_du['dvr'], label=f'{plotter.labels[i]}')
    axs[0].set(ylabel="ΔV Esquerda (m/s)")
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    axs[1].set(ylabel="ΔV Direita (m/s)", xlabel="Tempo (s)")
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_steering_and_delta_steering_custom(plotter):
    """Imagem 6: Ângulo de Esterçamento e MPC du para delta steering."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # fig.suptitle("Imagem 6: Atuação do Esterçamento", fontsize=16)
    for i, log in enumerate(plotter.processed_logs):
        df_delta = log.get("control_delta")
        if not df_delta.empty:
            axs[0].plot(df_delta['time'], np.rad2deg(df_delta['delta_rad']), label=f'{plotter.labels[i]}')
    axs[0].set_title("Perfil do Ângulo de Esterçamento (δ)")
    axs[0].set(ylabel="Ângulo (°)")
    axs[0].grid(True)
    axs[0].legend()
    for i, log in enumerate(plotter.processed_logs):
        df_mpc_du = log.get("mpc_du")
        if not df_mpc_du.empty:
            time_points = df_mpc_du['time'].to_numpy()
            values_deg = np.rad2deg(df_mpc_du['d_delta'].to_numpy())
            if len(time_points) > 1:
                dt = time_points[1] - time_points[0]
                edges = np.append(time_points, time_points[-1] + dt)
                if len(values_deg) + 1 == len(edges):
                    axs[1].stairs(values_deg, edges, label=f'{plotter.labels[i]}')
    axs[1].set_title("Entrada de Controle MPC (Δδ)")
    axs[1].set(ylabel="Variação do Ângulo (°)", xlabel="Tempo (s)")
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_constraints_in_action(plotter, speed_limit, angle_limit_deg):
    """Imagem 7: Mostra as restrições de velocidade e ângulo funcionando."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # fig.suptitle("Imagem 7: Verificação de Restrições", fontsize=16)
    for i, log in enumerate(plotter.processed_logs):
        df_speeds = log.get("vehicle_speeds")
        if not df_speeds.empty:
            axs[0].plot(df_speeds['time'], df_speeds['linear_v'], label=f'{plotter.labels[i]}')
    axs[0].axhline(y=speed_limit, color='r', linestyle='--', label=f'Limite ({speed_limit} m/s)')
    axs[0].set_title("Velocidade Linear do Veículo")
    axs[0].set(ylabel="Velocidade (m/s)")
    axs[0].grid(True)
    axs[0].legend()
    for i, log in enumerate(plotter.processed_logs):
        df_delta = log.get("control_delta")
        if not df_delta.empty:
            axs[1].plot(df_delta['time'], np.rad2deg(df_delta['delta_rad']), label=f'{plotter.labels[i]}')
    axs[1].axhline(y=angle_limit_deg, color='r', linestyle='--', label=f'Limite ({angle_limit_deg}°)')
    axs[1].axhline(y=-angle_limit_deg, color='r', linestyle='--')
    axs[1].set_title("Ângulo de Esterçamento δ (°)")
    axs[1].set(ylabel="Ângulo (°)", xlabel="Tempo (s)")
    axs[1].grid(True)
    axs[1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_comparative_constraints(log_real, label_real, log_baixo, label_baixo, v_limit_real, v_limit_baixo, angle_limit_real, angle_limit_baixo):
    """
    Imagem 7 (Versão Comparativa): Mostra o mesmo controlador rodando com dois conjuntos
    de limites (real e baixo) para provar que as restrições funcionam.
    """
    # Processa os logs para DataFrame do Pandas para facilitar o acesso
    df_real_speeds = pd.DataFrame(log_real.get("vehicle_speeds", []), columns=['time', 'linear_v', 'angular_omega_rads', 'v_left_actual', 'v_right_actual'])
    df_real_delta = pd.DataFrame(log_real.get("control_delta", []), columns=['time', 'delta_rad'])

    df_baixo_speeds = pd.DataFrame(log_baixo.get("vehicle_speeds", []), columns=['time', 'linear_v', 'angular_omega_rads', 'v_left_actual', 'v_right_actual'])
    df_baixo_delta = pd.DataFrame(log_baixo.get("control_delta", []), columns=['time', 'delta_rad'])

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Gráfico de Cima: Comparativo de Velocidade ---
    # Plota as curvas de velocidade
    axs[0].plot(df_real_speeds['time'], df_real_speeds['linear_v'], label=label_real, color='green')
    axs[0].plot(df_baixo_speeds['time'], df_baixo_speeds['linear_v'], label=label_baixo, color='blue')

    # Plota as linhas de limite
    # axs[0].axhline(y=v_limit_real, color='blue', linestyle='--', label=f'Limite Real ({v_limit_real} m/s)')
    axs[0].axhline(y=v_limit_baixo, color='red', linestyle='--', label=f'Limite Baixo ({v_limit_baixo} m/s)')

    axs[0].set_ylabel("Velocidade Linear (m/s)")
    axs[0].grid(True)
    axs[0].legend(loc='lower right')

    # --- Gráfico de Baixo: Comparativo de Ângulo de Esterçamento ---
    # Plota as curvas de ângulo
    axs[1].plot(df_real_delta['time'], np.rad2deg(df_real_delta['delta_rad']), label=label_real, color='green')
    axs[1].plot(df_baixo_delta['time'], np.rad2deg(df_baixo_delta['delta_rad']), label=label_baixo, color='blue')

    # Plota as linhas de limite
    # axs[1].axhline(y=angle_limit_real, color='blue', linestyle='--', label=f'Limite Real ({angle_limit_real}°)')
    # axs[1].axhline(y=-angle_limit_real, color='blue', linestyle='--')
    axs[1].axhline(y=angle_limit_baixo, color='red', linestyle='--', label=f'Limite Baixo ({angle_limit_baixo}°)')
    axs[1].axhline(y=-angle_limit_baixo, color='red', linestyle='--')

    axs[1].set_ylabel("Ângulo de Esterçamento δ (°)")
    axs[1].set_xlabel("Tempo (s)")
    axs[1].grid(True)
    axs[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()
# ==============================================================================
# Funções de Plotagem para comparação de velocidades
# ==============================================================================

def plot_error_analysis_detailed(results_list, ref_x, ref_y):
    """
    Cria um painel com 3 gráficos, com o eixo do tempo normalizado (0-100%).
    - Esquerda (grande): Trajetórias.
    - Direita (superior): Erro de Posição (x,y).
    - Direita (inferior): Erro de Orientação (theta).
    """
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle("Análise Detalhada de Erro vs. Velocidade (Tempo Normalizado)", fontsize=16)

    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_xy_err = fig.add_subplot(gs[0, 1])
    ax_theta_err = fig.add_subplot(gs[1, 1])

    # --- Painel da Esquerda: Trajetórias (sem alteração) ---
    ax_traj.plot(ref_x, ref_y, "k-", label="Referência", linewidth=2.5)
    for result in results_list:
        velocity = result['velocity']
        df_pose = result['log']
        if df_pose is not None and not df_pose.empty:
            ax_traj.plot(df_pose['x'], df_pose['y'], linestyle='--',
                         label=f'V = {velocity:.2f} m/s')

    ax_traj.set_title("Trajetórias por Velocidade")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_ylim(-0.5, 12.5)
    ax_traj.legend(loc='best')
    # ax_traj.axis("equal")
    ax_traj.grid(True)

    # --- Painel Superior Direito: Erro de Posição (x,y) com TEMPO NORMALIZADO ---
    for result in results_list:
        velocity = result['velocity']
        error_data = result['error_data']
        if error_data:
            time, xy_error, _ = zip(*error_data)

            # Normaliza o eixo do tempo para 0-100%
            max_time = time[-1] if time else 0
            if max_time > 1e-6:
                normalized_time = (np.array(time) / max_time) * 100
            else:
                normalized_time = np.zeros_like(time)

            # Plota o tempo normalizado no eixo X e o erro real no eixo Y
            ax_xy_err.plot(normalized_time, xy_error, label=f'V = {velocity:.2f} m/s')

    ax_xy_err.set_title("Erro de Posição (x,y) vs. Duração")
    ax_xy_err.set_ylabel("Erro (m)") # <-- Rótulo revertido para metros
    ax_xy_err.legend(loc='upper left')
    ax_xy_err.grid(True)
    ax_xy_err.set_ylim(bottom=0)

    # --- Painel Inferior Direito: Erro de Orientação (theta) com TEMPO NORMALIZADO ---
    for result in results_list:
        velocity = result['velocity']
        error_data = result['error_data']
        if error_data:
            time, _, theta_error_rad = zip(*error_data)

            # Normaliza o eixo do tempo para 0-100%
            max_time = time[-1] if time else 0
            if max_time > 1e-6:
                normalized_time = (np.array(time) / max_time) * 100
            else:
                normalized_time = np.zeros_like(time)

            # Plota o tempo normalizado no eixo X e o erro angular real no eixo Y
            ax_theta_err.plot(normalized_time, np.rad2deg(theta_error_rad), label=f'V = {velocity:.2f} m/s')

    ax_theta_err.set_title("Erro de Orientação (Theta) vs. Duração")
    ax_theta_err.set_xlabel("Duração da Trajetória (%)") # <-- Rótulo do eixo X atualizado
    ax_theta_err.set_ylabel("Erro Angular (°)") # <-- Rótulo revertido para graus
    ax_theta_err.legend(loc='upper left')
    ax_theta_err.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
