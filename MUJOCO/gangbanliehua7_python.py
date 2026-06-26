import numpy as np
from scipy.stats import norm
from scipy.io import loadmat
import time
import torch
import os
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_factor_As_16 = None
global_factor_As_18 = None
global_DL_bias = None
global_R_bias = None
global_T_sp_samples = None
global_a_samples = None
global_b_samples = None
global_cl_cr = None
global_cl_surface = None
global_cover = None
global_fc = None
global_fy = None
global_fy_p = None
global_fy_pl = None
global_fy_plate = None
global_w_c = None

def init_global_matrices(use_cuda=True):
    global global_factor_As_16, global_factor_As_18, device
    global global_DL_bias, global_R_bias, global_T_sp_samples, global_a_samples, global_b_samples
    global global_cl_cr, global_cl_surface, global_cover, global_fc, global_fy
    global global_fy_p, global_fy_pl, global_fy_plate, global_w_c
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        pass
    else:
        device = torch.device('cpu')
        pass
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        pass
        factor_As_16_path = os.path.join(current_dir, 'factor_As_16.mat')
        factor_As_18_path = os.path.join(current_dir, 'factor_As_18.mat')
        samples_path = os.path.join(current_dir, 'samples.mat')
        pass
        factor_As_16_data = loadmat(factor_As_16_path)
        factor_As_18_data = loadmat(factor_As_18_path)
        samples_data = loadmat(samples_path)
        global_factor_As_16 = torch.tensor(factor_As_16_data['factor_As'], dtype=torch.float32, device=device)
        global_factor_As_18 = torch.tensor(factor_As_18_data['factor_As'], dtype=torch.float32, device=device)
        pass
        global_DL_bias = torch.tensor(samples_data['DL_bias'], dtype=torch.float32, device=device)
        global_R_bias = torch.tensor(samples_data['R_bias'], dtype=torch.float32, device=device)
        global_T_sp_samples = torch.tensor(samples_data['T_sp_samples'], dtype=torch.float32, device=device)
        global_a_samples = torch.tensor(samples_data['a_samples'], dtype=torch.float32, device=device)
        global_b_samples = torch.tensor(samples_data['b_samples'], dtype=torch.float32, device=device)
        global_cl_cr = torch.tensor(samples_data['cl_cr'], dtype=torch.float32, device=device)
        global_cl_surface = torch.tensor(samples_data['cl_surface'], dtype=torch.float32, device=device)
        global_cover = torch.tensor(samples_data['cover'], dtype=torch.float32, device=device)
        global_fc = torch.tensor(samples_data['fc'], dtype=torch.float32, device=device)
        global_fy = torch.tensor(samples_data['fy'], dtype=torch.float32, device=device)
        global_fy_p = torch.tensor(samples_data['fy_p'], dtype=torch.float32, device=device)
        global_fy_pl = torch.tensor(samples_data['fy_pl'], dtype=torch.float32, device=device)
        global_fy_plate = torch.tensor(samples_data['fy_plate'], dtype=torch.float32, device=device)
        global_w_c = torch.tensor(samples_data['w_c'], dtype=torch.float32, device=device)
        pass
        return True
    except Exception as e:
        pass
        return False

def run_simulation(maintenance_times):
    global global_factor_As_16, global_factor_As_18
    global global_DL_bias, global_R_bias, global_T_sp_samples, global_a_samples, global_b_samples
    global global_cl_cr, global_cl_surface, global_cover, global_fc, global_fy
    global global_fy_p, global_fy_pl, global_fy_plate, global_w_c
    if global_factor_As_16 is None or global_fc is None:
        success = init_global_matrices()
        if not success:
            raise RuntimeError('无法加载全局矩阵和样本数据，请检查文件路径')
    start_time = time.time()
    Nsample = int(100000.0)
    time_years = np.arange(1, 201)
    time_years_tensor = torch.tensor(time_years, dtype=torch.float32, device=device)
    maintenance_times = sorted(maintenance_times)
    h = 0.4
    b = 1
    num_steel_18 = 5
    num_steel_16 = 5
    diameter_18 = 18
    diameter_16 = 16
    Es = 200000000000.0
    factor_As_16 = global_factor_As_16
    factor_As_18 = global_factor_As_18
    fc = global_fc
    fy = global_fy.repeat(1, 200)
    diameter_18_tensor = torch.full((Nsample, 1), diameter_18, dtype=torch.float32, device=device)
    diameter_16_tensor = torch.full((Nsample, 1), diameter_16, dtype=torch.float32, device=device)
    diameter = torch.cat([diameter_18_tensor, diameter_16_tensor], dim=0)
    As_one = math.pi * diameter ** 2 / 4 / 1000000
    As_total_steel = num_steel_18 * As_one[0] * factor_As_18 + num_steel_16 * As_one[1] * factor_As_16
    plate_thickness = 6
    plate_width = 60
    num_plates = 10
    fy_pl = global_fy_pl.repeat(1, 200)
    Es_plate = 210000000000.0
    T_sp_samples = global_T_sp_samples
    a_samples = global_a_samples
    b_samples = global_b_samples
    last_maintenance_times = torch.zeros(len(time_years), dtype=torch.float32, device=device)
    for j in range(len(time_years)):
        t = time_years[j]
        last_m = 0
        for m_time in maintenance_times:
            if t >= m_time:
                last_m = m_time
            else:
                break
        last_maintenance_times[j] = torch.tensor(last_m, dtype=torch.float32, device=device)
    times_since_maintenance = time_years_tensor - last_maintenance_times
    times_since_maintenance_expanded = times_since_maintenance.unsqueeze(0).expand(Nsample, -1)
    T_sp_expanded = T_sp_samples.expand(-1, len(time_years))
    a_samples_expanded = a_samples.expand(-1, len(time_years))
    b_samples_expanded = b_samples.expand(-1, len(time_years))
    in_protection = times_since_maintenance_expanded < T_sp_expanded
    time_diff = torch.clamp(times_since_maintenance_expanded - T_sp_expanded, min=0)
    corrosion_depth = a_samples_expanded * torch.pow(time_diff, b_samples_expanded)
    d_sp = torch.where(in_protection, torch.zeros_like(corrosion_depth), corrosion_depth)
    effective_plate_thickness = plate_thickness / 1000 - d_sp
    effective_plate_thickness = torch.clamp(effective_plate_thickness, min=0)
    first_maintenance_time = maintenance_times[0] if maintenance_times else float('inf')
    is_before_first_maintenance = time_years_tensor < first_maintenance_time
    try:
        effective_plate_thickness[:, is_before_first_maintenance] = 0
    except Exception:
        for idx, flag in enumerate(is_before_first_maintenance.tolist()):
            if flag:
                effective_plate_thickness[:, idx] = 0
    As_plate = num_plates * effective_plate_thickness * plate_width / 1000
    d_steel = h - 0.0586
    d_plate = h + effective_plate_thickness / 2
    strain_concrete = 0.0033
    stress_bar = fy * 1000000
    stress_ban = fy_pl * 1000000
    alpha_c = 1.0
    beta_c = 0.8
    epsilon = 1e-10
    compression_depth = torch.max((stress_bar * As_total_steel + stress_ban * As_plate) / (alpha_c * beta_c * fc * b * 1000000), torch.tensor(epsilon, dtype=torch.float32, device=device))
    stress_steel = torch.zeros((Nsample, len(time_years)), dtype=torch.float32, device=device)
    stress_plate = torch.zeros((Nsample, len(time_years)), dtype=torch.float32, device=device)
    iteration_num = 0
    while iteration_num < 10:
        iteration_num += 1
        stress_store_steel = stress_steel.clone()
        stress_store_plate = stress_plate.clone()
        strain_steel = strain_concrete * (d_steel - compression_depth) / (compression_depth + epsilon)
        strain_plate = strain_concrete * (d_plate - compression_depth) / (compression_depth + epsilon)
        stress_steel = strain_steel * Es
        stress_plate = strain_plate * Es_plate
        stress_steel = torch.min(stress_steel, stress_bar)
        stress_plate = torch.min(stress_plate, stress_ban)
        stress_change_steel = torch.abs(stress_steel - stress_store_steel)
        stress_change_plate = torch.abs(stress_plate - stress_store_plate)
        if torch.all(stress_change_steel <= 0.1) and torch.all(stress_change_plate <= 0.1):
            break
    moment_capacity_steel = stress_steel * As_total_steel * (d_steel - compression_depth / 2) / 1000
    moment_capacity_plate = stress_plate * As_plate * (d_plate - compression_depth / 2) / 1000
    moment_capacity_total = moment_capacity_steel + moment_capacity_plate
    mean_moment_capacity_plate = torch.mean(moment_capacity_plate)
    moment_DW = 119
    R_bias = global_R_bias
    DL_bias = global_DL_bias
    pf = torch.zeros(len(time_years), dtype=torch.float32, device=device)
    beta = torch.zeros(len(time_years), dtype=torch.float32, device=device)
    first_maintenance_time = maintenance_times[0] if maintenance_times else float('inf')
    for i in range(len(time_years)):
        if time_years[i] >= first_maintenance_time:
            g = moment_capacity_total[:, i] * R_bias.squeeze() - moment_DW * DL_bias.squeeze()
        else:
            g = moment_capacity_steel[:, i] * R_bias.squeeze() - moment_DW * DL_bias.squeeze()
        pf[i] = torch.sum(g < 0).float() / Nsample
        if pf[i] == 0:
            beta[i] = torch.tensor(4.4172, dtype=torch.float32, device=device)
        else:
            beta[i] = torch.tensor(norm.ppf(1 - pf[i].cpu().numpy()), dtype=torch.float32, device=device)
    end_time = time.time()
    run_time = end_time - start_time
    return {'beta': beta.cpu().numpy(), 'pf': pf.cpu().numpy(), 'corrosion_depth': torch.mean(d_sp, dim=0).cpu().numpy(), 'moment_capacity_steel': moment_capacity_steel.cpu().numpy(), 'moment_capacity_plate': moment_capacity_plate.cpu().numpy(), 'moment_capacity_total': moment_capacity_total.cpu().numpy(), 'run_time': run_time, 'time_years': time_years}

def set_device(use_cuda=True):
    global device
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda:1')
        pass
        memory_allocated = torch.cuda.memory_allocated(1)
        memory_total = torch.cuda.get_device_properties(1).total_memory
        memory_used_ratio = memory_allocated / memory_total
        pass
    else:
        device = torch.device('cpu')
        pass
    init_global_matrices(use_cuda)
    return device
if __name__ == '__main__':
    set_device(use_cuda=True)
    example_maintenance_times = [91, 123, 166, 199]
    results = run_simulation(example_maintenance_times)
    pass
    pass
    pass
    pass
    pass
    time_years = results['time_years']
    pf = results['pf']
    beta = results['beta']
    d_sp = results['corrosion_depth']
    moment_capacity_steel = results['moment_capacity_steel']
    moment_capacity_plate = results['moment_capacity_plate']
    moment_capacity_total = results['moment_capacity_total']
    plt.figure()
    plt.plot(time_years, pf, linewidth=1.5)
    plt.xlabel('Time (years)')
    plt.ylabel('Probability of Failure')
    plt.title('Time-Varying Probability of Failure')
    plt.grid(True)
    plt.savefig('time_varying_pf_gangban.tiff', dpi=600)
    plt.figure()
    plt.plot(time_years, beta, linewidth=1.5)
    plt.xlabel('Time (years)')
    plt.ylabel('Reliability Index')
    plt.title('Time-Varying Reliability Index')
    plt.grid(True)
    plt.savefig('time_varying_beta_gangban.tiff', dpi=600)
    plt.figure()
    plt.plot(time_years, d_sp, linewidth=1.5)
    plt.xlabel('Time (years)')
    plt.ylabel('Corrosion Depth (m)')
    plt.title('Time-Varying Corrosion Depth')
    plt.grid(True)
    plt.savefig('time_varying_corrosion_depth.tiff', dpi=600)
    plt.figure()
    plt.plot(time_years, np.mean(moment_capacity_steel, axis=0), linewidth=1.5, label='Steel Moment Capacity')
    plt.plot(time_years, np.mean(moment_capacity_plate, axis=0), linewidth=1.5, label='Plate Moment Capacity')
    plt.plot(time_years, np.mean(moment_capacity_total, axis=0), linewidth=1.5, label='Total Moment Capacity')
    plt.xlabel('Time (years)')
    plt.ylabel('Moment Capacity (kNm)')
    plt.title('Moment Capacity Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('moment_capacity_over_time.tiff', dpi=600)
