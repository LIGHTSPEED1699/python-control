"""
Generate plots for "PID Tuning with Relay Feedback" tutorial.

Plant:  G(s) = 20 * exp(-s) / (10*s + 1)
        K = 20,  T = 10 s,  L = 1 s

Uses python-control (ct.nlsys) for relay feedback experiment,
then PIDController from control_utils.py for discrete-time
closed-loop validation.

Run with:
    python generate_relay_feedback_plots.py

Output files (written to ../../website/public/images/tutorials/):
    - relay-feedback-waveform.png       (0–60 s, detail view)
    - relay-feedback-closed-loop.png    (comparison: open-loop ZN vs relay ZN)
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import control as ct
import sys
import os

# ---------------------------------------------------------------------------
# 1. Add control_utils.py to Python path
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)  # ../python-control/
sys.path.insert(0, _PARENT_DIR)
from control_utils import PIDController, PIDGains, FOLPD

WEBSITE_IMG_DIR = '/home/hongbin/Documents/opencode-projects/hongbinli-website/public/images/tutorials'
assert os.path.isdir(WEBSITE_IMG_DIR), f"Directory not found: {WEBSITE_IMG_DIR}"
# ---------------------------------------------------------------------------
K, T, L = 20.0, 10.0, 1.0

G0 = ct.tf([K], [T, 1], name='G0')
[num_delay, den_delay] = ct.pade(L, 3)
G_delay = ct.tf(num_delay, den_delay)
G = ct.series(G0, G_delay, name='G', inputs=['u'], outputs=['y'])

# ---------------------------------------------------------------------------
# 3. Relay feedback experiment
# ---------------------------------------------------------------------------
d = 5.0  # relay amplitude


def relay_state(t, xc, uc, p):
    return [0]


def relay_output(t, xc, uc, p):
    e = uc[0]
    return [np.sign(e) * p['b']]


relay_ctrl = ct.nlsys(
    relay_state, relay_output,
    inputs=('e'), outputs=('u'),
    states=0, params={'b': d},
    name='relay', dt=0
)

sum_junc = ct.summing_junction(inputs=['r', '-y'], output='e')

T_ry = ct.interconnect(
    [G, relay_ctrl, sum_junc],
    inplist='r', outlist='y'
)
T_ru = ct.interconnect(
    [G, relay_ctrl, sum_junc],
    inplist='r', outlist='u'
)

T_sim = 200
dt = 0.01
t = np.arange(0, T_sim, dt)
r_vec = np.ones_like(t)

print("Running relay feedback simulation …")
resp_y = ct.input_output_response(T_ry, t, r_vec)
resp_u = ct.input_output_response(T_ru, t, r_vec)

# ---------------------------------------------------------------------------
# 4. Measure limit cycle (discard first 120 s)
# ---------------------------------------------------------------------------
from scipy.signal import find_peaks

idx_start = int(120 / dt)  # t ≥ 120 s
y_ss = np.asarray(resp_y.outputs[idx_start:]).flatten()
t_ss = np.asarray(resp_y.time[idx_start:]).flatten()

mean_y = float(np.mean(y_ss))

# Find extrema
peaks, _ = find_peaks(y_ss, height=mean_y, distance=int(2.0 / dt))
valleys, _ = find_peaks(-y_ss, height=-mean_y, distance=int(2.0 / dt))

if len(peaks) < 2 or len(valleys) < 2:
    raise RuntimeError('Not enough oscillation cycles found.')

a = (np.mean(y_ss[peaks]) - np.mean(y_ss[valleys])) / 2.0
periods = [t_ss[peaks[i + 1]] - t_ss[peaks[i]] for i in range(len(peaks) - 1)]
Tu = float(np.mean(periods))

Ku = 4.0 * d / (np.pi * a)

# Verify against linear margin()
gm, pm, wg, wp = ct.margin(G)
print(f"\n--- Relay experiment results ---")
print(f"  Measured amplitude a  = {a:.4f}")
print(f"  Measured period  Tu   = {Tu:.4f} s")
print(f"  Ultimate gain    Ku   = {Ku:.4f}")
print(f"  (True gm={gm:.4f}, true Tu={2*np.pi/wg:.4f})")

# ---------------------------------------------------------------------------
# 5. Compute PID gains for both methods
# ---------------------------------------------------------------------------

# --- Open-loop Ziegler-Nichols (reaction curve) ---
plant = FOLPD(K=K, T=T, L=L)
gains_ol_zn = plant.ziegler_nichols_tuning('PID')
print(f"\n--- Open-loop ZN (K={K}, T={T}, L={L}) ---")
print(f"  Kp = {gains_ol_zn.kp:.4f}")
print(f"  Ki = {gains_ol_zn.ki:.4f}")
print(f"  Kd = {gains_ol_zn.kd:.4f}")

# --- Ziegler-Nichols frequency domain (from relay) ---
# Classic PID

gains_freq_zn = PIDGains(
    kp=0.6 * Ku,
    ki=1.2 * Ku / Tu,
    kd=0.075 * Ku * Tu
)
print(f"\n--- Relay ZN Classic ---")
print(f"  Kp = {gains_freq_zn.kp:.4f}")
print(f"  Ki = {gains_freq_zn.ki:.4f}")
print(f"  Kd = {gains_freq_zn.kd:.4f}")

# Some overshoot
gains_some_os = PIDGains(
    kp=0.333 * Ku,
    ki=0.667 * Ku / Tu,
    kd=0.111 * Ku * Tu
)
print(f"\n--- Some overshoot ---")
print(f"  Kp = {gains_some_os.kp:.4f}")
print(f"  Ki = {gains_some_os.ki:.4f}")
print(f"  Kd = {gains_some_os.kd:.4f}")

# No overshoot
gains_no_os = PIDGains(
    kp=0.2 * Ku,
    ki=0.4 * Ku / Tu,
    kd=0.067 * Ku * Tu
)
print(f"\n--- No overshoot ---")
print(f"  Kp = {gains_no_os.kp:.4f}")
print(f"  Ki = {gains_no_os.ki:.4f}")
print(f"  Kd = {gains_no_os.kd:.4f}")

# Also compute Cohen-Coon for comparison
gains_cc = plant.coon_cohen_tuning('PID')
print(f"\n--- Cohen-Coon ---")
print(f"  Kp = {gains_cc.kp:.4f}")
print(f"  Ki = {gains_cc.ki:.4f}")
print(f"  Kd = {gains_cc.kd:.4f}")

# ---------------------------------------------------------------------------
# 6. Discretised closed-loop step-response helper
# ---------------------------------------------------------------------------
def simulate_pid_closed_loop(plant_tf, pid_gains, t_vec, u_min=-1e6, u_max=1e6):
    """
    Closed-loop step response using PIDController (discrete time).
    Plant simulated step-by-step via state-space.
    Returns t, y, u arrays.
    """
    dt = float(t_vec[1] - t_vec[0])
    ctrl = PIDController(pid_gains, dt=dt, output_limits=(u_min, u_max))

    ss_plant = ct.ss(plant_tf)
    n_states = ss_plant.nstates

    # Flatten B / C for 1-D state vector math
    A = ss_plant.A
    B = np.asarray(ss_plant.B).flatten()
    C = np.asarray(ss_plant.C).flatten()
    D = float(np.asarray(ss_plant.D).flatten()[0])

    x = np.zeros(n_states)
    y_out = np.zeros_like(t_vec)
    u_out = np.zeros_like(t_vec)

    for i in range(len(t_vec)):
        # Output
        y_meas = float(C @ x + D * u_out[max(0, i - 1)])
        if i == 0:
            y_meas = 0.0
        y_out[i] = y_meas

        # Controller
        u = ctrl.update(setpoint=1.0, measurement=y_meas)
        u_out[i] = u

        # Plant update (Euler)
        dx = A @ x + B * u
        x = x + dx * dt

    return t_vec, y_out, u_out


print("\nSimulating closed-loop responses …")
t_cl = np.linspace(0, 120, 12000)

_, y_ol_zn, u_ol_zn = simulate_pid_closed_loop(
    G, gains_ol_zn, t_cl, u_min=-50, u_max=50
)
_, y_freq_zn, u_freq_zn = simulate_pid_closed_loop(
    G, gains_freq_zn, t_cl, u_min=-50, u_max=50
)
_, y_some_os, u_some_os = simulate_pid_closed_loop(
    G, gains_some_os, t_cl, u_min=-50, u_max=50
)
_, y_no_os, u_no_os = simulate_pid_closed_loop(
    G, gains_no_os, t_cl, u_min=-50, u_max=50
)
_, y_cc, u_cc = simulate_pid_closed_loop(
    G, gains_cc, t_cl, u_min=-50, u_max=50
)

# ---------------------------------------------------------------------------
# 7. Compute actual overshoot & settling time for ALL methods
# ---------------------------------------------------------------------------
def metrics(y, t, y_sp=1.0, tol=0.02):
    """Return overshoot (%), peak time, settling time, IAE for step response."""
    y_final = float(np.mean(y[-500:]))
    if y_final < 0.1 or np.isnan(y_final):
        return 0.0, 0.0, 0.0, 0.0

    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]
    peak_time = t[peak_idx]

    overshoot = max(0.0, (peak_val - y_final) / y_final * 100)
    
    # IAE (Integrated Absolute Error)
    iae = float(np.trapezoid(np.abs(y - y_final), t))

    # Settling time: first time after which signal stays within ±tol of final
    settling_time = t[-1]
    for i in range(len(y) - 1, -1, -1):
        if np.abs(y[i] - y_final) > tol * np.abs(y_final):
            settling_time = t[min(i + 1, len(t) - 1)]
            break

    return overshoot, peak_time, settling_time, iae

results = {
    'Open-loop ZN':        metrics(y_ol_zn, t_cl),
    'Relay ZN Classic':    metrics(y_freq_zn, t_cl),
    'Some overshoot':      metrics(y_some_os, t_cl),
    'No overshoot':        metrics(y_no_os, t_cl),
    'Cohen-Coon':          metrics(y_cc, t_cl),
}

print(f"\n{'Method':<20} {'OS (%)':>8} {'Peak(s)':>9} {'Settle(s)':>10} {'IAE':>10}")
print("-" * 60)
for name, (os_val, pt, st, iae) in results.items():
    print(f"{name:<20} {os_val:>8.1f} {pt:>9.1f} {st:>10.1f} {iae:>10.1f}")

# ---------------------------------------------------------------------------
# 8. PLOT 1 — Relay waveform (0–60 s, zoomed detail)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True,
                         gridspec_kw={'hspace': 0.08})

t_start = int(0 / dt)
t_end = int(60 / dt)

t_plot = resp_y.time[t_start:t_end]
y_plot = resp_y.outputs[t_start:t_end]
u_plot = resp_u.outputs[t_start:t_end]
r_plot = r_vec[t_start:t_end]

axes[0].plot(t_plot, r_plot, 'k--', lw=1.0, label='r(t) = 1')
axes[0].set_ylabel('Setpoint r(t)')
axes[0].set_title(f'Relay Feedback Experiment (K={K}, T={T}s, L={L}s, d={d})')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1.5])

axes[1].plot(t_plot, y_plot, 'b-', lw=1.2, label='y(t)', alpha=0.9)
# Mark peaks/valleys that fall inside plot window
for p in peaks:
    actual_idx = p + idx_start
    if t_start <= actual_idx < t_end:
        axes[1].plot(resp_y.time[actual_idx], resp_y.outputs[actual_idx],
                     'ro', markersize=5, zorder=5)
for v in valleys:
    actual_idx = v + idx_start
    if t_start <= actual_idx < t_end:
        axes[1].plot(resp_y.time[actual_idx], resp_y.outputs[actual_idx],
                     'go', markersize=5, zorder=5)
axes[1].set_ylabel('Output y(t)')
axes[1].text(0.98, 0.95, f'a = {a:.2f},  Tu = {Tu:.2f} s',
             transform=axes[1].transAxes, ha='right', va='top',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1].grid(True, alpha=0.3)

axes[2].step(t_plot, u_plot, 'r-', lw=1.0, where='post', label='u(t)')
axes[2].set_ylabel('Relay u(t)')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper right')

plt.savefig(
    os.path.join(WEBSITE_IMG_DIR, 'relay-feedback-waveform.png'),
    dpi=150, bbox_inches='tight'
)
print(f"\nSaved: relay-feedback-waveform.png")

# ---------------------------------------------------------------------------
# 9. PLOT 2 — Comprehensive closed-loop step response comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                         gridspec_kw={'hspace': 0.08})

# Extract all y-values truncated to 80 s
t_plot_cl = t_cl[:int(80 / float(t_cl[1] - t_cl[0]))]
len_plot = len(t_plot_cl)

# Unpack metrics
os_ol, pt_ol, st_ol, iae_ol = results['Open-loop ZN']
os_cl, pt_cl, st_cl, iae_cl = results['Relay ZN Classic']
os_some, pt_some, st_some, iae_some = results['Some overshoot']
os_no, pt_no, st_no, iae_no = results['No overshoot']
os_cc, pt_cc, st_cc, iae_cc = results['Cohen-Coon']

# Data dicts for plotting
methods_data = [
    ('Open-loop ZN',     y_ol_zn[:len_plot],     u_ol_zn[:len_plot],     os_ol,  'C0'),
    ('Cohen-Coon',        y_cc[:len_plot],        u_cc[:len_plot],        os_cc,  'C2'),
    ('Relay ZN Classic',  y_freq_zn[:len_plot],   u_freq_zn[:len_plot],   os_cl,  'C1'),
    ('Some overshoot',    y_some_os[:len_plot],   u_some_os[:len_plot],   os_some, 'C3'),
    ('No overshoot',      y_no_os[:len_plot],     u_no_os[:len_plot],     os_no,  'C4'),
]

axes[0].axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
for name, y_plt, _, os_val, color in methods_data:
    label = f'{name}  (OS≈{os_val:.0f}%)'
    axes[0].plot(t_plot_cl, y_plt, color=color, lw=1.4, label=label, alpha=0.9)
    axes[0].fill_between(t_plot_cl, y_plt, alpha=0.05, color=color)
axes[0].set_ylabel('Output y(t)')
axes[0].set_title('Closed-Loop Step Response : All ZN-Based Methods Compared')
axes[0].legend(loc='upper right', ncol=2, fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 2.2])

for name, _, u_plt, _, color in methods_data:
    axes[1].plot(t_plot_cl, u_plt, color=color, lw=1.0, alpha=0.8)
axes[1].set_ylabel('Control u(t)')
axes[1].set_xlabel('Time (s)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 80])

plt.savefig(
    os.path.join(WEBSITE_IMG_DIR, 'relay-feedback-closed-loop.png'),
    dpi=150, bbox_inches='tight'
)
print(f"Saved: relay-feedback-closed-loop.png")
print("\nDone.")
