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
gains_freq_zn = PIDGains(
    kp=0.6 * Ku,
    ki=1.2 * Ku / Tu,
    kd=0.075 * Ku * Tu
)
print(f"\n--- Relay ZN frequency ---")
print(f"  Kp = {gains_freq_zn.kp:.4f}")
print(f"  Ki = {gains_freq_zn.ki:.4f}")
print(f"  Kd = {gains_freq_zn.kd:.4f}")

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

# ---------------------------------------------------------------------------
# 7. Compute actual overshoot & settling time
# ---------------------------------------------------------------------------
def metrics(y, t, y_sp=1.0, tol=0.02):
    """Return overshoot (%), peak time, settling time for step response."""
    y_final = np.mean(y[-500:])  # last samples
    if y_final < 0.1:
        return 0.0, 0.0, 0.0

    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]
    peak_time = t[peak_idx]

    overshoot = max(0.0, (peak_val - y_final) / y_final * 100)

    # 2% settling time
    within = np.where(np.abs(y - y_final) <= tol * np.abs(y_final))[0]
    settling_time = t[within[-1]] if len(within) else t[-1]
    # Find first time after which it stays within
    for i in range(len(y) - 1, -1, -1):
        if np.abs(y[i] - y_final) > tol * np.abs(y_final):
            settling_time = t[min(i + 1, len(t) - 1)]
            break

    return overshoot, peak_time, settling_time

os_ol, pt_ol, st_ol = metrics(y_ol_zn, t_cl)
os_freq, pt_freq, st_freq = metrics(y_freq_zn, t_cl)

print(f"\n--- Performance metrics ---")
print(f"  Open-loop ZN   : OS={os_ol:.1f}%, peak@t={pt_ol:.1f}s, settle≈{st_ol:.1f}s")
print(f"  Relay ZN freq  : OS={os_freq:.1f}%, peak@t={pt_freq:.1f}s, settle≈{st_freq:.1f}s")

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
# 9. PLOT 2 — Comparative closed-loop step responses
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                         gridspec_kw={'hspace': 0.08})

# Output comparison
t_plot_cl = t_cl[:int(80 / dt)]
y_plot_ol = y_ol_zn[:len(t_plot_cl)]
y_plot_freq = y_freq_zn[:len(t_plot_cl)]

axes[0].axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
axes[0].plot(t_plot_cl, y_plot_ol, 'C0-', lw=1.4,
             label=f'Open-loop ZN  (OS≈{os_ol:.0f}%)')
axes[0].plot(t_plot_cl, y_plot_freq, 'C1-', lw=1.4,
             label=f'Relay ZN freq  (OS≈{os_freq:.0f}%)')
axes[0].fill_between(t_plot_cl, y_plot_ol, alpha=0.08, color='C0')
axes[0].fill_between(t_plot_cl, y_plot_freq, alpha=0.08, color='C1')
axes[0].set_ylabel('Output y(t)')
axes[0].set_title('Closed-Loop Step Response Comparison')
axes[0].legend(loc='upper right', ncol=2)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, max(2.0, np.max(y_plot_ol) * 1.1)])

# Control signal comparison
u_plot_ol = u_ol_zn[:len(t_plot_cl)]
u_plot_freq = u_freq_zn[:len(t_plot_cl)]

axes[1].plot(t_plot_cl, u_plot_ol, 'C0-', lw=1.0, alpha=0.8)
axes[1].plot(t_plot_cl, u_plot_freq, 'C1-', lw=1.0, alpha=0.8)
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
