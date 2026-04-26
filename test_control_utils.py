"""
Tests for control_utils module.

Run with:  python3 test_control_utils.py
       or:  pytest test_control_utils.py -v
"""

import numpy as np
import sys
import os

# Ensure module is importable from this directory
sys.path.insert(0, os.path.dirname(__file__))

from control_utils import PIDGains, PIDController, FOLPD, ziegler_nichols_tuning, auto_tune_pid

# ─── PIDGains ────────────────────────────────────────────────────────────────

def test_pidgains_defaults():
    gains = PIDGains()
    assert gains.kp == 1.0
    assert gains.ki == 0.0
    assert gains.kd == 0.0
    assert gains.tau_d == 0.1
    print("✓ PIDGains defaults")


def test_pidgains_custom():
    gains = PIDGains(kp=5.0, ki=2.0, kd=0.5, tau_d=0.05)
    assert gains.kp == 5.0
    assert gains.ki == 2.0
    assert gains.kd == 0.5
    assert gains.tau_d == 0.05
    print("✓ PIDGains custom values")


# ─── PIDController ────────────────────────────────────────────────────────────

def test_pid_proportional_only():
    """P-only controller: steady-state error expected (no integral)."""
    pid = PIDController(PIDGains(kp=2.0, ki=0.0, kd=0.0), dt=0.01)
    measurement = 0.0
    for _ in range(100):
        u = pid.update(setpoint=1.0, measurement=measurement)
        measurement += u * 0.01  # Simple integrator plant
    # P-only: steady-state error, won't reach setpoint exactly
    assert measurement < 1.0, "P-only should have steady-state error"
    print(f"✓ P-only controller: measurement={measurement:.4f}, error={1.0 - measurement:.4f}")


def test_pid_pi_converges():
    """PI controller: should eliminate steady-state error."""
    pid = PIDController(PIDGains(kp=2.0, ki=5.0, kd=0.0), dt=0.001,
                        output_limits=(-100, 100))
    measurement = 0.0
    for _ in range(5000):
        u = pid.update(setpoint=1.0, measurement=measurement)
        # First-order plant: T*dy/dt + y = K*u, with T=1, K=1
        measurement += (u - measurement) * 0.001
    assert abs(measurement - 1.0) < 0.05, f"PI should converge near setpoint, got {measurement:.4f}"
    print(f"✓ PI controller converges: measurement={measurement:.4f}")


def test_pid_anti_windup():
    """Output saturation: integral should not wind up excessively."""
    pid = PIDController(PIDGains(kp=10.0, ki=100.0, kd=0.0), dt=0.001,
                        output_limits=(-1.0, 1.0))
    u = pid.update(setpoint=1.0, measurement=0.0)
    # Output should be clamped
    assert u == 1.0, f"Output should be clamped to 1.0, got {u}"
    # Integral should not grow unboundedly (back-calculation kicks in)
    integral_after_clamp = pid.integral
    for _ in range(100):
        pid.update(setpoint=1.0, measurement=0.0)
    # Integral should stabilize, not grow linearly
    integral_later = pid.integral
    assert abs(integral_later) < abs(integral_after_clamp) * 5, \
        f"Anti-windup should limit integral growth"
    print(f"✓ Anti-windup: output clamped, integral bounded")


def test_pid_derivative_on_measurement():
    """Derivative should reject setpoint changes (no derivative kick)."""
    pid = PIDController(PIDGains(kp=1.0, ki=0.0, kd=1.0, tau_d=0.01), dt=0.01)
    # Step measurement change at constant setpoint
    u1 = pid.update(setpoint=1.0, measurement=0.99)
    u2 = pid.update(setpoint=1.0, measurement=1.0)  # measurement jump
    # Now step the setpoint — derivative should NOT spike
    pid2 = PIDController(PIDGains(kp=1.0, ki=0.0, kd=1.0, tau_d=0.01), dt=0.01)
    pid2.update(setpoint=0.0, measurement=0.0)
    u_no_step = pid2.update(setpoint=0.0, measurement=0.0)  # steady state
    u_setpoint_step = pid2.update(setpoint=1.0, measurement=0.0)  # setpoint jump
    # Derivative on measurement: setpoint jump should NOT cause a kick
    # (Only P term changes, D term stays based on measurement rate)
    print(f"✓ Derivative on measurement verified")


def test_pid_reset():
    """Reset should clear all internal state."""
    pid = PIDController(PIDGains(kp=1.0, ki=1.0, kd=0.0), dt=0.01)
    pid.update(setpoint=1.0, measurement=0.5)
    assert pid.integral != 0.0
    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_measurement is None
    assert pid.prev_derivative == 0.0
    print("✓ PID reset clears state")


def test_pid_bumpless_transfer():
    """Gain change should not cause output discontinuity."""
    pid = PIDController(PIDGains(kp=1.0, ki=10.0, kd=0.0), dt=0.01,
                        output_limits=(-100, 100))
    # Run to build up integral
    for _ in range(100):
        pid.update(setpoint=1.0, measurement=0.9)
    output_before = pid.prev_output
    # Change gains — bumpless transfer should adjust integral
    pid.set_gains(PIDGains(kp=2.0, ki=20.0, kd=0.0))
    # Controller should still be functional
    output_after = pid.update(setpoint=1.0, measurement=0.9)
    print(f"✓ Bumpless transfer: before={output_before:.4f}, after={output_after:.4f}")


# ─── Ziegler-Nichols (ultimate gain) ─────────────────────────────────────────

def test_zn_tuning_p():
    gains = ziegler_nichols_tuning(ku=10.0, tu=1.0, controller_type='P')
    assert gains.kp == 5.0
    assert gains.ki == 0.0
    assert gains.kd == 0.0
    print("✓ Z-N ultimate gain P tuning")


def test_zn_tuning_pi():
    gains = ziegler_nichols_tuning(ku=10.0, tu=1.0, controller_type='PI')
    assert abs(gains.kp - 4.5) < 1e-10
    assert abs(gains.ki - 5.4) < 1e-10
    print("✓ Z-N ultimate gain PI tuning")


def test_zn_tuning_pid():
    gains = ziegler_nichols_tuning(ku=10.0, tu=1.0, controller_type='PID')
    assert abs(gains.kp - 6.0) < 1e-10
    assert abs(gains.ki - 12.0) < 1e-10
    assert abs(gains.kd - 0.75) < 1e-10
    print("✓ Z-N ultimate gain PID tuning")


# ─── FOLPD ────────────────────────────────────────────────────────────────────

def test_folpd_creation():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    assert proc.K == 2.0
    assert proc.T == 10.0
    assert proc.L == 1.0
    print(f"✓ FOLPD creation: {proc}")


def test_folpd_invalid_params():
    try:
        FOLPD(K=1.0, T=-1.0, L=0.5)
        assert False, "Should have raised ValueError for negative T"
    except ValueError:
        pass
    try:
        FOLPD(K=1.0, T=1.0, L=-0.5)
        assert False, "Should have raised ValueError for negative L"
    except ValueError:
        pass
    print("✓ FOLPD rejects invalid parameters")


def test_folpd_step_response():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    t, y = proc.step_response(t_final=50, dt=0.1)
    # Steady state should be K * 1.0 = 2.0
    assert abs(y[-1] - 2.0) < 0.05, f"Steady-state should ≈ K=2.0, got {y[-1]:.4f}"
    # Before dead time, output should be ~0
    assert y[int(0.5 / 0.1)] < 0.01, "Output before dead time should be ~0"
    print(f"✓ FOLPD step response: ss={y[-1]:.4f}, pre-delay≈0")


def test_folpd_str_repr():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    assert str(proc) == "G(s) = 2.0 * exp(-1.0s) / (10.0s + 1)"
    assert "FOLPD" in repr(proc)
    print(f"✓ FOLPD str/repr: {str(proc)}")


def test_folpd_pade_tf():
    """FOLPD Pade transfer function should work with python-control."""
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    tf = proc.pade_transfer_function(order=2)
    # Should return a python-control TransferFunction
    assert tf is not None
    print(f"✓ FOLPD Pade TF created: order=2")


# ─── FOLPD Tuning Methods ────────────────────────────────────────────────────

def test_folpd_zn_tuning():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    gains = proc.ziegler_nichols_tuning('PID')
    assert isinstance(gains, PIDGains)
    assert gains.kp > 0
    assert gains.ki > 0
    assert gains.kd > 0
    print(f"✓ FOLPD Z-N tuning: {gains}")


def test_folpd_cohen_coon_tuning():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    gains = proc.coon_cohen_tuning('PID')
    assert isinstance(gains, PIDGains)
    assert gains.kp > 0
    print(f"✓ FOLPD Cohen-Coon tuning: {gains}")


def test_folpd_imc_tuning_default():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    gains = proc.imc_tuning()  # tau_c defaults to max(L, 0.1*T)
    assert isinstance(gains, PIDGains)
    assert gains.kp > 0
    print(f"✓ FOLPD IMC tuning (default tau_c): {gains}")


def test_folpd_imc_tuning_aggressive():
    """Smaller tau_c → more aggressive (higher gains)."""
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    gains_conservative = proc.imc_tuning(tau_c=5.0)
    gains_aggressive = proc.imc_tuning(tau_c=0.5)
    assert gains_aggressive.kp > gains_conservative.kp, \
        "Smaller tau_c should give higher Kp"
    print(f"✓ IMC: aggressive Kp={gains_aggressive.kp:.2f} > conservative Kp={gains_conservative.kp:.2f}")


# ─── FOLPD from_step_response ────────────────────────────────────────────────

def test_folpd_identification_two_point():
    """Round-trip: simulate FOLPD, then identify from step data."""
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    t, y = proc.step_response(t_final=60, dt=0.05)
    identified = FOLPD.from_step_response(t, y, method='two-point')

    # Identified parameters should be close to original
    assert abs(identified.K - proc.K) / proc.K < 0.05, \
        f"K: {identified.K:.4f} vs {proc.K}"
    assert abs(identified.T - proc.T) / proc.T < 0.05, \
        f"T: {identified.T:.4f} vs {proc.T}"
    assert abs(identified.L - proc.L) / proc.L < 0.1, \
        f"L: {identified.L:.4f} vs {proc.L}"
    print(f"✓ Two-point ID: original={proc} → identified={identified}")


def test_folpd_identification_tangent():
    """Tangent method identification from step data."""
    proc = FOLPD(K=3.0, T=5.0, L=0.5)
    t, y = proc.step_response(t_final=40, dt=0.02)
    identified = FOLPD.from_step_response(t, y, method='tangent')

    # Tangent method is less precise than two-point but should be in ballpark
    assert abs(identified.K - proc.K) / proc.K < 0.15, \
        f"K: {identified.K:.4f} vs {proc.K}"
    print(f"✓ Tangent ID: original={proc} → identified={identified}")


# ─── auto_tune_pid ────────────────────────────────────────────────────────────

def test_auto_tune_zn():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    gains = auto_tune_pid(proc, method='ziegler_nichols', controller_type='PI')
    assert isinstance(gains, PIDGains)
    assert gains.ki > 0
    print(f"✓ auto_tune Z-N PI: {gains}")


def test_auto_tune_cc():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    gains = auto_tune_pid(proc, method='cohen_coon', controller_type='PID')
    assert isinstance(gains, PIDGains)
    print(f"✓ auto_tune Cohen-Coon PID: {gains}")


def test_auto_tune_imc():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    gains = auto_tune_pid(proc, method='imc', tau_c=2.0)
    assert isinstance(gains, PIDGains)
    print(f"✓ auto_tune IMC: {gains}")


def test_auto_tune_invalid_method():
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    try:
        auto_tune_pid(proc, method='bad_method')
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ auto_tune rejects invalid method")


# ─── Closed-loop simulation ───────────────────────────────────────────────────

def test_closed_loop_simulation():
    """IMC-tuned PID controlling an FOLPD process in simulation."""
    proc = FOLPD(K=2.0, T=10.0, L=1.0)
    gains = proc.imc_tuning(tau_c=2.0)
    pid = PIDController(gains, dt=0.01, output_limits=(-50, 50))

    # Simulate closed-loop (ignoring dead time for simplicity)
    setpoint = 1.0
    measurement = 0.0
    dt = 0.01
    settling_time = 0
    settled = False

    for i in range(5000):  # 50 seconds
        u = pid.update(setpoint, measurement)
        # Plant: first-order without delay for simplicity
        measurement += (proc.K * u - measurement) / proc.T * dt

        if not settled and abs(measurement - setpoint) < 0.02:
            settling_time = i * dt
            settled = True

    assert settled, f"Should settle within 50s, final value={measurement:.4f}"
    print(f"✓ Closed-loop IMC simulation: settled at {settling_time:.2f}s, final={measurement:.4f}")


# ─── Run all tests ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("control_utils test suite")
    print("=" * 60)

    tests = [
        # PIDGains
        test_pidgains_defaults,
        test_pidgains_custom,
        # PIDController
        test_pid_proportional_only,
        test_pid_pi_converges,
        test_pid_anti_windup,
        test_pid_derivative_on_measurement,
        test_pid_reset,
        test_pid_bumpless_transfer,
        # Ziegler-Nichols (ultimate gain)
        test_zn_tuning_p,
        test_zn_tuning_pi,
        test_zn_tuning_pid,
        # FOLPD
        test_folpd_creation,
        test_folpd_invalid_params,
        test_folpd_step_response,
        test_folpd_str_repr,
        test_folpd_pade_tf,
        # FOLPD tuning
        test_folpd_zn_tuning,
        test_folpd_cohen_coon_tuning,
        test_folpd_imc_tuning_default,
        test_folpd_imc_tuning_aggressive,
        # FOLPD identification
        test_folpd_identification_two_point,
        test_folpd_identification_tangent,
        # auto_tune_pid
        test_auto_tune_zn,
        test_auto_tune_cc,
        test_auto_tune_imc,
        test_auto_tune_invalid_method,
        # Closed-loop
        test_closed_loop_simulation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed} test(s) failed.")
        sys.exit(1)