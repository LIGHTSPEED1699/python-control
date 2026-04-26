"""
Control systems utilities for python-control notebooks.

Provides:
  - PIDGains: PID gain dataclass
  - PIDController: PID with anti-windup, derivative filter, bumpless transfer
  - FOLPD: First-Order Lag Plus Dead-time process model
  - ziegler_nichols_tuning: Ziegler-Nichols tuning helper
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ─── PID Controller ──────────────────────────────────────────────────────────

@dataclass
class PIDGains:
    """PID gains with optional derivative filter."""
    kp: float = 1.0      # Proportional gain
    ki: float = 0.0      # Integral gain
    kd: float = 0.0      # Derivative gain
    tau_d: float = 0.1   # Derivative filter time constant


class PIDController:
    """PID controller with anti-windup and derivative filtering.

    Features:
    - Integral anti-windup (clamping and back-calculation)
    - Derivative on measurement (avoids derivative kick)
    - Low-pass filter on derivative term
    - Bumpless transfer for gain changes
    """

    def __init__(self, gains: PIDGains, dt: float,
                 output_limits: Tuple[float, float] = (-np.inf, np.inf)):
        self.gains = gains
        self.dt = dt
        self.output_min, self.output_max = output_limits

        # State
        self.integral = 0.0
        self.prev_measurement = None
        self.prev_derivative = 0.0
        self.prev_output = 0.0

    def update(self, setpoint: float, measurement: float) -> float:
        """Compute control output.

        Args:
            setpoint: Desired value
            measurement: Current measured value

        Returns:
            Control output (clamped to limits)
        """
        error = setpoint - measurement

        # Proportional term
        p_term = self.gains.kp * error

        # Integral term with clamping anti-windup
        self.integral += error * self.dt
        i_term = self.gains.ki * self.integral

        # Derivative term on measurement (not error)
        # Avoids derivative kick on setpoint changes
        if self.prev_measurement is None:
            d_term = 0.0
        else:
            # Raw derivative
            d_raw = -(measurement - self.prev_measurement) / self.dt

            # Low-pass filter on derivative
            alpha = self.dt / (self.gains.tau_d + self.dt)
            d_filtered = alpha * d_raw + (1 - alpha) * self.prev_derivative
            self.prev_derivative = d_filtered

            d_term = self.gains.kd * d_filtered

        self.prev_measurement = measurement

        # Compute output
        output_unsat = p_term + i_term + d_term

        # Clamp output
        output = np.clip(output_unsat, self.output_min, self.output_max)

        # Back-calculation anti-windup
        if self.gains.ki != 0:
            saturation_error = output - output_unsat
            self.integral += saturation_error / self.gains.ki

        self.prev_output = output
        return output

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_measurement = None
        self.prev_derivative = 0.0

    def set_gains(self, gains: PIDGains):
        """Update gains with bumpless transfer."""
        if self.gains.ki != 0 and gains.ki != 0:
            self.integral *= self.gains.ki / gains.ki
        self.gains = gains


def ziegler_nichols_tuning(ku: float, tu: float,
                            controller_type: str = 'PID') -> PIDGains:
    """Compute PID gains using Ziegler-Nichols method.

    Args:
        ku: Ultimate gain (gain at which oscillation occurs)
        tu: Ultimate period (period of oscillation)
        controller_type: 'P', 'PI', or 'PID'

    Returns:
        Tuned PID gains
    """
    if controller_type == 'P':
        return PIDGains(kp=0.5 * ku)
    elif controller_type == 'PI':
        return PIDGains(kp=0.45 * ku, ki=0.54 * ku / tu)
    else:  # PID
        return PIDGains(
            kp=0.6 * ku,
            ki=1.2 * ku / tu,
            kd=0.075 * ku * tu
        )


# ─── FOLPD (First-Order Lag Plus Dead-time) ──────────────────────────────────

class FOLPD:
    """First-Order Lag Plus Dead-time process model.

    Transfer function:  G(s) = K * exp(-L*s) / (T*s + 1)

    Parameters:
        K: Process gain (steady-state change per unit input)
        T: Time constant (lag)
        L: Dead time (transport delay)

    This is the standard model used in process control for
    PID tuning (Ziegler-Nichols, Cohen-Coon, Lambda, IMC, etc.).

    Can also be constructed from step response data via:
        FOLPD.from_step_response()
    """

    def __init__(self, K: float, T: float, L: float):
        if T <= 0:
            raise ValueError(f"Time constant T must be positive, got {T}")
        if L < 0:
            raise ValueError(f"Dead time L must be non-negative, got {L}")
        self.K = K
        self.T = T
        self.L = L

    @property
    def transfer_function(self):
        """Return python-control TransferFunction object (no delay).

        Note: python-control approximates delay via Pade or treats it separately.
        Use self.pade_transfer_function() for a rational approximation.
        """
        try:
            import control as ct
            return ct.tf([self.K], [self.T, 1])
        except ImportError:
            raise ImportError("python-control library required. Install with: pip install control")

    def pade_transfer_function(self, order: int = 1):
        """Return TransferFunction with Pade approximation of dead time.

        Args:
            order: Pade approximation order (1 or 2 typical)

        Returns:
            python-control TransferFunction including delay approximation
        """
        import control as ct
        num_delay, den_delay = ct.pade(self.L, order)
        delay_tf = ct.tf(num_delay, den_delay)
        plant_tf = ct.tf([self.K], [self.T, 1])
        return ct.series(plant_tf, delay_tf)

    def step_response(self, t=None, dt=0.01, t_final=None):
        """Compute step response of the FOLPD process.

        Args:
            t: Time vector (optional, auto-generated if None)
            dt: Time step for auto-generated vector
            t_final: Final time (defaults to 5*T + 5*L)

        Returns:
            t, y arrays
        """
        if t is None:
            if t_final is None:
                t_final = 5 * self.T + 5 * self.L + 1
            t = np.arange(0, t_final, dt)

        # Time-shifted first-order response
        y = np.where(t >= self.L,
                     self.K * (1 - np.exp(-(t - self.L) / self.T)),
                     0.0)
        return t, y

    def ziegler_nichols_tuning(self, controller_type: str = 'PID') -> PIDGains:
        """Ziegler-Nichols tuning from FOLPD parameters (reaction curve method).

        Uses the process reaction curve parameters K, L, T to compute
        PID gains. This is the Ziegler-Nichols "process reaction curve" method
        (also called the open-loop method), distinct from the ultimate gain method.

        Args:
            controller_type: 'P', 'PI', or 'PID'

        Returns:
            PIDGains with computed parameters
        """
        R = self.K / self.T  # Reaction rate (slope of tangent at inflection)
        L = self.L

        if controller_type == 'P':
            return PIDGains(kp=self.T / (self.K * L))
        elif controller_type == 'PI':
            return PIDGains(kp=0.9 * self.T / (self.K * L),
                            ki=0.9 * self.T / (self.K * L) / (3.33 * L))
        else:  # PID
            kp = 1.2 * self.T / (self.K * L)
            ki = 1.2 * self.T / (self.K * L) / (2 * L)
            kd = 1.2 * self.T / (self.K * L) * (0.5 * L)
            return PIDGains(kp=kp, ki=ki, kd=kd)

    def coon_cohen_tuning(self, controller_type: str = 'PID') -> PIDGains:
        """Cohen-Coon tuning from FOLPD parameters.

        Better than Z-N for processes with significant dead time.
        Reduces the proportional band and eliminates steady-state error.

        Args:
            controller_type: 'P', 'PI', or 'PID'

        Returns:
            PIDGains with computed parameters
        """
        r = self.L / self.T  # Controllability ratio
        K = self.K

        if controller_type == 'P':
            return PIDGains(kp=(1 / K) * (r + 1 / 3))
        elif controller_type == 'PI':
            return PIDGains(kp=(1 / K) * (0.9 * r + 0.167),
                            ki=(1 / K) * (0.9 * r + 0.167) /
                            (3.33 * self.L + 0.333 * self.T))
        else:  # PID
            kp = (1 / K) * (1.33 * r + 0.167)
            ki = kp / (2.67 * self.L + 0.5 * self.T)
            kd = kp * (0.5 * self.L)
            return PIDGains(kp=kp, ki=ki, kd=kd)

    def imc_tuning(self, tau_c: Optional[float] = None) -> PIDGains:
        """Internal Model Control (IMC) tuning for FOLPD process.

        Provides good robustness and performance. The closed-loop time
        constant tau_c is the primary tuning knob:
          - Small tau_c → aggressive (fast response, less robust)
          - Large tau_c → conservative (slower, more robust)
          - Rule of thumb: tau_c >= L for robustness

        For FOLPD with filter: Kp = (T+L/2) / (K*tau_c),
                                Ki = Kp / (T+L/2),
                                Kd = T*L / (2*(T+L/2))

        Args:
            tau_c: Desired closed-loop time constant.
                   Defaults to max(L, 0.1*T) if not specified.

        Returns:
            PIDGains with IMC-tuned parameters
        """
        if tau_c is None:
            tau_c = max(self.L, 0.1 * self.T)

        Kp = (self.T + self.L / 2) / (self.K * tau_c)
        Ki = Kp / (self.T + self.L / 2)
        Kd = self.T * self.L / (2 * (self.T + self.L / 2))

        return PIDGains(kp=Kp, ki=Ki, kd=Kd)

    @staticmethod
    def from_step_response(t, y, method: str = 'two-point') -> 'FOLPD':
        """Identify FOLPD parameters from open-loop step response data.

        Args:
            t: Time array
            y: Step response output array
            method: Identification method
                - 'two-point': 63.2% and 28.3% points (Smith, 1972)
                - 'tangent': Tangent at inflection point (classic Z-N)

        Returns:
            FOLPD model with identified K, T, L
        """
        y_ss = y[-1]  # Steady-state value
        y0 = y[0]     # Initial value
        K = y_ss - y0  # Process gain

        if method == 'two-point':
            # Two-point method (63.2% and 28.3% of final value)
            t63_y = y0 + 0.632 * K
            t28_y = y0 + 0.283 * K

            # Find times at these amplitudes
            t63 = np.interp(t63_y, y, t)
            t28 = np.interp(t28_y, y, t)

            T = 1.5 * (t63 - t28)
            L = t63 - T

            # L can't be negative
            L = max(L, 0.0)

        elif method == 'tangent':
            # Tangent at inflection point
            # Find inflection (max second derivative)
            dy = np.gradient(y, t)
            d2y = np.gradient(dy, t)

            # Inflection point: where d2y is maximum (before settling)
            # Use first half of response
            mid = len(t) // 2
            idx_inflection = np.argmax(d2y[:mid])
            t_inflection = t[idx_inflection]
            y_inflection = y[idx_inflection]
            slope = dy[idx_inflection]

            # Project tangent line to t-axis and y_ss
            L = t_inflection - (y_inflection - y0) / slope
            T = K / slope

            L = max(L, 0.0)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'two-point' or 'tangent'.")

        return FOLPD(K=K, T=max(T, 1e-6), L=L)

    def __repr__(self):
        return f"FOLPD(K={self.K}, T={self.T}, L={self.L})"

    def __str__(self):
        return f"G(s) = {self.K} * exp(-{self.L}s) / ({self.T}s + 1)"


# ─── Convenience aliases ──────────────────────────────────────────────────────

# Common tuning rules mapped for easy access
TUNING_METHODS = {
    'ziegler_nichols': 'reaction_curve_zn',
    'cohen_coon': 'cohen_coon',
    'imc': 'imc',
}


def auto_tune_pid(process: FOLPD, method: str = 'imc',
                  controller_type: str = 'PID',
                  tau_c: Optional[float] = None) -> PIDGains:
    """Auto-tune PID for a given FOLPD process.

    Args:
        process: FOLPD process model
        method: 'ziegler_nichols', 'cohen_coon', or 'imc'
        controller_type: 'P', 'PI', or 'PID'
        tau_c: Closed-loop time constant (only for IMC method)

    Returns:
        PIDGains with tuned parameters
    """
    if method == 'ziegler_nichols':
        return process.ziegler_nichols_tuning(controller_type)
    elif method == 'cohen_coon':
        return process.coon_cohen_tuning(controller_type)
    elif method == 'imc':
        return process.imc_tuning(tau_c)
    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Use 'ziegler_nichols', 'cohen_coon', or 'imc'.")