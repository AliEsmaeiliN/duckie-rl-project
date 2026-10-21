# coding=utf-8
"""
Numpy port of duckietown_world's DB18 PWM dynamics (pwm_dynamics.DynamicModel).

The original runs on PyGeometry, which validates every SE(2) matrix with numpy.testing
assertions; that made one integration step cost ~0.7 ms, three times per env step. This
version does the same math in closed form on plain floats. tests: debug/check_dynamics.py
compares it step by step against the original.
"""
import math

# Parameters of get_DB18_nominal / get_DB18_uncalibrated
U1, U2, U3 = 5.0, 0.0, 0.0  # autonomous longitudinal dynamics
W1, W2, W3 = 4.0, 0.0, 0.0  # autonomous angular dynamics
U_ALPHA = 1.5  # forced longitudinal response, both wheels
W_ALPHA = 15.0  # forced angular response, scaled by trim


class DB18Dynamics:
    """
    State of a DB18 Duckiebot: pose (x, y, theta) in the cartesian frame and body
    velocities (u longitudinal, w angular), integrated in place.
    """

    def __init__(self, x: float, y: float, theta: float, trim: float = 0.0):
        self.x, self.y, self.theta = x, y, theta
        self.u = 0.0
        self.w = 0.0
        self.w_alpha_r = W_ALPHA * (1.0 + trim)
        self.w_alpha_l = W_ALPHA * (1.0 - trim)

    @property
    def pose(self):
        # Heading wrapped to (-pi, pi], as PyGeometry's translation_angle_from_SE2 returns it
        return self.x, self.y, math.atan2(math.sin(self.theta), math.cos(self.theta))

    def integrate(self, dt: float, motor_left: float, motor_right: float):
        right = min(max(float(motor_right), -1.0), 1.0)
        left = min(max(float(motor_left), -1.0), 1.0)
        u, w = self.u, self.w

        # Acceleration: autonomous response plus forced response B @ [right, left]
        u_dd = -U1 * u - U2 * w + U3 * w * w + U_ALPHA * right + U_ALPHA * left
        w_dd = -W1 * w - W2 * u - W3 * u * w + self.w_alpha_r * right - self.w_alpha_l * left

        # Forward Euler on the velocities, lateral velocity is always zero
        u = u + dt * u_dd
        w = w + dt * w_dd
        self.u, self.w = u, w

        # Hold (u, w) for dt: exponential map of se(2), then compose with the current pose
        dtheta = w * dt
        if abs(dtheta) < 1e-9:
            dx, dy = u * dt, 0.0
        else:
            r = u / w
            dx, dy = r * math.sin(dtheta), r * (1.0 - math.cos(dtheta))
        c, s = math.cos(self.theta), math.sin(self.theta)
        self.x += c * dx - s * dy
        self.y += s * dx + c * dy
        self.theta += dtheta
