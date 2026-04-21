"""
===============================================================================
MuJoCo Inverted Pendulum with LQR Control — Design Tool for a Real Cart-Pole
===============================================================================

PURPOSE
-------
This script is your DESIGN TOOL. It lives on your PC. Its job is to:
  1. Simulate the cart-pole with parameters matching your physical hardware.
  2. Linearize the dynamics around the upright equilibrium.
  3. Compute the LQR gain vector K using your chosen Q and R weights.
  4. Verify the controller stabilizes the system in sim, with realistic
     disturbances (friction, motor lag, sensor noise, input delay).

The final output you care about is the four-number vector K. You copy that
into your Arduino/Pi firmware, which runs u = -K*(x - x_ref) in a tight
real-time loop.

HOW TO USE THIS FILE
--------------------
Step 1: Measure your real hardware and fill in the PHYSICAL PARAMETERS section.
Step 2: Add realistic friction and actuator dynamics (FRICTION & ACTUATOR
        REALISM section). Start idealized, add realism incrementally.
Step 3: Tune Q and R in the LQR DESIGN section until the response is what
        you want: fast vs gentle, aggressive vs smooth.
Step 4: Run the simulation. Watch it balance. If it fails, either the gains
        are wrong or you've modeled something in a way that makes the
        controller unable to handle real effects (good — that tells you
        something about what your real hardware needs).
Step 5: When happy, print K and copy those four numbers into your firmware.

Two run modes (set USE_VIEWER below):
  - VIEWER MODE: opens an interactive 3D window. Ctrl + right-click-drag on
    the pole to kick it and watch the controller recover.
  - OFFSCREEN MODE: renders to a video file (for headless machines).

Windows/Mac/Linux with display:  python mujoco_cartpole_lqr.py
Headless Linux:                  MUJOCO_GL=egl python mujoco_cartpole_lqr.py
"""

import numpy as np
import mujoco
import scipy.linalg
import matplotlib.pyplot as plt

# ============================================================================
# RUN MODE
# ============================================================================
USE_VIEWER = True        # True = interactive 3D window, False = save video file
SAVE_PLOT  = True        # Save the state/control time-history plot either way


# ============================================================================
# PHYSICAL PARAMETERS — MEASURE THESE ON YOUR REAL HARDWARE
# ============================================================================
#
# Every number here should reflect your physical rig. When you change one,
# the linearization changes, so K changes. That's the whole point.
#
# UNITS: SI throughout. Meters, kilograms, seconds, radians, Newtons.
# ----------------------------------------------------------------------------

# --- Cart ---
# Mass of the cart assembly: cart body + anything that moves with it along
# the rail (belt attachment bracket, any wiring strain-relief, etc.).
# HOW TO MEASURE: put the cart on a kitchen scale. If the cart is connected
# to the belt and the belt has non-negligible mass, add roughly half the
# belt mass here (the moving half).
CART_MASS = 0.5          # kg

# Cart physical dimensions — used ONLY for visualization. These DO NOT
# affect dynamics (mass is what affects dynamics). Make them match your
# real cart so the animation looks right, so you can spot sign errors.
CART_SIZE = (0.15, 0.10, 0.05)   # half-sizes (x, y, z) in meters

# --- Pole / Pendulum ---
# Mass of the pole (rod + any tip weight you added).
# HOW TO MEASURE: weigh the complete pole assembly on a scale.
POLE_MASS = 0.2          # kg

# POLE_HALFLEN is half of the pole's total length. MuJoCo builds the pole
# as a capsule from (0,0,0) to (0,0,2*POLE_HALFLEN) attached at the pivot,
# so total length = 2 * POLE_HALFLEN.
# HOW TO MEASURE: measure from the pivot axis (center of the hinge shaft)
# to the far tip of the pole, in meters. Divide by 2 and put that here.
# NOTE: For a uniform rod of length L, MuJoCo computes the moment of inertia
# automatically from the capsule geometry and density. If your real pole
# has a concentrated tip mass or non-uniform weight distribution, you need
# to match its inertia more carefully — see POLE_MASS notes below.
POLE_HALFLEN = 0.3       # meters (so total pole length = 0.6 m)

# Visual radius of the pole capsule — doesn't affect dynamics meaningfully.
POLE_RADIUS = 0.02       # meters

# --- Rail ---
# Length of the track. For a 1 m rail, use 0.5 here (half-length).
# Your cart's allowed travel range will be slightly less (limits set below).
RAIL_HALF = 0.5          # meters (1 m total rail)


# ============================================================================
# FRICTION & ACTUATOR REALISM — THE STUFF THAT BREAKS REAL CONTROLLERS
# ============================================================================
#
# This is where your simulation earns its keep. An LQR tuned on a frictionless
# massless-belt perfect-motor model will fail on your real hardware because
# the real hardware has:
#   - Cart friction (belt drag, pulley bearings, air resistance)
#   - Pole hinge friction (bearing in the pivot)
#   - Motor dynamics (it doesn't produce requested force instantly)
#   - Sensor noise and quantization (encoders only report discrete counts)
#   - Control latency (time from sensor read to motor command)
#   - Static friction / stiction (especially bad — causes limit cycles)
#
# Model these here. Start with the dominant ones; don't get lost tuning
# every parameter on day one.
# ----------------------------------------------------------------------------

# CART_DAMPING: viscous friction on the slider joint, N per (m/s).
# This models belt drag, pulley bearing friction, and air drag combined.
# HOW TO MEASURE ON YOUR RIG:
#   1. Disconnect the motor.
#   2. Give the cart a firm push.
#   3. Use your encoder to log position vs time as it coasts to a stop.
#   4. Fit v(t) = v0 * exp(-b/M * t) — the decay rate tells you b/M.
#   5. Multiply by CART_MASS to get b.
# Typical hobby rigs: 0.1 - 2.0 N/(m/s). Err high if unsure; LQR handles
# more damping gracefully but gets surprised by less damping than modeled.
CART_DAMPING = 0.1       # N/(m/s)

# POLE_DAMPING: viscous friction on the hinge joint, N·m per (rad/s).
# This is the bearing friction in the pole's pivot. Usually tiny for a
# decent bearing — a drop of oil on a 608 skate bearing gives <0.001.
# HOW TO MEASURE: hold the cart still, let the pole swing freely (pointing
# DOWN, so it's a regular pendulum), log the angle, measure how many
# oscillations it takes for amplitude to halve.
POLE_DAMPING = 0.001     # N·m/(rad/s)

# ARMATURE (reflected motor inertia on the slider):
# Your DC motor + gearhead + belt pulley has rotational inertia that the
# cart has to accelerate when it moves. It "feels" like extra cart mass.
# For belt-driven carts, effective inertia on the cart = J_motor / r^2
# where r is the pulley radius. You can either:
#   (a) lump this into CART_MASS directly (easiest), or
#   (b) use MuJoCo's <joint armature=...> attribute, which adds inertia
#       to the joint's DOF without adding mass to the body.
# For a typical 6mm pulley and a small DC motor, reflected inertia is often
# comparable to or larger than the cart mass itself — do not skip this.
CART_ARMATURE = 0.01     # kg (effective additional cart mass)

# MOTOR_TAU: first-order motor time constant (seconds).
# When your control loop commands "10 N," the motor doesn't produce 10 N
# instantly — there's electrical (L/R) and mechanical lag. A first-order
# filter u_actual = lag_filter(u_commanded, tau) captures this.
# Typical small DC motors: 10-50 ms. With a fast current-mode motor driver,
# can be <5 ms.
# If you don't model this, your LQR will command fast force reversals that
# the motor can't actually produce, and your real system will oscillate.
# We implement this in the control loop (not MJCF) — see run_*() functions.
MOTOR_TAU = 0.02         # seconds (20 ms motor lag)

# CONTROL_LATENCY: time from sensor read to motor command actually changing
# on the real system. Includes: encoder read time, serial comms between Pi
# and Arduino, control math, PWM update. Typically 1-10 ms.
# We model this as a fixed delay on u in the control loop.
CONTROL_LATENCY_STEPS = 2    # integer multiples of sim timestep (2ms each)

# ENCODER_RESOLUTION: counts per meter for cart, counts per radian for pole.
# Real encoders report integer counts, not smooth floats. This causes
# quantization noise in your state estimate — worse at low speeds.
# HOW TO COMPUTE:
#   Cart: (encoder CPR * 4 for quadrature) / (belt pulley circumference)
#         e.g. 500 CPR encoder * 4 = 2000 counts/rev
#         On a 20mm-diameter pulley (circ = 0.0628 m) → 31,847 counts/m
#   Pole: encoder CPR * 4 (quadrature) / (2*pi)
#         e.g. 2000 quadrature counts / 2π = 318 counts/rad
# Set these to None to disable quantization in sim.
ENCODER_CART_COUNTS_PER_M   = 30000   # ~ 500 CPR quadrature on 20mm pulley
ENCODER_POLE_COUNTS_PER_RAD = 1000    # ~ 1000 CPR quadrature encoder

# SENSOR_NOISE: standard deviation of Gaussian noise added to each state
# element. Real encoders are quite clean (quantization dominates), but
# velocity estimates (from differentiating position) are noisy.
# State order: [cart_pos, pole_angle, cart_vel, pole_angvel]
SENSOR_NOISE_STD = np.array([0.0, 0.0, 0.01, 0.05])   # m, rad, m/s, rad/s


# ============================================================================
# INITIAL CONDITIONS — WHERE THE SIM STARTS
# ============================================================================
#
# The linearization is valid for small angles. Initial tilts up to ~0.3 rad
# (~17°) usually still stabilize; beyond that, the linear approximation
# breaks down and you may need swing-up control (a different topic entirely).
# ----------------------------------------------------------------------------
INITIAL_POLE_ANGLE = 0.20    # radians (~11.5°)
INITIAL_POLE_RATE  = 0.0     # rad/s — nonzero to test recovery from a push
INITIAL_CART_POS   = 0.0     # meters
INITIAL_CART_VEL   = 0.0     # m/s


# ============================================================================
# MJCF MODEL — TRANSLATES PHYSICAL PARAMETERS INTO A MUJOCO SIMULATION
# ============================================================================
#
# MJCF is MuJoCo's XML format. You rarely need to edit the structure here —
# just adjust the parameters above, and the string substitutions below fill
# the right values in.
#
# Key MJCF pieces:
#   <option> — global solver settings (integrator, timestep, gravity)
#   <default> — default values for joints/geoms (inherited unless overridden)
#   <body> — a rigid body, can contain joints, geoms, and child bodies
#   <joint> — a degree of freedom connecting a body to its parent
#   <geom> — a collision/visual shape with mass and inertia
#   <actuator> — a motor or force input driven by ctrl values
# ----------------------------------------------------------------------------

MJCF = f"""
<mujoco model="cartpole">
  <!--
  INTEGRATOR NOTE: 'implicitfast' is MuJoCo's default semi-implicit integrator.
  It is REQUIRED for mjd_transitionFD (RK4 is not supported by the derivative
  tool). Timestep 0.002 s = 500 Hz simulation. Reduce to 0.001 or 0.0005 if
  you see instability with very stiff systems.
  -->
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

  <default>
    <!-- Default joint properties. Individual joints override these below. -->
    <joint damping="0.0"/>
    <geom friction="1 0.005 0.0001"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.8"
             rgb2="0.6 0.6 0.6" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 -1 2" dir="0 0.3 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
    <light pos="0 1 2" dir="0 -0.3 -1" diffuse="0.6 0.6 0.6"/>
    <light pos="2 0 2" dir="-0.5 0 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Ground plane, with contact enabled (contype/conaffinity = 1). -->
    <geom name="floor" type="plane" size="5 1 0.1" rgba="0.9 0.9 0.9 1"
          material="grid" contype="1" conaffinity="1"/>

    <!-- Rail: visual only (no contact) so the cart doesn't collide with it. -->
    <geom name="rail" type="capsule"
          fromto="-{RAIL_HALF} 0 0.05  {RAIL_HALF} 0 0.05"
          size="0.01" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>

    <!-- CART BODY -->
    <body name="cart" pos="0 0 0.05">
      <!--
      SLIDER JOINT:
        - type="slide" = prismatic joint (linear motion along one axis)
        - axis="1 0 0" = moves along world x-axis (the rail direction)
        - damping = cart friction, in N/(m/s). This is CART_DAMPING above.
        - armature = reflected motor inertia, in kg. Adds to effective mass
                     on this DOF without physically showing up as a body mass.
        - range = soft limit on travel, so the cart can't leave the rail.
      -->
      <joint name="slider" type="slide" axis="1 0 0"
             damping="{CART_DAMPING}" armature="{CART_ARMATURE}"
             range="-{RAIL_HALF - 0.1} {RAIL_HALF - 0.1}"/>

      <!--
      CART GEOM: the visual and mass-carrying box for the cart.
        - contype/conaffinity = 2 puts cart+pole in a contact group that
          does not match the floor (1), so the cart doesn't collide with it.
        - mass = CART_MASS. MuJoCo computes inertia from the box dimensions
          and this mass automatically.
      -->
      <geom name="cart_geom" type="box"
            size="{CART_SIZE[0]} {CART_SIZE[1]} {CART_SIZE[2]}"
            rgba="0.2 0.4 0.8 1" mass="{CART_MASS}"
            contype="2" conaffinity="2"/>

      <!-- POLE BODY — child of cart, pivots at the top of the cart -->
      <body name="pole" pos="0 0 {CART_SIZE[2]}">
        <!--
        HINGE JOINT:
          - type="hinge" = revolute joint (rotation about one axis)
          - axis="0 1 0" = rotates about world y-axis (pole tips in x-z plane)
          - damping = POLE_DAMPING, bearing friction at the pivot.
        At qpos=0 the pole points straight up (unstable equilibrium).
        -->
        <joint name="hinge" type="hinge" axis="0 1 0"
               damping="{POLE_DAMPING}"/>

        <!--
        POLE GEOM: capsule from pivot to tip. Distributed mass, so
        MuJoCo computes the moment of inertia from the geometry + mass.
        For a uniform rod of length L = 2*POLE_HALFLEN and mass m:
          I_about_pivot = (1/3) m L^2
        which MuJoCo handles automatically from the capsule dimensions.
        If your real pole has a heavy tip (like a steel ball), you need
        either a separate <geom> at the tip or tweak the capsule mass/size
        to match the real inertia. The capsule approximation is usually
        close enough for a first-pass design.
        -->
        <geom name="pole_geom" type="capsule"
              fromto="0 0 0  0 0 {2 * POLE_HALFLEN}"
              size="{POLE_RADIUS}" rgba="0.8 0.2 0.2 1" mass="{POLE_MASS}"
              contype="2" conaffinity="2"/>

        <!-- Yellow marker at the tip, for visualization only. -->
        <site name="tip" pos="0 0 {2 * POLE_HALFLEN}" size="0.025"
              rgba="1 1 0 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!--
    MOTOR ACTUATOR:
      - joint="slider" = this actuator applies force to the slider joint
      - gear="1" = no mechanical advantage; ctrl value equals force in Newtons
      - ctrlrange = saturation limit. Match to your real motor + driver's
        peak force capability. For a small DC motor + L298N driver on a
        cart-pole rig, ~10-30 N is realistic; we use 50 N here to give
        headroom.
    TO CONVERT FROM MOTOR SPECS:
      Peak force at cart (N) = (motor torque, N·m) * (gear ratio)
                               / (pulley radius, m)
      Account for gearing and the pulley diameter of your belt drive.
    -->
    <motor name="cart_motor" joint="slider" gear="1" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
"""


# ============================================================================
# LQR DESIGN — THE CONTROLLER SYNTHESIS
# ============================================================================
#
# LQR (Linear Quadratic Regulator) finds the optimal feedback gain K that
# minimizes the cost function:
#
#     J = sum over time of [ x^T Q x  +  u^T R u ]
#
# where x is the state error and u is the control input.
#
# - Q is a diagonal matrix of weights on each state. LARGER Q[i,i] means
#   "I really care about this state being small."
# - R is the weight on control effort. LARGER R means "use less force."
# - The ratio Q/R determines the tradeoff: big Q/R = aggressive but
#   possibly jerky controller; small Q/R = gentle but slower recovery.
#
# Default weights below are a reasonable starting point. TUNE THESE:
#   - Pole falls over? Increase Q[pole_angle] (position 1 in the diagonal).
#   - Pole balances but cart drifts off the rail? Increase Q[cart_pos] (0).
#   - Controller is jerky / motor whines? Increase R.
#   - Controller is sluggish? Decrease R, or increase Q uniformly.
# ----------------------------------------------------------------------------

def design_lqr(model, data, verbose=True):
    """
    Linearize the system at the upright fixed point and compute the
    discrete-time LQR gain K.

    State vector order (from MuJoCo's [qpos; qvel] convention):
        x[0] = cart position  (m)
        x[1] = pole angle     (rad, 0 = upright, + = tipping in +x direction)
        x[2] = cart velocity  (m/s)
        x[3] = pole angular velocity (rad/s)

    Control:
        u = force on cart (N), positive = push in +x direction

    Returns K such that the control law is:
        u = -K @ (x - x_ref)
    where x_ref is usually zero (upright, at origin, stationary).
    """
    # ----- Step 1: Put the system at the equilibrium point -----
    # mjd_transitionFD linearizes around whatever state it finds in `data`.
    # The upright fixed point is (all zeros). We set it and then call
    # mj_forward to populate derived quantities (Jacobians etc).
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    # ----- Step 2: Finite-difference the dynamics -----
    # mjd_transitionFD computes A = df/dx and B = df/du where
    #   x_{k+1} = f(x_k, u_k)
    # is MuJoCo's discrete-time dynamics (one call to mj_step).
    # These are DISCRETE-TIME matrices — they describe what happens in one
    # timestep, not in continuous time. So we need DISCRETE-TIME LQR below.
    nx = 2 * model.nv    # state dim = qpos + qvel = 2 + 2 = 4
    nu = model.nu        # input dim = 1
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, 1, A, B, None, None)

    # ----- Step 3: Design Q and R weights -----
    # STATE ORDER: [cart_pos, pole_angle, cart_vel, pole_angvel]
    # Start here and TUNE FOR YOUR HARDWARE:
    #   - If the pole wobbles but doesn't fall, you're fine.
    #   - If the cart drifts to one end of the rail, raise Q[0] (cart pos).
    #   - If the motor is too aggressive/loud, raise R.
    #   - If recovery is too slow, raise Q[1] (pole angle) or lower R.
    Q = np.diag([
        1.0,    # cart position — raise if cart drifts off rail
        50.0,   # pole angle — usually the most important; raise if pole wobbles
        1.0,    # cart velocity
        5.0,    # pole angular velocity — adds damping to the response
    ])
    R = np.array([[0.05]])   # control effort — raise to use less force

    # ----- Step 4: Solve the discrete-time algebraic Riccati equation -----
    # DARE: P = A'PA - A'PB(R + B'PB)^-1 B'PA + Q
    # Then optimal gain: K = (R + B'PB)^-1 B'PA
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    if verbose:
        print("\n=== LQR Design Results ===")
        print("Discrete-time A (one-step state transition):")
        print(np.array2string(A, precision=4, suppress_small=True))
        print("\nDiscrete-time B (input response per step):")
        print(np.array2string(B, precision=4, suppress_small=True))
        print("\nQ (state cost, diagonal):", np.diag(Q))
        print("R (control cost):         ", R.flatten())
        print("\nLQR gain K = ", K.flatten())
        print("Control law: u = -K @ [cart_pos, pole_angle, cart_vel, pole_angvel]")
        print("===========================\n")

    return np.asarray(K)


# ============================================================================
# CONTROL LOOP HELPERS — APPLY REALISTIC EFFECTS
# ============================================================================

class MotorLag:
    """
    First-order motor dynamics: the motor can't produce requested force
    instantly. u_actual evolves toward u_commanded with time constant tau.
        du_actual/dt = (u_commanded - u_actual) / tau
    Discretized as exponential filter.
    """
    def __init__(self, tau, dt):
        self.alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
        self.u = 0.0

    def step(self, u_commanded):
        self.u += self.alpha * (u_commanded - self.u)
        return self.u


def quantize(x, counts_per_unit):
    """Encoder quantization: real encoders report integer counts only."""
    if counts_per_unit is None or counts_per_unit <= 0:
        return x
    return np.round(x * counts_per_unit) / counts_per_unit


def apply_sensor_model(x_true, rng):
    """Simulate what the real controller would actually measure."""
    x = x_true.copy()
    # Encoder quantization on positions (velocities come from differentiation,
    # which is handled by noise instead).
    x[0] = quantize(x[0], ENCODER_CART_COUNTS_PER_M)
    x[1] = quantize(x[1], ENCODER_POLE_COUNTS_PER_RAD)
    # Additive Gaussian noise (mainly affects velocity estimates).
    x += rng.normal(0.0, SENSOR_NOISE_STD)
    return x


# ============================================================================
# RUN MODES — INTERACTIVE VIEWER AND OFFSCREEN VIDEO
# ============================================================================

def _setup_initial_state(model, data):
    """Put the system at the configured initial conditions."""
    slider_qpos = model.jnt_qposadr[model.joint("slider").id]
    hinge_qpos  = model.jnt_qposadr[model.joint("hinge").id]
    slider_qvel = model.jnt_dofadr [model.joint("slider").id]
    hinge_qvel  = model.jnt_dofadr [model.joint("hinge").id]

    mujoco.mj_resetData(model, data)
    data.qpos[slider_qpos] = INITIAL_CART_POS
    data.qpos[hinge_qpos]  = INITIAL_POLE_ANGLE
    data.qvel[slider_qvel] = INITIAL_CART_VEL
    data.qvel[hinge_qvel]  = INITIAL_POLE_RATE
    mujoco.mj_forward(model, data)


def _control_step(x_true, K, x_ref, rng, motor, u_delay_buf, u_max):
    """
    One step of the realistic control pipeline:
        true state -> sensor model -> LQR -> motor lag -> delay -> plant
    """
    # 1. Sensor: quantize + noise
    x_measured = apply_sensor_model(x_true, rng)

    # 2. LQR law: u_commanded = -K (x - x_ref)
    u_commanded = float((-K @ (x_measured - x_ref))[0])

    # 3. Saturate to motor capability
    u_commanded = float(np.clip(u_commanded, -u_max, u_max))

    # 4. Push through control latency delay buffer
    if CONTROL_LATENCY_STEPS > 0:
        u_delay_buf.append(u_commanded)
        u_delayed = u_delay_buf.pop(0)
    else:
        u_delayed = u_commanded

    # 5. Motor lag (first-order dynamics)
    u_actual = motor.step(u_delayed)

    return u_actual, u_commanded


def run_viewer(model, data, K, t_final=30.0, u_max=50.0):
    """Interactive 3D viewer with live LQR control."""
    import mujoco.viewer
    import time

    _setup_initial_state(model, data)
    x_ref = np.zeros(2 * model.nv)
    rng = np.random.default_rng(0)

    motor = MotorLag(MOTOR_TAU, model.opt.timestep)
    u_delay_buf = [0.0] * CONTROL_LATENCY_STEPS

    t_hist, state_hist, u_hist = [], [], []

    print("\nOpening interactive viewer...")
    print("  - Drag with left mouse to rotate, right mouse to pan, scroll to zoom")
    print("  - Ctrl + right-click-drag on the pole to perturb it")
    print("  - Press ESC or close the window to exit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10

        sim_time_start = data.time
        while viewer.is_running() and (data.time - sim_time_start) < t_final:
            step_start = time.time()

            x_true = np.concatenate([data.qpos, data.qvel])
            u_actual, u_cmd = _control_step(
                x_true, K, x_ref, rng, motor, u_delay_buf, u_max)

            data.ctrl[0] = u_actual

            t_hist.append(data.time)
            state_hist.append(x_true.copy())
            u_hist.append(u_actual)

            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    return np.array(t_hist), np.array(state_hist), np.array(u_hist)


def run_offscreen(model, data, K, t_final=6.0, u_max=50.0,
                  video_path="mujoco_cartpole.mp4",
                  gif_path="mujoco_cartpole.gif"):
    """Run the sim and save a rendered video."""
    import imageio.v2 as imageio

    dt_sim = model.opt.timestep
    n_steps = int(t_final / dt_sim)

    _setup_initial_state(model, data)
    x_ref = np.zeros(2 * model.nv)
    rng = np.random.default_rng(0)

    motor = MotorLag(MOTOR_TAU, dt_sim)
    u_delay_buf = [0.0] * CONTROL_LATENCY_STEPS

    t_hist = np.zeros(n_steps)
    state_hist = np.zeros((n_steps, 2 * model.nv))
    u_hist = np.zeros(n_steps)

    fps = 30
    render_every = max(1, int(round(1.0 / (fps * dt_sim))))
    renderer = mujoco.Renderer(model, height=360, width=640)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat = np.array([0.0, 0.0, 0.35])
    cam.distance = 2.8
    cam.azimuth = 90
    cam.elevation = -10

    frames = []

    for i in range(n_steps):
        x_true = np.concatenate([data.qpos, data.qvel])
        u_actual, u_cmd = _control_step(
            x_true, K, x_ref, rng, motor, u_delay_buf, u_max)

        data.ctrl[0] = u_actual

        t_hist[i] = data.time
        state_hist[i] = x_true
        u_hist[i] = u_actual

        if i % render_every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        mujoco.mj_step(model, data)

    renderer.close()

    try:
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)
        print(f"Saved video: {video_path}")
    except Exception as e:
        print(f"MP4 save failed ({e}); saving GIF only.")

    imageio.mimsave(gif_path, frames[::2], fps=fps // 2, loop=0)
    print(f"Saved GIF:   {gif_path}")

    return t_hist, state_hist, u_hist


def plot_response(t_hist, state_hist, u_hist, model,
                  path="mujoco_cartpole_response.png"):
    """Plot cart, pole, and control histories."""
    slider_qpos = model.jnt_qposadr[model.joint("slider").id]
    hinge_qpos  = model.jnt_qposadr[model.joint("hinge").id]
    slider_qvel = model.jnt_dofadr [model.joint("slider").id]
    hinge_qvel  = model.jnt_dofadr [model.joint("hinge").id]

    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    axs[0].plot(t_hist, state_hist[:, slider_qpos], label="Cart position (m)")
    axs[0].plot(t_hist, state_hist[:, model.nv + slider_qvel],
                label="Cart velocity (m/s)", alpha=0.7)
    axs[0].axhline(RAIL_HALF, color="red", linestyle=":", alpha=0.4,
                   label="Rail limit")
    axs[0].axhline(-RAIL_HALF, color="red", linestyle=":", alpha=0.4)
    axs[0].set_ylabel("Cart"); axs[0].legend(loc="upper right"); axs[0].grid(True)

    axs[1].plot(t_hist, np.degrees(state_hist[:, hinge_qpos]),
                label="Pole angle (deg)", color="C2")
    axs[1].plot(t_hist, np.degrees(state_hist[:, model.nv + hinge_qvel]),
                label="Pole angular vel (deg/s)", color="C3", alpha=0.7)
    axs[1].set_ylabel("Pole"); axs[1].legend(loc="upper right"); axs[1].grid(True)

    axs[2].plot(t_hist, u_hist, label="Control force (N)", color="C4")
    axs[2].set_ylabel("Control"); axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc="upper right"); axs[2].grid(True)

    fig.suptitle("MuJoCo Cart-Pole — LQR Control (with realistic effects)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot:  {path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)

    print("=" * 70)
    print("MuJoCo Cart-Pole LQR Design Tool")
    print("=" * 70)
    print(f"  Cart mass:    {CART_MASS} kg")
    print(f"  Pole mass:    {POLE_MASS} kg")
    print(f"  Pole length:  {2*POLE_HALFLEN} m")
    print(f"  Cart damping: {CART_DAMPING} N/(m/s)")
    print(f"  Pole damping: {POLE_DAMPING} N·m/(rad/s)")
    print(f"  Motor tau:    {MOTOR_TAU*1000:.1f} ms")
    print(f"  Latency:      {CONTROL_LATENCY_STEPS * model.opt.timestep*1000:.1f} ms")
    print(f"  Timestep:     {model.opt.timestep*1000:.1f} ms")
    print()

    # DESIGN: compute the LQR gain. THIS is what goes into your firmware.
    K = design_lqr(model, data)

    # SIMULATE: verify it works with realistic disturbances.
    if USE_VIEWER:
        t_hist, state_hist, u_hist = run_viewer(model, data, K)
    else:
        t_hist, state_hist, u_hist = run_offscreen(model, data, K)

    if SAVE_PLOT and len(t_hist) > 1:
        plot_response(t_hist, state_hist, u_hist, model)

    # Print K one more time in a copy-paste-friendly format.
    print("\n" + "=" * 70)
    print("COPY THIS INTO YOUR ARDUINO/PI FIRMWARE:")
    print("=" * 70)
    print(f"float K[4] = {{ {K[0,0]:.6f}f, {K[0,1]:.6f}f, "
          f"{K[0,2]:.6f}f, {K[0,3]:.6f}f }};")
    print("// Order: [cart_pos_m, pole_angle_rad, cart_vel_mps, pole_angvel_radps]")
    print("// Control law: u = -(K[0]*cart_pos + K[1]*pole_angle + ")
    print("//                   K[2]*cart_vel + K[3]*pole_angvel)")
    print("=" * 70)


if __name__ == "__main__":
    main()
