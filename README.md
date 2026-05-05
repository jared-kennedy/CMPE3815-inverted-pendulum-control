# Inverted Pendulum Cart-Pole — CMPE3815
**University of Vermont | Jared Kennedy Eva Yost Emmett Cline Lucey

A belt-driven cart-pole inverted pendulum balanced by a cascaded PD/PI controller running entirely on an Arduino Uno R3.

---

## How It Works

The pole is balanced using two nested control loops:

**Inner loop — pole balancing (PD)**
A proportional-derivative controller drives the cart motor to keep the pole upright. The proportional term corrects angle error; the derivative term damps angular velocity to prevent oscillation.

**Outer loop — cart centering (PI)**
A proportional-integral controller on cart position outputs a small angle offset fed into the pole controller as a setpoint correction. The idea: to move the cart left, lean the pole slightly left — the pole controller then drives the cart in that direction as a side effect of keeping the pole upright.

**Startup sequence**
1. Power on. The system runs a ~0.8 s motor calibration to estimate mechanical force bias (belt drag asymmetry).
2. After calibration, the motor sits idle. Hold the pole upright within 2° of vertical to arm the controller.
3. Release. The controller runs at 500 Hz until a safety limit trips (end-stop or pole fallen past ~75°). Bring the pole upright again to re-arm.

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | Arduino Uno R3 |
| Motor driver | BTS7960 (IBT-2) H-bridge |
| Motor | 12V DC, belt-driven |
| Rail | 1m aluminum extrusion |
| Encoders | 600 P/R incremental quadrature (×2), AB 2-phase, 5–24V |
| Cart encoder | Mounted on belt-return pulley (50.64mm diameter) |
| Pole encoder | Direct-drive on pivot shaft |

**Encoder scaling (X4 quadrature decoding)**
- Cart: 2400 counts/rev ÷ (π × 0.05064 m) = **15,092 counts/meter**
- Pole: 2400 counts/rev ÷ (2π) = **381.97 counts/radian**

---

## Pin Assignments

| Pin | Function |
|---|---|
| 2 | Cart encoder A (INT0, CHANGE) |
| 3 | Pole encoder A (INT1, CHANGE) |
| 8 | Cart encoder B (PCINT0) |
| 9 | Pole encoder B (PCINT1) |
| 5 | BTS7960 LPWM — backward |
| 6 | BTS7960 RPWM — forward |
| 4 | BTS7960 R_EN |
| 7 | BTS7960 L_EN |

**Wiring notes**
- Encoder VCC → Arduino 5V
- 12V supply GND → Arduino GND (common ground is critical)
- Swap M+ / M− on the BTS7960 if the cart moves the wrong direction

---

## Tuned Parameters

**Control gains**
```
kp_pole   = 800.0     pole proportional gain
pole_damp =   3.1     pole derivative (velocity damping) gain
kp_cart   =   0.0009  cart position proportional gain
ki_cart   =   0.0     cart position integral gain (currently off)
```

**Motor**
```
FORCE_TO_PWM  = 22.0   Newtons to PWM scaling
PWM_DEADBAND  = 18     added to every command to overcome stiction
PWM_BIAS      =  8     forward/backward asymmetry correction
FORCE_DEADBAND = 0.2   commands below this (N) are zeroed
MAX_PWM       = 255
```

**Safety limits**
```
CART_LIMIT_M  = 0.42 m     software end-stop (physical rail half = 0.5 m)
POLE_FAIL_RAD = 1.30 rad   ~75°, disarms controller if exceeded
START_ANGLE   = 0.035 rad  ~2°, pole must be within this to arm
```

---

## Tuning Sequence (hardware bring-up)

1. **Upload `cartpole_pid.ino`.** Open Serial Monitor at any baud (the sketch does not use serial).

2. **Verify encoder directions.** Power on with the motor disconnected. Push the cart toward the motor end — `cart_pos` should go positive (check by adding a `Serial.println` temporarily). Tilt the pole toward the motor end — `pole_angle` should increase. If either is wrong, negate the relevant line in `cartUpdate()` or `poleUpdate()`.

3. **Verify motor direction.** With low gains, arm the controller and tilt the pole slightly in the +x direction. The cart should move in the +x direction. If it moves the wrong way, swap M+ and M− on the BTS7960.

4. **Tune `FORCE_TO_PWM`.** Support the pole by hand, arm the controller, release slowly. If the cart barely moves, decrease `FORCE_TO_PWM`. If it immediately slams the end-stop, increase it.

5. **Tune `kp_pole` and `pole_damp`.** Start with `kp_pole` low (~200) and increase until the pole holds. Add `pole_damp` to kill oscillation. Then increase `kp_pole` further for tighter balance.

6. **Tune `kp_cart`.** Once the pole holds, the cart will drift. Increase `kp_cart` slowly until it self-centers without destabilizing the pole.

---


## Project Status

- [x] Physical mechanism built and wired
- [x] BTS7960 motor driver tested
- [x] Quadrature encoder decoding (X4 via lookup table)
- [x] 500 Hz Timer1 control loop
- [x] PD pole controller + PI cart centering
- [x] Motor calibration routine
- [x] Safety limits (end-stop, pole fallen, overcurrent)
- [x] First successful unassisted balance
