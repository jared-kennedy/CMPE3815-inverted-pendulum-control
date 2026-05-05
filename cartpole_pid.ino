/*
 * =============================================================================
 * cartpole_pid.ino — Inverted Pendulum PID Controller
 * CMPE3815 — University of Vermont
 * Jared Kennedy 
 * Eva Yost
 * Emmett Cline Lucey
 * =============================================================================
 *
 * OVERVIEW
 * --------
 * Standalone PID balance controller for a belt driven cart-pole inverted
 * pendulum. Runs entirely on an Arduino Uno R3 — no Raspberry Pi or PC
 * required during operation.
 *
 * The control loop fires at 500 Hz via a Timer1 interrupt.
 * Two quadrature encoders (X4 decoding via lookup table) measure cart
 * position and pole angle. A cascaded PD/PI controller computes a force
 * command, which is converted to a PWM signal for the BTS7960 H-bridge.
 *
 * CONTROL ARCHITECTURE
 * --------------------
 * Outer loop (cart centering):
 *   - PI controller on cart position error
 *   - Output: a small angle setpoint offset fed to the pole controller
 *   - Idea: to move the cart left, lean the pole slightly left so the
 *     pole controller drives the cart that direction. Adjusted from typical PID
 *     by having a non linear cubic mutiplier zone.
 *
 * Inner loop (pole balancing):
 *   - PD controller on pole angle error (angle relative to upright)
 *   - Setpoint is pi radians (upright) plus the cart centering correction
 *   - D term is a direct damping term on pole angular velocity
 *
 * STARTUP SEQUENCE
 * ----------------
 *   1. Power on. System runs a short motor calibration routine (~0.8 s)
 *      to estimate mechanical force bias (belt drag asymmetry, etc.).
 *   2. After calibration, system waits with motor off until the pole is
 *      held upright within START_ANGLE of vertical.
 *   3. Once armed, the control loop runs continuously. Safety limits
 *      disarm the controller if the cart hits the end-stop or the pole
 *      falls past ~75 degrees. To restart, bring the pole upright again.
 *
 * HARDWARE
 * --------
 *   - Arduino Uno R3
 *   - BTS7960 (IBT-2) H-bridge motor driver, 12V supply
 *   - Two 600 P/R incremental quadrature encoders (AB 2-phase, 5V)
 *       Cart encoder: mounted on belt-return pulley (50.64mm diameter)
 *       Pole encoder: direct-drive on pivot shaft
 *   - DC motor (12V) driving a belt and pulley cart on a 1m rail
 *
 * PIN ASSIGNMENTS
 * ---------------
 *   Pin 2  — Cart encoder A  (INT0, CHANGE interrupt)
 *   Pin 3  — Pole encoder A  (INT1, CHANGE interrupt)
 *   Pin 8  — Cart encoder B  (PCINT0 pin-change interrupt, also read in ISR)
 *   Pin 9  — Pole encoder B  (PCINT1 pin-change interrupt, also read in ISR)
 *   Pin 6  — BTS7960 RPWM   (PWM, forward / +x direction)
 *   Pin 5  — BTS7960 LPWM   (PWM, backward / -x direction)
 *   Pin 4  — BTS7960 R_EN   (enable right half-bridge, tied HIGH)
 *   Pin 7  — BTS7960 L_EN   (enable left half-bridge, tied HIGH)
 *
 * TUNED GAINS (as of last successful balance run)
 * ------------------------------------------------
 *   kp_pole  = 800.0    — pole proportional gain
 *   pole_damp = 3.1     — pole derivative (velocity damping) gain
 *   kp_cart  = 0.0009   — cart position proportional gain
 *   ki_cart  = 0.0000   — cart position integral gain (currently off)
 * =============================================================================
 */

#include <Arduino.h>

// ===================== PINS =====================
// Motor driver and encoder pin assignments.
// See wiring notes in the header above.
const int PIN_CART_ENC_A = 2;   // cart encoder channel A — INT0
const int PIN_CART_ENC_B = 8;   // cart encoder channel B — PCINT0
const int PIN_POLE_ENC_A = 3;   // pole encoder channel A — INT1
const int PIN_POLE_ENC_B = 9;   // pole encoder channel B — PCINT1

const int PIN_RPWM = 6;   // BTS7960 RPWM — forward (positive x) PWM
const int PIN_LPWM = 5;   // BTS7960 LPWM — backward (negative x) PWM
const int PIN_R_EN  = 4;  // BTS7960 right half-bridge enable (set HIGH in setup)
const int PIN_L_EN  = 7;  // BTS7960 left half-bridge enable  (set HIGH in setup)

// ===================== PHYSICS =====================
// Encoder-to-SI unit conversion factors.
//
// Cart encoder sits on the belt-return pulley (50.64 mm diameter).
//   Circumference = π × 0.05064 = 0.15904 m/rev
//   With X4 quadrature (4 edges per pulse, 600 PPR): 2400 counts/rev
//   → 2400 / 0.15904 = 15,092 counts/m
//
// Pole encoder is direct-drive on the pivot shaft.
//   With X4 quadrature: 2400 counts/rev
//   → 2400 / (2π) = 381.97 counts/rad
const float CART_M_PER_COUNT   = 1.0f / 15092.0f;   // meters per encoder count
const float POLE_RAD_PER_COUNT = 1.0f / 381.97f;    // radians per encoder count

// ===================== TIMING =====================
// Control loop period in seconds. Timer1 fires at 500 Hz → DT = 2 ms.
// Used for velocity differentiation and integrator accumulation.
const float DT = 0.002f;

// ===================== LIMITS =====================
// CART_LIMIT_M: software end-stop. If the cart travels this far from center
//   (in meters), the controller disarms and the motor stops. This protects
//   the physical end-stops from being hit at speed. Set slightly inside the
//   physical rail half-length (0.5 m).
const float CART_LIMIT_M = 0.42f;

// POLE_FAIL_RAD: if the pole angle exceeds this (in radians, ~75°), the
//   controller disarms. Recovery with a linear PD is impossible beyond ~60°.
const float POLE_FAIL_RAD = 1.3f;

// START_ANGLE: how close to upright (in radians) the pole must be before
//   the controller will arm. User must hold the pole within this window.
//   2° in radians ≈ 0.035 rad.
const float START_ANGLE = 2.0f * PI / 180.0f;

// ===================== MOTOR =====================
// MAX_PWM: maximum PWM value sent to the H-bridge (0–255).
//   255 = full 12V to motor. Currently uncapped; adjust if motor runs hot.
const int MAX_PWM = 255;

// FORCE_TO_PWM: converts a force command in Newtons to a PWM integer.
//   Tune this so that a 1 N command produces a physically reasonable
//   cart acceleration. Too high → cart slams end-stops. Too low → sluggish.
const float FORCE_TO_PWM = 22.0f;

// PWM_DEADBAND: minimum PWM added to every nonzero command to overcome
//   static friction / stiction in the belt drive. The motor won't move at
//   very low PWM, so we add this floor after computing the commanded value.
const int PWM_DEADBAND = 18;

// PWM_BIAS: small asymmetry correction. The BTS7960 or motor may have
//   slightly different forward vs backward efficiency. This adds a fixed
//   offset to the forward direction and subtracts it from backward.
const int PWM_BIAS = 8;

// FORCE_DEADBAND: if the commanded force magnitude is below this (Newtons),
//   just stop the motor instead. Avoids buzzing the motor near zero.
const float FORCE_DEADBAND = 0.2f;

// ===================== STATE =====================
// Encoder counts — modified only inside ISRs, so declared volatile.
// The main loop reads these atomically with interrupts disabled.
volatile long cart_counts = 0;
volatile long pole_counts = 0;

// X4 quadrature decoder lookup table.
// Index = (prev_AB << 2) | new_AB (4-bit value, 0–15).
// Value = direction: +1 forward, -1 backward, 0 no-change transition.
// This is faster and more reliable than if/else chains in the ISR.
const int8_t QDEC_TABLE[16] = {
   0, +1, -1,  0,
  -1,  0,  0, +1,
  +1,  0,  0, -1,
   0, -1, +1,  0
};

// Previous AB state for each encoder — used by the quadrature decoder.
volatile uint8_t cart_ab_prev = 0;
volatile uint8_t pole_ab_prev = 0;

// Encoder count snapshots taken at the moment of zeroing (GO or arm event).
// Positions are computed relative to these offsets so the controller
// always sees zero at the point it was zeroed.
float cart_offset = 0;
float pole_offset = 0;

// Current estimated position in SI units (meters and radians).
float cart_pos   = 0;
float pole_angle = 0;

// Velocity estimates (differentiated from position each control tick).
// Suffix _f is for historical reasons; these are not filtered beyond
// the implicit low-pass effect of the finite-difference at 500 Hz.
float cart_vel_f = 0;
float pole_vel_f = 0;

// ===================== CONTROL =====================
// armed: true when the controller is actively driving the motor.
//   Set to true when pole is within START_ANGLE of upright.
//   Cleared by safety() on end-stop or pole-fallen events.
bool armed = false;

// calibrating: true during the startup motor bias estimation routine.
//   Blocks normal control until the calibration is complete.
bool calibrating = true;

// calib_step: counts 500 Hz ticks during calibration.
int calib_step = 0;

// bias_accum: accumulates cart velocity during calibration to estimate
//   the average velocity drift caused by mechanical asymmetry.
float bias_accum = 0;

// cart_force_bias: learned offset (Newtons) added to every force command
//   to compensate for belt/motor asymmetry. Computed during calibration.
float cart_force_bias = 0.0f;

// pole_setpoint: the angle (radians) the pole controller tries to reach.
//   Normally near zero (upright). The cart centering controller shifts
//   this slightly to steer the cart back toward center.
float pole_setpoint = 0;

// ===================== GAINS =====================
// Pole inner-loop PD gains.
//   kp_pole: proportional gain on pole angle error (rad → N)
//   ki_pole: integral gain on pole angle error (currently unused, set to 0)
//   pole_damp: derivative gain — directly multiplies pole angular velocity
float kp_pole = 800.0;
float ki_pole = 0.0;
float pole_damp = 3.1f;

// Cart outer-loop PI gains.
//   kp_cart: proportional gain on cart position error (m → rad setpoint offset)
//   ki_cart: integral gain on cart position error (currently unused)
float kp_cart = 0.0009;
float ki_cart = 0.0000;

// cart_lean_limit: maximum angle offset (radians) the cart controller can
//   request from the pole controller. Prevents over-leaning.
float cart_lean_limit = 0.12f;

// ===================== MODE =====================
// recovery_mode: engaged when the cart is too close to the rail end-stop.
//   Uses an aggressive cubic restoring law instead of the normal PI to
//   snap the cart back toward center before the end-stop safety trips.
bool recovery_mode = false;

// Thresholds for entering and exiting recovery mode (as a fraction of
// CART_LIMIT_M). 
const float RECOVERY_ENTER = 0.27f * CART_LIMIT_M;  // enter if |pos| > this
const float RECOVERY_EXIT  = 0.22f * CART_LIMIT_M;  // exit  if |pos| < this

// ===================== INTEGRATORS =====================
// Accumulated integral terms for the pole and cart PI loops.
// Reset on arm, disarm, and mode switches to prevent integrator windup
// from carrying over across state transitions.
float i_pole = 0;
float i_cart = 0;

// ===================== MOTOR =====================

/*
 * motorStop()
 * Sets both PWM outputs to zero, coasting the motor to a stop.
 * Called by safety(), disarm events, and during calibration pauses.
 */
void motorStop() {
  analogWrite(PIN_RPWM, 0);
  analogWrite(PIN_LPWM, 0);
}

/*
 * applyForce(f)
 * Converts a signed force command (Newtons) to a PWM signal on the H-bridge.
 *
 * Pipeline:
 *   1. Add the learned mechanical bias correction.
 *   2. If magnitude is below FORCE_DEADBAND, stop motor (avoids buzzing).
 *   3. Scale force to PWM using FORCE_TO_PWM.
 *   4. Add PWM_DEADBAND to overcome stiction.
 *   5. Add/subtract PWM_BIAS to correct forward/backward asymmetry.
 *   6. Clamp to [0, MAX_PWM] and write to the appropriate PWM pin.
 *
 * Positive f → forward (RPWM). Negative f → backward (LPWM).
 */
void applyForce(float f) {

  //  apply learned force bias correction
  f += cart_force_bias;

  if (fabs(f) < FORCE_DEADBAND) {
    motorStop();
    return;
  }

  int pwm = (int)(fabs(f) * FORCE_TO_PWM);
  pwm += PWM_DEADBAND;

  if (f > 0) pwm += PWM_BIAS;
  else       pwm -= PWM_BIAS;

  pwm = constrain(pwm, 0, MAX_PWM);

  if (f > 0) {
    analogWrite(PIN_RPWM, pwm);
    analogWrite(PIN_LPWM, 0);
  } else {
    analogWrite(PIN_RPWM, 0);
    analogWrite(PIN_LPWM, pwm);
  }
}

// ===================== ENCODERS =====================

/*
 * cartUpdate() / poleUpdate()
 * X4 quadrature decoder — called from every encoder ISR (both INT and PCINT).
 * Reads both A and B channels, builds a 4-bit index from (prev_AB, new_AB),
 * and looks up the direction in QDEC_TABLE. Updates the count accordingly.
 *
 * The pole count is negated (note the minus sign) to match the physical
 * sign convention: positive pole angle = tipping toward the +x (motor) end.
 *
 * Why a lookup table?
 *   ISRs must be fast. The table collapses the direction logic to a single
 *   array index + read, avoiding branches. It also correctly handles all
 *   16 AB transition combinations, including illegal glitches (→ 0 delta).
 */
static inline void cartUpdate() {
  uint8_t a = digitalRead(PIN_CART_ENC_A);
  uint8_t b = digitalRead(PIN_CART_ENC_B);
  uint8_t ab_new = (a << 1) | b;
  uint8_t idx = (cart_ab_prev << 2) | ab_new;
  cart_counts += QDEC_TABLE[idx];
  cart_ab_prev = ab_new;
}

static inline void poleUpdate() {
  uint8_t a = digitalRead(PIN_POLE_ENC_A);
  uint8_t b = digitalRead(PIN_POLE_ENC_B);
  uint8_t ab_new = (a << 1) | b;
  uint8_t idx = (pole_ab_prev << 2) | ab_new;
  pole_counts += -QDEC_TABLE[idx];  // negated: positive = tip toward +x
  pole_ab_prev = ab_new;
}

// INT0 fires on every CHANGE of cart encoder A channel.
void cartISR_A() { cartUpdate(); }

// INT1 fires on every CHANGE of pole encoder A channel.
void poleISR_A() { poleUpdate(); }

/*
 * ISR(PCINT0_vect)
 * Pin-change interrupt for PORTB (pins 8 and 9 = encoder B channels).
 * Fires whenever either B channel changes, giving us the 4th edge per
 * quadrature cycle (hence X4 decoding). Both encoders share this ISR,
 * so we update both every time it fires.
 */
ISR(PCINT0_vect) {
  cartUpdate();
  poleUpdate();
}

// ===================== STATE =====================

/*
 * updateState()
 * Called every control tick (500 Hz). Atomically snapshots encoder counts,
 * converts to SI units using the offset established at arm time, and
 * estimates velocities by finite difference over one DT period.
 *
 * Velocity estimation is simple first-difference: v = Δx / DT.
 * At 500 Hz with these encoder resolutions, this is clean enough without
 * additional filtering.
 */
void updateState() {
  long cc, pc;

  // Disable interrupts for the minimum time needed to read two longs.
  // On AVR (8-bit), reading a 4-byte long is not atomic by default.
  noInterrupts();
  cc = cart_counts;
  pc = pole_counts;
  interrupts();

  float new_cart = (cc - cart_offset) * CART_M_PER_COUNT;
  float new_pole = (pc - pole_offset) * POLE_RAD_PER_COUNT;

  cart_vel_f = (new_cart - cart_pos) / DT;
  pole_vel_f = (new_pole - pole_angle) / DT;

  cart_pos   = new_cart;
  pole_angle = new_pole;
}

// ===================== CART CONTROL =====================

/*
 * cartControl(x)
 * Outer-loop cart centering controller. Returns an angle offset (radians)
 * that is fed to the pole controller as a setpoint correction.
 *
 * Normal mode (|pos| < RECOVERY_ENTER):
 *   Standard PI on position error. The integral term is bled off at high
 *   cart velocities to prevent windup during fast transients.
 *   A small deadband (5 mm) suppresses noise-driven integrator drift near
 *   center.
 *
 * Recovery mode (|pos| > RECOVERY_ENTER):
 *   Uses an aggressive cubic restoring law:
 *     u = 0.020*e - 0.135*vel + 14.0*e³
 *   The cubic term makes the correction force grow rapidly as the cart
 *   approaches the end-stop, snapping it back to center quickly.
 *
 * Note: error is defined as e = -x so that a positive cart position (cart
 * has moved in +x direction) produces a negative correction (lean toward
 * -x to drive it back).
 */
float cartControl(float x) {

  float e = -x;

  if (recovery_mode) {
    float u = 0.020f * e - 0.135f * cart_vel_f + 14.0f * e * e * e;
    return u;
  }

  if (fabs(e) < 0.005f) return 0;

  if (fabs(cart_vel_f) > 0.4f) {
    i_cart *= 0.9f;  // bleed integrator during fast motion to prevent windup
  }

  i_cart += e * DT;
  i_cart = constrain(i_cart, -1.5f, 1.5f);  // anti-windup clamp

  return kp_cart * e + ki_cart * i_cart;
}

// ===================== POLE CONTROL =====================

/*
 * poleControl(setpt, x, vel)
 * Inner-loop pole balancing controller. Returns a force command (Newtons).
 *
 *   setpt: desired pole angle (rad) — normally ~PI plus cart correction
 *   x:     current pole angle (rad)
 *   vel:   current pole angular velocity (rad/s)
 *
 * Implements a PD controller:
 *   u = kp_pole * (setpt - x)  -  pole_damp * vel
 *
 * The D term (velocity damping) is applied directly to pole_vel_f rather
 * than to the derivative of the error, which is equivalent here since the
 * setpoint changes slowly relative to the 500 Hz loop.
 */
float poleControl(float setpt, float x, float vel) {

  float e = setpt - x;

  float P = kp_pole * e;
  float D = -pole_damp * vel;  // negative: opposes angular velocity

  return P + D;
}

// ===================== SAFETY =====================

/*
 * safety(pole_u)
 * Checks two hardware protection conditions every control tick:
 *   1. Cart end-stop: |cart_pos| > CART_LIMIT_M
 *   2. Pole fallen:   |pole_angle_error| > POLE_FAIL_RAD (~75°)
 *
 * If either triggers: motor stops, controller disarms, returns false.
 * The main loop checks this return value and skips control output if false.
 * To restart after a safety trip, bring the pole upright — armed will reset
 * in the arm-check block.
 *
 * Note: the argument is pole_u (= pole_angle - PI), the angle error from
 * upright, not the raw encoder angle. This keeps the check symmetric around
 * the balance point regardless of encoder offset.
 */
bool safety(float pole_u) {

  if (fabs(cart_pos) > CART_LIMIT_M) {
    motorStop();
    armed = false;
    return false;
  }

  if (fabs(pole_u) > POLE_FAIL_RAD) {
    motorStop();
    armed = false;
    return false;
  }

  return true;
}

// ===================== TIMER =====================

// Flag set by Timer1 ISR, cleared in loop(). Drives the 500 Hz control tick.
volatile bool tick = false;

/*
 * ISR(TIMER1_COMPA_vect)
 * Timer1 compare-match interrupt, fires at 500 Hz.
 * Only sets the tick flag — all real work is done in loop() to keep the
 * ISR as short as possible and avoid re-entrancy issues.
 */
ISR(TIMER1_COMPA_vect) {
  tick = true;
}

/*
 * setupTimer()
 * Configures Timer1 for CTC mode at 500 Hz.
 *
 * Formula:  OCR1A = (F_CPU / (prescaler × frequency)) - 1
 *                 = (16,000,000 / (8 × 500)) - 1
 *                 = 3999
 *
 * Timer1 is a 16-bit timer so 3999 fits comfortably. Timer1 is NOT used
 * by analogWrite (that uses Timer0 and Timer2 on the Uno), so no conflict.
 * Avoid using the Servo library in this sketch — it also uses Timer1.
 */
void setupTimer() {
  noInterrupts();
  TCCR1A = 0;                    // clear control register A (no PWM output)
  TCCR1B = 0;                    // clear control register B
  OCR1A = 3999;                  // compare match value for 500 Hz
  TCCR1B |= (1 << WGM12);       // CTC mode: reset counter on compare match
  TCCR1B |= (1 << CS11);        // prescaler = 8
  TIMSK1 |= (1 << OCIE1A);      // enable compare match A interrupt
  interrupts();
}

// ===================== SETUP =====================

/*
 * setup()
 * Runs once on power-on or reset.
 *
 * 1. Configure motor driver pins and enable H-bridge.
 * 2. Configure encoder pins with pull-ups (safe for both push-pull and
 *    open-collector encoder outputs).
 * 3. Initialize quadrature decoder state from current pin levels so the
 *    first ISR fires with a valid prev_AB state.
 * 4. Attach INT0/INT1 interrupts for encoder A channels (CHANGE = both edges).
 * 5. Enable PCINT on PORTB for encoder B channels (pins 8 and 9).
 * 6. Start Timer1 at 500 Hz.
 * 7. Wait 1 second for everything to settle, then snapshot encoder offsets.
 * 8. Start the calibration routine (runs in loop() via the calibrating flag).
 */
void setup() {

  // Motor driver output pins
  pinMode(PIN_RPWM, OUTPUT);
  pinMode(PIN_LPWM, OUTPUT);
  pinMode(PIN_R_EN, OUTPUT);
  pinMode(PIN_L_EN, OUTPUT);

  // Enable both H-bridge half-bridges permanently
  digitalWrite(PIN_R_EN, HIGH);
  digitalWrite(PIN_L_EN, HIGH);

  // Encoder input pins with internal pull-ups
  pinMode(PIN_CART_ENC_A, INPUT_PULLUP);
  pinMode(PIN_CART_ENC_B, INPUT_PULLUP);
  pinMode(PIN_POLE_ENC_A, INPUT_PULLUP);
  pinMode(PIN_POLE_ENC_B, INPUT_PULLUP);

  // Initialize decoder state from current pin levels so the first edge
  // decoded has a valid "previous" state to compare against.
  cart_ab_prev = (digitalRead(PIN_CART_ENC_A) << 1) | digitalRead(PIN_CART_ENC_B);
  pole_ab_prev = (digitalRead(PIN_POLE_ENC_A) << 1) | digitalRead(PIN_POLE_ENC_B);

  // INT0 (pin 2) and INT1 (pin 3): A-channel interrupts, fire on any edge
  attachInterrupt(digitalPinToInterrupt(PIN_CART_ENC_A), cartISR_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_POLE_ENC_A), poleISR_A, CHANGE);

  // PCINT on PORTB: enables pin-change interrupts for pins 8 (PCINT0)
  // and 9 (PCINT1) — the B channels of each encoder.
  PCICR  |= (1 << PCIE0);
  PCMSK0 |= (1 << PCINT0) | (1 << PCINT1);

  setupTimer();

  delay(1000);  // settle time: let encoder signals stabilize before zeroing

  // Record current encoder counts as the position reference (zero point).
  noInterrupts();
  cart_offset = cart_counts;
  pole_offset = pole_counts;
  interrupts();

  // Begin the motor calibration routine
  calibrating = true;
  calib_step  = 0;
  bias_accum  = 0;
}

// ===================== LOOP =====================

/*
 * loop()
 * Main execution loop. Returns immediately unless the 500 Hz Timer1 tick
 * has fired, keeping CPU usage low between control steps.
 *
 * Each tick runs the following pipeline:
 *
 * [CALIBRATION] (first ~0.8 s after power-on)
 *   Alternates small forward/backward force pulses and accumulates the
 *   resulting cart velocity. A net drift indicates mechanical asymmetry
 *   (belt tension, motor winding imbalance, etc.). The average drift is
 *   used to compute cart_force_bias, which is added to every subsequent
 *   force command in applyForce().
 *
 * [ARM CHECK]
 *   Once calibration is done, waits until the pole is held within
 *   START_ANGLE (2°) of upright. Arms the controller and zeros the
 *   cart position reference at that moment.
 *
 * [SAFETY CHECK]
 *   Checks cart end-stop and pole-fallen conditions. Disarms and stops
 *   motor if either is triggered.
 *
 * [MODE SWITCH]
 *   Transitions between normal (PI cart centering) and recovery (cubic
 *   restoring force) modes based on cart distance from center, with
 *   hysteresis to prevent chattering.
 *
 * [CONTROL]
 *   1. Outer loop: cartControl() computes a pole setpoint offset.
 *   2. The offset is clamped to ±cart_lean_limit and added to the
 *      nominal upright setpoint (0 rad).
 *   3. Inner loop: poleControl() computes the force command.
 *   4. applyForce() sends the command to the H-bridge.
 */
void loop() {

  if (!tick) return;   // wait for 500 Hz timer tick
  tick = false;

  updateState();  // read encoders, compute position and velocity

  // pole_u: pole angle error from upright.
  // The encoder is zeroed with the pole hanging DOWN (natural rest position),
  // so the upright position corresponds to π radians. Subtracting π gives
  // an error that is 0 when perfectly upright, positive when tipping in +x.
  float pole_u = pole_angle - PI;

  // ===================== CALIBRATION =====================
  if (calibrating) {

    motorStop();

    // Alternate between small positive and negative force pulses.
    // If the cart consistently drifts in one direction under balanced
    // forcing, that reveals the mechanical bias.
    float test_force = (calib_step % 2 == 0) ? 1.5f : -1.5f;
    applyForce(test_force);

    bias_accum += cart_vel_f;  // accumulate velocity to detect net drift
    calib_step++;

    if (calib_step > 400) {  // 400 ticks × 2 ms = 0.8 s calibration window

      float avg = bias_accum / calib_step;

      // The bias correction opposes the average drift, scaled by 2× for
      // a stronger correction (empirically tuned factor).
      cart_force_bias = -avg * 2.0f;

      calibrating = false;

      // Reset integrators so no accumulated error carries into control
      i_cart = 0;
      i_pole = 0;

      motorStop();
    }

    return;  // skip rest of loop during calibration
  }

  // ===================== ARM =====================
  // Wait for the user to hold the pole upright before enabling the motor.
  // Once armed, zero the cart position so the controller targets the
  // current cart location as "home."
  if (!armed) {
    if (fabs(pole_u) < START_ANGLE) {
      armed  = true;
      i_cart = 0;
      i_pole = 0;
      cart_pos = 0;  // re-zero cart reference at arm point
    } else {
      motorStop();
      return;
    }
  }

  if (!safety(pole_u)) return;  // disarm and stop if end-stop or pole fallen

  // ===================== MODE SWITCH =====================
  // Switch to recovery mode if the cart is getting too close to the end-stop.
  // Hysteresis (RECOVERY_ENTER > RECOVERY_EXIT) prevents rapid toggling.
  float dist = fabs(cart_pos);

  if (!recovery_mode && dist > RECOVERY_ENTER) {
    recovery_mode = true;
    i_cart = 0;  // clear integrator to avoid windup spike on mode entry
  }

  if (recovery_mode && dist < RECOVERY_EXIT) {
    recovery_mode = false;
    i_cart = 0;
  }

  // ===================== CONTROL =====================
  // Outer loop: compute a lean-angle correction to steer the cart to center.
  float cart_cmd = cartControl(cart_pos);

  // Convert cart correction to pole setpoint offset (negative: lean opposite
  // to the error direction to push the cart back). Clamp to safe lean range.
  pole_setpoint = -cart_cmd;
  pole_setpoint = constrain(pole_setpoint, -cart_lean_limit, cart_lean_limit);

  // Inner loop: compute force to drive pole to setpoint.
  float u = poleControl(pole_setpoint, pole_u, pole_vel_f);

  // Send force command to motor driver.
  applyForce(u);
}
