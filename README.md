# Inverted Pendulum Control Project

A cart-pole inverted pendulum stabilized by an LQR controller.
- Design in MuJoCo (PC)
- Supervision and logging on a Raspberry Pi 3B
- Real-time control on an Arduino Uno R3

## Architecture
PC (design) ──K──▶ Raspberry Pi 3B (supervisor, Python)
│ USB serial
▼
Arduino Uno R3 (real-time C)
│
▼
H-bridge → DC motor → belt → cart
+ encoder on belt return pulley
+ encoder on pole pivot

## Repository layout

- `sim/` — MuJoCo design tool. Produces the LQR gain K.
- `supervisor/` — Pi-side Python: serial protocol, telemetry logging, web dashboard.
- `firmware/` — Arduino sketch (real-time control loop).
- `logs/` — run telemetry (gitignored).
- `docs/` — writeups, wiring diagrams, photos.

## Status

- [x] MuJoCo simulation and LQR design
- [ ] Pi ↔ Arduino serial protocol (with fake Arduino)
- [ ] Telemetry logging
- [ ] Web dashboard
- [ ] Physical cart assembled
- [ ] First balance