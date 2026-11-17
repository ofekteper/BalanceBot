# BalanceBot simulation: progress notes, 17 Nov 2026

MuJoCo simulation of the 3-servo ball-and-plate platform (BaBot design). All code is in
`testing_MuJoCo/v2_cad/`.

## How to run

```
cd testing_MuJoCo/v2_cad
py run_sim.py      # viewer + control panel; starts in Balance mode
py diagnose.py     # headless tests: joints, tilt accuracy, IR sensors, firmware balancing
```

## Files

| File | What it does |
|---|---|
| `ballplate_model.py` | Platform model (geometry from CAD), inverse kinematics, `ServoCommander` (servo speed limit), joint gap/load checks |
| `ir_sensors.py` | Simulated IR sensor PCB plus the firmware's sensing pipeline |
| `babot_firmware.py` | Python port of `BaBot PID/main.ino` (PID, IK, servo output, timing) |
| `run_sim.py` | MuJoCo viewer plus tkinter control panel |
| `diagnose.py` | Headless test suite |

## What was done today

### 1. Plate "disconnecting" from the joints
- **The physics joint never actually opened.** Gaps stayed under 0.3 mm even with ±45° servo jumps.
- **Real cause: the model geometry didn't match the CAD.** The rod length was a guess, and the holder was drawn 5.5 mm too far inward and 3 mm too high, so on screen the plate looked detached.
- **Fix: geometry is now measured from the STL files** by fitting the actual holes and surfaces:
  - servo shaft at r = 19.00 mm, z = 23.63 mm
  - elbow pin at r = 62.30 mm, z = 48.63 mm
  - crank 50.0 mm at 30°
  - Ø12 mm steel magnetic ball between the arm tip and the holder
- **The real STL parts are drawn in the viewer** (base, both arm parts, holders). The physics still uses simple shapes.
- **Stiffer joint constraint:** the gap is now about 0.002 mm.
- **New `joint_loads()`** reports the force along each rod. A real magnet lets go when the plate pulls on it, so the panel warns when the pull exceeds `MAGNET_PULL_N`. That value is a 2 N guess and needs measuring.

### 2. Servo control
- **Control panel modes:** Balance (firmware), Plate tilt (IK), Servos directly, Circle demo. All sliders move in 0.1° steps.
- **Small tilts are accurate:** commanding 0.2° gives 0.20° measured, which needs about 0.3° at the servo.
- **Servo range** follows the firmware: −10°…85° crank angle above horizontal.

### 3. Spec decisions
- **Upper arm = 39.2 mm** (spec and firmware), not the 40.9 mm from the CAD. Side effect: in the viewer the steel ball overlaps the printed arm tip by about 1.7 mm. To switch back, change `ROD_LENGTH` in `ballplate_model.py`.
- **Neutral height:** the firmware's 65 mm is measured from the servo-shaft plane to the ball-joint plane.

### 4. IR sensing PCB
Sources: the Gerber zip (pick-and-place and netlist from `FlyingProbeTesting.json`), the BOM, and the datasheets.
- **Board:** Ø100 mm with 3 tabs at 90°, 210° and 330°. The tabs clip into slots in the holders' inner tabs (slot z = 93.3–95.2 mm), with the PMMA plate on top of the holders. The gap between the PCB and the PMMA is 9.25 mm.
- **LEDs:** 21 × Everlight IR204/H60, 3 mm, ±25° beam. They sit on a 21.34 mm grid, 5×5 with the corners removed.
- **Sensors:** 16 × Würth 1540031NC6090 phototransistors, ±30°, in a 4×4 grid. Q1 is top-left (−X, +Y), numbered row by row.
- **Other parts:** an AO3400A MOSFET switches all the LEDs. U1 is a **20-pin FPC connector**; the multiplexer (CD74HC4067) is on the main board.
- **Optics model:** each LED beam (with a little side light) reflects off the ball, partly diffusely and partly like a mirror. The PMMA surfaces add a constant background. The brightest sensor is the one directly under the ball, as the design description says.
- **Firmware pipeline, copied exactly:**
  - LEDs off / on subtraction to remove ambient light
  - smoothing with α = 0.7
  - ball present when max − min ≥ 350 counts
  - cubic weighted centroid with mirrored edges
  - 30 Hz
- **Viewer:** the phototransistors light up with their reading, and a green disc shows the estimated ball position. The panel shows a live 4×4 heat map.

### 5. Firmware PID port (`main.ino` → `babot_firmware.py`)
- **Ported:**
  - `handleOnMode` (2-cycle skip, 1 s ball-lost hold)
  - `pidControl` (including the soft derivative deadband)
  - output smoothing (α = 0.9)
  - `movePlatform` (the firmware's own IK)
  - `moveServos` (clamp, whole degrees)
  - about 20 ms sensor-to-servo latency
- **Firmware values** (they differ from the spec): **30 Hz**, P = 3.0, **I = 0.04, D = 25**. The spec says 40 Hz, I = 0.02 and D = 30.
- **Frame mapping:** firmware X = servo-0 direction, firmware Y = PCB +X. This is mirrored relative to the PCB, and it's the only mapping where the firmware's signs balance the ball. Firmware legs 0/1/2 map to sim legs 0/2/1.
- **Panel:** live P/I/D sliders, setpoint in mm, firmware ON/OFF and a "Kick ball" button.

## Key results (`diagnose.py`)

| Test | Result |
|---|---|
| Joint gap, all scenarios | ≤ 0.06 mm |
| Static tilt accuracy | within 0.03° up to 8° |
| IR position error, ping-pong ball | 7.5 mm mean inside the grid, 4.4 mm outside, no false detections |
| 20 mm plastic or steel ball | never detected (peaks 300 and 170 counts are below the 350 threshold) |
| Firmware balancing, drops up to 46 mm off-centre | settles within 2.4 mm in about 1.5 s |
| Setpoint (15, 10) mm | ends 5–8 mm short |
| Kick of 0.18 m/s | ball moves 17 mm, recovers within 3 s |

## Findings worth noting
- **Centroid bias:** cubing the weights pulls the firmware's position estimate toward the nearest sensor. Error is low at sensor positions and up to about 10 mm between them, which is why off-centre setpoints are missed.
- **Coverage:** the sensors span ±32 mm, but the plate is Ø140 mm. The mirrored edges let the estimate reach a little beyond the grid.
- **Ball choice:** the firmware only works with the ping-pong ball, because smaller balls don't reach the 350-count threshold.
- **Realism:** the simulation is probably more forgiving than the real robot. It has no servo backlash, plate flex or calibration error, and the optics are partly estimated.

## Open items / next steps
- [ ] **Verify the sensor channel order on the real board.** The FPC cable is reverse-direction, and the sim assumes mux channel i = Q(i+1). Enable `sendSerialData()` in `main.ino` and move the ball over one known sensor.
- [ ] **Calibrate the sensor model** against the real serial output (steps at the bottom of `ir_sensors.py`): `ADC_REF_SIGNAL`, `LED_SIDE_LIGHT`, `EMIT_HEIGHT`, `BALL_OPTICS`.
- [ ] **Measure the magnet pull-off force** and set `MAGNET_PULL_N`.
- [ ] **Replace the mass guesses** (PCB, plate size, print infill) with weighed values.
- [ ] **Decide the arm length:** 39.2 mm spec or 40.9 mm CAD. Check the printed part.
- [ ] **Model the real servo's imperfections** (backlash, deadband, calibration offsets) to make the PID tuning results more realistic.
- [ ] **Possible improvement to test in the sim:** a better position estimate than the cubic centroid, to remove the bias between sensors.

## Housekeeping
- **Duplicate folder:** there's an older copy at `testing MuJoCo/v2_cad` (with a space). All current work is in `testing_MuJoCo/v2_cad`, so the old copy can be deleted once checked.
- **Nothing committed yet:** none of today's work is in git.
