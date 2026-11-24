# Mochi deformable tasks

Four batched RL tasks in which two arms handle a soft body: a foam sponge, a terry towel, a cable and a T-shirt. They
run on the Mochi solver, Eden's host of the Genesis port of Meta's SuperDex implicit solver. Read this page to train
or replay them, and for what Mochi changes about contact-rich deformable manipulation. The code is in
`eden/tasks/mochi/`.

Three of the tasks are Eden's versions of RoboSimStudio's three demos (`examples/08_demos` in that project), built at
each demo's numbers on its bimanual i2RT YAM workcell and scored with the demo's own goal. The fourth,
`mochi_fold_the_shirt`, is Eden's version of ClothDojo's `shirt_fold` benchmark task for one garment, on the same
workcell and scored with ClothDojo's fold quality. Every env carries its own soft body in one Mochi solve.

<div class="grid" markdown>

![mochi_wipe_the_bar: a trained policy wipes a marker stripe off a grab bar with a foam sponge](../assets/readme/mochi_wipe.webp)

![mochi_fold_the_towel: two arms fold a towel in half](../assets/readme/mochi_fold.webp)

![mochi_route_the_cable: a cable routed round two spools into a port](../assets/readme/mochi_route.webp)

</div>

The README labels the wiping clip as a trained policy and the folding and routing clips as scripted runs.

## The tasks

| Task (`--task`) | Demo | Soft body | Goal (success) | Episode |
|-----------------|------|-----------|----------------|---------|
| `mochi_wipe_the_bar` | `wipe_the_bar` (`surface_prep`) | 50 × 75 × 30 mm `FIRM_FOAM` sponge (E 80 kPa) | 90 % of a 243-cell marker stripe on a grab bar's crown worn clean | 15 s |
| `mochi_fold_the_towel` | `fold_the_towel` (`cloth_fold` / `towel_half`) | 30 cm terry towel, 500 g/m², 30 × 30 cells | fold score ≥ 0.85, fold extent ≤ 0.6, the towel at rest, both hands clear | 12 s |
| `mochi_route_the_cable` | `route_the_cable` (VABAR `cable_route/port_easy`) | 600 mm, 7 mm cable of 50 segments, welded at one end | two segments in each spool's seat, three in the port with the free end, hands clear, at rest | 20 s |
| `mochi_fold_the_shirt` | ClothDojo `shirt_fold`, garment `rand_ZZCWITH5HY` | GarmentCodeData T-shirt at 0.7 of its size, 6888 vertices, 180 g/m², 0.55 × 0.36 m flat | ClothDojo fold quality ≥ 0.142, the shirt at rest, both hands clear | 40 s |

```bash
eden smoke --task mochi_wipe_the_bar
eden train --task mochi_wipe_the_bar        # 256 envs, default fp32 precision
```

Each task trains with PPO at 30 Hz control: one 1/30 s Mochi step (`decimation=1`) with 4 Newton iterations
(`MochiOptions(n_newton_iterations=4)`), 256 envs (128 for the shirt, whose Mochi state is about 32 MiB an env).

### `mochi_wipe_the_bar`

A 25 mm chrome tube with a 340 mm straight run on two 32 mm elbows, its legs 50 mm into 76 × 80 × 70 mm blocks on a
10 mm mat; the sponge lies on the mat 70 mm past the near block. The stripe is the middle 200 mm of the bar's crown,
9 degrees either side: 243 cells of 2.5 mm, each wearing by Preston's law with the demo's constants
(`custom_terms.wipe.BarStripeWipe`).

The right arm does the job: it takes the sponge off the mat with its jaws across the 50 mm side, carries it over the
near block and strokes the stripe. Its jaws squeeze at most 8 N a finger, the demo's cap. Once the grip has settled,
the demo's latch (`SettledGraspLatch`) pins the foam under the pads to the hand until the jaws open. The left arm
stands at its ready posture. The episode ends on success.

### `mochi_fold_the_towel`

The towel lies flat at `(-0.05, +0.06)` on the bench; the crease is its centreline across y, and the far half is
carried over onto the near half. The fold is scored with the cloth suite's measures (`custom_terms.fold`:
`ClothFoldScore`, SoftGym's ClothFold with a drag term and one layer's 7.5 mm forgiven, and `ClothFoldExtent`).
"At rest" means a 95th-percentile particle speed under 0.02 m/s; "hands clear" means no particle within 2 cm of a
tool centre.

The left arm takes the far edge and carries it over. The right arm, its jaws shut, presses the near half to the
bench so the flap does not drag it along.

### `mochi_route_the_cable`

The cable is welded under an anchor bracket and lies along +y. Two ManipulationNet F1 spools stand 240 mm apart
(a 4.75 mm shaft between 12.5 mm flanges); the cable must pass the first on the far side and the second on the
robot's side. A port 61 mm out from the second spool is a 10 mm channel between 12.5 mm walls, facing the run. The
goal is the board's `routed` (`custom_terms.route`), counted on the rod's segments; "at rest" means no node faster
than 50 mm/s and "hands clear" more than 50 mm from the cable.

Both arms act: the right makes the three placements and the left plugs the end. The jaws hold the cable by
friction at the demo's 6000 N/m servo.

### `mochi_fold_the_shirt`

ClothDojo (clothdojo.github.io, arXiv 2611.01190) benchmarks bimanual T-shirt flattening and folding on Newton's
Style3D cloth with two AgileX Piper arms. This task takes its `shirt_fold` task for one garment, `rand_ZZCWITH5HY`:
each episode starts from one of the garment's four benchmark fold starts (the shirt settled flat, sleeves out, at
0, 90, 180 and 270 degrees). Both arms move their tool centres by differential IK, turn them about the vertical, and
close with `FeatureGraspLatch`, which seizes at most 32 vertices within 1.5 cm of the tool centre, close to
ClothDojo's own particle grasp.

The garment is ClothDojo's at 0.7 of its size, laid 0.19 m nearer the arms than ClothDojo lays it. The YAMs are
smaller arms than ClothDojo's Pipers: at 3 cm over the bench each reaches its own half of the bench to about 0.10 m
past the middle, and only to about 2 cm short of the centre line. At full size and ClothDojo's place even a half
fold, hem to shoulders, needs the hem carried from x = -0.14 m to +0.36 m. The reset and the metric scale the
benchmark's starts and flat reference in the plane (`planar_scale`), heights kept, so the layers stay as thick as
before. That makes the same cloth relatively thicker: a flat start scores -1.62 here against ClothDojo's -1.39.
The hands' turn is free (`orientation_scale=[0, 0, 0.1]`, as in `mochi_route_the_cable`): with it held, the wrist
rolls against every turn of the base and its ±2.09 rad stops end the reach well short of the middle.

Success is ClothDojo's fold quality Q (`custom_terms.shirt.ShirtFoldQuality`) at or above its threshold, 0.142,
with the shirt at rest and both hands clear. Q is a fixed linear model over five geometric features of the vertex
positions: footprint compression, flatness, how much of the top is one planar rectangle, layer overlap and
closeness to the table. ClothDojo's flat shirt scores about -1.4 and its expert folds +0.3 to +1.0. The port
reproduces ClothDojo's NumPy and OpenCV implementation exactly (the top-down raster runs batched on the GPU), and
costs about 10 ms per env, so the metric is scored every 5 control steps and held in between. ClothDojo also asks
Q to hold for 3 s; "at rest and let go" stands in for that hold, as for the towel.

The threshold was checked on Mochi rather than refitted. Eight frames of ClothDojo's published expert fold, scaled
like the task's starts and laid into its scene, were settled for 4 s. Seven settled at Q +0.19 to +1.14, all above
the threshold (change -0.09 to +0.54, mean +0.21), so it is kept; one frame's solve diverged.
`examples/recipes/mochi/fold_the_shirt_threshold.py` reruns the measurement.

The task's scripted demonstrator, `ShirtFolder` (`eden/tasks/mochi/scripted_demos.py`), folds the sleeves in and
then the length in thirds with both YAMs, through the task's own actions. `eden datagen` records it from the two
starts with the sleeves across the bench; the far sleeve of the other two is out of reach:

```bash
eden datagen --task mochi_fold_the_shirt --num_envs 16 \
    --override 'event_options: {fold_start: {state_indices: [1, 3]}}'
```

It does not reach the task's success: its final fold quality runs about -0.9 to -2.3, held back by a lumpy top
(the planar-top feature) and by sleeve folds that sometimes slip. `eden datagen --show_viewer` shows it as it records,
and `examples/recipes/mochi/fold_the_shirt_nyx.py` path traces it in LIBERO's kitchen.

The garment's files are converted from ClothDojo's Hub dataset by `examples/recipes/mochi/convert_clothdojo_shirt.py`
into `objects/clothdojo/rand_ZZCWITH5HY/` of Eden's asset dataset: the mesh (`shirt.obj`, in the USD's vertex order)
and the four starts with the flat reference (`fold_starts.npz`, in ClothDojo's world frame).

### Rewards and terminations

The rewards stage the demo's job: reach the part, hold it (gated on the grip or the latch), bring it to the goal,
and the task's own measure. The wipe ends on success. The fold and the routing hold a state, so they do not
terminate on success: they pay a bonus every step the goal holds (`custom_terms.MetricsHold`) and run to the time-out.
Ending those episodes on success forfeited the dense terms' remaining pay, and the policy learned to keep hold of the
towel.

## The workcell

Positions are in RoboSimStudio's work-surface frame: the origin at the middle of the bench top, +x away from the
arms, +y toward the left arm, z up from the bench top, which is the ground plane. Two Eden `Yam` arms
(`eden.options.robots.i2rt_yam.Yam`) stand at the demo's mounts, 0.3475 m behind the middle and 0.62 m apart
(`eden.tasks.mochi.common.YAM_BASES`).

- **Gravity-free arms**, as the demo's gravity-compensated arms are (Mochi has no compensation term).
- **The demo's servo gains**, not the asset's sim-to-real defaults: kp 300 / kv 30 for the arm (1600 / 70 for the
  wiping arm, which the demo stiffens), and each demo's calibrated squeeze for the jaws.
- **A squeeze cap is a gain.** Mochi clamps no drive force, so the asset's effort limits are dropped and the cap is
  set through kp: closed on a part `w` wide, each finger squeezes with `finger_kp × w / 2`. 320 N/m puts the demo's
  8 N cap on the 50 mm sponge.
- **Actions.** Each acting arm moves its tool centre (between the fingertip pads) by differential IK with its tilt
  held pointing down, turns it about the vertical where the demo does (the cable), and opens and closes its jaws.
- **No fingertip micro-spheres.** The YAM's MJCF carries six 0.6 mm collision spheres per finger for MuJoCo's point
  contacts. Mochi samples every collision triangle, so they were 55 % of the arm's contact samples and slowed a
  step sevenfold when a policy pushed its hand into the sponge. The task loads a copy without them
  (`common.yam_mochi_file`); the pads' boxes carry the grip.

## Replaying the demo routines

`examples/recipes/mochi/rss_demo_routines.py` plays each demo's own routine (the same moves, speeds and marks)
through `env.step` and the task's own actions. A routine that meets the goal shows the goal is reachable from the
policy's action space.

```bash
uv run python examples/recipes/mochi/rss_demo_routines.py --task mochi_wipe_the_bar
uv run python examples/recipes/mochi/rss_demo_routines.py --task mochi_route_the_cable -b 4
```

<!-- TODO(material): current measured results (routine success per task, trained-policy success at the current
mounts, env-steps/s). The old page's numbers were measured before the arms collided and are not reproduced here. -->

## Recording the wipe

The stripe is the metric's ink, not geometry, so a plain camera shows a clean bar. `eden inference --rec` records the
task's `recording_camera()`, whose `PlanePaintOverlay` paints each cell as dark as the ink it holds, in
`STRIPE_COLOR`, shaded as the renderer shaded the bar. The training config declares no camera. See
[Sensors and rendering](../guides/sensors-and-rendering.md) for camera overlays.

## Where Mochi differs from the demos' physics

Mochi solves the arms, the parts and the fixtures in one implicit Newton system. RoboSimStudio runs MuJoCo for the
arms and couples SuperDex (the sponge) or VBD (the towel, the cable's capsules) to it. Where that mattered:

- **The sponge needs the latch, for the demo's reason.** Friction alone picks and carries the sponge, but a stroke
  drags it out of the jaws, as it did in the demo.
- **The towel is taken by the benchmark's latch, not the demo's pinch.** The demo's jaws gather the two faces of a
  5 mm VBD slab. A Mochi shell is one surface (here with a 2.5 mm collider either side), and the fingertips press it
  into the bench instead of gathering it. So closing the jaws seizes the edge strip within 3 cm of the tool centre
  (`FeatureGraspLatch`, RoboSimStudio's `attach`); a close with nothing in reach seizes nothing.
- **The cable lies on its own bed.** A rod touches through its centreline, so the bench must push back at the
  cable's radius; the same plane would also stop the fingertips at the centreline, and the jaws would ride over the
  cable. So the cable lies on an invisible bed in the bench's plane, the arms stop on the bench, and each plane's
  pairs with the other's bodies are off.
- **The cable bed is grippier than the demo's bench** (0.3 combined friction against the demo's 0.02): the demo's
  contacts are sticky, which is static friction, and Mochi's friction grows with slip below its falloff.
- **The arms collide**, where the demo's pass through each other. So the right hand presses the towel at
  `(0.08, -0.04)`, 80 mm farther out than the demo's mark: at the demo's mark the folding arm stopped against the
  pressing one 45 mm short of its place.
- **The shirt's rest shape is its sewn 3D form.** Mochi takes a shell's bending rest from the mesh it is built from;
  ClothDojo's Style3D cloth bends toward its flat sewing panels. With the bending derived from the material (D ≈
  9e-6 N m) a shirt laid flat sprang 1 cm back toward its 3D form and lost 0.5 of fold quality settling, so the task
  sets D = 1e-6 N m outright, near ClothDojo's 1.25e-6, and the loss is 0.1.
- **The shirt's layers are ClothDojo's 5 mm.** At a 4 mm collider radius (8 mm layers) a folded stack settled 1-2 cm
  taller and lost 0.3-0.6 of fold quality; at ClothDojo's 2.5 mm particle radius it kept it.
- **The sponge is meshed 6 × 9 × 4** (350 nodes, about 8 mm apart) rather than the demo's 9 × 13 × 6 (980): twice
  as fast.
- **The cable** is a Mochi discrete elastic rod with the demo cable's bending rigidity (its per-joint stiffness
  times the segment length) and an axial stiffness that keeps it all but inextensible, as rigid capsules are.

## Physics notes

- **Contact stiffness is 1e7 Pa/m** (`common.CONTACT_STIFFNESS`), which Mochi conditions well in single precision,
  so the tasks train under the default fp32. Mochi's 1e9 default needs `eden.init(precision="64")`.
- **A Mochi scene is all-Mochi.** Every entity, the bench plane included, takes a `Mochi*MaterialOptions`. See
  [Physics](../guides/physics.md).
- **A reset restores the pins of the build.** A pin, and where it holds the vertex, are part of the state a reset
  restores, taken at the end of the env's build. The cable's weld is pinned at every reset (`PinSoftBodyVertices`),
  and a reset releases whatever a latch held.
- **A diverged env is reset alone.** An env whose implicit solve fails is truncated and reset
  (`Episode_Termination/solver_error`) while the others go on (`EnvBase.get_solver_error_envs_mask`); its reward is
  zeroed for that step.

## See also

- [Physics](../guides/physics.md) for the Mochi solver and its options
- `examples/recipes/mochi/` for standalone Mochi scenes (cloth, rods, soft bodies, grippers)
- [Task suites overview](index.md)
