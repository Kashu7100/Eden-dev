"""``mochi_fold_the_shirt``: two YAMs fold a ClothDojo T-shirt from its benchmark's fold starts, scored by its fold quality.

Eden's version of ClothDojo's ``shirt_fold`` task (clothdojo.github.io, arXiv 2611.01190) for one of its garments,
``rand_ZZCWITH5HY`` (GarmentCodeData, 6888 vertices, 0.78 x 0.51 m laid flat; here at 0.7 of that), on the Mochi
solver. Each episode starts from one of the garment's four benchmark fold starts
(``benchmark/conditions/nocs-fold``: the shirt settled flat, sleeves out, at 0, 90, 180 and 270 degrees), and
success is ClothDojo's fold quality Q (:class:`~eden.tasks.mochi.custom_terms.shirt.ShirtFoldQuality`, its five
geometric features and linear model) at or above the threshold, with the shirt at rest and both hands clear of it. ClothDojo also asks Q to hold for
3 s; here "at rest and let go" stands in for that hold, as in ``mochi_fold_the_towel``.

The shirt is ClothDojo's garment at 0.7 of its size (``SHIRT_SCALE``: 0.55 x 0.36 m laid flat, a child's
T-shirt), laid 19 cm nearer the arms than ClothDojo lays it (``SHIRT_OFFSET``). ClothDojo's Pipers fold an adult's
shirt; the YAMs at 3 cm over the bench reach about 0.10 m past the bench's middle on their own side and only up to it
across the centre line, so at full size and ClothDojo's place even a half fold, hem to shoulders, is out of reach
(the hem would travel from x = -0.14 m to +0.36 m). At this size and place the 270-degree start (hem toward the
arms) spans x = -0.25 to +0.10 m and y = +/-0.27 m, and folding it in thirds along its length (sleeves in, then the
hem up twice; ``examples/recipes/mochi/fold_the_shirt_routine.py``) is within reach. The starts and flat reference come
from ``examples/recipes/mochi/convert_clothdojo_shirt.py`` at full size, in ClothDojo's world; the reset and the
metric scale them in the plane (``planar_scale``) and shift them onto the bench by ``SHIRT_OFFSET``.

Both arms act. Each moves its tool centre by differential IK with its tilt held pointing down, turns it about the
vertical, and closes its jaws
with the benchmark's deformable grasp (:class:`~eden.tasks.mochi.custom_terms.grasp.FeatureGraspLatch`): a close
seizes the shirt within 1.5 cm of the tool centre, at most 32 vertices, close to ClothDojo's own grasp (four seeds
within 1 cm grown two mesh rings, at most 32). Control is 30 Hz, as in the other Mochi tasks (ClothDojo's is 50 Hz),
over ClothDojo's 40 s fold horizon.

The shirt is a Mochi shell with the GarmentCode mesh's sewn 3D shape as its rest geometry, 180 g/m^2 (ClothDojo's
areal density), an in-plane stiffness E t = 100 N/m, softer than woven cotton, as ClothDojo's Style3D cloth is
(50 N/m along each panel axis), and a bending stiffness near ClothDojo's. Measured on an RTX 3080 with the shirt at
rest: 11 ms per env-step at 4 Newton iterations, 5.6 ms at 2; self-contact costs little.
"""

from __future__ import annotations

import genesis as gs

from eden import EventMode, observations, rewards, terminations
from eden.engine.solvers.mochi.options import MochiOptions
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options import ObservationGroupOptions
from eden.options.camera import CameraOptions
from eden.options.entities import EntityOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.materials import MochiShellMaterialOptions
from eden.tasks.mochi import custom_terms
from eden.tasks.mochi.common import (
    CONTACT_STIFFNESS,
    YAM_ARM_DOFS,
    YAM_FINGER_DOFS,
    YAM_FINGERS_CLOSED,
    YAM_FINGERS_OPEN,
    YAM_TCP_OFFSET,
    bench,
    yam,
)
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import DoneTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

# ---- the garment: ClothDojo's rand_ZZCWITH5HY, as convert_clothdojo_shirt.py writes it ----------------------------
GARMENT_DIR = "objects/clothdojo/rand_ZZCWITH5HY"
SHIRT_FILE = f"{GARMENT_DIR}/shirt.obj"
# The four fold starts ("starts") and the flat reference ("flat_reference"), in ClothDojo's world frame.
STARTS_FILE = f"{GARMENT_DIR}/fold_starts.npz"
# ClothDojo's world onto the bench: its Piper bases at (0.25, +/-0.30) onto the YAMs' x = -0.3475, its table top
# (z = 0.80) onto the bench. The starts' lowest vertex is 2.9 mm over the table, clear of the shell's collider.
CLOTHDOJO_TO_BENCH = (-0.5975, 0.0, -0.80)
# The garment's size against ClothDojo's (the mesh, and its starts and flat reference in the plane).
SHIRT_SCALE = 0.7
# The starts and flat reference onto the bench: ClothDojo's layout, then 0.19 m toward the arms. All four starts are
# centred at x = 0.113 m in ClothDojo's layout, so here at x = -0.077 m; the 270-degree start's hem lies at x = -0.25 m,
# 0.10 m in front of the arms' bases.
SHIRT_OFFSET = (CLOTHDOJO_TO_BENCH[0] - 0.19, 0.0, CLOTHDOJO_TO_BENCH[2])
# ClothDojo's garment placement before its starts were settled, mapped onto the bench and raised 0.4 m: where the
# sewn rest shape is built, clear of the arms, before the first reset lays a start down.
REST_POS = (0.0525, 0.0, 0.764)
# Each side of the shell collides 2.5 mm out, ClothDojo's particle radius: a 5 mm layer. At 4 mm (8 mm layers) a folded
# stack of ClothDojo's expert fold settled 1-2 cm taller and lost 0.3-0.6 of fold quality; at 2.5 mm it kept it.
COLLIDER_RADIUS = 0.0025
# The shell's bending stiffness (N m); see the material below.
BENDING_STIFFNESS = 1e-6
# 24 vertices spread over the flat shirt (farthest-point sampling of the flat reference's xy from its outermost
# vertex): its shape for the policy. They index rand_ZZCWITH5HY's mesh.
KEYPOINTS = [5, 6, 15, 151, 162, 167, 285, 309, 338, 383, 472, 1091, 1506, 1549, 1857, 1904, 2618, 3596, 3649,
             3881, 4185, 4419, 4985, 6740]  # fmt: skip

# ClothDojo's fold-quality threshold, kept for Mochi's cloth. Eight frames of ClothDojo's published expert fold
# (garment rand_004NVCOKV7, Q +0.05 to +1.0 on its cloth), scaled like the starts, laid into this task's scene and
# settled 4 s: seven settled at Q +0.19 to +1.14 (change -0.09 to +0.54, mean +0.21), one diverged. Its flat starts
# score -1.62 as laid here (-1.39 on its cloth at full size).
# examples/recipes/mochi/fold_the_shirt_threshold.py reruns the measurement.
FOLD_QUALITY_THRESHOLD = custom_terms.shirt.FOLD_QUALITY_THRESHOLD

_EPISODE_S = 40.0
CONTROL_DT = 1.0 / 30.0
_LEFT_TCP = {"robot_name": "left", "link_name": "link_6", "link_offset": YAM_TCP_OFFSET}
_RIGHT_TCP = {"robot_name": "right", "link_name": "link_6", "link_offset": YAM_TCP_OFFSET}
# The i2RT jaws' servo, as in mochi_fold_the_towel.
_FINGER_KP, _FINGER_KD = 600.0, 20.0


def _hand(robot: str) -> DifferentialIKController:
    return DifferentialIKController.configure(
        entity_name=robot,
        ee_link_name="link_6",
        dofs_name=YAM_ARM_DOFS,
        position_scale=0.015,
        # The tilt held (pointing down), the turn about the vertical free: 0.1 rad a step, as in mochi_route_the_cable.
        # A hand whose turn is held too must roll its wrist against every turn of the base, and the wrist's +/-2.09 rad
        # stops then end the reach well short of the shirt's middle (the right arm's ready posture leaves it 0.26 rad).
        orientation_scale=[0.0, 0.0, 0.1],
        damping=0.02,
        max_dq=0.1,
    )


def _gripper(robot: str) -> custom_terms.FeatureGraspLatch:
    return custom_terms.FeatureGraspLatch.configure(
        entity_name=robot,
        dofs_name=YAM_FINGER_DOFS,
        open_action=YAM_FINGERS_OPEN,
        close_action=YAM_FINGERS_CLOSED,
        soft_entity_name="shirt",
        tcp_offset=YAM_TCP_OFFSET,
        radius=0.015,
        max_vertices=32,
    )


_Observations = {
    "left_dofs_pos": observations.DofsPos.configure(entity_name="left"),
    "left_dofs_vel": observations.DofsVel.configure(entity_name="left"),
    "right_dofs_pos": observations.DofsPos.configure(entity_name="right"),
    "right_dofs_vel": observations.DofsVel.configure(entity_name="right"),
    "left_tool": custom_terms.ToolCentre.configure(**_LEFT_TCP),
    "right_tool": custom_terms.ToolCentre.configure(**_RIGHT_TCP),
    "shirt_from_left": custom_terms.SoftBodyPoints.configure(entity_name="shirt", vertices=KEYPOINTS, **_LEFT_TCP),
    "shirt_from_right": custom_terms.SoftBodyPoints.configure(entity_name="shirt", vertices=KEYPOINTS, **_RIGHT_TCP),
    "fold_quality": observations.MetricObservation.configure(metric_name="fold_quality"),
    "left_holds": observations.MetricObservation.configure(metric_name="left_holds"),
    "right_holds": observations.MetricObservation.configure(metric_name="right_holds"),
    "last_action": ObsTerm.configure(func=observations.last_action),
}


@TASK_REGISTRY.register()
class MochiFoldTheShirtConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        # About 32 MiB of Mochi state an env (the 3593-vertex shirt's 16.6, doubled): 128 envs fit a 10 GB card.
        num_envs=128,
        episode_length_s=_EPISODE_S,
        sim_dt=CONTROL_DT,
        sim_substeps=1,
        decimation=1,
        # At most 100 conjugate-gradient iterations a Newton step, as in mochi_fold_the_towel.
        mochi_options=MochiOptions(n_newton_iterations=4, n_pcg_iterations=100),
    )

    scene_options: SceneOptions = SceneOptions(
        # The bench pushes back at the shell's collider radius (see mochi_fold_the_towel).
        plane=bench(friction=0.8, penalty_threshold=COLLIDER_RADIUS),
        left=yam("left", ready="jaws_x", finger_kp=_FINGER_KP, finger_kd=_FINGER_KD),
        right=yam("right", ready="jaws_x", finger_kp=_FINGER_KP, finger_kd=_FINGER_KD),
        shirt=EntityOptions(
            file=SHIRT_FILE,
            scale=SHIRT_SCALE,
            default_root_pos=REST_POS,
            material=MochiShellMaterialOptions(
                E=1e5,
                nu=0.3,
                rho=180.0,  # 180 g/m^2 over the 1 mm membrane: ClothDojo's cloth_density
                thickness=1e-3,
                # Bending stiffness D = 1e-6 N m (ClothDojo's is 1.25e-6 along a panel's stiffer axis), set outright:
                # derived from E and the thickness it would be 9e-6. The rest geometry is the sewn 3D shirt, whose
                # curvature Mochi takes as the bending rest (Style3D bends toward flat panels); at 9e-6 a shirt laid
                # flat springs 1 cm back up toward it and loses 0.5 of fold quality settling, at 1e-6 0.14.
                bending_alpha=0.3 * BENDING_STIFFNESS,  # D nu
                bending_beta=0.7 * BENDING_STIFFNESS,  # D (1 - nu)
                stiffness_damping=1e-3,
                # ClothDojo's cloth-body friction; Mochi takes the geometric mean with the bench's 0.8.
                friction=0.6,
                penalty_coefficient=CONTACT_STIFFNESS,
                collider_radius=COLLIDER_RADIUS,
                # A fold must not pass through itself.
                self_contact=True,
            ),
            surface=gs.surfaces.Default(color=(0.36, 0.52, 0.74)),
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # One of the four benchmark fold starts, drawn per env.
        fold_start=custom_terms.ResetSoftBodyFromStates.configure(
            mode=EventMode.RESET,
            entity_name="shirt",
            states_file=STARTS_FILE,
            offset=SHIRT_OFFSET,
            planar_scale=SHIRT_SCALE,
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        fold_quality=custom_terms.ShirtFoldQuality.configure(
            entity_name="shirt",
            reference_file=STARTS_FILE,
            offset=SHIRT_OFFSET,
            planar_scale=SHIRT_SCALE,
            interval=5,
            success_threshold=FOLD_QUALITY_THRESHOLD,
        ),
        settled=custom_terms.SoftBodySettled.configure(entity_name="shirt", success_threshold=0.02),
        hands_clear=custom_terms.HandsClear.configure(
            entity_name="shirt", robots_name=["left", "right"], link_offset=YAM_TCP_OFFSET, success_threshold=0.02
        ),
        left_holds=custom_terms.LatchHolds.configure(action_name="left_gripper", contributes_to_success=False),
        right_holds=custom_terms.LatchHolds.configure(action_name="right_gripper", contributes_to_success=False),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Observations, enable_corruption=True),
        critic=ObservationGroupOptions(**_Observations, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        left_hand=_hand("left"),
        right_hand=_hand("right"),
        left_gripper=_gripper("left"),
        right_gripper=_gripper("right"),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        # Dense: Q rises as the shirt is folded into a neat, compact stack.
        fold_quality=rewards.MetricReward.configure(metric_name="fold_quality", weight=1.0),
        # Staged on the way to the goal, as in mochi_fold_the_towel: folded; folded and let go by both hands; folded
        # and graded by how far both hands are from it, up to the goal's 2 cm.
        folded=custom_terms.MetricsHold.configure(metrics_name=["fold_quality"], weight=2.0),
        folded_released=custom_terms.MetricsHold.configure(
            metrics_name=["fold_quality"], zero_metrics_name=["left_holds", "right_holds"], weight=4.0
        ),
        folded_clear=custom_terms.MetricsHold.configure(
            metrics_name=["fold_quality"], scale_metric_name="hands_clear", scale_target=0.02, weight=8.0
        ),
        # Every step the whole goal holds (let go and at rest included).
        success=rewards.TaskSuccess.configure(weight=50.0),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        left_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "left", "max_vel": 1.0}
        ),
        right_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits, weight=-0.01, params={"entity_name": "right", "max_vel": 1.0}
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        # No termination on success: a folded shirt is a state to hold, as the folded towel is.
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[256, 128, 64],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[256, 128, 64]),
        num_steps_per_env=24,
        max_iterations=3000,
        save_interval=100,
        experiment_name="mochi_fold_the_shirt",
        run_name="",
    )

    def recording_camera(self) -> CameraOptions:
        """In front of the bench and above it, looking back at the arms: the whole shirt and both hands."""
        return CameraOptions(
            cam_pos=(0.45, -0.45, 0.55),
            cam_lookat=(-0.12, 0.0, 0.03),
            cam_fov=50.0,
            cam_resolution=(960, 540),
        )
