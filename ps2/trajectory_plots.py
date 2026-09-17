"""Look at a gripper trajectory as twelve numbers against time.

`pick_and_place.timed_trajectories` builds a pose trajectory, differentiates it
into the velocity trajectory the Jacobian controller eats, and then throws the
pose away.  Nothing in the simulation ever shows either one directly: what you
watch in Meshcat is the arm's answer to them, filtered through IK and the
integrator.  This file puts the two trajectories themselves on a graph.

Nothing here is imported by pick_and_place -- it only ever imports *from* it,
the same way dead_code.py does, so the pset file stays untouched.

There are two ways to look at it.  `--view traces` is twelve numbers against
time, which is where you go to read a value off.  `--view path` is the same
trajectory drawn in the room it happens in: the hand's path through space,
colored by how fast it is going, with the gripper's frame stamped along it.
The staircase and the slerp are much easier to believe once you have seen the
second one.

And there are two timings to look at either way, which is the point of drawing
them side by side.  `--timing keyframe` is the rule the simulation actually
runs: two seconds a leg, whatever the leg has to do, so the speed of the hand
is whatever the geometry leaves it.  `--timing speed` is the rule from the note
to the TAs in pick_and_place: fix the speed and let each leg take as long as it
takes.  The waypoints, the poses and the path through space are identical
between the two -- only the clock differs -- so every difference you can see in
these plots is the timing rule and nothing else.

Figures drawn in the same run are put on common scales, across the timings as
well as within each one.  Without that, per-figure autoscaling would quietly
renormalize the slow trajectory back up to full height and hide the one thing
the comparison is for.

    python trajectory_plots.py                  # 4 waypoints out of test4's plan
    python trajectory_plots.py --view path      # the same four, drawn in 3D
    python trajectory_plots.py --full           # the whole sixteen-waypoint plan
    python trajectory_plots.py --timing speed   # just the fixed-speed rule
"""

import sys
from pathlib import Path

# The shared modules live in utils/, a sibling of this directory, and these
# scripts are run from inside their own pset directory -- so the repository
# root has to go on the path before anything imports from it.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os

# Nothing here simulates anything, but pick_and_place decides at import time
# whether it will want a Meshcat server, and we never do.
os.environ.setdefault("HEADLESS", "1")

import numpy as np
from manipulation.station import MakeMultibodyPlant
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pydrake.all import PiecewisePose, RollPitchYaw

from pick_and_place import (
    BLOCK_LAYOUT,
    BLOCK_SIZE,
    GRIPPER_REACH,
    block_pose,
    make_scenario,
    plan,
    trajectory_stops,
    waypoint_times,
    waypoint_times_at_a_fixed_speed,
)
from utils.plotting import plt

# Which of test4's sixteen waypoints to keep.  These four are the places the
# gripper actually does something -- close on B, let go of B, close on C, let
# go of C -- so the trajectory between them is the plan in miniature, and the
# last leg is the only one in the whole pset where the hand has to turn a block
# on its end, which is what makes the orientation plot worth looking at.
FOUR_WAYPOINTS = (1, 5, 9, 13)

# What each of the twelve numbers is called, in the order the trajectories hand
# them over: a pose is (rotation, translation) and a spatial velocity is
# (angular, translational).
POSE_LABELS = ["roll", "pitch", "yaw", "x", "y", "z"]
VELOCITY_LABELS = [
    r"$\omega_x$",
    r"$\omega_y$",
    r"$\omega_z$",
    r"$v_x$",
    r"$v_y$",
    r"$v_z$",
]
POSE_UNITS = ["rad", "rad", "rad", "m", "m", "m"]
VELOCITY_UNITS = ["rad/s", "rad/s", "rad/s", "m/s", "m/s", "m/s"]


def gripper_start_pose():
    """Where the gripper is before anything has moved.

    The same pose compose_jacobian_control reads off the station, computed off
    a bare plant instead so that nothing has to be built or simulated.
    """
    plant = MakeMultibodyPlant(make_scenario())
    return plant.EvalBodyPoseInWorld(
        plant.CreateDefaultContext(),
        plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")),
    )


def pose_and_velocity_trajectories(waypoints, speed=0.05, timing=waypoint_times):
    """The pose trajectory and its derivative, plus the waypoint times.

    This is `timed_trajectories` with the pose kept rather than dropped, and
    the fingers left out: the fingers are a zero-order hold on one number and
    have nothing to do with where the hand goes.

    `timing` is the rule for when to be where.  The default is the one the
    simulation actually runs, which gives every leg the same two seconds and
    ignores `speed` entirely -- that is why the dashed lines come out evenly
    spaced and the velocity steps are all different heights.  Passing
    `waypoint_times_at_a_fixed_speed` instead is the other way round, and the
    plots are the argument for it: the dashed lines stop being evenly spaced
    and the tallest steps come down by an order of magnitude.  They do not come
    out equal, because the speed it holds fixed is `pose_distance`, which mixes
    rotation into the same number as translation, so a leg that also turns the
    hand spends some of its budget there.
    """
    stops = trajectory_stops(waypoints, gripper_start_pose())
    times = timing(stops, speed)
    traj_X_G = PiecewisePose.MakeLinear(times, [stop.X_WG for stop in stops])
    return traj_X_G, traj_X_G.MakeDerivative(), times


def sample(traj_X_G, traj_V_G, samples=600):
    """Evaluate both trajectories on one shared grid of times.

    Returns (t, pose, velocity), each of the last two `samples` x 6, with the
    rotation first in both -- roll-pitch-yaw for the pose and the angular
    velocity vector for the derivative.  Those are *not* the same three numbers
    differentiated; see `rpy_rates`, which is where the honest derivative comes
    from.

    The angles come back unwrapped, so that a joint passing +pi draws as a line
    going up rather than as a cliff down to -pi.
    """
    t = np.linspace(traj_X_G.start_time(), traj_X_G.end_time(), samples)
    pose = np.array(
        [
            np.concatenate(
                [
                    RollPitchYaw(traj_X_G.GetPose(ti).rotation()).vector(),
                    traj_X_G.GetPose(ti).translation(),
                ]
            )
            for ti in t
        ]
    )
    pose[:, :3] = np.unwrap(pose[:, :3], axis=0)
    velocity = np.array([np.ravel(traj_V_G.value(ti)) for ti in t])
    return t, pose, velocity


def rpy_rates(pose, velocity):
    """The actual derivatives of the roll, pitch and yaw traces.

    Worth having because omega is *not* it, and this trajectory is a good
    example of how confusing that can get.  Drake's RollPitchYaw is the ZYX
    convention, R = Rz(yaw) Ry(pitch) Rx(roll), and stacking those three
    rotations gives

        wx = rdot cos(p) cos(y) - pdot sin(y)
        wy = rdot cos(p) sin(y) + pdot cos(y)
        wz = ydot - rdot sin(p)

    -- so every component of omega is a mix, and which rpy rate lands in which
    component of omega depends on where the wrist happens to be pointing.  This
    function is that 3x3 map run backwards.  It blows up at cos(p) = 0, which
    is gimbal lock and exactly the reason nobody controls a robot in rpy; here
    the pitch stays within a hundredth of a radian of zero, so it is safe.

    On the four-waypoint plan the map does something worth seeing.  Over most
    of it the yaw sits at exactly pi/2 and the pitch at zero, and there the
    three lines above collapse to

        wx = -pdot        wy = rdot        wz = ydot

    which is why the plan's one big reorientation -- standing the last block on
    its end, from t = 6 s to t = 8 s -- shows up as a change in *roll* and a
    spike in *omega_y*, one row apart on the plot.  The rows are labeled
    honestly and it is the pairing that is a fiction, which is what the dashed
    lines are drawn to say.
    """
    r, p, y = pose[:, 0], pose[:, 1], pose[:, 2]
    wx, wy, wz = velocity[:, 0], velocity[:, 1], velocity[:, 2]
    rdot = (wx * np.cos(y) + wy * np.sin(y)) / np.cos(p)
    pdot = wy * np.cos(y) - wx * np.sin(y)
    ydot = wz + rdot * np.sin(p)
    return np.stack([rdot, pdot, ydot], axis=1)


def share_scale(axes):
    """Put every one of `axes` on the union of their y limits.

    Only ever called on a group that is in the same units, so the comparison
    it invites is a fair one: with rad against rad and m against m, the height
    of a wiggle finally means something across panels and not just within one.
    Autoscaling each panel on its own is what makes a hundredth of a radian of
    slerp wobble look like an event.
    """
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(low, high)


def plot_traces(t, pose, velocity, times, title):
    """Twelve stacked axes on one shared clock: pose left, velocity right.

    Every panel gets the same x limits and the same dashed lines at the
    waypoint times, so a corner in a position lines up with the step in the
    velocity that causes it.  The velocities are piecewise constant, because
    the pose trajectory is piecewise linear -- that staircase is the whole
    story of what the controller is handed.

    The bottom three rows really are a quantity above its own derivative.  The
    top three are not, and the plot would be lying if it left it at that: what
    is drawn on the right is omega, which is what the Jacobian eats, and it is
    not the derivative of the roll-pitch-yaw beside it.  So each of those three
    panels also carries the true derivative as a dashed line, and the gap
    between the two is `rpy_rates`' 3x3 map made visible.  Read across a row on
    the top half and you will match the wrong things up; the dashed line is
    there to stop you.

    The panels come in four groups of three -- rad, m, rad/s, m/s -- and each
    group shares one scale, so that the three components of a quantity can be
    compared with each other and not only with themselves.  What this costs is
    that a component which does almost nothing draws as a flat line: the pitch
    really does stay inside a hundredth of a radian here, and on a scale that
    has to hold the yaw's full pi it has nothing left to show.  That is the
    honest picture of it.
    """
    rates = rpy_rates(pose, velocity)
    fig, axes = plt.subplots(6, 2, sharex=True, figsize=(11, 12))
    fig.suptitle(title)
    axes[0, 0].set_title("gripper pose $X^W{}_G$")
    axes[0, 1].set_title("gripper spatial velocity $V^W{}_G$")

    for row in range(6):
        for col, (data, labels, units) in enumerate(
            [
                (pose, POSE_LABELS, POSE_UNITS),
                (velocity, VELOCITY_LABELS, VELOCITY_UNITS),
            ]
        ):
            ax = axes[row, col]
            ax.plot(t, data[:, row], color=f"C{row}", label=labels[row])
            if col == 1 and row < 3:
                ax.plot(
                    t,
                    rates[:, row],
                    color="0.35",
                    ls="--",
                    lw=1.2,
                    label=f"d({POSE_LABELS[row]})/dt",
                )
                ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85)
            ax.set_ylabel(f"{labels[row]}\n[{units[row]}]")
            ax.grid(alpha=0.3)
            for waypoint_time in times:
                ax.axvline(waypoint_time, color="0.7", ls="--", lw=0.8)
            if col == 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")

    # Rotations with rotations and translations with translations, on each
    # side.  The dashed rpy rates are in rad/s too, so they are inside the
    # group they are drawn on and pull its scale along with them.
    for group in (axes[:3, 0], axes[3:, 0], axes[:3, 1], axes[3:, 1]):
        share_scale(group)

    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    return fig


######################################################################
## The same trajectory, drawn in the room it happens in
######################################################################


# How many gripper frames to stamp along the path.  Enough to read the
# orientation off, few enough that the triads do not become a hedge.
TRIAD_COUNT = 22

# How long to draw each triad's arms, in meters.  About a third of the
# gripper's reach, which is small next to the blocks and still visible.
TRIAD_LENGTH = 0.035

# x, y, z of a frame, in the colors Drake's own triads use.
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")


def block_edges(X_WO):
    """The twelve edges of a block sitting at `X_WO`, as pairs of world points.

    A block's frame is at the middle of its bottom face -- the same convention
    make_overhead_box_grasp relies on -- so the corners run half a length and
    half a width either way, and a full height up.
    """
    length, width, height = BLOCK_SIZE
    corners = np.array(
        [
            [sx * length / 2, sy * width / 2, sz * height]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (0, 1)
        ]
    )
    world = np.array([X_WO @ corner for corner in corners])
    # Two corners share an edge exactly when they differ in one coordinate,
    # which in this ordering means their indices differ by one bit.
    return [
        (world[i], world[j])
        for i in range(8)
        for j in range(i + 1, 8)
        if bin(i ^ j).count("1") == 1
    ]


def draw_the_scene(ax):
    """The three blocks where make_scenario starts them, as wireframe boxes.

    Context, not physics: the path means much more when you can see which
    block it is reaching into.  These are the blocks' *initial* poses, so the
    later half of a plan reaches for places where the boxes drawn here are not
    any more -- that is the plan moving them.

    Returns every corner it drew, so that the framing can allow for the blocks
    as well as for the path.
    """
    corners = []
    for name, *_, rgb in BLOCK_LAYOUT:
        edges = block_edges(block_pose(name))
        for a, b in edges:
            ax.plot(*zip(a, b), color=rgb, lw=1.2, alpha=0.8)
        corners += [a for a, _ in edges]
    return np.array(corners)


def pad_path(traj_X_G, t):
    """Where the middle of the finger pads is, GRIPPER_REACH out along G's y.

    Worth drawing next to G's own path: the two are the same shape while the
    hand only translates, and pull apart the moment it turns.  It is the
    cheapest picture of why "where the gripper is" needs a pose and not a
    point -- and of why the pads, not G, are what has to land on a block.
    """
    return np.array(
        [traj_X_G.GetPose(ti) @ np.array([0, GRIPPER_REACH, 0]) for ti in t]
    )


def draw_the_path(ax, pose, velocity, pads):
    """The gripper origin's path, colored by how fast it is going.

    One short line segment per sample, so the color can change along the path;
    the speed is the translational half of the spatial velocity, which is the
    speed of G's origin and so exactly the speed of the line being drawn.
    """
    p_G = pose[:, 3:]
    segments = np.stack([p_G[:-1], p_G[1:]], axis=1)
    speed = np.linalg.norm(velocity[:, 3:], axis=1)
    lines = Line3DCollection(segments, cmap="viridis", lw=2.5)
    lines.set_array(speed[:-1])
    ax.add_collection3d(lines)
    ax.plot(*pads.T, color="0.45", ls="--", lw=1.0, label="middle of the pads")
    return lines


def draw_the_triads(ax, traj_X_G, t):
    """Stamp the gripper's frame along the path at `TRIAD_COUNT` even times.

    Even in *time*, not in distance, so the triads also read as a clock: they
    bunch up where the hand is dawdling and spread out where it is hurrying.
    """
    for ti in t[:: max(1, len(t) // TRIAD_COUNT)]:
        X_WG = traj_X_G.GetPose(ti)
        origin = X_WG.translation()
        R = X_WG.rotation().matrix()
        for axis in range(3):
            ax.quiver(
                *origin,
                *(TRIAD_LENGTH * R[:, axis]),
                color=AXIS_COLORS[axis],
                lw=1.2,
                arrow_length_ratio=0.25,
            )


def draw_the_waypoints(ax, traj_X_G, times):
    """A labeled dot at every stop, numbered the way trajectory_stops orders
    them: 0 is where the gripper already was, and the last is the repeat that
    makes the plan end standing still."""
    for i, ti in enumerate(times):
        p = traj_X_G.GetPose(ti).translation()
        ax.scatter(*p, color="black", s=18, depthshade=False, zorder=5)
        # Lift the number off its dot, so that two stops in nearly the same
        # place -- the repeat at the end is exactly that -- stay readable.
        ax.text(p[0], p[1], p[2] + 0.012, f"{i}", fontsize=9, color="black")


def equal_aspect(ax, points, margin=0.03):
    """Make one meter the same length on all three axes.

    Without this the trajectory's 0.25 m of z gets stretched to the height of
    the figure, every triad comes out sheared, and a straight-line leg looks
    bent.  Squaring the *limits* would fix that too, but it would also pad the
    two short axes out to the length of the longest one and leave the picture
    mostly empty; giving the box the data's own proportions keeps the scale
    honest and the drawing tight.
    """
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    for setter, a, b in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], low, high):
        setter(a, b)
    ax.set_box_aspect(high - low)


def plot_path(t, pose, velocity, traj_X_G, times, title):
    """The whole trajectory as one picture: path, speed, orientation, scene."""
    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)

    pads = pad_path(traj_X_G, t)
    corners = draw_the_scene(ax)
    lines = draw_the_path(ax, pose, velocity, pads)
    draw_the_triads(ax, traj_X_G, t)
    draw_the_waypoints(ax, traj_X_G, times)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    fig.colorbar(lines, ax=ax, shrink=0.6, pad=0.1, label="speed of G [m/s]")
    ax.legend(loc="upper left", fontsize=8)
    equal_aspect(ax, np.vstack([pose[:, 3:], pads, corners]))
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    return fig


######################################################################
## Putting several figures on common scales
######################################################################


# The timing rules, by the name the command line calls them.  "keyframe" is
# what timed_trajectories uses and so what the simulation runs; "speed" is the
# alternative pick_and_place's note to the TAs argues for.
TIMINGS = {
    "keyframe": waypoint_times,
    "speed": waypoint_times_at_a_fixed_speed,
}


def trace_grid(fig):
    """The 6 x 2 array of axes inside a `plot_traces` figure.

    plt.subplots hands its axes back row-major, and `fig.axes` keeps them in
    the order they were created, so the shape can simply be put back on.
    """
    return np.array(fig.axes).reshape(6, 2)


def match_trace_scales(figs):
    """Put corresponding panels of several `plot_traces` figures on one scale.

    Same four groups of units as `share_scale` uses within a figure, only
    unioned across all of them, plus a common time axis.  This is what makes
    two timings of the same plan comparable: the fixed-speed run takes about
    three and a half times as long and its velocity steps are about a fifth as
    tall, and both of those facts are invisible if each figure is allowed to
    scale itself to fill its own axes.
    """
    grids = [trace_grid(fig) for fig in figs]
    for rows, col in [(slice(0, 3), 0), (slice(3, 6), 0), (slice(0, 3), 1), (slice(3, 6), 1)]:
        share_scale([ax for grid in grids for ax in grid[rows, col]])
    end = max(ax.get_xlim()[1] for grid in grids for ax in grid.ravel())
    for grid in grids:
        for ax in grid.ravel():
            ax.set_xlim(0, end)


def match_speed_colors(figs):
    """Put the speed colormaps of several `plot_path` figures on one range.

    The colorbar follows its mappable, so setting the limits is enough.  With
    it done, the fixed-speed path comes out uniformly dark against the
    two-seconds-a-leg one instead of being recolored to use the whole map.

    The triads are Line3DCollections too -- that is what ax.quiver builds one
    out of -- so being the right type is not enough to identify the path.
    Carrying an array of values to color by is: only the path has one.
    """
    paths = [
        artist
        for fig in figs
        for artist in fig.axes[0].collections
        if isinstance(artist, Line3DCollection) and artist.get_array() is not None
    ]
    low = min(path.get_array().min() for path in paths)
    high = max(path.get_array().max() for path in paths)
    for path in paths:
        path.set_clim(low, high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        choices=["traces", "path", "both"],
        default="both",
        help="twelve numbers against time, the path in 3D, or both",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="plot the whole plan instead of four of its waypoints",
    )
    parser.add_argument(
        "--timing",
        choices=["keyframe", "speed", "both"],
        default="both",
        help="two seconds a leg as the simulation does, a fixed hand speed, "
        "or both drawn on common scales",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.05,
        help="m/s, and only used by the speed timing",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="write the figures here instead of showing them; whichever of "
        "the timing and the view has more than one value is inserted "
        "before the extension",
    )
    args = parser.parse_args()

    whole_plan = plan("red", "blue", "yellow")
    if args.full:
        waypoints = whole_plan
        title = f"test4's plan: {len(waypoints)} waypoints"
    else:
        waypoints = [whole_plan[i] for i in FOUR_WAYPOINTS]
        title = f"four waypoints from test4's plan: {list(FOUR_WAYPOINTS)}"

    timings = list(TIMINGS) if args.timing == "both" else [args.timing]
    views = ["traces", "path"] if args.view == "both" else [args.view]

    # Every figure first, then the scales, then the saving: a scale can only be
    # matched against figures that already exist.
    figures = {}
    for timing in timings:
        rule = TIMINGS[timing]
        traj_X_G, traj_V_G, times = pose_and_velocity_trajectories(
            waypoints, args.speed, rule
        )
        t, pose, velocity = sample(traj_X_G, traj_V_G)
        for view in views:
            heading = f"{title}, timed by {rule.__name__}"
            figures[timing, view] = (
                plot_traces(t, pose, velocity, times, heading)
                if view == "traces"
                else plot_path(t, pose, velocity, traj_X_G, times, heading)
            )

    if len(timings) > 1:
        if "traces" in views:
            match_trace_scales([figures[timing, "traces"] for timing in timings])
        if "path" in views:
            match_speed_colors([figures[timing, "path"] for timing in timings])

    for (timing, view), fig in figures.items():
        if not args.save:
            continue
        tag = "_".join(
            part
            for part, several in [(timing, len(timings) > 1), (view, len(views) > 1)]
            if several
        )
        stem, dot, extension = args.save.rpartition(".")
        if not tag:
            name = args.save
        elif dot:
            name = f"{stem}_{tag}{dot}{extension}"
        else:
            # No extension to insert before, so put the tag on the end -- the
            # alternative is several figures quietly overwriting one file.
            name = f"{args.save}_{tag}"
        fig.savefig(name, dpi=150)
        print(f"wrote {name}")
    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
