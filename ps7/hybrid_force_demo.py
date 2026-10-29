"""Run PS7's hybrid force-position exercise from a terminal.

From the course root, after completing compute_ctrl:
    python ps7/hybrid_force_demo.py --verify
    python ps7/hybrid_force_demo.py --simulate
    python ps7/hybrid_force_demo.py --simulate --replay ps7/results/hybrid.html

Install numpy, drake and manipulation, or use the course environment. This
script is the terminal version of hybrid_force_position.ipynb, and it does
not use the ps6 folder.
"""

import argparse
from pathlib import Path

import numpy as np

from hybrid_force_control import build_and_simulate, compute_ctrl


def main():
    """Run the stages named on the command line, or print usage if none is named."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="print the wrench for the PDF's fixed verification inputs")
    parser.add_argument("--simulate", action="store_true",
                        help="run the supplied 6.5-second book-dragging simulation")
    parser.add_argument("--replay", type=Path, metavar="HTML",
                        help="save a Meshcat replay as HTML (requires --simulate)")
    args = parser.parse_args()
    if args.replay and not args.simulate:
        parser.error("--replay requires --simulate")
    if not (args.verify or args.simulate):
        parser.print_help()
        return

    if args.verify:
        wrench = compute_ctrl(np.array([0.1, 0.8, 0.05]),
                              np.array([0.2, -0.1, 0.0]), 0.65, 2.0)
        print("[tau_y (N m), f_x (N), f_z (N)] =",
              np.array2string(np.asarray(wrench), precision=4, floatmode="fixed"))

    if args.simulate:
        meshcat = None
        if args.replay:
            from pydrake.all import StartMeshcat
            meshcat = StartMeshcat()
        log, plant, context = build_and_simulate(compute_ctrl, meshcat=meshcat)
        book = plant.GetFreeBodyPose(context, plant.GetBodyByName("book"))
        pose = log.data()
        tip_z = pose[2] - 0.12 * np.cos(pose[0])  # finger tip is 0.12 m below the gripper body
        print("Final book center (m):", book.translation())
        print("Minimum finger-tip height (m):", tip_z.min())
        print("Expected: book x between 0.35 and 0.55 m; tip z at least 0.035 m.")
        if args.replay:
            args.replay.parent.mkdir(parents=True, exist_ok=True)
            args.replay.write_text(meshcat.StaticHtml(), encoding="utf-8")
            print("Replay saved to", args.replay.resolve())


if __name__ == "__main__":
    main()
