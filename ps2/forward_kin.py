# python libraries
# ruff: noqa: F401
import sys
from pathlib import Path

# The shared modules live in utils/, a sibling of this directory, and these
# scripts are run from inside their own pset directory -- so the repository
# root has to go on the path before anything imports from it.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import mpld3
import numpy as np
from manipulation import running_as_notebook
from manipulation.exercises.pick.plot_planar_manipulator import (
    plot_planar_manipulator,
    plot_two_planar_manipulators,
)
from pydrake.all import RigidTransform, RotationMatrix

from utils.plotting import plt

######################################################################
## Display code
######################################################################


def test_plot_fk():
    q = 2 * np.pi * np.random.rand(2)
    # q = [np.pi/2, np.pi/2]
    plot_planar_manipulator(q, forward_kinematics(q))


def test_plot_ik():
    theta = 2 * np.pi * np.random.rand()
    r = 2 * np.random.rand()
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    q1, q2 = inverse_kinematics(x, y)
    print(q1, q2)
    plot_two_planar_manipulators(q1, q2, np.array([x, y]))


def test_plot_manipulability(n: int = 200):
    """Heat map of manipulability(Jacobian(q)) over the joint space."""
    grid = np.linspace(-np.pi, np.pi, n)
    Q0, Q1 = np.meshgrid(grid, grid)
    w = np.array(
        [manipulability(Jacobian(np.array(q))) for q in zip(Q0.ravel(), Q1.ravel())]
    ).reshape(Q0.shape)

    plt.figure()
    # Nonnegative magnitude, anchored at zero. viridis has no white at either
    # end, so low manipulability can't be confused with empty background.
    plt.pcolormesh(Q0, Q1, w, cmap="viridis", vmin=0, vmax=np.max(w), shading="auto")
    plt.colorbar(label="manipulability  |det(J)|")
    # Near-zero level set = the singular configurations
    # plt.contour(Q0, Q1, w, levels=[0.05 * np.max(w)], colors="0.35", linewidths=1)

    ticks = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    labels = [r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"]
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("$q_0$")
    plt.ylabel("$q_1$")
    plt.title("Manipulability over joint space")
    plt.gca().set_aspect("equal", adjustable="box")


def test_plot_manipulability_task_space(n: int = 200):
    """Heat map of manipulability over task space, at the IK configuration.

    The two IK solutions are elbow-up/elbow-down mirrors (q1 and -q1), so
    |det(J)| is the same for both; we take the first.
    """
    grid = np.linspace(-2.2, 2.2, n)
    X, Y = np.meshgrid(grid, grid)
    w = np.array(
        [
            manipulability(Jacobian(inverse_kinematics(x, y)[0]))
            if x**2 + y**2 <= 4.0  # unreachable outside, IK has no solution
            else np.nan
            for x, y in zip(X.ravel(), Y.ravel())
        ]
    ).reshape(X.shape)

    plt.figure()
    plt.pcolormesh(X, Y, w, cmap="viridis", vmin=0, vmax=np.nanmax(w), shading="auto")
    plt.colorbar(label="manipulability  |det(J)|")
    # Unreachable cells are nan, so they show the axes background: a neutral
    # gray that appears nowhere in the colormap.
    plt.gca().set_facecolor("0.85")
    # Outer boundary of the reachable set
    th = np.linspace(0, 2 * np.pi, 200)
    plt.plot(2 * np.cos(th), 2 * np.sin(th), "-", color="0.5", linewidth=1)

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Manipulability over task space")
    plt.gca().set_aspect("equal", adjustable="box")


######################################################################
## Code for students to study
######################################################################


def X2D(x, y, th):
    return RigidTransform(RotationMatrix.MakeZRotation(th), [x, y, 0])


def X2p(X):
    return X.translation()[:2]


######################################################################
## Code for students to implement
######################################################################

## One of the "bookwork problems"
X_WA = X2D(1, 2, np.pi / 6)
X_WB = X2D(2, 3, -np.pi / 4)
X_WC = X2D(6, 4, -np.pi * 2 / 3)

X_BW = X_WB.inverse()
X_CW = X_WC.inverse()
X_BC = X_BW @ X_WC

p_BC_B = X2p(X_BC)
# print(p_BC_B)


# Use Drake tools!
def forward_kinematics(q: np.ndarray) -> np.ndarray:
    (q0, q1) = q
    X_a0 = X2D(0, 0, 0)
    X_0b = lambda q: X2D(0, 0, q)  # noqa: E731
    X_b1 = X2D(1, 0, 0)
    X_1c = lambda q: X2D(0, 0, q)  # noqa: E731
    X_cC = X2D(1, 0, 0)
    return X2p(X_a0 @ X_0b(q0) @ X_b1 @ X_1c(q1) @ X_cC)


def inverse_kinematics(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    r2 = x**2 + y**2
    q1_1 = np.arccos(r2 / 2 - 1)
    q2_1 = -q1_1

    # The solution for q_0 has been implemented for you
    q1_0 = np.arctan2(y, x) - np.arctan2(np.sin(q1_1), 1 + np.cos(q1_1))
    q2_0 = np.arctan2(y, x) - np.arctan2(np.sin(q2_1), 1 + np.cos(q2_1))

    return (np.array([q1_0, q1_1]), np.array([q2_0, q2_1]))


def Jacobian(q: np.ndarray) -> np.ndarray:
    q0 = q[0]
    q1 = q[1]
    s0 = np.sin(q0)
    s1 = np.sin(q1)
    c0 = np.cos(q0)
    c1 = np.cos(q1)
    dxdq0 = -s0 - c1 * s0 - c0 * s1
    dxdq1 = -s1 * c0 - s0 * c1
    dydq0 = c0 + c0 * c1 - s0 * s1
    dydq1 = -s0 * s1 + c0 * c1
    J = np.array(
        [
            [dxdq0, dxdq1],
            [dydq0, dydq1],
        ]
    )
    return J


def manipulability(J):
    return np.abs(np.linalg.det(J))


if __name__ == "__main__":
    # test_plot_fk()
    # test_plot_ik()
    test_plot_manipulability()
    test_plot_manipulability_task_space()
    plt.show()
