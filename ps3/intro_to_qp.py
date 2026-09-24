"""MathematicalProgram, and a small QP to solve with it."""

import sys
from pathlib import Path

# The shared modules live in utils/, a sibling of this directory, and these
# scripts are run from inside their own pset directory -- so the repository
# root has to go on the path before anything imports from it.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from pydrake.all import MathematicalProgram, Solve, eq, le

from utils.plotting import plt
from utils.viz import HEADLESS

######################################################################
## Code for students to study
######################################################################


def scalar_example() -> float:
    """min x^2 subject to x >= 3."""
    # 1. make an instance of MathematicalProgram
    prog = MathematicalProgram()

    # 2. Make the variables; x is an array
    x = prog.NewContinuousVariables(1)

    # 3. Define a cost function on x
    prog.AddCost(x.dot(x))

    # 4. Define a constraint as an ordinary Python expression, or using eq, le on vectors
    prog.AddConstraint(x[0] >= 3)

    # 5. Call the solver. It will pick an appropriate algorithm automatically
    result = Solve(prog)

    # 6. Extract the answer
    if not result.is_success():
        raise RuntimeError("the scalar example failed to solve")
    return result.GetSolution(x)[0]


def vector_example() -> np.ndarray:
    """A QP in three variables, with vector-valued constraints.

        min  x0^2 + x1^2 + x2^2
        s.t. [[2, 3, 1], [5, 1, 0]] x = [1, 1]
             x <= [2, 2, 2]

    A quadratic cost and linear constraints is what makes a problem a QP, and
    QPs are solved fast enough to run one inside a control loop -- which is
    exactly what `diffik_optimization.py` does next.

    `eq`, `le` and `ge` build one constraint per element, so a whole vector of
    them can be written on one line.
    """
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(3)

    prog.AddCost(x.dot(x))
    prog.AddConstraint(eq(np.array([[2, 3, 1], [5, 1, 0]]).dot(x), [1, 1]))
    prog.AddConstraint(le(x, 2 * np.ones(3)))

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("the vector example failed to solve")
    return result.GetSolution(x)


## Drawing the scalar problems

# Only the scalar examples can be drawn like this -- one variable on the x axis
# and the cost on the y axis


def plot_scalar_problem(cost, x_star, xlim, feasible, title) -> None:
    """Draw a one-variable problem: the cost, the feasible set, the optimum.

    Draws into the current axes, so the caller can put several of these in one
    figure with `plt.subplot` and call `plt.show()` once at the end.

    Args:
        cost: the cost, as an ordinary Python function of one float.
        x_star: what the solver returned.
        xlim: (low, high) range of x to draw.
        feasible: (low, high) of the feasible interval; either may be None,
            meaning unbounded on that side.
        title: heading for this subplot.
    """
    xs = np.linspace(xlim[0], xlim[1], 400)
    plt.plot(xs, [cost(x) for x in xs], "k-", label="cost")

    lo = xlim[0] if feasible[0] is None else feasible[0]
    hi = xlim[1] if feasible[1] is None else feasible[1]
    plt.axvspan(lo, hi, color="0.85", label="feasible")

    plt.plot([x_star], [cost(x_star)], "ro", label=f"x* = {x_star:.4f}")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("cost")
    plt.legend()


######################################################################
## Code for students to write
######################################################################


def scalar_example_interior() -> float:
    """min x^2 - 2x subject to -1 <= x <= 3."""
    raise NotImplementedError("your code here")


## The problem to solve:
##
##     min  2 x0^2 + x1^2 + 4 x2^2
##     s.t. [[1, 2, 3], [2, 7, 4]] x = [1, 1]
##          |x| <= [0.35, 0.35, 0.35]
##
## written out as data so that the program below is only the transcription.

# The cost is x' Q x with Q diagonal, so the weights are all there is to it.
COST_WEIGHTS = np.array([2.0, 1.0, 4.0])

A_EQ = np.array([[1.0, 2.0, 3.0], [2.0, 7.0, 4.0]])
B_EQ = np.array([1.0, 1.0])

# The last constraint is on |x|, which cannot be written down directly: an
# absolute value is not linear.  It is two linear constraints, -b <= x <= b,
# and a two-sided bound on a variable is exactly what a bounding box is.
BOX = np.array([0.35, 0.35, 0.35])


def solve_the_qp() -> np.ndarray:
    """The optimal x for the problem above."""
    raise NotImplementedError("your code here")


######################################################################
## Testing
######################################################################


def test1():
    """The first scalar problem on its own: the bound is active at x* = 3."""
    x_star = scalar_example()
    print(f"scalar example:          x* = {x_star:.4f}")
    plot_scalar_problem(
        lambda x: x * x, x_star, (-1.0, 5.0), (3.0, None), "min x^2, x >= 3"
    )
    plt.tight_layout()
    if not HEADLESS:
        plt.show()


def test2():
    """Both scalar problems side by side, active bound then inactive."""
    x_star = scalar_example()
    print(f"scalar example:          x* = {x_star:.4f}")
    plt.subplot(1, 2, 1)
    plot_scalar_problem(
        lambda x: x * x, x_star, (-1.0, 5.0), (3.0, None), "min x^2, x >= 3"
    )

    x_star = scalar_example_interior()
    print(f"interior scalar example: x* = {x_star:.4f}")
    plt.subplot(1, 2, 2)
    plot_scalar_problem(
        lambda x: x * x - 2 * x,
        x_star,
        (-2.0, 4.0),
        (-1.0, 3.0),
        "min x^2 - 2x, -1 <= x <= 3",
    )
    plt.tight_layout()
    if not HEADLESS:
        plt.show()


def test3():
    print(f"vector example: x* = {np.round(vector_example(), 4)}")


def test4():
    """The answer to hand in."""
    x = solve_the_qp()
    print(f"   cost      {COST_WEIGHTS.dot(x * x):.4f}")
    print(f"x* = {np.round(x, 4)}")
    print(f"   A x - b   {np.round(A_EQ.dot(x) - B_EQ, 12)}")
    print(f"   |x| <= b  {np.all(np.abs(x) <= BOX + 1e-9)}")


if __name__ == "__main__":
    test1()
    #test2()
    #test3()
    #test4()
