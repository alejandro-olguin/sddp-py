"""Vehicle location (https://sddp.dev/stable/examples/vehicle_location/).

A faithful port of ``docs/src/examples/vehicle_location.jl``: an ambulance dispatch problem
on the number line 0..100 with bases at 20, 40, 60, 80, 100, three vehicles, ten stages and
a call location ``request in 0:10:100`` each stage. ``parameterize`` sets the stage objective.

Note: the Julia example's own check (``calculate_bound(model) >= 1000`` after 20 iterations
with ``cut_deletion_minimum = 100``) is commented out upstream with
``TODO(odow): find out why this fails``; see ``tests/test_vehicle_location.py``.
"""

from __future__ import annotations

from typing import Any

import sddp

HOSPITAL_LOCATION = 0
BASES = [HOSPITAL_LOCATION, 20, 40, 60, 80, 100]
VEHICLES = [1, 2, 3]
REQUESTS = list(range(0, 101, 10))


def shift_cost(src: int, dest: int) -> float:
    return abs(src - dest)


def dispatch_cost(base: int, request: int) -> float:
    return 2 * (abs(request - HOSPITAL_LOCATION) + abs(request - base))


def initial_state(b: int, v: int) -> float:
    """All ambulances start at the hospital."""
    return 1.0 if b == HOSPITAL_LOCATION else 0.0


def build_vehicle_location(stages: int = 10) -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        # Current location of each vehicle at each base.
        location = {
            (b, v): sp.add_state(
                f"location[{b},{v}]", lb=0.0, ub=1.0, initial_value=initial_state(b, v)
            )
            for b in BASES
            for v in VEHICLES
        }
        # Which vehicle is dispatched?
        dispatch = {
            (b, v): sp.add_variable(f"dispatch[{b},{v}]", binary=True)
            for b in BASES
            for v in VEHICLES
        }
        # Shifting vehicles between bases: [src, dest, vehicle]
        shift = {
            (b, d, v): sp.add_variable(f"shift[{b},{d},{v}]", binary=True)
            for b in BASES
            for d in BASES
            for v in VEHICLES
        }

        # Flow of vehicles in and out of bases:
        def base_balance(b: int, v: int) -> Any:
            return (
                location[(b, v)].in_
                - dispatch[(b, v)]
                - sum(shift[(b, d, v)] for d in BASES)
                + sum(shift[(s, b, v)] for s in BASES)
            )

        # Only one vehicle dispatched to call.
        sp.add_constraint(sum(dispatch.values()) == 1)
        for b in BASES:
            for v in VEHICLES:
                # Can only dispatch vehicle from base if vehicle is at that base.
                sp.add_constraint(dispatch[(b, v)] <= location[(b, v)].in_)
                # Can only shift vehicle if vehicle is at that src base.
                sp.add_constraint(sum(shift[(b, d, v)] for d in BASES) <= location[(b, v)].in_)
                # Can only shift vehicle if vehicle is not being dispatched.
                sp.add_constraint(sum(shift[(b, d, v)] for d in BASES) + dispatch[(b, v)] <= 1)
                # Can't shift to same base.
                sp.add_constraint(shift[(b, b, v)] == 0)
        # Update states for non-home/non-hospital bases.
        for b in BASES[1:]:
            for v in VEHICLES:
                sp.add_constraint(location[(b, v)].out == base_balance(b, v))
        # Update states for home/hospital bases.
        for v in VEHICLES:
            sp.add_constraint(
                location[(HOSPITAL_LOCATION, v)].out
                == base_balance(HOSPITAL_LOCATION, v) + sum(dispatch[(b, v)] for b in BASES)
            )

        @sp.parameterize(REQUESTS)
        def _(request: int) -> None:
            sp.set_stage_objective(
                sum(
                    # Distance to travel from base to emergency and then to hospital.
                    dispatch[(b, v)] * dispatch_cost(b, request)
                    # Distance travelled by vehicles relocating bases.
                    + sum(shift_cost(b, dest) * shift[(b, dest, v)] for dest in BASES)
                    for b in BASES
                    for v in VEHICLES
                )
            )

    return sddp.LinearPolicyGraph(builder, stages=stages, lower_bound=0.0, optimizer=sddp.HiGHS)


def vehicle_location_model(
    duality_handler: Any = None, iteration_limit: int = 20, seed: int | None = 1, **kwargs: Any
) -> sddp.PolicyGraph:
    """Train as on the page: 20 iterations, ``cut_deletion_minimum=100``."""
    model = build_vehicle_location()
    sddp.train(
        model,
        iteration_limit=iteration_limit,
        log_frequency=10,
        cut_deletion_minimum=100,
        duality_handler=duality_handler or sddp.ContinuousConicDuality(),
        seed=seed,
        **kwargs,
    )
    return model


def main() -> None:
    import time

    t = time.time()
    model = vehicle_location_model(sddp.ContinuousConicDuality())
    print("bound:", sddp.calculate_bound(model), f"({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
