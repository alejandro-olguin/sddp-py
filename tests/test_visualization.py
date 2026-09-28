import math

import pytest

import sddp
from tests.problems import build_hydro_thermal

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_publication_data_and_plots(tmp_path):
    model = build_hydro_thermal()
    sddp.train(model, iteration_limit=5, print_level=0, seed=1)
    sims = sddp.simulate(model, 20, ["volume"], seed=2)
    data = sddp.publication_data(sims, [0.0, 0.5, 1.0], lambda s: s["volume"].out)
    assert data.shape == (3, 3)
    assert all(data[0, t] <= data[1, t] <= data[2, t] for t in range(3))
    ax = sddp.publication_plot(sims, lambda s: s["stage_objective"], title="objective")
    ax.figure.savefig(tmp_path / "pub.png")
    ax2 = sddp.spaghetti_plot(sims, lambda s: s["volume"].out)
    ax2.figure.savefig(tmp_path / "spag.png")
    assert (tmp_path / "pub.png").stat().st_size > 0
    with pytest.raises(ValueError, match="not finite"):
        sddp.publication_data(sims, [0.5], lambda s: math.nan)
