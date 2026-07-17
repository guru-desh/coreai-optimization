# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy

import pytest
import torch
from torch.nn.utils.parametrize import is_parametrized

from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec


def _collect_fake_palettize_modules(model: torch.nn.Module) -> list[_KMeansFakePalettize]:
    """Return all _KMeansFakePalettize parametrizations attached to ``model``."""
    fps: list[_KMeansFakePalettize] = []
    for module in model.modules():
        if not hasattr(module, "parametrizations"):
            continue
        for parametrizations in module.parametrizations.values():
            for p in parametrizations:
                if isinstance(p, _KMeansFakePalettize):
                    fps.append(p)
    return fps


def _make_config(spec: PalettizationSpec) -> KMeansPalettizerConfig:
    return KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(op_state_spec={"weight": spec})
    )


class TestCrossLayerParallel:
    """Verify cross-layer parallel centroid calculation matches sequential output."""

    @pytest.mark.parametrize(
        "spec",
        [
            PalettizationSpec(n_bits=2, granularity=PerTensorGranularity(), cluster_dim=1),
            PalettizationSpec(
                n_bits=4,
                granularity=PerGroupedChannelGranularity(group_size=2, axis=0),
                cluster_dim=1,
            ),
            # lut_qspec populates quantized_lut / lut_quantization_scale /
            # lut_quantization_zero_point buffers, which must survive the
            # worker -> main state_dict round-trip.
            PalettizationSpec(
                n_bits=4,
                granularity=PerTensorGranularity(),
                cluster_dim=1,
                lut_qspec=QuantizationSpec(
                    dtype=torch.int8,
                    qscheme=QuantizationScheme.SYMMETRIC,
                ),
            ),
            # enable_per_channel_scale populates the per_channel_scale buffer,
            # which must also survive the round-trip.
            PalettizationSpec(
                n_bits=4,
                granularity=PerGroupedChannelGranularity(group_size=2, axis=0),
                cluster_dim=1,
                enable_per_channel_scale=True,
            ),
        ],
        ids=[
            "per_tensor",
            "per_grouped_channel",
            "lut_quantized",
            "per_channel_scale",
        ],
    )
    def test_parallel_matches_sequential(self, simple_conv_linear_model, simple_model_input, spec):
        """``num_workers > 1`` must produce identical per-module state to ``num_workers=1``."""
        config = _make_config(spec)

        seq_model = copy.deepcopy(simple_conv_linear_model)
        par_model = copy.deepcopy(simple_conv_linear_model)

        seq_palettizer = KMeansPalettizer(seq_model, config)
        seq_palettizer.prepare((simple_model_input,), num_workers=1)

        par_palettizer = KMeansPalettizer(par_model, config)
        par_palettizer.prepare((simple_model_input,), num_workers=2)

        seq_fps = _collect_fake_palettize_modules(seq_palettizer._model)
        par_fps = _collect_fake_palettize_modules(par_palettizer._model)

        assert len(seq_fps) == len(par_fps) > 0
        for seq_fp, par_fp in zip(seq_fps, par_fps, strict=True):
            assert seq_fp.lut is not None and par_fp.lut is not None
            assert seq_fp.indices is not None and par_fp.indices is not None
            torch.testing.assert_close(seq_fp.lut, par_fp.lut)
            torch.testing.assert_close(seq_fp.indices, par_fp.indices)

            # Optional buffers populated only when the corresponding feature
            # is enabled. Check each matches (including whether it was set).
            for buf_name in (
                "quantized_lut",
                "lut_quantization_scale",
                "lut_quantization_zero_point",
                "per_channel_scale",
            ):
                seq_buf = getattr(seq_fp, buf_name)
                par_buf = getattr(par_fp, buf_name)
                assert (seq_buf is None) == (par_buf is None), (
                    f"{buf_name} set in one path but not the other"
                )
                if seq_buf is not None:
                    assert torch.equal(seq_buf, par_buf)

        with torch.no_grad():
            seq_out = seq_model(simple_model_input)
            par_out = par_model(simple_model_input)
        torch.testing.assert_close(seq_out, par_out)

    def test_parallel_disabled_module(self, simple_conv_linear_model, simple_model_input):
        """A layer incompatible with the granularity must be not have FakePalettize.

        ``simple_conv_linear_model`` has ``conv`` (32 output channels) and
        ``linear`` (10 output channels). Under ``group_size=16``, the linear
        layer raises ``_IncompatibleGranularityError`` and is marked
        ``_disabled=True``, while the conv layer palettizes normally. This
        exercises ``_disabled`` propagation from worker to main process. After that
        the disabled FakePalett modules should be removed in prepare.
        """
        spec = PalettizationSpec(
            n_bits=4,
            granularity=PerGroupedChannelGranularity(group_size=16, axis=0),
            cluster_dim=1,
        )
        config = _make_config(spec)

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,), num_workers=2)

        assert is_parametrized(prepared_model.conv)
        assert not is_parametrized(prepared_model.linear)

    def test_parallel_with_no_palettize_modules_is_noop(
        self, simple_conv_linear_model, simple_model_input
    ):
        """``_calculate_centroids_parallel`` returns cleanly when nothing to palettize."""
        # global_config=None disables palettization for all modules.
        config = KMeansPalettizerConfig(global_config=None)
        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        # Should not raise; falls through with zero modules collected.
        palettizer.prepare((simple_model_input,), num_workers=2)

    def test_parallel_vector_palettization_matches_sequential(self):
        """Vector palettization (``cluster_dim > 1``) matches sequential when clusters are obvious.

        ``_cluster_weights_2d`` runs ``_EfficientKMeans._kmeans_pp``, which uses
        unseeded ``torch.randint`` / ``np.random.choice`` — so spawned workers
        and the main process see different RNG states. To make the comparison
        deterministic anyway, the linear weight is hand-built so that, after
        ``_vectorize`` (transpose + reshape into 2D pairs), the vectors form 4
        well-separated clusters. K-means converges to those 4 centers from any
        reasonable initialization, so both paths produce the same reconstructed
        weights (up to a cluster-ID permutation), and the model output matches.
        """
        # 4 well-separated centers in 2D, 4 vectors per center -> 16 (N, 2) vectors.
        centers = torch.tensor([[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0], [10.0, -10.0]])
        vectors = centers.repeat_interleave(4, dim=0)  # (16, 2)
        weight = vectors.reshape(8, 4).transpose(0, 1).contiguous()

        class _LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 4, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(weight)

            def forward(self, x):
                return self.linear(x)

        example_input = torch.randn(2, 8)
        spec = PalettizationSpec(
            n_bits=2,  # 4 palettes, matching the 4 hand-built clusters
            granularity=PerTensorGranularity(),
            cluster_dim=2,
        )
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": spec},
                enable_fast_kmeans_mode=False,
            )
        )

        seq_model = _LinearModel()
        par_model = _LinearModel()

        KMeansPalettizer(seq_model, config).prepare((example_input,), num_workers=1)
        KMeansPalettizer(par_model, config).prepare((example_input,), num_workers=2)

        with torch.no_grad():
            seq_out = seq_model(example_input)
            par_out = par_model(example_input)

        torch.testing.assert_close(seq_out, par_out)

    def test_invalid_num_workers_raises(self, simple_conv_linear_model, simple_model_input):
        """``num_workers < 1`` must raise ``ValueError`` instead of silently falling through."""
        config = _make_config(
            PalettizationSpec(n_bits=4, granularity=PerTensorGranularity(), cluster_dim=1)
        )
        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            palettizer.prepare((simple_model_input,), num_workers=0)

    def test_calibration_mode_respects_prepare_num_workers(
        self, simple_conv_linear_model, simple_model_input
    ):
        """``calibration_mode``'s recompute uses the ``num_workers`` from ``prepare()``.

        Runs calibration on two copies of the same model -- one prepared with
        ``num_workers=1``, the other with ``num_workers=2`` -- and asserts the
        post-calibration LUTs, indices, observer/fake_palett state, and model
        outputs match. Same calibration inputs are used for both so any
        divergence would point to the recompute path itself rather than RNG.
        """
        config = _make_config(
            PalettizationSpec(n_bits=2, granularity=PerTensorGranularity(), cluster_dim=1)
        )
        target = torch.randint(0, 10, (1,))

        seq_model = copy.deepcopy(simple_conv_linear_model)
        par_model = copy.deepcopy(simple_conv_linear_model)

        seq_palettizer = KMeansPalettizer(seq_model, config)
        seq_palettizer.prepare((simple_model_input,), num_workers=1)
        with seq_palettizer.calibration_mode(loss_fn=torch.nn.functional.cross_entropy) as skm:
            skm.step(seq_model(simple_model_input), target)

        par_palettizer = KMeansPalettizer(par_model, config)
        par_palettizer.prepare((simple_model_input,), num_workers=2)
        with par_palettizer.calibration_mode(loss_fn=torch.nn.functional.cross_entropy) as skm:
            skm.step(par_model(simple_model_input), target)

        seq_fps = _collect_fake_palettize_modules(seq_palettizer._model)
        par_fps = _collect_fake_palettize_modules(par_palettizer._model)
        assert len(seq_fps) == len(par_fps) > 0
        for seq_fp, par_fp in zip(seq_fps, par_fps, strict=True):
            assert seq_fp.lut is not None and par_fp.lut is not None
            torch.testing.assert_close(seq_fp.lut, par_fp.lut)
            torch.testing.assert_close(seq_fp.indices, par_fp.indices)

            # After calibration_mode exits, both paths should have restored
            # fake_palett=on / observer=off. The parallel path swaps fp_modules
            # into parametrization slots, so the subsequent apply(_enable_fake_palett)
            # and apply(_disable_observer) must reach the new modules.
            assert seq_fp.fake_palett_enabled.item() == 1
            assert par_fp.fake_palett_enabled.item() == 1
            assert seq_fp.observer_enabled.item() == 0
            assert par_fp.observer_enabled.item() == 0

        with torch.no_grad():
            torch.testing.assert_close(
                seq_palettizer._model(simple_model_input),
                par_palettizer._model(simple_model_input),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestWholeModelBatching:
    """Verify vectorize=True modules routed through the whole-model GPU
    batching path (``_cluster_group_batched``, bypassing the multiprocessing
    pool) produce results consistent with the sequential per-module path,
    and that SMAWK modules sharing a model with vectorize=True modules are
    unaffected -- see the batching investigation's Phase 14."""

    def test_mixed_vectorize_and_smawk_matches_sequential(
        self, simple_conv_linear_model, simple_model_input
    ):
        """``simple_conv_linear_model``'s conv (SMAWK) and linear
        (vectorize=True, multi-block) must both match a fully-sequential
        (``num_workers=1``) run when palettized in parallel."""
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={
                    "weight": PalettizationSpec(
                        n_bits=4,
                        granularity=PerGroupedChannelGranularity(group_size=2, axis=0),
                        cluster_dim=1,
                    )
                },
                vectorize=True,
            ),
            module_type_configs={
                torch.nn.Conv2d: ModuleKMeansPalettizerConfig(
                    op_state_spec={
                        "weight": PalettizationSpec(
                            n_bits=4, granularity=PerTensorGranularity(), cluster_dim=1
                        )
                    },
                    vectorize=False,
                ),
            },
        )

        seq_model = copy.deepcopy(simple_conv_linear_model)
        par_model = copy.deepcopy(simple_conv_linear_model)

        seq_palettizer = KMeansPalettizer(seq_model, config)
        seq_palettizer.prepare((simple_model_input,), num_workers=1)

        par_palettizer = KMeansPalettizer(par_model, config)
        par_palettizer.prepare((simple_model_input,), num_workers=2)

        for name in ("conv", "linear"):
            seq_fp = _collect_fake_palettize_modules(dict(seq_model.named_modules())[name])[0]
            par_fp = _collect_fake_palettize_modules(dict(par_model.named_modules())[name])[0]
            assert seq_fp.lut is not None and par_fp.lut is not None
            torch.testing.assert_close(seq_fp.indices, par_fp.indices)
            torch.testing.assert_close(seq_fp.lut, par_fp.lut, rtol=1e-6, atol=1e-6)

        with torch.no_grad():
            seq_out = seq_model(simple_model_input)
            par_out = par_model(simple_model_input)
        torch.testing.assert_close(seq_out, par_out, rtol=1e-6, atol=1e-6)

    def test_multiple_vectorize_modules_sharing_k_matches_sequential(
        self, gated_mlp_model, gated_mlp_model_input
    ):
        """``gated_mlp_model``'s three Linear layers (gate_proj, up_proj,
        down_proj) all share n_bits, so the whole-model batching path groups
        all of them into a single cross-module batched CUDA kernel call.
        Must still match the fully-sequential (``num_workers=1``) output."""
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={
                    "weight": PalettizationSpec(
                        n_bits=4,
                        granularity=PerGroupedChannelGranularity(group_size=8, axis=0),
                        cluster_dim=1,
                    )
                },
                vectorize=True,
            )
        )

        seq_model = copy.deepcopy(gated_mlp_model)
        par_model = copy.deepcopy(gated_mlp_model)

        seq_palettizer = KMeansPalettizer(seq_model, config)
        seq_palettizer.prepare((gated_mlp_model_input,), num_workers=1)

        par_palettizer = KMeansPalettizer(par_model, config)
        par_palettizer.prepare((gated_mlp_model_input,), num_workers=2)

        for name in ("gate_proj", "up_proj", "down_proj"):
            seq_fp = _collect_fake_palettize_modules(dict(seq_model.named_modules())[name])[0]
            par_fp = _collect_fake_palettize_modules(dict(par_model.named_modules())[name])[0]
            torch.testing.assert_close(seq_fp.indices, par_fp.indices)
            torch.testing.assert_close(seq_fp.lut, par_fp.lut, rtol=1e-6, atol=1e-6)

        with torch.no_grad():
            seq_out = seq_model(gated_mlp_model_input)
            par_out = par_model(gated_mlp_model_input)
        torch.testing.assert_close(seq_out, par_out, rtol=1e-6, atol=1e-6)

    def test_smawk_module_unaffected_by_sibling_vectorize_modules(
        self, simple_conv_linear_model, simple_model_input
    ):
        """The conv (SMAWK, pool-dispatched) module must produce identical
        results whether or not a sibling vectorize=True module is present
        in the same model -- isolation check for Phase 14's partition of
        ``_calculate_centroids_parallel``'s dispatch."""
        conv_spec = ModuleKMeansPalettizerConfig(
            op_state_spec={
                "weight": PalettizationSpec(
                    n_bits=4, granularity=PerTensorGranularity(), cluster_dim=1
                )
            },
            vectorize=False,
        )
        conv_only_config = KMeansPalettizerConfig(
            global_config=None,
            module_type_configs={torch.nn.Conv2d: conv_spec},
        )
        mixed_config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={
                    "weight": PalettizationSpec(
                        n_bits=4,
                        granularity=PerGroupedChannelGranularity(group_size=2, axis=0),
                        cluster_dim=1,
                    )
                },
                vectorize=True,
            ),
            module_type_configs={torch.nn.Conv2d: conv_spec},
        )

        conv_only_model = copy.deepcopy(simple_conv_linear_model)
        mixed_model = copy.deepcopy(simple_conv_linear_model)

        KMeansPalettizer(conv_only_model, conv_only_config).prepare(
            (simple_model_input,), num_workers=2
        )
        KMeansPalettizer(mixed_model, mixed_config).prepare((simple_model_input,), num_workers=2)

        conv_only_fp = _collect_fake_palettize_modules(conv_only_model.conv)[0]
        mixed_fp = _collect_fake_palettize_modules(mixed_model.conv)[0]
        torch.testing.assert_close(conv_only_fp.lut, mixed_fp.lut)
        torch.testing.assert_close(conv_only_fp.indices, mixed_fp.indices)
