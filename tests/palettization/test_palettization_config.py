# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import torch.nn as nn

from coreai_opt.palettization.config import (
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    OpKMeansPalettizerConfig,
)
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    default_weight_palettization_spec,
)
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec

DISABLED_MODULE_CONFIG = ModuleKMeansPalettizerConfig(
    op_input_spec=None,
    op_output_spec=None,
    op_state_spec=None,
    module_input_spec=None,
    module_output_spec=None,
    module_state_spec=None,
)

# ==================================
# Tests for OpKMeansPalettizerConfig
# ==================================


def test_op_kmeans_palettizer_config_default():
    """Test OpKMeansPalettizerConfig with default settings."""
    config = OpKMeansPalettizerConfig()

    # Should have default state spec for weight and in_proj_weight
    assert config.op_state_spec is not None
    assert config.op_state_spec["weight"] == default_weight_palettization_spec()
    assert config.op_state_spec["in_proj_weight"] == default_weight_palettization_spec()

    # Input and output specs should be empty (weight-only compression)
    assert config.op_input_spec == {}
    assert config.op_output_spec == {}


def test_op_kmeans_palettizer_config_custom_state_spec():
    """Test OpKMeansPalettizerConfig with custom state spec."""
    custom_spec = PalettizationSpec(n_bits=2)
    config = OpKMeansPalettizerConfig(op_state_spec={"weight": custom_spec, "bias": None})

    assert config.op_state_spec["weight"] == custom_spec
    assert config.op_state_spec["bias"] is None
    assert config.op_input_spec == {}
    assert config.op_output_spec == {}


def test_op_kmeans_palettizer_config_rejects_input_spec():
    """Test that OpKMeansPalettizerConfig rejects input specs."""
    with pytest.raises(ValueError, match="does not support op_input_spec"):
        OpKMeansPalettizerConfig(op_input_spec={0: PalettizationSpec(n_bits=4)})


def test_op_kmeans_palettizer_config_rejects_output_spec():
    """Test that OpKMeansPalettizerConfig rejects output specs."""
    with pytest.raises(ValueError, match="does not support op_output_spec"):
        OpKMeansPalettizerConfig(op_output_spec={0: PalettizationSpec(n_bits=4)})


def test_op_kmeans_palettizer_config_with_all_state_tensors():
    """Test OpKMeansPalettizerConfig with wildcard for all state tensors."""
    custom_spec = PalettizationSpec(n_bits=6)
    config = OpKMeansPalettizerConfig(op_state_spec={"*": custom_spec})

    assert config.op_state_spec["*"] == custom_spec
    assert config.op_input_spec == {}
    assert config.op_output_spec == {}


def test_op_kmeans_palettizer_config_disable_weight():
    """Test OpKMeansPalettizerConfig with weight palettization disabled."""
    config = OpKMeansPalettizerConfig(op_state_spec={"weight": None})

    assert config.op_state_spec["weight"] is None
    assert config.op_input_spec == {}
    assert config.op_output_spec == {}


# ======================================
# Tests for ModuleKMeansPalettizerConfig
# ======================================


def test_module_kmeans_palettizer_config_defaults():
    """Test ModuleKMeansPalettizerConfig with default settings."""
    config = ModuleKMeansPalettizerConfig()

    # Should have empty activation specs (weight-only compression)
    assert config.op_input_spec == {}
    assert config.op_output_spec == {}
    assert config.module_input_spec == {}
    assert config.module_output_spec == {}

    # Should have default op_state_spec
    spec = default_weight_palettization_spec()
    assert config.op_state_spec == {"weight": spec, "in_proj_weight": spec}

    # Should have default module_state_spec (empty)
    assert config.module_state_spec == {}

    # vectorize defaults to False (opt-in, unbenchmarked-for-most-block-sizes
    # algorithm swap; see ModuleKMeansPalettizerConfig's docstring).
    assert config.vectorize is False


def test_module_kmeans_palettizer_config_vectorize_reaches_kmeans_fake_palettize():
    """vectorize must survive _get_compressor_specific_settings() and land on the
    instantiated _KMeansFakePalettize, the same generic kwargs-passthrough path
    enable_fast_kmeans_mode/rounding_precision already use."""
    config = ModuleKMeansPalettizerConfig(vectorize=True)
    settings = config._get_compressor_specific_settings()
    assert settings["vectorize"] is True

    palettizer = _KMeansFakePalettize(
        n_bits=4,
        lut_qspec=None,
        granularity=PerGroupedChannelGranularity(axis=0, group_size=32),
        cluster_dim=1,
        enable_per_channel_scale=False,
        **settings,
    )
    assert palettizer.vectorize is True


def test_module_kmeans_palettizer_config_vectorize_with_cluster_dim_gt_1_does_not_raise():
    """Unlike enable_fast_kmeans_mode (genuinely unimplemented for cluster_dim > 1,
    and validated against), vectorize only ever affects the cluster_dim == 1 kmeans1d
    path; cluster_dim > 1 never calls it, so vectorize=True there is simply inert,
    not forbidden. Regression guard against an overzealous validator being added
    later by analogy with enable_fast_kmeans_mode's constraint."""
    ModuleKMeansPalettizerConfig(
        op_state_spec={"weight": PalettizationSpec(n_bits=4, cluster_dim=2)},
        enable_fast_kmeans_mode=False,
        vectorize=True,
    )


def test_module_kmeans_palettizer_config_with_op_state_spec():
    """Test ModuleKMeansPalettizerConfig with custom op_state_spec."""
    custom_spec = PalettizationSpec(n_bits=2)
    config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": custom_spec})

    assert config.op_state_spec["weight"] == custom_spec
    assert config.op_input_spec == {}
    assert config.op_output_spec == {}


def test_module_kmeans_palettizer_config_with_module_state_spec():
    """Test ModuleKMeansPalettizerConfig with module_state_spec."""
    custom_spec = PalettizationSpec(n_bits=6)
    config = ModuleKMeansPalettizerConfig(module_state_spec={"weight": custom_spec})

    assert config.module_state_spec["weight"] == custom_spec
    assert config.module_input_spec == {}
    assert config.module_output_spec == {}


def test_module_kmeans_palettizer_config_with_op_configs():
    """Test ModuleKMeansPalettizerConfig with op_type_config and op_name_config."""
    linear_op_config = OpKMeansPalettizerConfig(
        op_state_spec={"weight": PalettizationSpec(n_bits=4)}
    )

    config = ModuleKMeansPalettizerConfig(
        op_type_config={"aten.linear.default": linear_op_config},
        op_name_config={
            "special_op": OpKMeansPalettizerConfig(
                op_state_spec={"weight": PalettizationSpec(n_bits=2)}
            )
        },
    )

    assert len(config.op_type_config) == 1
    assert config.op_type_config["aten.linear.default"] == linear_op_config
    assert len(config.op_name_config) == 1
    assert config.op_name_config["special_op"].op_state_spec["weight"].n_bits == 2


def test_module_kmeans_palettizer_config_rejects_op_input_spec():
    """Test that ModuleKMeansPalettizerConfig rejects op_input_spec."""
    with pytest.raises(ValueError, match="does not support op_input_spec"):
        ModuleKMeansPalettizerConfig(op_input_spec={0: PalettizationSpec(n_bits=4)})


def test_module_kmeans_palettizer_config_rejects_op_output_spec():
    """Test that ModuleKMeansPalettizerConfig rejects op_output_spec."""
    with pytest.raises(ValueError, match="does not support op_output_spec"):
        ModuleKMeansPalettizerConfig(op_output_spec={0: PalettizationSpec(n_bits=4)})


def test_module_kmeans_palettizer_config_rejects_module_input_spec():
    """Test that ModuleKMeansPalettizerConfig rejects module_input_spec."""
    with pytest.raises(ValueError, match="does not support module_input_spec"):
        ModuleKMeansPalettizerConfig(module_input_spec={0: PalettizationSpec(n_bits=4)})


def test_module_kmeans_palettizer_config_rejects_module_output_spec():
    """Test that ModuleKMeansPalettizerConfig rejects module_output_spec."""
    with pytest.raises(ValueError, match="does not support module_output_spec"):
        ModuleKMeansPalettizerConfig(module_output_spec={0: PalettizationSpec(n_bits=4)})


def test_module_kmeans_palettizer_config_combined_state_specs():
    """Test ModuleKMeansPalettizerConfig with both op and module state specs."""
    op_spec = PalettizationSpec(n_bits=4)
    module_spec = PalettizationSpec(n_bits=2)

    config = ModuleKMeansPalettizerConfig(
        op_state_spec={"weight": op_spec}, module_state_spec={"bias": module_spec}
    )

    assert config.op_state_spec["weight"] == op_spec
    assert config.module_state_spec["bias"] == module_spec
    assert config.op_input_spec == {}
    assert config.op_output_spec == {}
    assert config.module_input_spec == {}
    assert config.module_output_spec == {}


# ======================================
# Tests for KMeansPalettizerConfig
# ======================================


def test_kmeans_palettizer_config_default():
    """Test KMeansPalettizerConfig with default settings."""
    config = KMeansPalettizerConfig()

    # Should have default global config
    assert config.global_config is not None
    assert isinstance(config.global_config, ModuleKMeansPalettizerConfig)
    assert config.global_config == ModuleKMeansPalettizerConfig()
    assert len(config.module_type_configs) == 0
    assert len(config.module_name_configs) == 0


def test_kmeans_palettizer_config_custom_global():
    """Test KMeansPalettizerConfig with custom global config."""
    custom_spec = PalettizationSpec(
        n_bits=6,
        lut_qspec=QuantizationSpec(dtype=torch.uint8, qscheme=QuantizationScheme.SYMMETRIC),
        granularity=PerGroupedChannelGranularity(axis=1, group_size=4),
    )

    global_config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": custom_spec})
    config = KMeansPalettizerConfig(global_config=global_config)

    assert config.global_config == global_config
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}


def test_kmeans_palettizer_config_module_type_configs():
    """Test KMeansPalettizerConfig with module type configurations."""
    linear_spec = PalettizationSpec(n_bits=4)
    conv_spec = PalettizationSpec(n_bits=8)

    linear_config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": linear_spec})
    conv_config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": conv_spec})

    config = KMeansPalettizerConfig(
        module_type_configs={
            nn.Linear: linear_config,
            nn.Conv2d: conv_config,
            nn.BatchNorm2d: None,  # Skip palettization
        }
    )

    assert len(config.module_type_configs) == 3
    # Module types are normalized to fully-qualified names
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == linear_config
    assert config.module_type_configs["torch.nn.modules.conv.Conv2d"] == conv_config
    assert (
        config.module_type_configs["torch.nn.modules.batchnorm.BatchNorm2d"]
        == DISABLED_MODULE_CONFIG
    )

    assert config.global_config == ModuleKMeansPalettizerConfig()
    assert config.module_name_configs == {}


def test_kmeans_palettizer_config_module_name_configs():
    """Test KMeansPalettizerConfig with module name configurations."""
    layer1_spec = PalettizationSpec(n_bits=2)
    layer2_spec = PalettizationSpec(n_bits=6)

    layer1_config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": layer1_spec})
    layer2_config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": layer2_spec})

    config = KMeansPalettizerConfig(
        module_name_configs={
            "layer1": layer1_config,
            "layer2": layer2_config,
            "skip_layer": None,  # Skip palettization
        }
    )

    assert len(config.module_name_configs) == 3
    assert config.module_name_configs["layer1"] == layer1_config
    assert config.module_name_configs["layer2"] == layer2_config
    assert config.module_name_configs["skip_layer"] == DISABLED_MODULE_CONFIG

    assert config.global_config == ModuleKMeansPalettizerConfig()
    assert config.module_type_configs == {}


def test_kmeans_palettizer_config_combined():
    """Test KMeansPalettizerConfig with all configuration types."""
    # Global config
    global_spec = PalettizationSpec(n_bits=4)
    global_config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": global_spec})

    # Type-specific configs
    linear_spec = PalettizationSpec(n_bits=8)
    linear_config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": linear_spec})
    # Name-specific configs
    special_spec = PalettizationSpec(n_bits=2)
    special_config = ModuleKMeansPalettizerConfig(op_state_spec={"weight": special_spec})

    config = KMeansPalettizerConfig(
        global_config=global_config,
        module_type_configs={nn.Linear: linear_config},
        module_name_configs={"special_layer": special_config},
    )

    assert config.global_config == global_config
    # Module types are normalized to fully-qualified names
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == linear_config
    assert config.module_name_configs["special_layer"] == special_config
