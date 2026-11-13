import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.fused_moe.moe_comm_method import _append_cann_megamoe_dummy_tokens
from vllm_ascend import utils
from vllm_ascend.utils import get_cann_megamoe_buffer_params


def test_dummy_routes_cover_all_experts_across_ep_ranks():
    routed_experts = []
    for ep_rank_id in range(4):
        hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        active_mask = torch.tensor([1, 0], dtype=torch.int8)

        hidden_states, topk_ids, topk_weights, active_mask, original_num_tokens = _append_cann_megamoe_dummy_tokens(
            hidden_states,
            topk_ids,
            topk_weights,
            active_mask,
            num_experts=8,
            ep_rank_id=ep_rank_id,
            ep_world_size=4,
        )

        assert original_num_tokens == 2
        assert torch.equal(hidden_states[-1], torch.ones(4, dtype=torch.bfloat16))
        assert torch.equal(topk_weights[-1], torch.full((2,), 0.5))
        assert active_mask.tolist() == [1, 0, 1]
        routed_experts.extend(topk_ids[-1].tolist())

    assert sorted(routed_experts) == list(range(8))


def test_receive_bound_uses_documented_worst_case():
    assert get_cann_megamoe_buffer_params(480, 32, 256, 8) == (512, 8, 32, 131072)


@pytest.mark.parametrize(
    "args",
    [
        (4090, 8, 256, 8),
        (480, 3, 256, 8),
        (480, 8, 256, 17),
    ],
)
def test_buffer_params_reject_unsupported_a2_shapes(args):
    with pytest.raises(ValueError):
        get_cann_megamoe_buffer_params(*args)


def test_a2_mode_2_selects_megamoe(monkeypatch):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_current_hardware_profile", lambda: SimpleNamespace(moe_comm_policy=None))
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=2,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            hidden_size=4096,
            moe_intermediate_size=1536,
            num_experts_per_tok=8,
            quantize="w8a8_dynamic",
        ),
        get_hidden_size=lambda: 4096,
        get_num_experts=lambda: 256,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        lora_config=None,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=32,
            pipeline_parallel_size=1,
        ),
    )

    assert afc.select_moe_comm_method(512, vllm_config) == MoECommType.FUSED_MC2


def test_cann_92_buffer_helper_is_loaded_from_nested_module(monkeypatch):
    def helper():
        pass
    modules = {
        "cann_ops_transformer.ops": SimpleNamespace(),
        "cann_ops_transformer.ops.mega_moe": SimpleNamespace(),
        "cann_ops_transformer.ops.mc2.mega_moe": SimpleNamespace(),
        "cann_ops_transformer.ops.mc2.mega_moe.mega_moe": SimpleNamespace(
            _get_mega_moe_ccl_buffer_size=helper
        ),
    }

    def fake_import_module(name):
        if name in modules:
            return modules[name]
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    assert utils._load_cann_megamoe_ccl_buffer_size() is helper


def test_cann_92_buffer_helper_receives_intermediate_hidden(monkeypatch):
    calls = {}

    def helper(
        ep_world_size,
        moe_expert_num,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        **kwargs,
    ):
        calls.update(
            ep_world_size=ep_world_size,
            moe_expert_num=moe_expert_num,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            kwargs=kwargs,
        )
        return 128

    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            hidden_size=4096,
            moe_intermediate_size=1536,
            num_experts_per_tok=8,
        ),
        get_hidden_size=lambda: 4096,
        get_num_experts=lambda: 256,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(world_size_across_dp=32, pipeline_parallel_size=1, tensor_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=960),
    )

    import vllm.config as vllm_config_module

    monkeypatch.setattr(utils, "_load_cann_megamoe_ccl_buffer_size", lambda: helper)
    monkeypatch.setattr(vllm_config_module, "get_current_vllm_config", lambda: vllm_config)

    assert utils.calculate_cann_megamoe_hccl_buffer_size() == 128
    assert calls["intermediate_hidden"] == 2 * (1536 // 2)
    assert calls["kwargs"]["max_recv_token_num"] == (960 // 2 + 32) * 32 * 8
