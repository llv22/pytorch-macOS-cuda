# Owner(s): ["oncall: distributed"]

import copy
import sys
from itertools import chain
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard, replicate
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin
from torch.utils._pytree import tree_all, tree_all_only


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestStateDict(DTensorTestBase, VerifyStateDictMixin):
    """Tests state_dict and load_state_dict"""

    @property
    def world_size(self) -> int:
        return 2

    def _test_save_load(
        self,
        init_model_optim: Callable,
        test_frozen: bool = False,
    ) -> None:
        options = StateDictOptions(ignore_frozen_params=test_frozen)
        # Initialize original model and distributed model.
        model, optim, copy_optim, dist_model, dist_optim = init_model_optim()

        # Train 10 steps.
        for i in range(10):
            batch = torch.rand(8, 100, device="cuda")
            model(batch).sum().backward()
            optim.step()
            dist_model(batch).sum().backward()
            if not isinstance(dist_optim, list):
                dist_optim.step()
                dist_optim.zero_grad()
            else:
                for _dist_optim in dist_optim:
                    _dist_optim.zero_grad()
            optim.zero_grad()

        # Get the state_dict, and compare the result
        msd = model.state_dict()
        osd = optim.state_dict()
        dist_msd, dist_osd = get_state_dict(
            dist_model, optimizers=dist_optim, options=options
        )
        self._verify_msd(msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        self._verify_osd(model, optim, osd, dist_osd)

        # Initialize a completely new model to simulate checkpoint load.
        _, _, _, dist_model, dist_optim = init_model_optim()

        # Simulate DCP distributed load. We need to first get the state_dict and
        # pass them to DCP to load the saved state_dict from the storage.
        # Then finally we can call set_state_dict().
        if not isinstance(dist_optim, list):
            dist_optim = [dist_optim]
        curr_dist_msd, curr_dist_osd = get_state_dict(
            dist_model, optimizers=dist_optim, options=options
        )
        if test_frozen:
            # We won't be able to load the partial state_dict back.
            return
        # Since we already have the state_dict saved before, no need to call DCP.
        # We can directly load them back. This asser is to ensure that optimizer
        # state storage are initialized.
        # self.assertEqual(len(curr_dist_osd[STATE]), len(dist_osd[STATE]))
        set_state_dict(
            dist_model,
            optimizers=dist_optim,
            model_state_dict=dist_msd,
            optim_state_dict=dist_osd,
            options=options,
        )

        # Check if the new state_dict are the same
        dist_msd, dist_osd = get_state_dict(
            dist_model, optimizers=dist_optim, options=options
        )
        self._verify_msd(msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        self._verify_osd(model, optim, osd, dist_osd)

        # Test _patch_model_state_dict, and _patch_optimizer_state_dict
        _patch_model_state_dict(dist_model, options=options)
        _patch_optimizer_state_dict(dist_model, optimizers=dist_optim, options=options)
        dist_msd = dist_model.state_dict()
        dist_osd = dist_optim[0].state_dict()
        self._verify_msd(msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        self._verify_osd(model, optim, osd, dist_osd)

    def _test_fsdp(
        self,
        *,
        use_orig_params: bool,
        use_composable: bool,
        use_dtensor: bool,
        wrapping: Tuple[nn.Module] = (),
    ) -> None:
        if not use_orig_params and use_composable:
            return

        # TODO: remove this return after we complete the composable API side change for device_mesh
        if use_composable and use_dtensor:
            return

        def init_model_optim():
            if use_dtensor:
                device_mesh = init_device_mesh("cuda", (self.world_size,))

            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            if wrapping:
                strategy = set(wrapping)
            else:
                strategy = {UnitModule}
            if use_composable:
                dist_model = fully_shard(
                    copy.deepcopy(orig_model), policy=ModuleWrapPolicy(strategy)
                )
            else:
                if use_dtensor:
                    device_mesh = init_device_mesh("cuda", (self.world_size,))
                    dist_model = FSDP(
                        copy.deepcopy(orig_model),
                        auto_wrap_policy=ModuleWrapPolicy(strategy),
                        use_orig_params=use_orig_params,
                        device_mesh=device_mesh,
                    )
                else:
                    dist_model = FSDP(
                        copy.deepcopy(orig_model),
                        auto_wrap_policy=ModuleWrapPolicy(strategy),
                        use_orig_params=use_orig_params,
                    )

            dist_optim = torch.optim.Adam(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp(self) -> None:
        self.run_subtests(
            {
                "use_orig_params": [True, False],
                "use_composable": [True, False],
                "use_dtensor": [True, False],
                "wrapping": [tuple(), (nn.Linear, UnitModule)],
            },
            self._test_fsdp,
        )

    def _test_ddp(self, use_composable: bool) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            if use_composable:
                dist_model = replicate(copy.deepcopy(orig_model))
            else:
                dist_model = DDP(copy.deepcopy(orig_model))
            dist_optim = torch.optim.Adam(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_ddp(self) -> None:
        self.run_subtests(
            {"use_composable": [True, False]},
            self._test_ddp,
        )

    def _test_fsdp_ddp(
        self,
        use_composable: bool,
        optim_in_backward: bool = False,
        test_frozen: bool = False,
    ) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            if test_frozen:
                for param in chain(
                    orig_model.u1.parameters(), orig_model.u2.parameters()
                ):
                    param.requires_grad = False
            orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            dist_model = copy.deepcopy(orig_model)
            if use_composable:
                replicate(dist_model.l)
                fully_shard(dist_model, policy=ModuleWrapPolicy({UnitModule}))
            else:
                dist_model.l = DDP(dist_model.l)
                dist_model = FSDP(
                    copy.deepcopy(orig_model),
                    auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                    use_orig_params=optim_in_backward,
                    ignored_modules=[dist_model.l],
                )
            if optim_in_backward:
                _apply_optimizer_in_backward(
                    torch.optim.Adam, dist_model.parameters(), {"lr": 1e-3}
                )
                dist_optim = [
                    p._in_backward_optimizers[0] for p in dist_model.parameters()
                ]
            else:
                dist_optim = torch.optim.Adam(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim, test_frozen)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp_ddp(self) -> None:
        self.run_subtests(
            {"use_composable": [True, False]},
            self._test_fsdp_ddp,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_frozen_parameters(self) -> None:
        self._test_fsdp_ddp(use_composable=False, test_frozen=True)

    # TODO: enable use_dtensor once 2D device_mesh support is fully landed.
    """
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_use_dtensor(self) -> None:
        self._test_fsdp_ddp(use_composable=False, use_dtensor=True)
    """

    # TODO: enable the test after FSDP + apply_optimizer_in_backward works.
    # Disable this test as it is broken after
    # https://github.com/pytorch/pytorch/pull/108298.
    """
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_apply_optimizer_in_backward(self) -> None:
        self.run_subtests(
            {"use_composable": [True, False]},
            self._test_fsdp_ddp,
            optim_in_backward=True,
        )
    """

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_single_gpu(self) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            model_copy = copy.deepcopy(orig_model)
            optim_copy = torch.optim.Adam(model_copy.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, model_copy, optim_copy

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_strict(self) -> None:
        model = CompositeParamModel(device=torch.device("cuda"))

        model_state_dict = get_model_state_dict(model)
        key = next(iter(model_state_dict.keys()))
        model_state_dict["abc"] = torch.zeros(10)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            set_model_state_dict(model, model_state_dict=model_state_dict)
        model_state_dict.pop(key)
        incompatible_keys = set_model_state_dict(
            model,
            model_state_dict=model_state_dict,
            options=StateDictOptions(strict=False),
        )
        self.assertEqual(incompatible_keys.missing_keys, [key])
        self.assertEqual(incompatible_keys.unexpected_keys, ["abc"])
        model_state_dict.pop("abc")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            set_model_state_dict(model, model_state_dict=model_state_dict)

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_partial(self) -> None:
        model = CompositeParamModel(device=torch.device("cuda"))

        model_state_dict1 = get_model_state_dict(model)
        model_state_dict1 = copy.deepcopy(model_state_dict1)
        model_state_dict2 = get_model_state_dict(model, submodules={model.l})
        model_state_dict2 = copy.deepcopy(model_state_dict2)
        model_state_dict3 = get_model_state_dict(
            model,
            submodules={model.l},
            options=StateDictOptions(keep_submodule_prefixes=False),
        )
        model_state_dict3 = copy.deepcopy(model_state_dict3)
        self.assertEqual(len(model_state_dict2), 2)
        self.assertEqual(len(model_state_dict3), 2)
        for key in model_state_dict3.keys():
            full_fqn = f"l.{key}"
            value1 = model_state_dict1[full_fqn]
            value2 = model_state_dict2[full_fqn]
            value3 = model_state_dict3[key]
            self.assertEqual(value1, value2)
            self.assertEqual(value2, value3)

        zeros_state_dict = {
            k: torch.zeros_like(v) for k, v in model_state_dict1.items()
        }
        model.load_state_dict(zeros_state_dict)
        set_model_state_dict(
            model,
            model_state_dict=model_state_dict2,
            options=StateDictOptions(strict=False),
        )
        self.assertEqual(model.l.weight, model_state_dict1["l.weight"])
        self.assertEqual(model.l.bias, model_state_dict1["l.bias"])

        model.load_state_dict(zeros_state_dict)
        set_model_state_dict(
            model,
            model_state_dict={model.l: model_state_dict3},
            options=StateDictOptions(strict=False),
        )
        self.assertEqual(model.l.weight, model_state_dict1["l.weight"])
        self.assertEqual(model.l.bias, model_state_dict1["l.bias"])

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_cpu_offload_full_state_dict(self) -> None:
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        dist_model = FSDP(
            copy.deepcopy(orig_model),
            auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
            use_orig_params=True,
            device_mesh=device_mesh,
        )

        dist_optim = torch.optim.Adam(dist_model.parameters(), lr=1e-3)

        mst, ost = get_state_dict(
            dist_model,
            dist_optim,
            options=StateDictOptions(cpu_offload=True),
        )

        cpu_device = torch.device("cpu")

        def is_cpu(v):
            if isinstance(v, DTensor):
                return v.device == cpu_device
            elif isinstance(v, ShardedTensor):
                shards = v.local_shards()
                if not shards:
                    return True
                return shards[0].tensor.device == cpu_device
            else:
                return v.device == cpu_device

        self.assertTrue(
            tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, mst)
        )
        self.assertTrue(
            tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, ost)
        )

        mst, ost = get_state_dict(
            dist_model, dist_optim, options=StateDictOptions(full_state_dict=True)
        )

        self.assertTrue(
            tree_all(lambda v: not isinstance(v, (DTensor, ShardedTensor)), mst)
        )
        self.assertTrue(
            tree_all(lambda v: not isinstance(v, (DTensor, ShardedTensor)), ost)
        )

        mst, ost = get_state_dict(
            dist_model,
            dist_optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        if self.rank == 0:
            self.assertTrue(
                tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, mst)
            )
            self.assertTrue(
                tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, ost)
            )
        else:
            self.assertEqual(mst, {})
            self.assertEqual(ost, {})


if __name__ == "__main__":
    run_tests()