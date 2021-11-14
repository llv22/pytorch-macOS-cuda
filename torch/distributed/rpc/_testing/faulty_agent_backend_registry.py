#!/usr/bin/env python3

import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.distributed_c10d as dc10d
from torch.distributed.rpc import constants as rpc_constants

from datetime import timedelta

# def _init_process_group(store, rank, world_size):
#     # Initialize ProcessGroup.
#     process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT

#     # We're using a bunch of private APIs here since `new_group` requires the
#     # default group to be initialized.
#     group = dist.ProcessGroupGloo(store, rank, world_size, process_group_timeout)

#     assert group is not None, "Failed to initialize default ProcessGroup."

#     if (rank != -1) and (rank != group.rank()):
#         raise RuntimeError(
#             "rank argument {} doesn't match pg rank {}".format(rank, group.rank())
#         )
#     if (world_size != -1) and (world_size != group.size()):
#         raise RuntimeError(
#             "world_size argument {} doesn't match pg size {}".format(
#                 world_size, group.size()
#             )
#         )
#     return group


# def _faulty_tensorpipe_construct_rpc_backend_options_handler(
#     rpc_timeout,
#     init_method,
#     num_worker_threads,
#     messages_to_fail,
#     messages_to_delay,
#     num_fail_sends,
#     **kwargs
# ):
#     from . import FaultyTensorPipeRpcBackendOptions

#     return FaultyTensorPipeRpcBackendOptions(
#         num_worker_threads=num_worker_threads,
#         rpc_timeout=rpc_timeout,
#         init_method=init_method,
#         messages_to_fail=messages_to_fail,
#         messages_to_delay=messages_to_delay,
#         num_fail_sends=num_fail_sends,
#     )


# def _faulty_tensorpipe_init_backend_handler(
#     store, name, rank, world_size, rpc_backend_options
# ):
#     from . import FaultyTensorPipeAgent
#     from . import FaultyTensorPipeRpcBackendOptions
#     from torch.distributed.rpc import api

#     if not isinstance(store, dist.Store):
#         raise TypeError("`store` must be a c10d::Store. {}".format(store))

#     if not isinstance(
#         rpc_backend_options, FaultyTensorPipeRpcBackendOptions
#     ):
#         raise TypeError(
#             "`rpc_backend_options` must be a `FaultyTensorPipeRpcBackendOptions`. {}".format(
#                 rpc_backend_options
#             )
#         )

#     group = _init_process_group(store, rank, world_size)
#     agent = FaultyTensorPipeAgent(
#         store,
#         name,
#         rank,
#         world_size,
#         group,
#         rpc_backend_options,
#         {},  # reverse_device_map
#         [],  # devices
#     )
#     api._init_rpc_states(agent)

#     return agent

# rpc.backend_registry.register_backend(
#     "FAULTY_PROCESS_GROUP",
#     _faulty_process_group_construct_rpc_backend_options_handler,
#     _faulty_process_group_init_backend_handler,
# )

def _faulty_process_group_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_send_recv_threads,
    messages_to_fail,
    messages_to_delay,
    num_fail_sends,
    **kwargs
):
    from . import FaultyProcessGroupRpcBackendOptions

    return FaultyProcessGroupRpcBackendOptions(
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        num_send_recv_threads=num_send_recv_threads,
        messages_to_fail=messages_to_fail,
        messages_to_delay=messages_to_delay,
        num_fail_sends=num_fail_sends,
    )
    
def _faulty_process_group_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    from . import FaultyProcessGroupAgent

    if dist.is_initialized():
        raise RuntimeError(
            "Process group must not be initialized before init_rpc."
        )

    process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT

    dist.init_process_group(
        backend=dist.Backend.GLOO,
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=process_group_timeout,
    )

    try:
        group = dc10d._get_default_group()
        assert group is not None, "Failed to initialize default ProcessGroup."

        if (rank != -1) and (rank != group.rank()):
            raise RuntimeError(
                "rank argument {} doesn't match pg rank {}".format(rank, group.rank())
            )
        if (world_size != -1) and (world_size != group.size()):
            raise RuntimeError(
                "world_size argument {} doesn't match pg size {}".format(
                    world_size, group.size()
                )
            )

        return FaultyProcessGroupAgent(
            store,
            name,
            group,
            rpc_backend_options.num_send_recv_threads,
            timedelta(seconds=rpc_backend_options.rpc_timeout),
            rpc_backend_options.messages_to_fail,
            rpc_backend_options.messages_to_delay,
            rpc_backend_options.num_fail_sends,
        )
    except Exception as ex:
        dist.destroy_process_group()
        raise ex

rpc.backend_registry.register_backend(
    "FAULTY_PROCESS_GROUP",
    _faulty_process_group_construct_rpc_backend_options_handler,
    _faulty_process_group_init_backend_handler,
)
