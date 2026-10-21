#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Guard: the pre-recv endpoint gate must compare GLOBAL ranks.

2026-10-20 hang: the gate compared self.rank (the executor's per-node
numbering) against the topology's cloud leader rank (global). With one
edge card, global = local + 1, so cloud local rank 1 won the gate, its
comm thread attached the MQs, consumed the hint and died on "rank 2
holds no lwd_up channel"; the true endpoint (global rank 1) never ran
the thread, the watermark stayed -1 and the gate held every prefill
forever. Pure AST/source scan, no torch required.
"""

import re
from pathlib import Path

WORKER = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "worker"
    / "lwd_cloud_model"
    / "lwd_cloud_worker.py"
)


def test_endpoint_gate_uses_distributed_global_rank() -> None:
    src = WORKER.read_text(encoding="utf-8")
    # The endpoint check must take dist.get_rank() as the left-hand value; a bare
    # self.rank must never be compared against the topology's global rank
    assert re.search(
        r"global_rank\s*=\s*dist\.get_rank\(\)", src
    ), "endpoint gate must read the distributed global rank"
    assert re.search(
        r"global_rank\s*\n?\s*==\s*self\.vllm_config\.lwd_config\.topology"
        r"\.clouds\[0\]\.dp\[0\]\.ranks\[0\]",
        src,
    ), "the endpoint comparison must use global_rank, not self.rank"
    assert not re.search(
        r"self\.rank\s*\n?\s*==\s*self\.vllm_config\.lwd_config\.topology",
        src,
    ), "a bare self.rank (executor-local numbering) leaked back into the gate"


def test_endpoint_log_names_both_rank_domains() -> None:
    src = WORKER.read_text(encoding="utf-8")
    assert "local_rank=%d global_rank=%d endpoint=%s" in src, (
        "the worker banner must print both rank domains so a future "
        "mismatch is visible at a glance"
    )


def test_lwd_runners_have_no_stale_pcp_attribute() -> None:
    """2026-10-20: the first gated batch died on self.pcp_size (an old-base
    attribute name; the current base only has dcp_size; the prefill-side
    parallel degree lives in parallel_config.prefill_context_parallel_size)."""
    from pathlib import Path as P
    root = P(__file__).resolve().parents[3] / "vllm_ascend" / "worker"
    for pkg in ("lwd_cloud_model", "lwd_edge_model", "lwd_base_model"):
        for path in (root / pkg).rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            assert "self.pcp_size" not in src, (
                f"{path} still references the retired self.pcp_size; use "
                "parallel_config.prefill_context_parallel_size"
            )


def test_lwd_mtp_proposer_matches_base_draft_contract() -> None:
    """2026-10-20: the first gated batch died on an override signature missing
    sampling_metadata (the base extended the contract for probabilistic draft
    sampling; the migrated override stayed on the old signature). The override
    must accept and forward it."""
    from pathlib import Path as P
    proposer = P(__file__).resolve().parents[3] / "vllm_ascend" / "worker" / "lwd_cloud_model" / "lwd_mtp_proposer.py"
    src = proposer.read_text(encoding="utf-8")
    assert "sampling_metadata:" in src, "override lost the base-contract parameter"
    assert "            sampling_metadata,\n        )" in src, "override must forward it to super()"


def test_bincount_kernel_stays_upstream() -> None:
    """2026-10-21 decision: the kernel stays identical to ghdc/main (the
    >=0 guard c0ccd8215 was reverted). That guard was measured to stop the
    reproducible curl->bench crash, but the path feeding negative values
    into the kernel has been cut at the root by the runner-side row-slot
    cleanup (1e0d4a98b/0c413f54d), so the guard rollback keeps upstream
    as-is. Silent contamination of the same poison in the centralized
    kernel is handled via an upstream issue."""
    from pathlib import Path as P
    src = P(__file__).resolve().parents[3].joinpath(
        "vllm_ascend/ops/triton/bincount.py").read_text(encoding="utf-8")
    assert "(token >= 0) & (token < vocab_size)" not in src, (
        "upstream bincount.py was edited again; localize with logs first"
    )
    assert "token_in_range = pos_mask & (token < vocab_size)" in src
