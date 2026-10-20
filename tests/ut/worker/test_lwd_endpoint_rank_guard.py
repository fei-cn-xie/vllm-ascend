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

import pytest

WORKER = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "worker"
    / "lwd_cloud_model"
    / "lwd_cloud_worker.py"
)


def test_endpoint_gate_uses_distributed_global_rank() -> None:
    src = WORKER.read_text(encoding="utf-8")
    # 端点判定必须以 dist.get_rank() 为左值,禁止裸 self.rank 与拓扑全局 rank 相比
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
    """2026-10-20:首个过闸批次死于 self.pcp_size(老底座属性名,现底座只有
    dcp_size;prefill 侧并行度住 parallel_config.prefill_context_parallel_size)。"""
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
    """2026-10-20:首个过闸批次死于覆写签名缺 sampling_metadata(底座为
    概率草稿采样扩了契约,迁移覆写停留在旧签名)。覆写必须接受并透传。"""
    from pathlib import Path as P
    proposer = P(__file__).resolve().parents[3] / "vllm_ascend" / "worker" / "lwd_cloud_model" / "lwd_mtp_proposer.py"
    src = proposer.read_text(encoding="utf-8")
    assert "sampling_metadata:" in src, "override lost the base-contract parameter"
    assert "            sampling_metadata,\n        )" in src, "override must forward it to super()"


def test_bincount_kernel_filters_negative_tokens() -> None:
    """2026-10-20 aivec 终审翻案(证据齐备,替代 14abf6abb 的上游钉子):
    惩罚输入 spec 段实测携带 -1 哨兵(移植版基线 4136 行 / LWD 4364 行,
    全 spec 段,analyzer --compare 终审两侧同毒);上游 GPU 版靠
    vocab_size+1 吸收列容纳 -1,移植版无吸收列,内核必须自带下界,
    否则每次带惩罚推理静默污染邻座计数(中地址彩票时 aivec fault)。"""
    from pathlib import Path as P
    src = P(__file__).resolve().parents[3].joinpath(
        "vllm_ascend/ops/triton/bincount.py").read_text(encoding="utf-8")
    assert "token_in_range = pos_mask & (token >= 0) & (token < vocab_size)" in src, (
        "bincount 下界守卫被移除;-1 哨兵将穿透为负偏移原子写"
    )


def test_bincount_kernel_ignores_negative_sentinels() -> None:
    """2026-10-20 aivec 回归(执行级,复现 bug 本体):惩罚输入携带 -1
    哨兵(async+MTP 数据契约,两侧实测 4136/4364 行)时,内核必须忽略
    它们且计数正确——修复前 -1 穿透为负偏移原子写(row0+段边界时
    aivec fault,否则静默污染邻座计数)。
    需 NPU 环境;轻量环境跳过,真实环境补跑(本文件其余用例为纯
    源码扫描,任何主机可跑)。"""
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        pytest.skip("需 NPU 环境(torch + torch_npu)补跑")
    from vllm_ascend.ops.triton.bincount import (
        get_token_bin_counts_and_mask_triton,
    )

    vocab = 100
    tokens = torch.tensor(
        [
            [5, -1, 7, 100, 5],  # -1 哨兵 + vocab padding
            [-1, -1, 3, 99, 0],
        ],
        dtype=torch.int32,
        device="npu",
    )
    counts, mask = get_token_bin_counts_and_mask_triton(tokens, vocab, 2)
    counts_cpu = counts.cpu()
    # 正常 token 计数正确
    assert int(counts_cpu[0, 5]) == 2 and int(counts_cpu[0, 7]) == 1
    assert int(counts_cpu[1, 3]) == 1 and int(counts_cpu[1, 99]) == 1
    assert int(counts_cpu[1, 0]) == 1
    # -1 与 vocab padding 均不计入任何桶(修复前 -1 写到桶外)
    assert int(counts_cpu.sum()) == 5
    assert not bool(mask[0, 0].cpu()) and bool(mask[0, 5].cpu())


def test_sampler_probe_installs_in_worker_process() -> None:
    """2026-10-20:探针必须装在 worker 进程(内核在这里发射);引擎进程
    的模块补丁对 spawn 出的 worker 无效。"""
    from pathlib import Path as P
    src = P(__file__).resolve().parents[3].joinpath(
        "vllm_ascend/worker/lwd_cloud_model/lwd_cloud_worker.py"
    ).read_text(encoding="utf-8")
    assert "def _lwd_install_sampler_probe" in src, "worker probe function missing"
    assert "self._lwd_install_sampler_probe()" in src, "probe not called in init_device"
    assert "bincount launch" in src, "kernel launch log format missing"
    assert "OUT-OF-RANGE" in src, "out-of-range escalation missing"
