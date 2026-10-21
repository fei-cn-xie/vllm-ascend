# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""2026-10-20 aivec 事故观测探针(采样惩罚/分箱内核侧)。

从 LwdCloudWorker 原地抽出为独立安装器,供两处共用同一份仪器:
- LWD 云 worker(init_device 调用);
- 原生昇腾服务(NPUWorker 以 VLLM_LWD_SAMPLER_PROBE=1 启动时调用),
  用于"移植版基线 vs LWD"的归属终审对比——同仪器才有可比性。

只做宿主侧元数据/纯 Python 列表统计,零设备同步(early 2026-10-20 版本
的 .min()/.max()/.item() 会排空设备队列、掩盖时序,已去除)。
根因定位后整体删除。
"""

from __future__ import annotations

import os


def install_lwd_sampler_probe() -> None:
    """Idempotent per worker process; patches the penalty kernel entry."""
    import vllm_ascend.ops.triton.penalty as penalty_mod
    from vllm.lwd_logger import lwd_logger
    from vllm_ascend.sample import sampler as sampler_mod

    if getattr(penalty_mod, "_lwd_probe_installed", False):
        return
    penalty_mod._lwd_probe_installed = True

    budget = {"left": int(os.getenv("VLLM_LWD_SAMPLER_PROBE_BUDGET", "200"))}
    orig_entry = penalty_mod.get_token_bin_counts_and_mask_triton
    # 首次 launch 一次性打显存池快照(此时 HCCL/wire 通信域已建好),
    # 用于对比两侧(LWD 9-rank 共享世界 vs 原生独立世界)的分配器布局。
    first_launch = {"done": False}
    # 2026-10-21 崩溃取证模式(VLLM_LWD_CRASH_HUNT=1):同步读出每次内核
    # 输入的最小值/负值个数,按 prompt/output 配对标注(apply_penalties
    # 每轮先 prompt 后 output 各调一次)。带 D2H 同步,仅在崩溃可复现的
    # 取证期开,平时保持关闭零开销。
    crash_hunt = os.getenv("VLLM_LWD_CRASH_HUNT", "0") == "1"
    hunt_seq = {"n": 0}

    def probed_entry(tokens, vocab_size, num_seqs=None):
        if crash_hunt:
            hunt_seq["n"] += 1
            t_min = int(tokens.min().item())
            t_neg = int((tokens < 0).sum().item())
            lwd_logger.warning(
                "[LWD][sampler-probe] CRASH HUNT seq=%d half=%s min=%d "
                "neg=%d shape=%s",
                hunt_seq["n"],
                "prompt" if hunt_seq["n"] % 2 == 1 else "output",
                t_min, t_neg, tuple(tokens.shape),
            )
        # 值检查(.min/.max/.item)是 D2H 同步点,会扰动 async 占位覆写
        # 时序,已去除;只留宿主侧元数据行(零设备读)。
        lwd_logger.info(
            "[LWD][sampler-probe] bincount launch shape=%s stride=%s "
            "ptr=0x%x vocab=%d num_seqs=%s",
            tuple(tokens.shape), tokens.stride(),
            tokens.data_ptr(), vocab_size, num_seqs,
        )
        if not first_launch["done"]:
            first_launch["done"] = True
            try:
                import torch

                lwd_logger.info(
                    "[LWD][sampler-probe] bincount first-launch mem "
                    "reserved=%d allocated=%d",
                    torch.npu.memory_reserved(),
                    torch.npu.memory_allocated(),
                )
            except Exception:  # pragma: no cover - npu 内存 API 差异
                pass
        result = orig_entry(tokens, vocab_size, num_seqs)
        # 2026-10-20 崩溃不对称排查:负偏移写是否 fault 取决于 bin_counts
        # 是否落在分配器段边界。记录返回的 counts 实际地址与 2MB 段对齐
        # 标志(段起点对齐),对比两侧分配布局;data_ptr 是宿主元数据,
        # 零设备同步。守卫修复后该测量仍有效(量的是分配,不是 fault)。
        counts_ptr = int(result[0].data_ptr())
        lwd_logger.info(
            "[LWD][sampler-probe] bincount return counts_ptr=0x%x align2m=%d",
            counts_ptr, 1 if counts_ptr % (2 << 20) == 0 else 0,
        )
        return result

    penalty_mod.get_token_bin_counts_and_mask_triton = probed_entry

    orig_apply = sampler_mod.AscendSampler.apply_penalties

    def probed_apply(logits, sampling_metadata, output_token_ids):
        # 惩罚输入 = [*output(修复链管), *spec] 拼接;此处清点修复后输入
        # 与 spec 段各自的 -1 行——"毒是否到达内核"的真判据。纯 Python
        # 列表扫描,零设备同步。
        if not sampling_metadata.no_penalties and budget["left"] > 0:
            budget["left"] -= 1
            prompt_ids = sampling_metadata.prompt_token_ids
            poison_rows = sum(1 for row in output_token_ids if -1 in row)
            spec_ids_meta = sampling_metadata.spec_token_ids
            spec_poison_rows = (
                sum(1 for row in spec_ids_meta if -1 in row)
                if spec_ids_meta else 0
            )
            if poison_rows:
                # 2026-10-20 崩溃不对称排查:负偏移写仅当毒行=批内第 0 行
                # 且 bin_counts 落段边界才 fault(行号>=1 恒静默写前一行
                # 尾部)。min_idx=0 即"第 0 行带毒"这一崩溃前置的直接读数,
                # 用于对比 LWD(可复现崩溃)与原生基线(不崩)的行序差异。
                poison_idx = [
                    i for i, r in enumerate(output_token_ids) if -1 in r
                ]
                lwd_logger.warning(
                    "[LWD][sampler-probe] PENALTY INPUT POISON "
                    "rows=%d/%d spec_rows=%d min_idx=%d idx_sample=%s",
                    poison_rows, len(output_token_ids), spec_poison_rows,
                    min(poison_idx), poison_idx[:8],
                )
            lwd_logger.info(
                "[LWD][sampler-probe] penalties ACTIVE logits=%s "
                "prompt_ids=%s poison_rows=%d spec_poison=%d",
                tuple(logits.shape),
                ("None" if prompt_ids is None
                 else tuple(prompt_ids.shape)),
                poison_rows, spec_poison_rows,
            )
        return orig_apply(logits, sampling_metadata, output_token_ids)

    sampler_mod.AscendSampler.apply_penalties = staticmethod(probed_apply)
    lwd_logger.info(
        "[LWD][sampler-probe] installed on worker (budget=%d)",
        budget["left"],
    )
