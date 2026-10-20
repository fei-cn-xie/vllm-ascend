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

    def probed_entry(tokens, vocab_size, num_seqs=None):
        # 值检查(.min/.max/.item)是 D2H 同步点,会扰动 async 占位覆写
        # 时序,已去除;只留宿主侧元数据行(零设备读)。
        lwd_logger.info(
            "[LWD][sampler-probe] bincount launch shape=%s stride=%s "
            "ptr=0x%x vocab=%d num_seqs=%s",
            tuple(tokens.shape), tokens.stride(),
            tokens.data_ptr(), vocab_size, num_seqs,
        )
        return orig_entry(tokens, vocab_size, num_seqs)

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
                lwd_logger.warning(
                    "[LWD][sampler-probe] PENALTY INPUT POISON "
                    "rows=%d/%d spec_rows=%d",
                    poison_rows, len(output_token_ids), spec_poison_rows,
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
