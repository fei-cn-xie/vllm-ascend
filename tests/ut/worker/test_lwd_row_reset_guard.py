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
"""Guard: the reused-row reset must stay hooked on request admission.

2026-10-21 MTE root cause (closed loop, all links measured): LWD cloud
requests ship embeddings without token ids, so the native add_request
never overwrites the row's prompt region; the previous occupant's
residue (19 unrepaired async -1 placeholders from a penalty-free curl
request) was inherited by the next request's prompt region, copied
verbatim into the prompt-token tensor (the one penalty path without
masked_fill), and reached the bincount kernel where negative ids became
negative-offset atomic writes -> aivec fault on row 0 + segment
boundary. Native serves always overwrite the region with real ids,
which is why the port never crashed. The reset restores that missing
overwrite at admission. Pure source scan; see also the probe family
(lwd_sampler_probe) that captured the evidence.
"""

from pathlib import Path

RUNNER = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "worker"
    / "lwd_cloud_model"
    / "lwd_cloud_model_runner.py"
)


def test_row_reset_wired_into_admission() -> None:
    src = RUNNER.read_text(encoding="utf-8")
    # 动作与接线函数存在,且 __init__ 完成接线
    assert "def _lwd_reset_reused_row(" in src, "行槽清理动作函数缺失"
    assert "def _lwd_reset_reused_rows_on_admission(" in src, "入住挂接函数缺失"
    assert "self._lwd_reset_reused_rows_on_admission()" in src, "__init__ 未接线"


def test_row_reset_semantics() -> None:
    """只冲 ids=None 的行,填 vocab_size(padding 语义),原生入账先行。"""
    src = RUNNER.read_text(encoding="utf-8")
    assert "if request.prompt_token_ids is not None:" in src, "带真 ids 的行必须早退"
    assert "batch.token_ids_cpu[req_index, :num_prompt] = batch.vocab_size" in src, (
        "必须以 vocab_size 填充(无 token 语义),不得用其他值"
    )
    # 包装次序:原生入账在前,清理在后(任何读者见到行之前残渣已冲)
    wire = src.index("req_index = orig_add(request)")
    reset = src.index("self._lwd_reset_reused_row(request, req_index)")
    assert wire < reset, "清理必须发生在原生入账之后"
    # 实例级挂接(不碰共享类),幂等守卫在位
    assert "batch.add_request = add_request_then_reset_row" in src
    assert 'getattr(batch, "_lwd_row_reset_hooked", False)' in src
