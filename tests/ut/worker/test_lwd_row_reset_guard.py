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
    # The action and wiring functions exist, and __init__ completes the wiring
    assert "def _lwd_reset_reused_row(" in src, "row-slot cleanup action function is missing"
    assert "def _lwd_reset_reused_rows_on_admission(" in src, "admission hook function is missing"
    assert "self._lwd_reset_reused_rows_on_admission()" in src, "not wired"
    # 2026-10-21 acceptance incident: initialize_kv_cache rebuilds the
    # InputBatch (always triggered by hybrid KV geometry), and the hook on
    # the old instance dies with it — it must be re-hooked after the rebuild
    assert src.count("def initialize_kv_cache(") == 1, "the override must exist exactly once"
    init_kv = src.index("def initialize_kv_cache(")
    assert src.index("self._lwd_reset_reused_rows_on_admission()", init_kv) > init_kv, (
        "initialize_kv_cache must re-hook at the end (root cause of the first acceptance failure)"
    )


def test_row_reset_semantics() -> None:
    """Flush only rows with ids=None, fill with vocab_size (padding semantics), native bookkeeping first."""
    src = RUNNER.read_text(encoding="utf-8")
    assert "if request.prompt_token_ids is not None:" in src, "rows carrying real ids must bail out early"
    assert "batch.token_ids_cpu[req_index, :num_prompt] = batch.vocab_size" in src, (
        "must fill with vocab_size (no-token semantics); no other value allowed"
    )
    # Wrapping order: native bookkeeping first, cleanup after (residue is flushed before any reader sees the row)
    wire = src.index("req_index = orig_add(request)")
    reset = src.index("self._lwd_reset_reused_row(request, req_index)")
    assert wire < reset, "cleanup must happen after native bookkeeping"
    # Instance-level hooking (does not touch the shared class); idempotency guard in place
    assert "batch.add_request = add_request_then_reset_row" in src
    assert 'getattr(batch, "_lwd_row_reset_hooked", False)' in src
