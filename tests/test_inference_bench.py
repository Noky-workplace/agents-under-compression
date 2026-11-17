import json
import math
import time

import pytest

from inference.bench import report, vram
from inference.bench.gguf import GGUFError, read_header
from inference.bench.gpu import GpuSample, GpuSampler, GpuStats, parse_smi, wait_for_cooldown
from inference.bench.run_bench import auto_ngl, parse_bench_json, run_one

from .gguf_factory import write_gguf


@pytest.fixture
def model(tmp_path):
    return write_gguf(tmp_path / "tiny-Q4_K_M.gguf")


# ---- GGUF header -----------------------------------------------------------
def test_header_metadata_and_tensor_sizes(model):
    h = read_header(model)
    assert (h.architecture, h.n_layer, h.n_embd, h.n_head, h.n_head_kv) == ("qwen3", 4, 64, 8, 2)
    assert h.head_dim == 8
    assert h.block_bytes() == [4096] * 4
    assert h.non_block_bytes() == {"input": 1024, "output": 512}
    assert sum(t.nbytes for t in h.tensors) == h.file_size - h.data_offset
    assert "tokenizer.ggml.scores" not in h.metadata  # big arrays are skipped


def test_header_can_keep_arrays(model):
    h = read_header(model, keep_arrays=True)
    assert h.metadata["tokenizer.ggml.tokens"] == ["a", "bb", "ccc"]


def test_rejects_non_gguf(tmp_path):
    bad = tmp_path / "x.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(GGUFError):
        read_header(bad)


# ---- VRAM model ------------------------------------------------------------
def test_offload_takes_last_blocks_then_output_head(model):
    h = read_header(model)
    assert vram.offloaded_weight_bytes(h, 0) == 0
    assert vram.offloaded_weight_bytes(h, 3) == 3 * 4096
    assert vram.offloaded_weight_bytes(h, 4) == 4 * 4096
    assert vram.offloaded_weight_bytes(h, 5) == 4 * 4096 + 512
    assert vram.offloaded_weight_bytes(h, 99) == 4 * 4096 + 512


def test_kv_cache_formula(model):
    h = read_header(model)
    # 2 (K and V) * ctx * head_dim * n_head_kv * 2 bytes (f16) * layers
    assert vram.kv_cache_bytes(h, 1024, 4) == 2 * 1024 * 8 * 2 * 2 * 4
    assert vram.kv_cache_bytes(h, 1024, 4, "f32") == 2 * vram.kv_cache_bytes(h, 1024, 4)
    with pytest.raises(ValueError):
        vram.kv_cache_bytes(h, 1024, 4, "nope")


def test_estimate_is_monotonic_and_cpu_only_has_fixed_overhead(model):
    h = read_header(model)
    totals = [vram.estimate(h, n, 2048).total_mib for n in range(6)]
    assert totals[0] == vram.DEFAULT_OVERHEAD_MIB
    assert totals == sorted(totals)


def test_max_ngl_respects_budget(model):
    h = read_header(model)
    full = vram.estimate(h, 5, 2048, overhead_mib=0)
    assert vram.max_ngl(h, 2048, full.total_mib, overhead_mib=0).ngl == 5
    two = vram.estimate(h, 2, 2048, overhead_mib=0)
    assert vram.max_ngl(h, 2048, two.total_mib, overhead_mib=0).ngl == 2
    assert vram.max_ngl(h, 2048, 0.0001, overhead_mib=0).ngl == 0


def test_calibrate_overhead_recovers_constant(model):
    h = read_header(model)
    samples = []
    for ngl in (1, 3, 5):
        e = vram.estimate(h, ngl, 2048)
        samples.append((e, e.weight_scale * e.weights_mib + e.kv_mib + 432.0))
    assert vram.calibrate_overhead(samples) == pytest.approx(432.0)


def test_vram_cli(model, capsys):
    assert vram.main([str(model), "--ctx", "1024", "--budget-mib", "9999"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ngl"] == 5 and out["fits_fully"] is True


# ---- GPU telemetry ---------------------------------------------------------
def test_parse_smi_handles_na():
    s = parse_smi("5120, 8188, 71, [N/A], 96\n")
    assert (s.mem_used_mib, s.mem_total_mib, s.temp_c, s.util_pct) == (5120, 8188, 71, 96)
    assert math.isnan(s.power_w)


def test_stats_flag_over_temperature():
    st = GpuStats()
    st.add(GpuSample(1000, 8188, 60, 40, 10))
    st.add(GpuSample(3000, 8188, 86, 80, 99))
    assert st.peak_mem_mib == 3000 and st.over_temp and st.mean_power_w == 60


def test_sampler_calls_over_temp_hook_once():
    calls = []
    hot = GpuSample(1, 8188, 90, 50, 1)
    with GpuSampler(interval_s=0.01, reader=lambda: hot, on_over_temp=lambda: calls.append(1)) as s:
        time.sleep(0.08)
    assert calls == [1] and s.stats.samples >= 2


def test_cooldown_waits_until_target():
    temps = iter([80, 75, 64])
    slept = []
    last = wait_for_cooldown(
        target_c=65,
        reader=lambda: GpuSample(0, 0, next(temps), 0, 0),
        sleep=slept.append,
    )
    assert last == 64 and len(slept) == 2


# ---- llama-bench wrapper ---------------------------------------------------
BENCH_OUT = """log line before
[
  {"model_filename": "m.gguf", "n_prompt": 512, "n_gen": 0, "avg_ts": 1800.5, "stddev_ts": 12.0},
  {"model_filename": "m.gguf", "n_prompt": 0, "n_gen": 128, "avg_ts": 42.25, "stddev_ts": 0.5}
]
trailing log"""


def test_parse_bench_json():
    r = parse_bench_json(BENCH_OUT)
    assert r["pp_ts"] == 1800.5 and r["tg_ts"] == 42.25 and r["tg_n"] == 128
    with pytest.raises(ValueError):
        parse_bench_json("no json here")


def test_auto_ngl_covers_cpu_to_full():
    assert auto_ngl(36) == [0, 9, 18, 27, 36, 37]


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self._out, self._err, self.returncode = stdout, stderr, returncode
        self.terminated = False

    def communicate(self, timeout=None):
        return self._out, self._err

    def terminate(self):
        self.terminated = True

    def kill(self):
        pass


COOL = GpuSample(900, 8188, 55, 30, 5)


def _run(model, proc, sample=COOL):
    return run_one(
        model,
        4,
        popen=lambda *a, **k: proc,
        gpu_reader=lambda: sample,
        cooldown=lambda reader: 55.0,
    )


def test_run_one_success(model):
    rec = _run(model, FakeProc(stdout=BENCH_OUT))
    assert rec["tg_ts"] == 42.25 and "error" not in rec
    assert rec["ngl"] == 4 and rec["predicted_mib"] > 0 and rec["baseline_mib"] == 900


def test_run_one_reports_oom(model):
    rec = _run(model, FakeProc(stderr="ggml: CUDA error: out of memory", returncode=1))
    assert rec["error"] == "oom" and "tg_ts" not in rec


def test_run_one_aborts_when_too_hot(model):
    proc = FakeProc(stdout=BENCH_OUT)
    rec = _run(model, proc, sample=GpuSample(900, 8188, 91, 30, 5))
    # the sampler may or may not catch a sample before the fake exits; if it did, it must abort
    if rec["over_temp"]:
        assert proc.terminated and "over temperature" in rec["error"]


# ---- report ----------------------------------------------------------------
def test_report_table_and_overhead_fit(tmp_path):
    rows = [
        {"model": "m", "file_mib": 5000, "ngl": 0, "pp_ts": 50, "tg_ts": 5.0, "used_mib": 0,
         "predicted_mib": 0, "predicted_weights_mib": 0, "predicted_kv_mib": 0,
         "peak_temp_c": 50, "mean_power_w": 20},
        {"model": "m", "file_mib": 5000, "ngl": 37, "pp_ts": 1800, "tg_ts": 42.0, "used_mib": 5600,
         "predicted_mib": 5500, "predicted_weights_mib": 4900, "predicted_kv_mib": 100,
         "peak_temp_c": 74, "mean_power_w": 95},
        {"model": "m", "file_mib": 5000, "ngl": 37, "error": "oom", "used_mib": 0,
         "predicted_weights_mib": 4900, "predicted_kv_mib": 100},
    ]  # fmt: skip
    path = tmp_path / "bench.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
    loaded = report.latest(report.load(path))
    md = report.to_markdown(loaded)
    assert "| m | 5000 | 37 | 1800 | 42.0 | 5600 |" in md
    assert report.overhead_fit(loaded) == {
        "overhead_mib": 600.0, "mae_mib": 0.0, "max_err_mib": 0.0, "n": 1,
    }  # fmt: skip
    assert report.overhead_fit([rows[2]]) is None


def test_cpu_calibration_and_impossible_budget(model):
    h = read_header(model)
    e = vram.estimate(h, 0, 1024)
    assert vram.calibrate_overhead([(e, 972)]) == 972
    with pytest.raises(ValueError, match="fixed overhead"):
        vram.max_ngl(h, 1024, 900)
    with pytest.raises(ValueError, match="at least one sample"):
        vram.calibrate_overhead([])


def test_weight_fit_and_held_out_errors():
    rows = [
        {
            "ngl": n,
            "n_layer": 2,
            "predicted_weights_mib": w,
            "predicted_kv_mib": k,
            "predicted_mib": old,
            "used_mib": 100 + 0.8 * w + k,
        }
        for n, w, k, old in [(0, 0, 0, 0), (1, 50, 5, 555), (3, 100, 10, 610)]
    ]
    fit = report.weight_fit(rows + [{"error": "oom"}])
    assert fit == pytest.approx({"overhead_mib": 100, "weight_scale": 0.8})
    held_out = [dict(rows[1], used_mib=152)]
    errors = report.model_errors(held_out, **fit)
    assert errors["partial"]["after_max"] == pytest.approx(7)
    assert errors["partial"]["before_mae"] == 403
    with pytest.raises(ValueError, match="distinct"):
        report.weight_fit(rows[:1])
    with pytest.raises(ValueError, match="successful"):
        report.weight_fit([])


def test_recorded_fit_matches_defaults():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "research/results/phase2"
    fit = report.weight_fit(report.load(root / "bench.jsonl"))
    assert fit == pytest.approx(
        {"overhead_mib": vram.DEFAULT_OVERHEAD_MIB, "weight_scale": vram.DEFAULT_WEIGHT_SCALE}
    )
    errors = report.model_errors(report.load(root / "bench-rerun.jsonl"), **fit)
    assert errors["full"]["after_max"] == pytest.approx(329.864004623985, abs=0.01)


def test_report_labels_historical_predictions_and_scaled_fit():
    rows = [
        {
            "model": "tiny",
            "file_mib": 100,
            "ngl": n,
            "used_mib": 100 + 0.8 * w,
            "predicted_weights_mib": w,
            "predicted_kv_mib": 0,
        }
        for n, w in [(0, 0), (1, 50)]
    ]
    text = report.to_markdown(rows)
    assert "fixed 100.0 MiB, weight scale 0.800000" in text
    assert "predictions recorded at benchmark time" in text
