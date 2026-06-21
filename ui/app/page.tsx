"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import {
  CATALOG_GPUS,
  CATALOG_MODELS,
  DTYPES,
  TRAFFIC_CLASSES,
  type ScenarioCreate,
} from "@/lib/types";

// ── Custom model form — HuggingFace lookup ────────────────────────────────────

interface HFSpec {
  name: string;
  display_name: string;
  num_layers: number;
  d_model: number;
  num_q_heads: number;
  num_kv_heads: number;
  head_dim: number;
  context_len: number;
  native_dtype: string;
  weight_bytes_per_param: number;
  kv_dtype_bytes: number;
  geometry_source: "estimated";
  resident_weights_gb?: number;
  total_params: number;
  active_params: number;
  is_moe: boolean;
  num_experts?: number;
  experts_per_token?: number;
}

function fmtParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return String(n);
}

function CustomModelFields({
  onChange,
}: {
  onChange: (spec: Record<string, unknown> | null) => void;
}) {
  const [hfId, setHfId] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [spec, setSpec] = useState<HFSpec | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);

  const handleFetch = async () => {
    const trimmed = hfId.trim();
    if (!trimmed || !trimmed.includes("/")) {
      setFetchError('Enter a HuggingFace model ID like "meta-llama/Meta-Llama-3-8B"');
      return;
    }
    setLoading(true);
    setFetchError(null);
    setSpec(null);
    setWarnings([]);
    onChange(null);

    try {
      const headers: Record<string, string> = {};
      if (hfToken.trim()) headers["x-hf-token"] = hfToken.trim();

      const res = await fetch(
        `/api/hf-config?model=${encodeURIComponent(trimmed)}`,
        { headers }
      );
      const data = await res.json() as
        | { spec: HFSpec; warnings: string[] }
        | { error: string; message: string };

      if (!res.ok || "error" in data) {
        const msg = "message" in data ? data.message : `HTTP ${res.status}`;
        setFetchError(msg);
        return;
      }

      setSpec(data.spec);
      setWarnings(data.warnings);
      onChange(data.spec as unknown as Record<string, unknown>);
    } catch {
      setFetchError("Network error — could not reach the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-2 space-y-3">
      {/* HF model ID row */}
      <div className="flex gap-2">
        <input
          className="flex-1 border rounded-lg px-3 py-2 text-sm font-mono"
          placeholder="owner/model-name  e.g. meta-llama/Meta-Llama-3-8B"
          value={hfId}
          onChange={(e) => { setHfId(e.target.value); setSpec(null); setFetchError(null); onChange(null); }}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); handleFetch(); } }}
        />
        <button
          type="button"
          onClick={handleFetch}
          disabled={loading || !hfId.trim()}
          className="px-4 py-2 rounded-lg text-sm bg-brand-600 text-white hover:bg-brand-700 disabled:opacity-40 transition-colors"
        >
          {loading ? "Fetching…" : "Fetch"}
        </button>
      </div>

      {/* Gated model token toggle */}
      <div>
        <button
          type="button"
          onClick={() => setShowToken(!showToken)}
          className="text-xs text-gray-500 hover:text-gray-700"
        >
          {showToken ? "▾ Hide token" : "▸ Gated model? Add HF token"}
        </button>
        {showToken && (
          <input
            type="password"
            className="mt-1 w-full border rounded-lg px-3 py-2 text-sm font-mono"
            placeholder="hf_..."
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
          />
        )}
      </div>

      {/* Error */}
      {fetchError && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          {fetchError}
        </p>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <ul className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 space-y-1 list-disc list-inside">
          {warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}

      {/* Geometry preview */}
      {spec && (
        <div className="p-3 bg-green-50 border border-green-200 rounded-lg text-xs space-y-1">
          <p className="font-semibold text-green-800 text-sm">{spec.display_name}</p>
          <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-gray-700 mt-1">
            <span>Params: <strong>{fmtParams(spec.total_params)}</strong>{spec.is_moe ? ` (${fmtParams(spec.active_params)} active)` : ""}</span>
            {spec.resident_weights_gb != null && (
              <span>Weights: <strong>{spec.resident_weights_gb.toFixed(1)} GB</strong></span>
            )}
            <span>Layers: <strong>{spec.num_layers}</strong></span>
            <span>d_model: <strong>{spec.d_model}</strong></span>
            <span>Q heads: <strong>{spec.num_q_heads}</strong></span>
            <span>KV heads: <strong>{spec.num_kv_heads}</strong></span>
            <span>head_dim: <strong>{spec.head_dim}</strong></span>
            <span>Context: <strong>{(spec.context_len / 1024).toFixed(0)}K</strong></span>
            <span>Dtype: <strong>{spec.native_dtype}</strong></span>
          </div>
          <p className="text-amber-600 mt-1">Geometry is estimated — confidence will be capped at MEDIUM.</p>
        </div>
      )}
    </div>
  );
}

// ── Scenario Builder (screen 1) ───────────────────────────────────────────────

export default function ScenarioBuilder() {
  const router = useRouter();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelMode, setModelMode] = useState<"catalog" | "custom">("catalog");

  const [form, setForm] = useState<ScenarioCreate>({
    name: "my-deployment",
    model_name: "llama-3.1-8b",
    gpu_name: "h100_sxm",
    dtype: "bf16",
    requests_per_day: 1_000_000,
    peak_multiplier: 3,
    isl: 2048,
    osl: 256,
    ttft_slo_ms: 2000,
    tp: 1,
    traffic_class: "realtime",
    gpu_mem_util: 0.9,
    runtime: "vllm",
  });

  const set = (k: keyof ScenarioCreate, v: unknown) =>
    setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const scenario = await api.createScenario(form);
      router.push(`/estimate?sid=${scenario.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-1">Scenario Builder</h1>
      <p className="text-gray-500 text-sm mb-6">
        Describe your workload. We'll size replicas, estimate cost, and generate a benchmark plan.
      </p>

      <form onSubmit={handleSubmit} className="space-y-6">

        {/* ── Name ── */}
        <div>
          <label className="block text-sm font-medium mb-1">Scenario name</label>
          <input
            className="w-full border rounded-lg px-3 py-2 text-sm"
            value={form.name}
            onChange={(e) => set("name", e.target.value)}
            required
          />
        </div>

        {/* ── GPU (dropdown only) ── */}
        <div>
          <label className="block text-sm font-medium mb-1">
            GPU <span className="text-gray-400 font-normal">(curated catalog — new GPUs added by maintainer)</span>
          </label>
          <select
            className="w-full border rounded-lg px-3 py-2 text-sm"
            value={form.gpu_name}
            onChange={(e) => set("gpu_name", e.target.value)}
          >
            {CATALOG_GPUS.map((g) => (
              <option key={g.key} value={g.key}>{g.label}</option>
            ))}
          </select>
        </div>

        {/* ── Model (dropdown + custom) ── */}
        <div>
          <label className="block text-sm font-medium mb-1">Model</label>
          <div className="flex gap-2 mb-2">
            <button
              type="button"
              onClick={() => setModelMode("catalog")}
              className={`px-3 py-1 rounded text-sm ${modelMode === "catalog" ? "bg-brand-600 text-white" : "bg-white border"}`}
            >
              Catalog
            </button>
            <button
              type="button"
              onClick={() => setModelMode("custom")}
              className={`px-3 py-1 rounded text-sm ${modelMode === "custom" ? "bg-brand-600 text-white" : "bg-white border"}`}
            >
              Custom
            </button>
          </div>
          {modelMode === "catalog" ? (
            <select
              className="w-full border rounded-lg px-3 py-2 text-sm"
              value={form.model_name}
              onChange={(e) => set("model_name", e.target.value)}
            >
              {CATALOG_MODELS.map((m) => (
                <option key={m.key} value={m.key}>{m.label}</option>
              ))}
            </select>
          ) : (
            <CustomModelFields onChange={(spec) => {
              if (spec) set("model_name", JSON.stringify(spec));
            }} />
          )}
        </div>

        {/* ── Dtype + TP + GPU mem util ── */}
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Serving dtype</label>
            <select
              className="w-full border rounded-lg px-3 py-2 text-sm"
              value={form.dtype}
              onChange={(e) => set("dtype", e.target.value)}
            >
              {DTYPES.map((d) => <option key={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Tensor parallel</label>
            <input
              type="number" min={1} max={64}
              className="w-full border rounded-lg px-3 py-2 text-sm"
              value={form.tp}
              onChange={(e) => set("tp", parseInt(e.target.value))}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              GPU mem util
              <span className="ml-1 text-gray-400 font-normal text-xs" title="Fraction of VRAM reserved for weights + KV cache (default 0.9). Lower if you see OOM errors.">?</span>
            </label>
            <input
              type="number" min={0.5} max={1.0} step={0.05}
              className="w-full border rounded-lg px-3 py-2 text-sm"
              value={form.gpu_mem_util}
              onChange={(e) => set("gpu_mem_util", parseFloat(e.target.value))}
            />
          </div>
        </div>

        {/* ── Traffic ── */}
        <fieldset className="border rounded-lg p-4">
          <legend className="text-sm font-medium px-1">Traffic</legend>
          <div className="grid grid-cols-2 gap-4 mt-2">
            <div>
              <label className="block text-sm mb-1">Requests/day</label>
              <input
                type="number" min={1}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.requests_per_day}
                onChange={(e) => set("requests_per_day", parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Peak multiplier</label>
              <input
                type="number" min={1} step={0.5}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.peak_multiplier}
                onChange={(e) => set("peak_multiplier", parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">ISL (input tokens)</label>
              <input
                type="number" min={1}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.isl}
                onChange={(e) => set("isl", parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">OSL (output tokens)</label>
              <input
                type="number" min={1}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.osl}
                onChange={(e) => set("osl", parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">TTFT SLO (ms)</label>
              <input
                type="number" min={100}
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.ttft_slo_ms}
                onChange={(e) => set("ttft_slo_ms", parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Traffic class</label>
              <select
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.traffic_class}
                onChange={(e) => set("traffic_class", e.target.value)}
              >
                {TRAFFIC_CLASSES.map((tc) => (
                  <option key={tc.key} value={tc.key}>{tc.label}</option>
                ))}
              </select>
            </div>
          </div>
        </fieldset>

        {/* ── Advanced serving config ── */}
        <fieldset className="border rounded-lg p-4">
          <legend className="text-sm font-medium px-1">Advanced <span className="text-gray-400 font-normal">(optional)</span></legend>
          <div className="grid grid-cols-3 gap-4 mt-2">
            <div>
              <label className="block text-sm mb-1">
                Prefix cache len
                <span className="ml-1 text-gray-400 text-xs" title="Shared prefix length in tokens (e.g. system prompt). Reduces prefill compute; KV budget unchanged.">?</span>
              </label>
              <input
                type="number" min={0}
                placeholder="0"
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.prefix_cache_len ?? ""}
                onChange={(e) => set("prefix_cache_len", e.target.value === "" ? undefined : parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">
                Cache hit rate
                <span className="ml-1 text-gray-400 text-xs" title="Fraction of requests that hit the prefix cache (0.0–1.0).">?</span>
              </label>
              <input
                type="number" min={0} max={1} step={0.05}
                placeholder="0.0"
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.prefix_cache_hit_rate ?? ""}
                onChange={(e) => set("prefix_cache_hit_rate", e.target.value === "" ? undefined : parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">
                Max batch size
                <span className="ml-1 text-gray-400 text-xs" title="vLLM --max-num-seqs: scheduler concurrency cap, independent of KV budget.">?</span>
              </label>
              <input
                type="number" min={1}
                placeholder="unlimited"
                className="w-full border rounded-lg px-3 py-2 text-sm"
                value={form.max_num_seqs ?? ""}
                onChange={(e) => set("max_num_seqs", e.target.value === "" ? undefined : parseInt(e.target.value))}
              />
            </div>
          </div>
        </fieldset>

        {error && (
          <p className="text-red-600 text-sm bg-red-50 border border-red-200 rounded p-3">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="w-full bg-brand-600 hover:bg-brand-700 text-white font-medium py-3 rounded-lg disabled:opacity-50 transition-colors"
        >
          {submitting ? "Creating scenario…" : "Build Estimate →"}
        </button>
      </form>
    </div>
  );
}
