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

// ── Section card shell ────────────────────────────────────────────────────────

function SectionCard({
  icon,
  title,
  children,
}: {
  icon: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-slate-50 border-b border-slate-100">
        <span className="w-5 h-5 rounded bg-brand-50 flex items-center justify-center text-xs">{icon}</span>
        <span className="text-xs font-bold text-slate-500 uppercase tracking-wide">{title}</span>
      </div>
      <div className="px-4 py-4">{children}</div>
    </div>
  );
}

// ── Field label ───────────────────────────────────────────────────────────────

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
      {children}
    </label>
  );
}

// ── Input / Select shared styles ─────────────────────────────────────────────

const inputCls = "w-full border border-slate-200 rounded-lg px-3 py-2 text-sm bg-slate-50 text-slate-800 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent";

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
      <div className="flex gap-2">
        <input
          className={`flex-1 ${inputCls} font-mono`}
          placeholder='owner/model-name  e.g. meta-llama/Meta-Llama-3-8B'
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

      <div>
        <button
          type="button"
          onClick={() => setShowToken(!showToken)}
          className="text-xs text-slate-400 hover:text-slate-600"
        >
          {showToken ? "▾ Hide token" : "▸ Gated model? Add HF token"}
        </button>
        {showToken && (
          <input
            type="password"
            className={`mt-1 ${inputCls} font-mono`}
            placeholder="hf_..."
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
          />
        )}
      </div>

      {fetchError && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          {fetchError}
        </p>
      )}

      {warnings.length > 0 && (
        <ul className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 space-y-1 list-disc list-inside">
          {warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}

      {spec && (
        <div className="p-3 bg-green-50 border border-green-200 rounded-lg text-xs space-y-1">
          <p className="font-semibold text-green-800 text-sm">{spec.display_name}</p>
          <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-slate-700 mt-1">
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

// ── Scenario Builder ──────────────────────────────────────────────────────────

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
      <h1 className="text-2xl font-bold text-slate-900 mb-1">Scenario Builder</h1>
      <p className="text-slate-500 text-sm mb-6">
        Configure your workload — we&apos;ll size replicas, estimate cost, and generate a benchmark plan.
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">

        {/* ── Scenario ── */}
        <SectionCard icon="🏷" title="Scenario">
          <FieldLabel>Name</FieldLabel>
          <input
            className={inputCls}
            value={form.name}
            onChange={(e) => set("name", e.target.value)}
            required
          />
        </SectionCard>

        {/* ── Hardware ── */}
        <SectionCard icon="🖥" title="Hardware">
          <div className="grid grid-cols-3 gap-4">
            <div className="col-span-2">
              <FieldLabel>
                GPU <span className="text-slate-400 font-normal normal-case tracking-normal">(curated catalog)</span>
              </FieldLabel>
              <select
                className={inputCls}
                value={form.gpu_name}
                onChange={(e) => set("gpu_name", e.target.value)}
              >
                {CATALOG_GPUS.map((g) => (
                  <option key={g.key} value={g.key}>{g.label}</option>
                ))}
              </select>
            </div>
            <div>
              <FieldLabel>Serving dtype</FieldLabel>
              <select
                className={inputCls}
                value={form.dtype}
                onChange={(e) => set("dtype", e.target.value)}
              >
                {DTYPES.map((d) => <option key={d}>{d}</option>)}
              </select>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div>
              <FieldLabel>Tensor parallel</FieldLabel>
              <input
                type="number" min={1} max={64}
                className={inputCls}
                value={form.tp}
                onChange={(e) => set("tp", parseInt(e.target.value))}
              />
            </div>
            <div>
              <FieldLabel>
                GPU mem util
                <span className="ml-1 text-slate-400 font-normal normal-case tracking-normal text-xs" title="Fraction of VRAM reserved for weights + KV cache (default 0.9). Lower if you see OOM errors.">?</span>
              </FieldLabel>
              <input
                type="number" min={0.5} max={1.0} step={0.05}
                className={inputCls}
                value={form.gpu_mem_util}
                onChange={(e) => set("gpu_mem_util", parseFloat(e.target.value))}
              />
            </div>
          </div>
        </SectionCard>

        {/* ── Model ── */}
        <SectionCard icon="🤖" title="Model">
          <div className="flex gap-2 mb-3">
            <button
              type="button"
              onClick={() => setModelMode("catalog")}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                modelMode === "catalog"
                  ? "bg-brand-600 text-white"
                  : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
              }`}
            >
              Catalog
            </button>
            <button
              type="button"
              onClick={() => setModelMode("custom")}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                modelMode === "custom"
                  ? "bg-brand-600 text-white"
                  : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
              }`}
            >
              Custom / HF
            </button>
          </div>
          {modelMode === "catalog" ? (
            <select
              className={inputCls}
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
        </SectionCard>

        {/* ── Traffic ── */}
        <SectionCard icon="📊" title="Traffic">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <FieldLabel>Requests / day</FieldLabel>
              <input
                type="number" min={1}
                className={inputCls}
                value={form.requests_per_day}
                onChange={(e) => set("requests_per_day", parseInt(e.target.value))}
              />
            </div>
            <div>
              <FieldLabel>Peak multiplier</FieldLabel>
              <input
                type="number" min={1} step={0.5}
                className={inputCls}
                value={form.peak_multiplier}
                onChange={(e) => set("peak_multiplier", parseFloat(e.target.value))}
              />
            </div>
            <div>
              <FieldLabel>ISL (input tokens)</FieldLabel>
              <input
                type="number" min={1}
                className={inputCls}
                value={form.isl}
                onChange={(e) => set("isl", parseInt(e.target.value))}
              />
            </div>
            <div>
              <FieldLabel>OSL (output tokens)</FieldLabel>
              <input
                type="number" min={1}
                className={inputCls}
                value={form.osl}
                onChange={(e) => set("osl", parseInt(e.target.value))}
              />
            </div>
            <div>
              <FieldLabel>TTFT SLO (ms)</FieldLabel>
              <input
                type="number" min={100}
                className={inputCls}
                value={form.ttft_slo_ms}
                onChange={(e) => set("ttft_slo_ms", parseFloat(e.target.value))}
              />
            </div>
            <div>
              <FieldLabel>Traffic class</FieldLabel>
              <select
                className={inputCls}
                value={form.traffic_class}
                onChange={(e) => set("traffic_class", e.target.value)}
              >
                {TRAFFIC_CLASSES.map((tc) => (
                  <option key={tc.key} value={tc.key}>{tc.label}</option>
                ))}
              </select>
            </div>
          </div>
        </SectionCard>

        {/* ── Advanced ── */}
        <SectionCard icon="⚙️" title="Advanced (optional)">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <FieldLabel>
                Prefix cache len
                <span className="ml-1 text-slate-400 font-normal normal-case tracking-normal text-xs" title="Shared prefix length in tokens (e.g. system prompt). Reduces prefill compute; KV budget unchanged.">?</span>
              </FieldLabel>
              <input
                type="number" min={0}
                placeholder="0"
                className={inputCls}
                value={form.prefix_cache_len ?? ""}
                onChange={(e) => set("prefix_cache_len", e.target.value === "" ? undefined : parseInt(e.target.value))}
              />
            </div>
            <div>
              <FieldLabel>
                Cache hit rate
                <span className="ml-1 text-slate-400 font-normal normal-case tracking-normal text-xs" title="Fraction of requests that hit the prefix cache (0.0–1.0).">?</span>
              </FieldLabel>
              <input
                type="number" min={0} max={1} step={0.05}
                placeholder="0.0"
                className={inputCls}
                value={form.prefix_cache_hit_rate ?? ""}
                onChange={(e) => set("prefix_cache_hit_rate", e.target.value === "" ? undefined : parseFloat(e.target.value))}
              />
            </div>
            <div>
              <FieldLabel>
                Max batch size
                <span className="ml-1 text-slate-400 font-normal normal-case tracking-normal text-xs" title="vLLM --max-num-seqs: scheduler concurrency cap, independent of KV budget.">?</span>
              </FieldLabel>
              <input
                type="number" min={1}
                placeholder="unlimited"
                className={inputCls}
                value={form.max_num_seqs ?? ""}
                onChange={(e) => set("max_num_seqs", e.target.value === "" ? undefined : parseInt(e.target.value))}
              />
            </div>
          </div>
        </SectionCard>

        {error && (
          <p className="text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg p-3">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="w-full bg-brand-600 hover:bg-brand-700 text-white font-semibold py-3 rounded-xl disabled:opacity-50 transition-colors shadow-sm shadow-brand-200"
        >
          {submitting ? "Creating scenario…" : "Estimate →"}
        </button>
      </form>
    </div>
  );
}
