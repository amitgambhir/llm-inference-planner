"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { api } from "@/lib/api";
import type { EstimateOut } from "@/lib/types";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { ModeBadge } from "@/components/ModeBadge";
import { ReplicaRangeChart } from "@/components/ReplicaRangeChart";

function EstimateResultsInner() {
  const router = useRouter();
  const params = useSearchParams();
  const sid = params.get("sid");

  const [estimate, setEstimate] = useState<EstimateOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sid) return;
    api.runEstimate(parseInt(sid))
      .then(setEstimate)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sid]);

  if (!sid) return <p className="text-red-500">Missing scenario ID.</p>;
  if (loading) return <p className="text-slate-500">Running estimate…</p>;
  if (error) return <p className="text-red-600 bg-red-50 p-3 rounded">{error}</p>;
  if (!estimate) return null;

  const p = estimate.payload;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Estimate Results</h1>
        </div>
        <ModeBadge mode="estimate_only" />
      </div>

      {/* Confidence + binding constraint */}
      <div className="flex gap-2 flex-wrap">
        <ConfidenceBadge level={estimate.confidence} />
        <span className="inline-flex items-center px-3 py-1 rounded-md text-sm bg-slate-100 text-slate-600 border border-slate-200 font-mono text-xs">
          {estimate.binding_constraint}
        </span>
      </div>

      {/* Primary metric cards */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Replicas", value: estimate.replicas, accent: "indigo" },
          { label: "Range", value: `${estimate.replicas_low}–${estimate.replicas_high}`, accent: "amber" },
          { label: "Predicted TTFT", value: `${p.ttft_ms?.toFixed(0) ?? "–"} ms`, accent: "green" },
        ].map(({ label, value, accent }) => (
          <div
            key={label}
            className={`bg-white rounded-xl border border-slate-200 shadow-sm px-4 pt-4 pb-3 text-center relative overflow-hidden
              before:absolute before:top-0 before:left-0 before:right-0 before:h-0.5
              ${accent === "indigo" ? "before:bg-brand-500" : accent === "amber" ? "before:bg-amber-400" : "before:bg-green-500"}`}
          >
            <p className="text-3xl font-extrabold text-slate-900 tabular-nums leading-none mb-1">{value}</p>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">{label}</p>
          </div>
        ))}
      </div>

      {/* Secondary metric pills */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Total GPUs", value: p.total_gpus ?? "–", color: "bg-brand-500" },
          { label: "TPOT", value: `${p.tpot_ms?.toFixed(1) ?? "–"} ms`, color: "bg-amber-400" },
          { label: "KV ratio", value: p.kv_ratio?.toFixed(1) ?? "–", color: "bg-green-500" },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-white border border-slate-200 rounded-lg px-3 py-2.5 flex items-center gap-2.5 shadow-sm">
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${color}`} />
            <div>
              <p className="text-xs text-slate-500">{label}</p>
              <p className="text-sm font-bold text-slate-800 tabular-nums">{value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Replica range chart */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
        <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">Replica range</h2>
        <ReplicaRangeChart
          replicas={estimate.replicas}
          replicasLow={estimate.replicas_low}
          replicasHigh={estimate.replicas_high}
          confidence={estimate.confidence}
        />
      </div>

      {/* Workload stats */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-100">
          <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wide">Workload stats</h2>
        </div>
        <dl>
          {[
            ["Avg RPS", p.avg_rps?.toFixed(2)],
            ["Peak RPS", p.peak_rps?.toFixed(2)],
            ["Prefill ceiling (TP group)", `${p.prefill_tps_gpu?.toFixed(0)} tok/s`],
            ["Decode ceiling (TP group)", `${p.decode_tps_gpu?.toFixed(0)} tok/s`],
            ["Max concurrent seqs/replica", p.max_concurrent_seqs],
            ["Effective batch (70% fill)", p.eff_batch_used],
            ["MFU used", `${(p.mfu_used * 100).toFixed(1)}%`],
            ["Decode bw_eff (KV-adjusted)", `${((p.decode_bw_eff_used ?? 0) * 100).toFixed(1)}%`],
          ].map(([k, v], i) => (
            <div
              key={k as string}
              className={`flex justify-between px-4 py-2 text-sm ${i % 2 === 1 ? "bg-slate-50" : ""}`}
            >
              <dt className="text-slate-500">{k}</dt>
              <dd className="font-mono font-semibold text-slate-800">{v}</dd>
            </div>
          ))}
        </dl>
      </div>

      {/* Warnings */}
      {p.warnings?.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-amber-800 mb-2">Warnings</h2>
          <ul className="space-y-1">
            {p.warnings.map((w, i) => (
              <li key={i} className="text-sm text-amber-700">⚠ {w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Navigation */}
      <div className="flex gap-3">
        <button
          onClick={() => router.push("/")}
          className="px-4 py-2 border border-slate-200 rounded-lg text-sm text-slate-500 hover:bg-slate-50 transition-colors"
        >
          ← New scenario
        </button>
        <button
          onClick={() => router.push(`/benchmark-plan?sid=${sid}`)}
          className="flex-1 bg-brand-600 hover:bg-brand-700 text-white py-2 rounded-lg text-sm font-semibold transition-colors shadow-sm shadow-brand-200"
        >
          Generate Benchmark Plan →
        </button>
      </div>
    </div>
  );
}

export default function EstimateResults() {
  return (
    <Suspense fallback={<p className="text-slate-400">Loading…</p>}>
      <EstimateResultsInner />
    </Suspense>
  );
}
