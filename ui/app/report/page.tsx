"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { RecommendationOut } from "@/lib/types";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { ModeBadge } from "@/components/ModeBadge";

function ReportInner() {
  const router = useRouter();
  const params = useSearchParams();
  const sid = params.get("sid");

  const [rec, setRec] = useState<RecommendationOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exported, setExported] = useState(false);

  useEffect(() => {
    if (!sid) return;
    api.getRecommendation(parseInt(sid))
      .then(setRec)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sid]);

  const exportMarkdown = () => {
    if (!rec) return;
    const s = rec.summary;
    const md = [
      `# Capacity Recommendation`,
      ``,
      `> **${rec.summary.mode === "estimate_only" ? "⚠ ESTIMATE ONLY" : "✓ VALIDATED"}**`,
      ``,
      `| | |`,
      `|---|---|`,
      `| Model | \`${s.model}\` |`,
      `| GPU | \`${s.gpu}\` |`,
      `| Dtype | \`${s.dtype}\` |`,
      `| Confidence | \`${s.confidence.toUpperCase()}\` |`,
      `| Replicas | **${s.replicas}** |`,
      `| Range | ${s.replicas_low} – ${s.replicas_high} |`,
      `| Binding constraint | \`${s.binding_constraint}\` |`,
      `| Benchmark runs done | ${s.benchmark_runs_done} |`,
    ].join("\n");

    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `recommendation-scenario-${sid}.md`;
    a.click();
    URL.revokeObjectURL(url);
    setExported(true);
  };

  if (!sid) return <p className="text-red-500">Missing scenario ID.</p>;
  if (loading) return <p className="text-slate-500">Loading recommendation…</p>;
  if (error) return <p className="text-red-600 bg-red-50 p-3 rounded">{error}</p>;
  if (!rec) return null;

  const s = rec.summary;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-slate-900">Recommendation Report</h1>
        <ModeBadge mode={s.mode} />
      </div>

      <ConfidenceBadge level={rec.confidence} />

      {/* Summary card */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        {/* KPI row */}
        <div className="px-6 py-5 flex items-center gap-8 border-b border-slate-100">
          <div>
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-1">Recommended replicas</p>
            <p className="text-5xl font-extrabold text-brand-600 leading-none tabular-nums">{s.replicas}</p>
          </div>
          <div className="flex flex-col gap-2">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold">Range</p>
              <p className="text-lg font-bold text-slate-800 tabular-nums">{s.replicas_low} – {s.replicas_high}</p>
            </div>
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold">Binding constraint</p>
              <p className="text-sm font-mono font-semibold text-slate-700">{s.binding_constraint}</p>
            </div>
          </div>
        </div>
        {/* Detail grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-0 divide-x divide-slate-100">
          {[
            { label: "Model", value: s.model },
            { label: "GPU", value: s.gpu },
            { label: "Dtype", value: s.dtype },
            { label: "Benchmark runs", value: s.benchmark_runs_done },
          ].map(({ label, value }) => (
            <div key={label} className="px-4 py-3">
              <p className="text-xs text-slate-400 uppercase tracking-wide font-semibold mb-0.5">{label}</p>
              <p className="text-sm font-mono font-semibold text-slate-700 truncate">{value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Validation status */}
      <div className={`rounded-xl border p-4 ${
        s.mode === "validated_by_benchmark"
          ? "bg-green-50 border-green-200 border-l-4 border-l-green-500"
          : "bg-amber-50 border-amber-200 border-l-4 border-l-amber-400"
      }`}>
        <h2 className="text-sm font-semibold mb-1">
          {s.mode === "validated_by_benchmark" ? "✓ Benchmark evidence" : "⚠ No benchmark data"}
        </h2>
        <p className="text-sm text-slate-700">
          {s.benchmark_runs_done === 0
            ? "This recommendation is based on a roofline estimate only. Run the benchmark plan to improve confidence."
            : `${s.benchmark_runs_done} benchmark run(s) completed. Confidence calibrated from live GPU data.`}
        </p>
        {s.mode !== "validated_by_benchmark" && (
          <button
            onClick={() => router.push(`/benchmark-plan?sid=${sid}`)}
            className="mt-3 text-sm text-brand-600 hover:text-brand-700 font-semibold"
          >
            → View benchmark plan
          </button>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={() => router.push("/")}
          className="px-4 py-2 border border-slate-200 rounded-lg text-sm text-slate-500 hover:bg-slate-50 transition-colors"
        >
          ← New scenario
        </button>
        <button
          onClick={exportMarkdown}
          className="flex-1 border border-brand-200 text-brand-600 bg-white hover:bg-brand-50 py-2 rounded-lg text-sm font-semibold transition-colors"
        >
          {exported ? "✓ Exported" : "⬇ Export as Markdown"}
        </button>
      </div>
    </div>
  );
}

export default function ReportPage() {
  return (
    <Suspense fallback={<p className="text-slate-400">Loading…</p>}>
      <ReportInner />
    </Suspense>
  );
}
