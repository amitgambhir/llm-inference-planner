"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { BenchmarkPlanOut, BenchmarkRunOut } from "@/lib/types";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { ModeBadge } from "@/components/ModeBadge";
import { CopyButton } from "@/components/CopyButton";

type RunState = {
  runId: number | null;
  status: BenchmarkRunOut["status"] | null;
  error: string | null;
};

const STATUS_COLORS: Record<string, string> = {
  queued:  "bg-slate-100 text-slate-600",
  running: "bg-blue-100 text-blue-700",
  done:    "bg-green-100 text-green-700",
  failed:  "bg-red-100 text-red-700",
};

function BenchmarkPlanInner() {
  const router = useRouter();
  const params = useSearchParams();
  const sid = params.get("sid");

  const [plan, setPlan] = useState<BenchmarkPlanOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [endpoint, setEndpoint] = useState("http://localhost:8000/v1/completions");
  const [authUser, setAuthUser] = useState("");
  const [authPass, setAuthPass] = useState("");

  const [runs, setRuns] = useState<Record<number, RunState>>({});

  const isLocalEndpoint = (() => {
    try {
      const host = new URL(endpoint).hostname;
      return host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0" || host === "::1";
    } catch { return false; }
  })();

  useEffect(() => {
    if (!sid) return;
    api.getBenchmarkPlan(parseInt(sid))
      .then(setPlan)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sid]);

  useEffect(() => {
    const inFlight = Object.entries(runs).filter(
      ([, r]) => r.runId !== null && (r.status === "queued" || r.status === "running")
    );
    if (inFlight.length === 0) return;

    const timer = setInterval(async () => {
      for (const [idxStr, r] of inFlight) {
        if (!r.runId) continue;
        try {
          const updated = await api.getBenchmarkRun(r.runId);
          setRuns((prev) => ({
            ...prev,
            [parseInt(idxStr)]: { ...prev[parseInt(idxStr)], status: updated.status },
          }));
        } catch {
          // ignore transient poll errors
        }
      }
    }, 3000);

    return () => clearInterval(timer);
  }, [runs]);

  const handleRun = async (stepIndex: number) => {
    if (!sid) return;
    setRuns((prev) => ({ ...prev, [stepIndex]: { runId: null, status: "queued", error: null } }));

    let effectiveEndpoint = endpoint;
    if (authUser || authPass) {
      try {
        const u = new URL(endpoint);
        u.username = authUser;
        u.password = authPass;
        effectiveEndpoint = u.toString();
      } catch {
        // leave endpoint as-is if URL parsing fails
      }
    }

    try {
      const run = await api.runBenchmark(parseInt(sid), stepIndex, effectiveEndpoint);
      setRuns((prev) => ({ ...prev, [stepIndex]: { runId: run.id, status: run.status, error: null } }));
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setRuns((prev) => ({ ...prev, [stepIndex]: { runId: null, status: "failed", error: msg } }));
    }
  };

  if (!sid) return <p className="text-red-500">Missing scenario ID.</p>;
  if (loading) return <p className="text-slate-500">Generating benchmark plan…</p>;
  if (error) return <p className="text-red-600 bg-red-50 p-3 rounded">{error}</p>;
  if (!plan) return null;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-slate-900">Benchmark Plan</h1>
        <ModeBadge mode="estimate_only" />
      </div>

      <div className="flex gap-2 flex-wrap">
        <ConfidenceBadge level={plan.confidence} />
        <span className="inline-flex items-center px-3 py-1 rounded-md text-xs bg-slate-100 text-slate-600 border border-slate-200 font-mono">
          {plan.binding_constraint}
        </span>
      </div>

      {/* Rationale */}
      <div className="bg-white border border-indigo-100 border-l-4 border-l-brand-500 rounded-r-xl px-4 py-3 text-sm text-indigo-800">
        <strong className="font-semibold">Why this ordering: </strong>{plan.rationale}
      </div>

      {/* Endpoint input */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2.5 bg-slate-50 border-b border-slate-100">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wide">Inference endpoint</span>
        </div>
        <div className="px-4 py-3 space-y-2">
          <p className="text-xs text-slate-500">
            The OpenAI-compatible <code className="bg-slate-100 px-1 rounded">/v1/completions</code> URL of the running GPU server.
          </p>
          <input
            type="url"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
            placeholder="http://your-gpu-host:8000/v1/completions"
            className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm font-mono bg-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
          />
          {isLocalEndpoint && (
            <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
              Remote run unavailable for localhost endpoints — copy the commands and run them directly on your GPU machine.
            </p>
          )}
          <details className="mt-1">
            <summary className="text-xs text-slate-400 cursor-pointer select-none hover:text-slate-600">
              Basic auth (optional)
            </summary>
            <div className="mt-2 flex gap-2">
              <input
                type="text"
                value={authUser}
                onChange={(e) => setAuthUser(e.target.value)}
                placeholder="Username"
                autoComplete="username"
                className="flex-1 border border-slate-200 rounded-lg px-3 py-1.5 text-sm bg-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500"
              />
              <input
                type="password"
                value={authPass}
                onChange={(e) => setAuthPass(e.target.value)}
                placeholder="Password"
                autoComplete="current-password"
                className="flex-1 border border-slate-200 rounded-lg px-3 py-1.5 text-sm bg-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500"
              />
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Credentials are embedded in the URL sent to the backend — never stored.
            </p>
          </details>
        </div>
      </div>

      {/* Steps */}
      <div>
        <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">
          Steps ({plan.steps.length})
        </h2>
        <div className="space-y-3">
          {plan.steps.map((step, i) => {
            const run = runs[i];
            const isActive = run?.status === "queued" || run?.status === "running";
            return (
              <div key={i} className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
                <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-slate-100">
                  <div className="flex items-center gap-2.5">
                    <span className="w-6 h-6 rounded-full bg-brand-50 border border-brand-200 text-brand-600 text-xs font-bold flex items-center justify-center flex-shrink-0">
                      {step.priority}
                    </span>
                    <span className="font-semibold text-sm text-slate-800">{step.label}</span>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {run?.status && (
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${STATUS_COLORS[run.status]}`}>
                        {run.status}
                      </span>
                    )}
                    <CopyButton text={step.command} />
                    {!isLocalEndpoint && (
                      <button
                        onClick={() => handleRun(i)}
                        disabled={isActive || !endpoint}
                        className="text-xs px-3 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium"
                      >
                        {isActive ? "Running…" : "Run"}
                      </button>
                    )}
                  </div>
                </div>
                <div className="px-4 py-3">
                  <p className="text-xs text-slate-500 mb-2">{step.purpose}</p>
                  <pre className="text-xs bg-slate-900 text-slate-300 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap break-all leading-relaxed">
                    {step.command}
                  </pre>
                  {run?.error && (
                    <p className="text-xs text-red-600 mt-2">{run.error}</p>
                  )}
                  <p className="text-xs text-slate-400 mt-2">
                    Collapses: <em>{step.collapses_confidence_on}</em>
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* All commands block */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <div className="flex justify-between items-center px-4 py-2.5 bg-slate-50 border-b border-slate-100">
          <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wide">All commands</h2>
          <CopyButton text={plan.steps.map((s) => s.command).join("\n")} />
        </div>
        <pre className="text-xs bg-slate-900 text-slate-300 p-4 overflow-x-auto whitespace-pre-wrap break-all leading-relaxed">
          {plan.steps.map((s) => s.command).join("\n")}
        </pre>
      </div>

      {/* Navigation */}
      <div className="flex gap-3">
        <button
          onClick={() => router.push(`/estimate?sid=${sid}`)}
          className="px-4 py-2 border border-slate-200 rounded-lg text-sm text-slate-500 hover:bg-slate-50 transition-colors"
        >
          ← Estimate
        </button>
        <button
          onClick={() => router.push(`/report?sid=${sid}`)}
          className="flex-1 bg-brand-600 hover:bg-brand-700 text-white py-2 rounded-lg text-sm font-semibold transition-colors shadow-sm shadow-brand-200"
        >
          View Recommendation →
        </button>
      </div>
    </div>
  );
}

export default function BenchmarkPlanPage() {
  return (
    <Suspense fallback={<p className="text-slate-400">Loading…</p>}>
      <BenchmarkPlanInner />
    </Suspense>
  );
}
