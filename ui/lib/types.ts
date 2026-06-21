// API response types — mirrors api/schemas.py

export interface ScenarioCreate {
  name: string;
  model_name: string;
  gpu_name: string;
  dtype: string;
  requests_per_day: number;
  peak_multiplier: number;
  isl: number;
  osl: number;
  ttft_slo_ms: number;
  tp: number;
  traffic_class: string;
  gpu_mem_util: number;
  runtime: string;
  prefix_cache_len?: number;
  prefix_cache_hit_rate?: number;
  max_num_seqs?: number;
}

export interface ScenarioOut extends ScenarioCreate {
  id: number;
  created_at: string;
}

export interface EstimateOut {
  id: number;
  scenario_id: number;
  replicas: number;
  replicas_low: number;
  replicas_high: number;
  binding_constraint: string;
  confidence: "high" | "medium" | "low";
  payload: {
    avg_rps: number;
    peak_rps: number;
    input_tps_peak: number;
    output_tps_peak: number;
    ttft_ms: number;
    max_concurrent_seqs: number;
    mfu_used: number;
    bw_eff_used: number;
    decode_bw_eff_used: number;
    prefill_tps_gpu: number;
    decode_tps_gpu: number;
    total_gpus: number;
    tpot_ms: number;
    eff_batch_used: number;
    kv_ratio: number;
    warnings: string[];
    assumptions: string[];
  };
  created_at: string;
}

export interface BenchmarkStepOut {
  priority: number;
  label: string;
  command: string;
  purpose: string;
  collapses_confidence_on: string;
}

export interface BenchmarkPlanOut {
  steps: BenchmarkStepOut[];
  confidence: string;
  binding_constraint: string;
  rationale: string;
}

export interface BenchmarkRunOut {
  id: number;
  scenario_id: number;
  tag: string;
  command: string;
  status: "queued" | "running" | "done" | "failed";
  result_path: string | null;
  exit_code: number | null;
  created_at: string;
}

export interface RecommendationOut {
  id: number;
  scenario_id: number;
  summary: {
    mode: string;
    replicas: number;
    replicas_low: number;
    replicas_high: number;
    binding_constraint: string;
    confidence: string;
    benchmark_runs_done: number;
    model: string;
    gpu: string;
    dtype: string;
  };
  confidence: string;
  created_at: string;
}

// GPU and model catalog entries — keep in sync with catalog/gpus.yaml and catalog/models.yaml
export const CATALOG_GPUS: { key: string; label: string }[] = [
  // NVIDIA Hopper / Ada / Ampere
  { key: "h100_sxm", label: "NVIDIA H100 SXM (80 GB)" },
  { key: "h200_sxm", label: "NVIDIA H200 SXM (141 GB)" },
  { key: "a100_80gb_sxm", label: "NVIDIA A100 80GB SXM4 (80 GB)" },
  { key: "l40s", label: "NVIDIA L40S (48 GB)" },
  { key: "l4", label: "NVIDIA L4 (24 GB)" },
  { key: "h20", label: "NVIDIA H20 (96 GB)" },
  // NVIDIA Blackwell
  { key: "b100", label: "NVIDIA B100 (192 GB)" },
  { key: "b200", label: "NVIDIA B200 (192 GB)" },
  { key: "b300", label: "NVIDIA B300 (288 GB)" },
  { key: "b30a", label: "NVIDIA B30A (120 GB)" },
  // AMD Instinct
  { key: "mi250x", label: "AMD MI250X (128 GB)" },
  { key: "mi300x", label: "AMD MI300X (192 GB)" },
  { key: "mi325x", label: "AMD MI325X (256 GB)" },
  { key: "mi350x", label: "AMD MI350X (288 GB)" },
  { key: "mi300a", label: "AMD MI300A (128 GB)" },
];

export const CATALOG_MODELS: { key: string; label: string }[] = [
  // Llama family
  { key: "llama-3.1-8b", label: "Llama 3.1 8B" },
  { key: "llama-3.1-70b", label: "Llama 3.1 70B" },
  { key: "llama-3.3-70b", label: "Llama 3.3 70B" },
  { key: "llama-2-70b", label: "Llama 2 70B" },
  { key: "llama-4-maverick", label: "Llama 4 Maverick (MoE)" },
  // Gemma family
  { key: "gemma-2-9b", label: "Gemma 2 9B" },
  { key: "gemma-3-1b", label: "Gemma 3 1B" },
  { key: "gemma-3-4b", label: "Gemma 3 4B" },
  { key: "gemma-3-12b", label: "Gemma 3 12B" },
  { key: "gemma-3-27b", label: "Gemma 3 27B" },
  { key: "gemma-4-e2b", label: "Gemma 4 E2B" },
  { key: "gemma-4-e4b", label: "Gemma 4 E4B" },
  { key: "gemma-4-12b", label: "Gemma 4 12B" },
  { key: "gemma-4-26b-a4b", label: "Gemma 4 26B-A4B (MoE)" },
  { key: "gemma-4-31b", label: "Gemma 4 31B" },
  // Mistral / Mixtral
  { key: "mistral-7b", label: "Mistral 7B v0.3" },
  { key: "mistral-small-24b", label: "Mistral Small 24B" },
  { key: "mixtral-8x7b", label: "Mixtral 8×7B (MoE)" },
  // Qwen
  { key: "qwen3-8b", label: "Qwen3 8B" },
  { key: "qwen3-30b-a3b", label: "Qwen3 30B-A3B (MoE)" },
  // Other
  { key: "gpt-oss-20b", label: "GPT-OSS 20B (MoE)" },
  { key: "nemotron-4-340b", label: "Nemotron-4 340B" },
  { key: "glm-5.1", label: "GLM-5.1 (MoE)" },
  { key: "glm-5.2", label: "GLM-5.2 (MoE)" },
];

export const DTYPES = ["bf16", "fp16", "fp8", "mxfp4", "int8"];

export const TRAFFIC_CLASSES = [
  { key: "realtime", label: "Realtime (1.40× headroom)" },
  { key: "mixed", label: "Mixed (1.25× headroom)" },
  { key: "batch", label: "Batch (1.10× headroom)" },
];
