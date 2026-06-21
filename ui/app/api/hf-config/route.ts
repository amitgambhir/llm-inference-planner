import { NextRequest, NextResponse } from "next/server";

const HF_BASE = "https://huggingface.co";

export interface HFModelSpec {
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
  // Set when safetensors index is available — overrides weight_bytes_per_param × total_params
  resident_weights_gb?: number;
  total_params: number;
  active_params: number;
  is_moe: boolean;
  num_experts?: number;
  experts_per_token?: number;
}

export interface HFConfigResponse {
  spec: HFModelSpec;
  warnings: string[];
}

export interface HFErrorResponse {
  error: "gated" | "not_found" | "network_error" | "invalid_model_id";
  message: string;
}

async function hfFetch(url: string, token?: string): Promise<Response> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return fetch(url, { headers, next: { revalidate: 300 } });
}

export async function GET(
  req: NextRequest
): Promise<NextResponse<HFConfigResponse | HFErrorResponse>> {
  const { searchParams } = new URL(req.url);
  const modelId = searchParams.get("model")?.trim();

  if (!modelId || !modelId.includes("/")) {
    return NextResponse.json(
      {
        error: "invalid_model_id",
        message: 'Model ID must be in the format "owner/model-name"',
      },
      { status: 400 }
    );
  }

  const token = req.headers.get("x-hf-token") ?? undefined;
  const warnings: string[] = [];

  // ── Step 1: fetch config.json ────────────────────────────────────────────
  const configUrl = `${HF_BASE}/${modelId}/raw/main/config.json`;
  let configRes: Response;

  try {
    configRes = await hfFetch(configUrl, token);
  } catch {
    return NextResponse.json(
      {
        error: "network_error",
        message: "Could not reach HuggingFace. Check your network connection.",
      },
      { status: 503 }
    );
  }

  if (configRes.status === 403 || configRes.status === 401) {
    return NextResponse.json(
      {
        error: "gated",
        message:
          "This model is gated. Provide a HuggingFace access token to continue.",
      },
      { status: 403 }
    );
  }
  if (configRes.status === 404) {
    return NextResponse.json(
      {
        error: "not_found",
        message: `Model "${modelId}" was not found on HuggingFace.`,
      },
      { status: 404 }
    );
  }
  if (!configRes.ok) {
    return NextResponse.json(
      {
        error: "network_error",
        message: `HuggingFace returned HTTP ${configRes.status} for config.json`,
      },
      { status: 502 }
    );
  }

  // Sanitize JavaScript literals that aren't valid JSON
  let config: Record<string, unknown>;
  try {
    const text = await configRes.text();
    const sanitized = text
      .replace(/:\s*Infinity\b/g, ": 1e308")
      .replace(/:\s*-Infinity\b/g, ": -1e308")
      .replace(/:\s*NaN\b/g, ": null");
    config = JSON.parse(sanitized) as Record<string, unknown>;
  } catch {
    return NextResponse.json(
      {
        error: "network_error",
        message: "config.json from HuggingFace could not be parsed as JSON.",
      },
      { status: 502 }
    );
  }

  // ── Step 2: weight bytes from safetensors index ──────────────────────────
  let weightBytes: number | null = null;

  try {
    const indexRes = await hfFetch(
      `${HF_BASE}/${modelId}/resolve/main/model.safetensors.index.json`,
      token
    );
    if (indexRes.ok) {
      const index = (await indexRes.json()) as Record<string, unknown>;
      const meta = index.metadata as Record<string, unknown> | undefined;
      if (typeof meta?.total_size === "number") {
        weightBytes = meta.total_size as number;
      }
    }
  } catch {
    // fallthrough to HF API
  }

  // Fallback: derive from HF API dtype-bucketed parameter counts
  if (weightBytes === null) {
    try {
      const apiRes = await hfFetch(
        `${HF_BASE}/api/models/${modelId}`,
        token
      );
      if (apiRes.ok) {
        const meta = (await apiRes.json()) as Record<string, unknown>;
        const safetensors = meta.safetensors as Record<string, unknown> | undefined;
        const params = safetensors?.parameters as Record<string, number> | undefined;
        if (params) {
          const DTYPE_BYTES: Record<string, number> = {
            F32: 4, F16: 2, BF16: 2, F8_E4M3: 1, F8_E5M2: 1, I8: 1, I4: 0.5,
          };
          weightBytes = Object.entries(params).reduce(
            (sum, [dtype, count]) => sum + count * (DTYPE_BYTES[dtype] ?? 2),
            0
          );
        }
      }
    } catch {
      // ignore
    }
  }

  if (weightBytes === null) {
    warnings.push(
      "Could not retrieve weight size from HuggingFace — memory estimate will be approximate."
    );
  }

  // ── Step 3: extract planner geometry ────────────────────────────────────
  const numLayers = (config.num_hidden_layers as number) ?? 32;
  const hiddenSize = (config.hidden_size as number) ?? 4096;
  const numQHeads = (config.num_attention_heads as number) ?? 32;
  const numKvHeads =
    (config.num_key_value_heads as number) ?? numQHeads;
  // head_dim: use explicit field, fall back to hidden_size / num_q_heads
  const headDim =
    (config.head_dim as number) ??
    Math.round(hiddenSize / numQHeads);
  const contextLen =
    (config.max_position_embeddings as number) ?? 131072;

  // Native dtype
  const torchDtype = (config.torch_dtype as string) ?? "bfloat16";
  const nativeDtype = torchDtype.replace("bfloat16", "bf16").replace("float16", "fp16").replace("float32", "fp32");
  const DTYPE_BYTES_MAP: Record<string, number> = {
    bf16: 2, fp16: 2, fp32: 4, fp8: 1, int8: 1, mxfp4: 0.5, int4: 0.5,
  };
  const weightBytesPerParam = DTYPE_BYTES_MAP[nativeDtype] ?? 2.0;

  // MoE detection
  const isMoe = !!(config.num_local_experts as number);
  const numExperts = (config.num_local_experts as number) ?? undefined;
  const expertsPerToken = (config.num_experts_per_tok as number) ?? undefined;

  // total_params: derive from weight bytes (accurate for bf16 base models)
  // For quantized or multimodal, this overestimates — resident_weights_gb is the ground truth.
  const totalParams = weightBytes
    ? Math.round(weightBytes / weightBytesPerParam)
    : Math.round(
        // rough formula as fallback: embedding + layers + lm_head
        2 * ((config.vocab_size as number) ?? 32000) * hiddenSize +
          numLayers *
            (4 * numQHeads * headDim * hiddenSize +   // attention (Q,K,V,O)
             2 * hiddenSize * ((config.intermediate_size as number) ?? hiddenSize * 4))
      );

  // active_params for MoE: non-expert portion + active expert portion
  let activeParams = totalParams;
  if (isMoe && numExperts && expertsPerToken && weightBytes) {
    // Rough split: attention layers = ~30% of total, expert FFNs = ~70%
    // Active = attention + (experts_per_token / num_experts) × expert_weights
    const expertFraction = expertsPerToken / numExperts;
    activeParams = Math.round(
      totalParams * 0.30 + totalParams * 0.70 * expertFraction
    );
  }

  const isMultimodal = !!(
    config.text_config ||
    config.vision_config ||
    (Array.isArray(config.architectures) &&
      (config.architectures as string[]).some(
        (a: string) =>
          a.includes("Conditional") ||
          a.includes("VL") ||
          a.includes("Vision") ||
          a.includes("Multimodal")
      ))
  );
  if (isMultimodal) {
    warnings.push(
      "This appears to be a multimodal model. Weight memory includes vision encoder weights — " +
        "the planner will over-estimate VRAM needed for the language model portion."
    );
  }

  // Display name: last segment of model ID
  const displayName = modelId.split("/").pop() ?? modelId;

  const spec: HFModelSpec = {
    name: modelId.replace("/", "--"),
    display_name: displayName,
    num_layers: numLayers,
    d_model: hiddenSize,
    num_q_heads: numQHeads,
    num_kv_heads: numKvHeads,
    head_dim: headDim,
    context_len: contextLen,
    native_dtype: nativeDtype,
    weight_bytes_per_param: weightBytesPerParam,
    kv_dtype_bytes: 2,
    geometry_source: "estimated",
    total_params: totalParams,
    active_params: activeParams,
    is_moe: isMoe,
    ...(numExperts !== undefined && { num_experts: numExperts }),
    ...(expertsPerToken !== undefined && { experts_per_token: expertsPerToken }),
    ...(weightBytes !== null && {
      resident_weights_gb: Math.round((weightBytes / 1e9) * 100) / 100,
    }),
  };

  return NextResponse.json({ spec, warnings });
}
