import { getApiBaseUrl } from "@/lib/config";

// Types mirror services/api/app/schemas.py.
export type Label = "fake" | "real";

export interface ModelInfo {
  name: string;
  version: string | null;
  trained_at: string | null;
}

export interface AnalyzeResponse {
  mode: "model" | "demo";
  label: Label;
  confidence: number;
  probabilities: { fake: number; real: number };
  model: ModelInfo | null;
  analyzed_chars: number;
  disclaimer: string;
}

export interface HealthResponse {
  status: "ok";
  service: string;
  version: string;
  environment: string;
  model: {
    status: "ready" | "not_trained" | "error";
    loaded: boolean;
    detail: string;
    info: ModelInfo | null;
  };
  demo_mode_enabled: boolean;
}

export type ApiErrorKind =
  "model_not_trained" | "model_load_failed" | "invalid_request" | "network" | "timeout" | "server";

export class ApiError extends Error {
  constructor(
    readonly kind: ApiErrorKind,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const REQUEST_TIMEOUT_MS = 30_000;

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort("timeout"), REQUEST_TIMEOUT_MS);
  const callerSignal = init.signal;
  callerSignal?.addEventListener("abort", () => controller.abort(callerSignal.reason), {
    once: true,
  });

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, { ...init, signal: controller.signal });
  } catch (error) {
    if (callerSignal?.aborted) throw error;
    if (controller.signal.reason === "timeout") {
      throw new ApiError("timeout", "The analysis service took too long to respond. Try again.");
    }
    throw new ApiError(
      "network",
      "Could not reach the analysis service. Check that the API is running and NEXT_PUBLIC_API_BASE_URL is correct.",
    );
  } finally {
    clearTimeout(timeout);
  }

  if (response.ok) {
    return (await response.json()) as T;
  }
  throw await toApiError(response);
}

async function toApiError(response: Response): Promise<ApiError> {
  let code: string | undefined;
  let message: string | undefined;
  try {
    const body = (await response.json()) as { error?: { code?: string; message?: string } };
    code = body.error?.code;
    message = body.error?.message;
  } catch {
    // Non-JSON error body (e.g. a proxy error page): fall through to the generic message.
  }
  if (code === "model_not_trained" || code === "model_load_failed" || code === "invalid_request") {
    return new ApiError(code, message ?? "The request failed.");
  }
  return new ApiError(
    "server",
    message ?? `The service returned an error (HTTP ${response.status}).`,
  );
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/api/v1/health", { signal });
}

/** `demo: true` calls the clearly separate demo endpoint, which never runs the ML model. */
export function analyzeArticle(
  text: string,
  options: { demo?: boolean; signal?: AbortSignal } = {},
): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>(options.demo ? "/api/v1/demo/analyze" : "/api/v1/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal: options.signal,
  });
}
