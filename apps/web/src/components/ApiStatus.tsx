import type { HealthResponse } from "@/lib/api";

export type ApiStatusState =
  { phase: "checking" } | { phase: "unreachable" } | { phase: "ok"; health: HealthResponse };

export function ApiStatus({ state }: { state: ApiStatusState }) {
  let dot = "bg-slate-400";
  let text = "Checking service…";

  if (state.phase === "unreachable") {
    dot = "bg-rose-500";
    text = "API unreachable";
  } else if (state.phase === "ok") {
    const status = state.health.model.status;
    dot = status === "ready" ? "bg-emerald-500" : "bg-amber-500";
    text =
      status === "ready"
        ? "API online · model ready"
        : status === "not_trained"
          ? "API online · model not trained yet"
          : "API online · model failed to load";
  }

  return (
    <p
      role="status"
      className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300"
    >
      <span aria-hidden className={`size-2 rounded-full ${dot}`} />
      {text}
    </p>
  );
}
