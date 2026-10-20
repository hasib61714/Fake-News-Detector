import type { AnalyzeResponse, ApiError } from "@/lib/api";

export type AnalysisState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "success"; result: AnalyzeResponse }
  | { phase: "error"; error: ApiError };

const card =
  "rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6 dark:border-slate-800 dark:bg-slate-900";

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function ResultPanel({ state }: { state: AnalysisState }) {
  return (
    <section aria-labelledby="results-heading" aria-live="polite" className={card}>
      <h2 id="results-heading" className="text-lg font-semibold">
        Result
      </h2>
      <div className="mt-4">
        {state.phase === "idle" && <Idle />}
        {state.phase === "loading" && <Loading />}
        {state.phase === "success" && <Success result={state.result} />}
        {state.phase === "error" && <Failure error={state.error} />}
      </div>
    </section>
  );
}

function Idle() {
  return (
    <p className="text-sm text-slate-600 dark:text-slate-400">
      Your analysis will appear here. Paste an article and press <strong>Analyze</strong>.
    </p>
  );
}

function Loading() {
  return (
    <div role="status" className="space-y-3" aria-label="Analyzing article">
      <div className="h-6 w-1/2 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
      <div className="h-3 w-full animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
      <div className="h-3 w-5/6 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
      <p className="text-sm text-slate-600 dark:text-slate-400">Analyzing article…</p>
    </div>
  );
}

function Failure({ error }: { error: ApiError }) {
  const notTrained = error.kind === "model_not_trained";
  const tone = notTrained
    ? "border-amber-300 bg-amber-50 text-amber-950 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-100"
    : "border-rose-300 bg-rose-50 text-rose-950 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-100";

  return (
    <div role="alert" className={`rounded-xl border p-4 text-sm ${tone}`}>
      <p className="font-semibold">{notTrained ? "Model not trained yet" : "Analysis failed"}</p>
      <p className="mt-1">{error.message}</p>
      {notTrained && (
        <p className="mt-2">
          Train the model with <code className="rounded bg-black/10 px-1 py-0.5">make train</code>{" "}
          (see the README), then press Analyze again. No restart is needed.
        </p>
      )}
    </div>
  );
}

function Success({ result }: { result: AnalyzeResponse }) {
  const isDemo = result.mode === "demo";
  const isFake = result.label === "fake";

  const badge = isDemo
    ? "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200"
    : isFake
      ? "bg-rose-100 text-rose-900 dark:bg-rose-950 dark:text-rose-200"
      : "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200";
  const bar = isDemo ? "bg-slate-400" : isFake ? "bg-rose-500" : "bg-emerald-500";

  return (
    <div className="space-y-5">
      {isDemo && (
        <div
          role="note"
          className="rounded-xl border border-violet-300 bg-violet-50 p-3 text-sm font-medium text-violet-950 dark:border-violet-800 dark:bg-violet-950/40 dark:text-violet-100"
        >
          Demo preview. {result.disclaimer}
        </div>
      )}

      <div>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          {isDemo ? "Sample output" : "The classifier thinks this article looks"}
        </p>
        <p className={`mt-1 inline-block rounded-lg px-3 py-1.5 text-2xl font-semibold ${badge}`}>
          {isDemo ? "No real verdict" : isFake ? "Likely fake" : "Likely real"}
        </p>
      </div>

      <div>
        <div className="flex items-baseline justify-between text-sm">
          <span className="font-medium">{isDemo ? "Sample confidence" : "Confidence"}</span>
          <span className="tabular-nums">{percent(result.confidence)}</span>
        </div>
        <div
          role="progressbar"
          aria-label="Confidence"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(result.confidence * 100)}
          className="mt-2 h-2.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800"
        >
          <div className={`h-full ${bar}`} style={{ width: percent(result.confidence) }} />
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-lg bg-slate-50 p-3 dark:bg-slate-800/60">
          <dt className="text-slate-600 dark:text-slate-400">Fake probability</dt>
          <dd className="mt-0.5 font-semibold tabular-nums">
            {percent(result.probabilities.fake)}
          </dd>
        </div>
        <div className="rounded-lg bg-slate-50 p-3 dark:bg-slate-800/60">
          <dt className="text-slate-600 dark:text-slate-400">Real probability</dt>
          <dd className="mt-0.5 font-semibold tabular-nums">
            {percent(result.probabilities.real)}
          </dd>
        </div>
      </dl>

      <p className="text-xs text-slate-600 dark:text-slate-400">
        {result.model
          ? `Model: ${result.model.name}${result.model.version ? ` (v${result.model.version})` : ""} · `
          : "No model used · "}
        {result.analyzed_chars.toLocaleString("en-US")} characters analyzed
      </p>
      {!isDemo && <p className="text-xs text-slate-600 dark:text-slate-400">{result.disclaimer}</p>}
    </div>
  );
}
