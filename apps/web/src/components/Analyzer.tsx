"use client";

import { useEffect, useRef, useState } from "react";

import { ApiStatus, type ApiStatusState } from "@/components/ApiStatus";
import { type AnalysisState, ResultPanel } from "@/components/ResultPanel";
import { analyzeArticle, ApiError, getHealth } from "@/lib/api";
import { MAX_ARTICLE_CHARS, MIN_ARTICLE_CHARS, validateArticle } from "@/lib/validation";

export function Analyzer() {
  const [text, setText] = useState("");
  const [touched, setTouched] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisState>({ phase: "idle" });
  const [apiStatus, setApiStatus] = useState<ApiStatusState>({ phase: "checking" });
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then((health) => setApiStatus({ phase: "ok", health }))
      .catch(() => {
        if (!controller.signal.aborted) setApiStatus({ phase: "unreachable" });
      });
    return () => {
      controller.abort();
      inFlight.current?.abort();
    };
  }, []);

  const validation = validateArticle(text);
  const showError = touched && !validation.valid;
  const loading = analysis.phase === "loading";
  const demoAvailable = apiStatus.phase === "ok" && apiStatus.health.demo_mode_enabled;

  async function run(demo: boolean) {
    setTouched(true);
    if (!validation.valid) return;

    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setAnalysis({ phase: "loading" });

    try {
      const result = await analyzeArticle(text.trim(), { demo, signal: controller.signal });
      if (!controller.signal.aborted) setAnalysis({ phase: "success", result });
    } catch (error) {
      if (controller.signal.aborted) return;
      setAnalysis({
        phase: "error",
        error:
          error instanceof ApiError
            ? error
            : new ApiError("server", "Something went wrong. Please try again."),
      });
    }
  }

  function clear() {
    inFlight.current?.abort();
    setText("");
    setTouched(false);
    setAnalysis({ phase: "idle" });
  }

  const trimmedLength = text.trim().length;

  return (
    <div className="space-y-4">
      <ApiStatus state={apiStatus} />
      <div className="grid gap-6 lg:grid-cols-2">
        <form
          noValidate
          onSubmit={(event) => {
            event.preventDefault();
            void run(false);
          }}
          className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6 dark:border-slate-800 dark:bg-slate-900"
        >
          <label htmlFor="article" className="text-lg font-semibold">
            Article text
          </label>
          <textarea
            id="article"
            name="article"
            value={text}
            onChange={(event) => setText(event.target.value)}
            onBlur={() => setTouched(true)}
            rows={14}
            disabled={loading}
            aria-invalid={showError}
            aria-describedby="article-help article-error"
            placeholder="Paste the full text of a news article here…"
            className="mt-3 block w-full resize-y rounded-xl border border-slate-300 bg-white p-3 text-sm leading-6 text-slate-900 shadow-inner focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-500/40 disabled:opacity-60 aria-[invalid=true]:border-rose-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
          />
          <div className="mt-2 flex items-start justify-between gap-3 text-xs">
            <p
              id="article-error"
              role={showError ? "alert" : undefined}
              className="text-rose-600 dark:text-rose-400"
            >
              {showError && !validation.valid ? validation.message : null}
            </p>
            <p
              id="article-help"
              className="ml-auto shrink-0 tabular-nums text-slate-600 dark:text-slate-400"
            >
              {trimmedLength.toLocaleString("en-US")} / {MAX_ARTICLE_CHARS.toLocaleString("en-US")}{" "}
              characters (min {MIN_ARTICLE_CHARS})
            </p>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={loading}
              aria-busy={loading}
              className="inline-flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-violet-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-violet-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading && (
                <span
                  aria-hidden
                  className="size-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
                />
              )}
              {loading ? "Analyzing…" : "Analyze"}
            </button>
            <button
              type="button"
              onClick={clear}
              disabled={loading || (text === "" && analysis.phase === "idle")}
              className="rounded-xl border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              Clear
            </button>
            {demoAvailable && (
              <button
                type="button"
                onClick={() => void run(true)}
                disabled={loading}
                className="rounded-xl border border-violet-300 px-4 py-2.5 text-sm font-medium text-violet-800 transition hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-violet-700 dark:text-violet-200 dark:hover:bg-violet-950/40"
              >
                Preview demo result
              </button>
            )}
          </div>
          {demoAvailable && (
            <p className="mt-3 text-xs text-slate-600 dark:text-slate-400">
              Demo mode is on: &ldquo;Preview demo result&rdquo; shows a fixed sample to preview the
              interface. It does not use the model.
            </p>
          )}
        </form>

        <ResultPanel state={analysis} />
      </div>
    </div>
  );
}
