import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Analyzer } from "@/components/Analyzer";

const ARTICLE = "The ministry published its quarterly budget report to parliament today. ".repeat(
  2,
);

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function health(overrides: Record<string, unknown> = {}) {
  return {
    status: "ok",
    service: "fake-news-detector-api",
    version: "0.1.0",
    environment: "test",
    model: { status: "ready", loaded: true, detail: "Model loaded.", info: null },
    demo_mode_enabled: false,
    ...overrides,
  };
}

const analysis = (overrides: Record<string, unknown> = {}) => ({
  mode: "model",
  label: "fake",
  confidence: 0.87,
  probabilities: { fake: 0.87, real: 0.13 },
  model: { name: "tfidf-logistic-regression", version: "20260101000000", trained_at: null },
  analyzed_chars: ARTICLE.length,
  disclaimer: "Statistical classifier, not a fact-check.",
  ...overrides,
});

function stubApi(handlers: { health?: () => Response; analyze?: () => Response }) {
  const spy = vi.fn(async (url: string) => {
    if (url.endsWith("/api/v1/health")) return handlers.health?.() ?? json(health());
    if (url.endsWith("/analyze")) return handlers.analyze?.() ?? json(analysis());
    throw new Error(`unexpected request ${url}`);
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

beforeEach(() => {
  stubApi({});
});
afterEach(() => vi.unstubAllGlobals());

describe("<Analyzer />", () => {
  it("shows a validation message and does not call the API for too-short input", async () => {
    const fetchSpy = stubApi({});
    const user = userEvent.setup();
    render(<Analyzer />);

    await user.type(screen.getByLabelText("Article text"), "too short");
    await user.click(screen.getByRole("button", { name: "Analyze" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/too short/i);
    expect(fetchSpy.mock.calls.some(([url]) => String(url).endsWith("/analyze"))).toBe(false);
  });

  it("shows a loading state and then the result", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    const fetchSpy = vi.fn(async (url: string) => {
      if (url.endsWith("/health")) return json(health());
      await gate;
      return json(analysis());
    });
    vi.stubGlobal("fetch", fetchSpy);
    const user = userEvent.setup();
    render(<Analyzer />);

    await user.click(screen.getByLabelText("Article text"));
    await user.paste(ARTICLE);
    await user.click(screen.getByRole("button", { name: "Analyze" }));

    expect(await screen.findByRole("button", { name: "Analyzing…" })).toBeDisabled();
    release();

    expect(await screen.findByText("Likely fake")).toBeInTheDocument();
    expect(screen.getByText("87.0%", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyze" })).toBeEnabled();
  });

  it("explains clearly when the model has not been trained", async () => {
    stubApi({
      analyze: () =>
        json(
          { error: { code: "model_not_trained", message: "Model not trained yet.", fields: [] } },
          503,
        ),
    });
    const user = userEvent.setup();
    render(<Analyzer />);

    await user.click(screen.getByLabelText("Article text"));
    await user.paste(ARTICLE);
    await user.click(screen.getByRole("button", { name: "Analyze" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Model not trained yet");
    expect(alert).toHaveTextContent("make train");
    expect(screen.queryByText("Likely fake")).not.toBeInTheDocument();
    expect(screen.queryByText("Likely real")).not.toBeInTheDocument();
  });

  it("shows a friendly error when the API is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    const user = userEvent.setup();
    render(<Analyzer />);

    expect(await screen.findByText("API unreachable")).toBeInTheDocument();
    await user.click(screen.getByLabelText("Article text"));
    await user.paste(ARTICLE);
    await user.click(screen.getByRole("button", { name: "Analyze" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not reach the analysis service",
    );
  });

  it("hides the demo button unless the backend enables demo mode", async () => {
    render(<Analyzer />);
    await screen.findByText(/model ready/i);
    expect(screen.queryByRole("button", { name: /demo/i })).not.toBeInTheDocument();
  });

  it("labels demo results as not produced by the model", async () => {
    const fetchSpy = vi.fn(async (url: string) => {
      if (url.endsWith("/health")) return json(health({ demo_mode_enabled: true }));
      if (url.endsWith("/demo/analyze")) {
        return json(
          analysis({
            mode: "demo",
            model: null,
            confidence: 0.5,
            probabilities: { fake: 0.5, real: 0.5 },
            disclaimer: "DEMO ONLY: not produced by the model.",
          }),
        );
      }
      throw new Error(`unexpected request ${url}`);
    });
    vi.stubGlobal("fetch", fetchSpy);
    const user = userEvent.setup();
    render(<Analyzer />);

    await user.click(screen.getByLabelText("Article text"));
    await user.paste(ARTICLE);
    await user.click(await screen.findByRole("button", { name: "Preview demo result" }));

    await waitFor(() => expect(screen.getByRole("note")).toHaveTextContent("Demo preview"));
    expect(screen.getByText("No real verdict")).toBeInTheDocument();
    expect(screen.queryByText("Likely fake")).not.toBeInTheDocument();
  });
});
