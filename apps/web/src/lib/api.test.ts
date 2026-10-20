import { afterEach, describe, expect, it, vi } from "vitest";

import { analyzeArticle, ApiError, getHealth } from "@/lib/api";

function mockFetch(impl: (url: string, init?: RequestInit) => Promise<Response>) {
  const spy = vi.fn(impl);
  vi.stubGlobal("fetch", spy);
  return spy;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("api client", () => {
  it("posts the article to the base URL from the environment", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://api.test/");
    const fetchSpy = mockFetch(async () => json({ label: "real" }));

    await analyzeArticle("hello world");

    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("http://api.test/api/v1/analyze");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ text: "hello world" });
  });

  it("uses the separate demo endpoint only when asked", async () => {
    const fetchSpy = mockFetch(async () => json({ mode: "demo" }));
    await analyzeArticle("hello world", { demo: true });
    expect(fetchSpy.mock.calls[0][0]).toContain("/api/v1/demo/analyze");
  });

  it("maps the 'model not trained' response to a typed error", async () => {
    mockFetch(async () =>
      json(
        { error: { code: "model_not_trained", message: "Model not trained yet.", fields: [] } },
        503,
      ),
    );
    await expect(analyzeArticle("x")).rejects.toMatchObject({
      kind: "model_not_trained",
      message: "Model not trained yet.",
    });
  });

  it("maps network failures to a friendly error", async () => {
    mockFetch(async () => {
      throw new TypeError("Failed to fetch");
    });
    const error = await getHealth().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ kind: "network" });
  });

  it("handles non-JSON error bodies", async () => {
    mockFetch(async () => new Response("<html>Bad gateway</html>", { status: 502 }));
    await expect(getHealth()).rejects.toMatchObject({ kind: "server" });
  });
});
