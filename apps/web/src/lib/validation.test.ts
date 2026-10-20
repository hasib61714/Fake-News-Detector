import { describe, expect, it } from "vitest";

import { MAX_ARTICLE_CHARS, MIN_ARTICLE_CHARS, validateArticle } from "@/lib/validation";

describe("validateArticle", () => {
  it("rejects empty and whitespace-only input", () => {
    expect(validateArticle("")).toMatchObject({ valid: false });
    expect(validateArticle("   \n\t ")).toMatchObject({ valid: false });
  });

  it("rejects text shorter than the minimum, measured after trimming", () => {
    const result = validateArticle(`  ${"a".repeat(MIN_ARTICLE_CHARS - 1)}  `);
    expect(result).toMatchObject({ valid: false });
    expect(result.valid === false && result.message).toContain("too short");
  });

  it("accepts text at the minimum and maximum lengths", () => {
    expect(validateArticle("a".repeat(MIN_ARTICLE_CHARS))).toEqual({ valid: true });
    expect(validateArticle("a".repeat(MAX_ARTICLE_CHARS))).toEqual({ valid: true });
  });

  it("rejects text longer than the maximum", () => {
    const result = validateArticle("a".repeat(MAX_ARTICLE_CHARS + 1));
    expect(result.valid === false && result.message).toContain("too long");
  });
});
