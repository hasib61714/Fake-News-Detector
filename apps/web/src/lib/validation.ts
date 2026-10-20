// Keep in sync with services/api/app/schemas.py (MIN_ARTICLE_CHARS / MAX_ARTICLE_CHARS).
export const MIN_ARTICLE_CHARS = 50;
export const MAX_ARTICLE_CHARS = 20_000;

export type ValidationResult = { valid: true } | { valid: false; message: string };

/** Mirrors the server rule: length is measured after trimming whitespace. */
export function validateArticle(text: string): ValidationResult {
  const length = text.trim().length;
  if (length === 0) {
    return { valid: false, message: "Paste the article text to analyze." };
  }
  if (length < MIN_ARTICLE_CHARS) {
    return {
      valid: false,
      message: `Article is too short (${length}/${MIN_ARTICLE_CHARS} characters minimum).`,
    };
  }
  if (length > MAX_ARTICLE_CHARS) {
    return {
      valid: false,
      message: `Article is too long (${length.toLocaleString("en-US")}/${MAX_ARTICLE_CHARS.toLocaleString("en-US")} characters maximum).`,
    };
  }
  return { valid: true };
}
