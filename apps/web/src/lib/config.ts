const DEFAULT_API_BASE_URL = "http://localhost:8000";

/** Backend base URL, from NEXT_PUBLIC_API_BASE_URL (inlined at build time). */
export function getApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  return (raw || DEFAULT_API_BASE_URL).replace(/\/+$/, "");
}
