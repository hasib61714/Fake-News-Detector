import { Analyzer } from "@/components/Analyzer";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6 lg:py-14">
      <header className="mb-8 max-w-2xl">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Fake News Detector</h1>
        <p className="mt-3 text-base text-slate-600 dark:text-slate-400">
          Paste a news article to see whether a text classifier trained on labeled articles thinks
          it resembles fake or real news. It is an estimate based on writing patterns, not a
          fact-check.
        </p>
      </header>
      <Analyzer />
    </main>
  );
}
