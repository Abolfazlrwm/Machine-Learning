/* Retrieval core shared by the browser demo and the Node tests.
 * Ports the same ideas as rag-farsi-assistant: lexical BM25-style scoring,
 * dense cosine similarity and Reciprocal-Rank-Fusion (RRF, k = 60). */

export const STOPWORDS = new Set([
  "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای", "آن",
  "یک", "های", "می", "شد", "کند", "می‌شود", "the", "a", "an", "of", "and",
]);

export function normalise(text) {
  return text
    .normalize("NFC")
    .replace(/\u064a/g, "\u06cc")   /* Arabic Yeh -> Persian Yeh */
    .replace(/\u0643/g, "\u06a9");  /* Arabic Keh -> Persian Keh */
}

export function tokenize(text) {
  return normalise(text).toLowerCase().match(/[\w\u0600-\u06FF]+/g) || [];
}

export function contentTokens(text) {
  return tokenize(text).filter((t) => !STOPWORDS.has(t) && t.length > 1);
}

/* ----------------------------- BM25-style scorer ------------------------- */
export function buildBm25(chunks) {
  const docTokens = chunks.map((c) => contentTokens(c.text));
  const df = new Map();
  for (const toks of docTokens) {
    for (const t of new Set(toks)) df.set(t, (df.get(t) || 0) + 1);
  }
  const n = chunks.length;
  const idf = new Map();
  for (const [t, d] of df) idf.set(t, Math.log((n - d + 0.5) / (d + 0.5) + 1));
  return { docTokens, idf, n };
}

export function lexicalScores(bm25, question) {
  const qTokens = contentTokens(question);
  return bm25.docTokens.map((toks) => {
    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
    let score = 0;
    for (const t of qTokens) {
      const f = tf.get(t);
      if (f && bm25.idf.has(t)) score += bm25.idf.get(t) * ((3.0 * f) / (f + 1.2));
    }
    return score;
  });
}

/* --------------------------------- fusion -------------------------------- */
export function argsortDesc(scores) {
  return scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).map((p) => p[1]);
}

export function rrfFuse(rankings, k = 60) {
  const fused = new Map();
  for (const ranking of rankings) {
    ranking.forEach((docIdx, rank) => {
      fused.set(docIdx, (fused.get(docIdx) || 0) + 1 / (k + rank + 1));
    });
  }
  return [...fused.entries()].sort((a, b) => b[1] - a[1]).map(([i]) => i);
}

export function dot(vec, row) {
  let s = 0;
  for (let i = 0; i < row.length; i++) s += row[i] * vec[i];
  return s;
}

/* ------------------------ extractive cited answer ------------------------ */
export function extractiveAnswer(question, hits) {
  const qTokens = new Set(contentTokens(question));
  const parts = ["پاسخ بر پایه‌ی منابع بازیابی‌شده:", ""];
  for (let rank = 0; rank < hits.length; rank++) {
    const hit = hits[rank];
    const sentences = hit.text.split(/(?<=[.!؟])\s+|\n+/);
    const scored = [];
    for (const s of sentences) {
      const overlap = contentTokens(s).filter((t) => qTokens.has(t)).length;
      if (overlap) scored.push([overlap, s.trim()]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    const best = scored.slice(0, 2).map((p) => p[1]).join(" ") || hit.text.slice(0, 200);
    parts.push(`[${rank + 1}] ${best}`);
    parts.push("");
  }
  parts.push("برای پاسخِ تولیدشده با مدل زبانی، نسخهٔ کامل مخزن را ببینید.");
  return parts.join("\n");
}
