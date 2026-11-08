/* Browser-side Persian RAG demo: dense (E5 via Transformers.js) + lexical
 * (BM25-style) retrieval fused with RRF — no server, everything runs locally. */
import {
  buildBm25, lexicalScores, argsortDesc, rrfFuse, dot, extractiveAnswer,
} from "./retrieval.js";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const progressWrap = $("progress-wrap");
const progressBar = $("progress-bar");
const modeBadge = $("mode-badge");
const answerBox = $("answer");
const tableBody = $("sources-body");
const tableWrap = $("sources-wrap");

let DATA = null;
let BM25 = null;
let EXTRACTOR = null;   // dense encoder (null => lexical-only fallback)

function setStatus(text) { statusEl.textContent = text; }

function showProgress(pct) {
  progressWrap.style.display = "block";
  progressBar.style.width = `${Math.min(100, Math.round(pct))}%`;
  if (pct >= 100) setTimeout(() => { progressWrap.style.display = "none"; }, 600);
}

/* ------------------------------- bootstrapping --------------------------- */
async function boot() {
  setStatus("در حال بارگذاری شاخص (embeddings.json) …");
  DATA = await (await fetch("./embeddings.json")).json();
  BM25 = buildBm25(DATA.chunks);
  $("corpus-info").textContent =
    `${DATA.chunks.length} قطعه از ۶ یادداشت آموزشی · مدل: ${DATA.model}`;

  setStatus("در حال دانلود مدل زبانی (~۱۲۰ مگابایت، فقط بار اول) …");
  try {
    const { pipeline, env } = await import(
      "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3"
    );
    env.allowLocalModels = false;
    if (new URLSearchParams(location.search).has("local")) {
      env.allowLocalModels = true;   // test hook: serve model files from ./models/
    }
    let lastFile = "";
    EXTRACTOR = await pipeline("feature-extraction", DATA.model, {
      dtype: "q8",
      progress_callback: (p) => {
        if (p.status === "progress" && p.file !== lastFile) lastFile = p.file;
        if (p.status === "progress" && p.total) showProgress(p.progress);
      },
    });
    modeBadge.textContent = "حالت: هیبریدی (E5 + BM25 + RRF)";
    modeBadge.className = "badge hybrid";
    setStatus("آماده است — یک پرسش بنویس 🙂");
  } catch (err) {
    console.error("dense backend unavailable:", err);
    modeBadge.textContent = "حالت: واژگانی (BM25) — دانلود مدل ناموفق بود";
    modeBadge.className = "badge lexical";
    setStatus("آماده است (حالت جایگزین) — یک پرسش بنویس 🙂");
  }
  $("ask-btn").disabled = false;
}

/* --------------------------------- answering ----------------------------- */
async function answer() {
  const q = $("question").value.trim();
  if (!q) { answerBox.value = "یک پرسش بنویس 🙂"; return; }
  $("ask-btn").disabled = true;
  setStatus("در حال بازیابی …");

  try {
    const rankings = [];
    if (EXTRACTOR) {
      const out = await EXTRACTOR(DATA.prefix + q, {
        pooling: "mean", normalize: true,
      });
      const qv = out.data;                       // L2-normalised Float32Array
      const scores = DATA.matrix.map((row) => dot(qv, row));
      rankings.push(argsortDesc(scores));
    }
    rankings.push(argsortDesc(lexicalScores(BM25, q)));

    const fused = rrfFuse(rankings).slice(0, 3);
    const hits = fused.map((idx, r) => ({
      ...DATA.chunks[idx],
      ref: r + 1,
      dense: rankings[0].indexOf(idx) + 1,
      lex: rankings[rankings.length - 1].indexOf(idx) + 1,
    }));

    answerBox.value = extractiveAnswer(q, hits);
    renderSources(hits);
    setStatus("آماده است — پرسش بعدی؟");
  } catch (err) {
    console.error(err);
    setStatus("خطا! کنسول مرورگر را ببین.");
  } finally {
    $("ask-btn").disabled = false;
  }
}

function renderSources(hits) {
  tableBody.innerHTML = "";
  for (const h of hits) {
    const tr = document.createElement("tr");
    const docName = document.createElement("td");
    docName.textContent = h.title;
    const ranks = document.createElement("td");
    ranks.textContent = `dense: ${EXTRACTOR ? h.dense : "—"} · bm25: ${h.lex}`;
    const snippet = document.createElement("td");
    snippet.textContent = h.text.slice(0, 160) + "…";
    tr.append(docName, ranks, snippet);
    tableBody.appendChild(tr);
  }
  tableWrap.style.display = "block";
}

/* ---------------------------------- wiring ------------------------------- */
$("ask-btn").addEventListener("click", answer);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); answer(); }
});
document.querySelectorAll(".example").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("question").value = chip.textContent.trim();
    answer();
  });
});

boot();
