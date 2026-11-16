/* TinyGPT static Space — loads Pyodide (Python/WASM), vendors the repo's
   src/ stack + the committed checkpoint into the Pyodide FS, then wires the
   Persian UI. The model runs fully client-side. */
"use strict";

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const PY_FILES = ["__init__.py", "autograd.py", "model.py", "tokenizer.py"];
const MODEL_FILES = ["params.npz", "tokenizer.json", "train_log.json"];
const EXAMPLES = [
  { prompt: "یادگیری ماشین", temperature: 0.8, topk: 12, nchars: 220, seed: 7 },
  { prompt: "شبکهٔ عصبی", temperature: 0.9, topk: 20, nchars: 300, seed: 21 },
  { prompt: "بیش‌برازش چیست", temperature: 0.7, topk: 8, nchars: 200, seed: 3 },
];

const $ = (id) => document.getElementById(id);

function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + (cls || "");
}

async function fetchBytes(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fetch ${path} → HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function boot() {
  const btn = $("generate");
  btn.disabled = true;
  setStatus("بارگذاری Pyodide — Python در WebAssembly… (بار اول ~۱۰ مگابایت)", "busy");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });

  setStatus("بارگذاری NumPy…", "busy");
  await pyodide.loadPackage("numpy");

  setStatus("بارگذاری کد و چک‌پوینت مدل…", "busy");
  pyodide.FS.mkdir("/space");
  pyodide.FS.mkdir("/space/src");
  pyodide.FS.mkdir("/model");
  for (const f of PY_FILES) pyodide.FS.writeFile(`/space/src/${f}`, await fetchBytes(`src/${f}`));
  for (const f of MODEL_FILES) pyodide.FS.writeFile(`/model/${f}`, await fetchBytes(`model/${f}`));

  await pyodide.runPythonAsync($("py-bootstrap").textContent);
  window.__generate = pyodide.globals.get("generate");

  const info = JSON.parse(pyodide.runPython("json.dumps(INFO, ensure_ascii=False)"));
  $("stats").textContent =
    `واژگان ${info.vocab} نویسه · ${info.params.toLocaleString("fa-IR")} پارامتر · ` +
    `آموزش ${info.train_sec} ثانیه روی CPU · loss نهایی ${info.train_loss} / اعتبارسنجی ${info.val_loss}`;

  setStatus("آماده است — کل مدل NumPy در مرورگر شما اجرا می‌شود ✔", "ready");
  btn.disabled = false;
}

async function generate() {
  const btn = $("generate");
  const args = [
    $("prompt").value,
    parseFloat($("temperature").value),
    parseInt($("topk").value, 10),
    parseInt($("nchars").value, 10),
    parseInt($("seed").value || "7", 10),
  ];
  btn.disabled = true;
  setStatus("در حال تولید… (چند صد پاس رو به جلو در WebAssembly)", "busy");
  await new Promise((r) => setTimeout(r, 30)); // let the status paint first
  try {
    const res = JSON.parse(window.__generate(...args));
    $("output").value = res.text;
    $("stats").textContent = res.note;
    setStatus("آماده است ✔", "ready");
  } catch (err) {
    setStatus("خطا در تولید: " + err.message, "error");
    console.error(err);
  } finally {
    btn.disabled = false;
  }
}

function wire() {
  $("generate").addEventListener("click", generate);
  for (const [slider, label] of [["temperature", "t-val"], ["topk", "k-val"], ["nchars", "n-val"]]) {
    $(slider).addEventListener("input", (e) => { $(label).textContent = e.target.value; });
  }
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const ex = EXAMPLES.find((e) => e.prompt === chip.dataset.prompt) || EXAMPLES[0];
      $("prompt").value = ex.prompt;
      $("temperature").value = ex.temperature; $("t-val").textContent = ex.temperature;
      $("topk").value = ex.topk; $("k-val").textContent = ex.topk;
      $("nchars").value = ex.nchars; $("n-val").textContent = ex.nchars;
      $("seed").value = ex.seed;
    });
  });
}

wire();
boot().catch((err) => {
  setStatus("بارگذاری ناموفق: " + err.message, "error");
  console.error(err);
});
