/**
 * Binwise — offline-first recycling verdicts in the browser.
 * MIT License · 2026 Dongjin Kim
 *
 * Everything runs on the device: the image never leaves the page and
 * inference happens in WebGPU/WASM through transformers.js.
 */

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";
const MODEL_ID = "Xenova/clip-vit-base-patch32";
const DEFAULT_RULES = "rules/kr.json";

// An item's score is the summed probability of its prompts, so a real match lands
// around 0.35–0.60 and a photo of something outside the catalogue stays flat near
// 0.05–0.15. The floor catches the second case; the margin catches a photo that two
// catalogue items could each explain.
const CONFIDENCE_FLOOR = 0.25;  // below this the model is guessing
const CONFIDENCE_MARGIN = 0.08; // top-2 too close -> ask for another photo

const state = {
  lang: localStorage.getItem("binwise.lang") || "ko",
  rules: null,
  rulesUrl: DEFAULT_RULES,
  classifier: null,
  modelDevice: null,
  modelError: null,
  loading: false,
  imageUrl: null,
  results: null,
  selectedItemId: null,
};

const $ = (sel) => document.querySelector(sel);

/* ------------------------------------------------------------------ */
/* i18n                                                                */
/* ------------------------------------------------------------------ */

const KO_ORIGINALS = new Map();

// Strings that are only ever produced by script (not present as data-i18n in the
// HTML) need their Korean text here, otherwise the key would leak into the UI.
const KO = {
  scanning: "사진을 살펴보는 중…",
  model_loading: "모델을 불러오는 중… {pct}",
  model_loading_plain: "모델을 불러오는 중…",
  model_ready: "모델 준비 완료 · {device} · 사진은 이 기기를 벗어나지 않습니다.",
  model_error: "모델을 불러오지 못했습니다 ({err}). 아래에서 품목을 직접 고를 수 있습니다.",
  confidence: "확신도 {pct}%",
  prep_title: "이렇게 준비하세요",
  notes_title: "주의",
  candidates_title: "모델이 본 것",
  applied_rule: "적용 규칙: {file} · 품목: {item}",
  uncertain_title: "확신도가 낮아 판정을 보류했습니다",
  uncertain_body: "사진이 모호합니다. 물건 하나가 화면 가운데 오게 다시 찍거나, 아래에서 품목을 직접 골라 주세요.",
  food_hint: "액체는 따라 버리고 배출하세요.",
  disclaimer: "지자체 조례에 따라 다를 수 있습니다. 헷갈리면 주민센터에 문의하세요.",
  rules_loaded: "{n}개 품목 · {date} 갱신",
};

const EN = {
  tagline: "Sorting assistant",
  nav_tool: "Sort it",
  nav_how: "How it works",
  nav_rules: "Rule data",
  hero_title: "One photo, and you know exactly which bin it goes in.",
  hero_lede: "Binwise recognises the item inside your browser and checks it against local sorting rules to show <strong>the stream, the prep steps and the caveats</strong>. Photos never leave the device, and once the model is cached it works offline.",
  hero_p1: "<span aria-hidden=\"true\">🔒</span> No upload — inference happens on your device",
  hero_p2: "<span aria-hidden=\"true\">🌏</span> Rules are JSON — swap in your own district",
  hero_p3: "<span aria-hidden=\"true\">🧾</span> Evidence shown — the rule that decided it is cited",
  drop_title: "Take a photo or choose a file",
  drop_hint: "One item, centred in the frame",
  clear: "Retake",
  model_idle: "Preparing the model… (about 60–120MB downloaded once on first run)",
  retry_model: "Retry",
  manual_summary: "Pick the item without a photo",
  empty_state: "Add a photo and the <strong>disposal stream</strong> and <strong>prep steps</strong> appear here.",
  how_title: "How a verdict is made",
  how_1_t: "1. Recognition on the device",
  how_1_b: "A zero-shot CLIP image classifier runs on WebGPU or WASM. No server, no API key, no account.",
  how_2_t: "2. Matched against rules",
  how_2_b: "The recognised item is mapped to a code in the local rule file, which supplies the stream, prep steps and exceptions.",
  how_3_t: "3. Explained, not asserted",
  how_3_b: "When confidence is low the app refuses to guess: it shows the top three candidates with probabilities and the rule id it applied.",
  how_note: "<strong>Honest limits:</strong> Binwise helps you decide; it is not a legal ruling. Local ordinances differ, so follow your district office for contaminated or ambiguous items. Refusing to sound certain when it is not is the core design choice here.",
  rules_title: "Rules are data",
  rules_body: "Decision logic and the item catalogue live in <code>rules/*.json</code>. Add a file per district or country and the same app speaks that jurisdiction. See the README for the schema.",
  rules_doc: "Active ruleset",
  rules_items: "Items",
  rules_model: "Model",
  impact_title: "Why this exists",
  impact_1_t: "Problem",
  impact_1_b: "Recycling is decided less by what you throw away than by how you prepare it. One bottle left with a lid, a label or a splash of drink contaminates the whole bin. Rules vary by district and people are in a hurry.",
  impact_2_t: "Approach",
  impact_2_b: "One photo recognises the item and returns the prep steps as a checklist. It only gets used if there is nothing to install, join or upload — so it runs entirely in the browser.",
  impact_3_t: "How we would measure it",
  impact_3_b: "① Share of contaminated-risk items (coated cups, greasy paper, soiled film) correctly routed to general waste. ② Share of verdicts that surface prep steps. ③ Share of verdicts refused for low confidence — read as avoided false certainty, not as an error rate. Validating against field error rates needs bin-audit sampling with a district partner.",
  footer_1: "Binwise · MIT License · 2026 Dongjin Kim",
  footer_2: "Photos never leave this browser. Only the model file is fetched from a public CDN, once.",
  scanning: "Looking at the photo…",
  model_loading_plain: "Loading the model…",
  model_loading: "Loading the model… {pct}%",
  model_ready: "Model ready on {device}. Photos stay on this device.",
  model_error: "Could not load the model ({err}). You can still pick the item manually below.",
  confidence: "confidence {pct}%",
  prep_title: "How to prepare it",
  notes_title: "Watch out",
  candidates_title: "What the model saw",
  applied_rule: "rule: {file} · item: {item}",
  uncertain_title: "Not sure enough",
  uncertain_body: "The photo is ambiguous, so no verdict is given. Try again with one item filling the frame, or pick the item manually below.",
  food_hint: "Drain liquid before the bin.",
  disclaimer: "Local rules can differ. When in doubt, ask your district office.",
  rules_loaded: "{n} items · updated {date}",
};

function t(key, vars) {
  const table = state.lang === "en" ? EN : KO;
  const dict = table || {};
  const fromDict = dict[key];
  let text;
  if (fromDict !== undefined) text = fromDict;
  else if (KO_ORIGINALS.has(key)) text = KO_ORIGINALS.get(key);
  else text = key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) text = text.replaceAll(`{${k}}`, String(v));
  }
  return text;
}

function captureOriginals() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (!KO_ORIGINALS.has(key)) KO_ORIGINALS.set(key, el.innerHTML);
  });
}

function applyLanguage() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    const original = KO_ORIGINALS.get(key);
    if (state.lang === "en") {
      if (EN[key] !== undefined) el.innerHTML = EN[key];
    } else if (original !== undefined) {
      el.innerHTML = original;
    }
  });
  const langButton = $("#langButton");
  if (langButton) {
    langButton.textContent = state.lang === "ko" ? "EN" : "한국어";
    langButton.setAttribute("aria-label", state.lang === "ko" ? "영어로 보기" : "View in Korean");
  }
  renderStatus();
  renderManualGrid();
  renderRulesMeta();
  if (state.results) renderVerdict(state.results);
  else renderEmpty();
}

const L = (obj) => (obj && typeof obj === "object" ? obj[state.lang] || obj.ko || obj.en : obj);

/* ------------------------------------------------------------------ */
/* Rules                                                               */
/* ------------------------------------------------------------------ */

async function loadRules() {
  const params = new URLSearchParams(location.search);
  const url = params.get("rules") || DEFAULT_RULES;
  state.rulesUrl = url;
  try {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!Array.isArray(data.items)) throw new Error("rules file has no items[]");
    state.rules = data;
  } catch (err) {
    state.rules = null;
    console.error("Binwise: could not load rules", err);
  }
  renderRulesMeta();
  renderManualGrid();
}

function itemById(id) {
  return state.rules?.items.find((it) => it.id === id) || null;
}

function renderRulesMeta() {
  const label = $("#rulesLabel");
  const count = $("#rulesCount");
  const model = $("#modelLabel");
  if (label) label.textContent = state.rules ? L(state.rules.name) : "—";
  if (count) count.textContent = state.rules ? String(state.rules.items.length) : "—";
  if (model) model.textContent = state.modelDevice ? `${MODEL_ID} · ${state.modelDevice}` : MODEL_ID;
}

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

// `"gpu" in navigator` is true in many browsers where no adapter can be created
// (software rendering, headless, blocked GPU). Asking for an adapter costs little
// and keeps a dead WebGPU attempt from poisoning the WASM fallback.
async function probeWebGPU() {
  try {
    if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

function setStatus(kind, text) {
  const dot = document.querySelector("#modelStatus .dot");
  const span = $("#modelStatusText");
  if (dot) dot.className = `dot ${kind}`;
  if (span) span.textContent = text;
}

function renderStatus() {
  if (state.modelError) {
    setStatus("err", t("model_error", { err: state.modelError }));
    $("#warmButton")?.removeAttribute("hidden");
  } else if (state.classifier) {
    setStatus("ok", t("model_ready", { device: state.modelDevice }));
    $("#warmButton")?.setAttribute("hidden", "");
  } else if (state.loading) {
    setStatus("busy", t("model_loading_plain"));
  } else {
    setStatus("", t("model_idle"));
  }
}

async function loadModel() {
  if (state.classifier || state.loading) return;
  state.loading = true;
  state.modelError = null;
  renderStatus();
  try {
    const mod = await import(TRANSFORMERS_URL);
    const { pipeline, env } = mod;
    env.allowLocalModels = false;
    env.useBrowserCache = true;

    const hasWebGPU = await probeWebGPU();
    const attempts = hasWebGPU ? ["webgpu", "wasm"] : ["wasm"];
    const errors = [];

    for (const device of attempts) {
      try {
        state.classifier = await pipeline("zero-shot-image-classification", MODEL_ID, {
          dtype: "q8",
          device,
        });
        state.modelDevice = device;
        break;
      } catch (err) {
        errors.push(`${device}: ${err?.message || err}`);
        state.classifier = null;
        console.warn(`Binwise: ${device} failed`, err);
      }
    }
    if (!state.classifier) throw new Error(errors.join(" / "));
  } catch (err) {
    state.classifier = null;
    state.modelError = err?.message || String(err);
  } finally {
    state.loading = false;
    renderStatus();
    renderRulesMeta();
  }
}

/* ------------------------------------------------------------------ */
/* Classification                                                      */
/* ------------------------------------------------------------------ */

function buildLabelMap() {
  const map = [];
  for (const item of state.rules.items) {
    for (const prompt of item.prompts) map.push({ label: prompt, itemId: item.id });
  }
  return map;
}

async function classifyImage(url) {
  if (!state.classifier) throw new Error("model not loaded");
  const labelMap = buildLabelMap();
  const labels = labelMap.map((entry) => entry.label);
  const raw = await state.classifier(url, labels);

  // Aggregate to an item posterior: an item's score is the probability mass of all
  // of its prompts. Taking the single best prompt instead would penalise the items
  // that describe themselves in more ways and reward the terse ones.
  const totals = new Map();
  for (const row of raw) {
    const mapped = labelMap.find((m) => m.label === row.label);
    if (!mapped) continue;
    const current = totals.get(mapped.itemId) || { itemId: mapped.itemId, prob: 0, best: 0, label: row.label };
    current.prob += row.score;
    if (row.score > current.best) {
      current.best = row.score;
      current.label = row.label;
    }
    totals.set(mapped.itemId, current);
  }
  const ranked = [...totals.values()].sort((a, b) => b.prob - a.prob);

  // The sum is the probability the image matches that item at all, which is what the
  // gate and the reader both need; it is not re-normalised again on top.
  return ranked;
}

function decide(ranked) {
  if (!ranked.length) return { uncertain: true, reason: "empty" };
  const [first, second] = ranked;
  if (first.prob < CONFIDENCE_FLOOR) return { uncertain: true, reason: "low" };
  if (second && first.prob - second.prob < CONFIDENCE_MARGIN) return { uncertain: true, reason: "close" };
  return { uncertain: false, item: itemById(first.itemId), ranked };
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */

function renderEmpty() {
  const panel = $("#verdictPanel");
  if (!panel) return;
  panel.innerHTML = `<p class="empty">${t("empty_state")}</p>`;
}

function renderManualGrid() {
  const grid = $("#manualGrid");
  if (!grid || !state.rules) return;
  grid.innerHTML = "";
  for (const item of state.rules.items) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = L(item.name);
    button.addEventListener("click", () => {
      state.results = { manual: true, item };
      state.selectedItemId = item.id;
      renderVerdict(state.results);
    });
    grid.appendChild(button);
  }
}

function streamLabel(stream) {
  const s = state.rules?.streams?.[stream];
  return s ? L(s) : stream;
}

function renderVerdict(results) {
  const panel = $("#verdictPanel");
  if (!panel) return;

  if (results.manual) {
    panel.innerHTML = "";
    panel.appendChild(verdictNode(results.item, null));
    return;
  }

  const decision = decide(results.ranked);
  panel.innerHTML = "";

  if (decision.uncertain) {
    const wrap = document.createElement("div");
    wrap.className = "uncertain";
    wrap.innerHTML = `
      <p class="verdict-item">${t("uncertain_title")}</p>
      <p class="note">${t("uncertain_body")}</p>
    `;
    wrap.appendChild(candidateList(results.ranked.slice(0, 3)));
    panel.appendChild(wrap);
    return;
  }

  panel.appendChild(verdictNode(decision.item, results.ranked));
}

function candidateList(ranked) {
  const box = document.createElement("div");
  box.className = "candidates";
  box.innerHTML = `<h4>${t("candidates_title")}</h4>`;
  for (const row of ranked) {
    const item = itemById(row.itemId);
    const pct = Math.round((row.prob || row.score) * 100);
    const line = document.createElement("div");
    line.className = "bar-row";
    line.innerHTML = `
      <div>
        <div style="display:flex;justify-content:space-between;gap:.5rem">
          <span>${item ? L(item.name) : row.label}</span>
        </div>
        <div class="bar"><span style="width:${Math.max(3, Math.min(100, pct * 2.5))}%"></span></div>
      </div>
      <span class="pct">${pct}%</span>
    `;
    box.appendChild(line);
  }
  return box;
}

function verdictNode(item, ranked) {
  const node = document.createElement("div");
  const pct = ranked ? Math.round((ranked[0].prob || ranked[0].score) * 100) : null;

  const head = document.createElement("div");
  head.className = "verdict-head";
  head.innerHTML = `
    <div>
      <p class="verdict-item">${L(item.name)}</p>
      <span class="stream ${item.stream}">${streamLabel(item.stream)}</span>
    </div>
    ${pct !== null ? `<span class="verdict-conf">${t("confidence", { pct })}</span>` : ""}
  `;
  node.appendChild(head);

  if (item.prep?.length) {
    const list = document.createElement("ol");
    list.className = "prep";
    item.prep.forEach((step, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="num" aria-hidden="true">${i + 1}</span><span>${L(step)}</span>`;
      list.appendChild(li);
    });
    node.appendChild(list);
  }

  if (item.notes) {
    const note = document.createElement("p");
    note.className = "note";
    note.innerHTML = `<strong>${t("notes_title")}</strong> ${L(item.notes)}`;
    node.appendChild(note);
  }

  if (ranked && ranked.length > 1) node.appendChild(candidateList(ranked.slice(0, 3)));

  const rule = document.createElement("p");
  rule.className = "rule-id";
  rule.textContent = t("applied_rule", { file: state.rulesUrl, item: item.id });
  node.appendChild(rule);

  const disclaimer = document.createElement("p");
  disclaimer.className = "rule-id";
  disclaimer.textContent = t("disclaimer");
  node.appendChild(disclaimer);

  return node;
}

/* ------------------------------------------------------------------ */
/* Input wiring                                                        */
/* ------------------------------------------------------------------ */

function setPreview(url) {
  const wrap = $("#previewWrap");
  const img = $("#preview");
  const drop = $("#dropZone");
  if (!wrap || !img) return;
  if (!url) {
    wrap.setAttribute("hidden", "");
    img.removeAttribute("src");
    drop?.removeAttribute("hidden");
    return;
  }
  img.src = url;
  wrap.removeAttribute("hidden");
  // The photo replaces the empty drop target instead of stacking under it.
  drop?.setAttribute("hidden", "");
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = URL.createObjectURL(file);
  setPreview(state.imageUrl);
  state.results = null;

  if (!state.classifier) {
    await loadModel();
  }
  if (!state.classifier) {
    renderEmpty();
    return;
  }

  setStatus("busy", t("scanning"));
  try {
    const ranked = await classifyImage(state.imageUrl);
    state.results = { ranked };
    renderVerdict(state.results);
  } catch (err) {
    console.error("Binwise: classification failed", err);
    state.results = null;
    renderEmpty();
  } finally {
    renderStatus();
  }
}

function wireInputs() {
  const input = $("#fileInput");
  const drop = $("#dropZone");

  input?.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
  });

  if (drop) {
    ["dragenter", "dragover"].forEach((type) =>
      drop.addEventListener(type, (event) => {
        event.preventDefault();
        drop.classList.add("is-over");
      })
    );
    ["dragleave", "drop"].forEach((type) =>
      drop.addEventListener(type, (event) => {
        event.preventDefault();
        drop.classList.remove("is-over");
      })
    );
    drop.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) handleFile(file);
    });
  }

  $("#clearButton")?.addEventListener("click", () => {
    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
    state.results = null;
    setPreview(null);
    if ($("#fileInput")) $("#fileInput").value = "";
    renderEmpty();
  });

  $("#warmButton")?.addEventListener("click", () => loadModel());

  $("#langButton")?.addEventListener("click", () => {
    state.lang = state.lang === "ko" ? "en" : "ko";
    localStorage.setItem("binwise.lang", state.lang);
    applyLanguage();
  });

  // Expose a tiny hook so reviewers can verify the pipeline from the console.
  window.Binwise = {
    version: "1.0.0",
    rules: () => state.rules,
    model: () => ({ id: MODEL_ID, device: state.modelDevice, ready: Boolean(state.classifier) }),
    classify: classifyImage,
    decide,
  };
}

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

async function boot() {
  captureOriginals();
  wireInputs();
  await loadRules();
  applyLanguage();
  renderEmpty();
  loadModel();
}

boot();
