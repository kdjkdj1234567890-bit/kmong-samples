# Binwise

**Photo → recycling verdict, entirely in the browser.**

Binwise recognises what you are about to throw away and tells you how to prepare it for
disposal, using a rules file you can swap per district or country. There is no server, no
account, no API key and no upload: the image is classified on the device with a quantised
CLIP model running through WebGPU or WebAssembly.

Live demo: <https://kdjkdj1234567890-bit.github.io/binwise/>

---

## The problem

Recycling outcomes are decided less by *what* you throw away than by *how you prepare it*.
A bottle left with its lid, label or a splash of drink contaminates everything around it in
the bin, and one contaminated bag can send a whole collection to landfill. Meanwhile the rules
differ by district, change often, and are read at the bin, in a hurry, from a laminated sheet.

Existing recycling apps are mostly lookups that require you to type a product name, or
upload a photo to somebody's server. Neither survives contact with a kitchen at 11pm.

## What Binwise does

1. **Recognise** — one photo, zero-shot CLIP classification over a catalogue of 22 common
   household items, each with several natural-language prompts (82 in total).
2. **Decide** — the winning item maps to a code in `rules/*.json`, which carries the disposal
   stream, ordered prep steps, and caveats.
3. **Explain** — the verdict shows the top three candidates with probabilities, the rule file
   and item id that produced it, and a confidence number. If the top two candidates are too
   close, or confidence is below the floor, Binwise **refuses to give a verdict** and asks for
   a better photo or a manual pick. A wrong confident answer is worse than no answer here.

## Why the browser

- **Privacy is structural, not a promise.** The photo never leaves the page; there is no
  endpoint that could receive it. The network panel shows exactly one class of request: the
  model and ruleset being fetched from a public CDN.
- **It runs where the decision is made.** No install, no login, no app-store review.
- **Offline after first load.** The model is cached by the browser via transformers.js.
- **Deployable by anyone.** Static files on any host, including a district office's own.

## How it is built

| Layer | Choice | Why |
| --- | --- | --- |
| Inference | [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) 4.x, `Xenova/clip-vit-base-patch32` (q8) | Zero-shot: new items need prompts, not training data |
| Runtime | WebGPU, falling back to WASM | Works on phones without WebGPU |
| UI | Plain ES modules, no framework, no build step | The whole app is four files; a district can host it |
| Rules | JSON schema (see `rules/kr.json`) | Translators and district clerks can contribute without touching code |
| Accessibility | Semantic HTML, focus-visible styles, `prefers-reduced-motion`, ARIA live regions, keyboard-only file input | Usable at a bin, one-handed, in daylight |

### Rule file schema

```jsonc
{
  "id": "kr-national",
  "name":      { "ko": "…", "en": "…" },
  "updated":   "2026-11",
  "disclaimer":{ "ko": "…", "en": "…" },
  "streams":   { "recycling": { "ko": "재활용", "en": "Recycling" } },
  "items": [{
    "id": "pet_bottle",
    "name":    { "ko": "페트병", "en": "PET bottle" },
    "prompts": ["a clear plastic water bottle", "an empty PET soda bottle"],
    "stream":  "recycling",
    "material":{ "ko": "플라스틱류", "en": "Plastics" },
    "prep":    [{ "ko": "…", "en": "…" }],
    "notes":   { "ko": "…", "en": "…" }
  }]
}
```

Load another file with `?rules=rules/your-district.json`. `rules/template.json` is a
starting point and is **not** a real ordinance.

#### Adding an item

1. Add an object to `items[]` with a stable `id`, both languages, and 2–4 English prompts
   (CLIP is English-trained; write what a person would say, not taxonomy terms).
2. Point `stream` at one of the keys in `streams`.
3. Add the prep steps in the order a person performs them.
4. Re-check the manual picker — it is generated from the same file.

## Honest limitations

- **Zero-shot CLIP confuses similar materials.** A coated paper cup and a paper cup look
  alike; PET and rigid PP look alike. This is exactly why confidence gating exists, and why
  the prep steps, not the label, carry most of the value.
- **Prompts are a design surface.** Accuracy is bounded by how well the prompt set matches
  real households, and English prompts can miss locally specific packaging.
- **Rules are not law.** `rules/kr.json` follows national guidance; districts add their own
  exceptions. The disclaimer is rendered with every verdict on purpose.
- **No measurements yet.** See below for how this should be evaluated before claiming impact.

## How impact should be measured

1. **Contamination routing accuracy** — share of high-risk items (coated cups, greasy paper,
   soiled film) correctly routed to general waste, on a held-out set labelled by a district
   officer.
2. **Prep-step coverage** — share of verdicts that surface at least one prep step.
3. **Refusal rate** — share of verdicts withheld for low confidence. Higher is not worse: it
   is false certainty avoided, and should be reported next to (1), never alone.
4. **Field validation** — bin-audit sampling with a district partner before and after rollout,
   comparing contamination rates in serviced buildings.

## Run it locally

```bash
git clone https://github.com/kdjkdj1234567890-bit/binwise
cd binwise
python3 -m http.server 8000
# open http://localhost:8000
```

Requires a static file server (ES modules and `fetch` of the rules file do not work from
`file://`). First load downloads the quantised model (~60–120MB) once; later loads are served
from the browser cache.

## Review hooks

`window.Binwise` exposes `version`, `rules()`, `model()`, `classify(url)`, and `decide(ranked)`
so a reviewer can verify the pipeline from the console without trusting the UI.

## License

MIT — see [LICENSE](LICENSE). Model: CLIP ViT-B/32 by OpenAI, MIT-licensed weights,
ONNX port by Xenova.
