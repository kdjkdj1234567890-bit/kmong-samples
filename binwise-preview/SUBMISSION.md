# Arbiter Hacks V1 — submission text

**Track:** Rethinking Waste · **Team:** solo (Dongjin Kim) · **Repo:** `https://github.com/kdjkdj1234567890-bit/binwise` · **Demo:** `https://kdjkdj1234567890-bit.github.io/binwise/`

---

## Project name

Binwise

## Tagline (60 chars)

Photo in, correct bin out — on-device, no upload.

## What it does

Binwise is a browser app that looks at what you are about to throw away and tells you how to
prepare it for disposal. One photo runs a zero-shot CLIP classifier over a catalogue of 22
common household items — PET bottles, coated paper cups, styrofoam trays, greasy pizza boxes,
batteries, dead cables, ceramics and so on — entirely on the device. The winning item is then
looked up in a **rules file** (`rules/kr.json`) that carries the disposal stream, the ordered
prep steps a person actually performs, and the caveats that decide whether something is
recyclable at all.

Three behaviours matter more than the classifier:

1. **It refuses when it is not sure.** If the top score is below the floor, or the top two
   candidates are within a margin, no verdict is shown. The app says "not sure enough", shows
   the three candidates with probabilities, and asks for a better photo or a manual pick.
2. **It shows its work.** Every verdict carries the rule file, the item id, the confidence and
   the top-3 candidates, so a person can see why the app said what it said.
3. **It never uploads.** There is no endpoint to receive a photo. The only network request is
   the model file itself, once, from a public CDN.

## The waste problem this targets

Recycling yield is decided at the sink, not at the sorting plant. A PET bottle left with its
cap, its label and a splash of drink is still "a bottle" to the person holding it, but it is a
contaminant to the collection: it makes the bale wet, mixes resin types, and in a curbside
system it is exactly the kind of item that turns a recyclable load into residue. Coated paper
cups, greasy pizza boxes and food-soiled film are worse — they *look* recyclable and are not,
and people put them in the recycling bin precisely because they are trying to do the right
thing.

**How we would measure whether it worked.** Three numbers, collected in a district with a
partner: (1) *contamination routing accuracy* — the share of high-risk items (coated cups,
greasy paper, soiled film) that the tool routes to general waste, checked against a set labelled
by a district officer; (2) *prep-step coverage* — the share of verdicts that surface at least
one prep step, because prep is where the contamination is actually prevented; (3) *refusal
rate* — the share of verdicts withheld for low confidence, reported next to (1) and read as
false certainty avoided, never as an error rate on its own. Field validation would compare bin
contamination rates before and after rollout in serviced buildings. None of these numbers are
claimed yet; this build is the instrument for collecting them.

## How we built it

- **Inference:** `@huggingface/transformers` 4.x (`transformers.js`) with
  `Xenova/clip-vit-base-patch32` at q8, on WebGPU with an automatic WASM fallback. Zero-shot was
  chosen deliberately: adding a new packaging type means adding a prompt, not labelling a
  dataset, which matters for a tool that has to track packaging that changes every season.
- **Aggregation:** the model scores 82 prompts; an item's score is the summed probability of
  its own prompts, so "a clear plastic water bottle" and "an empty PET soda bottle" add up for
  the same item instead of competing, and an item that describes itself in four ways is not
  penalised against one that has three.
- **Decision layer:** a confidence floor plus a top-2 margin, both tuned on real photos, and a
  manual grid generated from the same rules file so the tool is still usable when the model is
  wrong or the photo is bad.
- **Rules as data:** streams, prep steps, notes and disclaimers live in JSON with Korean and
  English strings. `?rules=rules/your-district.json` swaps the jurisdiction; a district clerk
  can fix a rule without touching code.
- **UI:** no framework, no build step — four files, static hosting, works on a phone at a bin,
  keyboard accessible, honours `prefers-reduced-motion`, with ARIA live regions for verdicts.

## Challenges we ran into

- **The vocabulary of prompts is the real model.** "A plastic container" gets confused with "a
  styrofoam container"; separating coated paper cups from plain paper cups needed prompts that
  describe the *object*, not the material.
- **Honest uncertainty is hard to design.** A recycling app that is confidently wrong is worse
  than no app, so the gate had to be tuned and then surfaced in the UI as a first-class state
  rather than an error message.
- **Mobile first, offline first.** Loading a quantised CLIP model is a 60–120MB download on the
  first visit. Caching plus a visible model status line turned a scary wait into something a
  user can reason about.

## What we learned

Recycling guidance is a *rules problem wearing an ML costume*. The classifier only has to
narrow 22 items down to one; everything the user needs — rinse, remove the cap, flatten, do not
put this in paper — is in the rules layer. That split is what makes the tool portable to other
districts and other countries.

## What's next

- Validate the refusal gate and the high-risk routing with a district partner and publish the
  three numbers above with the model card.
- Add a camera-first capture flow with a framing guide, and barcode lookup as a second signal
  for packaged goods.
- Publish a rule schema so district offices and community translators can contribute their own
  `rules/*.json`, with a diff view showing what changed between published versions.
- On-device fine-tuning of the prompt set on a small labelled set of local packaging.

## Built with

`transformers.js` · `CLIP ViT-B/32` · `WebGPU` · `WebAssembly` · `JavaScript` · `HTML` ·
`CSS` · `JSON rulesets` · `GitHub Pages`
