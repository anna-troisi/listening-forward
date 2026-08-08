# Listening Forward — integrated room scene full website

This is the complete website code and visual asset update.

## What is fixed

- The blurred room/table background is baked directly into `assets/radio-reference.png`.
- There is no separate studio backdrop image or CSS background.
- The main tuning knob is the working live control, correctly sized so the coloured orientation ring remains visible.
- The old main knob face is removed beneath it while the original fixed shadow remains on the radio image.
- The static volume knob is removed from the radio image.
- The live volume knob is centred precisely on the printed MIN/MAX scale.
- The original volume shadow remains fixed while only the knob face rotates.
- The waveform has stronger adaptive visual gain and is aligned to the black display.
- Repair, Automation, Habitation and Biodiversity remain connected to the existing audio engine.

## Install

Replace in `docs/`:

- `index.html`
- `styles.css`
- `app.js`
- `audio-engine.js`

Replace in `docs/assets/`:

- `radio-reference.png`
- `main-dial-face.png`
- `volume-dial-face.png`

Remove the obsolete file if it remains:

- `docs/assets/studio-backdrop.jpg`

Keep unchanged:

- `docs/audio/`
- `docs/data/`
