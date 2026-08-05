# Listening Forward web update v2

Replace these three files in your existing `docs/` folder:

- `index.html`
- `styles.css`
- `audio-engine.js`

Keep your existing:

- `docs/app.js`
- `docs/assets/`
- `docs/audio/`
- `docs/data/`

Changes:

1. The rotary volume knob now has visible **MIN** and **MAX** labels.
2. **Biodiversity** and **Habitation** headings use the same bold weight as
   **Repair** and **Automation**.
3. Active continuous layers react to the dial immediately.
4. Fades are shorter and layer replacement begins sooner.
5. Weakly relevant layers can fade to silence instead of remaining at a fixed
   minimum level, reducing long water/wind overlap across orientations.
