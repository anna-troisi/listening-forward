# Listening Forward — working web audio setup

This package uses the four orientations already tested in the Python radio engine:

1. Repair
2. Automation
3. Habitation
4. Biodiversity

## Install into the project

- Copy the contents of `docs/` over the existing project `docs/` folder.
- Copy `export_web_package.py` into the project root, beside:
  - `generate_soundscape_v12.py`
  - `listening_forward_audio_v3.py`
  - the metadata workbook
  - `sound_embeddings.csv`
  - `sounds/`

## Export browser-ready audio

From the project root:

```bash
python3 export_web_package.py \
  --metadata Soundscape_Metadata2.xlsx \
  --embeddings sound_embeddings.csv \
  --sounds-dir sounds \
  --docs-dir docs \
  --clean
```

Use the actual workbook filename after `--metadata` if yours is different.

The command creates:

- `docs/data/sounds.json`
- `docs/audio/*.wav`

## Test locally

Do not double-click `index.html`. Start a local server:

```bash
cd docs
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Turn the large dial once to start audio. Stop the server with `Control+C`.

## GitHub

After local testing, commit the changed `docs/` folder and push it. GitHub Pages can then serve the same static audio instrument.
