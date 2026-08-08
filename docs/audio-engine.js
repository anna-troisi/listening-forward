(() => {
  "use strict";

  const MANIFEST_URL = "data/sounds.json";
  const ORIENTATIONS = ["repair", "automation", "habitation", "biodiversity"];

  const BED_LAYER_COUNT = 2;
  const RECONCILE_DELAY_MS = 260;
  const GAIN_RESPONSE_SECONDS = 0.9;
  const BED_FADE_SECONDS = 1.8;
  const EVENT_RICH_CROSSFADE_SECONDS = 2.2;
  const REPLACEMENT_MARGIN = 0.025;
  const MIN_KEEP_SCORE = 0.18;
  const EVENT_MIN_SCORE = 0.38;
  const EVENT_GAP_MS = 6500;
  const AUTOMATION_ENTER_THRESHOLD = 0.55;
  const AUTOMATION_DOMINANT_THRESHOLD = 0.78;
  const AUTOMATION_CROWD_FADE_SECONDS = 0.75;
  const AUTOMATION_MACHINE_FADE_SECONDS = 1.05;
  const AUTOMATION_EVENT_GAP_MS = 2600;

  const ROLE_GAINS = {
    base: 0.34,
    texture: 0.24,
    foreground: 0.17,
    accent: 0.12
  };

  const STRONG_MACHINE_CUES = new Set([
    "machine", "machines", "machinery", "mechanical",
    "saw", "sander", "tools", "construction", "industrial"
  ]);

  const SECONDARY_MACHINE_CUES = new Set([
    "engine", "engines", "traffic", "car", "cars",
    "helicopter", "boat", "boats", "foghorn", "signal", "signals"
  ]);

  const CROWD_CUES = new Set([
    "people", "person", "voices", "voice", "crowd", "crawd",
    "footsteps", "steps", "walkers", "walking", "children", "kids",
    "music", "street music", "song", "songs"
  ]);

  const HABITATION_CUES = new Set([
    "people", "person", "voices", "voice", "children", "kids",
    "footsteps", "walkers", "walking", "dogs", "dog", "crowd",
    "crawd", "music", "song", "songs", "work", "workers",
    "construction", "boat", "boats", "engine", "engines", "swimming"
  ]);

  const BIODIVERSITY_CUES = new Set([
    "bird", "birds", "birdsong", "monkey", "monkeys", "insects",
    "water", "waves", "wind", "crabs", "vegetation"
  ]);

  const NATURAL_ONLY_TERMS = new Set([
    "water", "waves", "rain", "wind", "bird", "birds", "birdsong"
  ]);

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function mulberry32(seed) {
    return function random() {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hasAny(values, vocabulary) {
    return values.some(value => vocabulary.has(String(value).toLowerCase()));
  }

  function isNaturalOnly(elements) {
    return elements.length > 0
      && elements.every(value => NATURAL_ONLY_TERMS.has(String(value).toLowerCase()));
  }

  class ListeningForwardAudioEngine {
    constructor() {
      this.context = null;
      this.masterGain = null;
      this.analyser = null;
      this.waveformCanvas = document.getElementById("soundscapeWaveform");
      this.waveformContext = this.waveformCanvas?.getContext("2d") || null;
      this.waveformData = null;
      this.waveformFrame = null;
      this.waveformVisualGain = 3.6;
      this.manifest = null;
      this.soundsById = new Map();
      this.bufferCache = new Map();

      // A "bed" may now be either:
      // - continuous: stable looping ambience;
      // - event_rich base/texture: rotating non-identical excerpts.
      this.activeBeds = new Map();

      this.activeEvents = new Set();
      this.cooldownUntil = new Map();

      this.weights = {
        repair: 1,
        automation: 0,
        habitation: 0,
        biodiversity: 0
      };

      this.volume = 0.72;
      this.started = false;
      this.starting = false;
      this.reconcileTimer = null;
      this.followUpTimer = null;
      this.automationEventTimer = null;
      this.lastEventAt = 0;
      this.lastAutomationEventAt = 0;

      this.sessionSeed =
        (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0;
      this.random = mulberry32(this.sessionSeed);

      this.statusElement = document.getElementById("audioStatus");
      this.prepareWaveformCanvas();
      this.drawWaveform();
      this.manifestPromise = this.loadManifest();

      window.addEventListener("listeningforward:start", () => this.start());
      window.addEventListener("listeningforward:tune", event => {
        this.setWeights(event.detail?.orientations || {});
      });
      window.addEventListener("listeningforward:volume", event => {
        this.setVolume(Number(event.detail?.value ?? this.volume));
      });
    }


    prepareWaveformCanvas() {
      if (!this.waveformCanvas || !this.waveformContext) return;

      const rect = this.waveformCanvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(rect.width * ratio));
      const height = Math.max(1, Math.round(rect.height * ratio));

      if (
        this.waveformCanvas.width !== width
        || this.waveformCanvas.height !== height
      ) {
        this.waveformCanvas.width = width;
        this.waveformCanvas.height = height;
      }
    }

    drawWaveform() {
      if (!this.waveformCanvas || !this.waveformContext) return;

      this.prepareWaveformCanvas();

      const canvas = this.waveformCanvas;
      const context = this.waveformContext;
      const width = canvas.width;
      const height = canvas.height;
      const centre = height / 2;

      context.clearRect(0, 0, width, height);

      const background = context.createLinearGradient(0, 0, 0, height);
      background.addColorStop(0, "rgba(5, 8, 10, 0.98)");
      background.addColorStop(0.5, "rgba(11, 16, 19, 0.99)");
      background.addColorStop(1, "rgba(3, 5, 6, 0.99)");
      context.fillStyle = background;
      context.fillRect(0, 0, width, height);

      // A barely visible centre line gives the display a calm instrument feel.
      context.beginPath();
      context.moveTo(0, centre);
      context.lineTo(width, centre);
      context.strokeStyle = "rgba(166, 188, 200, 0.10)";
      context.lineWidth = Math.max(1, width / 900);
      context.stroke();

      context.beginPath();

      if (this.analyser && this.waveformData) {
        this.analyser.getByteTimeDomainData(this.waveformData);

        const slice = width / Math.max(1, this.waveformData.length - 1);

        let peak = 0;

        for (let index = 0; index < this.waveformData.length; index += 1) {
          const value = Math.abs((this.waveformData[index] - 128) / 128);
          if (value > peak) peak = value;
        }

        const targetVisualGain = Math.min(
          14.0,
          Math.max(3.0, 0.78 / Math.max(peak, 0.025))
        );

        this.waveformVisualGain +=
          (targetVisualGain - this.waveformVisualGain) * 0.14;

        for (let index = 0; index < this.waveformData.length; index += 1) {
          const normalised = (this.waveformData[index] - 128) / 128;
          const amplified = Math.max(
            -0.92,
            Math.min(0.92, normalised * this.waveformVisualGain)
          );
          const x = index * slice;
          const y = centre + amplified * height * 0.485;

          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
      } else {
        context.moveTo(0, centre);
        context.lineTo(width, centre);
      }

      const glow = context.createLinearGradient(0, 0, width, 0);
      glow.addColorStop(0, "rgba(178, 207, 220, 0.48)");
      glow.addColorStop(0.5, "rgba(222, 237, 244, 0.92)");
      glow.addColorStop(1, "rgba(178, 207, 220, 0.48)");

      context.strokeStyle = glow;
      context.lineWidth = Math.max(1.9, width / 430);
      context.shadowColor = "rgba(191, 220, 233, 0.42)";
      context.shadowBlur = Math.max(6, width / 68);
      context.stroke();
      context.shadowBlur = 0;

      this.waveformFrame = window.requestAnimationFrame(
        () => this.drawWaveform()
      );
    }

    setStatus(message, state = "") {
      if (!this.statusElement) return;
      this.statusElement.textContent = message;
      this.statusElement.dataset.state = state;
    }

    async loadManifest() {
      try {
        const response = await fetch(MANIFEST_URL, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Could not load ${MANIFEST_URL} (${response.status})`);
        }

        const manifest = await response.json();
        if (!Array.isArray(manifest.sounds) || manifest.sounds.length === 0) {
          throw new Error(
            "The audio manifest contains no sounds. Rerun export_web_package.py."
          );
        }

        this.manifest = manifest;
        for (const sound of manifest.sounds) {
          this.soundsById.set(sound.id, sound);
        }

        this.setStatus("turn dial to begin", "ready");
        return manifest;
      } catch (error) {
        console.error(error);
        this.setStatus("audio package not exported", "error");
        throw error;
      }
    }

    async start() {
      if (this.started || this.starting) {
        if (this.context?.state === "suspended") {
          await this.context.resume();
        }
        return;
      }

      this.starting = true;

      try {
        await this.manifestPromise;

        const AudioContextClass =
          window.AudioContext || window.webkitAudioContext;

        if (!AudioContextClass) {
          throw new Error("This browser does not support the Web Audio API.");
        }

        this.context = new AudioContextClass();
        this.masterGain = this.context.createGain();
        this.masterGain.gain.value = this.volume;

        this.analyser = this.context.createAnalyser();
        this.analyser.fftSize = 2048;
        this.analyser.smoothingTimeConstant = 0.72;
        this.waveformData = new Uint8Array(this.analyser.fftSize);

        this.masterGain.connect(this.analyser);
        this.analyser.connect(this.context.destination);

        if (this.context.state === "suspended") {
          await this.context.resume();
        }

        this.started = true;
        this.setStatus("listening", "playing");
        await this.reconcile(true);
      } catch (error) {
        console.error(error);
        this.setStatus("audio unavailable", "error");
      } finally {
        this.starting = false;
      }
    }

    setWeights(incoming) {
      const previousAutomation = this.weights.automation || 0;

      const next = Object.fromEntries(
        ORIENTATIONS.map(name => [name, 0])
      );

      for (const name of ORIENTATIONS) {
        next[name] = clamp(Number(incoming[name] || 0), 0, 1);
      }

      const total = Object.values(next)
        .reduce((sum, value) => sum + value, 0);

      if (total > 0) {
        for (const name of ORIENTATIONS) {
          next[name] /= total;
        }
      }

      this.weights = next;

      if (this.started) {
        this.updateBedGains(GAIN_RESPONSE_SECONDS);

        if (
          next.automation >= AUTOMATION_ENTER_THRESHOLD
          && previousAutomation < AUTOMATION_ENTER_THRESHOLD
        ) {
          this.scheduleAutomationEvent(220);
        } else if (next.automation >= AUTOMATION_ENTER_THRESHOLD) {
          this.scheduleAutomationEvent(1200);
        } else {
          clearTimeout(this.automationEventTimer);
        }
      }

      clearTimeout(this.reconcileTimer);
      this.reconcileTimer = window.setTimeout(() => {
        if (this.started) this.reconcile(false);
      }, RECONCILE_DELAY_MS);
    }

    setVolume(value) {
      this.volume = clamp(value, 0, 1);

      if (!this.masterGain || !this.context) return;

      const now = this.context.currentTime;
      this.masterGain.gain.cancelScheduledValues(now);
      this.masterGain.gain.setTargetAtTime(this.volume, now, 0.06);
    }

    machineStrength(sound) {
      const elements = (sound.elements || [])
        .map(value => String(value).toLowerCase());
      const category = String(sound.category || "").toLowerCase();
      const environment = String(sound.environment || "").toLowerCase();

      let strength = 0;

      if (category === "machines") strength = 1;
      if (hasAny(elements, STRONG_MACHINE_CUES)) {
        strength = Math.max(strength, 0.95);
      }
      if (hasAny(elements, SECONDARY_MACHINE_CUES)) {
        strength = Math.max(strength, 0.58);
      }
      if (environment === "infrastructural" && strength > 0) {
        strength += 0.08;
      }

      return clamp(strength, 0, 1);
    }

    crowdStrength(sound) {
      const elements = (sound.elements || [])
        .map(value => String(value).toLowerCase());
      const category = String(sound.category || "").toLowerCase();
      const humanPresence =
        String(sound.humanPresence || "").toLowerCase();

      let strength = 0;

      if (hasAny(elements, CROWD_CUES)) strength = 0.9;
      if (category === "urban") strength = Math.max(strength, 0.35);
      if (humanPresence === "direct") strength = Math.max(strength, 0.55);
      if (humanPresence === "indirect") strength = Math.max(strength, 0.25);

      return clamp(strength, 0, 1);
    }

    machineDominance(sound) {
      return this.machineStrength(sound) - this.crowdStrength(sound);
    }

    scoreForOrientation(sound, orientation) {
      let score = clamp(
        Number(sound.scores?.[orientation] || 0),
        0,
        1
      );

      const elements = (sound.elements || [])
        .map(value => String(value).toLowerCase());

      const category = String(sound.category || "").toLowerCase();
      const environment = String(sound.environment || "").toLowerCase();
      const humanPresence =
        String(sound.humanPresence || "").toLowerCase();

      if (orientation === "automation") {
        const machine = this.machineStrength(sound);
        const crowd = this.crowdStrength(sound);

        if (machine >= 0.85) {
          score = clamp(score * 1.12 + 0.16, 0, 1);
        } else if (machine >= 0.45) {
          score = clamp(score * 0.86 + 0.07, 0, 1);
        } else {
          score *= 0.08;
        }

        // Bells, voices and crowds remain Habitation material unless
        // a machine presence is genuinely dominant.
        if (crowd > machine) score *= 0.20;
        if (isNaturalOnly(elements)) score *= 0.10;
      }

      if (orientation === "habitation") {
        const hasHabitationCue =
          hasAny(elements, HABITATION_CUES)
          || humanPresence === "direct"
          || humanPresence === "indirect"
          || category === "urban";

        if (!hasHabitationCue) score *= 0.24;
      }

      if (orientation === "biodiversity") {
        const hasBiodiversityCue =
          hasAny(elements, BIODIVERSITY_CUES)
          || category === "birds"
          || category === "water"
          || environment === "coastal";

        if (!hasBiodiversityCue) score *= 0.28;
      }

      return score;
    }

    orientationScore(sound) {
      return ORIENTATIONS.reduce(
        (sum, name) =>
          sum
          + (this.weights[name] || 0)
          * this.scoreForOrientation(sound, name),
        0
      );
    }

    similarity(sound, anchorId) {
      if (!anchorId || sound.id === anchorId) return 1;

      const item = (sound.neighbours || [])
        .find(neighbour => neighbour.id === anchorId);

      return item
        ? clamp(Number(item.similarity), -1, 1)
        : 0;
    }

    isBedEligible(sound) {
      if (sound.playbackCharacter === "continuous") return true;

      return sound.playbackCharacter === "event_rich"
        && (sound.role === "base" || sound.role === "texture");
    }

    rankedBeds() {
      const activeIds = new Set(this.activeBeds.keys());
      const currentAnchor =
        this.activeBeds.keys().next().value || null;
      const automationWeight = this.weights.automation || 0;

      return this.manifest.sounds
        .filter(sound => this.isBedEligible(sound))
        .map(sound => {
          let score = this.orientationScore(sound);

          if (currentAnchor) {
            score +=
              Math.max(0, this.similarity(sound, currentAnchor)) * 0.025;
          }

          if (activeIds.has(sound.id)) score += 0.015;
          if (sound.role === "base" || sound.role === "texture") {
            score += 0.025;
          }

          if (automationWeight >= AUTOMATION_ENTER_THRESHOLD) {
            const machine = this.machineStrength(sound);
            const crowd = this.crowdStrength(sound);

            score += machine * automationWeight * 0.34;

            if (crowd > machine) {
              score -= crowd * automationWeight * 0.42;
            }
          }

          return { sound, score };
        })
        .sort((left, right) => right.score - left.score);
    }

    desiredGain(sound) {
      const roleGain =
        ROLE_GAINS[sound.role] ?? ROLE_GAINS.texture;

      const relevance = clamp(this.orientationScore(sound), 0, 1);

      if (relevance <= 0.12) return 0;

      const normalised = (relevance - 0.12) / 0.88;
      let gain = roleGain * Math.pow(normalised, 1.15);

      const automationWeight = this.weights.automation || 0;

      if (automationWeight >= AUTOMATION_ENTER_THRESHOLD) {
        const machine = this.machineStrength(sound);
        const crowd = this.crowdStrength(sound);

        if (crowd > machine) {
          gain *= Math.max(0.025, 1 - automationWeight * 1.08);
        }

        if (machine >= 0.45) {
          gain *= 1 + automationWeight * 0.62;
        }
      }

      return Math.min(gain, roleGain * 1.48);
    }

    updateBedGains(seconds = GAIN_RESPONSE_SECONDS) {
      if (!this.context) return;

      for (const layer of this.activeBeds.values()) {
        this.rampGain(
          layer.gain,
          this.desiredGain(layer.sound),
          seconds
        );
      }
    }

    async ensureAutomationBeds(ranked) {
      const automationWeight = this.weights.automation || 0;

      const desiredMachineBeds =
        automationWeight >= AUTOMATION_DOMINANT_THRESHOLD ? 2 : 1;

      const activeMachineBeds = [...this.activeBeds.values()]
        .filter(layer => this.machineStrength(layer.sound) >= 0.45);

      if (activeMachineBeds.length >= desiredMachineBeds) {
        return false;
      }

      const candidate = ranked.find(item =>
        !this.activeBeds.has(item.sound.id)
        && this.machineStrength(item.sound) >= 0.45
      );

      if (!candidate) return false;

      const activeLayers = [...this.activeBeds.values()];

      const crowdLayer = activeLayers
        .filter(layer =>
          this.crowdStrength(layer.sound)
          > this.machineStrength(layer.sound)
        )
        .sort(
          (left, right) =>
            this.crowdStrength(right.sound)
            - this.crowdStrength(left.sound)
        )[0];

      const weakestLayer = activeLayers
        .sort(
          (left, right) =>
            this.orientationScore(left.sound)
            - this.orientationScore(right.sound)
        )[0];

      const outgoing = crowdLayer || weakestLayer;

      if (outgoing) {
        this.fadeOutBed(
          outgoing.sound.id,
          AUTOMATION_CROWD_FADE_SECONDS
        );
      }

      await this.startBed(
        candidate.sound,
        Math.max(
          this.desiredGain(candidate.sound),
          (ROLE_GAINS[candidate.sound.role] ?? ROLE_GAINS.texture) * 0.48
        ),
        AUTOMATION_MACHINE_FADE_SECONDS
      );

      return true;
    }

    async reconcile(initial = false) {
      if (!this.started || !this.context || !this.manifest) return;

      const ranked = this.rankedBeds();

      if (
        !initial
        && this.weights.automation >= AUTOMATION_ENTER_THRESHOLD
      ) {
        const changed = await this.ensureAutomationBeds(ranked);

        if (changed) {
          clearTimeout(this.followUpTimer);
          this.followUpTimer = window.setTimeout(
            () => this.reconcile(false),
            1050
          );
        }
      }

      if (ranked.length === 0) {
        this.setStatus("no suitable sound beds available", "error");
        return;
      }

      if (initial || this.activeBeds.size === 0) {
        const initialItems = ranked.slice(0, BED_LAYER_COUNT);
        await Promise.all(
          initialItems.map(item =>
            this.startBed(
              item.sound,
              this.desiredGain(item.sound)
            )
          )
        );
      } else {
        const activeRanked = ranked
          .filter(item => this.activeBeds.has(item.sound.id));

        const inactiveRanked = ranked
          .filter(item => !this.activeBeds.has(item.sound.id));

        const weakestActive = [...activeRanked]
          .sort((left, right) => left.score - right.score)[0];

        const strongestInactive = inactiveRanked[0];

        const shouldReplace =
          weakestActive
          && strongestInactive
          && (
            weakestActive.score < MIN_KEEP_SCORE
            || strongestInactive.score
              > weakestActive.score + REPLACEMENT_MARGIN
          );

        if (shouldReplace) {
          this.fadeOutBed(weakestActive.sound.id);
          await this.startBed(
            strongestInactive.sound,
            this.desiredGain(strongestInactive.sound)
          );

          clearTimeout(this.followUpTimer);
          this.followUpTimer = window.setTimeout(
            () => this.reconcile(false),
            1700
          );
        }
      }

      this.updateBedGains(1.1);
      await this.maybeTriggerEvent();
    }

    async loadBuffer(path) {
      if (this.bufferCache.has(path)) {
        return this.bufferCache.get(path);
      }

      const promise = fetch(path)
        .then(response => {
          if (!response.ok) {
            throw new Error(`Could not load audio file: ${path}`);
          }
          return response.arrayBuffer();
        })
        .then(arrayBuffer =>
          this.context.decodeAudioData(arrayBuffer)
        );

      this.bufferCache.set(path, promise);

      try {
        return await promise;
      } catch (error) {
        this.bufferCache.delete(path);
        throw error;
      }
    }

    chooseFile(sound, excludedPath = null) {
      const files = sound.files || [];
      if (files.length === 0) return null;

      const alternatives = excludedPath && files.length > 1
        ? files.filter(file => file.path !== excludedPath)
        : files;

      const pool = alternatives.length > 0
        ? alternatives
        : files;

      const index = Math.floor(this.random() * pool.length);
      return pool[index];
    }

    async startBed(sound, targetGain, fadeSeconds = BED_FADE_SECONDS) {
      if (this.activeBeds.has(sound.id)) return;

      const gain = this.context.createGain();
      gain.gain.value = 0;
      gain.connect(this.masterGain);

      const layer = {
        sound,
        gain,
        character: sound.playbackCharacter,
        sources: new Set(),
        timer: null,
        stopped: false,
        lastFilePath: null,
        buffers: []
      };

      this.activeBeds.set(sound.id, layer);

      try {
        if (sound.playbackCharacter === "continuous") {
          await this.startContinuousBed(layer);
        } else {
          await this.startEventRichBed(layer);
        }

        this.rampGain(gain, targetGain, fadeSeconds);
      } catch (error) {
        console.error(error);
        this.activeBeds.delete(sound.id);
        try { gain.disconnect(); } catch (_) { /* already disconnected */ }
      }
    }

    async startContinuousBed(layer) {
      const file = layer.sound.files?.[0];
      if (!file) return;

      const buffer = await this.loadBuffer(file.path);
      const source = this.context.createBufferSource();

      source.buffer = buffer;
      source.loop = true;
      source.connect(layer.gain);

      const offset = buffer.duration > 2
        ? this.random() * Math.min(buffer.duration - 1, 12)
        : 0;

      source.start(0, offset);
      layer.sources.add(source);

      source.addEventListener("ended", () => {
        layer.sources.delete(source);
        try { source.disconnect(); } catch (_) { /* already disconnected */ }
      });
    }

    async startEventRichBed(layer) {
      const files = layer.sound.files || [];
      if (files.length === 0) return;

      layer.buffers = await Promise.all(
        files.map(async file => ({
          file,
          buffer: await this.loadBuffer(file.path)
        }))
      );

      this.scheduleEventRichSegment(
        layer,
        this.context.currentTime + 0.04
      );
    }

    scheduleEventRichSegment(layer, startAt) {
      if (
        layer.stopped
        || !this.activeBeds.has(layer.sound.id)
        || layer.buffers.length === 0
      ) {
        return;
      }

      const choices = layer.lastFilePath && layer.buffers.length > 1
        ? layer.buffers.filter(
            item => item.file.path !== layer.lastFilePath
          )
        : layer.buffers;

      const item =
        choices[Math.floor(this.random() * choices.length)];

      layer.lastFilePath = item.file.path;

      const source = this.context.createBufferSource();
      const segmentGain = this.context.createGain();
      const buffer = item.buffer;

      source.buffer = buffer;
      segmentGain.gain.value = 0;

      source
        .connect(segmentGain)
        .connect(layer.gain);

      const fade = Math.min(
        EVENT_RICH_CROSSFADE_SECONDS,
        Math.max(0.6, buffer.duration * 0.22)
      );

      const endAt = startAt + buffer.duration;
      const fadeOutAt = Math.max(
        startAt + fade,
        endAt - fade
      );

      segmentGain.gain.setValueAtTime(0, startAt);
      segmentGain.gain.linearRampToValueAtTime(1, startAt + fade);
      segmentGain.gain.setValueAtTime(1, fadeOutAt);
      segmentGain.gain.linearRampToValueAtTime(0, endAt);

      source.start(startAt);
      source.stop(endAt + 0.05);
      layer.sources.add(source);

      source.addEventListener("ended", () => {
        layer.sources.delete(source);
        try { source.disconnect(); } catch (_) { /* already disconnected */ }
        try { segmentGain.disconnect(); } catch (_) { /* already disconnected */ }
      });

      const nextStartAt = endAt - fade;
      const schedulingLead = 0.35;
      const delayMs = Math.max(
        0,
        (nextStartAt - this.context.currentTime - schedulingLead) * 1000
      );

      clearTimeout(layer.timer);
      layer.timer = window.setTimeout(() => {
        this.scheduleEventRichSegment(layer, nextStartAt);
      }, delayMs);
    }

    fadeOutBed(id, fadeSeconds = BED_FADE_SECONDS) {
      const layer = this.activeBeds.get(id);
      if (!layer) return;

      this.activeBeds.delete(id);
      layer.stopped = true;
      clearTimeout(layer.timer);

      this.rampGain(layer.gain, 0, fadeSeconds);

      window.setTimeout(() => {
        for (const source of layer.sources) {
          try { source.stop(); } catch (_) { /* already stopped */ }
          try { source.disconnect(); } catch (_) { /* already disconnected */ }
        }

        layer.sources.clear();
        try { layer.gain.disconnect(); } catch (_) { /* already disconnected */ }
      }, (fadeSeconds + 0.35) * 1000);
    }

    rampGain(gainNode, target, seconds) {
      const now = this.context.currentTime;
      const parameter = gainNode.gain;

      parameter.cancelScheduledValues(now);
      parameter.setValueAtTime(parameter.value, now);
      parameter.linearRampToValueAtTime(
        Math.max(0, target),
        now + seconds
      );
    }

    scheduleAutomationEvent(delayMs = 500) {
      clearTimeout(this.automationEventTimer);

      if (
        !this.started
        || (this.weights.automation || 0) < AUTOMATION_ENTER_THRESHOLD
      ) {
        return;
      }

      this.automationEventTimer = window.setTimeout(
        () => this.triggerAutomationEvent(),
        delayMs
      );
    }

    async triggerAutomationEvent() {
      if (
        !this.started
        || (this.weights.automation || 0) < AUTOMATION_ENTER_THRESHOLD
      ) {
        return;
      }

      if (this.activeEvents.size > 0) {
        this.scheduleAutomationEvent(900);
        return;
      }

      const nowMs = Date.now();

      const candidates = this.manifest.sounds
        .filter(sound =>
          !this.activeBeds.has(sound.id)
          && this.machineStrength(sound) >= 0.45
          && (
            sound.playbackCharacter === "discrete_event"
            || sound.playbackCharacter === "event_rich"
          )
          && (this.cooldownUntil.get(sound.id) || 0) <= nowMs
        )
        .map(sound => ({
          sound,
          score:
            this.orientationScore(sound)
            + this.machineStrength(sound) * 0.72
            - this.crowdStrength(sound) * 0.28
        }))
        .sort((left, right) => right.score - left.score);

      if (candidates.length === 0) {
        this.scheduleAutomationEvent(1400);
        return;
      }

      const top = candidates.slice(0, Math.min(2, candidates.length));
      const chosen =
        top[Math.floor(this.random() * top.length)];

      const duration = await this.playEvent(
        chosen.sound,
        chosen.score,
        { gainMultiplier: 1.42 }
      );

      this.lastAutomationEventAt = Date.now();

      const nextDelay = Math.max(
        AUTOMATION_EVENT_GAP_MS,
        Math.round((duration || 2) * 1000 + 1200)
      );

      this.scheduleAutomationEvent(nextDelay);
    }

    async maybeTriggerEvent() {
      if (
        (this.weights.automation || 0) >= AUTOMATION_ENTER_THRESHOLD
      ) {
        this.scheduleAutomationEvent(320);
        return;
      }

      const nowMs = Date.now();

      if (
        this.activeEvents.size > 0
        || nowMs - this.lastEventAt < EVENT_GAP_MS
      ) {
        return;
      }

      const activeAnchor =
        this.activeBeds.keys().next().value || null;

      const candidates = this.manifest.sounds
        .filter(sound => {
          if (sound.playbackCharacter === "discrete_event") {
            return true;
          }

          // Event-rich foreground/accent recordings remain occasional events.
          return sound.playbackCharacter === "event_rich"
            && (sound.role === "foreground" || sound.role === "accent");
        })
        .filter(sound => !this.activeBeds.has(sound.id))
        .filter(sound =>
          (this.cooldownUntil.get(sound.id) || 0) <= nowMs
        )
        .map(sound => {
          let score = this.orientationScore(sound);

          if (activeAnchor) {
            score +=
              Math.max(0, this.similarity(sound, activeAnchor)) * 0.08;
          }

          return { sound, score };
        })
        .filter(item => item.score >= EVENT_MIN_SCORE)
        .sort((left, right) => right.score - left.score);

      if (candidates.length === 0) return;

      const top = candidates.slice(
        0,
        Math.min(3, candidates.length)
      );

      const chosen =
        top[Math.floor(this.random() * top.length)];

      await this.playEvent(chosen.sound, chosen.score);
    }

    async playEvent(sound, score, options = {}) {
      const file = this.chooseFile(sound);
      if (!file) return;

      try {
        const buffer = await this.loadBuffer(file.path);
        const source = this.context.createBufferSource();
        const gain = this.context.createGain();

        source.buffer = buffer;
        gain.gain.value = 0;

        source
          .connect(gain)
          .connect(this.masterGain);

        const roleGain =
          ROLE_GAINS[sound.role] ?? ROLE_GAINS.accent;

        const gainMultiplier =
          Number(options.gainMultiplier || 1);

        const target = Math.min(
          roleGain * 1.65,
          roleGain
            * (0.42 + clamp(score, 0, 1) * 0.58)
            * gainMultiplier
        );

        const now = this.context.currentTime;
        const discrete =
          sound.playbackCharacter === "discrete_event";

        const fadeIn = discrete ? 0.65 : 1.4;
        const fadeOut = discrete ? 1.1 : 1.8;

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(
          target,
          now + fadeIn
        );

        const fadeStart = Math.max(
          now + fadeIn,
          now + buffer.duration - fadeOut
        );

        gain.gain.setValueAtTime(target, fadeStart);
        gain.gain.linearRampToValueAtTime(
          0,
          now + buffer.duration
        );

        const token = `${sound.id}:${now}`;
        this.activeEvents.add(token);
        this.lastEventAt = Date.now();

        const cooldown = discrete ? 36000 : 18000;
        this.cooldownUntil.set(
          sound.id,
          Date.now() + cooldown
        );

        source.addEventListener("ended", () => {
          this.activeEvents.delete(token);
          try { source.disconnect(); } catch (_) { /* already disconnected */ }
          try { gain.disconnect(); } catch (_) { /* already disconnected */ }
        });

        source.start();
        return buffer.duration;
      } catch (error) {
        console.error(error);
        return 0;
      }
    }
  }

  window.listeningForwardAudio =
    new ListeningForwardAudioEngine();
})();
