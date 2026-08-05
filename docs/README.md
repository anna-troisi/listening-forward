# Listening Forward — physical-look web shell

This folder is ready to replace the contents of `docs/`.

## What works now

- The physical retro-instrument image remains the visual shell.
- The central knob can be dragged with mouse or touch.
- Arrow keys adjust the knob; Shift+arrow gives fine control.
- The live readout shows the blend between adjacent orientations.
- The volume control remains in the lower-left position.
- The left explanation and right monitor space remain part of the reference image.

## What is deliberately not connected yet

- Audio playback
- Active waveform
- Layer monitors
- Scenario visualisation

The JavaScript emits two custom browser events that the future audio engine can use:

- `listeningforward:tune`
- `listeningforward:volume`

## Version 2 adjustment

The horizontal volume slider has been replaced by a rotary knob aligned with the physical volume control in the reference image.

The knob supports pointer/touch rotation, the mouse wheel, arrow keys, Shift+arrow fine adjustment, and Home/End. It emits the existing `listeningforward:volume` event for the future Web Audio engine.


## Version 3 wording update

The left-hand copy is now editable HTML rather than fixed image text.

Current wording:
- describes a growing environmental field-recording practice;
- retains Mtwapa Creek as an important source without implying that the entire library comes from one location;
- introduces the listening action through emergence, recession, overlap and return.

Future wording changes can now be made directly in `index.html` without editing the radio image.


## Version 4 visual wording update

The interface now remains geographically open:

- the left-hand copy no longer names a specific field-recording location;
- the bottom-right wording is replaced by editable HTML;
- the new bottom-right statement explains the relation between human annotation,
  machine-derived acoustic similarity and the way sounds enter, persist and recede.

Current editable wording:

> This instrument recomposes environmental field recordings through human interpretation and machine listening.

> Move the dial slowly to tune between possible sonic futures. Listen for sounds that emerge, recede, overlap and return.

> Human annotation and machine-derived acoustic similarity shape how recordings enter, persist and recede.


## Version 5 visual integration

The old fixed body and bottom-right copy have been removed from the radio image itself. The revised HTML text now sits directly on the paper surface with a transparent background, so there are no tinted rectangles or visible text panels.

The original title remains part of the physical concept image.
