# Listening Forward web audio fix v4

Replace only `docs/audio-engine.js`.

This version makes the transition from Habitation to Automation more explicit.

Changes:
- generic crowds and bells no longer count as sufficient Automation material;
- crowd-dominant beds fade quickly above 55% Automation;
- machine-related beds are actively introduced;
- two machine-related beds are sought above 78% Automation when available;
- machine events such as machines-at-work and the electrical sander enter promptly and at a stronger level;
- generic Habitation events are suspended while Automation dominates.

No new audio export is required.
