# massage
Multitrack Analysis/SynthesiS for Annotation, auGmentation and Evaluation



Pitch
============================
Submodule containing different pitch trackers.

Each pitch tracker inputs audio and outputs arrays of times and frequencies:

audio ---> times, frequencies


Resynth
============================
Submodule containing different analysis - synthesis algorithms.

Each algorithm inputs audio (and optionally f0) and outputs resynthesized audio
and a [jams](http://github.com/marl/jams) file:

audio, [f0] --> resynthesized audio, jams

Remix
============================
Submodule containing different multitrack remix formulas.

Each formula inputs a [medleydb](http://github.com/marl/medleydb) multitrack
and outputs a mix and corresponding jams file

multitrack --> mix, jams
