# massage
Multitrack Analysis/SynthesiS for Annotation, auGmentation and Evaluation


[![Build Status](https://travis-ci.org/marl/massage.svg?branch=master)](https://travis-ci.org/marl/massage)

[![Coverage Status](https://coveralls.io/repos/github/marl/massage/badge.svg?branch=master)](https://coveralls.io/github/marl/massage?branch=master)

[![Documentation Status](https://readthedocs.org/projects/massage/badge/?version=latest)](http://massage.readthedocs.io/en/latest/?badge=latest)



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
