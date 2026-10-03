# Third-party notices

The harness's own code is MIT (see `LICENSE`). These bundled or required components keep their own licences.

| Component | Where | Licence |
| --- | --- | --- |
| GSAP (GreenSock) `gsap.min.js` | `packs/_core/scene-template/assets/vendor/`, copied into each run's `scenes/assets/vendor/` | GreenSock's standard "no charge" licence, <https://gsap.com/standard-license>; not MIT, read it before you ship |
| Inter | `packs/_core/scene-template/assets/fonts/`, example runs | SIL Open Font License 1.1 (`OFL.txt` alongside) |
| Instrument Serif, JetBrains Mono | `examples/promo/harness-launch/scenes/assets/fonts/` | SIL Open Font License 1.1 (`OFL-*.txt` alongside) |
| HyperFrames 0.8.98 | fetched by `npx hyperframes@0.8.98` at run time, not bundled | Apache-2.0, <https://github.com/heygen-com/hyperframes> |
| Supertonic, parakeet-mlx, mlx-whisper models | downloaded by their own tools on first use, not bundled | see each project |

Music, voices and sound effects you generate with Gemini, Lyria or ElevenLabs are subject to those providers' terms.
The frames in `examples/promo/harness-launch/scenes/assets/frames/` are stills from the Oleafly tour film by the same
author.
