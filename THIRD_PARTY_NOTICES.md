# Third-Party Notices

This project downloads some robot descriptions, meshes, and model checkpoints
at runtime. They are intentionally excluded from Git and are not covered by
the project license in `LICENSE`. Each artifact remains subject to its own
upstream terms.

## Robot assets

### SO-101

- Source: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- Downloaded path: `Simulation/SO101`
- Revision used by `scripts/fetch_assets.py`: `main`
- Upstream license: Apache-2.0
- Note: Apache-2.0 attribution and license requirements apply if these files
  are redistributed. The downloader does not copy the upstream `LICENSE` file
  into `assets/`.

### TurtleBot4

- Source: [narcispr/turtlebot4_mujoco](https://github.com/narcispr/turtlebot4_mujoco)
- Downloaded files: `turtlebot4.xml` and selected files under `assets/meshes`
- Revision used by `scripts/fetch_assets.py`: `main`
- Upstream license: no `LICENSE` file was found in the repository at the time
  this notice was written.
- Redistribution status: unresolved. Do not redistribute these downloaded
  files until the upstream license and the referenced original model source
  terms are confirmed.

## Model checkpoints

The model downloader accepts arbitrary Hugging Face repositories with
`--repo`. The license must therefore be checked for every repository and
revision before redistribution.

### SmolVLM

- Repository: [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- License shown by the model card: Apache-2.0
- Additional upstream components and training data may have their own terms.

### SmolVLA

- Repository: [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- License: no explicit license was present in the model metadata when checked.
- Redistribution status: unresolved. Treat the checkpoint as local-use-only
  until the model author confirms its terms.

### TurboVLA

- Repository: [H-EmbodVis/TurboVLA](https://huggingface.co/H-EmbodVis/TurboVLA)
- Project code: Apache-2.0
- Checkpoint parameters: subject to the included
  [DINOv3 License](https://huggingface.co/H-EmbodVis/TurboVLA/blob/main/DINOv3_LICENSE.md)
- Redistribution status: do not apply this project's Apache-2.0 license to the
  checkpoint files. Follow the model card and DINOv3 terms.

## Dependencies

Python packages installed from `requirements.txt` and optional dependencies
retain their own licenses. This notice does not replace the license notices
provided by those packages.

## Scope

This file records the source and license status known for the downloader
defaults. Re-check upstream terms when changing a repository, revision, or
redistribution method. This is project documentation, not legal advice.