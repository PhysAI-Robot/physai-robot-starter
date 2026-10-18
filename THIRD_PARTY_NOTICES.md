# Third-Party Notices

This project downloads some robot descriptions and meshes
at runtime. They are intentionally excluded from Git and are not covered by
the project license in `LICENSE`. Each artifact remains subject to its own
upstream terms.

## Robot assets

### SO-101

- Source: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- Downloaded path: `Simulation/SO101`
- Revision used by `scripts/fetch_assets.py`: `5f6d2b876a53a4872e405b991dd925556c9e38a4`
  (pinned; was tracking `main` until 2026-09-27)
- Upstream license: Apache-2.0
- Note: Apache-2.0 attribution and license requirements apply if these files
  are redistributed. The downloader does not copy the upstream `LICENSE` file
  into `assets/`.

### TurtleBot4

- Source: [narcispr/turtlebot4_mujoco](https://github.com/narcispr/turtlebot4_mujoco)
- Downloaded files: `turtlebot4.xml` and selected files under `assets/meshes`
- Revision used by `scripts/fetch_assets.py`: `e5d772caf426179b3d93eb91741965529239ae80`
  (pinned; was tracking `main` until 2026-09-27)
- Upstream license: no `LICENSE` file was found in the repository at the time
  this notice was written.
- Redistribution status: unresolved. Do not redistribute these downloaded
  files until the upstream license and the referenced original model source
  terms are confirmed.

### ViperX 300 S

- Source: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- Downloaded path: `trossen_vx300s` (`vx300s.xml`, the meshes under `assets/`, and `LICENSE`)
- Revision used by `scripts/fetch_assets.py`: `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`
  (pinned)
- Upstream license: BSD-3-Clause, Copyright (c) 2023 Trossen Robotics. The
  downloader copies the upstream `LICENSE` next to the model; keep it if the files
  are redistributed.
- Note: the MJCF is derived by Menagerie from the Interbotix `vx300s.urdf.xacro`.
  It is the single ViperX 300 S arm with its stock gripper, not the ALOHA 2
  follower's custom gripper.

## Model checkpoints

Checkpoints (for example an ACT policy you trained yourself, or a
Hugging Face model you download yourself) are never committed and carry their own licenses.
Check the license of every repository and revision before redistribution.

## Dependencies

Python packages installed from the project lockfile and optional dependencies
retain their own licenses. This notice does not replace the license notices
provided by those packages.

## Scope

This file records the source and license status known for the downloader
defaults. Re-check upstream terms when changing a repository, revision, or
redistribution method. This is project documentation, not legal advice.