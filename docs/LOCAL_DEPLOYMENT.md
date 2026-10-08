# Local deployment and migration

## Scope

The `HandPoseCollection` checkout contains the current collection and WiLoR reconstruction source. `pc_receiver/` retains the module layout needed by subprocesses and templates. Tests and documentation have their own directories; older Quest dataset utilities are in `legacy/`.

The original `~/Desktop/pc_receiver` was copied, not removed. Running receiver and replay processes were not restarted during organization. New launch commands use this checkout. Avoid making simultaneous changes in both source copies; treat this checkout as the maintained source after migration.

## Local-only resources

Ignored configuration files under `configs/` reference the existing camera, WiLoR, Anipose, and analysis environments. Ignored links under `data/` point to the existing dataset and calibration recording directories. Operations through those links modify the original data; they are not backups.

The exact source-copy inventory is stored locally in `logs/migration_manifest.json`. Runtime paths and recordings are intentionally absent from the Git repository. A new machine needs its own environments, model assets, local configs and data.

## Calibration status at migration

The local configuration still references the October 2, 2026 calibration. Subsequent camera geometry changes were observed, so this is not certification that the present rig is ready to collect accurate 3D ground truth.

Two October 7 calibration recordings and combined experiments produced candidate parameters, but none passed all existing held-out validation criteria. Those experiments did not replace formal calibration. A separate diagnostic replay uses one candidate with Anipose and bone/temporal constraints; its per-joint weighted stage has not been run. The candidate diagnostic replay and the original processing failure must not be presented as completed validated weighted processing.

Calibration files, historical session identifiers, and experimental results remain in the ignored local data tree. Update local configuration only when deliberately selecting the intended calibration, and preserve provenance for existing sessions.

## Changes made while organizing

- Added a launcher that resolves environment paths from local configuration and selects the proper Python for each stage.
- Converted the temporary calibration recording helper into `pc_receiver/record_calibration.py` with duration arguments, explicit board confirmation, and bounded recording/cleanup.
- Kept the existing reconstruction algorithm and model environments.
- Preserved example configurations for fresh checkouts and ignored machine-specific copies.
- Kept model source and third-party assets external. The legacy image-folder WiLoR adapter was copied from the local camera calibration workspace.

The historical synchronization document retains older commands for reference. Use the repository README for commands with the new layout.
