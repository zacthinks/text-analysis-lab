# Upgrading legacy TeAL projects

TeAL's strict frozen-operator format was introduced during pre-1.0 development. Projects created before that change can contain valid historical artifacts whose recorded operators use the older snapshot format.

TeAL does **not** upgrade these projects automatically when they are opened. Historical results remain readable, but current reusable-Pipeline APIs may reject an old operator until an explicit upgrade is performed.

## Inspect first

Open the project and run a dry run:

```python
import text_analysis_lab as teal

project = teal.Project.open("my_project")
report = project.upgrade_legacy_operators(dry_run=True)
print(report)
```

The report classifies each pre-strict operator. Only operators TeAL can prove are exactly migratable are eligible for a strict replacement. Historical-only or invalid snapshots are reported and left untouched.

## Apply the upgrade

After inspecting the report:

```python
report = project.upgrade_legacy_operators()
print(report)
```

For each eligible operator, TeAL:

- creates or reuses one exact strict snapshot;
- rebinds completed historical operations to that strict replacement only when it is reusable;
- moves operator aliases to the reusable replacement;
- preserves the original legacy operator snapshot;
- preserves existing artifact IDs and payloads;
- records the old-to-new mapping and operation rebinds for audit and recovery.

The upgrade does **not** rerun translations, refit models, regenerate artifacts, or reinterpret ambiguous historical state.

## Partial upgrades are expected

A project may contain a mixture of exactly migratable and historical-only operators. That does not block safe upgrades elsewhere.

For example, an older model-backed sentence decomposition stage may remain historical-only while a downstream count-vectorization stage is migrated exactly. In that case a Pipeline beginning at the already-materialized sentence artifact may become replayable even though a Pipeline beginning before the historical-only model stage remains correctly unavailable.

## Idempotency and interrupted upgrades

The project-level upgrade is idempotent. Running it again does not create another strict replacement after a canonical mapping has been recorded.

Operation descriptors live on the filesystem while operation references live in the SQLite catalog, so TeAL records descriptor-synchronization state during the upgrade. If the process is interrupted after the catalog rebind, rerunning `upgrade_legacy_operators()` repairs any pending operation descriptors before planning new work.

## Single-operator migration remains conservative

`Project.migrate_legacy_operator(ref)` still only creates a new strict snapshot from one exactly migratable legacy operator. It does **not** rewrite historical operations.

Use `Project.upgrade_legacy_operators()` when the goal is to make an existing project use current strict replay/Pipeline semantics.
