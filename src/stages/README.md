# Stage Structure (Problemstellung Mapping)

This directory mirrors the thesis problem statement exactly.

## Schritt 1

- 1.1 Suche nach fuer die Installation relevanten Dateien im Repository
  - Package: `stages/stage_1_repository_installation_analysis/relevant_file_discovery/`
- 1.2 Extraktion von Installationsbefehlen aus den relevanten Dateien
  - Package: `stages/stage_1_repository_installation_analysis/install_command_extraction/`

## Schritt 2

- Generierung des Installationsskriptes (Dockerfile)
  - Package: `stages/stage_2_dockerfile_generation/`

## Schritt 3

- Iterative Reparatur des Installationsskriptes
  - Package: `stages/stage_3_iterative_dockerfile_repair/`

## Orchestration

- Pipeline orchestration
  - Package: `stages/pipeline_orchestration/`

## Additional Stage

- Install guide generation
  - Package: `stages/stage_4_install_guide/`

## Current Script Mapping

- Step 1.1 and 1.2: `stages/stage_1_repository_installation_analysis/agent_classify.py`
  - L1 module (relevant file discovery): `stages/stage_1_repository_installation_analysis/relevant_file_discovery/l1_iterative_react.py`
  - L2 module (install command extraction / synthesis): `stages/stage_1_repository_installation_analysis/install_command_extraction/l2_synthesis.py`
  - Classify validation module: `stages/stage_1_repository_installation_analysis/validation_evidence/classify_validation_loop.py`
  - Architecture orchestration graph: `stages/stage_1_repository_installation_analysis/orchestration/architecture_state_graph.py`
  - Architecture scratchpad payloads: `stages/stage_1_repository_installation_analysis/architecture_artifacts/scratchpad_payloads.py`
- Step 2: `stages/stage_2_dockerfile_generation/agent_dockerfile.py`
- Step 3: `stages/stage_3_iterative_dockerfile_repair/agent_dockerfile_repair.py`
- Pipeline: `stages/pipeline_orchestration/agent_pipeline.py`
- Validation gate: `stages/stage_2_dockerfile_generation/agent_validation_gate.py`
- Install guide: `stages/stage_4_install_guide/agent_install_guide.py`

## Shared Tools

- ReAct/shared tool builders: `agent_tools/react_loop_tools.py`
