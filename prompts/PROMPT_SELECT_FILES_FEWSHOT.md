<example index="1">
<user>
Select relevant files for this repository:
Repo: https://github.com/sharkdp/bat
Structure (build-input files after doc/CI stripping):
- Cargo.toml
- Cargo.lock
- rust-toolchain.toml
- build.rs
- src/ (main.rs, lib.rs, ...)
- assets/, tests/, benches/
</user>
<assistant>
selected_files:
  - Cargo.toml
  - Cargo.lock
  - rust-toolchain.toml
  - build.rs
</assistant>
</example>

<example index="2">
<user>
Select relevant files for this repository:
Repo: https://github.com/numpy/numpy
Structure (build-input files after doc/CI stripping):
- pyproject.toml
- meson.build
- meson.options
- requirements/build_requirements.txt
- numpy/ (package sources)
- tools/, benchmarks/
</user>
<assistant>
selected_files:
  - pyproject.toml
  - meson.build
  - meson.options
  - requirements/build_requirements.txt
</assistant>
</example>
