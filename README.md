# PrincoML (Archived)

PrincoML is a personal research/engineering machine-learning framework I built to experiment with custom gradient-based supervised learning architectures.

This repository is now archived as a historical artifact.

## Archive Status
- `Maintenance`: No active development, no planned feature work, and no support.
- `Ecosystem relevance`: Not aligned with the current Python/ML tooling ecosystem anymore.
- `Compatibility`: Modern environments may break without manual dependency pinning and code updates.
- `Intended use`: Reference material for past experiments, not a recommended foundation for new projects.

## Project Background
A few years ago I wanted more direct control over model architecture, equations, and training behavior than I felt was convenient in mainstream frameworks at the time. PrincoML was built to prioritize configurability for experimentation rather than ease-of-use.

It was used for internal experimentation across regression and neural-network style workflows, with a strong emphasis on custom training behavior.

## What This Repo Contains
- Core package code in `princo_ml/`
- Historical experiment notebooks (`*_testbed.ipynb`)
- Legacy release notes in `release_notes.md`

## Example Notebooks
These notebooks show historical examples of how the library was used:
- [`classify_testbed.ipynb`](./classify_testbed.ipynb)
- [`regression_testbed.ipynb`](./regression_testbed.ipynb)
- [`cnn_testbed.ipynb`](./cnn_testbed.ipynb)
- [`state_network_testbed.ipynb`](./state_network_testbed.ipynb)

## Installation (Legacy)
The package is available on PyPI as `princoml`: <https://pypi.org/project/princoml/?>  
This repository should still be treated as legacy code.

If you still need to reproduce prior work, use an isolated environment and pin dependencies carefully:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Important Caveats
- APIs and patterns here predate many current best practices.
- No guarantees are made around correctness in modern runtime/library combinations.
- Notebook outputs and workflows may require environment-specific fixes.

## For New Work
For current projects, use actively maintained ML frameworks and ecosystem tooling.

## Release Notes
Historical release notes are available at [`release_notes.md`](./release_notes.md).

## Author
Jason Cherry  
`JCherry@gmail.com`
