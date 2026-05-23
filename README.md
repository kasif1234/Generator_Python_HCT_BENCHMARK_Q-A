# Generator Python HCT Benchmark Q-A

A Python-based pipeline for generating synthetic benchmark tables and question-answer data for HCT-style benchmarking tasks.

This repository is designed as a configurable data generation system. It uses metadata, metrics, and staged processing logic to construct synthetic tables, store intermediate outputs, and support flexible pipeline execution.

-----------------------------------------------------------------------------------------------------------
## Research Poster related to this project

<img width="1952" height="1379" alt="Image" src="https://github.com/user-attachments/assets/7d28c000-708a-4fd1-b916-358c6f99ec80" />
-----------------------------------------------------------------------------------------------------------

## Project Overview

The goal of this project is to generate structured synthetic data that can be used for benchmarking question-answering systems over tabular data.

The pipeline is split into multiple stages so that each part of the generation process can be inspected, debugged, and rerun independently. This makes the project easier to maintain, test, and extend.

At a high level, the workflow is:

```text
Configuration Files
        ↓
Pipeline Stages
        ↓
Intermediate Cache Files
        ↓
Generated Tables / Benchmark Data
        ↓
Question-Answer Outputs
```

---

## Key Features

- Config-driven synthetic table generation
- Six-stage pipeline architecture
- Flexible command-line execution
- Ability to run the full pipeline or selected stages only
- Intermediate cache storage for debugging and verification
- Organized project structure for future extension
- Useful for benchmark dataset creation and QA pipeline testing

---

## Repository Structure

```text
Generator_Python_HCT_BENCHMARK_Q-A/
│
├── configs/
│   └── Configuration files used by the generator
│
├── data/
│   └── Data files and generated outputs
│
├── data/cache/
│   └── Intermediate outputs created during pipeline execution
│
├── src/generator/
│   └── Main source code for the generator
│
├── src/generator/stages/
│   └── Stage-wise logic used by the pipeline
│
├── src/generator/pipeline.py
│   └── Main pipeline controller
│
├── requirements.txt
│   └── Python dependencies
│
├── pyproject.toml
│   └── Project configuration
│
├── .gitignore
│   └── Files and folders excluded from Git
│
└── README.md
    └── Project documentation
```

---

## Codebase Overview

### `configs/`

This folder contains the configuration files used to control the synthetic table generation process.

These configuration files may include:

- Metrics
- Metadata
- Table construction rules
- Values used during generation
- Parameters used by different pipeline stages

The pipeline reads these files and uses them as the foundation for generating the benchmark data.

---

### `src/generator/`

This folder contains the main Python implementation of the project.

It includes the pipeline controller and the logic responsible for reading configurations, processing data, generating synthetic tables, and saving outputs.

---

### `src/generator/stages/`

This directory contains the core staged logic of the repository.

The pipeline is divided into six sequential stages. Each stage is responsible for one part of the generation workflow.

This staged design makes it easier to:

- Debug individual steps
- Rerun only selected parts of the pipeline
- Inspect intermediate results
- Extend the system with new generation logic

---

### `data/cache/`

This folder stores intermediate outputs produced during pipeline execution.

The cache is useful for:

- Debugging
- Checking how data changes after each stage
- Verifying intermediate computations
- Avoiding confusion when tracing the pipeline flow

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kasif1234/Generator_Python_HCT_BENCHMARK_Q-A.git
```

Move into the project folder:

```bash
cd Generator_Python_HCT_BENCHMARK_Q-A
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

The pipeline can be executed from the command line using Python module execution.

### Run the Full Pipeline

This runs all stages from Stage 1 to Stage 6:

```bash
python -m src.generator.pipeline
```

---

### List All Available Stages

Use this command to display all available pipeline stages:

```bash
python -m src.generator.pipeline --list
```

---

### Run Specific Stages Only

For example, to run only Stage 3 and Stage 4:

```bash
python -m src.generator.pipeline --only 3,4
```

---

### Skip a Specific Stage

For example, to run the full pipeline but skip Stage 3:

```bash
python -m src.generator.pipeline --skip 3
```

---

### Run a Custom Range of Stages

For example, to run from Stage 2 to Stage 5:

```bash
python -m src.generator.pipeline --start 2 --end 5
```

---

## Pipeline Execution Modes

| Command | Purpose |
|---|---|
| `python -m src.generator.pipeline` | Runs the full pipeline |
| `python -m src.generator.pipeline --list` | Lists all available stages |
| `python -m src.generator.pipeline --only 3,4` | Runs only selected stages |
| `python -m src.generator.pipeline --skip 3` | Runs all stages except Stage 3 |
| `python -m src.generator.pipeline --start 2 --end 5` | Runs a custom stage range |

---

## Why the Pipeline is Stage-Based

The project is designed as a multi-stage pipeline instead of one large script.

This makes the system easier to understand because each stage performs a specific part of the generation process. It also makes debugging easier, since intermediate outputs can be inspected inside `data/cache/`.

For example, if the final generated benchmark data does not look correct, you can check the cached outputs from earlier stages to identify where the issue started.

---

## Typical Workflow

A typical workflow for using this repository is:

```text
1. Update or inspect the configuration files in configs/
2. Run the full pipeline
3. Check intermediate outputs in data/cache/
4. Verify the generated benchmark data
5. Rerun selected stages if changes are needed
```

---

## Example Usage

Run the full generation pipeline:

```bash
python -m src.generator.pipeline
```

Inspect the cached intermediate files:

```text
data/cache/
```

Modify configuration files if needed:

```text
configs/
```

Then rerun the required pipeline stages:

```bash
python -m src.generator.pipeline --only 3,4
```

---

## Debugging and Verification

The `data/cache/` folder is important for understanding how the pipeline works internally.

Use it to check:

- Whether each stage is producing the expected output
- Whether configuration values are being processed correctly
- Whether generated tables follow the expected structure
- Whether intermediate computed values are correct

This makes the project easier to test and improve over time.

---

## Extending the Project

This repository can be extended by adding:

- New configuration files
- New table generation rules
- New benchmark question types
- New metadata fields
- Additional pipeline stages
- More validation checks
- More output formats

When adding new logic, it is recommended to keep the staged design so the system remains easy to debug and maintain.

---

## Requirements

The project dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Project Purpose

This project is useful for anyone working on:

- Synthetic data generation
- Tabular question answering
- Benchmark dataset creation
- Natural language query testing
- Structured data evaluation
- Pipeline-based data generation systems

---

## Notes

- The pipeline is controlled through `src/generator/pipeline.py`.
- Configuration files are stored in `configs/`.
- Intermediate results are stored in `data/cache/`.
- The project is designed to support flexible execution of different stages.

---

## Author

**Mohammad Kashif**

GitHub: [@kasif1234](https://github.com/kasif1234)

---

## License

No license file has been added yet.

If this repository is intended for public reuse, consider adding an open-source license such as MIT, Apache 2.0, or BSD-3-Clause.
