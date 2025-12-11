1. Installation

Run the following command to install all required dependencies:

pip install -r requirements.txt

2. Codebase Overview
   
configs/
This folder contains all configuration files, including the metrics and metadata used to construct the synthetic tables.
Intermediate computed values referenced in these configs can be inspected in data/cache/.

generator/stages/
This directory contains the main logic of the repository.
The pipeline is divided into six sequential stages, all orchestrated through pipeline.py.

Pipeline Controls@
You can run the pipeline in multiple modes using the commands below:


               COMMANDS FOR FLEXIBLE PIPELINE EXECUTION:

1. Run the full pipeline (Stages 1 → 6):
   python -m src.generator.pipeline

2. List all available stages:
   python -m src.generator.pipeline --list

3. Run a specific subset of stages (example: only Stages 3 and 4):
   python -m src.generator.pipeline --only 3,4

4. Run the full pipeline but skip a stage (example: skip Stage 3):
   python -m src.generator.pipeline --skip 3

5. Run a custom range of stages (example: Stages 2 → 5):
   python -m src.generator.pipeline --start 2 --end 5
===============================================================================


data/cache/
This directory stores intermediate outputs generated at different stages of the pipeline.
It is useful for debugging, verification, and understanding how data evolves through the system.
