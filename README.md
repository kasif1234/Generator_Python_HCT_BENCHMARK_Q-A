1. pip install -r requirements.txt
   
2. Codebase - Understanding
    A. Configs folder contains the metrics/data that will be used to construct the synthetic tables (Check data/cache for details)
    B. generator/stages -> Main scripts of the repo, currently have 6 stages and all can be run and  controlled using pipeline.py
    ==============================================================================
    COMMANDS YOU CAN USE FOR BETTER CONTROL
    ==============================================================================
    (1) Full Pipeline (all stages): python -m src.generator.pipeline
    (2) List Stages: python -m src.generator.pipeline --list
    (3) Run a subset (e.g., only stages 3 and 4): python -m src.generator.pipeline --only 3,4
    (4) Run 1→6 but skip stage 3: python -m src.generator.pipeline --skip 3
    (5) Run 2→5: python -m src.generator.pipeline --start 2 --end 5

    C. data/cache contains intermediary results from running the stages
    
 
