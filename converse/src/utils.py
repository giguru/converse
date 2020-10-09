import os
import json

def load_qrels_json(fname: str, logger):
    # Original method from ORConvQA: train_pipeline.load_json
    logger.info(f'loading json file {fname}')
    if not os.path.isfile(fname):
        logger.error(f'Failed to open {fname}, file not found')
    with open(fname) as handle:
        return json.load(handle)