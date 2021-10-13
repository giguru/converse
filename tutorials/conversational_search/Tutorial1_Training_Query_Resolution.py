import argparse, sys, logging
from transformers import BertConfig
import numpy as np, random, torch

sys.path.append('..')
from haystack.query_rewriting.query_resolution_processor import QuretecProcessor
from haystack.query_rewriting.query_resolution import QueryResolution


# Voskarides et al. uses a seed of 42
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

config = BertConfig.from_pretrained("bert-large-uncased",
                                    num_labels=len(QuretecProcessor.get_labels()),
                                    finetuning_task="ner",
                                    hidden_dropout_prob=0.4,
                                    label2id=QuretecProcessor.label2id(),
                                    id2label=QuretecProcessor.id2label(),
                                    pad_token_id=QuretecProcessor.pad_token_id())

# Define the processor which converts the dataset into input features/tensors
processor = QuretecProcessor(max_seq_len=300)
processor.set_dataset()


query_resolution = QueryResolution(config=config,
                                   use_gpu=True,
                                   model_args={'bert_model': "bert-large-uncased", 'max_seq_len': 300},
                                   processor=processor)
query_resolution.train(evaluate_every=2500,
                       datasilo_args={"caching": False},
                       learning_rate=5e-5)
query_resolution.eval()

exit()