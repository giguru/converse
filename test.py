from converse.src.reader.farm import FARMReader
from converse.src.converse import Converse
from converse.src.preprocessor.utils import orconvqa_read_files

from converse.src.retriever.dense_passage_retriever import DensePassageRetriever
from converse.src.document_store.faiss import FAISSDocumentStore

import logging

from converse.src.retriever.orconvqa_retriever import ORConvQARetriever

logger = logging.getLogger(__name__)

logger.info('Creating document store...')
document_store = FAISSDocumentStore(vector_dim=128)

logger.info('Loading data files...')
pf = 'datasets/predefined/orconvqa/'
labels, documents = orconvqa_read_files(
    # Contains questions and answers (and a bunch of other things like history)
    filename=pf + 'preprocessed/dev.txt',
    # Links questions to docs
    qrelsfile=pf + 'qrels.txt',
    # Really just calls a separate function, doing this in tandem means we could check if doc-ids in qrels exist
    # We assume qrels is always correct for now
    buildCorpus=True,
    # "Block" file containing the raw text blocks and their ids
    corpusFile=pf + 'document_blocks/all_blocks.txt')

logger.info('Writing data into documents store...')

# add eval data calls this internally, we could add it to the eval data function or just do it like this
document_store.write_documents(documents)
document_store.write_labels(labels)

logger.info('Setting up retriever...')
retriever = ORConvQARetriever(
    document_store=document_store,
    use_gpu=True,
    embed_title=True,
    max_seq_len=256,
    batch_size=16,
    remove_sep_tok_from_untitled_passages=True
)
logger.info('Update embeddings...')
document_store.update_embeddings(retriever)

logger.info('Setting up reader...')
# Load a local model or any of the QA models on Hugging Face's model hub (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="bert-base-uncased", use_gpu=True, num_processes=2)

converse = Converse(reader, [retriever])

## Evaluate pipeline
# Evaluate combination of Reader and Retriever through Finder
logger.info('Evaluate...')
eval_results = converse.eval(top_k_retriever=1, top_k_reader=10)
converse.print_eval_results(eval_results)

