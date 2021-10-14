from datasets import load_dataset
from haystack import Document
from haystack.document_store import FAISSDocumentStore
from haystack.retriever import DensePassageRetriever
from tqdm import tqdm
import bz2, _pickle as cPickle, logging

logger = logging.getLogger(__name__)


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
    return cPickle.loads(bz2.BZ2File(file, 'rb').read())


# Load the component, in this case a retriever, with the model to embed the passages with
retriever = DensePassageRetriever(document_store=FAISSDocumentStore(),
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=False,
                                  progress_bar=False,
                                  use_fast_tokenizers=False)

collection = load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'test_collection', split="test")
TOTAL = collection.num_rows
START = 0
BATCH_SIZE = 10000

# Process embeddings in batches without storing them in the document store first
for i in tqdm(range(START, TOTAL, BATCH_SIZE)):
    dictionary = collection[i:i + BATCH_SIZE]
    document_objects = []
    n = len(dictionary['docno'])
    for j in range(n):
        document_objects.append(Document.from_dict({
            'id': dictionary['docno'][j],
            'text': dictionary['text'][j]
        }))

    batch_embeddings = list(retriever.embed_passages(document_objects))

    filename = f'data-{i}-{i+n}'
    logger.info(f"Writing data starting on {filename} with doc_id={dictionary['docno'][0]}")
    data = {}
    for j in range(n):
        doc_no = dictionary['docno'][j]
        data[doc_no] = [float(v) for v in list(batch_embeddings[j])]

    compressed_pickle(filename, data)