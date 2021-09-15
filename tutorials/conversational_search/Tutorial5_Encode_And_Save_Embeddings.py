from datasets import load_dataset
from haystack import Document
from haystack.document_store import FAISSDocumentStore
from haystack.retriever import DensePassageRetriever
from tqdm import tqdm
import json

collection = load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'test_collection', split="test")

document_store = FAISSDocumentStore()
# LOAD COMPONENTS
retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=False,
                                  progress_bar=False,
                                  use_fast_tokenizers=False)

# Process embeddings in batches
batch_size = 100000
M = 1000000
embeddings = {}
for i in tqdm(range(20 * M, collection.num_rows, batch_size)):
    dictionary = collection[i:i + batch_size]
    document_objects = []
    n = len(dictionary['docno'])
    for j in range(n):
        document_objects.append(Document.from_dict({
            'id': dictionary['docno'][j],
            'text': dictionary['text'][j]
        }))

    batch_embeddings = list(retriever.embed_passages(document_objects))

    filename = f'data-{i}-{i+n}.txt'
    with open(filename, 'w') as fp:
        for j in range(n):
            doc_no = dictionary['docno'][j]
            emb = [str(x) for x in list(batch_embeddings[j])[:256]]
            fp.write(f"{doc_no} {' '.join(emb)}\n")