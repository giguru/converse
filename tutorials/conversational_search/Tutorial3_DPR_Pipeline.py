from datasets import load_dataset
from haystack import Pipeline
from haystack.document_store import FAISSDocumentStore
from haystack.eval import EvalTREC
from haystack.retriever import DensePassageRetriever
from haystack.query_rewriting.transformer import GenerativeReformulator


# LOAD DATASET
topics = load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'topics', split="test")
qrels = load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'qrels', split="test")
collection = load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'test_collection_sample', split="test")

# convert data into the right format
qrels = {d['qid']: {d['qrels']['docno'][i]: d['qrels']['relevance'][i] for i in range(len(d['qrels']['docno']))} for d in qrels}
topics = [topic for qid, topic in enumerate(topics)]

document_store = FAISSDocumentStore()
# LOAD COMPONENTS
reformulator = GenerativeReformulator(pretrained_model_path="castorini/t5-base-canard")
retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=False,
                                  use_fast_tokenizers=False,
                                  )
# Important:
# Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
# previously indexed documents and update their embedding representation.
# While this can be a time consuming operation (depending on corpus size), it only needs to be done once.
# At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
# However, the size of the embedding index is HUGE. I recommend using the DenseAnseriniRetriever with binary models.
# It reduces the index size of the Natural Questions dataset from 62GB to 2GB. Pretty awesome...
document_store.update_embeddings(retriever, batch_size=1000)
document_store.save('faiss-index.idx')

eval_retriever = EvalTREC(top_k_eval_documents=1000)

# BUILD PIPELINE
p = Pipeline()
p.add_node(component=reformulator, name="Reformulator", inputs=["Query"])
p.add_node(component=retriever, name="Retriever", inputs=["Reformulator"])
p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["Retriever"])

p.eval_qrels(topics=topics, qrels=qrels, dump_results=True)
eval_retriever.print()
exit()
