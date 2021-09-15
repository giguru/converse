import logging
import sys
import datasets
from haystack import Pipeline
from haystack.eval import EvalDocuments
from haystack.query_rewriting.transformer import ClassificationReformulator
from haystack.ranker import FARMRanker
from haystack.retriever.anserini import SparseAnseriniRetriever

# Load data sets
topics = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'topics', split="test")
qrels = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'qrels', split="test")

# Convert into the right data format
qrels = {d['qid']: {d['qrels']['docno'][i]: d['qrels']['relevance'][i] for i in range(len(d['qrels']['docno']))} for d in qrels}
topics = [topic for qid, topic in enumerate(topics)]

# Load pipeline elements
reformulator = ClassificationReformulator(pretrained_model_path="uva-irlab/quretec")
retriever = SparseAnseriniRetriever(prebuilt_index_name='cast2019', searcher_config={"Dirichlet": {'mu': 2500}})
eval_retriever = EvalDocuments(top_k_eval_documents=1000, open_domain=False)
ranker = FARMRanker(model_name_or_path="castorini/monobert-large-msmarco-finetune-only", top_k=1000, progress_bar=False)
eval_reranker = EvalDocuments(top_k_eval_documents=1000, open_domain=False)

# Build pipeline
p = Pipeline()
p.add_node(component=reformulator, name="Rewriter", inputs=["Query"])
p.add_node(component=retriever, name="Retriever", inputs=["Rewriter"])
p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["Retriever"])
p.add_node(component=ranker, name="Reranker", inputs=["Retriever"])
p.add_node(component=eval_reranker, name="EvalReranker", inputs=["Reranker"])

p.eval_qrels(qrels=qrels, topics=topics, dump_results=True)

# Print metric results
eval_retriever.print()
eval_reranker.print()

# Many components register execution time, so you can print the total execution times. The execution time for the
# full pipeline 25s/query using a GeForce 1080Ti.
retriever.print_time()
reformulator.print_time()
ranker.print_time()
exit()