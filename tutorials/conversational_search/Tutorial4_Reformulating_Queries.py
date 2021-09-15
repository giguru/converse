from haystack import Pipeline
from haystack.eval import EvalTREC
from haystack.query_rewriting.transformer import GenerativeReformulator, ClassificationReformulator
from haystack.retriever.anserini import SparseAnseriniRetriever
import datasets


# LOAD DATASETS
topics = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'topics', split="test")
qrels = datasets.load_dataset('uva-irlab/trec-cast-2019-multi-turn', 'qrels', split="test")
# Convert into the right data format
topics = [topic for qid, topic in enumerate(topics)]
qrels = {d['qid']: {d['qrels']['docno'][i]: d['qrels']['relevance'][i] for i in range(len(d['qrels']['docno']))} for d in qrels}

# LOAD COMPONENTS
# You can either use the ClassificationReformulator with the 'uva-irlab/quretec' model...
reformulator = ClassificationReformulator(pretrained_model_path="uva-irlab/quretec")
# or use the GenerativeReformulator with any Seq2SeqLM model
reformulator2 = GenerativeReformulator(pretrained_model_path="castorini/t5-base-canard")

retriever = SparseAnseriniRetriever(prebuilt_index_name='cast2019', searcher_config={"BM25": {}})
eval_retriever = EvalTREC(top_k_eval_documents=1000)

# BUILD PIPELINE
for r in [reformulator, reformulator2]:
    p = Pipeline()
    p.add_node(component=r, name="Reformulator", inputs=["Query"])
    p.add_node(component=retriever, name="Retriever", inputs=["Reformulator"])
    p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["Retriever"])

    # Do evaluation
    p.eval_qrels(qrels=qrels, topics=topics, dump_results=True)

    # Print metric results.
    # ClassificationReformulator: recall_1000=0.6447, recip_rank=0.5098, map=0.1929
    # GenerativeReformulator: recall_1000= , recip_rank= , map=
    reformulator.print()
    eval_retriever.print()

    # Many components register execution time, so you can print the total execution times.
    # On a Macbook Pro 13 inch, 2020 with 2 GHz Quad-Core Intel Core i5, the retriever took 1.249s/query
    # the ClassificationReformulator with uva-irlab/quretec 2.282s/query and
    # the GenerativeReformulator with castorini/t5-base-canard 2.540s/query
    reformulator.print_time()
    retriever.print_time()
exit()