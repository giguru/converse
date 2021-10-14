# ConverSE

Converse is a framework for doing research on Conversational Search. It was built/forked from
[Haystack](https://github.com/deepset-ai/haystack).

## Tutorials
It's quite easy to build a conversational search pipelines. Please consult the [conversational search  tutorials](https://github.com/giguru/converse/tree/master/tutorials/conversational_search) to see how.


```python
# Example pipeline

reformulator = ClassificationReformulator(pretrained_model_path="uva-irlab/quretec")
retriever = SparseAnseriniRetriever(prebuilt_index_name='cast2019', searcher_config={"Dirichlet": {'mu': 2500}})
eval_retriever = EvalTREC(top_k_eval_documents=1000)
ranker = FARMRanker(model_name_or_path="castorini/monobert-large-msmarco-finetune-only", top_k=1000, progress_bar=False)
eval_reranker = EvalTREC(top_k_eval_documents=1000)

# Build pipeline
p = Pipeline()
p.add_node(component=reformulator, name="Rewriter", inputs=["Query"])
p.add_node(component=retriever, name="Retriever", inputs=["Rewriter"])
p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["Retriever"])
p.add_node(component=ranker, name="Reranker", inputs=["Retriever"])
p.add_node(component=eval_reranker, name="EvalReranker", inputs=["Reranker"])

p.eval_qrels(qrels=qrels, topics=topics, dump_results=True)
```

## Core Features added on top of Haystack

- **Prebuilt dataset indices** from Anserini and Terrier.
- **Anserini support**: Integrate sparse and dense retrieval methods from [Pyserini](https://github.com/castorini/pyserini/).
- **Terrier support**: Integrate retrieval functions from
- **pytrec_eval** support to do evaluation.

## Haystack classes useful for Conversational Search

Please consult the Haystack for the conceptual explanation of document stores.

- **Pipeline**: Stick building blocks together to highly custom pipelines that are represented as Directed Acyclic Graphs (DAG). Think of it as "Apache Airflow for search".
- **FaissDocumentStore**
- **MilvusDocumentStore**
- **ElasticSearchDocumentStore**
- **FARMRanker**: Neural network (e.g., BERT or RoBERTA) that re-ranks top-k retrieved documents. The Ranker is an optional component and uses a TextPairClassification model under the hood. This model calculates semantic similarity of each of the top-k retrieved documents with the query.
- **FARMReader**: Neural network (e.g., BERT or RoBERTA) that reads through texts in detail
    to find an answer. The Reader takes multiple passages of text as input and returns top-n answers. Models are trained via [FARM](https://github.com/deepset-ai/FARM) or [Transformers](https://github.com/huggingface/transformers) on SQuAD like tasks.  You can load a pre-trained model from [Hugging Face's model hub](https://huggingface.co/models) or fine-tune it on your domain data.
- **RAGenerator**: Neural network (e.g., RAG) that *generates* an answer for a given question conditioned on the retrieved documents from the retriever.