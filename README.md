# Converse
Converse is a Information Retrieval framework focused on Conversational Search built from [Haystack](https://github.com/deepset-ai/haystack) version
Sep 2020  (not around).

## Contents
1. [Get started.](#getstarted)
2. [How to use.](#howtouse)
3. [Acknowledgement.](#acknowledgement)

<a name="getstarted"></a>
## Get started
To get started, all you need to do is install this package via pip:
```
pip install git+https://github.com/giguru/converse 
```

The repository comes with a development data set. If you want to have a test data set, please run the following command
as well:
```
bash download_data.sh
```

<a name="howtouse"></a>
## How to use


### Document stores
Firstly, choose a document store which functions as a database and indexer.
```python
document_store = FAISSDocumentStore(vector_dim=128)
```

Converse comes with several document stores. Which type is optimal depends on your use case.

- FAISSDocumentStore: optimal for embedding retrieval.
- SQLDocumentStore: optimal for sparse retrieval.
- ElasticsearchDocumentStore: works well for both embedding and sparse retrieval.

### Add data to documents stores
Secondly, data needs to be added to the document store.
```python
label, documents = orconvqa_read_files(...)
document_store.write_documents(documents)
document_store.write_labels(labels)
```

### Define your components
Then define your retriever and reader
```python
retriever = ORConvQARetriever(
    document_store=document_store,
    use_gpu=True,
    embed_title=True,
    max_seq_len=256,
    batch_size=16,
    remove_sep_tok_from_untitled_passages=True
)
# Optionally create embeddings of the documents in the document store 
document_store.update_embeddings(retriever)

reader = FARMReader(model_name_or_path="converse/models/orconvqa/BertForORConvQAReader", use_gpu=True, num_processes=2)
``` 

### Tie it together and evaluate
```python
converse = Converse(reader, [retriever])

eval_results = converse.eval(top_k_retriever=1, top_k_reader=10)
converse.print_eval_results(eval_results)
```


### Example script
Their is a working script on the folder examples/test.py.

<a name="acknowledgement"></a>
## Acknowledgement
Converse is the results of the course Information Retrieval 2 of programme Artificial Intelligence of the University
of Amsterdam. The course was given by prof. dr. E. Kanoulas. The main authors are Giguru Scheuer and Melle Vessies.
Supervised by S. Bhargav, A. Krasakis and S. Vakulenko.

Please cite this work:
```

```