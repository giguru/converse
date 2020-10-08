from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers

from converse.src.reader.farm import FARMReader
from converse.src.reader.transformers import TransformersReader
from converse.src.retriever.dense_passage_retriever import DensePassageRetriever
from converse.src.converse import Converse


# Add document collection to a DocumentStore. The original text will be indexed. Conversion into embeddings can be
# is done below.
from converse.src.document_store.faiss import FAISSDocumentStore

document_store = FAISSDocumentStore()


# Let's first get some files that we want to use
# doc_dir = "data/article_txt_got"
# s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
# fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Download evaluation data, which is a subset of Natural Questions development set containing 50 documents
doc_dir = "../data/nq"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
# document_store.write_documents(dicts)

document_store.add_eval_data(filename="../data/nq/nq_dev_subset_v2.json")



from converse.src.retriever.dense_passage_retriever import DensePassageRetriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",  # TODO replace with ORConvQA model
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",  # TODO replace with ORConvQA model
    use_gpu=True,
    embed_title=True,
    max_seq_len=256,
    batch_size=16,
    remove_sep_tok_from_untitled_passages=True
)


# Load a local model or any of the QA models on Hugging Face's model hub (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True, num_processes=2)

finder = Converse(reader, [retriever])

#%% md

## Evaluate pipeline


# Evaluate combination of Reader and Retriever through Finder
finder_eval_results = finder.eval(top_k_retriever=1, top_k_reader=10)
finder.print_eval_results(finder_eval_results)

