import logging
import re
from typing import List, Tuple, Callable
import os
import pyterrier as pt
from datasets import Dataset

from haystack import Document
from haystack.retriever.base import BaseRetriever

log = logging.getLogger(__name__)


class TerrierRetriever(BaseRetriever):
    def __init__(self,
                 top_k: int,
                 searcher_args: dict,
                 indexer_args: dict = None,
                 indexer_class = None,
                 document_store = None,
                 prebuilt_index_name: str = None,
                 index_path: str = None,
                 huggingface_dataset: Dataset = None,
                 huggingface_dataset_converter: Callable = None,
                 max_doc_id_len: int = 50,
                 max_text_len: int = 4096,
                 debug: bool = True
                 ):
        """
        @type prebuilt_index_name: str
            PyTerrier provides several data sets. For the full list, please consult:
            https://pyterrier.readthedocs.io/en/latest/datasets.html
        @type document_store: BaseDocumentStore
            Any Haystack document store.
        @type top_k: int
        """
        super().__init__(debug=debug)

        self.top_k = top_k
        self._indexer_args = indexer_args or {'type': pt.IndexingType.CLASSIC, 'overwrite': False}
        self._indexer_class = indexer_class or pt.IterDictIndexer
        self.MAX_DOC_ID_LEN = max_doc_id_len
        self.MAX_TEXT_LEN = max_text_len

        if document_store is None and prebuilt_index_name is None and huggingface_dataset is None and index_path is None:
            raise KeyError(f"Please provide either a `document_store`, `index_path`, `prebuilt_index_name` or `huggingface_dataset`. "
                           f"A possible pt_datasets is e.g. 'trec-deep-learning-passages'. "
                           f"The following datasets are included in Terrier: {pt.list_datasets()}")

        self.config = searcher_args
        if 'wmodel' in searcher_args:
            self.wmodel = self.config["wmodel"]
        else:
            raise KeyError('The config should contain a "wmodel" key. Amongst the possible values are: TF_IDF, BM25,'
                           'BM25F, Dirichlet_LM. For more, please consult the Terrier documentation.')

        self.index_ref = None
        self.meta_keys = None

        self.pt_dataset = None
        self.document_store = None
        self.huggingface_dataset = None

        if huggingface_dataset is not None:
            self.index_path = index_path or self.config.get("index_path", './huggingface_' + huggingface_dataset.info.builder_name)
            self.huggingface_dataset = huggingface_dataset
            self._huggingface_dataset_converter = huggingface_dataset_converter
            self._build_index_using_huggingface_dataset()
        elif prebuilt_index_name is not None:
            self.index_path = index_path or self.config.get("index_path", './terrier_' + prebuilt_index_name)
            self.pt_dataset = pt.get_dataset(prebuilt_index_name)
            self._build_index_using_pt_dataset()
        elif document_store is not None:
            self.index_path = index_path or self.config.get("index_path", './terrier_indexed')
            self.document_store = document_store
            self._build_index_using_document_store()
        elif 'index_path' in self.config.keys() or index_path:
            self.index_path = index_path or self.config.get("index_path")
        log.info(f"{self.__class__.__name__} with config={searcher_args}")

    def _clean_text(self, text):
        """
        Terrier expects cleaned data only!
        """
        pattern = re.compile('[\W_]+') # include only alphanumeric
        return pattern.sub(' ', text).lower()

    def _document_store_iter(self, documents: List[Document]):
        N = len(documents)
        log.debug(f"Iterating over {N} documents")
        print_every = N // 25
        for i, doc in enumerate(documents, 1):
            if print_every > 0 and i % print_every == 0:
                log.debug(f"\t{i} of {N} done")
            if len(doc.id) > self.MAX_DOC_ID_LEN:
                log.warning(f"DocumentStore Document {doc.id} has an ID longer than {self.MAX_DOC_ID_LEN}")
            yield {
                "text": self._clean_text(doc.text),
                "docno": doc.id
            }

    def _huggingface_dataset_iter(self):
        for item in self.huggingface_dataset:
            return_item = self._huggingface_dataset_converter(item) if self._huggingface_dataset_converter else item
            if 'docno' not in return_item or 'text' not in return_item:
                raise KeyError('The item to be indexed should contain the keys `docno` and `text`. If the dataset does'
                               'not have these keys, please provide a `huggingface_dataset_converter` when'
                               'constructing')
            if len(return_item['docno']) > self.MAX_DOC_ID_LEN:
                log.warning(f"Dataset document {return_item['docno']} has an ID longer than {self.MAX_DOC_ID_LEN}")
            yield return_item

    def __get_iter_indexer(self):
        return self._indexer_class(self.index_path, **self._indexer_args)

    def __set_batch_retriever(self):
        index_inst = pt.IndexFactory.of(self.index_ref)

        log.debug(f"Index stats:\n{index_inst.getCollectionStatistics().toString()}")
        self.batch_retriever = pt.BatchRetrieve(index_inst,
                                                num_results=self.top_k,
                                                metadata=['docno', 'text'],
                                                **self.config)

    def _build_index_using_huggingface_dataset(self):
        log.debug("Building index using HuggingFace Dataset")
        iter_indexer = self.__get_iter_indexer()
        existing_index_path = self.index_path + "/data.properties"
        if os.path.isfile(existing_index_path):
            self.index_ref = pt.IndexRef.of(existing_index_path)
        else:
            self.index_ref = iter_indexer.index(self._huggingface_dataset_iter(),
                                                meta=['docno', 'text'],
                                                meta_lengths=[self.MAX_DOC_ID_LEN, self.MAX_TEXT_LEN])
        self.__set_batch_retriever()
        log.debug("Building index using PYTerrier Dataset complete!")

    def _build_index_using_pt_dataset(self):
        log.debug("Building index using Pyterrier Dataset")
        iter_indexer = self.__get_iter_indexer()
        existing_index_path = self.index_path + "/data.properties"
        if os.path.isfile(existing_index_path):
            self.index_ref = pt.IndexRef.of(existing_index_path)
        else:
            self.index_ref = iter_indexer.index(self.pt_dataset.get_corpus_iter(),
                                                meta=['docno', 'text'],
                                                meta_lengths=[self.MAX_DOC_ID_LEN, self.MAX_TEXT_LEN])
        self.__set_batch_retriever()
        log.debug("Building index using PYTerrier Dataset complete!")

    def _build_index_using_document_store(self):
        log.debug("Building index using Document Store")
        iter_indexer = self.__get_iter_indexer()
        documents = self.document_store.get_all_documents()
        self.index_ref = iter_indexer.index(self._document_store_iter(documents),
                                            meta=['docno', 'text'],
                                            meta_lengths=[self.MAX_DOC_ID_LEN, self.MAX_TEXT_LEN]
                                            )
        self.__set_batch_retriever()
        log.debug("Building index using Document Store complete!")

    def batch_retrieve(self, queries: List[Tuple[str, str]], filters: dict = None, index: str = None):
        raise NotImplementedError()
        pass

    def _get_document(self, id) -> Document:
        return self.document_store.get_document_by_id(id=id)

    def retrieve(self, **kwargs) -> List:
        query = kwargs.get('query', None)  # type: str
        filters = kwargs.get('filters', None)  # type: dict
        top_k = kwargs.get('top_k', None)  # type: int
        index = kwargs.get('index', None)  # type: str
        if filters or index or top_k:
            raise NotImplementedError("filters, index and top_k not supported yet")

        results = []
        cleaned_text = self._clean_text(query)
        from jnius import JavaException
        try:
            for _, row in self.batch_retriever.search(cleaned_text).iterrows():
                if self.pt_dataset or self.huggingface_dataset:
                    meta = {key: row[key] for key in row.keys() if key not in ['docno', 'text', 'score']}
                    results.append(Document(id=row['docno'],
                                            text=row['text'],
                                            score=row['score'],
                                            meta=meta
                                            ))
                elif self.document_store:
                    results.append(self._get_document(row["docno"]))
                else:
                    raise NotImplementedError("I don't know what's happening, but no results are being returned...")
        except JavaException as ja:
            print('\n\t'.join(ja.stacktrace))
        return results
