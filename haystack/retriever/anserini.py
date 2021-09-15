from abc import ABC
from typing import List, Dict, Callable
from datasets import Dataset
from pyserini.search import SimpleSearcher
from haystack import Document
from haystack.retriever.base import BaseRetriever
import logging
import os
import json

logger = logging.getLogger(__name__)

__all__ = ['SparseAnseriniRetriever']


class AnseriniRetriever(BaseRetriever, ABC):
    def _build_index_using_huggingface(self,
                                       dataset: Dataset,
                                       converter=None,
                                       storeRaw: bool = True,
                                       storeVectors: bool = False,
                                       storePositions: bool = False
                                       ):
        index_path = './huggingface_anserini_' + dataset.info.builder_name
        if os.path.isfile(index_path):
            logger.info(f"Using existing Lucene index: {index_path}")
        else:
            logger.info('Preparing .jsonl for indexing Anserini')
            temp_jsonl_file = './temp.jsonl'
            if os.path.isfile(temp_jsonl_file):
                os.remove(temp_jsonl_file)

            n_empty_documents = []
            writer = open(f'{temp_jsonl_file}', 'w')
            for doc in dataset:
                doc_to_index = converter(doc) if converter else doc
                if 'id' not in doc_to_index:
                    raise KeyError(f'Each document should have an ID! Object: {doc_to_index}')
                if 'content' not in doc_to_index or not doc_to_index['content']:
                    n_empty_documents.append(doc_to_index['id'])

                writer.write(json.dumps(doc_to_index, separators=(',', ':')) + '\n')
                break
            writer.close()

            if len(n_empty_documents) > 0:
                logger.info(f"{len(n_empty_documents)} documents were indexed: {', '.join(n_empty_documents)}")

            logger.info(f'Creating indexing with Anserini at {index_path}. May take a while since stemming and '
                        f'stopword removal is enabled...')

            from jnius import autoclass
            JIndexCollection = autoclass('io.anserini.index.IndexCollection')
            args_for_main = [
                '-generator DefaultLuceneDocumentGenerator',
                '-collection JsonCollection',
                f'-input {temp_jsonl_file}',
                f'-index {index_path}',
                '-pretokenized false',
                '-stemmer porter',
                f'-threads {self.num_threads}',
            ]
            if storeRaw:
                args_for_main.append('-storeRaw')
            if storeVectors:
                args_for_main.append('-storeDocvectors')
            if storePositions:
                args_for_main.append('-storePositions')
            JIndexCollection.main(args_for_main)

            os.remove(temp_jsonl_file)
        return index_path

    def retrieve(self, **kwargs) -> List:
        query = kwargs.get('query', None)  # type: str
        if not query:
            raise KeyError(f'Please provide a `query`. The args are: {kwargs}')

        top_k = kwargs.get('top_k', None) or self.top_k  # type: int

        hits = self.searcher.search(q=query, k=top_k)
        results = []
        for hit in hits:
            doc = self.searcher.doc(hit.docid)
            results.append(Document(id=hit.docid, score=hit.score, text=doc.raw()))
        return results

"""
class DenseAnseriniRetriever(AnseriniRetriever):
    
    def __init__(self,
                 query_encoder: QueryEncoder = None,
                 top_k: int = 1000,
                 prebuilt_index_name: str = None,
                 num_threads: int = 3,
                 huggingface_dataset: Dataset = None,
                 huggingface_dataset_converter: Callable = None,
                 ):
        
        By default, this retriever was designed to do Dense Passage Retrieval.

        @param prebuilt_index_name: str
            E.g. msmarco-passage-tct_colbert-hnsw. For all available prebuilt indexes,
            please call pyserini.dsearch.SimpleDenseSearcher.list_prebuilt_indexes() or search on Google.
        @param num_threads: int
            Indexing Anserini allows for multithreading
        
        logger.info(f'{self.__class__.__name__}')
        self.num_threads = num_threads

        encoder = query_encoder or DprQueryEncoder(encoder_dir="facebook/dpr-question_encoder-single-nq-base")
        if prebuilt_index_name is not None:
            self.searcher = SimpleDenseSearcher.from_prebuilt_index(prebuilt_index_name, encoder)
        elif huggingface_dataset is not None:
            # TODO add possibility to index your own datasets as well.
            raise NotImplementedError()
            # self.index_path = self._build_index_using_huggingface(huggingface_dataset, huggingface_dataset_converter)
            # self.searcher = SimpleDenseSearcher(self.index_path, encoder)
        else:
            raise ValueError('Please provide either a prebuilt_index_name or huggingface_dataset.')

        self.top_k = top_k
"""

class SparseAnseriniRetriever(AnseriniRetriever):
    def __init__(self,
                 searcher_config: Dict[str, dict],
                 top_k: int = 1000,
                 prebuilt_index_name: str = None,
                 huggingface_dataset: Dataset = None,
                 huggingface_dataset_converter: Callable = None,
                 num_threads: int = 3,
                 ):
        """
        @param prebuilt_index_name: str
            E.g. robust04, msmarco-passage-slim, msmarco-passage or cast2019. For all available prebuilt indexes,
            please call pyserini.SimpleSearcher.list_prebuilt_indexes() or search on Google.
        @param huggingface_dataset: Dataset
            Please check if you need to provided a huggingface_dataset_converter as well
        @param huggingface_dataset_converter: Callable
            The object provided to anserini for indexing requires two keys: `id` and `contents`. If the entries in the
            huggingface dataset do not provide these keys, please provider a converter function:
            e.g. lamda d: {'id': d[...], 'contents': d[...]}
        @param num_threads: int
            Indexing anserini allows for multithreading
        """
        logger.info(f'{self.__class__.__name__} with {searcher_config}')
        self.num_threads = num_threads
        if prebuilt_index_name is not None:
            self.searcher = SimpleSearcher.from_prebuilt_index(prebuilt_index_name)
        elif huggingface_dataset is not None:
            self.index_path = self._build_index_using_huggingface(huggingface_dataset, huggingface_dataset_converter)
            self.searcher = SimpleSearcher(self.index_path)
        else:
            raise ValueError('Please provide either a prebuilt_index_name or huggingface_dataset.')

        for key, params in searcher_config.items():
            if key == 'Dirichlet':
                self.searcher.set_qld(**params)
            elif key == 'BM25':
                self.searcher.set_bm25(**params)
            elif key == 'RM3':
                self.searcher.set_rm3(**params)
            elif key in self.searcher:
                getattr(self.searcher, key)(**params)
            else:
                raise KeyError("Invalid key in `searcher_config`. The allowed keys are: Dirichlet, BM25, RM3 or a function of SimpleSearcher")

        self.top_k = top_k
