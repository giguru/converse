from typing import Union, List, Optional, Dict
from haystack import Document, Label
from haystack.document_store.base import BaseDocumentStore
from pyserini.search import SimpleSearcher
from pyserini.search._base import Document as PyseriniDoc
import numpy as np


__all__ = ['AnseriniSparseDocumentStore']


def pyserini_hit_to_haystack_doc(doc: PyseriniDoc) -> Document:
    return Document(text=doc.contents(), id=doc.id())


class AnseriniSparseDocumentStore(BaseDocumentStore):
    def __init__(self, prebuilt_index_name:str = None):
        """
        :param prebuilt_index_name: For instance, 'msmarco-passage'
        """
        if prebuilt_index_name:
            self.searcher = SimpleSearcher.from_prebuilt_index(prebuilt_index_name)
        else:
            AnseriniSparseDocumentStore.print_prebuilt_indices()
            raise NotImplementedError(f"For now, only prebuilt indices are supported.")

    @classmethod
    def from_prebuilt_indices(cls, prebuilt_index_name: str):
        return SimpleSearcher.from_prebuilt_index(prebuilt_index_name=prebuilt_index_name)

    @staticmethod
    def print_prebuilt_indices():
        return SimpleSearcher.list_prebuilt_indexes()

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None):
        raise NotImplementedError("For now, only prebuilt indices are supported.")

    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        """
        pass

    def get_all_labels(self,
                       index: Optional[str] = None,
                       filters: Optional[Dict[str, List[str]]] = None
                       ) -> List[Label]:
        pass

    def write_labels(self, labels: Union[List[Label], List[dict]], index: Optional[str] = None):
        pass

    def get_label_count(self, index: Optional[str] = None) -> int:
        pass

    def delete_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        raise NotImplementedError("For now, only prebuilt indices are supported.")

    def query_by_embedding(self,
                           query_emb: np.ndarray,
                           filters: Optional[Optional[Dict[str, List[str]]]] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        raise NotImplementedError(f"For dense retrieval is not supported yet")

    def query(self, query: Optional[str], top_k: int = 10) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by the BM25 algorithm.

        :param query: The query
        :param top_k: How many documents to return per query.
        """
        results = self.searcher.search(q=query, k=top_k)
        return [pyserini_hit_to_haystack_doc(hit) for hit in results]

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        doc = self.searcher.doc(docid=id)
        if doc:
            return pyserini_hit_to_haystack_doc(doc)
        return None

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        return self.searcher.num_docs