import logging
from guppy import hpy; h=hpy()
from pathlib import Path
from sys import platform
from typing import Union, List, Optional, Dict

import faiss
import numpy as np
from tqdm import tqdm
from faiss.swigfaiss import IndexHNSWFlat

from converse.src.schema import Document
from converse.src.document_store.sql import SQLDocumentStore
from converse.src.retriever.neural_retriever_pipeline_step import NeuralRetrieverPipelineStep

if platform != 'win32' and platform != 'cygwin':
    import faiss
else:
    raise ModuleNotFoundError("FAISSDocumentStore on windows platform is not supported")

logger = logging.getLogger(__name__)


class FAISSDocumentStore(SQLDocumentStore):
    """
    Document store for very large scale embedding based dense retrievers like the DPR.
    It implements the FAISS library(https://github.com/facebookresearch/faiss)
    to perform similarity search on vectors.
    The document text and meta-data (for filtering) are stored using the SQLDocumentStore, while
    the vector embeddings are indexed in a FAISS Index.
    """

    def __init__(
        self,
        sql_url: str = "sqlite:///",
        index_buffer_size: int = 10_000,
        vector_dim: int = 768,
        faiss_index_factory_str: str = "Flat",
        faiss_index: Optional[faiss.swigfaiss.Index] = None,
        **kwargs,
    ):
        """
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended.
        :param index_buffer_size: When working with large datasets, the ingestion process(FAISS + SQL) can be buffered in
                                  smaller chunks to reduce memory footprint.
        :param vector_dim: the embedding vector size.
        :param faiss_index_factory_str: Create a new FAISS index of the specified type.
                                        The type is determined from the given string following the conventions
                                        of the original FAISS index factory.
                                        Recommended options:
                                        - "Flat" (default): Best accuracy (= exact). Becomes slow and RAM intense for > 1 Mio docs.
                                        - "HNSW": Graph-based heuristic. If not further specified,
                                                  we use a RAM intense, but more accurate config:
                                                  HNSW256, efConstruction=256 and efSearch=256
                                        - "IVFx,Flat": Inverted Index. Replace x with the number of centroids aka nlist.
                                                          Rule of thumb: nlist = 10 * sqrt (num_docs) is a good starting point.
                                        For more details see:
                                        - Overview of indices https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
                                        - Guideline for choosing an index https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
                                        - FAISS Index factory https://github.com/facebookresearch/faiss/wiki/The-index-factory
                                        Benchmarks: XXX
        :param faiss_index: Pass an existing FAISS Index, i.e. an empty one that you configured manually
                            or one with docs that you used in Haystack before and want to load again.
        """
        self.vector_dim = vector_dim

        if faiss_index:
            self.faiss_index = faiss_index
        else:
            self.faiss_index = self._create_new_index(vector_dim=self.vector_dim, index_factory=faiss_index_factory_str, **kwargs)

        self.index_buffer_size = index_buffer_size
        super().__init__(url=sql_url)

    def _create_new_index(self, vector_dim: int, index_factory: str = "Flat", metric_type=faiss.METRIC_INNER_PRODUCT, **kwargs):
        if index_factory == "HNSW" and metric_type == faiss.METRIC_INNER_PRODUCT:
            # faiss index factory doesn't give the same results for HNSW IP, therefore direct init.
            # defaults here are similar to DPR codebase (good accuracy, but very high RAM consumption)
            n_links = kwargs.get("n_links", 128)
            index = faiss.IndexHNSWFlat(vector_dim, n_links, metric_type)
            index.hnsw.efSearch = kwargs.get("efSearch", 20)#20
            index.hnsw.efConstruction = kwargs.get("efConstruction", 80)#80
            logger.info(f"HNSW params: n_links: {n_links}, efSearch: {index.hnsw.efSearch}, efConstruction: {index.hnsw.efConstruction}")
        else:
            index = faiss.index_factory(vector_dim, index_factory, metric_type)
        return index

    def write_documents(self,
                        documents: Union[List[dict],
                        List[Document]],
                        index: Optional[str] = None,
                        embeddingRetriever: NeuralRetrieverPipelineStep = None):
        """
        Add new documents to the DocumentStore.
        :param documents: List of `Documents`. If they already contain the embeddings, we'll index them right away
                          in FAISS. If not, you can later call update_embeddings() to create & index them.
        :param index: (SQL) index name for storing the docs and metadata
        :return:
        """
        # vector index
        if not self.faiss_index:
            raise ValueError("Couldn't find a FAISS index. Try to init the FAISSDocumentStore() again ...")
        # doc + metadata index
        index = index or self.index

        for i in tqdm(range(0, len(documents), self.index_buffer_size)):
            document_objects = documents[i: i + self.index_buffer_size]
            add_vectors = False if document_objects[0].embedding is None else True

            vector_id = self.faiss_index.ntotal
            if embeddingRetriever != None:
                embeddings = np.array(embeddingRetriever.embed_passages(document_objects), dtype="float32")
                self.faiss_index.add(embeddings)
                del embeddings

            if add_vectors:
                embeddings = [doc.embedding for doc in document_objects]
                embeddings = np.array(embeddings, dtype="float32")
                self.faiss_index.add(embeddings)
                del embeddings  # save memory

            docs_to_write_in_sql = []
            for doc in document_objects:
                meta = doc.meta
                if add_vectors or embeddingRetriever:
                    meta["vector_id"] = vector_id
                    vector_id += 1
                docs_to_write_in_sql.append(doc)

            super(FAISSDocumentStore, self).write_documents(docs_to_write_in_sql, index=index)
            del document_objects, docs_to_write_in_sql  # release memory

    def update_embeddings(self, retriever: NeuralRetrieverPipelineStep, index: Optional[str] = None, batch_size: int = 80):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).
        :param retriever: Retriever to use to get embeddings for text
        :param index: (SQL) index name for storing the docs and metadata
        :return: None
        """
        # To clear out the FAISS index contents and frees all memory immediately that is in use by the index
        self.faiss_index.reset()

        index = index or self.index

        total_number_of_documents = self.get_document_count(index=index)
        if total_number_of_documents == 0:  # only check empty result on first iteration
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            self.faiss_index = None
            return
        logger.info(f"Updating embeddings for with {total_number_of_documents} docs...")

        batch_number = 0
        for k in tqdm(range(0, total_number_of_documents, batch_size)):
            documents = self.get_batch_of_documents(index=index, batch_number=batch_number, batch_size=batch_size)
            if len(documents) == 0:
                break

            embeddings = retriever.embed_passages(documents, show_logging=False)  # type: ignore
            assert len(documents) == len(embeddings)

            vector_id_map = {}
            vector_id = self.faiss_index.ntotal
            for i, doc in enumerate(documents):
                doc.embedding = embeddings[i]
                embeddings = np.array(embeddings, dtype="float32")
                self.faiss_index.add(embeddings)

                vector_id_map[doc.id] = vector_id
                vector_id += 1

            self.update_vector_ids(vector_id_map, index=index)

            # increment batch number
            batch_number = batch_number + 1


    def train_index(self, documents: Optional[Union[List[dict], List[Document]]], embeddings: Optional[np.array] = None):
        """
        Some FAISS indices (e.g. IVF) require initial "training" on a sample of vectors before you can add your final vectors.
        The train vectors should come from the same distribution as your final ones.
        You can pass either documents (incl. embeddings) or just the plain embeddings that the index shall be trained on.
        :param documents: Documents (incl. the embeddings)
        :param embeddings: Plain embeddings
        :return: None
        """

        if embeddings and documents:
            raise ValueError("Either pass `documents` or `embeddings`. You passed both.")
        if documents:
            document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
            embeddings = [doc.embedding for doc in document_objects]
            embeddings = np.array(embeddings, dtype="float32")
        self.faiss_index.train(embeddings)

    def delete_all_documents(self, index=None):
        index = index or self.index
        self.faiss_index.reset()
        super().delete_all_documents(index=index)

    def query_by_embedding(
        self, query_emb: np.array, filters: Optional[dict] = None, top_k: int = 10, index: Optional[str] = None
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.
        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: (SQL) index name for storing the docs and metadata
        :return:
        """
        if filters:
            raise Exception("Query filters are not implemented for the FAISSDocumentStore.")
        if not self.faiss_index:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")

        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        score_matrix, vector_id_matrix = self.faiss_index.search(query_emb, top_k)
        vector_ids_for_query = [str(vector_id) for vector_id in vector_id_matrix[0] if vector_id != -1]

        documents = self.get_documents_by_vector_ids(vector_ids_for_query, index=index)

        #assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in zip(vector_id_matrix[0], score_matrix[0])}
        for doc in documents:
            doc.score = scores_for_vector_ids[doc.meta["vector_id"]]  # type: ignore
            doc.probability = (doc.score + 1) / 2
        return documents

    def save(self, file_path: Union[str, Path]):
        """
        Save FAISS Index to the specified file.
        :param file_path: Path to save to.
        :return: None
        """
        faiss.write_index(self.faiss_index, str(file_path))

    @classmethod
    def load(
            cls,
            faiss_file_path: Union[str, Path],
            sql_url: str,
            index_buffer_size: int = 10_000,
    ):
        """
        Load a saved FAISS index from a file and connect to the SQL database.
        Note: In order to have a correct mapping from FAISS to SQL,
              make sure to use the same SQL DB that you used when calling `save()`.
        :param faiss_file_path: Stored FAISS index file. Can be created via calling `save()`
        :param sql_url: Connection string to the SQL database that contains your docs and metadata.
        :param index_buffer_size: When working with large datasets, the ingestion process(FAISS + SQL) can be buffered in
                                  smaller chunks to reduce memory footprint.
        :return:
        """
        """
        """
        faiss_index = faiss.read_index(str(faiss_file_path))
        return cls(
            faiss_index=faiss_index,
            sql_url=sql_url,
            index_buffer_size=index_buffer_size,
            vector_dim=faiss_index.d
        )