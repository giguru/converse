from typing import List, Union
import numpy as np
import logging
from farm.infer import Inferencer

from converse.src.document_store.base import BaseDocumentStore
from converse.src.schema import Document

from converse.src.retriever.neural_retriever_pipeline_step import NeuralRetrieverPipelineStep

logger = logging.getLogger(__name__)


class EmbeddingRetriever(NeuralRetrieverPipelineStep):
    def __init__(self, document_store: BaseDocumentStore, embedding_model: str, use_gpu: bool = True,
                 model_format: str = "farm", pooling_strategy: str = "reduce_mean", emb_extraction_layer: int = -1):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such as ``'deepset/sentence_bert'``
        :param use_gpu: Whether to use gpu or not
        :param model_format: Name of framework that was used for saving the model. Options:

                             - ``'farm'``
                             - ``'transformers'``
                             - ``'sentence_transformers'``
        :param pooling_strategy: Strategy for combining the embeddings from the model (for farm / transformers models only).
                                 Options:

                                 - ``'cls_token'`` (sentence vector)
                                 - ``'reduce_mean'`` (sentence vector)
                                 - ``'reduce_max'`` (sentence vector)
                                 - ``'per_token'`` (individual token vectors)
        :param emb_extraction_layer: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
                                     Default: -1 (very last layer).
        """
        super().__init__(document_store)
        self.document_store = document_store
        self.model_format = model_format
        self.embedding_model = embedding_model
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer

        logger.info(f"Init retriever using embeddings of model {embedding_model}")
        if model_format == "farm" or model_format == "transformers":
            self.embedding_model = Inferencer.load(
                embedding_model, task_type="embeddings", extraction_strategy=self.pooling_strategy,
                extraction_layer=self.emb_extraction_layer, gpu=use_gpu, batch_size=4, max_seq_len=512, num_processes=0
            )

        elif model_format == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
            # e.g. 'roberta-base-nli-stsb-mean-tokens'
            if use_gpu:
                device = "cuda"
            else:
                device = "cpu"
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        else:
            raise NotImplementedError

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        if index is None:
            index = self.document_store.index
        query_emb = self.embed(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], filters=filters,
                                                           top_k=top_k, index=index)
        return documents

    def embed(self, texts: Union[List[str], str]) -> List[np.array]:
        """
        Create embeddings for each text in a list of texts using the retrievers model (`self.embedding_model`)

        :param texts: Texts to embed
        :return: List of embeddings (one per input text). Each embedding is a list of floats.
        """

        # for backward compatibility: cast pure str input
        if type(texts) == str:
            texts = [texts]  # type: ignore
        assert type(texts) == list, "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"

        if self.model_format == "farm" or self.model_format == "transformers":
            emb = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])  # type: ignore
            emb = [(r["vec"]) for r in emb]
        elif self.model_format == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            # get back list of numpy embedding vectors
            emb = self.embedding_model.encode(texts)  # type: ignore
            emb = [r for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of queries. For this Retriever type: The same as calling .embed()

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        return self.embed(texts)

    def embed_passages(self, docs: List[Document], show_logging: bool = True) -> List[np.array]:
        """
        Create embeddings for a list of passages. For this Retriever type: The same as calling .embed()

        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        texts = [d.text for d in docs]

        return self.embed(texts)
