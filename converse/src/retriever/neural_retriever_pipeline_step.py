from abc import abstractmethod
import numpy as np

from typing import List

from converse.src.retriever.retriever_pipeline_step import RetrieverPipelineStep
from converse.src.schema import Document


class NeuralRetrieverPipelineStep(RetrieverPipelineStep):

    @abstractmethod
    def embed_queries(self, texts: List[str]) -> List[np.array]:
        pass

    @abstractmethod
    def embed_passages(self, docs: List[Document], show_logging: bool = True) -> List[np.array]:
        pass
