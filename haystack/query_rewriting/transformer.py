import logging, torch, re
from typing import Any, List, Union, Callable

from farm.data_handler.processor import Processor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
from farm.data_handler.dataloader import NamedDataLoader
from haystack.query_rewriting.base import BaseReformulator
from haystack.query_rewriting.query_resolution import QueryResolutionModel
from haystack.query_rewriting.query_resolution_processor import QuretecProcessor

logger = logging.getLogger(__name__)


__all__ = ['GenerativeReformulator', 'ClassificationReformulator']


class ClassificationReformulator(BaseReformulator):
    """
    Query reformulation can be done by classification.
    This component was created for the model in the publication:
    "Query Resolution for Conversational Search with Limited Supervision
    """
    outgoing_edges = 1

    def __init__(self,
                 processor: Processor = None,
                 config = None,
                 bert_model: str = 'bert-large-uncased',
                 pretrained_model_path: str = None,
                 use_gpu: bool = True,
                 progress_bar: bool = True,
                 tokenizer_args: dict = {},
                 model_args: dict = {},
                 debug: bool = True
                 ):
        super().__init__(use_gpu=use_gpu, debug=debug)
        self.debug = debug
        self.set_config(config=config, bert_model=bert_model)
        # Directly store some arguments as instance properties
        self.progress_bar = progress_bar

        # Set derived instance properties
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, **tokenizer_args)

        if 'quretec' in pretrained_model_path:
            self.model = QueryResolutionModel.from_pretrained(pretrained_model_path, device=self.device, config=config,
                                                              **model_args)
            self.processor = processor or QuretecProcessor(tokenizer=self.tokenizer, max_seq_len=300)
        else:

            self.model = AutoModelForTokenClassification(pretrained_model_path, config=config, device=self.device,
                                                         **model_args)
            self.processor = processor
        self.model = self.model.to(self.device)

        if self.processor is not None and not self.processor.tokenizer:
            self.processor.tokenizer = self.tokenizer

    def _filter_predicted_terms(self, predicted_terms: List[str], query: str):
        # The predicted terms can contain duplicates. Only get the distinct values
        predicted_terms = [w.lower() for w in list(set(predicted_terms))]

        # Do not append the terms that are already in the query, so filter term
        query_words = re.split(r"[\W_]+", query)  # Get query terms by splitting on spaces and punctuation
        query_words = [w.lower() for w in query_words]
        return [term for term in predicted_terms if term not in query_words]

    def run_query(self, query, **kwargs):
        history = kwargs.get('history', [])
        if len(history) > 0:  # reformulating to include history is irrelevant when there is no history...
            self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)
            dataset, tensor_names, problematic_samples = self.processor.dataset_from_dicts(
                dicts=[{'query': query, **kwargs}])
            data_loader = NamedDataLoader(dataset=dataset, batch_size=1, tensor_names=tensor_names)
            for step, batch in enumerate(data_loader):
                batch = {key: batch[key].to(self.device) for key in batch}
                with torch.no_grad():
                    logits = self.model.forward(**batch)

                    # Check if the processor has a logits_to_terms function.
                    if callable(getattr(self.processor, 'logits_to_terms', None)):
                        predicted_terms = self.processor.logits_to_terms(logits=logits, batch=batch)
                    else:  # This assumes the QuReTEcProcesssor and QuretecModel
                        preds_per_head = self.model.logits_to_preds(logits=logits, **batch)
                        # Right now, only one prediction head is supported
                        predicted_terms = self.processor.predictions_to_terms(batch=batch, predictions=preds_per_head[0])[0]

                predicted_terms = self._filter_predicted_terms(predicted_terms, query)
                output = {
                    **kwargs,
                    'original_query': query,
                    'query': f"{query} {' '.join(predicted_terms)}" if len(predicted_terms) > 0 else query
                }
        else:
            output = {
                **kwargs,
                'original_query': query,
                'query': query
            }
        return output, "output_1"


class GenerativeReformulator(BaseReformulator):
    outgoing_edges = 1

    def __init__(self,
                 pretrained_model_path: str,
                 max_length: int = 64,
                 num_beams: int = 10,
                 use_gpu: bool = True,
                 tokenizer_args: dict = {},
                 model_args: dict = {},
                 early_stopping: bool = True,
                 history_processor: Callable = None,
                 debug: bool = True
                 ):
        """
        Reformulate queries using transformers.
        Since AutoModel and AutoTokenizer

        Combinations that can be used are:
        - pretrained_model_path='castorini/t5-base-canard',

        @param pretrained_model_path:
        @param max_length:
        @param num_beams:
        @param use_gpu:
        @param tokenizer_args:
        @param model_args:
        """
        super().__init__(use_gpu=use_gpu, debug=debug)

        self.max_length = max_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self._history_processor = history_processor

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, **tokenizer_args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path, **model_args)
        self.model = self.model.to(self.device)

    def _default_history_processor(self, query: str, history: Union[str, List[str]]):
        """
        This default history processor was designed to be compatible with the pretrained model
        'castorini/t5-base-canard'.
        """
        history_separator = '|||'
        src_text = f" {history_separator} ".join(history) if isinstance(history, list) else history
        src_text = f"{src_text} {history_separator} {query}"
        return src_text

    def run_query(self, query: str, history: Union[str, List[str]], **kwargs):
        original_query = query
        if len(history) > 0:  # reformulating to include history is irrelevant when there is no history...
            if self._history_processor:
                src_text = self._history_processor(query=query, history=history)
            else:
                src_text = self._default_history_processor(query=query, history=history)
            input_ids = self.tokenizer(src_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=self.early_stopping,
            )
            rewritten_query = self.tokenizer.decode(
                output_ids[0, 0:],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            output = {
                **kwargs,
                "query": rewritten_query,
                "original_query": original_query,
            }
        else:
            output = {
                **kwargs,
                "query": query,
                "original_query": query,
            }

        return output, "output_1"




