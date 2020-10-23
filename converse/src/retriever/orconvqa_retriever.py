from converse.models.orconvqa.modeling import AlbertForRetrieverOnlyPositivePassage
from typing import List, Union, Tuple, Optional

from transformers import AlbertConfig, AlbertTokenizer

from converse.src.document_store.base import BaseDocumentStore
from converse.src.query_formatters import conversational_question_formatter
from converse.src.schema import Document

from transformers.modeling_dpr import DPRContextEncoder, DPRQuestionEncoder
from transformers.tokenization_dpr import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer

from converse.src.retriever.pipeline_composition_error import PipelineCompositionError
from converse.src.retriever.neural_retriever_pipeline_step import NeuralRetrieverPipelineStep

import logging
import torch
import numpy as np
import os

logger = logging.getLogger(__name__)


class ORConvQARetriever(NeuralRetrieverPipelineStep):
    """
        Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
        See the original paper for more details:
        Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
        (https://arxiv.org/abs/2004.04906).
    """

    def __init__(self, document_store: BaseDocumentStore,
                 max_seq_len: int = 256,
                 use_gpu: bool = True, batch_size: int = 16, embed_title: bool = True,
                 remove_sep_tok_from_untitled_passages: bool = True
                 ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param max_seq_len: Longest length of each sequence
        :param use_gpu: Whether to use gpu or not
        :param batch_size: Number of questions or passages to encode at once
        :param embed_title: Whether to concatenate title and passage to a text pair that is then used to create the embedding
        :param remove_sep_tok_from_untitled_passages: If embed_title is ``True``, there are different strategies to deal with documents that don't have a title.
        If this param is ``True`` => Embed passage as single text, similar to embed_title = False (i.e [CLS] passage_tok1 ... [SEP]).
        If this param is ``False`` => Embed passage as text pair with empty title (i.e. [CLS] [SEP] passage_tok1 ... [SEP])
        """

        super().__init__(document_store)
        self.document_store = document_store
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len


        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_used = "GPU"
        else:
            self.device = torch.device("cpu")
            device_used = "CPU"

        self.embed_title = embed_title
        self.remove_sep_tok_from_untitled_passages = remove_sep_tok_from_untitled_passages

        # Load pretrained retrievers
        binary_dir = './converse/models/orconvqa/pipeline_checkpoint/checkpoint-45000/retriever'
        token_dir = './converse/models/orconvqa/retriever_checkpoint'

        self.query_encoder = AlbertForRetrieverOnlyPositivePassage.from_pretrained(binary_dir, force_download=True).to(self.device)
        self.query_tokenizer = AlbertTokenizer.from_pretrained(token_dir)

        self.passage_tokenizer = AlbertTokenizer.from_pretrained(token_dir)
        self.passage_encoder = AlbertForRetrieverOnlyPositivePassage.from_pretrained(binary_dir, force_download=True).to(self.device)
        logger.info(f"ORConvQARetriever initialised with {type(document_store).__name__} Document Store, torch using {device_used} and model found in location {binary_dir} and tokenizer in location {token_dir}. The batch_size is {batch_size} and the max_seq_len is {max_seq_len}.")


    def retrieve(self, questions: List[str], previous_documents: List[Document], filters: dict = None, top_k: int = 10) -> List[Document]:
        if previous_documents is not None:
            raise PipelineCompositionError('the Dense Passage Retriever was taken from Haystack and was not designed to be used as a follow retriever')
        # Use the document store default index
        index = self._index or self.document_store.index
        query = self._query_formatter(questions)
        query_emb = self.embed_queries(texts=[query])
        return self.document_store.query_by_embedding(query_emb=query_emb[0], top_k=top_k, index=index)

    def embed_queries(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        queries = [self._normalize_query(q) for q in texts]
        result = self._generate_batch_predictions(texts=queries, model=self.query_encoder,
                                                  tokenizer=self.query_tokenizer,
                                                  embedding_queries=True,
                                                  batch_size=self.batch_size)
        return result

    def embed_passages(self, docs: List[Document], show_logging: bool = True) -> List[np.array]:
        """
        Create embeddings for a list of passages using the passage encoder

        :param docs: List of Document objects used to represent documents / passages in a standardized way within Haystack.
        :return: Embeddings of documents / passages shape (batch_size, embedding_dim)
        """
        texts = [d.text for d in docs]
        result = self._generate_batch_predictions(texts=texts,
                                                  model=self.passage_encoder,
                                                  tokenizer=self.passage_tokenizer,
                                                  embedding_passages=True,
                                                  show_logging=show_logging,
                                                  batch_size=self.batch_size)
        return result

    def _normalize_query(self, query: str) -> str:
        if query[-1] == '?':
            query = query[:-1]
        return query

    def _query_formatter(self, questions: List[str]):
        if len(questions) == 0:
            raise ValueError('The list of questions should contain at least one question')

        return conversational_question_formatter(questions)


    def _tensorizer(self, tokenizer: Union[AlbertTokenizer],
                    text: List[str],
                    title: Optional[List[str]] = None,
                    add_special_tokens: bool = True):
        """
        Creates tensors from text sequences
        :Example:
            >>> ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained()
            >>> dpr_object._tensorizer(tokenizer=ctx_tokenizer, text=passages, title=titles)

        :param tokenizer: An instance of DPRQuestionEncoderTokenizer or DPRContextEncoderTokenizer.
        :param text: list of text sequences to be tokenized
        :param title: optional list of titles associated with each text sequence
        :param add_special_tokens: boolean for whether to encode special tokens in each sequence

        Returns:
                token_ids: list of token ids from vocabulary
                token_type_ids: list of token type ids
                attention_mask: list of indices specifying which tokens should be attended to by the encoder
        """

        # combine titles with passages only if some titles are present with passages
        if self.embed_title and title:
            final_text = [tuple((title_, text_)) for title_, text_ in zip(title, text)] #type: Union[List[Tuple[str, ...]], List[str]]
        else:
            final_text = text
        out = tokenizer.batch_encode_plus(final_text, add_special_tokens=add_special_tokens, truncation=True,
                                              max_length=self.max_seq_len,
                                              pad_to_max_length=True)

        token_ids = torch.tensor(out['input_ids']).to(self.device)
        token_type_ids = torch.tensor(out['token_type_ids']).to(self.device)
        attention_mask = torch.tensor(out['attention_mask']).to(self.device)
        return token_ids, token_type_ids, attention_mask

    def _remove_sep_tok_from_untitled_passages(self, titles, ctx_ids_batch, ctx_attn_mask):
        """
        removes [SEP] token from untitled samples in batch. For batches which has some untitled passages, remove [SEP]
        token used to segment titles and passage from untitled samples in the batch
        (Official DPR code do not encode [SEP] tokens in untitled passages)

        :Example:
            # Encoding passages with 'embed_title' = True. 1st passage is titled, 2nd passage is untitled
            >>> texts = ['Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions.',
                          'Democratic Republic of the Congo to the south. Angola\'s capital, Luanda, lies on the Atlantic coast in the northwest of the country.'
                        ]
            >> titles = ["0", '']
            >>> token_ids, token_type_ids, attention_mask = self._tensorizer(self.passage_tokenizer, text=texts, title=titles)
            >>> [self.passage_tokenizer.ids_to_tokens[tok.item()] for tok in token_ids[0]]
            ['[CLS]', '0', '[SEP]', 'aaron', 'aaron', '(', 'or', ';', ....]
            >>> [self.passage_tokenizer.ids_to_tokens[tok.item()] for tok in token_ids[1]]
            ['[CLS]', '[SEP]', 'democratic', 'republic', 'of', 'the', ....]
            >>> new_ids, new_attn = self._remove_sep_tok_from_untitled_passages(titles, token_ids, attention_mask)
            >>> [self.passage_tokenizer.ids_to_tokens[tok.item()] for tok in token_ids[0]]
            ['[CLS]', '0', '[SEP]', 'aaron', 'aaron', '(', 'or', ';', ....]
            >>> [self.passage_tokenizer.ids_to_tokens[tok.item()] for tok in token_ids[1]]
            ['[CLS]', 'democratic', 'republic', 'of', 'the', 'congo', ...]

        :param titles: list of titles for each sample
        :param ctx_ids_batch: tensor of shape (batch_size, max_seq_len) containing token indices
        :param ctx_attn_mask: tensor of shape (batch_size, max_seq_len) containing attention mask

        Returns:
                ctx_ids_batch: tensor of shape (batch_size, max_seq_len) containing token indices with [SEP] token removed
                ctx_attn_mask: tensor of shape (batch_size, max_seq_len) reflecting the ctx_ids_batch changes
        """
        # Skip [SEP] removal if passage encoder not bert model
        if self.passage_encoder.ctx_encoder.base_model_prefix != 'bert_model':
            logger.warning("Context encoder is not a BERT model. Skipping removal of [SEP] tokens")
            return ctx_ids_batch, ctx_attn_mask

        # create a mask for titles in the batch
        titles_mask = torch.tensor(list(map(lambda x: 0 if x == "" else 1, titles))).to(self.device)

        # get all untitled passage indices
        no_title_indices = torch.nonzero(1 - titles_mask).squeeze(-1)

        # remove [SEP] token index for untitled passages and add 1 pad to compensate
        ctx_ids_batch[no_title_indices] = torch.cat((ctx_ids_batch[no_title_indices, 0].unsqueeze(-1),
                                                     ctx_ids_batch[no_title_indices, 2:],
                                                     torch.tensor([self.passage_tokenizer.pad_token_id]).expand(len(no_title_indices)).unsqueeze(-1).to(self.device)),
                                                    dim=1)
        # Modify attention mask to reflect [SEP] token removal and pad addition in ctx_ids_batch
        ctx_attn_mask[no_title_indices] = torch.cat((ctx_attn_mask[no_title_indices, 0].unsqueeze(-1),
                                                     ctx_attn_mask[no_title_indices, 2:],
                                                     torch.tensor([self.passage_tokenizer.pad_token_id]).expand(len(no_title_indices)).unsqueeze(-1).to(self.device)),
                                                    dim=1)

        return ctx_ids_batch, ctx_attn_mask

    def _generate_batch_predictions(self,
                                    texts: List[str],
                                    model: AlbertForRetrieverOnlyPositivePassage,
                                    tokenizer: Union[AlbertTokenizer],
                                    embedding_passages: bool = False,
                                    embedding_queries: bool = False,
                                    show_logging: bool = True,
                                    batch_size: int = 16) -> List[Tuple[object, np.array]]:
        n = len(texts)
        total = 0
        results = []
        for batch_start in range(0, n, batch_size):
            # create batch of text
            ctx_text = texts[batch_start:batch_start + batch_size]

            # tensorize the batch
            ctx_ids_batch, _, ctx_attn_mask = self._tensorizer(tokenizer, text=ctx_text, title=None)
            ctx_seg_batch = torch.zeros_like(ctx_ids_batch).to(self.device)

            with torch.no_grad():
                if embedding_passages is True and embedding_queries is True:
                    raise ValueError('Only one of the two can be true')
                elif embedding_queries is True:
                    out = model(query_input_ids=ctx_ids_batch, query_attention_mask=ctx_attn_mask, query_token_type_ids=ctx_seg_batch)
                elif embedding_passages is True:
                    out = model(passage_input_ids=ctx_ids_batch, passage_attention_mask=ctx_attn_mask, passage_token_type_ids=ctx_seg_batch)
                # TODO revert back to when updating transformers
                # out = out.pooler_output
                out = out[0]
            out = out.cpu()

            total += ctx_ids_batch.size()[0]

            results.extend([
                (out[i].view(-1).numpy())
                for i in range(out.size(0))
            ])

            if show_logging and total % 10 == 0:
                logger.info(f'Embedded {total} / {n} texts')

        return results