import logging
from typing import List, Union, Tuple, Optional
import torch
import numpy as np

from converse.src.document_store.base import BaseDocumentStore
from haystack import Document
from haystack.retriever.sparse import logger

from transformers.modeling_dpr import DPRContextEncoder, DPRQuestionEncoder
from transformers.tokenization_dpr import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer

from converse.src.retriever.pipeline_composition_error import PipelineCompositionError
from converse.src.retriever.retriever_pipeline_step import RetrieverPipelineStep

logger = logging.getLogger(__name__)


class DensePassageRetriever(RetrieverPipelineStep):
    """
        Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
        See the original paper for more details:
        Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
        (https://arxiv.org/abs/2004.04906).
    """

    def __init__(self, document_store: BaseDocumentStore,
                 query_embedding_model: str = "facebook/dpr-question_encoder-single-nq-base",
                 passage_embedding_model: str = "facebook/dpr-ctx_encoder-single-nq-base", max_seq_len: int = 256,
                 use_gpu: bool = True, batch_size: int = 16, embed_title: bool = True,
                 remove_sep_tok_from_untitled_passages: bool = True
                 ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by hugging-face transformers' modelhub models
                                      Currently available remote names: ``"facebook/dpr-question_encoder-single-nq-base"``
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by hugging-face transformers' modelhub models
                                        Currently available remote names: ``"facebook/dpr-ctx_encoder-single-nq-base"``
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
        else:
            self.device = torch.device("cpu")

        self.embed_title = embed_title
        self.remove_sep_tok_from_untitled_passages = remove_sep_tok_from_untitled_passages

        # Init & Load Encoders
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_embedding_model)
        self.query_encoder = DPRQuestionEncoder.from_pretrained(query_embedding_model).to(self.device)

        self.passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained(passage_embedding_model)
        self.passage_encoder = DPRContextEncoder.from_pretrained(passage_embedding_model).to(self.device)

    def initial_retrieve(self, questions: List[str], filters: dict = None, top_k: int = 10) -> List[Document]:
        """
        :param index:
        :param questions: Questions from the same conversation
        :param filters:
        :param top_k:
        :return:
        """
        query = self._query_formatter(questions)
        return self.retrieve(query, filters=filters, top_k=top_k)

    def follow_up_retrieve(self, questions: List[str], documents: List[Document], filters: dict = None, top_k: int = 10) -> List[Document]:
        raise PipelineCompositionError('the Dense Passage Retriever was taken from Haystack and was not designed to be used as a follow retriever')

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        if index is None:
            # Use the document store default index
            index = self.document_store.index
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], top_k=top_k, index=index)
        return documents

    def embed_queries(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        queries = [self._normalize_query(q) for q in texts]
        result = self._generate_batch_predictions(texts=queries, model=self.query_encoder,
                                                  tokenizer=self.query_tokenizer,
                                                  batch_size=self.batch_size)
        return result

    def embed_passages(self, docs: List[Document]) -> List[np.array]:
        """
        Create embeddings for a list of passages using the passage encoder

        :param docs: List of Document objects used to represent documents / passages in a standardized way within Haystack.
        :return: Embeddings of documents / passages shape (batch_size, embedding_dim)
        """
        texts = [d.text for d in docs]
        titles = None
        if self.embed_title:
            titles = [d.meta["name"] if d.meta and "name" in d.meta else "" for d in docs]

        result = self._generate_batch_predictions(texts=texts, titles=titles,
                                                  model=self.passage_encoder,
                                                  tokenizer=self.passage_tokenizer,
                                                  batch_size=self.batch_size)
        return result

    def _normalize_query(self, query: str) -> str:
        if query[-1] == '?':
            query = query[:-1]
        return query

    def _tensorizer(self, tokenizer: Union[DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer],
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
                                    model: torch.nn.Module,
                                    tokenizer: Union[DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer],
                                    titles: Optional[List[str]] = None, #useful only for passage embedding with DPR!
                                    batch_size: int = 16) -> List[Tuple[object, np.array]]:
        n = len(texts)
        total = 0
        results = []
        for batch_start in range(0, n, batch_size):
            # create batch of titles only for passages
            ctx_title = None
            if self.embed_title and titles:
                ctx_title = titles[batch_start:batch_start + batch_size]

            # create batch of text
            ctx_text = texts[batch_start:batch_start + batch_size]

            # tensorize the batch
            ctx_ids_batch, _, ctx_attn_mask = self._tensorizer(tokenizer, text=ctx_text, title=ctx_title)
            ctx_seg_batch = torch.zeros_like(ctx_ids_batch).to(self.device)

            # remove [SEP] token from untitled passages in batch
            if self.embed_title and self.remove_sep_tok_from_untitled_passages and ctx_title:
                ctx_ids_batch, ctx_attn_mask = self._remove_sep_tok_from_untitled_passages(ctx_title,
                                                                                           ctx_ids_batch,
                                                                                           ctx_attn_mask)

            with torch.no_grad():
                out = model(input_ids=ctx_ids_batch, attention_mask=ctx_attn_mask, token_type_ids=ctx_seg_batch)
                # TODO revert back to when updating transformers
                # out = out.pooler_output
                out = out[0]
            out = out.cpu()

            total += ctx_ids_batch.size()[0]

            results.extend([
                (out[i].view(-1).numpy())
                for i in range(out.size(0))
            ])

            if total % 10 == 0:
                logger.info(f'Embedded {total} / {n} texts')

        return results