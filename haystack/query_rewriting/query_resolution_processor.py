import os, logging, datasets, spacy, re
from pathlib import Path
# Use external dependency Spacy, because QuReTec also uses Spacy
from farm.data_handler.processor import Processor
from farm.data_handler.samples import Sample
from farm.evaluation.metrics import register_metrics
from spacy.tokens import Token
from transformers import BertTokenizer
from typing import List, Optional, Tuple
from haystack import Label

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")
__all__ = ['QuretecProcessor']

cached_parsed_spacy_token = {}

def get_entities(seq: list):
    """Gets entities from sequence.
    @param seq:
        sequence of labels.
    @return: list
        list of (chunk_type, index).

    Example:
        >>> seq = ['REL', 'REL', 'O', '[SEP]']
        >>> get_entities(seq)
        [('REL', 0), ('REL', 1), ('SEP', 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    return [(label, i) for i, label in enumerate(seq) if label != 'O']


def f1_micro(preds, labels):
    true_entities = set(get_entities(labels))
    pred_entities = set(get_entities(preds))

    correct = len(true_entities & pred_entities)
    pred = len(pred_entities)
    true = len(true_entities)

    micro_precision = correct / pred if pred > 0 else 0
    micro_recall = correct / true if true > 0 else 0

    if micro_precision + micro_recall == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    return {
        'micro_recall': micro_recall * 100,
        'micro_precision': micro_precision * 100,
        'micro_f1': micro_f1 * 100
    }


register_metrics('f1_micro', f1_micro)


def get_spacy_parsed_word(word: str) -> Token:
    if word not in cached_parsed_spacy_token:
        cached_parsed_spacy_token[word] = nlp(word)[0]
    return cached_parsed_spacy_token[word]


class QuretecProcessor(Processor):
    label_name_key = "question"
    gold_terms = "gold"

    labels = {
        "NOT_RELEVANT": "O",
        "RELEVANT": "REL",
    }
    """
    Used to handle the CANARD that come in json format.
    For more details on the dataset format, please visit: https://huggingface.co/datasets/uva-irlab/canard_quretec
    For more information on using custom datasets, please visit: https://github.com/deepset-ai/FARM/blob/master/tutorials/2_Build_a_processor_for_your_own_dataset.ipynb
    """

    def __init__(self, tokenizer: BertTokenizer = None, max_seq_len=300):
        """
        :param max_seq_len. The original authors of QuReTec have provided 300 as the max sequence length.
        """
        # Always log this, so users have a log of the settings of their experiments
        logger.info(f"{self.__class__.__name__} with max_seq_len={max_seq_len}")

        super(QuretecProcessor, self).__init__(tokenizer=tokenizer,
                                               max_seq_len=max_seq_len,
                                               # To set the args train_filename, dev_filename, test_filename and
                                               # data_dir, please use self.set_dataset
                                               train_filename=None,
                                               dev_filename=None,
                                               test_filename=None,
                                               data_dir=None,
                                               dev_split=0)
        self.add_task(name="ner",
                      metric="f1_micro",
                      label_list=self.get_labels(),
                      label_name="label"  # The label tensor name without "_ids" at the end
                      )
        self.label_to_id = self.label2id()
        self.id_to_label = self.id2label()

    def set_dataset(self,
                    dataset_name: Optional[str] = 'uva-irlab/canard_quretec',
                    split: dict=None
                    ):
        """
        Transfer a dataset from HuggingFace into the Processor. If you want to use your own dataset, simply set
        the self.data_dir, self.datasets, self.train_filename, self.test_filename and/or self.dev_filename of the
        processor instance.

        @param dataset_name: The processor class was designed to work with the HuggingFace dataset
                             'uva-irlab/canard_quretec'. Can be 'None' if you are using it for evaluation.
        @param split: Split dict as accepted by datasets.load_dataset from HuggingFace datasets
        @return:
        """
        logger.info(f"Prepare {self.__class__.__name__} with dataset {dataset_name}")
        if split is None:
            split = {'train': 'train', 'test': 'test', 'validation': 'validation'}

        loaded_datasets = datasets.load_dataset(dataset_name, split=list(split.values()))
        self.data_dir = Path(os.path.dirname(loaded_datasets[0].cache_files[0]['filename']))
        self.datasets = {}

        keys = [k for k in split.keys()]
        for i in range(len(split)):
            k = keys[i]
            self.datasets[k] = loaded_datasets[i]
            if k == 'train':
                self.train_filename = Path(os.path.basename(self.datasets[k].cache_files[0]['filename']))
            elif k == 'test':
                self.test_filename = Path(os.path.basename(self.datasets[k].cache_files[0]['filename']))
            elif k == 'validation' or k == 'dev':
                self.dev_filename = Path(os.path.basename(self.datasets['validation'].cache_files[0]['filename']))

    @staticmethod
    def get_labels():
        return [
            '[PAD]',
            QuretecProcessor.labels['NOT_RELEVANT'],
            QuretecProcessor.labels['RELEVANT'],
            "[CLS]",
            "[SEP]"
        ]

    @staticmethod
    def label2id():
        return {label: i for i, label in enumerate(QuretecProcessor.get_labels())}

    @staticmethod
    def id2label():
        return {i: label for i, label in enumerate(QuretecProcessor.get_labels())}

    @staticmethod
    def pad_token_id():
        return QuretecProcessor.get_labels().index('[PAD]')

    def file_to_dicts(self, file: str) -> [dict]:
        if self.test_filename:
            test_filename_path = self.data_dir / self.test_filename
            if file == test_filename_path:
                return self.datasets['test']

        if self.train_filename:
            train_filename_path = self.data_dir / self.train_filename
            if file == train_filename_path:
                return self.datasets['train']

        if self.dev_filename:
            dev_filename_path = self.data_dir / self.dev_filename
            if file == dev_filename_path:
                return self.datasets['validation']

        raise ValueError(f'Please use the training file {train_filename_path}\n, test file {test_filename_path} or dev file {dev_filename_path}')

    def relevant_terms(self, history: str, gold_source: str) -> Tuple[List[str], List[str]]:
        """
        Use the gold source to label the words/terms in the the history as relevant, non-relevant.

        :param history:
        :param gold_source:
        """
        exp = r"[\w|\[SEP\]|\[CLS\]|\[PAD\]]+|[\"\[\].,!?:;\-\(\)]|'s"
        word_list = re.findall(exp, history)
        gold_list = re.findall(exp, gold_source)
        gold_lemmas = set([get_spacy_parsed_word(w).lemma_ for w in gold_list])
        label_list = []

        for w in word_list:
            if len(gold_lemmas) > 0:
                if get_spacy_parsed_word(w).lemma_ in gold_lemmas:
                    label_list.append(QuretecProcessor.labels['RELEVANT'])
                else:
                    label_list.append(QuretecProcessor.labels['NOT_RELEVANT'])
            elif w == '[SEP]' or w == '[CLS]':
                label_list.append(w)
            else:
                label_list.append(QuretecProcessor.labels['NOT_RELEVANT'])

        return word_list, label_list

    def __combine_history_and_question(self, history: str, question: str):
        # The authors of QuReTec by Voskarides et al. decided to separate the history and the current question
        # with a [SEP] token
        return f"{history} {self.tokenizer.sep_token} {question}"

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> [Sample]:
        """
        :param dictionary

        """
        if 'cur_question' not in dictionary and 'query' not in dictionary:
            raise KeyError(f'Please provide a `query` or `cur_question` key for the dict: {dictionary}')
        question = dictionary.get('cur_question', dictionary.get('query')).lower()

        if 'prev_questions' not in dictionary and 'history' not in dictionary:
            raise KeyError(f'Please provide a `history` or `prev_questions` key for the dict: {dictionary}')

        history = dictionary.get('prev_questions', dictionary.get('history'))
        history = " ".join(history) if isinstance(history, list) else history
        history = history.lower()

        tokenized_text = self.__combine_history_and_question(history=history, question=question)

        # The HuggingFace dataset uva-irlab/canard_quretec is preprocessed by Voskarides et al. and contains a
        # word and label list.
        if 'bert_ner_overlap' in dictionary:
            word_list = dictionary.get('bert_ner_overlap')[0]
            label_list = dictionary.get('bert_ner_overlap')[1]
        else:
            gold_terms = dictionary.get('gold_terms', "")  # type: str
            word_list, label_list = self.relevant_terms(history=tokenized_text, gold_source=gold_terms)

        tokenized = self._quretec_tokenize_with_metadata(words_list=word_list, labellist=label_list)

        if len(tokenized["tokens"]) == 0:
            logger.warning(f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {tokenized_text}")

        return [
            Sample(id=QuretecProcessor.__get_dictionary_id(dictionary),
                   clear_text={
                       QuretecProcessor.label_name_key: question,
                       'tokenized_text': tokenized_text,
                   },
                   tokenized=tokenized)
        ]

    @staticmethod
    def __get_dictionary_id(d):
        return d['id'] if 'id' in d else 'NO-ID'

    def _quretec_tokenize_with_metadata(self, words_list: List[str], labellist: List[str]):
        """

        :param: text
            The entire text without a initial [CLS] and without closing [SEP] token.
            E.g. "Who are you? [SEP] I am your father"
        """
        tokens, labels, valid, label_mask = [], [], [], []

        if len(words_list) != len(labellist):
            raise ValueError(f"The word list (n={len(words_list)}) should be just as long as label list (n={len(labellist)})")

        for i, word in enumerate(words_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)

        # truncate lists to match short sequence
        if len(tokens) >= self.max_seq_len - 1:
            # Minus two, because CLS will be prepended and a SEP token will be appended
            tokens = tokens[0:(self.max_seq_len - 2)]
            labels = labels[0:(self.max_seq_len - 2)]
            valid = valid[0:(self.max_seq_len - 2)]
            label_mask = label_mask[0:(self.max_seq_len - 2)]

        return {
            "tokens": tokens,
            "labels": labels,
            "label_mask": label_mask,
            'valid': valid,
        }

    @staticmethod
    def to_haystack_label(d: dict, origin: str = '') -> Label:
        """
        Convert an entry from CANARD into a Haystack Label
        """
        doc_id = QuretecProcessor.__get_dictionary_id(d)
        return Label(
            question=d['cur_question'],
            answer=d['answer_text_with_window'],
            is_correct_answer=True,
            is_correct_document=True,
            origin=origin,
            document_id=doc_id,
            id=doc_id,
            meta=d
        )

    def _sample_to_features(self, sample: Sample) -> List[dict]:
        """
        convert Sample into features for a PyTorch model
        """

        label_mask = sample.tokenized['label_mask']
        labels = sample.tokenized['labels']
        tokens = sample.tokenized['tokens']
        valid = sample.tokenized['valid']

        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token

        # Prepend CLS token to the start
        segment_ids = [0]
        label_ids = [self.label_to_id[cls_token]]
        ntokens = [cls_token]

        valid.insert(0, 1)
        label_mask.insert(0, 1)

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(self.label_to_id[labels[i]])
        ntokens.append(sep_token)
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(self.label_to_id[sep_token])
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)

        input_mask = [1] * len(input_ids)

        # mask out labels for current turn.
        cur_turn_index = label_ids.index(self.label_to_id[sep_token])

        label_mask = [1] * cur_turn_index + [0] * (len(label_ids) - cur_turn_index)

        # Pad the features
        while len(input_ids) < self.max_seq_len:
            input_ids.append(self.tokenizer.pad_token_id)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)

        assert len(label_ids) == len(label_mask),\
            f"label_ids has a different length than label_mask. label_ids={label_ids}, label_mask={label_mask}"
        while len(label_ids) < self.max_seq_len:
            label_ids.append(0)
            label_mask.append(0)

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(label_ids) == self.max_seq_len
        assert len(valid) == self.max_seq_len
        assert len(label_mask) == self.max_seq_len

        return [{
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "label_ids": label_ids,
            "valid_ids": valid,
            "label_attention_mask": label_mask,
        }]

    def predictions_to_terms(self, batch: dict, predictions: List[List[str]]):
        input_ids = batch['input_ids']
        mask = batch['label_attention_mask']
        valid = batch['valid_ids']  # valid_ids has ones for each start of a word.

        terms = []  # type: List[List[str]]
        positions = []
        for b in range(len(input_ids)):
            relevant_filter = mask[b] == 1  # type: List[bool]
            relevant_inputs = input_ids[b][relevant_filter].tolist()
            assert len(relevant_inputs) == len(predictions[b])

            terms.append([])
            positions.append([])
            p = 0
            current = self.tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
            for t, token in enumerate(current):
                if relevant_filter[t]:
                    if predictions[b][p] == self.labels['RELEVANT'] and valid[b][t] == 1:
                        terms[b].append(token)

                        # BERT break up some words, e.g. 'Neverending' becomes 'never' and '##ending'.
                        # Merge those word parts together again by looping forward.
                        for f in range(t+1, len(current)):
                            next_term = current[f]  # type: str
                            if next_term.startswith('##'):
                                terms[b][-1] += next_term.replace('##', '')
                            else:
                                break
                    p += 1
        return terms

