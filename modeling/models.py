from transformers import BertConfig, BertTokenizer, AlbertConfig, AlbertTokenizer
from modeling.orconvqa import BertForOrconvqaGlobal, AlbertForRetrieverOnlyPositivePassage

predefined_model_sets = {
    'orconvqa_default_reader': '',
    'orconvqa_default_retriever': '',
}