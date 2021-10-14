import json, re, torch, logging
from typing import List
from pathlib import Path

from farm.data_handler.dataloader import NamedDataLoader
from farm.data_handler.processor import Processor
from farm.eval import Evaluator
from farm.modeling.prediction_head import TokenClassificationHead
from farm.train import Trainer, EarlyStopping
from torch import nn
from farm.modeling.tokenization import Tokenizer
from transformers import BertForTokenClassification, BertConfig
from farm.modeling.optimization import initialize_optimizer
from farm.data_handler.data_silo import DataSilo
from haystack import BaseComponent

__all__ = ['QuReTecTokenClassicationHead', 'QueryResolution', 'QueryResolutionModel']
logger = logging.getLogger(__name__)


def dict_to_string(d: dict):
    regex = '#(\W|\.)+#'
    params_strings = []
    for k,v in d.items():
        params_strings.append(f"{re.sub(regex, '_', str(k))}_{re.sub(regex, '_', str(v))}")
    return "_".join(params_strings)


class QuReTecTokenClassicationHead(TokenClassificationHead):
    def __init__(self, **kwargs):
        super(QuReTecTokenClassicationHead, self).__init__(**kwargs)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0,  # 0, because Voskarides does so
                                            reduction='none'  # none, because Haystack does so
                                            )


class QueryResolutionModel(BertForTokenClassification):
    model_binary_file_name = "query_resolution"

    def __init__(self,
                 config: BertConfig,
                 device: str,
                 bert_model: str = 'bert-base-uncased',
                 max_seq_len: int = 300,
                 ):
        super(QueryResolutionModel, self).__init__(config)
        self.bert_model = bert_model
        self.max_seq_len = max_seq_len
        self.config = config
        self._device = device
        self.prediction_heads = [QuReTecTokenClassicationHead(num_labels=config.num_labels)]
        logger.info(f"Initiated QueryResolution model device={device}, config={config}")

    def verify_vocab_size(self, vocab_size):
        pass

    def connect_heads_with_processor(self, tasks, require_labels=True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels)
        :return:
        """

        # Drop the next sentence prediction head if it does not appear in tasks. This is triggered by the interaction
        # setting the argument BertStyleLMProcessor(next_sent_pred=False)
        if "nextsentence" not in tasks:
            idx = None
            for i, ph in enumerate(self.prediction_heads):
                if ph.task_name == "nextsentence":
                    idx = i
            if idx is not None:
                logger.info(
                    "Removing the NextSentenceHead since next_sent_pred is set to False in the BertStyleLMProcessor")
                del self.prediction_heads[i]

        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task \'{head.task_name}\' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            if "RegressionHead" in str(type(head)):
                # This needs to be explicitly set because the regression label_list is being hijacked to store
                # the scaling factor and the mean
                num_labels = 1
            else:
                pass
            head.metric = tasks[head.task_name]["metric"]

    def logits_to_loss_per_head(self, logits, **kwargs):
        label_attention_mask = kwargs.get('label_attention_mask')
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head,
                                                  padding_mask=label_attention_mask,
                                                  initial_mask=None,
                                                  **kwargs))
        return all_losses

    def logits_to_loss(self, logits, global_step=None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.
        """
        loss_per_head = self.logits_to_loss_per_head(logits, **kwargs)
        return sum(loss_per_head)

    def logits_to_preds(self, logits, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: logits can vary in shape and type, depending on task
        :return: A list of all predictions from all prediction heads
        """
        label_attention_mask = kwargs.get('label_attention_mask')
        all_preds = []
        tensors_copy = label_attention_mask.detach()
        for t in tensors_copy:
            t[0] = 0  # mask CLS
        # collect preds from all heads
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=nn.functional.log_softmax(logits_for_head, dim=2),
                                         initial_mask=tensors_copy, **kwargs)
            all_preds.append(preds)
        return all_preds

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.
        :return: labels in the right format
        """
        label_attention_mask = kwargs.get('label_attention_mask')
        all_labels = []
        tensors_copy = label_attention_mask.detach()
        for t in tensors_copy:
            t[0] = 0
        for head in self.prediction_heads:
            labels = head.prepare_labels(initial_mask=tensors_copy, **kwargs)
            all_labels.append(labels)
        return all_labels

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None, **kwargs):
        all_logits = []
        for ph in self.prediction_heads:
            outputs = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)
            sequence_output = outputs[0]
            valid_output = self.transfer_valid_output(sequence_output=sequence_output, valid_ids=valid_ids)
            sequence_output = self.dropout(valid_output).to(self._device)
            all_logits.append(self.classifier(sequence_output).to(self._device))
        return all_logits

    def transfer_valid_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        return valid_output

    def save(self, save_dir: str, id_to_label: dict):
        """
        Save the model state_dict and its config file so that it can be loaded again.
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        logger.info("Saving model to folder: "+str(save_dir))
        model_to_save = self.module if hasattr(self, 'module') else self  # Only save the model it-self
        model_to_save.save_pretrained(Path(save_dir))
        self._save_config(save_dir, id_to_label=id_to_label)

    def _save_config(self, save_dir: str, id_to_label: dict):
        """
        Saves the config as a json file.
        :param save_dir: Path to save config to
        """
        model_config = self.config.to_dict()
        model_config['bert_model'] = self.bert_model
        model_config['max_seq_len'] = self.max_seq_len
        model_config['id2label'] = id_to_label
        output_config_file = Path(save_dir) / f"config.json"
        with open(output_config_file, "w") as file:
            json.dump(model_config, file)


class QueryResolution(BaseComponent):
    outgoing_edges = 1

    def __init__(self,
                 processor: Processor = None,
                 config: BertConfig = None,
                 bert_model: str = 'bert-large-uncased',
                 pretrained_model_path: str = None,
                 use_gpu: bool = True,
                 progress_bar: bool = True,
                 tokenizer_args: dict = {},
                 model_args: dict = {},
    ):
        self.set_config(
            config=config,
            bert_model=bert_model
        )
        """
        Query resolution for Session based pipeline runs. This component is based on the paper:
        Query Resolution for Conversational Search with Limited Supervision
        """
        # Directly store some arguments as instance properties
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar

        # Set derived instance properties
        self.tokenizer = Tokenizer.load(bert_model, **tokenizer_args)
        if use_gpu and torch.cuda.is_available():
            device = 'cuda'
            self.n_gpu = torch.cuda.device_count()
        else:
            device = 'cpu'
            self.n_gpu = 1

        if pretrained_model_path:
            self.model = QueryResolutionModel.from_pretrained(pretrained_model_path, device=device, config=config, **model_args)
        else:
            self.model = QueryResolutionModel(config=config, device=device, **model_args)
        self.device = torch.device(device)
        self.processor = processor

        # modify
        if self.processor is not None and not self.processor.tokenizer:
            self.processor.tokenizer = self.tokenizer
        self.model = self.model.to(self.device)

    def train(self,
              learning_rate: float = 5e-5,
              batch_size: int = 4,
              gradient_clipping: float = 1.0,
              optimizer_name: str = 'AdamW',
              evaluate_every: int = 10000,
              epsilon: float = 1e-8,
              n_epochs: int = 3,
              save_dir: str = "saved_models",
              grad_acc_steps: int = 1,
              weight_decay: float = 0.01,
              datasilo_args: dict = None,
              early_stopping: int = None,
              ):
        """
        :param: n_epochs: int
            Default value is 3, because Voskarides et al. their QuReTec model was trained in 3 epochs.
        """
        logger.info(f'Training {self.__class__.__name__} with batch_size={batch_size}, gradient_clipping={gradient_clipping}, '
                    f'epsilon={epsilon}, n_gpu={self.n_gpu}, grad_acc_steps={grad_acc_steps}, evaluate_every={evaluate_every}, '
                    f'early_stopping={early_stopping}')
        if datasilo_args is None:
            datasilo_args = { "caching": False }

        self.data_silo = DataSilo(processor=self.processor,
                                  batch_size=batch_size,
                                  distributed=False,
                                  max_processes=1,
                                  **datasilo_args)

        self.model, optimizer, lr_schedule = initialize_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            optimizer_opts={"name": optimizer_name,
                            "correct_bias": True,
                            "weight_decay": weight_decay,
                            "eps": epsilon,
                            "no_decay": ['bias', 'LayerNorm.weight']
                            },
            n_batches=len(self.data_silo.loaders["train"]),
            grad_acc_steps=grad_acc_steps,
            n_epochs=n_epochs,
            device=self.device,
        )

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            data_silo=self.data_silo,
            epochs=n_epochs,
            n_gpu=self.n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            grad_acc_steps=grad_acc_steps,
            device=self.device,
            max_grad_norm=1.0,
            early_stopping=EarlyStopping(metric="micro_f1", mode="max")
        )
        trainer.train()

        # TODO save the best performing model
        params_dict = {
            'learning_rate': learning_rate,
            'eps': epsilon,
            'weight_decay': weight_decay,
            'hidden_dropout_prob': self.model.config.hidden_dropout_prob
        }
        model_save_dir = Path(save_dir) / ("query_resolution_" + dict_to_string(params_dict))
        self.model.save(model_save_dir, id_to_label=self.processor.id_to_label)
        self.tokenizer.save_pretrained(save_directory=str(model_save_dir))

    def _get_results(self, batch, logits):
        """
        :param: logits
            Two dimensional logits. N x M where N is the number of samples in a batch and M is the tensor length of one
            sample.
        """
        logits = torch.argmax(nn.functional.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()

        label_map = self.model.config.id2label
        label_to_id_map = self.model.config.label2id
        y_true, y_pred, terms = [], [], []
        label_ids = batch['label_ids'].to('cpu').numpy()
        input_ids = batch['input_ids'].to('cpu').numpy()
        valid_ids = batch['valid_ids'].to('cpu').numpy()
        for i, label in enumerate(label_ids):
            temp_1, temp_2, temp_3 = [], [], []

            for j, m in enumerate(label):
                if j == 0:  # skip initial CLS
                    continue
                elif label_ids[i][j] == label_to_id_map['[SEP]']:  # break at the first [SEP] token
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    tmp = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    x_input_tokens = []
                    for jj in range(1, len(tmp)):  # skip initial CLS
                        token = tmp[jj]
                        if token == '[PAD]':
                            break
                        if valid_ids[i][jj] == 1:
                            x_input_tokens.append(token)
                        else:
                            x_input_tokens[-1] += token

                    # remove bert tokenization chars ## from tokens
                    terms.append([s.replace('##', '') for s in x_input_tokens])
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map.get(logits[i][j], 'O'))
                    temp_3.append(input_ids[i][j])

        return y_pred, y_true

    def _print_classifier_result(self,
                                 input_token_ids: torch.Tensor,
                                 target_tensor: torch.Tensor,
                                 output_tensor: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 verbose: bool = True):
        relevant_token_ids = {'output': [], 'target': [], 'input': []}

        # QuReTec is a binary classifier. The paper does say it uses a sigmoid layer, but it does not say how to get to
        # an output of zeros and ones. Rounding makes sense, i.e. splitting on 0.5
        rounded_output = output_tensor.float().round()
        for idx, token_id in enumerate(input_token_ids):
            token_id_int = int(token_id)
            if attention_mask[idx] == 1.0:
                relevant_token_ids['input'].append(token_id_int)
            if target_tensor[idx] == 1.0:
                relevant_token_ids['target'].append(token_id_int)
            if rounded_output[idx] == 1.0:
                relevant_token_ids['output'].append(token_id_int)

        if verbose:
            relevant_tokens = {k: self.tokenizer.convert_ids_to_tokens(ids=relevant_token_ids[k]) for k in relevant_token_ids}
            logger.info(f'\nGiven input: {" ".join(self.tokenizer.convert_ids_to_tokens(ids=input_token_ids)).replace(" [PAD]", "")}\n'
                        f'and attention: {str(self.tokenizer.convert_ids_to_tokens(ids=relevant_token_ids["input"]))}\n'
                        f'and target     {str(relevant_tokens["target"])}\n'
                        f'the result is: {str(relevant_tokens["output"])}')

    def eval(self,
             data_set: str = 'test',
             eval_report: bool = True,
             batch_size: int = 4,
             datasilo_args: dict = None):
        logger.info(f"QueryResolution eval on: device={self.device}")
        if datasilo_args is None:
            datasilo_args = {"caching": False}

        self.model = self.model.to(self.device)
        self.data_silo = DataSilo(processor=self.processor,
                                  batch_size=batch_size,
                                  distributed=False,
                                  max_processes=1,
                                  **datasilo_args)
        self.model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        evaluator = Evaluator(
            data_loader=self.data_silo.get_data_loader(data_set),
            tasks=self.processor.tasks,
            device=self.device,
            report=eval_report
        )
        results = evaluator.eval(self.model, return_preds_and_labels=True)
        evaluator.log_results(results, data_set, 0)

    def predictions_to_terms(self, batch: dict, predictions: List[List[str]]):
        relevant_inputs = [batch['input_ids'][i][batch['label_attention_mask'][i] == 1].tolist() for i in range(len(batch['input_ids']))]

        terms = []
        for b in range(len(relevant_inputs)):
            assert len(relevant_inputs[b]) == len(predictions[b])
            terms.append([])
            for i in range(len(predictions[b])):
                # TODO take valid ids into account
                if predictions[b][i] == 'REL':
                    terms[b].append(relevant_inputs[b][i])
        return [self.processor.tokenizer.convert_ids_to_tokens(terms[i]) for i in range(len(terms))]

    def run(self, query, **kwargs):
        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)
        dataset, tensor_names, problematic_samples = self.processor.dataset_from_dicts(dicts=[{'query': query, **kwargs}])
        data_loader = NamedDataLoader(dataset=dataset, batch_size=1, tensor_names=tensor_names)

        for step, batch in enumerate(data_loader):
            batch = {key: batch[key].to(self.device) for key in batch}
            with torch.no_grad():
                logits = self.model.forward(**batch)
                preds_per_head = self.model.logits_to_preds(logits=logits, **batch)
                predicted_terms = self.predictions_to_terms(batch=batch, predictions=preds_per_head[0])[0]

                # The predicted terms can contain duplicates. Only get the distinct values
                predicted_terms = list(set(predicted_terms))

            output = {
                **kwargs,
                'original_query': query,
                'query': f"{query} {' '.join(predicted_terms)}"
            }
        return output, "output_1"
