from abc import abstractmethod
from functools import wraps
from time import perf_counter
from typing import Any, Union, List
import torch
from haystack import BaseComponent


class BaseReformulator(BaseComponent):
    query_count = 0
    reformulate_time = 0.0

    def __init__(self, use_gpu: bool = True, debug: bool = True):
        self.debug = debug
        self.log: List = []
        self.use_gpu = use_gpu

        if use_gpu and torch.cuda.is_available():
            device = 'cuda'
            self.n_gpu = torch.cuda.device_count()
        else:
            device = 'cpu'
            self.n_gpu = 1
        self.device = torch.device(device)

    def timing(self, fn, attr_name):
        """Wrapper method used to time functions. """

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if attr_name not in self.__dict__:
                self.__dict__[attr_name] = 0
            tic = perf_counter()
            ret = fn(*args, **kwargs)
            toc = perf_counter()
            self.__dict__[attr_name] += toc - tic
            return ret

        return wrapper

    @abstractmethod
    def run_query(self, query: str, history: Union[str, List[str]], **kwargs):
        raise NotImplementedError("Please implement this method in the extended class")
        pass

    def run(self, **kwargs: Any):
        self.query_count += 1
        run_query_timed = self.timing(self.run_query, "reformulate_time")
        output, stream = run_query_timed(**kwargs)
        if self.debug:
            self.log.append(output)
        return {**kwargs, **output}, stream

    def print_time(self):
        print("\nReformulator (Speed)")
        print("---------------")

        if not self.query_count:
            print("No querying performed via Reformulator.run()")
        else:
            print(f"Queries Performed: {self.query_count}")
            print(f"Query time: {self.reformulate_time}s")
            print(f"{self.reformulate_time / self.query_count} seconds per query")
        print("\n")

    def print(self):
        print("\nReformulator Log")
        print("---------------")
        for output in self.log:
            print({
                'query': output['query'],
                'qid': output['qid'] if 'qid' in output else '',
                'original_query': output['original_query'],
            })