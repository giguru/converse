from typing import Union, List, Callable
from haystack.query_rewriting.base import BaseReformulator


class CustomFilterReformulator(BaseReformulator):
    def __init__(self, filter_func: Callable, separator: str, debug: bool = True):
        """
        This reformulator provides an interface to add a custom filter/function to combine the history and query into
        a context-independent query.

        :param filter_func
            Please provide a callable which can accept two arguments: query and the history. It should return a string
        :param separator:
            The string used to concatenate the different history elements and the current query.
        :param debug:
            Set True if the component needs to log its output.
        """
        super().__init__(debug=debug)
        self.separator = separator
        self._filter_func = filter_func

    def run_query(self, query: str, history: Union[str, List[str]], **kwargs):
        extended_query = self._filter_func(query=query, history=history)
        if not isinstance(extended_query, str):
            raise ValueError("The filter function of the CustomFilterReformulator should return a string.")

        return {
            **kwargs,
            'query': extended_query,
            'original_query': query,
        }, "output_1"


class ConcatenationReformulator(BaseReformulator):
    def __init__(self, separator: str, debug: bool = True):
        """
        This reformulator simply prepends all history to the query.

        :param separator:
            The string used to concatenate the different history elements and the current query.
        :param debug:
            Set True if the component needs to log its output.
        """
        super().__init__(debug=debug)
        self.separator = separator

    def run_query(self, query: str, history: Union[str, List[str]], **kwargs):
        sep = self.separator
        extended_query = sep.join(history) + sep + query
        return {
            **kwargs,
            'query': extended_query,
            'original_query': query,
        }, "output_1"


class PrependingReformulator(BaseReformulator):
    def __init__(self,
                 history_window: int,
                 separator: str,
                 debug: bool = True,
                 always_add_first: bool = False,
                 ):
        """
        This reformulator prepends a given history window to the

        :param history_window:
            The history window that will be prepended to the current query.
        :param separator:
            The string used to concatenate the different history elements and the current query.
        :param debug:
            Set True if the component needs to log its output.
        :param always_add_first:
            Set True if the component always needs to append the first element of the history.
        """
        super().__init__(debug=debug)
        self.always_add_first = always_add_first
        self.history_window = history_window
        self.separator = separator

    def run_query(self, query: str, history: Union[str, List[str]], **kwargs):
        sep = self.separator
        if len(history) <= self.history_window:
            extended_query = sep.join(history) + sep + query
        else:
            extended_query = sep.join(history[-self.history_window:]) + sep + query
            if self.always_add_first:
                extended_query = f"{history[0]}{sep}{extended_query}"

        return {
            **kwargs,
            'query': extended_query,
            'original_query': query,
        }, "output_1"
