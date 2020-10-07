import re
import os
from haystack.preprocessor.utils import fetch_archive_from_http


default_data_set_mapping = {  # paths WITHOUT trailing slash
    'orconvqa-dev': './predefined/orconvqa/dev.txt',
    'orconvqa-test': './predefined/orconvqa/test.txt'
}

"""
Collections suitable for Haystack's DocumentStore, In Haystack, DocumentStores expect Documents in a dictionary format.
Example:
document_store = ElasticsearchDocumentStore()
dicts = [
    {
        'text': DOCUMENT_TEXT_HERE,
        'meta': {'name': DOCUMENT_NAME, ...}
    }, ...
]
document_store.write_documents(dicts)
"""
default_collections = {
    ''
}


class DataSet:
    def __init__(self, output_dir: str = './data'):
        """

        :param output_dir:
        """
        self.__output_dir = output_dir

    def __is_url(self, url):
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(regex, url) is not None

    def __url_to_folder_name(self, url):

        for char in ['/', ':', '.', '\\', '&', '?']:
            url = url.replace(char, 'new')
        return url

    def get_dataset(self, url_or_path_or_predefined: str):
        if url_or_path_or_predefined in default_data_set_mapping.keys():
            # replace predefined string by its path
            url_or_path_or_predefined = default_data_set_mapping[url_or_path_or_predefined]

        data_set_path = self.__output_dir + '/' + url_or_path_or_predefined
        if os.path.isdir(data_set_path):  #
            return data_set_path

        if self.__is_url(url_or_path_or_predefined):  # TODO download and extract. Make sure it is in QUAC format
            url = url_or_path_or_predefined
            location = self.__output_dir + '/' + self.__url_to_folder_name(url)
            fetch_archive_from_http(url, output_dir=location)
            return location

        raise FileNotFoundError('Cannot verify path: '+data_set_path)



