import os
import sys
import requests
import gzip
import shutil

from enums import lang

_server_url = "https://"
short_paths = {
    "all_blocks": "datasets/predefined/orconvqa/documents/all_blocks.txt",
    "dev_blocks": "datasets/predefined/orconvqa/documents/dev_blocks.txt"
}
files_on_server = {
    "datasets/predefined/orconvqa/documents/all_blocks.txt": _server_url+"datasets/predefined/orconvqa/documents/all_blocks.txt.gz",
    "datasets/predefined/orconvqa/documents/dev_blocks.txt": _server_url+"datasets/predefined/orconvqa/documents/dev_blocks.txt.gz",
}

def get_full_path(short_or_full_path):
    if short_or_full_path in short_paths:
        return short_paths[short_or_full_path]
    return short_or_full_path

def exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok

def download_file(url, filename):
    if not exists(url):
        raise FileNotFoundError(f"Was looking for {url} on the {lang.framework} server, but it could not be found. Please contact the authors.")
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

def unzip_file(zip_file_path, destination_file_path, delete_archive:bool = True):
    with gzip.open(zip_file_path, 'rb') as f_in:
        with open(destination_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if delete_archive:
        os.remove(zip_file_path)

def check_and_rewrite_file_path(path):
    """
    Check if the provided path is a short-hand path for a full path. In addition, if the file does not exists, download
    it from the server.
    :param path:
    :return: The full path of the file
    """
    full_path = get_full_path(path)

    if not os.path.isfile(full_path):
        if full_path in files_on_server:

            # Get the file from the server
            service_file_url = files_on_server[full_path]
            download_file(service_file_url, full_path)

            # Optionally, unzip the archive
            if service_file_url.endswith(".gz"):
                unzip_file(full_path+".gz", full_path, delete_archive=True)
        else:
            raise FileNotFoundError(f"The file {full_path} does not exist locally or on the {lang.framework} server.")

    return full_path