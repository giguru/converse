import re
import logging
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import orjson
import json
import os

from farm.data_handler.utils import http_get

from haystack.file_converter.pdf import PDFToTextConverter
from haystack.file_converter.tika import TikaConverter
from converse.src.schema import Document, Label
from file_manager.manager import check_and_rewrite_file_path

logger = logging.getLogger(__name__)

def eval_data_from_file(filename: str) -> Tuple[List[Document], List[Label]]:
    """
    Read Documents + Labels from a SQuAD-style file.
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    :param filename: Path to file in SQuAD format
    :return: (List of Documents, List of Labels)
    """
    docs = []
    labels = []

    with open(filename, "r") as file:
        data = json.load(file)
        for document in data["data"]:
            # get all extra fields from document level (e.g. title)
            meta_doc = {k: v for k, v in document.items() if k not in ("paragraphs", "title")}
            for paragraph in document["paragraphs"]:
                cur_meta = {"name": document["title"]}
                # all other fields from paragraph level
                meta_paragraph = {k: v for k, v in paragraph.items() if k not in ("qas", "context")}
                cur_meta.update(meta_paragraph)
                # meta from parent document
                cur_meta.update(meta_doc)
                # Create Document
                cur_doc = Document(text=paragraph["context"], meta=cur_meta)
                docs.append(cur_doc)

                # Get Labels
                for qa in paragraph["qas"]:
                    if len(qa["answers"]) > 0:
                        for answer in qa["answers"]:
                            label = Label(
                                question=qa["question"],
                                answer=answer["text"],
                                is_correct_answer=True,
                                is_correct_document=True,
                                document_id=cur_doc.id,
                                offset_start_in_doc=answer["answer_start"],
                                no_answer=qa["is_impossible"],
                                origin="gold_label",
                            )
                            labels.append(label)
                    else:
                        label = Label(
                            question=qa["question"],
                            answer="",
                            is_correct_answer=True,
                            is_correct_document=True,
                            document_id=cur_doc.id,
                            offset_start_in_doc=0,
                            no_answer=qa["is_impossible"],
                            origin="gold_label",
                        )
                        labels.append(label)
        return docs, labels


def convert_files_to_dicts(dir_path: str, clean_func: Optional[Callable] = None, split_paragraphs: bool = False) -> \
List[dict]:
    """
    Convert all files(.txt, .pdf) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.

    :return: None
    """

    file_paths = [p for p in Path(dir_path).glob("**/*")]
    if ".pdf" in [p.suffix.lower() for p in file_paths]:
        pdf_converter = PDFToTextConverter()  # type: Optional[PDFToTextConverter]
    else:
        pdf_converter = None

    documents = []
    for path in file_paths:
        if path.suffix.lower() == ".txt":
            with open(path) as doc:
                text = doc.read()
        elif path.suffix.lower() == ".pdf" and pdf_converter:
            document = pdf_converter.convert(path)
            text = document["text"]
        else:
            raise Exception(f"Indexing of {path.suffix} files is not currently supported.")

        if clean_func:
            text = clean_func(text)

        if split_paragraphs:
            for para in text.split("\n\n"):
                if not para.strip():  # skip empty paragraphs
                    continue
                documents.append({"text": para, "meta": {"name": path.name}})
        else:
            documents.append({"text": text, "meta": {"name": path.name}})

    return documents


def tika_convert_files_to_dicts(
        dir_path: str,
        clean_func: Optional[Callable] = None,
        split_paragraphs: bool = False,
        merge_short: bool = True,
        merge_lowercase: bool = True
) -> List[dict]:
    """
    Convert all files(.txt, .pdf) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.

    :return: None
    """
    converter = TikaConverter(remove_header_footer=True)
    file_paths = [p for p in Path(dir_path).glob("**/*")]

    documents = []
    for path in file_paths:
        document = converter.convert(path)
        meta = document["meta"] or {}
        meta["name"] = path.name
        text = document["text"]
        pages = text.split("\f")

        if split_paragraphs:
            if pages:
                paras = pages[0].split("\n\n")
                # pop the last paragraph from the first page
                last_para = paras.pop(-1) if paras else ''
                for page in pages[1:]:
                    page_paras = page.split("\n\n")
                    # merge the last paragraph in previous page to the first paragraph in this page
                    if page_paras:
                        page_paras[0] = last_para + ' ' + page_paras[0]
                        last_para = page_paras.pop(-1)
                        paras += page_paras
                if last_para:
                    paras.append(last_para)
                if paras:
                    last_para = ''
                    for para in paras:
                        para = para.strip()
                        if not para: continue
                        # merge paragraphs to improve qa
                        # merge this paragraph if less than 10 characters or 2 words
                        # or this paragraph starts with a lower case and last paragraph does not end with a punctuation
                        if merge_short and len(para) < 10 or len(re.findall('\s+', para)) < 2 \
                                or merge_lowercase and para and para[0].islower() and last_para and last_para[
                            -1] not in '.?!"\'\]\)':
                            last_para += ' ' + para
                        else:
                            if last_para:
                                documents.append({"text": last_para, "meta": meta})
                            last_para = para
                    # don't forget the last one
                    if last_para:
                        documents.append({"text": last_para, "meta": meta})
        else:
            if clean_func:
                text = clean_func(text)
            documents.append({"text": text, "meta": meta})

    return documents


def fetch_archive_from_http(url: str, output_dir: str, proxies: Optional[dict] = None):
    """
    Fetch an archive (zip or tar.gz) from a url via http and extract content to an output directory.

    :param url: http address
    :type url: str
    :param output_dir: local path
    :type output_dir: str
    :param proxies: proxies details as required by requests library
    :type proxies: dict
    :return: bool if anything got fetched
    """
    # verify & prepare local directory
    path = Path(output_dir)
    if not path.exists():
        path.mkdir(parents=True)

    is_not_empty = len(list(Path(path).rglob("*"))) > 0
    if is_not_empty:
        logger.info(
            f"Found data stored in `{output_dir}`. Delete this first if you really want to fetch new data."
        )
        return False
    else:
        logger.info(f"Fetching from {url} to `{output_dir}`")

        # download & extract
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(url, temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible
            # extract
            if url[-4:] == ".zip":
                zip_archive = zipfile.ZipFile(temp_file.name)
                zip_archive.extractall(output_dir)
            elif url[-7:] == ".tar.gz":
                tar_archive = tarfile.open(temp_file.name)
                tar_archive.extractall(output_dir)
            # temp_file gets deleted here
        return True


def orconvqa_build_corpus(filename: str, limit_lines: int = 0) -> List[Document]:
    """
    :param filename - Name of json file containing the text blocks, each line should be in json format and contain
        at least a 'text' and 'id' entry
    :param limit_lines - For testing purposes, use only N lines instead of using the entire corpus.
        This because creating embeddings for the entire development corpus takes a lot of time. The corpus
        contains ca. 65000 lines. Make the limit 0 (=zero) to use everything.
    :return: List of Documents
    """

    docs = []
    with open(filename, 'r') as file:
        for idx, block in enumerate(file.readlines()):
            if 0 < limit_lines <= idx:
                # stop reading lines.
                break
            try:
                block = json.loads(block)
            except:
                raise ValueError(f'Error occurred reading json block on line {idx} of file: {filename}')

            cur_meta = {"name": block["title"]}
            # all other fields on block level f.e. id, aid, bid
            block_meta = {k: v for k, v in block.items() if k not in ['text', 'id']}
            cur_meta.update(block_meta)

            # Create Document
            cur_doc = Document(id=block['id'], text=block["text"], meta=cur_meta)
            docs.append(cur_doc)
    return docs


def orconvqa_read_files(filename: str, qrelsfile: str, buildCorpus: bool = False, corpusFile: str = ""):
    """
    Read and combine files from the OR-Quac dataset

    :param filename - Name of json file containing the questions, qids, answers and the question history
    :param qrelsfile - File in json format linking the qids to the doc ids of the golden passage (the passage where the answer can be found)
    :param buildCorpus - Whether or not the corpus should be build while parsing the questions file (requires corpus file)
    :param corpusFile - If buildCorpus is set, the function will load the documents from the corpus file as using the buildCorpus function

    :return: (List of Labels, None|List of Documents)
    """

    docs = None
    if buildCorpus:
        if not os.path.isfile(corpusFile):
            raise ValueError(f'Could not find corpus file: {corpusFile}')
        docs = orconvqa_build_corpus(corpusFile)

    with open(qrelsfile, 'r') as f:
        qrels = json.load(f)

    labels = []
    # filename = check_and_rewrite_file_path(filename)

    with open(filename, "r") as file:
        for question in file.readlines():
            question = orjson.loads(question)

            try:
                q_doc_rel = qrels[question['qid']]
            except:
                logger.warning(f'Qid {question["qid"]} not found in qrels, skipping question')

            if len(q_doc_rel.keys()) > 1:
                logger.warning('Found qrel with multiple docs, golden passage is unknown, assuming first')

            bla = q_doc_rel.keys()

            document_id = next(iter(q_doc_rel.keys()))

            label = Label(
                question=question["rewrite"],
                original_question=question['question'],
                answer=question["answer"]['text'],
                is_correct_answer=True,
                is_correct_document=True,
                # We do not do an extra check if the document id exists in the corpus, this may cause issues later
                document_id=document_id,
                offset_start_in_doc=question["answer"]['answer_start'],
                no_answer=question["answer"]['text'] == 'CANNOTANSWER',
                origin=filename,
                # TODO we do have some extra data here in the preprossed file -> pq['answer'], ['answer_start'] and ['bid']
                previous_questions_in_conversation=[pq['question'] for pq in question['history']]
            )
            labels.append(label)

    return labels, docs


def CoQA_read_file(filename: str) -> Tuple[List[Document], List[Label]]:
    """
    Read Documents + Labels from a CoQA style file
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    :param filename: Path to file in CoQA format
    :return: (List of Documents, List of Labels)
    """
    docs = []
    labels = []

    with open(filename, "r") as file:
        data = json.load(file)
        for document in data["data"]:
            # get all extra fields from document level (e.g. title)
            meta_doc = {k: v for k, v in document.items() if k not in ("questions", "answers")}
            cur_doc = Document(id=document['id'], text=document["story"], meta=meta_doc)

            docs.append(cur_doc)
            # Get Labels
            for q, a in zip(document["questions"], document['answers']):

                label = Label(
                    question=q["input_text"],
                    # TODO these are very short answers and may not allways match with the span_start
                    # The retrieved answer on span_text is longer and input_text is taken from that
                    answer=a['input_text'],
                    is_correct_answer=True,
                    is_correct_document=True,
                    # We do not do an extra check if the document id exists in the corpus, this may cause issues later
                    document_id=cur_doc.id,
                    offset_start_in_doc=a["span_start"],
                    origin=filename,
                    previous_questions_in_conversation=[pq['input_text'] for pq in document['questions'] if pq['turn_id'] < q['turn_id']]
                )
                labels.append(label)

    return docs, labels