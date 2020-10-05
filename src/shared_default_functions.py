from typing import List


question_close_token = '<CLS>'  # From BERT without surrounding spaces
question_separator_token = '<SEP>'  # From BERT without surrounding spaces

def orconvqa_question_formatter(questions: List[str], prepend_initial_question: bool, history_window: int):
    """
    In the paper Open-retrieval Conversational Question Answering, the question string provided to
    the retriever/reader is formatted by concatenating the current question with historical question
    including BERT tokens <CLS> and <SEP>
    """
    # Plus one, because the new question should have been added to the list of questions. So if the user wants a
    # history of two, then the number of question needed are two + the current question, which makes 3 in total.
    h = history_window + 1
    questions_to_take_into_account = questions[h:]

    # optionally prepend initial question
    initial_question_was_included = len(questions_to_take_into_account) < len(questions)
    if prepend_initial_question and initial_question_was_included:
        questions_to_take_into_account = [questions[0]] + questions_to_take_into_account

    token_with_spaces = ' ' + question_separator_token + ' '
    return '{0} {1} {2}'.format(question_close_token,
                                token_with_spaces.join(questions_to_take_into_account),
                                question_separator_token)