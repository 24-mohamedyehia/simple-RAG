from enum import Enum

class ResponseSignal(Enum):

    INVALID_INPUT_QUESTION = "invalid_input_question"
    NO_VALID_ANSWER = "no_valid_answer"
    INVALID_ANSWER_EMPTY = "invalid_answer_empty"