"""Module to map risk-estimation questions to different prompting techniques.

e.g.,
- multiple-choice Q&A vs direct numeric Q&A;
- zero-shot vs few-shot vs CoT;
"""
from __future__ import annotations

import logging

import pandas as pd
from transformers import AutoTokenizer

from .dataset import Dataset
from .qa_interface import QAInterface
from .task import TaskMetadata

SYSTEM_PROMPT = """\
You are a helpful assistant. You answer multiple-choice questions based on the information provided.
"""

ACS_TASK_DESCRIPTION = """\
The following data corresponds to a survey respondent. \
The survey was conducted among US residents in 2018. \
Please answer the question based on the information provided. \
The data provided is enough to reach an approximate answer.
"""

ACS_FEW_SHOT_TASK_DESCRIPTION = """\
The following data corresponds to different survey respondents. \
The survey was conducted among US residents in 2018. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

ANTHROPIC_CHAT_PROMPT = """If had to select one of the options, my answer would be"""
GEMMA_CHAT_PROMPT = """The provided information suggests that the answer is"""

def serialize_row(
    row: pd.Series,
    task: TaskMetadata,
    format: str = 'bullet',
    connector: str = "is",
    standardized_sentence=True,
    **kwargs
):
    logging.warn(f"{kwargs} are currently ignored.")
    if format == "bullet":
        return (
            "\n".join(
                [
                    "- "
                    + f"{task.cols_to_text[col].short_description} {connector} {task.cols_to_text[col].value_map(val)}"
                    for (col, val) in row.items()
                ]
            )
            + "\n"
        )
    elif format == "text":
        return (
            " ".join(
                [
                    (
                        f"The {task.cols_to_text[col].short_description} {connector} {task.cols_to_text[col].value_map(val)}."
                        if standardized_sentence
                        else task.cols_to_text[col].verbalize(val)
                    )
                    for (col, val) in row.items()
                ]
            )
            + "\n"
        )
    else:
        raise NotImplementedError(
            "Style not implemented, currently only 'bullet' list and 'text' are supported."
        )

def encode_row_prompt(
    row: pd.Series,
    task: TaskMetadata,
    question: QAInterface = None,
    custom_prompt_prefix: str = None,
    add_task_description: bool = True,
    custom_prompt_suffix: str = None,
    prompt_style: dict | None = None,
) -> str:
    """Encode a question regarding a given row."""
    if custom_prompt_prefix and custom_prompt_prefix[-1] != "\n":
        custom_prompt_prefix = custom_prompt_prefix + "\n"
    task_description = (ACS_TASK_DESCRIPTION if add_task_description else "") + (
        f"\n{custom_prompt_prefix}" if custom_prompt_prefix else ""
    )
    # ensure only feature defined for the task are used
    row = row[task.features]
    serialized_row = serialize_row(row, task, **prompt_style)
    # Get the question to ask
    question = question or task.question
    return (
        task_description
        + ("\n" if len(task_description) > 0 else "")
        + f"Information:\n{serialized_row}"
        + "\n"
        + f"{question.get_question_prompt()}"
        + (custom_prompt_suffix if custom_prompt_suffix else "")
    )


def encode_row_prompt_few_shot(
    row: pd.Series,
    task: TaskMetadata,
    dataset: Dataset,
    n_shots: int,
    question: QAInterface = None,
    reuse_examples: bool = False,
    class_balancing: bool = False,
    custom_prompt_prefix: str = None,
    prompt_style: dict | None = None,
) -> str:
    """Encode a question regarding a given row using few-shot prompting.

    Parameters
    ----------
    row : pd.Series
        The row that the question will be about.
    task : TaskMetadata
        The task that the row belongs to.
    n_shots : int, optional
        The number of example questions and answers to use before prompting
        about the given row, by default 3.
    reuse_examples : bool, optional
        Whether to reuse the same examples for consistency. By default will
        resample new examples each time (`reuse_examples=False`).

    Returns
    -------
    prompt : str
        The encoded few-shot prompt.
    """
    # Take `n_shots` random samples from the train set
    X_examples, y_examples = dataset.sample_n_train_examples(
        n_shots,
        reuse_examples=reuse_examples,
        class_balancing=class_balancing,
    )

    # Start with task description
    prompt = ACS_FEW_SHOT_TASK_DESCRIPTION + "\n"

    # Get the question to ask
    question = question or task.question

    # Add `n` example rows with respective labels
    for i in range(n_shots):
        prompt += (
            encode_row_prompt(
                X_examples.iloc[i],
                task=task,
                add_task_description=False,
                custom_prompt_prefix=custom_prompt_prefix,
                prompt_style = prompt_style,
            )
            + f" {question.get_answer_key_from_value(y_examples.iloc[i])}"
            + "\n\n"
        )

    # Add the target row without its label
    prompt += encode_row_prompt(
        row,
        task=task,
        add_task_description=False,
        custom_prompt_prefix=custom_prompt_prefix,
        question=question,
        prompt_style = prompt_style,
    )
    return prompt


def encode_row_prompt_chat(
    row: pd.Series,
    task: TaskMetadata,
    tokenizer: AutoTokenizer,
    question: QAInterface = None,
    **chat_template_kwargs,
) -> str:
    # TODO: implement two functions
    # - one for gemma-like models that are not compatible with system prompts
    # - and another for regular models compatible with system prompts
    logging.warning("NOTE :: Untested feature!!")

    return apply_chat_template(
        tokenizer,
        (
            SYSTEM_PROMPT
            + encode_row_prompt(row, task, question=question)
        ),
        **chat_template_kwargs,
    )


def apply_chat_template(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str = None,
    chat_prompt: str = ANTHROPIC_CHAT_PROMPT,
    **kwargs,
) -> str:
    # Add system prompt
    conversation = ([
        {"role": "system", "content": system_prompt}
    ] if system_prompt else [])

    # Add user prompt
    conversation.append({"role": "user", "content": user_prompt})

    # Using the Anthropic-style chat prompt
    conversation.append({"role": "assistant", "content": chat_prompt})

    # Default kwargs
    kwargs.setdefault("add_generation_prompt", False)

    # Apply prompt template
    filled_prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        **kwargs,
    )

    # Make sure no special tokens follow the `CHAT_PROMPT`;
    # > some models add a newline character and/or a <end_of_turn> token
    filled_prompt = filled_prompt[: len(chat_prompt) + filled_prompt.find(chat_prompt)]
    return filled_prompt
