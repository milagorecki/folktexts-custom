"""Module to map risk-estimation questions to different prompting techniques.

e.g.,
- multiple-choice Q&A vs direct numeric Q&A;
- zero-shot vs few-shot vs CoT;
"""

from __future__ import annotations

import logging

from copy import deepcopy
import inspect

import pandas as pd
from transformers import AutoTokenizer

from .dataset import Dataset
from .qa_interface import QAInterface
from .task import TaskMetadata

from folktexts.acs.acs_columns_alt import simplified_value_maps
from folktexts.acs._utils import reset_cache
from folktexts.acs import ACS_TASK_DESCRIPTION, ACS_TASK_DESCRIPTION_DEFAULTS
from folktexts.ts import TABLESHIFT_TASK_DESCRIPTION, TABLESHIFT_TASK_DESCRIPTION_DEFAULTS


SYSTEM_PROMPT = """\
You are a helpful assistant. You answer multiple-choice questions based on the information provided.
"""

# ACS_TASK_DESCRIPTION = """\
# The following data corresponds to a survey respondent. \
# The survey was conducted among US residents in 2018. \
# Please answer the question based on the information provided. \
# The data provided is enough to reach an approximate answer.
# """

# ACS_FEW_SHOT_TASK_DESCRIPTION = """\
# The following data corresponds to different survey respondents. \
# The survey was conducted among US residents in 2018. \
# Please answer each question based on the information provided. \
# The data provided is enough to reach an approximate answer for each person.
# """

ANTHROPIC_CHAT_PROMPT = """If had to select one of the options, my answer would be"""
GEMMA_CHAT_PROMPT = """The provided information suggests that the answer is"""

_valid_keys_cache = {}


class PromptVariation:
    def __init__(self, description, task):
        self.description = description
        self.task = deepcopy(task)

    def __call__(self, row: pd.Series, **kwds):
        raise NotImplementedError
        return row


class VaryFormat(PromptVariation):
    def __init__(self, task, format: str = "bullet"):
        description = "Vary the format of the prompt, by default 'bullet' is used."
        super().__init__(description, task)
        assert format in [
            "bullet",
            "comma",
            "text",
            "textbullet",
        ], "Currently only 'bullet', 'comma', 'text', 'textbullet' implemented."
        self.format = format
        logging.warning(
            "VaryFormat should be applied after the value mapping and adding the connector."
        )

    def __call__(self, row: pd.Series, **kwds) -> pd.Series:
        for col, val in row.items():
            if col in self.task.features:
                if self.format == "bullet":
                    row[col] = f"- {val}\n"
                elif self.format == "comma":
                    row[col] = f"{val}, "
                elif self.format == "text":
                    row[col] = f"The {val}. "
                elif self.format == "textbullet":
                    row[col] = f"- The {val}.\n"
        return row


class VaryConnector(PromptVariation):
    def __init__(self, task, connector: str = "is"):
        description = "Vary the connector word or symbol used between feature name and value, by default 'is' is used."
        super().__init__(description, task)
        if connector == ':':
            self.connector = f"{connector} "
        else:
            self.connector = f" {connector} "
        logging.warning("VaryConnector should be applied after value mapping has already been applied.")

    def __call__(self, row: pd.Series, **kwds) -> pd.Series:
        for col, val in row.items():
            if col in self.task.features:
                row[col] = (
                    f"{self.task.cols_to_text[col].short_description}{self.connector}{val}"
                )
        return row


class VaryValueMap(PromptVariation):
    def __init__(self, task, granularity="original"):
        description = "Vary the granulariy of the feature map, default: original (higher granularity) value map."
        super().__init__(description, task)
        assert granularity in ["original", "low"]
        self.granularity = granularity
        if self.granularity == 'low':
            # empty cache of previously parsed pums codes to overwrite with new postprocessing for value map
            reset_cache()

    def __call__(self, row: pd.Series, **kwds) -> pd.Series:
        for col, val in row.items():
            if col in self.task.features:
                # overwrite value map if wanted
                if self.granularity == "low" and col in simplified_value_maps.keys():
                    self.task.cols_to_text[col]._value_map = simplified_value_maps[col]
                # apply value map
                row[col] = self.task.cols_to_text[col][val]
        return row


class VaryFeatureOrder(PromptVariation):
    def __init__(self, task, order: list | str = None):
        description = "Vary the order of the features."
        super().__init__(description, task)
        if order:
            if isinstance(order, str):
                order = list(order.split(','))
            else: 
                assert isinstance(order, list), "Expected order provided as list"  # mutable
            assert set(order) == set(self.task.features), 'Provide a complete ordering of all features'
        self.order = order

    def __call__(self, row: pd.Series, **kwds):
        if not self.order:
            return row
        feature_set = set(self.task.features)  # for fast lookup
        # Build a mapping from original position → whether to use reordered feature or keep original
        reordered_iter = iter(self.order)
        order_all = [
            next(reordered_iter) if col in feature_set else col
            for col in row.index
        ]
        return row[order_all]


class VaryPrefix(PromptVariation):

    def __init__(
        self,
        task: TaskMetadata,
        add_task_description: bool = True,
        custom_prompt_prefix: str = None,
        task_description: str = None
    ):
        description = "Vary the prefix printed before the prompt, by default the task description is printed."
        super().__init__(description, task)
        if add_task_description:
            assert task_description is not None, 'Provide a task description to add.'
        self.task_description = task_description
        self.add_task_description = add_task_description
        self.custom_prefix = custom_prompt_prefix

    def __call__(
        self,
        row: pd.Series,
        **kwds,
    ) -> pd.Series:
        if self.add_task_description:
            if self.custom_prefix:
                if self.custom_prefix[-1] != "\n":
                    self.custom_prefix = self.custom_prefix + "\n"
                prefix = self.task_description + self.custom_prefix
            else:
                prefix = self.task_description
            # add new line after prefix
            row = pd.Series(
                {
                    "_PREFIX": prefix + "\nInformation:\n",
                    **{index: val for index, val in row.items()},
                }
            )
        else:
            row = pd.Series(
                {
                    '_PREFIX': "Information:\n",
                    **{index: val for index, val in row.items()},
                }
            )
        return row


class VarySuffix(PromptVariation):
    def __init__(
        self, task, question: QAInterface = None, custom_prompt_suffix: str = None
    ):
        description = "Vary the suffix, in particular the question."
        super().__init__(description, task)
        self.question = question if question else task.question
        self.custom_suffix = custom_prompt_suffix

    def __call__(
        self,
        row,
        **kwds,
    ):
        row = pd.Series(
            {
                **{index: val for index, val in row.items()},
                "_SUFFIX": f"\n{self.question.get_question_prompt()}{self.custom_suffix if self.custom_suffix else ''}",
            }
        )
        return row


def get_valid_keys(cls):
    if cls not in _valid_keys_cache:
        params = inspect.signature(cls.__init__).parameters
        _valid_keys_cache[cls] = set(params) - {'self', 'args', 'kwargs'}
    return _valid_keys_cache[cls]


_building_blocks_cache = []


def reset_building_block_cache():
    global _building_blocks_cache
    _building_blocks_cache.clear()


def encode_row_prompt(
    row: pd.Series,
    task: TaskMetadata,
    question: QAInterface = None,
    custom_prompt_prefix: str = None,
    add_task_description: bool = True,
    custom_prompt_suffix: str = None,
    prompt_variation: dict | None = None,
) -> str:
    """Encode a question regarding a given row."""
    global _building_blocks_cache
    # ensure only feature defined for the task are used
    row = row[task.features]
    # Get the question to ask
    question = question or task.question

    def use_variation(cls, default_kwargs):
        if prompt_variation is None:
            return cls(task=task, **default_kwargs)
        # get parameters from class __init__
        valid_keys = get_valid_keys(cls)

        # merge and overwrite defaults with variations
        merged = {**default_kwargs, **prompt_variation}

        # filter out keys not in class __init__
        filtered_kwargs = {k: v for k, v in merged.items() if k in valid_keys}

        return cls(task=task, **filtered_kwargs)

    if len(_building_blocks_cache) == 0:
        _building_blocks_cache = [
            use_variation(
                VaryPrefix,
                {
                    "add_task_description": add_task_description,
                    "custom_prompt_prefix": custom_prompt_prefix,
                    "task_description": (
                        ACS_TASK_DESCRIPTION.substitute(ACS_TASK_DESCRIPTION_DEFAULTS)
                        if task.name.startswith("ACS")
                        else TABLESHIFT_TASK_DESCRIPTION.substitute(TABLESHIFT_TASK_DESCRIPTION_DEFAULTS)
                        )
                },
            ),
            use_variation(
                VarySuffix,
                {
                    "question": question,
                    "custom_prompt_suffix": custom_prompt_suffix,
                },
            ),
            use_variation(VaryFeatureOrder, {"order": None}),
            # order of value map, connector and format should not be changed
            use_variation(VaryValueMap, {"granularity": "original"}),
            use_variation(VaryConnector, {"connector": "is"}),
            use_variation(VaryFormat, {"format": "textbullet"}),
        ]

    for fun in _building_blocks_cache:
        row = fun(row)
    return "".join(row.values)


def encode_row_prompt_few_shot(
    row: pd.Series,
    task: TaskMetadata,
    dataset: Dataset,
    n_shots: int,
    question: QAInterface = None,
    reuse_examples: bool = False,
    class_balancing: bool = False,
    custom_prompt_prefix: str = None,
    prompt_variation: dict | None = None,
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
    prompt = ""  # ACS_FEW_SHOT_TASK_DESCRIPTION + "\n"

    # Get the question to ask
    question = question or task.question

    # Add `n` example rows with respective labels
    for i in range(n_shots):
        prompt += (
            encode_row_prompt(
                X_examples.iloc[i],
                task=task,
                add_task_description=(i == 0),
                custom_prompt_prefix=custom_prompt_prefix,
                prompt_variation={
                    **(prompt_variation or {}),
                    "task_description": (
                        ACS_TASK_DESCRIPTION.substitute(
                            {
                                **ACS_TASK_DESCRIPTION_DEFAULTS,
                                "respondent": "different survey respondents",
                                "suffix": " for each person",
                            }
                        )
                        if task.name.startswith("ACS")
                        else TABLESHIFT_TASK_DESCRIPTION.substitute(
                            {
                                **TABLESHIFT_TASK_DESCRIPTION_DEFAULTS,
                                "respondent": "different survey respondents",
                                "suffix": " for each person",
                            }
                        )
                    ),
                    "custom_prompt_suffix": f" {question.get_answer_key_from_value(y_examples.iloc[i])}\n\n"
                },
            )
        )
        # only add task description before first examples
        if i == 0:
            reset_building_block_cache()

    # Add the target row without its label
    prompt += encode_row_prompt(
        row,
        task=task,
        add_task_description=False,
        custom_prompt_prefix=custom_prompt_prefix,
        question=question,
        prompt_variation=prompt_variation,
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
        (SYSTEM_PROMPT + encode_row_prompt(row, task, question=question)),
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
    conversation = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )

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
