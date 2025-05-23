"""A collection of instantiated ACS column objects and ACS tasks."""

from __future__ import annotations

from folktexts.col_to_text import ColumnToText
from folktexts.qa_interface import DirectNumericQA as _DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA as _MultipleChoiceQA

from . import brfss_columns

# Map of ACS column names to ColumnToText objects
brfss_columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in brfss_columns.__dict__.values()
    if isinstance(col_mapper, ColumnToText)
}

# Map of numeric ACS questions
brfss_numeric_qa_map: dict[str, object] = {
    question.column: question
    for question in brfss_columns.__dict__.values()
    if isinstance(question, _DirectNumericQA)
}

# Map of multiple-choice ACS questions
brfss_multiple_choice_qa_map: dict[str, object] = {
    question.column: question
    for question in brfss_columns.__dict__.values()
    if isinstance(question, _MultipleChoiceQA)
}

# ... include all multiple-choice questions defined in the column descriptions
brfss_multiple_choice_qa_map.update(
    {
        col_to_text.name: col_to_text.question
        for col_to_text in brfss_columns_map.values()
        if (isinstance(col_to_text, ColumnToText) and col_to_text._question is not None)
    }
)
