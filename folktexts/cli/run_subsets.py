import os
import pandas as pd
from monoculture.analysis.utils import model_to_key
from monoculture.analysis.setup import LLM_MODELS, ACS_TASKS
import folktexts.acs.acs_tasks as acs_tasks
import numpy as np
from argparse import ArgumentParser
import logging


def setup_arg_parser() -> ArgumentParser:
    # Init parser
    parser = ArgumentParser(description="Run subset experiments.")
    parser.add_argument(
        "--subset-rel-size",
        type=float,
        help="[float] Relative size of subset to use",
        default=0.8,
        required=False,
    )

    parser.add_argument(
        "--task",
        type=str,
        help="[string] ACS task name to run experiments on - can provide multiple!",
        required=False,
        action="append",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="[string] Model name on huggingface hub - can provide multiple!",
        required=False,
        action="append",
    )

    return parser


def choose_subset_by_feature_importance(task: str, model_key: str, cutoff: int):
    df = pd.read_csv("./results/feature-importance.csv", index_col=0)
    if model_key in df.index:
        sorted_means = (
            df.loc[model_key].filter(like="mean").sort_values(axis=0, ascending=False)
        )
        feature_by_importance = [
            feature.split("_")[0] for feature in sorted_means.index
        ]
        # reduce features by at least 1 up to the 3 most important
        for cutoff in range(3, len(feature_by_importance)):
            os.system(
                f"python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_acs_benchmark.py --results-dir ../monoculture/results/folktexts/subsets  --task {task} --model {model_key} --use-feature-subset={','.join(feature_by_importance[:cutoff])}"
            )
    else:
        print(f"{model_key} not found, skipping model.")


def choose_random_subset(task, subset_rel_size: float):
    task_features = {
        "ACSIncome": acs_tasks.acs_income_task.features,
        "ACSEmployment": acs_tasks.acs_employment_task.features,
        "ACSPublicCoverage": acs_tasks.acs_public_coverage_task.features,
        "ACSMobility": acs_tasks.acs_mobility_task.features,
        "ACSTravelTime": acs_tasks.acs_travel_time_task.features,
    }
    feature_list = task_features[task]
    num_features = len(feature_list)

    # get random subset of features
    rnd_indices = np.random.randint(
        0, num_features, size=int(np.ceil(num_features * subset_rel_size))
    )
    subset = np.array(feature_list)[rnd_indices].tolist()

    return subset


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    tasks = args.task or ACS_TASKS
    models = args.model or LLM_MODELS
    subset_rel_size = args.subset_rel_size

    for task in tasks:
        subset = choose_random_subset(task, subset_rel_size=subset_rel_size)
        for model in models:
            model_key = model_to_key(model)
            logging.info(f"Running {model_key} on {task} with subset {subset}")
            os.system(
                f"python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_acs_benchmark.py --results-dir ../monoculture/results/folktexts/subsets/{len(subset)}  --task {task} --model {model_key} --use-feature-subset={','.join(subset)}"
            )


if __name__ == "__main__":
    main()
