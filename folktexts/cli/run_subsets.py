import os
import pandas as pd
from monoculture.analysis.utils import model_to_key
from monoculture.analysis.setup import LLM_MODELS


def main(models, tasks):
    df = pd.read_csv("./results/feature-importance.csv", index_col=0)
    for task in tasks:
        for model in models:
            model_key = model_to_key(model)
            if model_key in df.index:
                sorted_means = (
                    df.loc[model_key]
                    .filter(like="mean")
                    .sort_values(axis=0, ascending=False)
                )
                feature_by_importance = [
                    feature.split("_")[0] for feature in sorted_means.index
                ]
                # reduce features by at least 1 up to the 3 most important
                for cutoff in range(3, len(feature_by_importance)):
                    os.system(
                        f"python -m folktexts.cli.launch_experiments_htcondor --executable-path ./folktexts/cli/run_acs_benchmark.py --results-dir ../monoculture/results/folktexts/subsets  --task {task} --model {model} --use-feature-subset={','.join(feature_by_importance[:cutoff])}"
                    )
            else:
                print(f"{model_key} skipped.")


if __name__ == "__main__":
    models = LLM_MODELS
    tasks = ["ACSIncome"]
    main(models, tasks)
