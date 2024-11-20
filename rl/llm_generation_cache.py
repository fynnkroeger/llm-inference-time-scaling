from collections import defaultdict
from typing import Optional, Annotated
from human_eval.data import write_jsonl, read_problems
from dataclasses import dataclass
import json
from rl.runner import evaluate_only_functional_correctness
from pathlib import Path
import random
import pickle


@dataclass
class TaskCompletion:
    task_id: str
    completion: str
    result: str
    passed: bool


@dataclass
class CachedResults:
    num_passed_samples: int = 0
    num_failed_samples: int = 0


out_dir = Path("outputs/human_eval_cache")
out_dir.mkdir(exist_ok=True, parents=True)


def get_prompts_from_task_ids(problems, task_ids: list[str]) -> list[str]:
    prompts = []
    for task_id in problems:
        if task_id in task_ids:
            prompt = problems[task_id]["prompt"]
            prompts.append(prompt)
    return prompts


def read_samples(file_path: str) -> list[TaskCompletion]:
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            data.append(TaskCompletion(**json.loads(line)))
    return data


def judge_problems(outputs: list, task_ids: list[str]) -> dict[str, CachedResults]:
    samples = []
    for task_id, output in zip(task_ids, outputs):
        for i in range(len(output.outputs)):
            generated_text = output.outputs[i].text
            samples.append(dict(task_id=task_id, completion=generated_text))

    generation_file_name = str(out_dir) + "/" + f"samples.jsonl"
    evaluated_generations_file_name = generation_file_name + "_results.jsonl"

    write_jsonl(generation_file_name, samples)
    evaluate_only_functional_correctness(generation_file_name, n_workers=64)

    results = read_samples(evaluated_generations_file_name)

    problems = defaultdict(CachedResults)
    for result in results:
        if result.passed:
            problems[result.task_id].num_passed_samples += 1
        else:
            problems[result.task_id].num_failed_samples += 1

    return problems


from os import path


class HumanEvalLLMGenerationCache:
    SAVE_PATH = path.join(path.dirname(__file__), ".human_eval_model_cache.bin")

    def __init__(
        self,
        cache: Optional[
            dict[
                str,
                dict[
                    Annotated[str, "task_id"],
                    dict[Annotated[tuple, "sampling_params"], CachedResults],
                ],
            ]
        ] = None,
    ):
        self.cache = cache if cache is not None else {}
        self.model = None
        self.llm = None
        self.human_eval_problems = read_problems()

    @classmethod
    def from_disk(cls, save_path: Optional[str] = None):
        cache_path = save_path or cls.SAVE_PATH
        if path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            return cls(cache)
        else:
            return cls()

    def use_model(self, model: str):
        if model != self.model:
            self.model = model
            if self.model not in self.cache:
                self.cache[self.model] = {}
            # Setting this to None meaning that is needs to be newly loaded
            self.llm = None

    def save(self):
        with open(self.SAVE_PATH, "wb") as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def tranform_sampling_parameters_into_dict_key(
        self, sampling_parameters: dict, ignored_parameters: list = []
    ) -> tuple:
        HASHED_PARAMETERS = ["temperature", "top_p", "max_tokens"]
        for param in sampling_parameters.keys():
            assert (
                param in HASHED_PARAMETERS or param in ignored_parameters
            ), f"{param} must either be hashed or specified as ignored!"

        hashable_tuple = (
            (param, sampling_parameters[param])
            for param in HASHED_PARAMETERS
            if param in sampling_parameters
        )
        return tuple(hashable_tuple)

    def _newly_generate(
        self, task_ids: list[str], sampling_parameters: dict, n_samples: int
    ):
        assert self.model is not None, "Model parameter must first be set"
        if self.llm is None:
            from vllm import LLM, SamplingParams  # type: ignore

            self.llm = LLM(model=self.model)

        if (
            "temperature" in sampling_parameters
            and sampling_parameters["temperature"] == 0
        ):
            n_samples = 1

        prompts = get_prompts_from_task_ids(self.human_eval_problems, task_ids)
        sampling_params = SamplingParams(**(sampling_parameters | {"n": n_samples}))
        outputs = self.llm.generate(prompts, sampling_params)
        new_cache_results = judge_problems(outputs, task_ids)

        for task_id, result in new_cache_results.items():
            if task_id not in self.cache[self.model]:
                self.cache[self.model][task_id] = {}
            self.cache[self.model][task_id][
                self.tranform_sampling_parameters_into_dict_key(sampling_parameters)
            ] = result

        self.save()

    def generate(
        self,
        task_ids: list[str],
        sampling_parameters: dict,
        use_expected_value: bool = False,
    ) -> dict[str, bool]:
        assert self.model is not None, "Model parameter must first be set"

        if "n" in sampling_parameters:
            n_samples = sampling_parameters["n"]
            del sampling_parameters["n"]
        else:
            n_samples = 1

        required_existing_samples = max(10, 2 * n_samples)
        non_cached_task_ids = []
        model_specific_cache = self.cache[self.model]

        for task_id in task_ids:
            tuple_sampling_paramters = self.tranform_sampling_parameters_into_dict_key(
                sampling_parameters
            )
            if (
                task_id not in model_specific_cache
                or tuple_sampling_paramters not in model_specific_cache[task_id]
            ):
                if (
                    "temperature" in sampling_parameters
                    and sampling_parameters["temperature"] != 0
                ):
                    task_results = model_specific_cache[task_id][
                        tuple_sampling_paramters
                    ]
                    num_already_exisiting_samples = (
                        task_results.num_failed_samples
                        + task_results.num_passed_samples
                    )
                    if required_existing_samples > num_already_exisiting_samples:
                        non_cached_task_ids.append(task_id)

        self._newly_generate(
            non_cached_task_ids, sampling_parameters, required_existing_samples
        )

        results = {}

        for task_id in task_ids:
            task_id_stats = self.cache[self.model][task_id][
                self.tranform_sampling_parameters_into_dict_key(sampling_parameters)
            ]
            p_sample_correct = task_id_stats.num_passed_samples / (
                task_id_stats.num_failed_samples + task_id_stats.num_passed_samples
            )
            if not use_expected_value:
                results[task_id] = False
                for _ in range(n_samples):
                    if random.random() <= p_sample_correct:
                        results[task_id] = True
                        break
            else:
                # p von in n versuchen gelöst
                results[task_id] = 1 - (1 - p_sample_correct) ** n_samples

        return results
