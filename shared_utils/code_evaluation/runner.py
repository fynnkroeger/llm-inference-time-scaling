from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import ast
from multiprocessing.managers import SyncManager
import tqdm

from human_eval.data import HUMAN_EVAL, read_problems

# Required for check_correctness execution
from human_eval.execution import create_tempdir, reliability_guard, swallow_io, time_limit, TimeoutException
from typing import Optional, Dict
import multiprocessing
import operator
from shared_utils.code_evaluation.radix_tree import RadixTree

# Supported operators
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}


SUPPORTED_FUNCTIONS = {
    "str": str
}
# 5:10 with cache
test_results_cache = RadixTree.from_disk()

# Adds the ability to evaluate a bit more complicated expressions than ast.literal_eval can handle
def eval_ast(node, context: dict={}):
    if isinstance(node, ast.BinOp):
        left = eval_ast(node.left, context)
        right = eval_ast(node.right, context)
        op = OPS[type(node.op)] # type: ignore
        return op(left, right)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        # Resolve variable from context
        if node.id in context:
            return context[node.id]
        raise ValueError(f"Variable {node.id} is not in the context.")
    elif isinstance(node, ast.Call):
        # Optional: handle function calls if needed
        if node.func.id not in SUPPORTED_FUNCTIONS.keys(): # type: ignore
            raise ValueError("Function calls are not supported in eval_ast.")
        
        return SUPPORTED_FUNCTIONS[node.func.id](*[eval_ast(x, context) for x in node.args]) # type: ignore
    else:
        return ast.literal_eval(node)
    
def extract_input_output_pairs_from_assert_statements_in_function(test_function_definition: str) -> list[tuple]:
    # Remove the starting function signature def check(canidate)... 
    start_of_assert_statements = "assert"
    start_of_assert_statements_index = test_function_definition.find(start_of_assert_statements)
    if start_of_assert_statements_index == -1:
        raise ValueError(f"Couldn't find any assert statements because: Prefix {start_of_assert_statements} not found in test function definition: {test_function_definition}")
    
    test_string = test_function_definition[start_of_assert_statements_index:]
    assert_statements = [x.strip() for x in test_string.split("\n")]
    input_output_pairs = []
    for assert_statement in assert_statements:
        if not assert_statement.startswith("assert"):
            continue
        assert_test = ast.parse(assert_statement).body[0].test # type: ignore

        # Skip gigachad openai developer assert True statements
        if isinstance(assert_test, ast.Constant):
            continue
        input_args = [eval_ast(arg) for arg in assert_test.left.args]
        outputs = eval_ast(assert_test.comparators[0])
        input_output_pairs.append((input_args, outputs))

    return input_output_pairs

def make_object_hashable(object):
        if isinstance(object, dict):
            return tuple(object.sorted()) # type: ignore
        elif isinstance(object, list):
            return tuple(object)
        else:
            return object

def hash_object(object) -> int:
    return hash(make_object_hashable(object))

def execute_and_extract_output(function_name: str, function_definition: str, positional_function_input):
    extractor_function = f"""
{function_definition}


function_output = {function_name}(*raw_function_input)
"""
    exec_locals = {}
    exec(extractor_function, {"raw_function_input": positional_function_input}, exec_locals)
    return exec_locals["function_output"]

def check_correctness(problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None, input_output_pairs: Optional[list[tuple]] = None,
                       hash_function_outputs: bool = True) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    :param input_output_pairs: Optional input output pairs to get function outputs
    """

    def unsafe_execute(input_output_pairs: Optional[list[tuple]]):
        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            if input_output_pairs is None:
                check_program = (
                    problem["prompt"] + completion + "\n" +
                    problem["test"] + "\n" +
                    f"check({problem['entry_point']})"
                )
                try:
                    exec_globals = {}

                    with swallow_io():
                        with time_limit(timeout):
                            exec(check_program, exec_globals)
                            test_results.append(True)
                except TimeoutException:
                    function_outputs.append("timed out")
                    test_results.append(False)
                except BaseException as e:
                    function_outputs.append(f"failed: {e}")
                    test_results.append(False)
            else:
                assert not any(test_results), "We assume test_results is initialized with False"
                # Disable functionalities that can make destructive changes to the test.
                reliability_guard()            
                generated_function = problem["prompt"] + completion + "\n"
                for i, (function_input, target_ouput) in enumerate(input_output_pairs):
                    try:
                        with swallow_io():
                            with time_limit(timeout / len(input_output_pairs)):
                                function_output = execute_and_extract_output(problem["entry_point"], generated_function, function_input)
                                function_outputs[i] = hash_object(function_output) if hash_function_outputs else function_output
                                test_results[i] = function_output == target_ouput
                    except TimeoutException:
                        function_outputs[i] = "TimeoutException"
                    except BaseException as e:
                        function_outputs[i] = str(e)

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
  
    complete_program = problem["prompt"] + completion
    complete_test_results = test_results_cache.get(complete_program)
    if complete_test_results is None:
        manager: SyncManager = multiprocessing.Manager()
        test_results = manager.list()
        function_outputs = manager.list()

        if input_output_pairs is not None:
            for _ in range(len(input_output_pairs)):
                test_results.append(False)
                function_outputs.append("TimeoutException")

        p = multiprocessing.Process(target=unsafe_execute, args=[input_output_pairs])
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()
        complete_test_results = dict(
            passed=all(test_results),
            function_outputs=tuple(list(function_outputs)), # We need to convert the multiprocessing proxy list to a normal list.
            test_results=list(test_results),
            failed_to_parse_test_cases=input_output_pairs is None
        )
        test_results_cache.add(complete_program, complete_test_results)

    return dict(
        task_id=problem["task_id"],
        completion_id=completion_id
    ) | complete_test_results


problems = read_problems(HUMAN_EVAL)
task_id_to_input_output_pairs = {}

for task_id, problem in problems.items():
    try:
        pair = extract_input_output_pairs_from_assert_statements_in_function(problem["test"])
    except Exception:
        pair = None
    task_id_to_input_output_pairs[task_id] = pair


def evaluate_only_functional_correctness(
    samples: list[dict],
    n_workers: int = 128,
    timeout: float = 1.5,
    extract_function_outputs: bool = False,
    hash_function_outputs: bool = True,
    verbose: bool = False
) -> list[dict]:
    """
    Evaluates the functional correctness of generated samples
    """
    n_samples = len(samples)

    # Initialize results storage
    completion_id = Counter()
    results = defaultdict(list)

    
    # Run test cases in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for sample in samples:
            # Extract input output pair if not cached yet
            if extract_function_outputs:
                input_output_pair = task_id_to_input_output_pairs[sample["task_id"]]
            else:
                input_output_pair = None

            future = executor.submit(
                check_correctness,
                problems[sample["task_id"]],
                sample["completion"],
                timeout,
                completion_id[sample["task_id"]],
                input_output_pair,
                hash_function_outputs
            )

            futures[future] =  (sample["task_id"], completion_id[sample["task_id"]])

        # Collect results as they complete
        futures_iterator = tqdm.tqdm( as_completed(futures),desc="Running test", total=n_samples) if verbose else as_completed(futures)
        for future in futures_iterator:
            task_id, comp_id = futures[future]
            try:
                result = future.result()
                results[task_id].append((comp_id, result))
            except Exception as e:
                print(f"Error processing task {task_id}: {e}")

    # Combine results and save them in one go
    def combine_results():
        for sample in samples:
            task_id = sample["task_id"]
            index_and_result = results[task_id].pop(0)
            result = index_and_result[1]
            sample["test_results"] = result["test_results"]
            sample["function_outputs"] = result["function_outputs"]
            sample["passed"] = result["passed"]
            sample["failed_to_parse_test_cases"] = result["failed_to_parse_test_cases"]
            yield sample

    return list(combine_results())
