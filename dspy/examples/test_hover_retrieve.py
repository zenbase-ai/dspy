from copy import deepcopy
from pprint import pprint

import dspy
import random
from dotenv import load_dotenv
import weave
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from dspy import ColBERTv2
from datasets import load_dataset
from hover_retrieve_discrete import HoverRetrieveProgram, discrete_retrieval_eval as discrete_eval
from hover_divide_and_conquer import HoverDivideAndConquerProgram, discrete_retrieval_eval as divide_conquer_eval


def load_hover_dataset(num_examples=100, filter_3hop=True):
    """
    Load and preprocess the HoVer dataset.
    
    Args:
        num_examples: Number of examples to load
        filter_3hop: Whether to filter for only 3-hop examples
        
    Returns:
        List of dspy.Example objects
    """
    dataset = load_dataset("hover")
    hf_trainset = dataset["validation"]
    
    reformatted_examples = []
    
    for example in hf_trainset:
        claim = example["claim"]
        supporting_facts = example["supporting_facts"]
        label = example["label"]
        
        # Count unique documents in supporting facts
        unique_docs = len(set([fact["key"] for fact in supporting_facts]))
        
        # Filter for 3-hop examples if requested
        if not filter_3hop or unique_docs == 3:
            reformatted_examples.append(dict(
                claim=claim, 
                supporting_facts=supporting_facts, 
                label=label
            ))
    
    # Shuffle with fixed seed for reproducibility
    rng = random.Random(0)
    rng.shuffle(reformatted_examples)
    
    # Convert to dspy.Example objects
    examples = [dspy.Example(**x).with_inputs("claim") for x in reformatted_examples[:num_examples]]
    
    return examples


def get_titles_from_docs(docs):
    return [d.split("|")[0].strip() for d in docs]


def summarize_prediction(prediction):
    prediction_summary = deepcopy(prediction)

    for t in prediction_summary['trace']:
        if "retrieved_docs" not in t:
            continue

        t["retrieved_docs"] = get_titles_from_docs(t["retrieved_docs"])

    prediction_summary["retrieved_docs"] = get_titles_from_docs(prediction_summary["retrieved_docs"])

    return prediction_summary


@weave.op()
def run_single_example(program, example, eval_func, verbose=False):
    """
    Run a single example through the program and evaluate it.
    
    Args:
        program: The program instance to run
        example: A single dspy.Example to process
        eval_func: The evaluation function to use
        verbose: Whether to print detailed output

    Returns:
        evaluation_result
    """
    prediction = program(example.claim)
    evaluation, found_titles, gold_titles = eval_func(example, prediction)

    if verbose:
        print()
        print(f"Claim: {example.claim}")
        print(f"Gold titles: {[doc['key'] for doc in example.supporting_facts]}")
        print()
        pprint(summarize_prediction(prediction))
        print()
        print(f"Evaluation result: {evaluation}")
        print()

    return evaluation


def run_program(program_class, eval_func, examples, num_threads=20, verbose=False):
    """Run a program on all examples using multiple threads."""
    program = program_class()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(
            partial(run_single_example, program, eval_func=eval_func, verbose=verbose), 
            examples
        ))
    
    num_correct = sum([1 for result in results if result])
    accuracy = 100. * num_correct / len(examples)
    
    return num_correct, accuracy


if __name__ == "__main__":
    MODEL = "gpt-4o-mini"
    COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
    NUM_EXAMPLES = 1
    NUM_THREADS = 20
    VERBOSE = True

    load_dotenv()
    weave.init(project_name="hover-retrieve-program")

    # Set up models
    lm = dspy.LM(model=MODEL)
    retriever = ColBERTv2(url=COLBERT_V2_ENDPOINT)
    dspy.settings.configure(lm=lm, rm=retriever)

    # Load dataset
    examples = load_hover_dataset(num_examples=NUM_EXAMPLES)

    # Run both programs
    programs = [
        # ("Discrete Retrieval", HoverRetrieveProgram, discrete_eval),
        ("Divide and Conquer", HoverDivideAndConquerProgram, divide_conquer_eval)
    ]

    for name, program_class, eval_func in programs:
        print(f"\nRunning {name} program...")
        num_correct, accuracy = run_program(
            program_class, 
            eval_func,
            examples, 
            num_threads=NUM_THREADS, 
            verbose=VERBOSE
        )
        print(f"{num_correct} Examples out of {NUM_EXAMPLES} were correctly retrieved: {accuracy:.2f}% accurate")
