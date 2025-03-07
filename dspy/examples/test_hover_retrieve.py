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
from hover_retrieve_discrete import HoverRetrieveProgram, discrete_retrieval_eval


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
def run_single_example(program, example, verbose=False):
    """
    Run a single example through the program and evaluate it.
    
    Args:
        program: The HoverRetrieveProgram instance
        example: A single dspy.Example to process
        verbose: Whether to print detailed output

    Returns:
        evaluation_result
    """
    prediction = program(example.claim)
    evaluation = discrete_retrieval_eval(example, prediction)

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


if __name__ == "__main__":
    MODEL = "gpt-4o-mini"
    COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
    NUM_EXAMPLES = 200
    NUM_THREADS = 1
    VERBOSE = False

    load_dotenv()
    weave.init(project_name="hover-retrieve-program")

    lm = dspy.LM(model=MODEL)
    retriever = ColBERTv2(url=COLBERT_V2_ENDPOINT)
    dspy.settings.configure(lm=lm, rm=retriever)

    examples = load_hover_dataset(num_examples=NUM_EXAMPLES)
    program = HoverRetrieveProgram()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(partial(run_single_example, program, verbose=VERBOSE), examples))

    num_correct_predictions = sum([1 for result in results if result])
    print(f"{num_correct_predictions} Examples out of {NUM_EXAMPLES} were correctly retrieved: {100. * num_correct_predictions/NUM_EXAMPLES}% accurate")
