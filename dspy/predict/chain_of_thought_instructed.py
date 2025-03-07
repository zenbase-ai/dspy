import dspy
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.signatures.signature import ensure_signature


class ChainOfThoughtInstructed(ChainOfThought):
    """
    A ChainOfThought that is instructed to perform a specific task.
    """
    def __init__(self, signature, instructions, rationale_type=None, **config):
        signature = ensure_signature(signature, instructions)
        super().__init__(signature, rationale_type, **config)
