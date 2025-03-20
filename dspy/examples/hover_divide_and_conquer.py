import dspy
from instructor import Mode, OpenAISchema
from pydantic import Field
from instructor import patch
import openai


def discrete_retrieval_eval(example, pred, trace=None):
    """
    Evaluation function for hover retrieve discrete tasks.
    Checks if all gold titles are found in the retrieved documents.
    """
    retrieved_docs = pred['retrieved_docs']

    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in retrieved_docs],
        )
    )
    return gold_titles.issubset(found_titles), found_titles, gold_titles


class CheckAtomicOutput(OpenAISchema):
    """Output schema for checking if a task is atomic or can be broken down."""
    reasoning: str = Field(
        ...,
        description="Explanation for why the task should or should not be broken down"
    )
    is_breakable: bool = Field(
        ...,
        description="True if the task should be broken down, False if it should be solved directly"
    )


class AbstractionOutput(OpenAISchema):
    """Output schema for creating task abstractions."""
    abstract_task: str = Field(
        ...,
        description="The abstract version of the task"
    )
    mapping: dict[str, str] = Field(
        ...,
        description="Mapping between concrete and abstract terms"
    )


class Abstraction:
    """Handles task abstraction and de-abstraction."""
    
    def __init__(self, client):
        """Initialize the abstraction handler with an OpenAI client."""
        self.patched_client = patch(client, mode=Mode.MD_JSON)
        self.mapping = {}
    
    def create_abstraction(self, task: str) -> AbstractionOutput:
        """Create an abstract version of a task and store the mapping."""
        prompt = (f"""You are an expert at creating abstract versions of tasks while preserving their core meaning.

Task to abstract: {task}

Create an abstract version of this task by:
1. Replacing specific terms with abstract placeholders
2. Maintaining the logical structure and relationships
3. Creating a clear mapping between concrete and abstract terms

Respond in JSON format with:
1. abstract_task: The abstract version of the task
2. mapping: A dictionary mapping concrete terms to abstract placeholders

Example:
Input task: "Find articles about the impact of climate change on polar bears in the Arctic"
"""
"""Output:
{
    "abstract_task": "Find articles about the impact of [PHENOMENON] on [SPECIES] in the [LOCATION]",
    "mapping": {
        "climate change": "[PHENOMENON]",
        "polar bears": "[SPECIES]",
        "Arctic": "[LOCATION]"
    }
}
""")
        
        result = self.patched_client.chat.completions.create(
            model="o3-mini",
            response_model=AbstractionOutput,
            messages=[
                {"role": "system", "content": "You are a task abstraction expert. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Store the mapping for later use
        self.mapping = result.mapping
        return result.abstract_task
    
    def abstract(self, text: str) -> str:
        """Convert concrete text to abstract form using the stored mapping."""
        if not self.mapping:
            raise ValueError("No abstraction mapping available. Call create_abstraction first.")
        
        abstract_text = text
        for concrete, abstract in self.mapping.items():
            abstract_text = abstract_text.replace(concrete, abstract)
        return abstract_text
    
    def deabstract(self, text: str) -> str:
        """Convert abstract text back to concrete form using the stored mapping."""
        if not self.mapping:
            raise ValueError("No abstraction mapping available. Call create_abstraction first.")
        
        concrete_text = text
        for concrete, abstract in self.mapping.items():
            concrete_text = concrete_text.replace(abstract, concrete)
        return concrete_text


def create_check_atomic(client):
    """Create a function that checks if a task is atomic using Instructor."""
    patched_client = patch(client, mode=Mode.MD_JSON)
    
    def check_atomic(task: str) -> CheckAtomicOutput:
        """Check if a task should be broken down or solved directly."""
        prompt = f"""You are an expert at analyzing tasks and determining whether they should be broken down into smaller sub-tasks or solved directly.

Task to analyze: {task}

Analyze this task and determine if it should be broken down into smaller sub-tasks or solved directly.
Remember:
- Only use the context provided in the task
- You are following a divide and conquer approach, and are currently focused on dividing
- You want to decide to either break down the task or say it is already trivial and further dividing would diverge us from the solution.

Respond in JSON format with two fields:
1. reasoning: A clear explanation of why the task should or should not be broken down
2. is_breakable: A boolean indicating if the task should be broken down (true) or solved directly (false)
```"""
        
        return patched_client.chat.completions.create(
            model="o3-mini",
            response_model=CheckAtomicOutput,
            messages=[
                {"role": "system", "content": "You are a task analysis expert. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ]
        )
    
    return check_atomic


class HoverDivideAndConquerProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7  # Number of documents to retrieve per hop
        self.max_calls = 40

        # Define the modules for creating queries and summarizing information
        self.divide = dspy.ChainOfThoughtInstructed(
            "task->sub_tasks:list[str]",
            ("Divide the claim into smaller sub-tasks.\n"
             "Each sub-task should be a self-contained standalone question that can be answered independently.\n"
             "The sub-tasks should be mutually exclusive and collectively exhaustive.\n"
             "Avoid making the question or its scope any bigger; only include what is necessary in the sub-tasks.\n"
             "Avoid adding additional information and only use what is available in the task itself.\n"
             "Make sure to use only the [PLACEHOLDER]s that are previously used and use them exactly as they are so that they can be found and replaced later.")
        )

        # Using Instructor for check_atomic
        self.check_atomic = create_check_atomic(openai.OpenAI())

        self.make_query = dspy.ChainOfThoughtInstructed(
            "task->query",
            ("Make a effective query for finding relevant documents for doing the given task.\n"
             "This query should be a list of important keywords, or a question to be used on an specific retriever.\n"
             "Avoid adding additional information and only use what is available in the task itself.")
        )

        self.solve = dspy.ChainOfThoughtInstructed(
            "task,context->is_done:bool,note",
            ("Is the context enough for doing the task? "
             "If yes, write True in is_done and a concise note of the solution in note.\n"
             "Otherwise, write False in is_done and ask for additional context that can potentially help in note.")
        )

        self.integrate = dspy.ChainOfThoughtInstructed(
            "task,context->simplified_task",
            ("You are given a task, and steps taken to do a part of it.\n"
             "Use the context, and articles provided in the context to simplify the original task with relevant information from the context and redefine it.\n"
             "Include everything required in the new task and output it in simplified_task.\n"
             "Only put the new task in simplified_task with no addition, prefix, or postfix.\n")
        )

        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        self._calls = 0
        trace = []

        task = f"Find articles either supporting or rejecting the claim. Claim: {claim}"
        self.abstraction = Abstraction(openai.OpenAI())
        abstract_task = self.abstraction.create_abstraction(task)

        new_abstract_task = abstract_task
        for i in range(5):
            context = self.divide_recursive(new_abstract_task, trace)
            new_task = self.integrate(task=task, context=self.make_string_context(context)).simplified_task
            new_abstract_task = self.abstraction.create_abstraction(new_task)
            print("\nNew Task: ", new_task)
        
        return {
            "trace": trace,
            "retrieved_docs": context
        } 
    
    def divide_recursive(self, task, trace):
        if self._calls >= self.max_calls:
            raise RuntimeError(f"Max number of calls reached {self._calls}")

        self._calls += 1
        check_result = self.check_atomic(task)
        print(f"Is this task breakable: {task}")
        print(check_result.is_breakable, check_result.reasoning)
        if not check_result.is_breakable:
            trace.append({
                "check_atomic_reasoning": check_result.reasoning
            })
            return self.solve_atomic_task(task, trace)

        self._calls += 1
        sub_tasks = self.divide(
            task=task
        ).sub_tasks

        trace.append({
            "sub_tasks": sub_tasks,
        })

        context = []
        for sub_task in [sub_tasks[0]]:
            context += [{"task": task, "sub_tasks":self.divide_recursive(sub_task, trace)}]

        return context
    
    def solve_atomic_task(self, task, trace):
        self._calls += 2
        query = self.make_query(task=task).query
        query_with_context = self.abstraction.deabstract(query)
        docs = self.retrieve_k(query_with_context).passages
        abstract_docs = [self.abstraction.abstract(doc) for doc in docs]
        solution = self.solve(task=task, context=abstract_docs)
        is_done = solution.is_done
        note = solution.note

        trace.append({
            "query": query,
            "retrieved_docs": docs,
            "is_done": is_done,
            "solution": note
        })

        if is_done:
            return [{"task":task, "docs":docs}]
        else:
            return [{"task":task, "sub_tasks":self.divide_recursive(note, trace)}]

    def make_string_context(self, context):
        output = ["In order to help solve the task, we select smaller parts of it until it's easy to solve:"]
        c = context[0]
        output += c["task"]
        while "sub_tasks" in c:
            c = c["sub_tasks"][0]
            output += [c["task"]]

        output += ["\nHere are documents relavant to the smallest task:\n", "\n".join(c["docs"])]

        return self.abstraction.deabstract("\n".join(output))
