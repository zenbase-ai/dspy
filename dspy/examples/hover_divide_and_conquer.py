import dspy


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


class HoverDivideAndConquerProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7  # Number of documents to retrieve per hop

        # Define the modules for creating queries and summarizing information
        self.divide = dspy.ChainOfThoughtInstructed(
            "task->sub_tasks:list[str]",
            """Divide the claim into smaller sub-tasks. Each sub-task should be a self-contained standalone question that can be answered independently. The sub-tasks should be mutually exclusive and collectively exhaustive. Avoid adding additional information and only use what is available in the task itself."""
        )

        self.check_atomic = dspy.ChainOfThoughtInstructed(
            "task->is_atomic:bool",
            """Is the task easy to understand and is it a question that can have a conclusive answer? If yes, return True. Otherwise, return False."""
        )

        self.make_query = dspy.ChainOfThoughtInstructed(
            "task->query",
            """Make a effective query for finding relevant documents for doing the given task. This query should be a list of important keywords, or a question to be used on an specific retriever."""
        )

        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        trace = []

        task = f"Verify if the claim is true or false. Claim: {claim}"

        context = self.divide_recursive(task, trace)
        
        return {
            "trace": trace,
            "retrieved_docs": context
        } 
    
    def divide_recursive(self, task, trace):
        sub_tasks = self.divide(
            task=task
        ).sub_tasks

        trace.append({
            "sub_tasks": sub_tasks,
        })

        context = []
        for sub_task in sub_tasks:
            is_atomic = self.check_atomic(task=sub_task).is_atomic
            if is_atomic:
                context += self.solve_atomic_task(sub_task, trace)
            else:
                context += self.divide_recursive(sub_task, trace)

        return context
    
    def solve_atomic_task(self, task, trace):
        query = self.make_query(task=task).query
        docs = self.retrieve_k(query).passages

        trace.append({
            "query": query,
            "retrieved_docs": docs,
        })

        return docs

