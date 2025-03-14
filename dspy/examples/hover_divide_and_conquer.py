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
        self.max_calls = 20

        # Define the modules for creating queries and summarizing information
        self.divide = dspy.ChainOfThoughtInstructed(
            "task->sub_tasks:list[str]",
            ("Divide the claim into smaller sub-tasks.\n"
             "Each sub-task should be a self-contained standalone question that can be answered independently.\n"
             "The sub-tasks should be mutually exclusive and collectively exhaustive.\n"
             "Avoid making the question or its scope any bigger; only include what is necessary in the sub-tasks.\n"
             "Avoid adding additional information and only use what is available in the task itself.")
        )

        self.check_atomic = dspy.ChainOfThoughtInstructed(
            "task->is_breakable:bool",
            ("Everything in this chat is symbolic. They might seen to have connection to real entities but they don't. \n"
             "In analyzing this task, only use the context provide and avoid using information that is not provided here.\n"
             "Sometimes, breaking a task down helps understand it better, and being more accurate.\n"
             "However, there are times when breaking tasks just adds unnecessary context to it diverges our focus from the actual task.\n"
             "In this case, is breaking the task worthy?"
             "If yes, return True. Otherwise, return False.")
        )

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

        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        self._calls = 0
        trace = []

        task = f"Verify if the claim is true or false. Claim: {claim}"

        context = self.divide_recursive(task, trace)
        
        return {
            "trace": trace,
            "retrieved_docs": context
        } 
    
    def divide_recursive(self, task, trace):
        if self._calls >= self.max_calls:
            raise RuntimeError(f"Max number of calls reached {self._calls}")

        self._calls += 1
        if not self.check_atomic(task=task).is_breakable:
            return self.solve_atomic_task(task, trace)

        self._calls += 1
        sub_tasks = self.divide(
            task=task
        ).sub_tasks

        trace.append({
            "sub_tasks": sub_tasks,
        })

        context = []
        for sub_task in sub_tasks:
            context += self.divide_recursive(sub_task, trace)

        return context
    
    def solve_atomic_task(self, task, trace):
        self._calls += 2
        query = self.make_query(task=task).query
        docs = self.retrieve_k(query).passages
        solution = self.solve(task=task, context=docs)
        is_done = solution.is_done
        note = solution.note

        trace.append({
            "query": query,
            "retrieved_docs": docs,
            "is_done": is_done,
            "solution": note
        })

        if is_done:
            return docs
        else:
            return self.divide_recursive(note, trace)