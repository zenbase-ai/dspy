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


class HoverRetrieveProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7  # Number of documents to retrieve per hop

        # Define the modules for creating queries and summarizing information
        self.identify_entities = dspy.ChainOfThoughtInstructed(
            "claim->definitions_and_relationships:list[str]",
            """Identify Key Entities
Replace specific names (people, places, organizations) with abstract variables (e.g., A, B, X, Y).
If an entity is defined by another entity or attribute (e.g., “the band that released The Balcony”), assign it a function (e.g., B = f(X)).
Define relationships between entities as you identify them (e.g., R(A, B) for "A has a relationship with B").
Output all the definitions and relationships in the definitions_and_relationships field and make sure it's self-contained."""
        )

        self.assumptions = dspy.ChainOfThoughtInstructed(
            "claim,definitions_and_relationships->missing_information:list[str]",
            """given claim, and the definitions and relationships extracted from it, go through each statement of definition and relationships and identify the missing information in order to resolve them. make sure the missing information listed are independent and atomic"""
        )

        self.sorted_missing_information = dspy.ChainOfThoughtInstructed(
            "claim,definitions_and_relationships,missing_information->sorted_missing_information:list[str]",
            "make sure the missing information is sorted by dependency (i.e.: the independent ones come first, and for each dependent one, the dependencies have come first) and make sure they are atomic and complete. Put the final list in sorted_missing_information field"
        )

        self.create_query = dspy.ChainOfThoughtInstructed(
            "claim,definitions_and_relationships,missing_information,focus->keywords:list[str]",
            "focus is a peace of missing information that we are focusing on right now, create a list of keywords in keywords field to search for the information referred to in focus"
        )

        self.refine = dspy.ChainOfThoughtInstructed(
            "claim, definitions_and_relationships,missing_information,context,passages->refined_missing_information:list[str]",
            "Use information in the passages to find missing_information then fill in all the information and simplify missing information. Now, create a new list of missing information with respect to the added context. Make sure to keep the order of missing information based on dependencies, so that independent ones come first and dependencies of each of them come before it. Do not include known information in missing information; only what we need to search for."
        )

        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        definitions_and_relationships = self.identify_entities(
            claim=claim
        ).definitions_and_relationships

        missing_information = self.assumptions(
            claim=claim,
            definitions_and_relationships=definitions_and_relationships
        ).missing_information

        sorted_missing_information = self.sorted_missing_information(
            claim=claim,
            definitions_and_relationships=definitions_and_relationships,
            missing_information=missing_information
        ).sorted_missing_information

        trace = [{
                    "definitions_and_relationships": definitions_and_relationships,
                    "missing_information": missing_information,
                    "sorted_missing_information": sorted_missing_information,
                }]

        refined_missing_information = sorted_missing_information
        context = []

        for i in range(5):
            if len(refined_missing_information) == 0:
                break

            keywords = self.create_query(
                claim=claim,
                definitions_and_relationships=definitions_and_relationships,
                missing_information=refined_missing_information,
                focus=refined_missing_information[0]
            ).keywords

            docs = self.retrieve_k(" ".join(keywords)).passages

            refined_missing_information = self.refine(
                claim=claim,
                definitions_and_relationships=definitions_and_relationships,
                missing_information=refined_missing_information,
                context=context,
                passages=docs
            ).refined_missing_information

            context += docs

            trace.append({
                "query_keywords": keywords,
                "retrieved_docs": docs,
                "refined_missing_information": refined_missing_information,
            })

        return {
            "trace": trace,
            "retrieved_docs": context
        }
