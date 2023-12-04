from langchain.prompts import PromptTemplate

_qa_documents_prompt_template = """You are a very helpful and accurate Question Answering assistant.
You are provided with following contexts to answer the questions at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Strictly answer only based on the provided contexts. Be as concise as possible.
If the answer cannot be inferred from the context, respond with 'Answer cannot be inferred'.
If you are asked question that's out of the provided context, just say you are a very helpful question answering assistant and say you cannot answer.

{context}

Question: {question}""".strip()

QA_DOCUMENTS_PROMPT = PromptTemplate(
    template=_qa_documents_prompt_template,
    input_variables=["context", "question"],
)

SQL_AGENT_QUERY_AUGMENTATION_PROMPT = """1. Use both the `similarity(<column_name>, <value>) >={threshold}` function as well as `ILIKE` operator for text matching.
2. Condolidate the final query based on both the operations removing any duplicate rows.
3. If the question is about finding solutions for satellites, look for `instrument`, `platform` columns for text matching and unify them.
4. If a proper response is not generated, just say 'I can't answer.', and nothing else.
5. Strictly, avoid unwanted answers that are not in the result. Avoid generating generic responses unrelated to the data.
6. Strictly avoid outputing sql query as the final answer.
7. Ignore tables "langchain_pg_collection" and "langchain_pg_embedding" in the database.
"""
