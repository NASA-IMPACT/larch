#!/usr/bin/env python3

import re
from typing import List, Tuple, Union

from langchain import LLMChain
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain_community.chat_models import ChatOpenAI

from ..structures import Response
from .engines import AbstractSearchEngine


class CustomAgentSearchEngine(AbstractSearchEngine):
    _TEMPLATE = """
        You are given a query. Choose the right tool to pass the query to get the right answer.
        Each of the tools is the search engine to call based on provided context.
        It could be performing Information Retrieval (IR) based search or metadata or SQL based searches.
        If one of the tools doesn't provide any answer, skip that and use other tools.
        Finally, if you have the answer, provide that answer. Consolidate the final answer in human readable form which should answer the query correctly.
        You have access to the following tools:

        {tools}

        Use the following format:

        Query: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: The result of the action. You must follow the observation with either an action or final answer
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: Now that I have the query and there isn't metadata fields that can be extracted from the query, I choose DocumentStoreRAG
        Action: DocumentStoreRAG
        Action Input: the query string from the user
        Observation: the result of the action
        Final Answer: Correctly extracted answer from the search engine, which is human readable form that directly answers the query

        Begin Loop:

        Query: {input}
        {agent_scratchpad}
    """.strip()

    class _CustomOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={
                        "output": llm_output.split("Final Answer:")[-1].strip(),
                    },
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(
                tool=action,
                tool_input=action_input.strip(" ").strip('"'),
                log=llm_output,
            )

    class _CustomPromptTemplate(BaseChatPromptTemplate):
        template: str
        tools: List[Tool]

        def format_messages(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join(
                [f"{tool.name}: {tool.description}" for tool in self.tools],
            )
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            formatted = self.template.format(**kwargs)
            return [HumanMessage(content=formatted)]

    def __init__(self, *engines, model: str = "gpt-3.5-turbo", debug: bool = False):
        super().__init__(debug=debug)
        self.engines = engines
        self.tools = self.create_tools()
        self.tool_names = list(map(lambda t: t.name, self.tools))
        self.model = model or "gpt-3.5-turbo"

        self.prompt = self._CustomPromptTemplate(
            template=CustomAgentSearchEngine._TEMPLATE,
            tools=self.tools,
            input_variables=["input", "intermediate_steps"],
        )

    @property
    def llm_chain(self):
        return LLMChain(
            llm=ChatOpenAI(model_name=self.model, temperature=0),
            prompt=self.prompt,
        )

    def create_tools(self) -> list:
        tools = []
        for engine in self.engines:
            tool = Tool(
                name=engine.__class__.__name__,
                description=engine.__doc__.strip(),
                func=engine,
            )
            tools.append(tool)
        return tools

    @property
    def agent_executor(self) -> Response:
        agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self._CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=self.tool_names,
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=self.debug,
            return_intermediate_steps=True,
        )

    def query(self, query: str, **kwargs) -> Response:
        res = self.agent_executor(query)
        return Response(
            text=res.get("output", ""),
            source=self.__classname__,
            evidences=self._intermediate_steps_to_evidences(
                res.get("intermediate_steps", []),
            ),
            # extras=dict(intermediate_steps=res.get("intermediate_steps", [])),
        )

    @staticmethod
    def _intermediate_steps_to_evidences(
        steps: List[Tuple[AgentAction, Response]],
    ) -> List[Response]:
        res = []
        # each step has a tuple of (<AgentAction>, <Response>)
        for step in steps:
            evidence = step[1]
            extras = evidence.extras or {}
            extras["agent_action"] = step[0]
            evidence.extras = extras
            res.append(evidence)
        return res


def main():
    pass


if __name__ == "__main__":
    main()
