import datetime
from dotenv import load_dotenv

load_dotenv()


from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas import AnswerQuestion, ReviseAnswer

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = JsonOutputToolsParser(
    return_id=True
)  # it is going to return us te function call we got back from the llm and transform it into a dictionary
parser_pydantic = PydanticToolsParser(
    tools=[AnswerQuestion]
)  # it is going to searech for the function calling invocation and it's going to parse iut and transform it into an answer querstion object
# it's going to take the answer of the llm and it's going to create an answer question object that we can easily work with


"""
actor_prompt_template is going to be a chat prompt template from the messages. The messages are going to be here, the main prompt,
and all of our history that we used to get so far
"""
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher.
            Curent time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvment.
            3. Recommend search queries to research information and improve your answer.
    
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the requireds format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 words answer"
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """"Revbise your previous answer using the new information.
                        - You should use the previous critique to add important information to your answer.
                        - You MUST include numerical citations in your revised answer to ensure it can be verified.
                        - Add a "References"section to the. bottom of your answer (which does not count towards the word limit). In form. of:
                            - [1] https://example.com
                            - [2] https://example.com
                        - You should use the previous critique to remove superflous information from your answer and make SURE it is not more than 250 words
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        "list startups that do that and raised capital."
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
