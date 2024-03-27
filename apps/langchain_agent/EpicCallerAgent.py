from vocode.streaming.models.transcript import Transcript
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.agent.base_agent import RespondAgent

import typing
from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM
from salesgpt.agents import SalesGPT
from salesgpt.tools import get_tools
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.prompts import SALES_AGENT_TOOLS_PROMPT


# Assuming SalesGPT and necessary imports are already handled
from salesgpt.agents import SalesGPT
from langchain_community.chat_models import ChatLiteLLM

class EpicCallerAgent(RespondAgent):
    def __init__(self, agent_config, *args, **kwargs):
        super().__init__(agent_config, *args, **kwargs)
        self.sales_agent = self.initialize_sales_gpt()

    def initialize_sales_gpt(self):
        # Configuration for SalesGPT
        config = {
            "verbose": True,
            "max_num_turns": 200,
            "model_name": "gpt-3.5-turbo",
            "product_catalog": "examples/sample_product_catalog.txt",
            "use_tools": True,
        }
        tools = get_tools("examples/sample_product_catalog.txt")
        prompt = CustomPromptTemplateForTools(
            template=SALES_AGENT_TOOLS_PROMPT,
            tools_getter=lambda x: tools,
            input_variables=[
                "input",
                "intermediate_steps",
                "salesperson_name",
                "salesperson_role",
                "company_name",
                "company_business",
                "company_values",
                "conversation_purpose",
                "conversation_type",
                "conversation_history",
            ],
        )
        llm = ChatLiteLLM(temperature=0.2, model_name=config["model_name"])
        sales_agent = SalesGPT.from_llm(llm, **config)
        sales_agent.seed_agent()
        return sales_agent

    async def respond(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[Optional[str], bool]:
        self.sales_agent.human_step(human_input)
        ai_log = self.sales_agent.step(stream=False)
        self.sales_agent.determine_conversation_stage()
        
        # Extract the response from SalesGPT's conversation history
        reply = (
            self.sales_agent.conversation_history[-1]
            if self.sales_agent.conversation_history
            else ""
        )
        response_message = ": ".join(reply.split(": ")[1:]).rstrip("<END_OF_TURN>")
        
        return response_message, False  # Continue the conversation

    # The generate_response method might not be needed if all interactions are synchronous
