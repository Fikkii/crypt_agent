from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, output_pydantic, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from pydantic import BaseModel
from typing import List

class TradeOrder(BaseModel):
    symbol: str    # e.g., "BTCUSDT"
    side: str      # "Buy" or "Sell"
    quantity: float
    take_profit: float
    stop_loss: float

class TradeSignal(BaseModel):
    orders: List[TradeOrder]

from crypt_agent.tools.custom_tool import (
    advanced_sliced_executor,
    calculate_technical_indicators,
    check_wallet_balance,
    execute_multiple_orders,
    get_latest_klines,
    math_tool, 
    place_market_order, 
    fetch_ticker_price, 
)

search_tool = SerperDevTool()

@CrewBase
class CryptAgent():
    """CryptAgent crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            tools=[search_tool, fetch_ticker_price, get_latest_klines], # type: ignore[list-item]
            llm="gemini/gemini-2.0-flash",
            verbose=True
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'], # type: ignore[index]
            verbose=True,
            llm="gemini/gemini-2.0-flash",
            tools=[calculate_technical_indicators, math_tool] # type: ignore[list-item]
        )

    @agent
    def strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['strategist'], # type: ignore[index]
            tools=[fetch_ticker_price, check_wallet_balance],
            llm="gemini/gemini-2.0-flash",
            verbose=True
        )

    @agent
    def trader(self) -> Agent:
        return Agent(
            config=self.agents_config['trader'], # type: ignore[index]
            tools=[place_market_order, execute_multiple_orders, advanced_sliced_executor],
            llm="gemini/gemini-2.0-flash",
            verbose=True
        )

    @agent
    def reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['reporter'], # type: ignore[index]
            llm="gemini/gemini-2.0-flash",
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'], # type: ignore[index]
        )

    @task
    def strategist_task(self) -> Task:
        return Task(
            config=self.tasks_config['strategist_task'], # type: ignore[index]
            output_pydantic=TradeSignal,
            output_file='output/trade_signal.json'
        )

    @task
    def trade_task(self) -> Task:
        return Task(
            config=self.tasks_config['trade_task'], # type: ignore[index]
        )

    @task
    def report_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_task'], # type: ignore[index]
            output_file='output/trade_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CryptAgent crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
