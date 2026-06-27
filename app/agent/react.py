from abc import abstractmethod

from app.agent.base import BaseAgent
from app.schema import AgentState


class ReActAgent(BaseAgent):
    def step(self) -> str:
        should_act = self.think()

        if not should_act:
            self.state = AgentState.FINISHED
            return "No action needed."

        observation = self.act()

        return observation

    @abstractmethod
    def think(self) -> bool:
        pass

    @abstractmethod
    def act(self) -> str:
        pass