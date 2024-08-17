from abc import ABC, abstractmethod


class ChatHandler(ABC):
    @abstractmethod
    def process_question(self, question: str) -> str:
        pass
