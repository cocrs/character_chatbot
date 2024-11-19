from abc import ABC, abstractmethod


class ChatHandler(ABC):
    @abstractmethod
    async def process_question(self, question: str) -> str:
        pass
    
    @abstractmethod
    def sync_with_current_setting(self, remove_memory=False):
        pass
