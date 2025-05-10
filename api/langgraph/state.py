from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

class RAGState(Dict[str, Any]):
    """State object passed between nodes in the LangGraph pipeline."""
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGState":
        state = cls()
        for key, value in data.items():
            state[key] = value
        return state
    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class DialogTurn:
    id: str
    user_message: Message
    assistant_message: Optional[Message] = None

class ConversationMemory:
    """Simple conversation management with a list of dialog turns."""
    def __init__(self):
        self.dialog_turns: List[DialogTurn] = []
    def add_user_message(self, content: str) -> str:
        from uuid import uuid4
        turn_id = str(uuid4())
        self.dialog_turns.append(DialogTurn(id=turn_id, user_message=Message(role="user", content=content)))
        return turn_id
    def add_assistant_message(self, turn_id: str, content: str) -> bool:
        for turn in self.dialog_turns:
            if turn.id == turn_id:
                turn.assistant_message = Message(role="assistant", content=content)
                return True
        return False
    def add_dialog_turn(self, user_content: str, assistant_content: str) -> str:
        from uuid import uuid4
        turn_id = str(uuid4())
        self.dialog_turns.append(DialogTurn(
            id=turn_id,
            user_message=Message(role="user", content=user_content),
            assistant_message=Message(role="assistant", content=assistant_content)
        ))
        return turn_id
    def get_messages(self, limit: int = None) -> List[Tuple[str, str]]:
        messages = []
        for turn in self.dialog_turns[-limit:] if limit else self.dialog_turns:
            messages.append(("user", turn.user_message.content))
            if turn.assistant_message:
                messages.append(("assistant", turn.assistant_message.content))
        return messages
    def to_dict(self) -> Dict:
        return {
            "dialog_turns": [
                {
                    "id": turn.id,
                    "user_message": {
                        "role": turn.user_message.role,
                        "content": turn.user_message.content,
                        "timestamp": turn.user_message.timestamp.isoformat(),
                    },
                    "assistant_message": {
                        "role": turn.assistant_message.role,
                        "content": turn.assistant_message.content,
                        "timestamp": turn.assistant_message.timestamp.isoformat(),
                    } if turn.assistant_message else None,
                }
                for turn in self.dialog_turns
            ]
        }

# Usage Example
if __name__ == "__main__":
    state = RAGState()
    state["query"] = "What does this repo do?"
    memory = ConversationMemory()
    turn_id = memory.add_user_message("Hello!")
    memory.add_assistant_message(turn_id, "Hi! How can I help?")
    print(memory.to_dict()) 