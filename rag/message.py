from langchain_core.documents import Document


class Message:

    def __init__(self, content, is_user: bool, docs: list[Document] = []):
        self.content = content
        self.is_user: bool = is_user
        self.docs = docs
