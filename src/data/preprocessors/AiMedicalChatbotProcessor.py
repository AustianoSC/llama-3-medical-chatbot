from .BaseProcessor import BaseProcessor

class AiMedicalChatbotProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def format_chat_template(self, row):
        row_json = [{"role": "user", "content": row["Patient"]},
                    {"role": "assistant", "content": row["Doctor"]}]
        row["text"] = self.tokenizer.apply_chat_template(row_json, tokenize=False)
        return row