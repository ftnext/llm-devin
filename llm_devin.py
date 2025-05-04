import os
import time

import httpx
import llm


class DevinModel(llm.KeyModel):
    needs_key = "devin"
    key_env_var = "LLM_DEVIN_KEY"
    can_stream = False

    def __init__(self) -> None:
        self.model_id = "devin"

    def execute(self, prompt, stream, response, conversation, key):
        create_session_response = httpx.post(
            "https://api.devin.ai/v1/sessions",
            headers={"Authorization": f"Bearer {os.getenv('LLM_DEVIN_KEY')}"},
            json={"prompt": prompt.prompt, "idempotent": True},
        )
        create_session_response.raise_for_status()

        session_id = create_session_response.json().get("session_id")
        print("Devin URL:", create_session_response.json()["url"])

        while True:
            session_detail = httpx.get(
                f"https://api.devin.ai/v1/session/{session_id}",
                headers={"Authorization": f"Bearer {os.getenv('LLM_DEVIN_KEY')}"},
            )
            session_detail.raise_for_status()
            session_detail_json = session_detail.json()
            if session_detail_json["status_enum"] in {"blocked", "stopped"}:
                break
            time.sleep(5)

        for message in session_detail_json["messages"]:
            if message["type"] == "devin_message":
                yield message["message"]


@llm.hookimpl
def register_models(register):
    register(DevinModel())
