import requests

class QwenToxicEvaluatorClient:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def evaluate(self, inputs: list[str], outputs: list[str]=None, th: float = 0.5) -> tuple[list[bool], list[float]]:
        """
        :param inputs: input list
        :param th: threshold for toxicity evaluation
        :return: (is_behavior_present, prediction)
        """
        url = f"{self.server_url}/evaluate/toxic"
        payload = {"inputs": inputs, "th": th}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["results"], data["predictions"]
        else:
            raise Exception(f"Error from server: {response.status_code}, {response.text}")


class QwenAnswerEvaluatorClient:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def evaluate(self, inputs: list[str], outputs: list[str], th: float = 0.5) -> tuple[list[bool], list[float]]:
        """
        :param inputs: input list
        :param outputs: output list of target model
        :param th: threshold for answer evaluation
        :return: (is_behavior_present, prediction)
        """
        url = f"{self.server_url}/evaluate/answer"
        payload = {"inputs": inputs, "outputs": outputs, "th": th}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["results"], data["predictions"]
        else:
            raise Exception(f"Error from server: {response.status_code}, {response.text}")

def get_server(server_url: str, port: int) -> str:
    """
    Get the server URL with port.
    :param server_url: server address (IP or domain)
    :param port: server port
    :return: IP address with port
    """
    if not server_url.startswith("http://"):
        server_url = "http://" + server_url
    if server_url.endswith("/"):
        server_url = server_url[:-1]
    return f"{server_url}:{port}"

if __name__ == "__main__":
    server = "127.0.0.1"
    port = 4397
    SERVER_URL = get_server(server, port)

    toxic_client = QwenToxicEvaluatorClient(SERVER_URL)
    answer_client = QwenAnswerEvaluatorClient(SERVER_URL)

    try:
        inputs = ["This is a test input for toxicity evaluation."]
        th = 0.5
        toxic_results, toxic_predictions = toxic_client.evaluate(inputs, th=th)
        print("Toxic Evaluation Results:", toxic_results)
        print("Toxic Predictions:", toxic_predictions)
    except Exception as e:
        print("Error during toxic evaluation:", e)

    try:
        inputs = ["What is the capital of France?"]
        outputs = ["The capital of France is Paris."]
        th = 0.5
        answer_results, answer_predictions = answer_client.evaluate(inputs, outputs, th)
        print("Answer Evaluation Results:", answer_results)
        print("Answer Predictions:", answer_predictions)
    except Exception as e:
        print("Error during answer evaluation:", e)