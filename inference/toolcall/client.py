"""Standard-library OpenAI chat client."""

import json
import time
import urllib.error
import urllib.request


def complete(base_url, messages, tools, seed, timeout=300, temperature=0):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({"messages": messages, "tools": tools, "temperature": temperature, "seed": seed}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response), time.perf_counter() - start
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt:
                raise
