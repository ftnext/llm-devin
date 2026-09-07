import json
from unittest.mock import patch

import httpx2
import pytest

API_KEY = "test-api-key"


class MockRoute:
    def __init__(self, method, url, json__eq, params__contains):
        self.method = method
        self.url = url
        self.json__eq = json__eq
        self.params__contains = params__contains
        self.responses = []
        self.called = False

    def mock(self, return_value=None, side_effect=None):
        self.responses = list(side_effect) if side_effect else [return_value]

    def matches(self, request):
        if request.method != self.method:
            return False
        if request.url.copy_with(query=None) != httpx2.URL(self.url):
            return False
        if self.json__eq is not None and json.loads(request.content) != self.json__eq:
            return False
        if self.params__contains is not None and any(
            request.url.params.get(k) != v for k, v in self.params__contains.items()
        ):
            return False
        return True

    def respond(self, request):
        self.called = True
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        response.request = request
        return response


class MockDevinApi:
    def __init__(self):
        self.routes = []
        self.calls = []

    def _route(self, method, url, json__eq=None, params__contains=None):
        route = MockRoute(method, url, json__eq, params__contains)
        self.routes.append(route)
        return route

    def get(self, url, **kwargs):
        return self._route("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._route("POST", url, **kwargs)

    def handler(self, request):
        self.calls.append(request)
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        for route in self.routes:
            if route.matches(request):
                return route.respond(request)
        pytest.fail(f"Unmocked request: {request.method} {request.url}")

    def assert_all_called(self):
        for route in self.routes:
            assert route.called, f"Route not called: {route.method} {route.url}"


def _patched_api(target):
    api = MockDevinApi()
    transport = httpx2.MockTransport(api.handler)

    def create_http_client(headers):
        return httpx2.Client(headers=headers, transport=transport)

    with patch(f"{target}.create_http_client", create_http_client):
        yield api
    api.assert_all_called()


@pytest.fixture
def mock_api():
    yield from _patched_api("llm_devin._devin")


@pytest.fixture
def mock_cli_api():
    yield from _patched_api("llm_devin._cli")
