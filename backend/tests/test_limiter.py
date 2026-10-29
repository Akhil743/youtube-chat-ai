from starlette.requests import Request

from app.limiter import DailyCap, client_ip


def make_request(headers: dict, client=("10.0.0.1", 1234)):
    scope = {
        "type": "http", "method": "GET", "path": "/", "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": client,
    }
    return Request(scope)


def test_prefers_cf_connecting_ip():
    req = make_request({"CF-Connecting-IP": "203.0.113.7", "X-Forwarded-For": "198.51.100.1, 10.0.0.2"})
    assert client_ip(req) == "203.0.113.7"


def test_falls_back_to_first_forwarded_hop():
    req = make_request({"X-Forwarded-For": " 198.51.100.1 , 10.0.0.2"})
    assert client_ip(req) == "198.51.100.1"


def test_falls_back_to_socket_address():
    assert client_ip(make_request({})) == "10.0.0.1"
    assert client_ip(make_request({"X-Forwarded-For": ""})) == "10.0.0.1"


def test_no_client_at_all():
    assert client_ip(make_request({}, client=None)) == "unknown"


def test_daily_cap_blocks_then_resets_next_utc_day():
    now = [1_700_000_000.0]
    cap = DailyCap(limit=2, clock=lambda: now[0])
    assert cap.try_consume() and cap.try_consume()
    assert not cap.try_consume()
    assert cap.remaining() == 0
    now[0] += 86_400
    assert cap.remaining() == 2 and cap.try_consume()
