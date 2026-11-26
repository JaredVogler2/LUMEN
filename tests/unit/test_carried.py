"""The store that keeps nothing (``lumen/storage/carried.py``): one request's content, and the envelope it travels in."""

import contextvars
import hashlib
import json
import threading
import zlib

import pytest

from lumen.errors import LumenError, NotFoundError
from lumen.storage.carried import (
    Carried, CarriedStore, PicturesNeeded, begin, current, end, pack, read_request, require_carrier, sign, unpack, write_response,
)

SID = "A" * 24
OTHER = "B" * 24


def within(carried, fn):
    token = begin(carried)
    try:
        return fn()
    finally:
        end(token)


def test_outside_a_request_there_is_no_job_and_none_can_be_made():
    store = CarriedStore()
    assert current() is None and store.exists(SID) is False and store.list_ids() == []
    for call in (lambda: store.load(SID), lambda: store.create(SID, {"id": SID}), lambda: store.save(SID, {"id": SID}), lambda: store.get_blob(SID, "f001.jpg"), lambda: store.put_blob(SID, "f001.jpg", b"x")):
        with pytest.raises(NotFoundError) as e:
            call()
        assert e.value.code == "session_not_found" and "travels with the phone" in e.value.message
    assert store.purge_expired(1.0) == 0 and store.enforce_capacity(1) is None
    with store.lock(SID):
        pass


def test_a_request_sees_what_it_brought_and_records_what_it_changed():
    store = CarriedStore()
    c = Carried(states={SID: {"id": SID, "n": 1}}, blobs={(SID, "f001.jpg"): b"one"})

    def serve():
        assert store.exists(SID) and not store.exists(OTHER)
        d = store.load(SID)
        d["n"] = 99                                  # what is handed out is a copy
        assert store.load(SID)["n"] == 1
        with pytest.raises(NotFoundError):
            store.load(OTHER)
        with pytest.raises(NotFoundError):
            store.save(OTHER, {"id": OTHER})
        assert store.get_blob(SID, "f001.jpg") == b"one"
        with pytest.raises(PicturesNeeded) as e:
            store.get_blob(SID, "f002.jpg")
        assert e.value.http_status == 428 and e.value.details == {"id": SID, "blobs": ["f002.jpg"]}
        with pytest.raises(NotFoundError):
            store.get_blob(SID, "../secret")
        with pytest.raises(NotFoundError):
            store.put_blob(OTHER, "f001.jpg", b"x")    # a picture for a job that is not here
        store.put_blob(SID, "f002.jpg", b"two")
        assert store.get_blob(SID, "f002.jpg") == b"two"
        store.delete_blob(SID, "f001.jpg")
        with pytest.raises(PicturesNeeded):
            store.get_blob(SID, "f001.jpg")
        store.save(SID, {"id": SID, "n": 2})
        store.create(OTHER, {"id": OTHER})
        with pytest.raises(FileExistsError):
            store.create(OTHER, {"id": OTHER})

    within(c, serve)
    assert c.changed == {SID, OTHER} and c.states[SID]["n"] == 2 and c.added == {(SID, "f002.jpg"): b"two"} and c.dropped == {(SID, "f001.jpg")}
    assert current() is None, "nothing is left when the request is over"

    def end_it():
        assert store.delete(SID) is True and store.delete(SID) is False and not store.exists(SID)
        with pytest.raises(NotFoundError):
            store.load(SID)

    within(c, end_it)
    assert c.removed == {SID} and SID not in c.changed and not [k for k in c.blobs if k[0] == SID]


def test_two_requests_at_once_do_not_see_each_other():
    store = CarriedStore()
    seen: dict[str, object] = {}
    gate = threading.Barrier(2)

    def serve(name, sid):
        def body():
            gate.wait(timeout=5)
            seen[name] = (store.exists(SID), store.exists(OTHER), store.load(sid)["who"])

        contextvars.copy_context().run(lambda: within(Carried(states={sid: {"id": sid, "who": name}}), body))

    threads = [threading.Thread(target=serve, args=("a", SID)), threading.Thread(target=serve, args=("b", OTHER))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {"a": (True, False, "a"), "b": (False, True, "b")}


def test_a_request_is_taken_apart_as_it_was_put_together():
    state = {"id": SID, "frames": [{"id": "f001", "blob": "f001.jpg"}], "text": "Fläche ±0,5 mm"}
    raw = json.dumps(state, ensure_ascii=False).encode()
    pic, body = b"\xff\xd8picture", b'{"frame_id":"f001"}'
    for encoding, sent in (("identity", raw), ("deflate", zlib.compress(raw, 6))):
        header = {
            "method": "POST", "path": f"api/sessions/{SID}/verify", "content_type": "application/json",
            "states": [{"id": SID, "size": len(sent), "encoding": encoding}], "blobs": [{"id": SID, "name": "f001.jpg", "size": len(pic)}], "body": len(body),
        }
        head, carried, got = read_request(pack(header, [sent, pic, body]))
        assert head["method"] == "POST" and carried.states == {SID: state} and carried.blobs == {(SID, "f001.jpg"): pic} and got == body
        assert carried.body_sha256 == hashlib.sha256(body).hexdigest() and not carried.changed
    # a state that would not stop growing when it is unpacked
    bomb = zlib.compress(b"0" * (9 * 1024 * 1024), 9)
    with pytest.raises(LumenError) as e:
        read_request(pack({"method": "GET", "path": "api/sessions", "states": [{"id": SID, "size": len(bomb), "encoding": "deflate"}], "blobs": [], "body": 0}, [bomb, b""]))
    assert e.value.code == "state_too_large"


def test_the_answer_carries_what_changed_and_the_pictures_the_client_does_not_have():
    upload = b"\xff\xd8the photo that was sent"
    c = Carried(states={SID: {"id": SID, "n": 1}, OTHER: {"id": OTHER}}, body_sha256=hashlib.sha256(upload).hexdigest())
    c.states[SID]["n"] = 2
    c.changed.add(SID)
    c.added[(SID, "f001.jpg")] = upload                  # stored from the request's own body
    c.added[(SID, "f002.jpg")] = b"made on the server"   # made here (a demo photo)
    c.dropped.add((SID, "f000.jpg"))
    c.removed.add(OTHER)
    out = write_response(200, "application/json", b'{"ok":true}', c, headers={"x-lumen-rect": "{}"})
    head, rest = unpack(out)
    assert head["status"] == 200 and head["content_type"] == "application/json" and head["headers"] == {"x-lumen-rect": "{}"}
    assert [s["id"] for s in head["states"]] == [SID] and head["removed"] == [OTHER]
    assert head["kept"] == [{"id": SID, "name": "f001.jpg", "sha256": hashlib.sha256(upload).hexdigest()}], "the client has this picture: only its name goes back"
    assert [(b["name"], b["size"]) for b in head["blobs"]] == [("f002.jpg", 18)] and head["dropped"] == [{"id": SID, "name": "f000.jpg"}]
    n = head["states"][0]["size"]
    assert json.loads(bytes(rest[:n])) == {"id": SID, "n": 2} and bytes(rest[n : n + 18]) == b"made on the server" and bytes(rest[n + 18 :]) == b'{"ok":true}'
    # with a secret every state is signed, and the signature is for this job and these bytes
    signed, _ = unpack(write_response(200, "application/json", b"", c, secret="k"))
    sig = signed["states"][0]["sig"]
    raw = bytes(rest[:n])
    assert sig == sign("k", SID, raw) and sig != sign("k", OTHER, raw) and sig != sign("other", SID, raw) and sig != sign("k", SID, raw + b" ")


def test_steps_that_outlive_their_request_are_refused_where_the_job_is_carried():
    from lumen.storage.base import MemorySessionStore

    require_carrier(MemorySessionStore())            # a server that keeps the job: nothing to refuse
    with pytest.raises(LumenError) as e:
        require_carrier(CarriedStore())
    assert e.value.code == "not_available_here" and e.value.http_status == 409
