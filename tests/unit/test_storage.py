import time

import pytest

from lumen.errors import LimitExceededError, NotFoundError
from lumen.storage.base import FileSessionStore, MemorySessionStore, check_blob_name, check_session_id, new_session_id


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path):
    return MemorySessionStore() if request.param == "memory" else FileSessionStore(tmp_path / "sessions")


def test_round_trip_and_blobs(store):
    sid = new_session_id()
    store.create(sid, {"id": sid, "n": 1})
    assert store.exists(sid) and store.load(sid)["n"] == 1
    store.save(sid, {"id": sid, "n": 2})
    assert store.load(sid)["n"] == 2
    store.put_blob(sid, "f001.jpg", b"\xff\xd8abc")
    assert store.get_blob(sid, "f001.jpg") == b"\xff\xd8abc"
    store.delete_blob(sid, "f001.jpg")
    with pytest.raises(NotFoundError):
        store.get_blob(sid, "f001.jpg")
    with store.lock(sid):
        store.save(sid, {"id": sid, "n": 3})
    assert store.delete(sid) and not store.exists(sid) and not store.delete(sid)
    with pytest.raises(NotFoundError):
        store.load(sid)


@pytest.mark.parametrize("bad", ["", "abc", "../../etc/passwd", "a" * 80, "aaaaaaaaaaaaaaaaaaaa/..", "aaaaaaaaaaaaaaaa\x00bbbb", None, 5])
def test_session_ids_are_validated_before_touching_storage(store, bad):
    with pytest.raises(NotFoundError):
        store.load(bad)
    with pytest.raises(NotFoundError):
        check_session_id(bad)


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "x" * 200, "a\\b", "state.json/../x"])
def test_blob_names_cannot_escape(store, bad):
    sid = new_session_id()
    store.create(sid, {})
    with pytest.raises(NotFoundError):
        store.put_blob(sid, bad, b"x")
    with pytest.raises(NotFoundError):
        check_blob_name(bad)


def test_ids_are_unguessable_and_distinct():
    ids = {new_session_id() for _ in range(200)}
    assert len(ids) == 200 and all(len(i) >= 22 for i in ids)


def test_expiry_and_capacity(store):
    a, b = new_session_id(), new_session_id()
    store.create(a, {})
    store.create(b, {})
    with pytest.raises(LimitExceededError):
        store.enforce_capacity(2)
    store.enforce_capacity(3)
    assert store.purge_expired(3600) == 0
    assert store.purge_expired(0.0, now=time.time() + 5) == 2
    assert store.list_ids() == []


def test_file_store_is_shared_between_instances(tmp_path):
    """Two worker processes are modelled by two store objects on one directory."""
    a = FileSessionStore(tmp_path / "s")
    b = FileSessionStore(tmp_path / "s")
    sid = new_session_id()
    a.create(sid, {"v": 1})
    a.put_blob(sid, "x.bin", b"123")
    assert b.load(sid) == {"v": 1} and b.get_blob(sid, "x.bin") == b"123"
    with b.lock(sid):
        b.save(sid, {"v": 2})
    assert a.load(sid) == {"v": 2}
    assert b.delete(sid) and not a.exists(sid)
    # nothing is left behind
    assert not any((tmp_path / "s").iterdir())


def test_file_store_directory_is_private(tmp_path):
    s = FileSessionStore(tmp_path / "priv")
    assert (s.root.stat().st_mode & 0o077) == 0
