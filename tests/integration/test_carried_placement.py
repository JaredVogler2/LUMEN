"""The tests of ``test_placement_flow.py`` once more, with the job carried by the client (live placement).

Not one assertion is changed: what the server answers must be the same whether it keeps the job or the client
brings it with every request.  The host here is one application that stays, as a warm instance of a function
does; a host that forgets everything between two requests is ``test_carried_flow.py``.
"""

import pytest

from tests.carry_client import warm_client
from tests.integration.test_placement_flow import *  # noqa: F401,F403  (the tests and their fixtures)


@pytest.fixture()
def client():
    yield from warm_client()
