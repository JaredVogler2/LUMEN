"""The tests of ``test_calibration_flow.py`` once more, with the workspace carried by the client.

A lens calibration is a workspace like a job: its board photos and what was measured in them are kept for as long
as it takes.  Where the browser carries the job it carries this too, and nothing may come out differently.
"""

import pytest

from tests.carry_client import warm_client
from tests.integration.test_calibration_flow import *  # noqa: F401,F403  (the tests and their fixtures)


@pytest.fixture()
def client():
    yield from warm_client()
