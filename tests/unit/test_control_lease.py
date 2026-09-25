import numpy as np
import pytest

from physai.contracts import Action
from physai.web.lease import ControlLease


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def command(value: float) -> Action:
    return Action(joint_position=np.array([value]))


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def lease(clock):
    return ControlLease(("arm", "base"), timeout=0.5, clock=clock)


def test_the_first_client_to_submit_controls_the_instance(lease):
    lease.submit("arm", command(1.0), "browser")

    with pytest.raises(PermissionError, match="controlled by another client"):
        lease.submit("arm", command(2.0), "other")
    assert lease.latest("arm").joint_position[0] == 1.0


def test_the_controller_renews_and_others_wait_for_expiry(lease, clock):
    lease.submit("arm", command(1.0), "browser")
    clock.now += 0.4
    lease.submit("arm", command(2.0), "browser")  # renewed
    clock.now += 0.4
    with pytest.raises(PermissionError):
        lease.submit("arm", command(3.0), "other")

    clock.now += 0.2  # past the renewed deadline
    lease.submit("arm", command(4.0), "other")

    assert lease.latest("arm").joint_position[0] == 4.0


def test_an_expired_lease_discards_its_queued_command(lease, clock):
    lease.submit("arm", command(1.0), "browser")
    clock.now += 0.6

    assert lease.latest("arm") is None


def test_a_newer_command_replaces_the_waiting_one(lease):
    lease.submit("arm", command(1.0), "browser")
    lease.submit("arm", command(2.0), "browser")

    assert lease.latest("arm").joint_position[0] == 2.0
    assert lease.latest("arm") is None


def test_instances_are_leased_independently(lease):
    lease.submit("arm", command(1.0), "browser")

    lease.submit("base", command(2.0), "other")  # a different instance


def test_release_frees_only_what_the_source_owns(lease):
    lease.submit("arm", command(1.0), "browser")
    lease.submit("base", command(2.0), "other")

    lease.release("browser")

    assert lease.latest("arm") is None
    lease.submit("arm", command(3.0), "third")  # free again
    with pytest.raises(PermissionError):
        lease.submit("base", command(4.0), "third")  # still owned by "other"
    assert lease.latest("base").joint_position[0] == 2.0


def test_discard_all_keeps_leases_but_drops_commands(lease):
    lease.submit("arm", command(1.0), "browser")

    lease.discard_all()

    assert lease.latest("arm") is None
    with pytest.raises(PermissionError):
        lease.submit("arm", command(2.0), "other")
