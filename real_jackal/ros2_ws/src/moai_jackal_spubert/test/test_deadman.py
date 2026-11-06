import pytest

from moai_jackal_spubert.deadman import Deadman


def test_press_release_timeout_and_invalid_button_message():
    gate = Deadman(1, 0.35)
    assert not gate.ready(0)
    gate.observe([0, 1], 1)
    assert gate.ready(1.3)
    assert not gate.ready(1.36)
    assert not gate.ready(0.9)
    gate.observe([0, 0], 2)
    assert not gate.ready(2)
    gate.observe([1], 3)
    assert not gate.ready(3)


@pytest.mark.parametrize("button,timeout", [(-1, 0.35), (0, 0), (0, float("nan"))])
def test_invalid_deadman_configuration_fails_closed(button, timeout):
    with pytest.raises(ValueError):
        Deadman(button, timeout)
