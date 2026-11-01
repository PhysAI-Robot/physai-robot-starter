from physai.sim.newton.core import _full_contact_params


def test_a_short_solref_and_solimp_are_filled_with_mujocos_defaults():
    """Newton reads the numbers MuJoCo's writer drops as zero, not as the defaults."""
    xml = '<geom solref="0.004" solimp="0.99 0.999"/><geom solref="0.01 0.5"/>'
    assert _full_contact_params(xml) == (
        '<geom solref="0.004 1" solimp="0.99 0.999 0.001 0.5 2"/><geom solref="0.01 0.5"/>'
    )
