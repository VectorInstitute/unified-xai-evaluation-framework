"""Integration test example."""

import pytest

from unified_xai_evaluation_framework.bar import bar as barfn
from unified_xai_evaluation_framework.foo import foo as foofn


@pytest.mark.integration_test()
def test_foofn_barfn(my_test_number: int) -> None:
    """Test foo and bar."""
    foobar = foofn("bar") + f" {my_test_number} " + barfn("foo")
    assert foobar == "foobar 42 barfoo"
