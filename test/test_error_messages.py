# Copyright (C) 2026 Michael Nowotny
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import pytest

import pyjags
from pyjags.console import JagsError


class TestAnnotatedErrorMessages:
    def test_syntax_error_shows_model_code(self):
        bad_code = """\
model {
    for (i in 1:N) {
        x[i] ~ this is not valid
    }
    p ~ dbeta(1, 1)
}"""
        with pytest.raises(JagsError, match="Model code:"):
            pyjags.Model(
                code=bad_code,
                data={"x": [1, 0, 1], "N": 3},
                chains=1,
                adapt=0,
            )

    def test_syntax_error_shows_arrow_at_offending_line(self):
        bad_code = """\
model {
    for (i in 1:N) {
        x[i] ~ this is not valid
    }
    p ~ dbeta(1, 1)
}"""
        with pytest.raises(JagsError, match="-->"):
            pyjags.Model(
                code=bad_code,
                data={"x": [1, 0, 1], "N": 3},
                chains=1,
                adapt=0,
            )

    def test_valid_model_no_error(self):
        # This should not raise
        model = pyjags.Model(
            code="""\
model {
    for (i in 1:N) {
        x[i] ~ dbern(p)
    }
    p ~ dbeta(1, 1)
}""",
            data={"x": [1, 0, 1], "N": 3},
            chains=1,
            adapt=0,
        )
        assert model is not None

    def test_error_from_file_has_no_annotation(self, tmp_path):
        """When loading from a file, code=None so annotation is minimal."""
        model_file = tmp_path / "bad_model.jags"
        model_file.write_text("""\
model {
    x ~ this is not valid
    p ~ dbeta(1, 1)
}""")
        with pytest.raises(JagsError):
            pyjags.Model(
                file=str(model_file),
                data={"x": [1]},
                chains=1,
                adapt=0,
            )
