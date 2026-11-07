"""A requested release ref must supply the compiler script as well as the files."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import tempfile

from tools import build_installer as B


class BuildReferenceTests(TestCase):
    def test_compile_uses_staged_script_even_when_checkout_differs(self):
        with tempfile.TemporaryDirectory() as d:
            stage = Path(d) / 'stage'
            (stage / 'installer').mkdir(parents=True)
            script = stage / 'installer' / 'atlas.iss'
            script.write_text('; script from requested ref', encoding='utf-8')
            args = SimpleNamespace(iscc='iscc', version='0.7.0', test=False, quick=False)
            with patch.object(B, 'ROOT', str(Path(d) / 'different-checkout')), patch.object(B.subprocess, 'run') as run:
                B.compile_setup(args, stage)
            self.assertEqual(run.call_args.args[0][-1], str(script))
            self.assertTrue(run.call_args.kwargs['check'])
