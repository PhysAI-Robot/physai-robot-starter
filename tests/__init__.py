"""Makes `tests` a regular package, not a namespace package.

Do not delete: `lerobot`'s `draccus` dependency installs its own test suite
as a top-level `tests` package into site-packages (a packaging mistake on
its part). Per PEP 420, a regular package anywhere on `sys.path` wins over
a namespace-package portion regardless of path order, so without this
file, `import tests` (and every `from tests.core... import ...` absolute
import in this suite) silently resolves to draccus's installed tests
instead of this directory whenever the `vla` extra is installed. See
docs/adr/0016-isaacsim-as-a-project-extra.md.
"""
