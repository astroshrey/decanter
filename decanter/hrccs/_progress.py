"""Progress bars for the HRCCS pipeline.

Uses :mod:`tqdm` when it is installed (it ships only in the optional ``hrccs``
extras), and otherwise falls back to a silent no-op with the same call surface.
Progress bars are purely cosmetic, so a base install without ``tqdm`` should
degrade gracefully rather than raising :class:`ModuleNotFoundError`.
"""

try:  # pragma: no cover - trivial import branch
    from tqdm.auto import tqdm  # noqa: F401  (re-exported)
except ModuleNotFoundError:  # pragma: no cover - exercised only without tqdm

    class tqdm:  # noqa: N801 - deliberately mirrors tqdm's public name
        """Minimal drop-in stand-in for ``tqdm.auto.tqdm``.

        Supports only the surface the HRCCS code uses: iteration, ``update``,
        ``close``, ``set_postfix``/``set_postfix_str``, ``set_description``, the
        ``total``/``n`` attributes, use as a context manager, and the static
        ``write`` helper.
        """

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get("total")
            self.n = 0

        def __iter__(self):
            for item in self.iterable or ():
                self.n += 1
                yield item

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, n=1):
            self.n += n

        def close(self):
            pass

        def refresh(self):
            pass

        def set_postfix_str(self, *args, **kwargs):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def set_description(self, *args, **kwargs):
            pass

        @staticmethod
        def write(message, file=None, end="\n"):
            print(message, file=file, end=end)
