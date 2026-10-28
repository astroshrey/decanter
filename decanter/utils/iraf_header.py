"""A FITS header as a list of card images, edited the way IRAF edits it.

IRAF tasks change a header card by card: an existing keyword keeps its
place and gets new value text, a new keyword is appended at the end of the
user area, a deleted keyword's card goes.  :class:`IrafHeader` holds the
80-character card images (a long string with ``CONTINUE`` cards stays one
entry) and implements exactly those edits, so a header built here is the
byte sequence WARP's files carry.  Values are parsed only when asked for;
numeric text is never re-formatted.

It is deliberately small (astropy's ``Header`` re-parses and re-formats
cards and is ~100x slower for this use).
"""
from __future__ import annotations

from typing import Iterator


class Card80:
    """One card: its keyword and image (80 characters, or a multiple of 80
    for a long-string card with ``CONTINUE`` cards)."""

    __slots__ = ("keyword", "image")

    def __init__(self, keyword: str, image: str):
        self.keyword = keyword
        self.image = image

    def __repr__(self) -> str:  # pragma: no cover
        return f"Card80({self.image!r})"

    @property
    def value_text(self) -> str:
        """The value field of a numeric card, as written (cols 11-30...)."""
        v = self.image[10:80]
        if "/" in v and not v.lstrip().startswith("'"):
            v = v.split("/", 1)[0]
        return v.strip()

    @property
    def value(self):
        """The parsed value (str, bool, int or float; None for no value)."""
        img = self.image
        if len(img) < 10 or img[8:10] != "= ":
            return None
        v = img[10:].lstrip()
        if v.startswith("'"):
            # a string; a long one continues over CONTINUE cards ('&' ends a piece)
            chunks = [img[k:k + 80] for k in range(0, len(img), 80)]
            text = ""
            for n, ch in enumerate(chunks):
                body = (ch[10:] if n == 0 else ch[8:]).lstrip()
                if not body.startswith("'"):
                    break
                j = 1
                piece = []
                while j < len(body):
                    if body[j] == "'":
                        if j + 1 < len(body) and body[j + 1] == "'":
                            piece.append("'")
                            j += 2
                            continue
                        break
                    piece.append(body[j])
                    j += 1
                s = "".join(piece)
                if s.endswith("&") and n + 1 < len(chunks):
                    text += s[:-1]
                    continue
                text += s
                break
            return text.rstrip()
        t = self.value_text
        if t == "T":
            return True
        if t == "F":
            return False
        try:
            return int(t)
        except ValueError:
            pass
        try:
            return float(t.replace("D", "E"))
        except ValueError:
            return t


def num_image(key: str, text: str) -> str:
    """A numeric card with IRAF's text right-justified in columns 11-30."""
    return f"{key:<8s}= {text:>20s}".ljust(80)


def str_image(key: str, value: str) -> str:
    """A string card as IRAF/astropy write it (at least 8 characters)."""
    v = str(value).replace("'", "''")
    return f"{key:<8s}= '{v:<8s}'".ljust(80)[:80]


class IrafHeader:
    """An ordered list of :class:`Card80` with IRAF's editing rules."""

    __slots__ = ("_cards",)

    def __init__(self, cards: list[Card80] | None = None):
        self._cards: list[Card80] = list(cards) if cards else []

    # -- construction --------------------------------------------------------
    @classmethod
    def from_astropy(cls, hdr) -> "IrafHeader":
        if isinstance(hdr, IrafHeader):
            return hdr.copy()
        return cls([Card80(c.keyword, c.image) for c in hdr.cards])

    @classmethod
    def from_images(cls, images: list[str]) -> "IrafHeader":
        return cls([Card80(img[:8].strip(), img) for img in images])

    def copy(self) -> "IrafHeader":
        return IrafHeader([Card80(c.keyword, c.image) for c in self._cards])

    # -- access --------------------------------------------------------------
    @property
    def cards(self) -> list[Card80]:
        return self._cards

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._cards)

    def keys(self) -> list[str]:
        return [c.keyword for c in self._cards]

    def index(self, key: str) -> int | None:
        for i, c in enumerate(self._cards):
            if c.keyword == key:
                return i
        return None

    def __contains__(self, key: str) -> bool:
        return self.index(key) is not None

    def card(self, key: str) -> Card80 | None:
        i = self.index(key)
        return None if i is None else self._cards[i]

    def get(self, key: str, default=None):
        c = self.card(key)
        return default if c is None else c.value

    def __getitem__(self, key: str):
        c = self.card(key)
        if c is None:
            raise KeyError(key)
        return c.value

    def text(self, key: str) -> str | None:
        """The value text of a numeric card (None when absent)."""
        c = self.card(key)
        return None if c is None else c.value_text

    # -- editing (IRAF semantics) ---------------------------------------------
    def append(self, card) -> None:
        if isinstance(card, Card80):
            self._cards.append(card)
        elif isinstance(card, str):
            self._cards.append(Card80(card[:8].strip(), card))
        else:                                  # an astropy Card
            self._cards.append(Card80(card.keyword, card.image))

    def put_image(self, key: str, image: str) -> None:
        """Replace ``key``'s card in place, or append it."""
        i = self.index(key)
        if i is None:
            self._cards.append(Card80(key, image))
        else:
            self._cards[i] = Card80(key, image)

    def put_num(self, key: str, text: str) -> None:
        self.put_image(key, num_image(key, text))

    def put_str(self, key: str, value: str) -> None:
        self.put_image(key, str_image(key, value))

    def delete(self, key: str) -> None:
        i = self.index(key)
        if i is not None:
            del self._cards[i]

    def set_string_keep_comment(self, key: str, value: str) -> None:
        """``impstr`` on an existing card: new value, the comment kept where
        it was (moved right, with a blank, when the value reaches it)."""
        i = self.index(key)
        if i is None:
            self.put_str(key, value)
            return
        old = self._cards[i].image[:80]
        v = str(value).replace("'", "''")
        prefix = f"{key:<8s}= '{v:<8s}'"
        slash = _comment_slash(old)
        if slash is None:
            img = prefix
        else:
            if len(prefix) < slash:
                prefix = prefix.ljust(slash)
            else:
                prefix = prefix + " "
            img = prefix + old[slash:]
        self._cards[i] = Card80(key, img.ljust(80)[:80])

    def images(self) -> list[str]:
        return [c.image for c in self._cards]


def _comment_slash(image: str) -> int | None:
    """Index of the ``/`` that starts a card's comment (outside the value)."""
    i = 10
    s = image
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] == "'":
        j = i + 1
        while j < len(s):
            if s[j] == "'":
                if j + 1 < len(s) and s[j + 1] == "'":
                    j += 2
                    continue
                break
            j += 1
        i = j + 1
    k = s.find("/", i)
    return None if k < 0 else k


def as_header(h) -> IrafHeader:
    """An :class:`IrafHeader` view of ``h`` (astropy Header or IrafHeader)."""
    if isinstance(h, IrafHeader):
        return h
    return IrafHeader.from_astropy(h)
