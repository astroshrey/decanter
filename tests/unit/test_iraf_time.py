"""IRAF's clock and date/time texts (decanter.utils.iraf_time) and the stamps the
WARP-tree writer builds with them: ``FIXPIX``'s run time and the ``# <cnvtime>``
line of the databases IRAF writes.

The expected texts are what the compiled IRAF wrote on this machine
(iraf-community 2.18.1 through pyraf, 2026-10-22): ``proto.fixpix`` (x_proto.e)
and ``apextract.apfind`` (x_apextract.e, ``ap_dbwrite`` -> ``dtptime``) run
under each time zone below at the Unix time given; and WARP's own trees.
Until 2026-10-22 decanter wrote ``FIXPIX``'s time with
``time.strftime("%b %e %H:%M")``: a zero-padded hour (``'Oct 22 00:03'`` where
IRAF writes ``'Oct 22  0:03'``) on the wall clock (daylight saving applied,
which IRAF's clock never applies).
"""
from __future__ import annotations

import calendar
import os
import random
import re
import time

import pytest

from decanter.utils import iraf_time as it

S1980 = 315532800                         # seconds 1970-01-01 -> 1980-01-01 (UTC)


@pytest.fixture
def tz():
    """Set the process time zone for one test (``TZ`` + ``tzset``), restored after."""
    old = os.environ.get("TZ")

    def set_tz(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield set_tz
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()


def lt(y, mo, d, h=0, mi=0, s=0) -> int:
    """IRAF's ltime (seconds since 1980-01-01 00:00:00 on IRAF's clock) of a broken-down time."""
    return calendar.timegm((y, mo, d, h, mi, s)) - S1980


# --- brktime: IRAF's calendar --------------------------------------------------

def _py_fields(ltime: int):
    g = time.gmtime(ltime + S1980)
    return (g.tm_sec, g.tm_min, g.tm_hour, g.tm_mday, g.tm_mon, g.tm_year,
            (g.tm_wday + 1) % 7 + 1, g.tm_yday)          # Python Mon == 0 -> IRAF Sun == 1


def test_brktime_is_the_calendar_from_1980_to_2099():
    cases = [0, 1, 59, 60, 3599, 3600, 86399, 86400]
    for y in range(1980, 2100):                           # every month boundary, both sides
        for mo in range(1, 13):
            t = lt(y, mo, 1)
            cases += [t - 1, t, t + 43200]
    rng = random.Random(1980)
    cases += [rng.randrange(0, lt(2100, 1, 1)) for _ in range(3000)]
    for t in cases:
        if t >= 0:                                        # IRAF: valid from 1980
            assert it.brktime(t) == _py_fields(t), t
    # before 1980 IRAF's truncating "/" and "mod" give what SPP gives (sec -1), not 1979
    assert it.brktime(-1) == (-1, 0, 0, 1, 1, 1980, 3, 1)


def test_brktime_fields_and_weekday():
    # 1980-01-01 was a Tuesday (Sun == 1 -> Tue == 3)
    assert it.brktime(0) == (0, 0, 0, 1, 1, 1980, 3, 1)
    assert it.brktime(lt(2024, 2, 29, 23, 59, 58)) == (58, 59, 23, 29, 2, 2024, 5, 60)
    assert [it.cnvtime(lt(2026, 10, 18 + k, 12))[:3] for k in range(7)] == \
        ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


# --- cnvdate / cnvtime: the texts ---------------------------------------------

@pytest.mark.parametrize("fields, text", [
    ((2026, 10, 22, 0, 3), "Oct 22  0:03"),     # the 2026-10-22 campaign's hour (IRAF's form)
    ((2026, 10, 22, 2, 52), "Oct 22  2:52"),
    ((2026, 10, 22, 9, 59), "Oct 22  9:59"),
    ((2026, 10, 22, 10, 0), "Oct 22 10:00"),
    ((2026, 10, 21, 23, 55), "Oct 21 23:55"),
    ((2026, 7, 3, 10, 33), "Jul  3 10:33"),     # WARP TOI406 NO1 (day space-padded)
    ((2026, 6, 26, 8, 26), "Jun 26  8:26"),     # WARP Qatar6 NO1
    ((2026, 1, 1, 0, 0), "Jan  1  0:00"),
    ((2026, 12, 31, 23, 59), "Dec 31 23:59"),
])
def test_cnvdate_time_of_day_is_space_padded(fields, text):
    t = lt(*fields)
    assert it.cnvdate(t, one_year_ago=t - 1) == text
    assert len(text) == 12                                # SZ_DATE
    # decanter's text until 2026-10-22, of the same fields: differs exactly for hours 0-9
    old = time.strftime("%b %e %H:%M", time.gmtime(t + S1980))
    assert (old != text) == (fields[3] < 10)


def test_cnvdate_year_form_at_and_before_one_year_ago():
    t = lt(2025, 7, 9, 14, 5)
    assert it.cnvdate(t, one_year_ago=t) == "Jul  9  2025"           # not later than: the year
    assert it.cnvdate(t, one_year_ago=t - 1) == "Jul  9 14:05"
    now = lt(2026, 10, 22, 2, 52)
    # one_year_ago = clktime(0) - 3600 * 24 * (365 - 31): 334 days back
    assert it.cnvdate(now - 334 * 86400, one_year_ago=now - 334 * 86400) == "Nov 22  2025"


def test_cnvdate_default_threshold_gives_the_time_of_day_for_now(tz):
    tz("Etc/GMT+10")
    t = it.clktime()
    assert re.fullmatch(r"[A-Z][a-z]{2} [ 123]\d [ 12]\d:[0-5]\d", it.cnvdate(t))


@pytest.mark.parametrize("fields, text", [
    ((2026, 10, 22, 2, 52, 42), "Thu 02:52:42 22-Oct-2026"),
    ((2026, 7, 3, 10, 33, 26), "Fri 10:33:26 03-Jul-2026"),     # WARP TOI406 apTOI406_NO1_s
    ((2026, 6, 26, 8, 26, 4), "Fri 08:26:04 26-Jun-2026"),      # WARP Qatar6 apQatar-6_NO1_s
    ((1980, 1, 1, 0, 0, 0), "Tue 00:00:00 01-Jan-1980"),
])
def test_cnvtime_is_zero_padded(fields, text):
    assert it.cnvtime(lt(*fields)) == text
    assert len(text) == 24                                # SZ_TIME


# --- clktime: IRAF's clock against the compiled IRAF ----------------------------

#: proto.fixpix's FIXPIX text (before " Bad pixel file is ...") at Unix time ``gmt`` under ``TZ``
FIXPIX_IRAF = [
    ("Etc/GMT+12", 1792666533, "Oct 21 22:55"),
    ("Etc/GMT+11", 1792666534, "Oct 21 23:55"),
    ("Etc/GMT+10", 1792666535, "Oct 22  0:55"),
    ("Etc/GMT+9", 1792666536, "Oct 22  1:55"),
    ("Etc/GMT+8", 1792666537, "Oct 22  2:55"),
    ("Etc/GMT+7", 1792666538, "Oct 22  3:55"),
    ("Etc/GMT+6", 1792666539, "Oct 22  4:55"),
    ("Etc/GMT+5", 1792666540, "Oct 22  5:55"),
    ("Etc/GMT+4", 1792666541, "Oct 22  6:55"),
    ("Etc/GMT+3", 1792666542, "Oct 22  7:55"),
    ("Etc/GMT+2", 1792666542, "Oct 22  8:55"),
    ("Etc/GMT+1", 1792666543, "Oct 22  9:55"),
    ("Etc/GMT", 1792666544, "Oct 22 10:55"),
    ("Etc/GMT-1", 1792666545, "Oct 22 11:55"),
    ("Etc/GMT-2", 1792666546, "Oct 22 12:55"),
    ("Etc/GMT-3", 1792666547, "Oct 22 13:55"),
    ("Etc/GMT-4", 1792666548, "Oct 22 14:55"),
    ("Etc/GMT-5", 1792666549, "Oct 22 15:55"),
    ("Etc/GMT-6", 1792666549, "Oct 22 16:55"),
    ("Etc/GMT-7", 1792666550, "Oct 22 17:55"),
    ("Etc/GMT-8", 1792666551, "Oct 22 18:55"),
    ("Etc/GMT-9", 1792666552, "Oct 22 19:55"),
    ("Etc/GMT-10", 1792666553, "Oct 22 20:55"),
    ("Etc/GMT-11", 1792666554, "Oct 22 21:55"),
    ("Etc/GMT-12", 1792666554, "Oct 22 22:55"),
    ("Etc/GMT-13", 1792666555, "Oct 22 23:55"),
    ("Etc/GMT-14", 1792666556, "Oct 23  0:55"),
    # daylight saving in effect (wall clock 03:55 PDT, 21:55 AEDT, 07:55 -03, 08:26 NDT,
    # 11:56 BST, 00:41 +1345): IRAF reads standard time
    ("America/Los_Angeles", 1792666557, "Oct 22  2:55"),
    ("Australia/Sydney", 1792666558, "Oct 22 20:55"),
    ("America/Santiago", 1792666559, "Oct 22  6:55"),
    ("America/St_Johns", 1792666562, "Oct 22  7:26"),
    ("Europe/London", 1792666563, "Oct 22 10:56"),
    ("Pacific/Chatham", 1792666564, "Oct 22 23:41"),
    # standard offset changed since 1980 (now +05:45, +13, -04; then +05:30, -11, -04):
    # IRAF keeps January 1980's
    ("Asia/Kathmandu", 1792666560, "Oct 22 16:26"),
    ("Pacific/Apia", 1792666560, "Oct 21 23:56"),
    ("America/Caracas", 1792666565, "Oct 22  6:56"),
    ("Asia/Kolkata", 1792666561, "Oct 22 16:26"),
    ("Asia/Pyongyang", 1792666565, "Oct 22 19:56"),
    ("America/Los_Angeles", 1792666566, "Oct 22  2:56"),      # TZ unset: /etc/localtime
]

#: apextract's database first line (ap_dbwrite -> dtptime) at Unix time ``gmt`` under ``TZ``
DTPTIME_IRAF = [
    ("Etc/GMT+10", 1792666597, "# Thu 00:56:37 22-Oct-2026"),
    ("Etc/GMT+3", 1792666598, "# Thu 07:56:38 22-Oct-2026"),
    ("Etc/GMT-9", 1792666599, "# Thu 19:56:39 22-Oct-2026"),
    ("America/Los_Angeles", 1792666600, "# Thu 02:56:40 22-Oct-2026"),
    ("Asia/Kathmandu", 1792666600, "# Thu 16:26:40 22-Oct-2026"),
    ("Pacific/Apia", 1792666601, "# Wed 23:56:41 21-Oct-2026"),
    ("Australia/Sydney", 1792666602, "# Thu 20:56:42 22-Oct-2026"),
    ("America/Los_Angeles", 1792666603, "# Thu 02:56:43 22-Oct-2026"),
]


@pytest.mark.parametrize("zone, gmt, text", FIXPIX_IRAF)
def test_fixpix_time_is_the_compiled_irafs(tz, monkeypatch, zone, gmt, text):
    tz(zone)
    assert it.cnvdate(it.clktime(gmt)) == text
    monkeypatch.setattr(it, "_now", lambda: gmt + 0.5)
    assert it.fixpix_time() == text


@pytest.mark.parametrize("zone, gmt, text", DTPTIME_IRAF)
def test_dtptime_is_the_compiled_irafs(tz, monkeypatch, zone, gmt, text):
    tz(zone)
    monkeypatch.setattr(it, "_now", lambda: gmt + 0.25)
    assert it.dtptime() == text
    from decanter.warptree_products import iraf_dtime
    assert iraf_dtime() == text


def test_clktime_is_standard_time_all_year(tz):
    """gmt_to_lst: one offset all year (January 1980's), no daylight saving."""
    tz("America/Los_Angeles")
    assert it.lst_epoch() == S1980 + 8 * 3600
    for mo in range(1, 13):
        g = calendar.timegm((2026, mo, 15, 12, 0, 0))
        assert it.clktime(g) == g - S1980 - 8 * 3600
    tz("Etc/GMT-3")
    assert it.clktime(S1980) == 3 * 3600


def test_the_old_strftime_text_differs_from_iraf_where_the_campaign_failed(tz):
    """2026-10-22 00:03:47 PDT (TOI406's phase-1 launch): f49c5d4 wrote 'Oct 22 00:03';
    IRAF writes 'Oct 21 23:03' (PST); at 01:03:47 PDT IRAF writes 'Oct 22  0:03'."""
    tz("America/Los_Angeles")
    g = calendar.timegm((2026, 10, 22, 7, 3, 47))
    assert time.strftime("%b %e %H:%M", time.localtime(g)) == "Oct 22 00:03"
    assert it.cnvdate(it.clktime(g)) == "Oct 21 23:03"
    assert it.cnvdate(it.clktime(g + 3600)) == "Oct 22  0:03"
    assert time.strftime("%b %e %H:%M", time.localtime(g + 3600)) == "Oct 22 01:03"
