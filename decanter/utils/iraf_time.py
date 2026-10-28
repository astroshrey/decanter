"""IRAF's clock and its date/time texts, for the run-time stamps IRAF tasks write.

Two stamps in a WARP tree are written by IRAF itself:

* ``FIXPIX = '<cnvdate> Bad pixel file is <mask>'`` -- ``proto.fixpix``
  (pkg/proto/t_fixpix.x:124-125: ``cnvdate (clktime(0), ...)``), carried by
  every product made from the fixed frames;
* the first line ``# <cnvtime>`` of an aperture database IRAF writes (apall's
  sky databases, apscatter's ``ap<OBJ>_NO<i>_s``, ``aplast``; ecidentify's
  ``ec<comp>`` in a calibration set): ``dtptime`` (pkg/xtools/dttext.x:564-576:
  ``cnvtime (clktime (0), ...)``) from ``ap_dbwrite`` / ``ec_dbwrite``.

The stamps WARP's own Python writes (``time.ctime()`` in ``status.txt``,
``datetime.datetime.today()`` in an SNratio ``.dat``, the UTC ``time.strftime``
of ``aperture.py:154``) are not IRAF's and are not made here.

This module ports the IRAF routines (iraf-community 2.18.1, the build WARP runs):

* :func:`clktime` -- ``clktime(0)`` (sys/etc/clktime.x) is ``ZGTIME``
  (unix/os/zgtime.c), ``gmt_to_lst (time(0))`` (unix/os/gmttolst.c): the seconds
  of ``time(0)`` minus ``mktime`` of 00:00:00 01-Jan-1980 with ``tm_isdst = 0``.
  IRAF's clock is therefore local STANDARD time with the zone's offset of
  January 1980: daylight saving time is never applied (during PDT it reads
  PST, one hour behind the wall clock) and a zone whose standard offset
  changed since 1980 keeps the old one.
* :func:`brktime` -- sys/etc/brktime.x: that count broken into the fields of
  ``<time.h>`` by IRAF's own calendar (a leap year every fourth year; IRAF
  states it valid from 1980 to 28-Feb-2100), with SPP's truncating ``/`` and
  ``mod``.
* :func:`cnvdate` -- sys/etc/cnvdate.x: ``"%3.3s %2d "`` (month, day) and then
  ``"%2d:%02d"`` (hour, minute) for a time later than ``one_year_ago`` (set to
  ``clktime(0) - 3600 * 24 * (365 - 31)`` on the process's first call), else
  ``"%5d"`` (year).  Day AND hour are space-padded: ``'Oct 22  2:52'``,
  ``'Jul  9  2025'`` -- not ``time.strftime("%b %e %H:%M")``, whose ``%H`` is
  zero-padded (decanter wrote ``'Oct 22 00:03'`` until 2026-10-22).
* :func:`cnvtime` -- sys/etc/cnvtime.x:
  ``"%3.3s %02d:%02d:%02d %02d-%3.3s-%04d"``, ``'Thu 02:52:42 22-Oct-2026'``,
  with IRAF's own English day and month names (no locale).

Checked against the compiled IRAF: ``proto.fixpix`` run through pyraf under
33 time zones at once (every IRAF hour 0-23, single-digit hours, DST and
southern-hemisphere zones, offsets changed since 1980) writes exactly
``cnvdate(clktime())`` of the same second (tests/unit/test_iraf_time.py keeps
the cases).
"""
from __future__ import annotations

import time

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
_MONDAY = 2                                    # brktime.x: define MONDAY 2
_MONTH = "JanFebMarAprMayJunJulAugSepOctNovDec"
_WEEKDAY = "SunMonTueWedThuFriSat"
_DAYS_PER_MONTH = (31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

#: the system clock IRAF's ``time(0)`` reads (a seam for the tests)
_now = time.time


def _sdiv(a: int, b: int) -> int:
    """SPP integer ``/`` (Fortran: truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _smod(a: int, b: int) -> int:
    """SPP ``mod`` (Fortran MOD: the sign of ``a``)."""
    return a - _sdiv(a, b) * b


def lst_epoch() -> int:
    """``mktime`` of 00:00:00 01-Jan-1980 with ``tm_isdst = 0`` in the current
    time zone (``TZ``): what ``gmt_to_lst`` subtracts."""
    return int(time.mktime((1980, 1, 1, 0, 0, 0, 0, 1, 0)))


def clktime(gmt: float | None = None) -> int:
    """``clktime(0)``: seconds since 00:00:00 01-Jan-1980 local standard time.

    ``gmt`` is ``time(0)``, the Unix time (default: now).
    """
    t = int(_now() if gmt is None else gmt)
    return t - lst_epoch()


def brktime(ltime: int) -> tuple[int, int, int, int, int, int, int, int]:
    """``brktime``: ``(sec, min, hour, mday, month, year, wday, yday)``, i.e.
    ``TM_SEC .. TM_YDAY`` of ``<time.h>`` (month 1-12, wday Sun == 1)."""
    ltime = int(ltime)
    seconds = _smod(ltime, SECONDS_PER_DAY)
    days = _sdiv(ltime, SECONDS_PER_DAY) + 1

    hour = _sdiv(seconds, SECONDS_PER_HOUR)
    temp = seconds - hour * SECONDS_PER_HOUR
    minute = _sdiv(temp, SECONDS_PER_MINUTE)
    sec = temp - minute * SECONDS_PER_MINUTE

    wday = _smod(days + _MONDAY, 7)
    if wday == 0:
        wday = 7

    temp = 0
    year = 1980
    days_per_year = 366
    while days > temp + days_per_year:
        temp = temp + days_per_year
        year = year + 1
        days_per_year = 366 if _smod(year, 4) == 0 else 365
    yday = days - temp

    nights = yday
    dpm = list(_DAYS_PER_MONTH)
    dpm[1] = 29 if _smod(year, 4) == 0 else 28
    temp = 0
    month = 1
    while nights > temp + dpm[month - 1]:
        temp = temp + dpm[month - 1]
        month = month + 1
    mday = nights - temp
    return sec, minute, hour, mday, month, year, wday, yday


def _month(m: int) -> str:
    return _MONTH[(m - 1) * 3:(m - 1) * 3 + 3]


def cnvdate(ltime: int, one_year_ago: int | None = None) -> str:
    """``cnvdate``: ``'Mmm dd hh:mm'`` with day and hour space-padded, or
    ``'Mmm dd  yyyy'`` for a time not later than ``one_year_ago``.

    ``one_year_ago`` defaults to ``clktime() - 3600 * 24 * (365 - 31)`` now
    (IRAF fixes it at the process's first call; for a stamp of the current
    time, as ``fixpix`` writes, either way gives the time of day).
    """
    if one_year_ago is None:
        one_year_ago = clktime() - 3600 * 24 * (365 - 31)
    _sec, minute, hour, mday, month, year, _wday, _yday = brktime(ltime)
    # the SPP format strings of cnvdate.x, verbatim
    s = "%3.3s %2d " % (_month(month), mday)  # noqa: UP031
    if ltime > one_year_ago:
        return s + "%2d:%02d" % (hour, minute)  # noqa: UP031
    return s + "%5d" % year  # noqa: UP031


def cnvtime(ltime: int) -> str:
    """``cnvtime``: ``'Dow hh:mm:ss dd-Mmm-yyyy'`` (zero-padded)."""
    sec, minute, hour, mday, month, year, wday, _yday = brktime(ltime)
    return "%3.3s %02d:%02d:%02d %02d-%3.3s-%04d" % (  # noqa: UP031 (cnvtime.x verbatim)
        _WEEKDAY[(wday - 1) * 3:(wday - 1) * 3 + 3], hour, minute, sec, mday,
        _month(month), year)


def fixpix_time() -> str:
    """The run time ``proto.fixpix`` puts at the start of ``FIXPIX``:
    ``cnvdate (clktime(0))`` now."""
    return cnvdate(clktime())


def dtptime() -> str:
    """The first line of a database IRAF writes (``dtptime``, without the
    newline): ``'# ' + cnvtime (clktime (0))`` now."""
    return "# " + cnvtime(clktime())


__all__ = ["brktime", "clktime", "cnvdate", "cnvtime", "dtptime", "fixpix_time", "lst_epoch"]
