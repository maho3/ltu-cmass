"""Unit tests for cmass.infer.tools — specifically the kcut helpers."""

import pytest
from cmass.infer.tools import (
    kcut_dirname, resolve_kmax, _kcut_keys, _sort_kmax_keys, iter_kcuts,
    split_experiments,
)


# ---------------------------------------------------------------------------
# kcut_dirname
# ---------------------------------------------------------------------------

class TestKcutDirname:
    def test_scalar_unchanged(self):
        """Scalar kmax must produce the legacy format byte-for-byte."""
        assert kcut_dirname(0.0, 0.4) == 'kmin-0.0_kmax-0.4'
        assert kcut_dirname(0.02, 0.5) == 'kmin-0.02_kmax-0.5'
        assert kcut_dirname(0., 0.1) == 'kmin-0.0_kmax-0.1'

    def test_mapping_two_families(self):
        """Mapping with two family keys uses new readable format."""
        result = kcut_dirname(0.0, {'zPk': 0.6, 'zQk': 0.2})
        # shorter key first, alphabetical tie-break
        assert result == 'kmin-0.0_kmax-zPk=0.6__zQk=0.2'

    def test_mapping_with_default(self):
        """'default' is rendered as 'def' and comes first."""
        result = kcut_dirname(0.0, {'zPk': 0.6, 'zPk4': 0.3, 'default': 0.2})
        assert result == 'kmin-0.0_kmax-def=0.2__zPk=0.6__zPk4=0.3'

    def test_mapping_ordering_stable(self):
        """Same mapping in different insertion orders yields identical dirnames."""
        m1 = {'default': 0.2, 'zPk': 0.6, 'zPk4': 0.3}
        m2 = {'zPk4': 0.3, 'zPk': 0.6, 'default': 0.2}
        m3 = {'zPk': 0.6, 'default': 0.2, 'zPk4': 0.3}
        assert kcut_dirname(0.0, m1) == kcut_dirname(0.0, m2)
        assert kcut_dirname(0.0, m1) == kcut_dirname(0.0, m3)

    def test_mapping_default_only(self):
        """Mapping with only a default key."""
        assert kcut_dirname(0.0, {'default': 0.4}) == 'kmin-0.0_kmax-def=0.4'

    def test_mapping_exact_key_coexists_with_family(self):
        """Exact-summary key and family key can coexist without ambiguity."""
        result = kcut_dirname(0.0, {'zBk': 0.3, 'zEqBk': 0.25, 'default': 0.2})
        # 'def' first, then shorter 'zBk', then longer 'zEqBk'
        assert result == 'kmin-0.0_kmax-def=0.2__zBk=0.3__zEqBk=0.25'


# ---------------------------------------------------------------------------
# _sort_kmax_keys
# ---------------------------------------------------------------------------

class TestSortKmaxKeys:
    def test_default_first(self):
        keys = _sort_kmax_keys({'zPk4': 0.3, 'default': 0.2, 'zPk': 0.6})
        assert keys[0] == 'default'

    def test_shorter_before_longer(self):
        keys = _sort_kmax_keys({'zPk4': 0.3, 'zPk': 0.6})
        assert keys == ['zPk', 'zPk4']

    def test_alphabetical_tiebreak(self):
        # 'zPk' and 'zQk' are the same length -> alphabetical
        keys = _sort_kmax_keys({'zQk': 0.2, 'zPk': 0.6})
        assert keys == ['zPk', 'zQk']


# ---------------------------------------------------------------------------
# _kcut_keys
# ---------------------------------------------------------------------------

class TestKcutKeys:
    def test_exact_name_with_digit(self):
        assert _kcut_keys('zPk0') == ['zPk0', 'zPk', 'default']

    def test_bk_tag_stripped(self):
        assert _kcut_keys('zEqQk0') == ['zEqQk0', 'zEqQk', 'zQk', 'default']

    def test_no_digit_suffix(self):
        # family name without trailing digit
        assert _kcut_keys('zPk') == ['zPk', 'default']

    def test_default_always_last(self):
        for summ in ['Pk0', 'zQk2', 'zEqBk0']:
            assert _kcut_keys(summ)[-1] == 'default'


# ---------------------------------------------------------------------------
# resolve_kmax
# ---------------------------------------------------------------------------

class TestResolveKmax:
    def test_scalar_passthrough(self):
        assert resolve_kmax(0.4, 'zPk0') == 0.4
        assert resolve_kmax(0.2, 'zQk0') == 0.2

    def test_exact_match_wins(self):
        kmax = {'zPk4': 0.3, 'zPk': 0.6, 'default': 0.2}
        assert resolve_kmax(kmax, 'zPk4') == 0.3

    def test_family_fallback(self):
        kmax = {'zPk': 0.6, 'default': 0.2}
        assert resolve_kmax(kmax, 'zPk0') == 0.6
        assert resolve_kmax(kmax, 'zPk2') == 0.6

    def test_default_fallback(self):
        kmax = {'zPk': 0.6, 'default': 0.2}
        assert resolve_kmax(kmax, 'zQk0') == 0.2

    def test_missing_key_raises(self):
        kmax = {'zPk': 0.6}
        with pytest.raises(KeyError):
            resolve_kmax(kmax, 'zQk0')

    def test_bk_tag_stripped_fallback(self):
        # zEqQk0 -> zEqQk -> zQk (strip 'Eq')
        kmax = {'zQk': 0.3, 'default': 0.2}
        assert resolve_kmax(kmax, 'zEqQk0') == 0.3
