"""Tests for seed policy and derivation."""

import pytest

from labelforge.core.seeds import (
    SeedPolicy,
    derive_stage_seed,
    derive_row_seed,
    derive_sampling_seed,
    validate_seed,
)


class TestSeedPolicy:
    """Tests for SeedPolicy class."""

    def test_create_policy(self):
        """Policy should be creatable with valid seed."""
        policy = SeedPolicy(run_seed=42)
        assert policy.run_seed == 42

    def test_invalid_seed_negative(self):
        """Negative seeds should be rejected."""
        with pytest.raises(ValueError):
            SeedPolicy(run_seed=-1)

    def test_invalid_seed_too_large(self):
        """Seeds >= 2^32 should be rejected."""
        with pytest.raises(ValueError):
            SeedPolicy(run_seed=2**32)

    def test_derive_stage_seed_stability(self):
        """Same stage should always get same seed."""
        policy = SeedPolicy(run_seed=42)
        seed1 = policy.derive_stage_seed("caption")
        seed2 = policy.derive_stage_seed("caption")
        assert seed1 == seed2

    def test_derive_stage_seed_different_stages(self):
        """Different stages should get different seeds."""
        policy = SeedPolicy(run_seed=42)
        seed1 = policy.derive_stage_seed("caption")
        seed2 = policy.derive_stage_seed("score")
        assert seed1 != seed2

    def test_derive_row_seed_stability(self):
        """Same row should always get same seed."""
        policy = SeedPolicy(run_seed=42)
        seed1 = policy.derive_row_seed("caption", "lf_1234567890abcdef")
        seed2 = policy.derive_row_seed("caption", "lf_1234567890abcdef")
        assert seed1 == seed2

    def test_derive_row_seed_different_rows(self):
        """Different rows should get different seeds."""
        policy = SeedPolicy(run_seed=42)
        seed1 = policy.derive_row_seed("caption", "lf_1234567890abcdef")
        seed2 = policy.derive_row_seed("caption", "lf_fedcba0987654321")
        assert seed1 != seed2


class TestSeedDerivation:
    """Tests for seed derivation functions."""

    def test_stage_seed_range(self):
        """Stage seeds should be valid 32-bit integers."""
        seed = derive_stage_seed(42, "test_stage")
        assert 0 <= seed < 2**32

    def test_row_seed_range(self):
        """Row seeds should be valid 32-bit integers."""
        stage_seed = derive_stage_seed(42, "test_stage")
        seed = derive_row_seed(stage_seed, "row_123")
        assert 0 <= seed < 2**32

    def test_sampling_seed_range(self):
        """Sampling seeds should be valid 32-bit integers."""
        row_seed = 12345
        seed = derive_sampling_seed(row_seed)
        assert 0 <= seed < 2**32

    def test_sampling_seed_retry(self):
        """Different retry attempts should get different seeds."""
        row_seed = 12345
        seed0 = derive_sampling_seed(row_seed, attempt=0)
        seed1 = derive_sampling_seed(row_seed, attempt=1)
        seed2 = derive_sampling_seed(row_seed, attempt=2)
        
        assert seed0 != seed1
        assert seed1 != seed2
        assert seed0 != seed2


class TestValidateSeed:
    """Tests for seed validation."""

    def test_valid_seed(self):
        """Valid seeds should pass."""
        assert validate_seed(0) == 0
        assert validate_seed(42) == 42
        assert validate_seed(2**32 - 1) == 2**32 - 1

    def test_none_generates_random(self):
        """None should generate a valid random seed."""
        seed = validate_seed(None)
        assert 0 <= seed < 2**32

    def test_invalid_type(self):
        """Non-integer should raise TypeError."""
        with pytest.raises(TypeError):
            validate_seed("42")  # type: ignore

    def test_out_of_range(self):
        """Out of range should raise ValueError."""
        with pytest.raises(ValueError):
            validate_seed(-1)
        with pytest.raises(ValueError):
            validate_seed(2**32)
