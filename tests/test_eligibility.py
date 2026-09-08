from __future__ import annotations

from app.eligibility.extractor import extract_crop, extract_farmer_type, extract_land_size_hectares, extract_profile_from_text, extract_state
from app.eligibility.matcher import assess_eligibility
from app.eligibility.rules import parse_eligibility_texts
from app.llm.schemas import FarmerProfile, SchemeEligibilityStatus


class TestExtractor:
    def test_extract_land_size_hectares(self):
        assert extract_land_size_hectares("I have 2 hectares of land") == 2.0

    def test_extract_land_size_acres_converted(self):
        assert extract_land_size_hectares("I farm 5 acres") == round(5 * 0.4047, 3)

    def test_extract_state(self):
        assert extract_state("I farm in Tamil Nadu") == "Tamil Nadu"

    def test_extract_crop(self):
        assert extract_crop("I grow rice and wheat") == "rice"

    def test_extract_farmer_type_compound_phrase(self):
        assert extract_farmer_type("I am a small and marginal farmer") == "small_marginal"

    def test_extract_profile_merges_onto_base(self):
        base = FarmerProfile(state="Kerala")
        profile = extract_profile_from_text("I grow rice on 1 hectare", base=base)
        assert profile.state == "Kerala"
        assert profile.crop == "rice"
        assert profile.land_size_hectares == 1.0


class TestRules:
    def test_parses_land_ceiling(self):
        criteria = parse_eligibility_texts(["Small and marginal farmers with landholding up to 2 hectares are eligible."])
        assert criteria.land_ceiling_hectares == 2.0
        assert "small and marginal farmer" in criteria.required_farmer_types or "small and marginal farmers" in criteria.required_farmer_types

    def test_parses_acre_ceiling_converted(self):
        criteria = parse_eligibility_texts(["Farmers with up to 5 acres are eligible."])
        assert criteria.land_ceiling_hectares == round(5 * 0.4047, 3)


class TestMatcher:
    ELIGIBILITY_TEXT = ["Small and marginal farmer families with combined landholding up to 2 hectares are eligible."]

    def test_likely_eligible(self):
        profile = FarmerProfile(land_size_hectares=1.0, farmer_type="small_marginal")
        result = assess_eligibility(profile, self.ELIGIBILITY_TEXT)
        assert result.status == SchemeEligibilityStatus.LIKELY_ELIGIBLE

    def test_likely_not_eligible_land_exceeds_ceiling(self):
        profile = FarmerProfile(land_size_hectares=10.0, farmer_type="small_marginal")
        result = assess_eligibility(profile, self.ELIGIBILITY_TEXT)
        assert result.status == SchemeEligibilityStatus.LIKELY_NOT_ELIGIBLE

    def test_insufficient_information_when_profile_empty(self):
        profile = FarmerProfile()
        result = assess_eligibility(profile, self.ELIGIBILITY_TEXT)
        assert result.status in (SchemeEligibilityStatus.INSUFFICIENT_INFORMATION, SchemeEligibilityStatus.POSSIBLY_ELIGIBLE)
        assert len(result.missing_information) > 0

    def test_no_eligibility_text_is_insufficient(self):
        result = assess_eligibility(FarmerProfile(land_size_hectares=1.0), [])
        assert result.status == SchemeEligibilityStatus.INSUFFICIENT_INFORMATION

    def test_never_asserts_eligible_without_matched_criteria(self):
        # Descriptive prose with no parseable ceiling/category/state/crop must not
        # be reported as LIKELY_ELIGIBLE just because a profile was supplied.
        profile = FarmerProfile(land_size_hectares=1.0, state="Kerala", crop="rice")
        result = assess_eligibility(profile, ["This scheme supports agricultural development broadly."])
        assert result.status != SchemeEligibilityStatus.LIKELY_ELIGIBLE
