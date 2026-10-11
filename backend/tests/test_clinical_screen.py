from ml.clinical_screen import findrisc_screen


def test_findrisc_young_active_low():
    out = findrisc_screen(age=32, bmi=22.0, sex="F", physically_active=True, family_history=False)
    assert out["score"] == 0
    assert out["tier"] == "low"


def test_findrisc_older_obese_family_high():
    out = findrisc_screen(
        age=66,
        bmi=32.0,
        sex="M",
        waist_cm=110,
        physically_active=False,
        family_history=True,
    )
    assert out["score"] == 18
    assert out["tier"] == "high"
