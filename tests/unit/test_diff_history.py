import pytest

from kitbash.critique.history import DiffHistory, DiffStatus, HistoryEntry, ScoreTrend
from kitbash.errors import PatchError
from kitbash.infra.patching import apply_diff, changed_lines, fingerprint, make_diff, parse_diff, similarity

SCRIPT = """import kitbash_bpy as kb

kb.reset_scene()
obj = kb.import_mesh()
kb.decimate(obj)
kb.fit_dimensions(obj, mode="height")
kb.origin_to_base(obj)
mat = kb.principled("wood", base_color=(0.5, 0.3, 0.1), roughness=0.6)
kb.assign(obj, mat)
kb.save_asset(obj)
"""


def test_make_and_apply_round_trip():
    new = SCRIPT.replace("roughness=0.6", "roughness=0.4")
    assert apply_diff(SCRIPT, make_diff(SCRIPT, new)) == new


def test_apply_tolerates_wrong_line_numbers_and_whitespace():
    diff = """--- a/script.py
+++ b/script.py
@@ -40,3 +40,3 @@
 kb.origin_to_base(obj)
-mat = kb.principled("wood", base_color=(0.5, 0.3, 0.1), roughness=0.6)
+mat = kb.principled("wood", base_color=(0.6, 0.35, 0.15), roughness=0.55)
 kb.assign(obj, mat)
"""
    patched = apply_diff(SCRIPT, diff)
    assert "roughness=0.55" in patched and "roughness=0.6)" not in patched


def test_apply_multiple_hunks_and_blank_context_lines():
    diff = """--- a/script.py
+++ b/script.py
@@ -1,4 +1,5 @@
 import kitbash_bpy as kb

+# asset build
 kb.reset_scene()
@@ -9,2 +10,3 @@
 kb.assign(obj, mat)
+kb.clean_mesh(obj)
 kb.save_asset(obj)
"""
    patched = apply_diff(SCRIPT, diff)
    assert patched.index("# asset build") < patched.index("kb.reset_scene()")
    assert patched.index("kb.clean_mesh(obj)") < patched.index("kb.save_asset(obj)")


def test_apply_fails_clearly_when_context_is_missing():
    diff = "@@ -1,2 +1,2 @@\n-obj = kb.load_everything()\n+obj = kb.import_mesh()\n"
    with pytest.raises(PatchError, match="Hunk 1 does not apply"):
        apply_diff(SCRIPT, diff)
    with pytest.raises(PatchError, match="no hunks"):
        parse_diff("just words")


def test_fingerprint_ignores_formatting_and_context():
    a = make_diff(SCRIPT, SCRIPT.replace("roughness=0.6", "roughness=0.4"))
    b = "@@ -8 +8 @@\n-mat = kb.principled(\"wood\",  base_color=(0.5, 0.3, 0.1), roughness=0.6)\n+mat = kb.principled(\"wood\", base_color=(0.5, 0.3, 0.1),   roughness=0.4)\n"
    assert fingerprint(a) == fingerprint(b)
    assert similarity(a, b) == 1.0
    assert "+kb.clean_mesh(obj)" not in changed_lines(a)


def entry(cycle: int, status: DiffStatus, diff: str, **kwargs) -> HistoryEntry:
    return HistoryEntry(cycle=cycle, status=status, diff=diff, **kwargs)


def test_history_detects_repeated_diffs_but_not_the_initial_script():
    initial = make_diff("", SCRIPT)
    tweak = make_diff(SCRIPT, SCRIPT.replace("roughness=0.6", "roughness=0.4"))
    history = DiffHistory()
    history.add(entry(1, DiffStatus.KEPT, initial, initial=True))
    history.add(entry(2, DiffStatus.REVERTED, tweak, score_before=0.6, score_after=0.4))
    assert history.find_repeat(initial) is None
    repeat = history.find_repeat(tweak)
    assert repeat is not None and repeat.cycle == 2 and repeat.forbidden
    other = make_diff(SCRIPT, SCRIPT.replace('mode="height"', 'mode="max"'))
    assert history.find_repeat(other) is None


def test_history_repeat_can_be_limited_to_forbidden_changes():
    kept = make_diff(SCRIPT, SCRIPT.replace("roughness=0.6", "roughness=0.4"))
    history = DiffHistory([entry(2, DiffStatus.KEPT, kept)])
    assert history.find_repeat(kept) is not None
    assert history.find_repeat(kept, include_kept=False) is None


def test_settle_updates_the_applied_entry_and_render_lists_statuses():
    history = DiffHistory()
    history.add(entry(1, DiffStatus.APPLIED, make_diff("", SCRIPT), initial=True))
    history.add(entry(2, DiffStatus.REJECTED, "@@\n-x\n+y\n", reason="does not apply"))
    history.add(entry(3, DiffStatus.APPLIED, make_diff(SCRIPT, SCRIPT + "# more\n"), score_before=0.5))
    history.settle(3, DiffStatus.REVERTED, 0.3)
    assert [e.status for e in history.entries] == [DiffStatus.APPLIED, DiffStatus.REJECTED, DiffStatus.REVERTED]
    text = history.render()
    assert "cycle 02: rejected (does not apply)" in text and "cycle 03: reverted, score 0.50 -> 0.30" in text
    assert DiffHistory().render() == "(no previous cycles)"


def test_render_truncates_old_diffs_first():
    big = "@@\n" + "\n".join(f"+line {i}" for i in range(2000)) + "\n"
    history = DiffHistory([entry(1, DiffStatus.KEPT, big), entry(2, DiffStatus.KEPT, "@@\n-a\n+b\n")])
    text = history.render(max_chars=500)
    assert "(truncated)" in text and "+b" in text


@pytest.mark.parametrize(
    "scores, stalled",
    [
        ([0.5], False),
        ([0.5, 0.6], False),
        ([0.5, 0.6, 0.7], False),
        ([0.7, 0.705, 0.69], True),
        ([0.7, 0.6, 0.65], True),
        ([0.7, 0.6, 0.8], False),
    ],
)
def test_score_trend_stall_detection(scores, stalled):
    assert ScoreTrend.of(scores, window=2, epsilon=0.01).stalled() is stalled
