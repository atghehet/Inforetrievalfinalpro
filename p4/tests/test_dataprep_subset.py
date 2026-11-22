"""Tests for p4.dataprep.subset_unarxive (filtering the unarXive tar by arXiv category)."""
import hashlib
import io
import json
import tarfile

import pytest

from p4.dataprep import subset_unarxive as S


def record(pid, cats, padding=0):
    """One unarXive-like record. `padding` pushes the categories field beyond the first 8 KB."""
    meta = {"id": pid, "title": "t", "authors": "a" * padding, "categories": cats}
    return json.dumps({"paper_id": pid, "metadata": meta, "body_text": []}).encode() + b"\n"


def make_tar(path, members):
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tar(tmp_path):
    (tmp_path / "src").mkdir()
    return make_tar(tmp_path / "src" / "u.tar", {
        "21/arXiv_src_2101_001.jsonl": record("2101.00001", "cs.CL cs.LG") + record("2101.00002", "math.PR"),
        "21/arXiv_src_2102_001.jsonl": record("2102.00001", "cs.LG") + record("2102.00002", "cs.AI cs.CL"),
        "98/arXiv_src_9807_001.jsonl": record("hep-th/9807001", "hep-th"),
        "readme.txt": b"not a record file\n"})


def test_keeps_exactly_the_records_with_the_category_token(tar, tmp_path):
    out = tmp_path / "out"
    manifest = S.filter_tar(tar, out, "cs.CL")
    kept = [json.loads(line)["paper_id"] for f in sorted(out.rglob("*.jsonl")) for line in f.read_text().splitlines()]
    assert kept == ["2101.00001", "2102.00002"]                                    # token may be anywhere in the list
    assert (manifest["records_scanned"], manifest["records_kept"], manifest["members_scanned"]) == (5, 2, 3)
    assert manifest["kept_per_year"] == {"2021": 2} and manifest["members_with_matches"] == 2


def test_output_mirrors_member_names_and_writes_nothing_for_members_without_matches(tar, tmp_path):
    out = tmp_path / "out"
    S.filter_tar(tar, out, "cs.CL")
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "21/arXiv_src_2101_001.jsonl", "21/arXiv_src_2102_001.jsonl", "MANIFEST.json"]    # nothing for 98/ or readme.txt


def test_lines_are_copied_byte_for_byte(tar, tmp_path):
    out = tmp_path / "out"
    S.filter_tar(tar, out, "cs.CL")
    assert (out / "21/arXiv_src_2101_001.jsonl").read_bytes() == record("2101.00001", "cs.CL cs.LG")


def test_category_must_match_a_whole_token(tmp_path):
    t = make_tar(tmp_path / "x.tar", {"20/arXiv_src_2001_001.jsonl": record("a", "cs.CLX") + record("b", "xcs.CL cs.CL.sub")
                                      + record("c", "cs.CL")})
    manifest = S.filter_tar(t, tmp_path / "out", "cs.CL")
    assert manifest["records_kept"] == 1                                              # only the exact token in record c


def test_categories_found_beyond_the_header_window_use_the_fallback_and_missing_ones_are_counted(tmp_path):
    long_record = record("late", "cs.CL", padding=S.HEADER_BYTES + 500)
    no_cats = json.dumps({"paper_id": "none", "metadata": {"id": "none"}}).encode() + b"\n"
    t = make_tar(tmp_path / "x.tar", {"20/arXiv_src_2001_001.jsonl": long_record + no_cats + b"\n" + record("ok", "cs.CL")})
    m = S.filter_tar(t, tmp_path / "out", "cs.CL")
    assert m["records_kept"] == 2 and m["header_fallbacks"] == 1                      # 'late' found by the whole-line search
    assert m["n_no_categories"] == 1 and m["records_scanned"] == 3                    # reported, neither kept nor hidden


def test_duplicate_paper_ids_are_reported_but_all_lines_are_kept(tmp_path):
    t = make_tar(tmp_path / "x.tar", {"20/arXiv_src_2001_001.jsonl": record("dup", "cs.CL") * 2 + record("x", "cs.CL")})
    m = S.filter_tar(t, tmp_path / "out", "cs.CL")
    assert m["records_kept"] == 3 and m["duplicate_paper_ids_among_kept"] == 1


def test_the_source_tar_is_never_modified(tar, tmp_path):
    before = sha(tar)
    S.filter_tar(tar, tmp_path / "out", "cs.CL")
    S.filter_tar(tar, tmp_path / "out2", "cs.LG", dry_run=True)
    assert sha(tar) == before


def test_dry_run_counts_without_writing(tar, tmp_path):
    out = tmp_path / "out"
    m = S.filter_tar(tar, out, "cs.CL", dry_run=True)
    assert m["records_kept"] == 2 and m["dry_run"] and not out.exists()


def test_manifest_is_written_with_the_counts(tar, tmp_path):
    out = tmp_path / "out"
    S.filter_tar(tar, out, "cs.CL")
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["records_kept"] == 2 and manifest["category"] == "cs.CL" and manifest["source_tar_bytes"] == tar.stat().st_size


def test_existing_output_is_never_overwritten(tar, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(S.SubsetError, match="not empty, refusing to overwrite"):
        S.filter_tar(tar, out, "cs.CL")
    assert (out / "keep.txt").read_text() == "mine"
    S.filter_tar(tar, tmp_path / "fresh", "cs.CL")                                    # an empty existing folder is fine
    empty = tmp_path / "empty"
    empty.mkdir()
    assert S.filter_tar(tar, empty, "cs.CL")["records_kept"] == 2


def test_output_inside_the_source_folder_is_refused(tar):
    with pytest.raises(S.SubsetError, match="must not be"):
        S.filter_tar(tar, tar.parent, "cs.CL")


@pytest.mark.parametrize("name", ["../evil.jsonl", "/abs/evil.jsonl", "C:/evil.jsonl", "a/../../evil.jsonl"])
def test_unsafe_member_names_are_refused(tmp_path, name):
    t = make_tar(tmp_path / "x.tar", {name: record("a", "cs.CL")})
    with pytest.raises(S.SubsetError, match="unsafe member name"):
        S.filter_tar(t, tmp_path / "out", "cs.CL")
    assert not (tmp_path.parent / "evil.jsonl").exists()


def test_missing_tar_and_cli(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        S.filter_tar(tmp_path / "nope.tar", tmp_path / "out")
    assert S.main(["--tar", str(tmp_path / "nope.tar"), "--out", str(tmp_path / "out")]) == 2
    assert "error: tar not found" in capsys.readouterr().err


def test_cli_prints_the_counts(tar, tmp_path, capsys):
    assert S.main(["--tar", str(tar), "--out", str(tmp_path / "out"), "--category", "cs.CL"]) == 0
    assert '"records_kept": 2' in capsys.readouterr().out
