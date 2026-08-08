# K3X 변환 CLI의 dry-run과 검증 경로를 검사합니다.
from pathlib import Path

from k3x_converter.cli import main


def test_dry_run_does_not_create_artifact(
    synthetic_source: Path, tmp_path: Path, capsys
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    assert main(["convert", str(synthetic_source), str(artifact), "--dry-run"]) == 0
    assert not artifact.exists()
    assert '"completed":false' in capsys.readouterr().out


def test_convert_then_validate_from_cli(
    synthetic_source: Path, tmp_path: Path, capsys
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    assert main(["convert", str(synthetic_source), str(artifact), "--chunk-bytes", "257"]) == 0
    assert main(["validate", str(artifact)]) == 0
    assert '"valid":true' in capsys.readouterr().out
