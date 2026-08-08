# 합성 source checkpoint를 재사용 가능한 pytest fixture로 제공합니다.
from pathlib import Path

import pytest

from k3x_ref.fixtures import write_source_checkpoint


@pytest.fixture
def synthetic_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    write_source_checkpoint(source)
    return source

