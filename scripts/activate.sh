# scripts/activate.sh
# zsh / bash 공용, 셸 종료 안 되게 최소 옵션만 사용

# 스크립트 파일 경로 구하기
if [ -n "${BASH_SOURCE:-}" ]; then
  _SELF="${BASH_SOURCE[0]}"       # bash
elif [ -n "${ZSH_NAME:-}" ]; then
  _SELF="${(%):-%N}"              # zsh
else
  _SELF="$0"                      # fallback
fi

# 레포 루트
ROOT="$(cd "$(dirname "$_SELF")/.." && pwd)"

# 가능한 Python 선택: 3.11 우선
if command -v python3.11 >/dev/null 2>&1; then
  PYBIN="python3.11"
else
  PYBIN="python3"
fi

# venv 생성(없으면)
if [ ! -d "$ROOT/.venv" ]; then
  "$PYBIN" -m venv "$ROOT/.venv"
fi

# 활성화
# shellcheck disable=SC1090
. "$ROOT/.venv/bin/activate"

# 경로
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

echo "✅ venv on: $(python -V) | PYTHONPATH=$PYTHONPATH"
