#!/bin/bash
#SBATCH --job-name=k3s-smoketest
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail

# Optional env overrides:
#   export K3S_VERSION="v1.33.1+k3s1"
#   export K3S_START_TIMEOUT=240
#   export K3S_KEEP_ALIVE_SECONDS=600
K3S_VERSION="${K3S_VERSION:-}"
K3S_START_TIMEOUT="${K3S_START_TIMEOUT:-180}"
K3S_KEEP_ALIVE_SECONDS="${K3S_KEEP_ALIVE_SECONDS:-120}"
K3S_API_PORT="${K3S_API_PORT:-6443}"

JOB_ROOT="${HOME}/.local/k3s-slurm/${SLURM_JOB_ID}"
BIN_DIR="${JOB_ROOT}/bin"
DATA_DIR="${JOB_ROOT}/data"
LOG_DIR="${JOB_ROOT}/logs"
mkdir -p "${BIN_DIR}" "${DATA_DIR}" "${LOG_DIR}"

K3S_BIN="${BIN_DIR}/k3s"
KUBECONFIG_PATH="${JOB_ROOT}/kubeconfig.yaml"
K3S_LOG="${LOG_DIR}/k3s-server.log"

cleanup() {
  if [[ -n "${K3S_PID:-}" ]] && kill -0 "${K3S_PID}" 2>/dev/null; then
    kill "${K3S_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${K3S_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date)] Job ${SLURM_JOB_ID} on node $(hostname)"
echo "[$(date)] Workspace: ${JOB_ROOT}"

ARCH="$(uname -m)"
K3S_ASSET="k3s"
case "${ARCH}" in
  x86_64) K3S_ASSET="k3s" ;;
  aarch64|arm64) K3S_ASSET="k3s-arm64" ;;
  *)
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

if [[ -z "${K3S_VERSION}" ]]; then
  K3S_VERSION="$(curl -fsSL https://update.k3s.io/v1-release/channels/stable)"
fi

K3S_URL="https://github.com/k3s-io/k3s/releases/download/${K3S_VERSION}/${K3S_ASSET}"
echo "[$(date)] Downloading ${K3S_URL}"
curl -fL "${K3S_URL}" -o "${K3S_BIN}"
chmod +x "${K3S_BIN}"

echo "[$(date)] Starting k3s (ephemeral cluster in Slurm job)"
"${K3S_BIN}" server \
  --https-listen-port "${K3S_API_PORT}" \
  --write-kubeconfig "${KUBECONFIG_PATH}" \
  --write-kubeconfig-mode 600 \
  --data-dir "${DATA_DIR}" \
  --node-name "slurm-${SLURM_JOB_ID}" \
  --disable traefik \
  --disable servicelb \
  --disable local-storage \
  --flannel-backend=none \
  --disable-network-policy \
  >"${K3S_LOG}" 2>&1 &
K3S_PID=$!

echo "[$(date)] Waiting up to ${K3S_START_TIMEOUT}s for API readiness..."
START_TS="$(date +%s)"
READY=0
while true; do
  if KUBECONFIG="${KUBECONFIG_PATH}" "${K3S_BIN}" kubectl get --raw=/readyz >/dev/null 2>&1; then
    READY=1
    break
  fi
  NOW_TS="$(date +%s)"
  if (( NOW_TS - START_TS > K3S_START_TIMEOUT )); then
    break
  fi
  sleep 2
done

if (( READY == 0 )); then
  echo "k3s did not become ready. Last server log lines:" >&2
  tail -n 120 "${K3S_LOG}" >&2 || true
  exit 1
fi

echo "[$(date)] k3s is up"
echo "KUBECONFIG=${KUBECONFIG_PATH}"
KUBECONFIG="${KUBECONFIG_PATH}" "${K3S_BIN}" kubectl cluster-info
KUBECONFIG="${KUBECONFIG_PATH}" "${K3S_BIN}" kubectl get nodes -o wide || true
KUBECONFIG="${KUBECONFIG_PATH}" "${K3S_BIN}" kubectl get pods -A || true

echo "[$(date)] Keeping cluster alive for ${K3S_KEEP_ALIVE_SECONDS}s (set K3S_KEEP_ALIVE_SECONDS to change)"
sleep "${K3S_KEEP_ALIVE_SECONDS}"
echo "[$(date)] Done. Cluster will be torn down by cleanup."
