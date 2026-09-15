#!/usr/bin/env bash
#
# hostinger_setup.sh — stand up openclaw on a fresh Hostinger VPS.
#
# SSH into your Hostinger VPS, then:
#   curl -fsSL https://raw.githubusercontent.com/Mohan-Kokkula/nifty-options-ml-trader/psar-engine/deploy/hostinger_setup.sh -o hostinger_setup.sh
#   less hostinger_setup.sh
#   bash hostinger_setup.sh
#
set -euo pipefail

REPO="https://github.com/Mohan-Kokkula/nifty-options-ml-trader.git"
BRANCH="psar-engine"
APP_DIR="/home/trader/openclaw"

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
die() { printf "\n\033[31mFAILED: %s\033[0m\n" "$*" >&2; exit 1; }

# ── 0. Check resources ─────────────────────────────────────────────────
say "checking system"
MEM_MB="$(free -m | awk '/^Mem:/{print $2}')"
DISK_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
echo "ram=${MEM_MB}MB  free-disk=${DISK_GB}GB"
[ "${MEM_MB}" -lt 1800 ] && die "need >=2GB RAM (4GB recommended for torch/FinBERT)"
[ "${DISK_GB}" -lt 15 ] && die "need >=15GB free disk"

# ── 1. Install Docker ──────────────────────────────────────────────────
say "installing docker"
if ! command -v docker >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl git
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | tee /etc/apt/sources.list.d/docker.list >/dev/null
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
else
    echo "docker already present: $(docker --version)"
fi

# ── 2. Create trader user ──────────────────────────────────────────────
say "creating trader user"
if ! id -u trader >/dev/null 2>&1; then
    useradd -m -s /bin/bash trader
    echo "created trader"
else
    echo "trader already exists"
fi
usermod -aG docker trader

# ── 3. Clone repo ──────────────────────────────────────────────────────
say "cloning ${BRANCH} branch"
if [ -d "${APP_DIR}/.git" ]; then
    echo "already cloned, pulling"
    sudo -u trader git -C "${APP_DIR}" fetch origin
    sudo -u trader git -C "${APP_DIR}" checkout "${BRANCH}"
    sudo -u trader git -C "${APP_DIR}" pull origin "${BRANCH}"
else
    sudo -u trader git clone -b "${BRANCH}" "${REPO}" "${APP_DIR}"
fi

# ── 4. Build Docker image ─────────────────────────────────────────────
say "building docker image (this takes 5-15 min)"
sudo -u trader bash -c "cd '${APP_DIR}' && docker compose build" \
    || die "build failed"

# ── 5. Fix permissions for bind mounts ─────────────────────────────────
say "fixing bind-mount permissions"
CUID="$(sudo -u trader bash -c \
    "cd '${APP_DIR}' && docker run --rm --entrypoint id \
     \$(docker compose config --images | head -1) -u")" \
    || die "could not read container UID"
echo "container uid = ${CUID}"
mkdir -p "${APP_DIR}"/{data,logs,models}
chown -R "${CUID}:trader" "${APP_DIR}"/{data,logs,models}
chmod -R g+w "${APP_DIR}"/{data,logs,models}
chown -R trader:trader "${APP_DIR}/.git"

# ── 6. Set timezone ────────────────────────────────────────────────────
say "setting timezone to IST"
timedatectl set-timezone Asia/Kolkata 2>/dev/null || ln -sf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime

# ── 7. Done ────────────────────────────────────────────────────────────
say "DONE — create your config and start"
cat <<EOF

Next steps:

1. Create config/settings.env:

    sudo -u trader nano ${APP_DIR}/config/settings.env

   Required settings:

    DRY_RUN=true
    KOTAK_CONSUMER_KEY=<your_key>
    KOTAK_MOBILE=<your_mobile>
    KOTAK_UCC=<your_ucc>
    KOTAK_MPIN=<your_mpin>
    KOTAK_TOTP_SECRET=<your_totp_secret>
    DEFAULT_QTY=65
    MAX_DAILY_LOSS=5000
    MAX_LOSS_PER_TRADE=3500
    ML_MODEL_VERSION=v9

2. Start the bot:

    sudo -u trader bash -c "cd ${APP_DIR} && docker compose up -d"

3. Check logs:

    sudo -u trader bash -c "cd ${APP_DIR} && docker compose logs -f nifty-trader"

4. Stop:

    sudo -u trader bash -c "cd ${APP_DIR} && docker compose down"

EOF
