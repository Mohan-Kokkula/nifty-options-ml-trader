#!/usr/bin/env bash
#
# oci_setup.sh — stand up openclaw-v9 on a fresh Oracle Cloud instance.
#
# Run as a sudo-capable user (Oracle images ship 'ubuntu' or 'opc'), NOT as
# root. Running the deploy as root is what produces root-owned files that
# later break `git pull` and the container's own log writes.
#
#   curl -fsSL https://raw.githubusercontent.com/Mohan-Kokkula/openclaw-v9-kotak/main/deploy/oci_setup.sh -o oci_setup.sh
#   less oci_setup.sh          # read it before running it
#   bash oci_setup.sh
#
# It deliberately does NOT create config/settings.env. That file holds your
# Kotak consumer key, MPIN and TOTP seed; it is not in git and no script
# should generate it. The last step prints exactly what to put there.
#
set -euo pipefail

REPO="https://github.com/Mohan-Kokkula/openclaw-v9-kotak.git"
APP_USER="trader"
APP_DIR="/home/${APP_USER}/openclaw-v9-docker"

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
die() { printf "\n\033[31mFAILED: %s\033[0m\n" "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] && die "do not run as root — use a sudo-capable user"

# ── 0. sanity: size the box before wasting 20 minutes on a build ─────────
say "checking the instance"
ARCH="$(uname -m)"
MEM_MB="$(free -m | awk '/^Mem:/{print $2}')"
DISK_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
echo "arch=${ARCH}  ram=${MEM_MB}MB  free-disk=${DISK_GB}GB"

# requirements.txt pulls torch + transformers (FinBERT, ~420MB of weights on
# first run) plus xgboost and lightgbm. The nightly retrain needs 2G on its
# own. OCI's Always Free AMD micro shape is 1 OCPU / 1GB and cannot run this.
[ "${MEM_MB}" -lt 3500 ] && die "need >=4GB RAM. OCI Always Free AMD micro (1GB) is too small —
       use the Ampere A1 (ARM) shape: 4 OCPU / 24GB is also Always Free."
[ "${DISK_GB}" -lt 25 ] && die "need >=25GB disk (268MB repo + ~6GB of images/wheels)"

if [ "${ARCH}" = "aarch64" ]; then
    echo "NOTE: ARM build. numpy/pandas/scikit-learn/xgboost/lightgbm/torch all"
    echo "      publish aarch64 wheels, but the image build takes noticeably"
    echo "      longer than x86 (torch alone is a large wheel). Expect 15-30min."
fi

# ── 1. docker ────────────────────────────────────────────────────────────
say "installing docker"
if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl git
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
else
    echo "docker already present: $(docker --version)"
fi

# ── 2. the trader user that owns and runs everything ─────────────────────
say "creating the ${APP_USER} user"
if ! id -u "${APP_USER}" >/dev/null 2>&1; then
    sudo useradd -m -s /bin/bash "${APP_USER}"
    echo "created ${APP_USER}"
else
    echo "${APP_USER} already exists"
fi
sudo usermod -aG docker "${APP_USER}"

# ── 3. clone ─────────────────────────────────────────────────────────────
# 268MB pack: models/*.pkl and data/*.csv are tracked, so the clone brings
# the trained V9 model and 11 years of bars with it. Only settings.env is
# missing.
say "cloning (268MB — models and price history are in the repo)"
if [ -d "${APP_DIR}/.git" ]; then
    echo "already cloned, pulling instead"
    sudo -u "${APP_USER}" git -C "${APP_DIR}" pull --ff-only
else
    sudo -u "${APP_USER}" git clone "${REPO}" "${APP_DIR}"
fi

# ── 4. build ─────────────────────────────────────────────────────────────
say "building the image (this is the slow part)"
sudo -u "${APP_USER}" bash -c "cd '${APP_DIR}' && docker compose build" \
    || die "build failed — on ARM, check which wheel had no aarch64 build"

# ── 5. THE OWNERSHIP STEP — do not skip ──────────────────────────────────
# The Dockerfile creates its user with `useradd -r`, a SYSTEM account whose
# UID is below 1000. The host trader is a normal UID (~1000). Same name,
# different number. docker-compose bind-mounts ./data ./logs ./models, so if
# those stay owned by the host trader the container cannot write its own log
# and crash-loops on:
#     PermissionError: [Errno 13] Permission denied: '/app/logs/<date>.log'
# Owner = container UID (bot can write). Group = host trader + g+w (you can
# still edit, and the retrain cron works). .git stays host-owned so git works.
say "aligning bind-mount ownership with the container's UID"
CUID="$(sudo -u "${APP_USER}" bash -c \
    "cd '${APP_DIR}' && docker run --rm --entrypoint id \
     \$(docker compose config --images | head -1) -u")" \
    || die "could not read the container UID"
echo "container uid = ${CUID}"
sudo mkdir -p "${APP_DIR}"/{data,logs,models}
sudo chown -R "${CUID}:${APP_USER}" "${APP_DIR}"/{data,logs,models}
sudo chmod -R g+w "${APP_DIR}"/{data,logs,models}
sudo chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}/.git"

# ── 6. firewall ──────────────────────────────────────────────────────────
# Oracle's Ubuntu images ship iptables rules that drop almost everything, and
# they are separate from the VCN Security List in the web console. BOTH must
# allow a port. The bot binds 5000 with network_mode: host.
say "firewall"
if sudo iptables -L INPUT -n 2>/dev/null | grep -q REJECT; then
    echo "Oracle's default iptables REJECT rules are present."
    echo "If you want to reach the dashboard from outside, run:"
    echo "    sudo iptables -I INPUT 5 -p tcp --dport 5000 -j ACCEPT"
    echo "    sudo netfilter-persistent save"
    echo "AND add an ingress rule for 5000 in the VCN Security List."
    echo "Leaving it closed is the safer default — use an SSH tunnel:"
    echo "    ssh -L 5000:localhost:5000 <user>@<instance-ip>"
fi

# ── 7. nightly retrain ───────────────────────────────────────────────────
say "installing the retrain cron"
CRON="30 2 * * 2-6 cd ${APP_DIR} && docker compose run --rm -T --build --name nifty-trader-retrain --entrypoint python nifty-trader scripts/retrain_weekly.py >> ${APP_DIR}/logs/retrain.log 2>&1"
( sudo -u "${APP_USER}" crontab -l 2>/dev/null | grep -v retrain_weekly; echo "${CRON}" ) \
    | sudo -u "${APP_USER}" crontab -
echo "installed (Tue-Sat 02:30 IST)"

# ── 8. what is left for a human ──────────────────────────────────────────
say "DONE — one manual step remains"
cat <<EOF

The bot will NOT start until you create ${APP_DIR}/config/settings.env.
It is not in git on purpose: it holds your broker credentials.

    sudo -u ${APP_USER} nano ${APP_DIR}/config/settings.env

Settings that matter for the current strategy:

    DRY_RUN=true                    # paper. Change only when you mean it.
    EXECUTION_MODE=futures          # trend_day_brain REFUSES to arm without
                                    # this; the same trades score PF 0.666
                                    # under options friction.
    TREND_DAY_BRAIN_ENABLED=true    # the only strategy with a confirmed
                                    # holdout edge (TEST PF 1.225, 3/3 folds)
    MAX_LOSS_PER_TRADE=3500         # 2000 blocked every entry: 65 lots x a
                                    # 4xATR stop is ~3452
    DEFAULT_QTY=65
    MAX_DAILY_LOSS=5000
    ML_MODEL_VERSION=v9
    # PILOT_TRADING_ENABLED stays absent — the pilot has no confirmed edge

Plus your Kotak Neo credentials: KOTAK_CONSUMER_KEY, KOTAK_MOBILE,
KOTAK_UCC, KOTAK_MPIN, KOTAK_TOTP_SECRET. Type them yourself; do not paste
them into a chat or a script.

Then:

    sudo -u ${APP_USER} bash -c "cd ${APP_DIR} && docker compose up -d"
    sudo -u ${APP_USER} bash -c "cd ${APP_DIR} && docker compose logs -f nifty-trader"

Healthy startup shows "Pilot entries DISABLED", the trend-day brain arming,
and NO "PILOT ENTRIES ARE ENABLED" banner.

EOF
