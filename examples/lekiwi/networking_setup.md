# LeKiwi Web Server Project — Phase 1: Networking Setup

Companion doc to `lekiwi_web_server.md`. Covers the networking topology from
section 4 of that spec: getting the Pi reachable for development (internet
access via the host PC over Ethernet) while also standing up the
`LeKiwi-RoboNet` hotspot so a "student device" can reach the future web
server, without the two networks colliding.

## Physical / topology overview

```
Home Wi-Fi router (AttIsBeter)
        │
        │ wlo1 (192.168.1.120/24) — host's internet source
        ▼
 ┌─────────────────┐   eth (10.55.0.0/24, host=.1, pi=.102)   ┌─────────────────┐
 │     Host PC     │ ───────────────────────────────────────▶ │  Raspberry Pi   │
 │                 │   host shares wlo1 internet over this    │   (LeKiwi)      │
 │                 │   link via NAT (ipv4.method=shared)      │                 │
 │                 │                                          │                 │
 │ wlx9cefd5f874c4 │◀──── wlan0 hotspot LeKiwi-RoboNet ────────│  wlan0 (AP)     │
 │ (10.42.0.63/24) │      (10.42.0.0/24, pi=.1)                │  10.42.0.1/24   │
 └─────────────────┘      "student device" test path           └─────────────────┘
```

Two separate subnets are used on purpose — `10.55.0.0/24` for the
host↔Pi Ethernet link, `10.42.0.0/24` for the Pi's hotspot — so that the
host PC (which has an interface on *both* networks) never has two
interfaces claiming the same subnet at once. Using the same subnet on both
would leave the host's routing table ambiguous about which interface owns
that range.

## Result

| Link | Purpose | Host-side IP | Pi-side IP |
|---|---|---|---|
| Ethernet (`enxe8ea6a951df5` ↔ `eth0`) | Dev/management access + internet sharing | `10.55.0.1/24` | `10.55.0.102/24` |
| Hotspot (`wlx9cefd5f874c4` ↔ `wlan0` AP `LeKiwi-RoboNet`) | Simulated student-device test path | `10.42.0.63/24` | `10.42.0.1/24` |

- Pi has internet access (DNS + HTTP verified) via the host's shared Ethernet connection.
- Pi's `wlan0` broadcasts `LeKiwi-RoboNet` (WPA-PSK, password `lekiwi_test`), matching `lekiwi_web_server.md` §4.1.
- Host's second Wi-Fi interface connects to that hotspot and can reach/SSH the Pi at `10.42.0.1`, standing in for a student's laptop/tablet/phone.
- Pi is **no longer reachable over the home Wi-Fi** — `wlan0` can't be both an AP and a home-network client at once on this hardware, so that path was intentionally given up in favor of the Ethernet link as the new management path.

## Prerequisites: passwordless sudo for network commands

Both machines needed passwordless `sudo` scoped to just the networking
tools, so the commands below could run without an interactive password
prompt (only needed if you want an agent/script driving this; a human at
the keyboard can just type the password each time instead).

Host PC (`/etc/sudoers.d/hao-nmcli`):
```
hao ALL=(ALL) NOPASSWD: /usr/bin/nmcli, /usr/sbin/nft, /usr/sbin/iptables
```
Created via:
```bash
echo "hao ALL=(ALL) NOPASSWD: /usr/bin/nmcli, /usr/sbin/nft, /usr/sbin/iptables" | sudo tee /etc/sudoers.d/hao-nmcli
sudo chmod 0440 /etc/sudoers.d/hao-nmcli
sudo visudo -c   # validates syntax
```

The Pi already had passwordless sudo configured for its user (default on
most Pi OS images), so no equivalent step was needed there.

## Step 1 — Host PC: share Ethernet internet to the Pi

The host's Ethernet interface (`enxe8ea6a951df5`) was set to `ipv4.method
auto` (DHCP client) by default, which meant it was waiting for a DHCP
server that didn't exist — the Pi's `eth0` was in the same state. Neither
side would ever get an address. Fix: make the host the DHCP/NAT server on
that link.

```bash
sudo nmcli connection modify "Wired connection 1" \
  ipv4.method shared \
  ipv4.addresses 10.55.0.1/24

sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"
```

`ipv4.method shared` makes NetworkManager: assign the static address, turn
on `ip_forward`, run a `dnsmasq` DHCP/DNS server on that interface, and add
NAT/masquerade + forwarding rules for that subnet automatically.

## Step 2 — Raspberry Pi: pick up the new DHCP lease

The Pi's `eth0` was already `ipv4.method auto` (correct — it should stay a
DHCP client), it just needed to reconnect to actually request a lease now
that the host is serving one:

```bash
sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"
```

Result: `eth0` → `10.55.0.102/24`, default route via `10.55.0.1`.

Verified internet access from the Pi:
```bash
curl -s -m 5 -o /dev/null -w 'HTTP %{http_code}\n' http://1.1.1.1   # HTTP 301
getent hosts github.com                                            # resolves fine
```

Note: right after bringing the link up, a `ping`/`curl` attempt hung with
no response for a bit — NAT/forwarding rules and conntrack were still
settling. It resolved itself within a few seconds; if it doesn't, check
`sudo nft list ruleset` on the host for the `nm-sh-fw-<iface>` forward
chain and the `ip nat POSTROUTING` masquerade rule.

## Step 3 — Raspberry Pi: create the `LeKiwi-RoboNet` hotspot
Password: lekiwi_test

```bash
sudo nmcli connection add type wifi ifname wlan0 con-name LeKiwi-RoboNet \
  autoconnect no ssid LeKiwi-RoboNet

sudo nmcli connection modify LeKiwi-RoboNet \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  ipv4.method shared \
  ipv4.addresses 10.42.0.1/24

sudo nmcli connection modify LeKiwi-RoboNet \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk lekiwi_test

sudo nmcli connection up LeKiwi-RoboNet
```

Gotcha: `wifi-sec.*` is *not* a valid shorthand for `nmcli connection
modify` in this NetworkManager version — it must be the full property name
`802-11-wireless-security.*`.

Activating this profile drops `wlan0`'s existing client connection to the
home Wi-Fi immediately (expected — see topology note above). Any SSH
session running over that old path dies right when you run `connection
up`; reconnect via the Ethernet IP (`10.55.0.102`) instead.

`autoconnect no` is deliberate for now (see the note in chat / top of this
doc) — flip it once you want the hotspot to come up automatically on
boot:
```bash
sudo nmcli connection modify LeKiwi-RoboNet connection.autoconnect yes
```

## Step 4 — Host PC: connect the second Wi-Fi interface to the hotspot

```bash
sudo nmcli device wifi rescan ifname wlx9cefd5f874c4
sudo nmcli device wifi connect LeKiwi-RoboNet password lekiwi_test ifname wlx9cefd5f874c4
```

Result: `wlx9cefd5f874c4` → `10.42.0.63/24`, confirmed `ping`/`ssh` to
`10.42.0.1` (the Pi).

## Known rough edges

- **SSH connect latency to the Pi is inconsistent** — anywhere from ~6s to
  a full timeout on an otherwise-healthy link (ping and raw TCP connect to
  port 22 are both instant). Not yet root-caused; it's not CPU load (`
  uptime` showed load average ~0.2 during a slow connection). Worth
  revisiting before this matters for a live WebSocket control loop —
  possible causes to check next: SSH client/server key-exchange entropy
  stalls, `GSSAPIAuthentication`, or something in `sshd_config`.
- New IPs meant new SSH host-key prompts (`10.55.0.102`, `10.42.0.1`) —
  accepted with `-o StrictHostKeyChecking=accept-new`. Fine for a lab
  setup; revisit if this machine's identity needs to be pinned down for
  anything security-sensitive later.
- **Hotspot is 2.4GHz `bg`-only** (`802-11-wireless.band bg`), and this is
  a likely contributor to jerky real-time control over the web app: driving
  the base from a phone connected to `LeKiwi-RoboNet` showed real WebSocket
  message delivery gaps (target ~33ms at 30Hz, observed gaps up to ~245ms
  during active use), while the same control loop felt smooth over the
  Ethernet-connected host. Root cause wasn't fully isolated (could be
  2.4GHz band/congestion, or bandwidth contention with the MJPEG video
  stream sharing the same link, or phone-side browser/CPU throttling) —
  deprioritized for now since it doesn't block development, but worth
  revisiting before this goes in front of students.
  **The Pi does support 5GHz** (`Raspberry Pi 4 Model B`, BCM43438 chip;
  `nmcli -f WIFI-PROPERTIES.5GHZ device show wlan0` → `yes`), so switching
  the hotspot to `802-11-wireless.band a` is a real option to try — no
  hardware limitation in the way.
