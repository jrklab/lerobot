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
        │ 2.4GHz, wlx9cefd5f874c4 (RT5370 USB dongle, 192.168.1.180/24) — host's internet source
        ▼
 ┌─────────────────┐   eth (10.55.0.0/24, host=.1, pi=.102)   ┌─────────────────┐
 │     Host PC     │ ───────────────────────────────────────▶ │  Raspberry Pi   │
 │                 │   host shares wlo1 internet over this    │   (LeKiwi)      │
 │                 │   link via NAT (ipv4.method=shared)      │                 │
 │                 │                                          │                 │
 │ wlo1 (built-in) │◀──── wlan0 hotspot LeKiwi-RoboNet ────────│  wlan0 (AP)     │
 │ (10.42.0.x/24)  │      5GHz ch 149, 10.42.0.0/24, pi=.1     │  10.42.0.1/24   │
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
| Hotspot (`wlo1` ↔ `wlan0` AP `LeKiwi-RoboNet`, 5GHz channel 149) | Simulated student-device test path | `10.42.0.x/24` (DHCP) | `10.42.0.1/24` |
| Home Wi-Fi (`wlx9cefd5f874c4` RT5370 dongle ↔ `AttIsBeter`, 2.4GHz) | Host's internet | `192.168.1.180/24` (DHCP) | n/a |
| Pi's second Wi-Fi adapter (`wlan1` ↔ home router `AttIsBeter`) | Lets home-Wi-Fi devices (e.g. the Quest) reach the Pi directly, and gives the Pi a real internet route | n/a (Pi-only adapter) | `192.168.1.180/24` (DHCP, may change) |

- Pi has internet access (DNS + HTTP verified) via the host's shared Ethernet connection.
- Pi's `wlan0` broadcasts `LeKiwi-RoboNet` (WPA-PSK, password `lekiwi_test`), matching `lekiwi_web_server.md` §4.1.
- Host's second Wi-Fi interface connects to that hotspot and can reach/SSH the Pi at `10.42.0.1`, standing in for a student's laptop/tablet/phone.
- Pi is **not reachable over the home Wi-Fi via `wlan0`** — it can't be both an AP and a home-network client at once on this hardware. (Step 5 below adds a *second*, USB, Wi-Fi adapter specifically to restore home-network reachability without giving up the hotspot -- two adapters, not a limitation of `wlan0` itself.)

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
  802-11-wireless.band a \
  802-11-wireless.channel 149 \
  ipv4.method shared \
  ipv4.addresses 10.42.0.1/24

sudo nmcli connection modify LeKiwi-RoboNet \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk lekiwi_test

sudo nmcli connection up LeKiwi-RoboNet
```

**5GHz, channel 149** (changed from the original 2.4GHz `band bg` on 2026-10-23). The Pi's
Wi-Fi country must match where it runs -- it was `GB`, which doesn't allow channel 149; set it
to `US` first, or the AP won't start on that channel:

```bash
sudo raspi-config nonint do_wifi_country US
sudo iw reg set US
```

Why 149: a scan from the Pi (`nmcli -f SSID,CHAN,SIGNAL dev wifi list --rescan yes`) showed
2.4GHz crowded (~10 networks on channels 1/6/9/10/11) and 5GHz in use at 44 (home router,
TP-Link) and 132/136 (DFS), but nothing on 149-165. 149 is non-DFS (no radar wait, never
forced off-channel) and full power in the US. The Pi 4's built-in radio runs the AP at 20MHz
width. Rescan and move if the room changes. Devices without 5GHz can't join any more; going
back is `802-11-wireless.band bg 802-11-wireless.channel 0`. Both bands at once would need a
second AP-capable adapter on the Pi -- `wlan0` serves one band at a time.

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

## Step 4 — Host PC: connect a Wi-Fi interface to the hotspot

The hotspot is 5GHz, so the host joins it with its built-in dual-band `wlo1`; the RT5370
USB dongle (`wlx9cefd5f874c4`, 2.4GHz only) took over the host's home Wi-Fi instead.

```bash
sudo nmcli device wifi connect AttIsBeter password '<password>' ifname wlx9cefd5f874c4
sudo nmcli device wifi connect LeKiwi-RoboNet password lekiwi_test ifname wlo1
sudo nmcli connection modify "LeKiwi-RoboNet 2" ipv4.never-default yes ipv6.never-default yes
sudo nmcli connection up "LeKiwi-RoboNet 2"
```

`never-default`: the hotspot's DHCP also hands out a default route (via the Pi, which may
have no internet), only one metric point behind the home Wi-Fi's -- if the dongle drops,
internet traffic would go to the Pi. With it set, only `10.42.0.0/24` goes over `wlo1`.
(`LeKiwi-RoboNet 2` is the profile name NetworkManager generated here; check with
`nmcli con show`.)

Result: `wlo1` → `10.42.0.182/24` on channel 149 at 117 Mbit/s, ping to `10.42.0.1`
~6 ms average, no loss.

A dual-band AIC8800D80 USB dongle (`a69c:8d80`, ships in "driver CD" mode `a69c:5723`) was
tried first and abandoned: it needs an out-of-tree driver (signed for Secure Boot), and with
the radxa-pkg driver its firmware upload always timed out at `0x170400` (this chip reports
`chip_id=7, chip_mcu_id=1`, which needs the `legacy-mcu1` branch of
`shenmintao/aic8800d80`, not tried).

## Step 5 — Raspberry Pi: second Wi-Fi adapter for direct home-network access

Added later, for the Quest 2 VR page (`CONTROLS.md`'s "Quest 2 VR" section): a USB Wi-Fi
dongle on the Pi, joined to the home Wi-Fi (`AttIsBeter`), so the headset can reach the web
server's HTTPS port while staying on the home network instead of having to switch onto the
isolated `LeKiwi-RoboNet` hotspot. `wlan0` keeps running the hotspot exactly as before --
this is a genuinely separate interface (`wlan1`), not a mode change to `wlan0`.

```bash
sudo nmcli connection add type wifi ifname wlan1 con-name AttIsBeter \
  ssid AttIsBeter wifi-sec.key-mgmt wpa-psk wifi-sec.psk '<password>'
sudo nmcli connection up AttIsBeter
```

Gotcha: plain `nmcli device wifi connect AttIsBeter password '<password>' ifname wlan1`
failed with `802-11-wireless-security.key-mgmt: property is missing` -- `device wifi
connect`'s auto-detection of the security type didn't work reliably for this adapter/AP
combination. Creating the connection profile directly with `wifi-sec.key-mgmt`/`wifi-sec.psk`
explicit (as above) worked first try.

Result: `wlan1` → `192.168.1.180/24` (DHCP from the home router; may change on lease
renewal), reachable from any other device on the same home Wi-Fi. The self-signed TLS cert
(`examples/lekiwi/web_server/certs/`) was regenerated to add this IP to its `subjectAltName`
list, avoiding an extra hostname-mismatch note on top of the usual self-signed warning.

Removing the dongle is safe at any time, including across a reboot: the `AttIsBeter`
connection is bound to the interface *name* `wlan1`, so with the dongle absent NetworkManager
has nothing to activate for that profile -- no retry loop, no boot delay. `lekiwi-web.service`
only depends on `network.target` (generic "networking subsystem present"), not
`network-online.target` (which actually waits for a live connection and could stall on a
missing device), so the service starts immediately regardless. Falls back to hotspot-only
access exactly as it was before this step. Plugging the dongle back in reconnects
automatically (`autoconnect` defaults to on).

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
- **Hotspot was 2.4GHz `bg`-only** until 2026-10-23 (now 5GHz, Step 3), and that was
  a likely contributor to jerky real-time control over the web app: driving
  the base from a phone connected to `LeKiwi-RoboNet` showed real WebSocket
  message delivery gaps (target ~33ms at 30Hz, observed gaps up to ~245ms
  during active use), while the same control loop felt smooth over the
  Ethernet-connected host. Root cause wasn't fully isolated (could be
  2.4GHz band/congestion, or bandwidth contention with the MJPEG video
  stream sharing the same link, or phone-side browser/CPU throttling) —
  deprioritized for now since it doesn't block development, but worth
  revisiting before this goes in front of students.
  Not yet re-measured from a phone on the 5GHz hotspot.
