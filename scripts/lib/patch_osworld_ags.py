"""Add Tencent Cloud Agent Sandbox (AGS) as an OSWorld provider on our pinned OSWorld checkout.

The AGS cookbook (TencentCloudAgentRuntime/ags-cookbook, examples/osworld-ags) ships an overlay for an older
OSWorld (84aee655); the vendored checkout is 044bc339. The overlay's provider package applies unchanged, and the only edits it needs in
OSWorld itself are two registrations, so we apply exactly those (idempotent) instead of replacing whole files:
  desktop_env/providers/ags/          copied from the cookbook
  desktop_env/providers/__init__.py   `elif provider_name == "ags": ... AGSVMManager(), AGSProvider()`
  desktop_env/desktop_env.py          "ags" in the set of providers that start from a clean state
"""
import re
import shutil
import sys
from pathlib import Path

osw, overlay = Path(sys.argv[1]), Path(sys.argv[2])
src = overlay / "desktop_env" / "providers" / "ags"
dst = osw / "desktop_env" / "providers" / "ags"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)

init = osw / "desktop_env" / "providers" / "__init__.py"
s = init.read_text()
if 'provider_name == "ags"' not in s:
    block = ('    elif provider_name == "ags":\n'
             '        from desktop_env.providers.ags.manager import AGSVMManager\n'
             '        from desktop_env.providers.ags.provider import AGSProvider\n'
             '        return AGSVMManager(), AGSProvider()\n')
    m = list(re.finditer(r"\n    else:\n", s))
    if not m:
        raise SystemExit("cannot find the provider factory's else-branch in providers/__init__.py")
    k = m[-1].start() + 1
    s = s[:k] + block + s[k:]
    init.write_text(s)

de = osw / "desktop_env" / "desktop_env.py"
s = de.read_text()
pat = re.compile(r'(if self\.provider_name in \{"docker", "aws"[^}]*)\}')
if '"ags"' not in s:
    s, n = pat.subn(lambda m: m.group(1) + ', "ags"}', s, count=1)
    if n != 1:
        raise SystemExit("cannot find the clean-start provider set in desktop_env.py")
    de.write_text(s)
# Keep-alive in the cookbook's local proxy: it opens a new aiohttp ClientSession (a new TLS handshake to the
# sandbox region) for every HTTP request, which costs up to a second per screenshot / action on a distant region.
# Reuse one session per event loop instead. Same requests and responses; only connection reuse changes.
prov = dst / "provider.py"
s = prov.read_text()
if "_nightwatch_session" not in s:
    helper = (
        "\n\n_NIGHTWATCH_SESSIONS = {}\n\n\n"
        "async def _nightwatch_session():\n"
        "    \"\"\"One keep-alive ClientSession per event loop (patched in by nightwatch; see scripts/lib/patch_osworld_ags.py).\"\"\"\n"
        "    loop = asyncio.get_running_loop()\n"
        "    s = _NIGHTWATCH_SESSIONS.get(loop)\n"
        "    if s is None or s.closed:\n"
        "        s = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=64, keepalive_timeout=120))\n"
        "        _NIGHTWATCH_SESSIONS[loop] = s\n"
        "    return s\n"
    )
    anchor = "    AIOHTTP_AVAILABLE = False\n"
    i = s.index(anchor)
    j = s.index("\n\n", i)
    s = s[:j] + helper + s[j:]
    s, n = re.subn(r"async with aiohttp\.ClientSession\(\) as session:\n(\s+)async with session\.request",
                   r"for session in [await _nightwatch_session()]:  # nightwatch: keep-alive\n\1async with session.request", s)
    if n < 1:
        raise SystemExit("proxy request pattern not found in provider.py")
    prov.write_text(s)

# Revert = kill. The cookbook's revert_to_snapshot only logs "not supported, skipping"; DesktopEnv.reset then
# calls _start_emulator, which creates a new sandbox and drops the reference to the old one, which keeps running
# (and billing) until AGS_TIMEOUT. Stop it first, like the docker provider does.
s = prov.read_text()
old = '        logger.warning("AGS snapshot revert not supported, skipping...")\n'
new = ('        # nightwatch: kill the current sandbox; DesktopEnv.reset starts a fresh one right after\n'
       '        self.stop_emulator(path_to_vm)\n')
if old in s:
    prov.write_text(s.replace(old, new))
elif new not in s:
    raise SystemExit("revert_to_snapshot body not found in provider.py")
# Home ownership. Some AGS OSWorld templates ship /home/user and /home/user/.config owned
# by root (775) while OSWorld Server runs as the guest user, so every task that uploads a file straight into
# /home/user (most LibreOffice tasks) fails at setup with "Permission denied". Give the top level of the home
# back to the server's user, as in the official images; nothing else changes.
s = prov.read_text()
anchor = '        run_as_root("true", "validate sandbox sudo access")\n'
fix = ('        run_as_root("find /home/user -maxdepth 1 -user root -exec chown --reference=/home/user/server {} +",\n'
       '                    "give /home/user back to the guest user")  # nightwatch: template home owned by root\n')
if fix not in s:
    if anchor not in s:
        raise SystemExit("sudo validation step not found in provider.py")
    prov.write_text(s.replace(anchor, anchor + fix, 1))
print(f"AGS provider installed into {osw}")
