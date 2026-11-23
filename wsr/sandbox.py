"""Where an agent's code runs: an isolated, CPU-only sandbox with read-only weights.

Runtimes (``sandbox.runtime`` in the config):

* ``docker``: one container per session, ``--network none`` (the built-in harness calls the model from the host,
  so the sandbox needs no network), no GPU, weights and edits mounted read-only. Recommended.
* ``apptainer``: the same on HPC clusters (``--containall --cleanenv``, no ``--nv``).
* ``local``: runs commands directly on this machine, in the session directory. Nothing is isolated: use it only
  for trusted agents or quick tests.

Inside a container the agent sees fixed paths: ``/work`` (its writable session directory), ``/weights/target``
(the target checkpoint), ``/interventions`` (the edit files to read) and ``/opt/wsr`` (the ``W`` helper).
With ``local`` the same roles point to the real directories.
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

CONTAINER_PATHS = {"work": "/work", "target": "/weights/target", "interventions": "/interventions", "pkg": "/opt/wsr",
                   "home": "/home/agent"}
# ``userns``: for running wsr itself inside a container (or any host) where no container runtime is available. Each
# command runs in fresh user, mount, network and PID namespaces (util-linux ``unshare``): no network, the mounts below
# on a tmpfs at /mnt (read-only where asked), the host paths in ``sandbox.hide`` covered by empty tmpfs mounts (default
# USERNS_HIDE; list your own data directories there), GPU device files replaced by /dev/null, and the command run by an
# unprivileged user in a nested user namespace, so it cannot undo any of this. Needs unprivileged user namespaces.
USERNS_PATHS = {"work": "/mnt/work", "target": "/mnt/weights/target", "interventions": "/mnt/interventions",
                "pkg": "/mnt/wsr", "home": "/mnt/home"}
USERNS_HIDE = ["/home", "/root", "/tmp", "/var/tmp", "/scratch", "/data", "/srv", "/media"]
MAX_OUTPUT = 20000


def clip(text: str, n: int = MAX_OUTPUT) -> str:
    if len(text) <= n:
        return text
    return text[: n // 2] + f"\n\n[... {len(text) - n} characters omitted ...]\n\n" + text[-n // 2:]


class Sandbox:
    """``mounts`` maps a role (work, target, interventions, pkg, home, or any name) to (host path, read_only)."""

    def __init__(self, cfg: dict, mounts: dict[str, tuple[str, bool]], env: dict[str, str] | None = None,
                 network: bool = False):
        sb = cfg["sandbox"]
        self.sb, self.runtime = sb, sb["runtime"]
        self.mounts = {k: (str(Path(h).resolve()), ro) for k, (h, ro) in mounts.items()}
        self.network = network
        self.container = None
        if self.runtime == "userns":
            self.paths = {k: USERNS_PATHS.get(k, f"/mnt/{k}") for k in self.mounts}
        else:
            self.paths = ({k: CONTAINER_PATHS.get(k, f"/mnt/{k}") for k in self.mounts} if self.runtime != "local"
                          else {k: h for k, (h, _) in self.mounts.items()})
        self._script = None
        work = self.paths["work"]
        self.python = "/usr/bin/python3" if self.runtime != "local" else sys.executable
        self.env = {"HOME": self.paths.get("home", work), "PATH": f"{work}/bin:/usr/local/bin:/usr/bin:/bin",
                    "PYTHONPATH": self.paths.get("pkg", ""), "WSR_TARGET_DIR": self.paths.get("target", ""),
                    "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
        if self.runtime == "local":
            self.env["PATH"] = f"{work}/bin:{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '/usr/bin:/bin')}"

    # ---- lifecycle
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self) -> None:
        if self.runtime == "docker" and self.container is None:
            name = f"wsr-{uuid.uuid4().hex[:10]}"
            cmd = ["docker", "run", "-d", "--rm", "--name", name, "-w", self.paths["work"]]
            if not self.network:
                cmd += ["--network", "none"]
            cmd += self._docker_mounts() + [x for k, v in self.env.items() for x in ("-e", f"{k}={v}")]
            cmd += list(self.sb.get("docker_args") or []) + [self.sb["image"], "sleep", "infinity"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"could not start the sandbox container: {r.stderr.strip()}")
            self.container = name

    def stop(self) -> None:
        if self._script and os.path.exists(self._script):
            os.remove(self._script)
            self._script = None
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
            self.container = None

    def _docker_mounts(self) -> list[str]:
        return [x for k, (h, ro) in self.mounts.items() for x in ("-v", f"{h}:{self.paths[k]}" + (":ro" if ro else ""))]

    def _apptainer_prefix(self) -> list[str]:
        binds = ",".join(f"{h}:{self.paths[k]}" + (":ro" if ro else "") for k, (h, ro) in self.mounts.items() if k != "home")
        cmd = ["apptainer", "exec", "--containall", "--cleanenv", "--pwd", self.paths["work"]]
        if "home" in self.mounts:
            cmd += ["--home", f"{self.mounts['home'][0]}:{self.paths['home']}"]
        return cmd + ["--bind", binds] + list(self.sb.get("apptainer_args") or []) + [self.sb["image"]]

    # ---- running code
    def exec(self, command: str, timeout: float = 600) -> tuple[int, str]:
        """Run a shell command in the session directory. -> (exit code, combined output)."""
        t = max(5, int(timeout))
        if self.runtime == "userns":
            argv = ["unshare", "-rnm", "--pid", "--fork", "--mount-proc", "--propagation", "private",
                    "bash", self._userns_script(), str(t), command]
            env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
        elif self.runtime == "docker":
            self.start()
            argv = ["docker", "exec", "-w", self.paths["work"], self.container, "timeout", "-k", "5", str(t),
                    "bash", "-lc", command]
            env = None
        elif self.runtime == "apptainer":
            argv = self._apptainer_prefix() + ["timeout", "-k", "5", str(t), "bash", "-lc", command]
            env = {**os.environ, **{f"APPTAINERENV_{k}": v for k, v in self.env.items()}}
        else:
            argv, env = ["bash", "-c", command], {**self.env, "HOME": self.env["HOME"]}
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                 cwd=self.mounts["work"][0] if self.runtime == "local" else None,
                                 start_new_session=True, text=True, errors="replace")
            out, _ = p.communicate(timeout=t + 15)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, rc = (p.communicate()[0] or "") + f"\n[killed: the command ran longer than {t} s]", 124
        if rc == 124 and "[killed" not in out:
            out += f"\n[killed: the command ran longer than {t} s]"
        return rc, out

    def _userns_script(self) -> str:
        """The namespace set-up script (kept in a host temporary file the sandboxed command cannot see)."""
        if self._script and os.path.exists(self._script):
            return self._script
        import tempfile
        L = ["set -e", "T=$1; CMD=$2", "mount -t tmpfs -o mode=755 tmpfs /mnt"]
        for k, (h, ro) in self.mounts.items():
            p = shlex.quote(self.paths[k])
            L += [f"mkdir -p {p}", f"mount --bind {shlex.quote(h)} {p}"]
            if ro:
                L.append(f"mount -o remount,bind,ro {p}")
        for d in self.sb.get("hide") or USERNS_HIDE:
            L.append(f"if [ -d {shlex.quote(d)} ]; then mount -t tmpfs -o mode=755 tmpfs {shlex.quote(d)}; fi")
        L.append("for d in /dev/nvidia* /dev/dri/*; do [ -c \"$d\" ] && mount --bind /dev/null \"$d\" || true; done")
        envs = " ".join(shlex.quote(f"{k}={v}") for k, v in self.env.items())
        L.append(f"cd {shlex.quote(self.paths['work'])}")
        L.append(f"exec unshare -U --map-user=1000 --map-group=1000 env -i {envs} timeout -k 5 \"$T\" bash -c \"$CMD\"")
        fd, path = tempfile.mkstemp(prefix="wsr_userns_", suffix=".sh")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(L) + "\n")
        self._script = path
        return path

    def cli(self, argv: list[str], extra_env: dict[str, str] | None = None) -> tuple[list[str], dict]:
        """Command + environment that run an agent CLI inside the sandbox (external harnesses)."""
        env = {**self.env, **(extra_env or {})}
        if self.runtime == "apptainer":
            return self._apptainer_prefix() + argv, {**os.environ, **{f"APPTAINERENV_{k}": v for k, v in env.items()}}
        if self.runtime == "docker":
            cmd = ["docker", "run", "--rm", "-i", "-w", self.paths["work"]] + self._docker_mounts()
            cmd += [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
            return cmd + list(self.sb.get("docker_args") or []) + [self.sb["image"]] + argv, dict(os.environ)
        return argv, {**os.environ, **env}

    def host_path(self, agent_path: str) -> Path | None:
        """The host file behind a path the agent names, if it lies in its writable session directory."""
        work_agent, work_host = self.paths["work"], self.mounts["work"][0]
        p = agent_path if os.path.isabs(agent_path) else os.path.join(work_agent, agent_path)
        p = os.path.normpath(p)
        if p == work_agent or p.startswith(work_agent.rstrip("/") + "/"):
            return Path(work_host) / os.path.relpath(p, work_agent)
        return None

    def readable_host_path(self, agent_path: str) -> Path | None:
        """The host file behind any mounted path the agent names (read access)."""
        p = os.path.normpath(agent_path if os.path.isabs(agent_path) else os.path.join(self.paths["work"], agent_path))
        for k, (h, _ro) in self.mounts.items():
            a = self.paths[k]
            if p == a or p.startswith(a.rstrip("/") + "/"):
                return Path(h) / os.path.relpath(p, a)
        return None


def install_python_logger(work_host: Path, sb: Sandbox) -> None:
    """Every Python invocation of the agent is logged (code and arguments) under work/logs/code."""
    work = sb.paths["work"]
    wrapper = ("#!/bin/sh\nts=$(date +%s%N)\n"
               f"if [ -f \"$1\" ]; then cp \"$1\" {work}/logs/code/${{ts}}_$(basename \"$1\"); fi\n"
               f"echo \"$ts $*\" >> {work}/logs/code/argv.log\nexec {shlex.quote(sb.python)} \"$@\"\n")
    (work_host / "bin").mkdir(parents=True, exist_ok=True)
    (work_host / "logs" / "code").mkdir(parents=True, exist_ok=True)
    for name in ["python", "python3"]:
        tmp = work_host / "bin" / f".{name}.tmp"       # write then rename: the old file may still be executing
        tmp.write_text(wrapper)
        tmp.chmod(0o755)
        os.replace(tmp, work_host / "bin" / name)


def install_helper(dest: Path) -> Path:
    """Copy the read-only weight helper (``from wsr.sandbox_lib import W``) into a directory for the sandbox."""
    pkg = dest / "wsr"
    pkg.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    (pkg / "__init__.py").write_text("")
    for f in ["checkpoint.py", "sandbox_lib.py"]:
        shutil.copy(here / f, pkg / f)
    return dest


def check_runtime(cfg: dict) -> tuple[bool, str]:
    rt = cfg["sandbox"]["runtime"]
    if rt == "userns":
        return (bool(shutil.which("unshare")), "userns (unshare)" if shutil.which("unshare") else "`unshare` not found")
    if rt == "local":
        return True, "local (not isolated: agents run code directly on this machine)"
    exe = "docker" if rt == "docker" else "apptainer"
    if not shutil.which(exe):
        return False, f"`{exe}` is not installed"
    if rt == "docker":
        r = subprocess.run(["docker", "image", "inspect", cfg["sandbox"]["image"]], capture_output=True, text=True)
        if r.returncode != 0:
            msg = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown error"
            if "permission denied" in msg.lower():
                return False, "no permission to use Docker (add yourself to the `docker` group)"
            return False, f"image `{cfg['sandbox']['image']}` not found: docker build -t {cfg['sandbox']['image']} sandbox/"
        return True, f"docker, image {cfg['sandbox']['image']}"
    img = cfg["sandbox"]["image"]
    return (Path(img).exists(), f"apptainer, image {img}" if Path(img).exists() else f"apptainer image `{img}` not found")
