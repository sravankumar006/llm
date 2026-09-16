#!/usr/bin/env python3
"""
CommandLLM: Dataset Acquisition, Synthetic Generation, and Balancing Pipeline
==============================================================================
Constructs a balanced dual-OS dataset of natural language prompt -> terminal command pairs:
- Linux: Ingests NL2Bash from Hugging Face or raw mirrors, with resilient offline fallback.
- PowerShell: Generates high-variety synthetic pairs spanning 5 enterprise sysadmin domains:
    1. File & Directory Operations
    2. Process Management
    3. Network & Port Diagnostics
    4. Service & System Info
    5. Package & Environment Management
- Cleans, filters (<= 150 cmd chars, <= 50 prompt words), balances 50/50, and splits into train/val.
- Exports to data/train.jsonl and data/val.jsonl with schema:
    {"os": "linux" | "powershell", "prompt": "...", "cmd": "..."}
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset")


# ==============================================================================
# 1. Text Cleaning and Validation
# ==============================================================================

PROMPT_PREFIX_REGEX = re.compile(r"^(?:PS\s+[A-Za-z]:\\[^>]*>|PS\s*>|[A-Za-z]:\\[^>]*>|[$#>])\s*")
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable characters."""
    if not text:
        return ""
    # Strip terminal escape codes if present
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    # Strip control chars
    text = "".join(ch for ch in text if ch.isprintable() or ch in " \t\n")
    # Normalize whitespace
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def clean_command(cmd: str) -> str:
    """Clean terminal commands by removing interactive shell prompts ($ or >) and excess spacing."""
    cmd = clean_text(cmd)
    # Remove leading prompt symbol like '$', '>', 'PS >', 'C:\>'
    cmd = PROMPT_PREFIX_REGEX.sub("", cmd).strip()
    # Strip any enclosing quotes if the entire command is quoted
    if (cmd.startswith('"') and cmd.endswith('"')) or (cmd.startswith("'") and cmd.endswith("'")):
        if len(cmd) > 2:
            cmd = cmd[1:-1].strip()
    return cmd


def is_valid_sample(prompt: str, cmd: str, max_cmd_len: int = 150, max_prompt_words: int = 50) -> bool:
    """Validate sample against length, content, and quality criteria."""
    if not prompt or not cmd:
        return False
    if len(cmd) < 2 or len(cmd) > max_cmd_len:
        return False
    words = prompt.split()
    if len(words) < 2 or len(words) > max_prompt_words:
        return False
    # Filter out code comments or markdown fences
    if cmd.startswith("#") or cmd.startswith("```"):
        return False
    return True


# ==============================================================================
# 2. Linux (NL2Bash) Data Ingestion & Resilient Offline Fallback
# ==============================================================================

CURATED_LINUX_PAIRS: List[Tuple[str, str]] = [
    # File Search & find
    ("find all files with extension .log modified in the last 7 days", "find . -type f -name '*.log' -mtime -7"),
    ("find and delete all empty directories in the current folder", "find . -type d -empty -delete"),
    ("recursively search for files larger than 100 megabytes", "find / -type f -size +100M 2>/dev/null"),
    ("find all Python files containing the string TODO", "find . -name '*.py' -exec grep -H 'TODO' {} +"),
    ("find all files owned by user www-data in /var/www", "find /var/www -user www-data"),
    ("find all files with 777 permissions and change them to 644", "find . -type f -perm 0777 -exec chmod 644 {} +"),
    ("search for files modified in the last 24 hours", "find . -type f -mtime -1"),
    ("find all hidden files in current directory", "find . -maxdepth 1 -name '.*' -type f"),
    ("search for all .conf configuration files in /etc", "find /etc -type f -name '*.conf'"),
    ("find files matching *.tmp and remove them", "find . -name '*.tmp' -type f -delete"),
    ("find directories with permission 755", "find . -type d -perm 755"),
    ("find all files created in the last hour", "find . -type f -cmin -60"),

    # Grep & Text Processing
    ("search for error messages case-insensitively in syslog", "grep -i 'error' /var/log/syslog"),
    ("recursively search for pattern API_KEY in the current directory", "grep -rn 'API_KEY' ."),
    ("count the number of occurrences of 404 in access.log", "grep -c ' 404 ' access.log"),
    ("search for lines that do not contain the word debug", "grep -v 'debug' app.log"),
    ("display the first 20 lines of a file", "head -n 20 output.txt"),
    ("view the last 50 lines of a log file in real time", "tail -n 50 -f /var/log/nginx/access.log"),
    ("count total lines words and characters in file.txt", "wc file.txt"),
    ("sort lines in a file in reverse numerical order", "sort -rn data.txt"),
    ("remove duplicate lines from a sorted text file", "uniq sorted.txt unique.txt"),
    ("extract the first and third columns separated by commas", "cut -d',' -f1,3 data.csv"),
    ("replace all occurrences of localhost with 127.0.0.1 in config.env", "sed -i 's/localhost/127.0.0.1/g' config.env"),
    ("print the second column of space-separated text", "awk '{print $2}' input.txt"),
    ("sum numbers in the third column of a table", "awk '{sum += $3} END {print sum}' data.tsv"),
    ("print lines matching regex from file", "grep -E '^[0-9]{3}-[0-9]{2}' records.txt"),

    # Compression & Archives
    ("create a compressed tar archive of a directory", "tar -czvf archive.tar.gz /path/to/folder"),
    ("extract a tar.gz compressed archive to current directory", "tar -xzvf backup.tar.gz"),
    ("list the contents of a tar.gz archive without extracting", "tar -ztvf backup.tar.gz"),
    ("create a zip archive of multiple files", "zip -r files.zip folder/"),
    ("unzip a zip file to a specific destination directory", "unzip project.zip -d /opt/project"),
    ("extract a tar.bz2 archive", "tar -xjvf backup.tar.bz2"),
    ("compress a single file using gzip", "gzip -k access.log"),
    ("decompress a gzip file keeping the original", "gzip -dk access.log.gz"),

    # Permissions & Ownership
    ("change owner and group of a folder recursively to user deploy", "chown -R deploy:deploy /var/www/app"),
    ("grant read write and execute permissions to the owner only", "chmod 700 secret_script.sh"),
    ("make a shell script executable", "chmod +x deploy.sh"),
    ("recursively grant read permissions to all users for a directory", "chmod -R a+r /shared/docs"),
    ("change ownership of file to current user", "chown $USER:$USER file.txt"),
    ("set read-only permissions on a file", "chmod 444 readonly.txt"),

    # Process Management
    ("display all running processes with full command details", "ps aux"),
    ("find the process id of an application named nginx", "pgrep -l nginx"),
    ("forcefully terminate a process with pid 1234", "kill -9 1234"),
    ("kill all processes named node", "killall node"),
    ("kill processes matching name python", "pkill -f python"),
    ("display top processes sorted by memory usage", "ps aux --sort=-%mem | head -n 10"),
    ("display top processes sorted by cpu usage", "ps aux --sort=-%cpu | head -n 10"),
    ("show process hierarchy tree", "pstree -p"),

    # System & Hardware Info
    ("check available disk space in human readable format", "df -h"),
    ("display the size of each directory in the current path", "du -sh *"),
    ("check system memory and swap usage in megabytes", "free -m"),
    ("display kernel version and system architecture", "uname -a"),
    ("show cpu hardware details", "lscpu"),
    ("list all block devices and storage partitions", "lsblk"),
    ("show system uptime and load average", "uptime"),
    ("list all PCI devices connected to the system", "lspci"),
    ("list all USB devices connected", "lsusb"),
    ("display total disk usage of current directory", "du -sh ."),

    # Network & Connectivity
    ("check active listening tcp ports and associated processes", "ss -tulpn"),
    ("display network interfaces and ip addresses", "ip addr show"),
    ("display default network gateway routing table", "ip route show"),
    ("test reachability of google.com by sending 4 ping packets", "ping -c 4 google.com"),
    ("download a file from URL using curl saving with remote name", "curl -O https://example.com/data.tar.gz"),
    ("fetch HTTP response headers from a web server", "curl -I https://api.example.com/health"),
    ("download a file silently with wget", "wget -q https://example.com/file.zip"),
    ("lookup dns A record for domain example.com", "dig +short example.com"),
    ("trace network packets to host", "traceroute 8.8.8.8"),
    ("display open network sockets using netstat", "netstat -tuln"),

    # Services & Daemons (Systemd)
    ("check the status of the nginx service", "systemctl status nginx"),
    ("start the postgresql service", "systemctl start postgresql"),
    ("stop the apache2 web server service", "systemctl stop apache2"),
    ("restart the docker daemon", "systemctl restart docker"),
    ("enable ssh service to start automatically on boot", "systemctl enable ssh"),
    ("disable bluetooth service on startup", "systemctl disable bluetooth"),
    ("reload systemd daemon configuration", "systemctl daemon-reload"),
    ("view live logs for the docker service using journalctl", "journalctl -u docker -f"),
    ("view system errors from the current boot", "journalctl -p err -b"),

    # Users & Groups
    ("create a new user with home directory", "useradd -m -s /bin/bash developer"),
    ("delete a user and their home directory", "userdel -r baduser"),
    ("add a user to the sudo administrative group", "usermod -aG sudo developer"),
    ("show groups that the current user belongs to", "groups"),
    ("display current logged-in users", "who"),
    ("switch user to root with environment loaded", "su -"),

    # Package Management
    ("update apt package cache on Ubuntu/Debian", "sudo apt update"),
    ("upgrade all installed apt packages to their latest versions", "sudo apt upgrade -y"),
    ("install build-essential and git packages", "sudo apt install -y build-essential git"),
    ("remove an unused package and its configuration files", "sudo apt purge -y oldpkg"),
    ("clean up obsolete package archives from apt cache", "sudo apt autoremove -y"),

    # Git Basics
    ("check status of git working tree", "git status"),
    ("show recent git commit history one line per commit", "git log --oneline -n 10"),
    ("create and switch to a new git branch", "git checkout -b feature/new-pipeline"),
    ("stash uncommitted changes with a message", "git stash save 'work in progress'"),
    ("discard all unstaged changes in git working tree", "git restore ."),

    # Docker Basics
    ("list all active and stopped docker containers", "docker ps -a"),
    ("display all local docker images", "docker images"),
    ("remove all stopped docker containers", "docker container prune -f"),
    ("view logs from container web-server", "docker logs -f --tail 100 web-server"),
    ("run an interactive ubuntu container", "docker run -it --rm ubuntu:22.04 bash"),
]


def load_linux_from_huggingface() -> List[Dict[str, str]]:
    """Attempt downloading NL2Bash dataset from Hugging Face mirrors."""
    hf_candidates = [
        ("AnishJoshi/nl2bash-custom", "train"),
        ("akankshat/nl2bash", "train"),
        ("Maggie/nl2bash", "train"),
    ]

    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("Hugging Face 'datasets' library not available, skipping direct HF API.")
        return []

    for repo, split in hf_candidates:
        try:
            logger.info(f"Attempting to load Linux dataset from Hugging Face: '{repo}'...")
            ds = load_dataset(repo, split=split)
            pairs: List[Dict[str, str]] = []
            for item in ds:
                prompt = (
                    item.get("prompt")
                    or item.get("nl")
                    or item.get("query")
                    or item.get("nl_command")
                    or item.get("description")
                    or ""
                )
                cmd = (
                    item.get("bash_code")
                    or item.get("cmd")
                    or item.get("bash")
                    or item.get("command")
                    or item.get("bash_command")
                    or ""
                )
                if prompt and cmd:
                    p_clean = clean_text(prompt)
                    c_clean = clean_command(cmd)
                    if is_valid_sample(p_clean, c_clean):
                        pairs.append({"os": "linux", "prompt": p_clean, "cmd": c_clean})
            if len(pairs) > 100:
                logger.info(f"Successfully loaded {len(pairs)} valid Linux pairs from '{repo}'.")
                return pairs
        except Exception as e:
            logger.warning(f"Could not load '{repo}' from Hugging Face ({e}). Trying next candidate...")

    return []


def load_linux_from_github_mirrors() -> List[Dict[str, str]]:
    """Download raw NL2Bash JSON/JSONL datasets from GitHub user-content mirrors."""
    urls = [
        "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash-queries.json",
        "https://raw.githubusercontent.com/skp98/NL2Bash/master/data/bash-queries.json",
    ]

    for url in urls:
        try:
            logger.info(f"Attempting download from raw GitHub mirror: {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "CommandLLM-DatasetBuilder/1.0"})
            with urllib.request.urlopen(req, timeout=12) as response:
                content = response.read().decode("utf-8")

            data = json.loads(content)
            pairs: List[Dict[str, str]] = []
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        prompt = v.get("query") or v.get("nl") or ""
                        cmd = v.get("cmd") or v.get("bash") or k
                    else:
                        prompt = str(k)
                        cmd = str(v)
                    p_clean = clean_text(prompt)
                    c_clean = clean_command(cmd)
                    if is_valid_sample(p_clean, c_clean):
                        pairs.append({"os": "linux", "prompt": p_clean, "cmd": c_clean})
            elif isinstance(data, list):
                for item in data:
                    prompt = item.get("query") or item.get("prompt") or item.get("nl") or ""
                    cmd = item.get("cmd") or item.get("command") or item.get("bash") or ""
                    p_clean = clean_text(prompt)
                    c_clean = clean_command(cmd)
                    if is_valid_sample(p_clean, c_clean):
                        pairs.append({"os": "linux", "prompt": p_clean, "cmd": c_clean})

            if len(pairs) > 50:
                logger.info(f"Loaded {len(pairs)} pairs from GitHub mirror {url}.")
                return pairs
        except Exception as e:
            logger.warning(f"Failed to load from {url}: {e}")

    return []


def generate_curated_linux_samples(multiplier: int = 1) -> List[Dict[str, str]]:
    """Return cleaned curated Linux pairs with synthetic variation to ensure rich coverage."""
    prefix_synonyms = [
        "",
        "how to ",
        "command to ",
        "please ",
        "can you ",
        "i need to ",
        "how do i ",
        "bash command for ",
        "terminal command to ",
        "script to ",
    ]

    samples: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for prompt, cmd in CURATED_LINUX_PAIRS:
        p_clean = clean_text(prompt)
        c_clean = clean_command(cmd)
        key = f"{p_clean}|{c_clean}"
        if key not in seen and is_valid_sample(p_clean, c_clean):
            seen.add(key)
            samples.append({"os": "linux", "prompt": p_clean, "cmd": c_clean})

    if multiplier > 1:
        base_samples = list(samples)
        for pre in prefix_synonyms[1:]:
            for item in base_samples:
                p_var = clean_text(pre + item["prompt"])
                key = f"{p_var}|{item['cmd']}"
                if key not in seen and is_valid_sample(p_var, item["cmd"]):
                    seen.add(key)
                    samples.append({"os": "linux", "prompt": p_var, "cmd": item["cmd"]})
                if len(samples) >= len(CURATED_LINUX_PAIRS) * multiplier:
                    break

    return samples


def get_linux_dataset(target_count: int = 2500) -> List[Dict[str, str]]:
    """Acquire Linux command dataset with automatic fallback."""
    logger.info("Ingesting Linux dataset...")
    pairs = load_linux_from_huggingface()

    if not pairs:
        pairs = load_linux_from_github_mirrors()

    if not pairs:
        logger.info("Internet/mirrors unreachable or rate-limited. Using built-in curated catalog...")
        pairs = generate_curated_linux_samples(multiplier=35)
    else:
        # Augment with curated pairs to guarantee coverage of foundational utilities
        curated = generate_curated_linux_samples(multiplier=1)
        existing_keys = {f"{x['prompt']}|{x['cmd']}" for x in pairs}
        for item in curated:
            k = f"{item['prompt']}|{item['cmd']}"
            if k not in existing_keys:
                pairs.append(item)
                existing_keys.add(k)

    logger.info(f"Total candidate Linux pairs collected: {len(pairs)}")
    return pairs


# ==============================================================================
# 3. Windows PowerShell Synthetic Generator (5 Admin Domains)
# ==============================================================================

def generate_powershell_synthetic_dataset(target_count: int = 3000) -> List[Dict[str, str]]:
    """
    Generate diverse, realistic synthetic PowerShell commands and natural language prompts
    spanning five enterprise sysadmin pillars:
      1. File & Directory Operations
      2. Process Management
      3. Network & Port Diagnostics
      4. Service & System Info
      5. Package & Environment
    """
    logger.info("Generating synthetic Windows PowerShell dataset...")
    results: List[Dict[str, str]] = []
    seen: Set[str] = set()

    exts = [".log", ".txt", ".json", ".csv", ".ps1", ".xml", ".zip", ".bak", ".config"]
    paths = [
        r"C:\Logs",
        r"C:\Backup",
        r"C:\Projects",
        r"C:\inetpub\wwwroot",
        r"D:\Data",
        r".\logs",
        r".\data",
        r"C:\Temp",
        r"C:\ProgramData\App",
    ]
    process_names = [
        "notepad", "chrome", "firefox", "powershell", "code", "msedge",
        "nginx", "python", "node", "dotnet", "teams", "slack", "java"
    ]
    service_names = [
        "wuauserv", "spooler", "w32time", "bits", "LanmanServer", "WinDefend",
        "sshd", "docker", "MSSQLSERVER", "MySQL", "nginx", "Themes"
    ]
    hosts = [
        "google.com", "github.com", "microsoft.com", "api.github.com",
        "192.168.1.1", "10.0.0.1", "127.0.0.1", "smtp.office365.com", "dns.google"
    ]
    ports = [80, 443, 22, 3389, 8080, 5432, 3306, 1433, 21, 53]
    pkgs = ["Git.Git", "Microsoft.VisualStudioCode", "Python.Python.3.12", "Docker.DockerDesktop", "7zip.7zip", "Mozilla.Firefox", "Node.js"]
    choco_pkgs = ["git", "vscode", "python3", "docker-cli", "7zip", "curl", "neovim", "terraform"]
    patterns = ["ERROR", "FATAL", "Exception", "Timeout", "Unauthorized", "404", "Warning", "ConnectionReset"]

    # Domain 1: File & Directory Operations
    file_templates = [
        (
            ["list all files in {path} recursively", "find all files inside {path} and subdirectories", "show all files in {path} including subfolders"],
            "Get-ChildItem -Path '{path}' -Recurse"
        ),
        (
            ["find all {ext} files in {path}", "search for {ext} files in {path}", "list {ext} files inside {path}"],
            "Get-ChildItem -Path '{path}' -Filter '*{ext}'"
        ),
        (
            ["recursively search for {ext} files in {path}", "find all {ext} files in {path} including subdirectories"],
            "Get-ChildItem -Path '{path}' -Filter '*{ext}' -Recurse"
        ),
        (
            ["list only directories in {path}", "show subfolders in {path}", "get folder list in {path}"],
            "Get-ChildItem -Path '{path}' -Directory"
        ),
        (
            ["list only files in {path} excluding folders", "get file listing for {path}"],
            "Get-ChildItem -Path '{path}' -File"
        ),
        (
            ["list hidden and system files in {path}", "show all files including hidden in {path}"],
            "Get-ChildItem -Path '{path}' -Force -Hidden"
        ),
        (
            ["get the 10 largest files in {path}", "find top 10 biggest files in {path}"],
            "Get-ChildItem -Path '{path}' -File -Recurse | Sort-Object Length -Descending | Select-Object -First 10"
        ),
        (
            ["create a new directory named {folder} in {path}", "make directory {path}\\{folder}", "create folder {folder} at {path}"],
            "New-Item -Path '{path}\\{folder}' -ItemType Directory -Force"
        ),
        (
            ["create an empty file named {name}{ext} in {path}", "create new file {path}\\{name}{ext}", "touch file {name}{ext} at {path}"],
            "New-Item -Path '{path}\\{name}{ext}' -ItemType File -Force"
        ),
        (
            ["copy file {path}\\source{ext} to {path}\\dest{ext}", "make a copy of {path}\\source{ext} at {path}\\dest{ext}"],
            "Copy-Item -Path '{path}\\source{ext}' -Destination '{path}\\dest{ext}' -Force"
        ),
        (
            ["copy directory {path} to C:\\Backup\\app recursively", "backup folder {path} to C:\\Backup\\app"],
            "Copy-Item -Path '{path}' -Destination 'C:\\Backup\\app' -Recurse -Force"
        ),
        (
            ["remove all {ext} files from {path}", "delete all {ext} files in {path}", "purge {ext} files under {path}"],
            "Remove-Item -Path '{path}\\*{ext}' -Force"
        ),
        (
            ["delete folder {path} and all its contents", "forcefully remove directory {path} recursively", "delete directory {path}"],
            "Remove-Item -Path '{path}' -Recurse -Force"
        ),
        (
            ["search for pattern '{pattern}' in all {ext} files in {path}", "find text '{pattern}' in {path} *{ext}"],
            "Get-ChildItem -Path '{path}\\*{ext}' | Select-String -Pattern '{pattern}'"
        ),
        (
            ["grep for '{pattern}' inside file {path}\\app.log", "search for '{pattern}' in {path}\\app.log"],
            "Select-String -Path '{path}\\app.log' -Pattern '{pattern}' -CaseSensitive"
        ),
        (
            ["view the first 25 lines of {path}\\app{ext}", "show top 25 lines of {path}\\app{ext}"],
            "Get-Content -Path '{path}\\app{ext}' -TotalCount 25"
        ),
        (
            ["read the last 50 lines of {path}\\app{ext}", "tail 50 lines from {path}\\app{ext}"],
            "Get-Content -Path '{path}\\app{ext}' -Tail 50"
        ),
        (
            ["monitor {path}\\app{ext} in real time", "tail -f {path}\\app{ext} in powershell"],
            "Get-Content -Path '{path}\\app{ext}' -Wait -Tail 20"
        ),
        (
            ["check if file {path}\\config.json exists", "test existence of {path}\\config.json"],
            "Test-Path -Path '{path}\\config.json'"
        ),
        (
            ["compress folder {path} into a zip archive", "zip folder {path} to C:\\Backup\\archive.zip"],
            "Compress-Archive -Path '{path}\\*' -DestinationPath 'C:\\Backup\\archive.zip' -Force"
        ),
        (
            ["unzip archive C:\\Backup\\archive.zip to {path}", "extract zip file to {path}"],
            "Expand-Archive -Path 'C:\\Backup\\archive.zip' -DestinationPath '{path}' -Force"
        ),
    ]

    # Domain 2: Process Management
    process_templates = [
        (
            ["list all running processes", "show all active processes", "display system process list"],
            "Get-Process"
        ),
        (
            ["check if {proc} is running", "get details for process {proc}", "find process named {proc}"],
            "Get-Process -Name '{proc}' -ErrorAction SilentlyContinue"
        ),
        (
            ["find top 5 processes using the most cpu", "show highest cpu consuming processes", "top cpu processes"],
            "Get-Process | Sort-Object CPU -Descending | Select-Object -First 5"
        ),
        (
            ["find top 5 processes using the most memory", "show processes consuming the most RAM", "top memory processes"],
            "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 5"
        ),
        (
            ["kill process named {proc}", "stop {proc} process", "terminate {proc}"],
            "Stop-Process -Name '{proc}' -Force"
        ),
        (
            ["terminate process with process id {pid}", "kill pid {pid}", "stop process by id {pid}"],
            "Stop-Process -Id {pid} -Force"
        ),
        (
            ["start a new instance of {proc}", "launch application {proc}", "run {proc}"],
            "Start-Process -FilePath '{proc}'"
        ),
        (
            ["start powershell as administrator", "launch elevated powershell prompt", "run powershell with admin privileges"],
            "Start-Process powershell -Verb RunAs"
        ),
        (
            ["wait for process {proc} to exit", "block until {proc} terminates"],
            "Wait-Process -Name '{proc}'"
        ),
    ]

    # Domain 3: Network & Port Diagnostics
    network_templates = [
        (
            ["test connection to {host} on port {port}", "check if port {port} is open on {host}", "verify connectivity to {host}:{port}"],
            "Test-NetConnection -ComputerName '{host}' -Port {port}"
        ),
        (
            ["ping host {host}", "test ping connectivity to {host}"],
            "Test-Connection -TargetName '{host}' -Count 4"
        ),
        (
            ["traceroute to host {host}", "trace network path to {host}"],
            "Test-NetConnection -ComputerName '{host}' -TraceRoute"
        ),
        (
            ["list all listening tcp ports", "show open listening ports", "check listening network sockets"],
            "Get-NetTCPConnection -State Listen | Select-Object LocalAddress, LocalPort, OwningProcess"
        ),
        (
            ["find process listening on port {port}", "check what is using port {port}", "identify process on port {port}"],
            "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue"
        ),
        (
            ["resolve dns records for {host}", "lookup dns for domain {host}", "query dns for {host}"],
            "Resolve-DnsName -Name '{host}'"
        ),
        (
            ["lookup MX mail records for domain {host}", "query MX records for {host}"],
            "Resolve-DnsName -Name '{host}' -Type MX"
        ),
        (
            ["download file from https://{host}/setup.exe", "fetch file from https://{host}/setup.exe and save to C:\\Temp\\setup.exe"],
            "Invoke-WebRequest -Uri 'https://{host}/setup.exe' -OutFile 'C:\\Temp\\setup.exe'"
        ),
        (
            ["send HTTP GET request to https://{host}/api/v1/health", "call REST api endpoint https://{host}/api/v1/health"],
            "Invoke-RestMethod -Uri 'https://{host}/api/v1/health' -Method Get"
        ),
        (
            ["get current IP addresses of this machine", "show active network adapters and ip addresses"],
            "Get-NetIPAddress -AddressFamily IPv4 | Select-Object IPAddress, InterfaceAlias"
        ),
    ]

    # Domain 4: Service & System Info
    service_templates = [
        (
            ["list all installed windows services", "show all services", "get list of windows services"],
            "Get-Service"
        ),
        (
            ["show only currently running services", "list services that are active"],
            "Get-Service | Where-Object Status -eq 'Running'"
        ),
        (
            ["show stopped services", "list services that are currently stopped"],
            "Get-Service | Where-Object Status -eq 'Stopped'"
        ),
        (
            ["check status of service {svc}", "is {svc} service running", "query service {svc}"],
            "Get-Service -Name '{svc}'"
        ),
        (
            ["start service {svc}", "start the {svc} service", "turn on service {svc}"],
            "Start-Service -Name '{svc}'"
        ),
        (
            ["stop service {svc}", "shut down {svc} service", "halt service {svc}"],
            "Stop-Service -Name '{svc}' -Force"
        ),
        (
            ["restart service {svc}", "bounce service {svc}", "reboot service {svc}"],
            "Restart-Service -Name '{svc}' -Force"
        ),
        (
            ["display general system and hardware information", "get windows computer information", "show system specs"],
            "Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, TotalPhysicalMemory"
        ),
        (
            ["check free disk space on all logical drives", "show disk space for all drives", "get storage information"],
            "Get-CimInstance -ClassName Win32_LogicalDisk | Select-Object DeviceID, FreeSpace, Size"
        ),
        (
            ["get operating system version and last boot time", "check system uptime and boot time"],
            "Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object LastBootUpTime, Version, OSArchitecture"
        ),
        (
            ["get cpu model and core count", "show processor information"],
            "Get-CimInstance -ClassName Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors"
        ),
    ]

    # Domain 5: Package & Environment Management
    env_templates = [
        (
            ["install {pkg} using winget", "use winget to install {pkg}", "install application {pkg} via winget silently"],
            "winget install --id '{pkg}' --silent --accept-source-agreements --accept-package-agreements"
        ),
        (
            ["search for package {pkg} using winget", "find winget package {pkg}"],
            "winget search '{pkg}'"
        ),
        (
            ["uninstall {pkg} using winget", "remove package {pkg} via winget"],
            "winget uninstall --id '{pkg}'"
        ),
        (
            ["install {choco_pkg} using chocolatey", "use choco to install {choco_pkg}", "choco install {choco_pkg} quietly"],
            "choco install '{choco_pkg}' -y"
        ),
        (
            ["upgrade all installed chocolatey packages", "update all software via choco"],
            "choco upgrade all -y"
        ),
        (
            ["list all installed chocolatey packages", "show local choco packages"],
            "choco list --local-only"
        ),
        (
            ["display all environment variables", "list all system environment variables"],
            "Get-ChildItem Env:"
        ),
        (
            ["get value of PATH environment variable", "show system PATH variable"],
            "$env:PATH"
        ),
        (
            ["temporarily add C:\\tools to current session PATH", "append C:\\tools to PATH environment variable"],
            "$env:PATH += ';C:\\tools'"
        ),
        (
            ["permanently add C:\\tools to machine system PATH", "add directory C:\\tools to system PATH permanently"],
            "[Environment]::SetEnvironmentVariable('Path', $env:PATH + ';C:\\tools', 'Machine')"
        ),
        (
            ["get current user profile directory path", "show USERPROFILE environment variable"],
            "$env:USERPROFILE"
        ),
        (
            ["get system temp directory path", "show TEMP directory path"],
            "$env:TEMP"
        ),
    ]

    all_domain_templates = [
        file_templates,
        process_templates,
        network_templates,
        service_templates,
        env_templates,
    ]

    random.seed(42)
    iterations = 0
    max_iterations = target_count * 10

    while len(results) < target_count and iterations < max_iterations:
        iterations += 1
        domain = random.choice(all_domain_templates)
        prompt_list, cmd_tmpl = random.choice(domain)

        ctx = {
            "path": random.choice(paths),
            "ext": random.choice(exts),
            "folder": random.choice(["output", "archive", "temp", "build", "dist", "workspace"]),
            "name": random.choice(["test", "config", "data", "report", "sample", "output"]),
            "pattern": random.choice(patterns),
            "proc": random.choice(process_names),
            "pid": random.choice([1042, 2834, 4920, 8192, 9304, 1248]),
            "svc": random.choice(service_names),
            "host": random.choice(hosts),
            "port": random.choice(ports),
            "pkg": random.choice(pkgs),
            "choco_pkg": random.choice(choco_pkgs),
        }

        raw_prompt = random.choice(prompt_list)
        try:
            rendered_prompt = raw_prompt.format(**ctx)
            rendered_cmd = cmd_tmpl.format(**ctx)
        except KeyError:
            continue

        p_clean = clean_text(rendered_prompt)
        c_clean = clean_command(rendered_cmd)

        key = f"{p_clean}|{c_clean}"
        if key not in seen and is_valid_sample(p_clean, c_clean):
            seen.add(key)
            results.append({"os": "powershell", "prompt": p_clean, "cmd": c_clean})

    logger.info(f"Generated {len(results)} distinct PowerShell pairs.")
    return results


# ==============================================================================
# 4. Balancing, Splitting, and Merging Pipeline
# ==============================================================================

def build_dataset(
    output_dir: str = "data",
    samples_per_os: int = 2500,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Execute end-to-end dataset pipeline:
    1. Collect Linux Bash subset
    2. Generate Windows PowerShell subset
    3. Balance 50/50
    4. Shuffle and split into 90% train / 10% validation
    5. Write to data/train.jsonl and data/val.jsonl
    """
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"

    linux_pool = get_linux_dataset(target_count=samples_per_os)
    powershell_pool = generate_powershell_synthetic_dataset(target_count=samples_per_os)

    n_samples = min(len(linux_pool), len(powershell_pool))
    if samples_per_os and samples_per_os < n_samples:
        n_samples = samples_per_os

    logger.info(f"Balancing dataset: selecting {n_samples} samples per OS (Total: {n_samples * 2})...")
    random.shuffle(linux_pool)
    random.shuffle(powershell_pool)

    linux_selected = linux_pool[:n_samples]
    powershell_selected = powershell_pool[:n_samples]

    combined = linux_selected + powershell_selected
    random.shuffle(combined)

    val_size = max(1, int(len(combined) * val_ratio))
    train_samples = combined[val_size:]
    val_samples = combined[:val_size]

    logger.info(f"Writing {len(train_samples)} samples to {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in tqdm(train_samples, desc="Writing train set"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Writing {len(val_samples)} samples to {val_file}...")
    with open(val_file, "w", encoding="utf-8") as f:
        for item in tqdm(val_samples, desc="Writing val set"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_linux_count = sum(1 for x in train_samples if x["os"] == "linux")
    train_pwsh_count = sum(1 for x in train_samples if x["os"] == "powershell")
    val_linux_count = sum(1 for x in val_samples if x["os"] == "linux")
    val_pwsh_count = sum(1 for x in val_samples if x["os"] == "powershell")

    logger.info("=" * 60)
    logger.info(" DATASET INGESTION & BALANCING COMPLETE")
    logger.info("=" * 60)
    logger.info(f" Total Train Records : {len(train_samples):,} (Linux: {train_linux_count:,}, PowerShell: {train_pwsh_count:,})")
    logger.info(f" Total Val Records   : {len(val_samples):,} (Linux: {val_linux_count:,}, PowerShell: {val_pwsh_count:,})")
    logger.info(f" Output Train Path   : {train_file.resolve()}")
    logger.info(f" Output Val Path     : {val_file.resolve()}")
    logger.info("=" * 60)

    return str(train_file), str(val_file)


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CommandLLM Dataset Pipeline: Ingest, Generate, Balance, and Split."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Destination directory for train.jsonl and val.jsonl (default: 'data').",
    )
    parser.add_argument(
        "--samples-per-os",
        type=int,
        default=2500,
        help="Target number of balanced samples per OS (default: 2500).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio (default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible balancing and shuffling (default: 42).",
    )

    args = parser.parse_args()
    build_dataset(
        output_dir=args.output_dir,
        samples_per_os=args.samples_per_os,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
