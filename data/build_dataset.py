#!/usr/bin/env python3
"""
CommandLLM: Scaled Dataset Acquisition & Synthetic Generation Pipeline
======================================================================
Constructs a balanced, diverse dual-OS dataset of natural language -> command pairs:
- Linux: Multi-constraint generators across files, grep, pipes, processes, networking,
         system diagnostics, permissions, archives, git, and docker.
- PowerShell: Enterprise sysadmin generators across files, registry, processes, networking,
              services, CimInstances, archives, git, docker, and diagnostics.
- Template-Level Holdouts: Dedicated template families reserved strictly for validation
  to prevent template-level memorization and data leakage.
- Generates 10,000 samples per OS (~20,000 total: ~18,000 train, ~2,000 val).
"""

import argparse
import json
import logging
import os
import random
import re
import sys
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

PROMPT_PREFIX_REGEX = re.compile(r"^(?:PS\s+[A-Za-z]:\\[^>]*>|PS\s*>|[A-Za-z]:\\[^>]*>|[$#>])\s*")
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable characters."""
    if not text:
        return ""
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in " \t\n")
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def clean_command(cmd: str) -> str:
    """Clean terminal commands by removing prompt prefixes ($ or >) and excess spacing."""
    cmd = clean_text(cmd)
    cmd = PROMPT_PREFIX_REGEX.sub("", cmd).strip()
    if (cmd.startswith('"') and cmd.endswith('"')) or (cmd.startswith("'") and cmd.endswith("'")):
        if len(cmd) > 2:
            cmd = cmd[1:-1].strip()
    return cmd


def is_valid_sample(prompt: str, cmd: str, max_cmd_len: int = 180, max_prompt_words: int = 50) -> bool:
    """Validate sample against length, content, and quality criteria."""
    if not prompt or not cmd:
        return False
    if len(cmd) < 2 or len(cmd) > max_cmd_len:
        return False
    words = prompt.split()
    if len(words) < 2 or len(words) > max_prompt_words:
        return False
    if cmd.startswith("#") or cmd.startswith("```"):
        return False
    return True


# ==============================================================================
# Shared Domain Vocabulary and Substitution Pools
# ==============================================================================

LINUX_PATHS = ["/var/log", "/etc", "/opt", "/tmp", "/home/user", "/var/www/html", "/usr/local/bin", ".", "..", "/data"]
PWSH_PATHS = [r"C:\Logs", r"C:\Backup", r"C:\Projects", r"C:\inetpub\wwwroot", r"D:\Data", r".\logs", r".\data", r"C:\Temp", r"C:\ProgramData\App"]
EXTENSIONS = [".log", ".txt", ".json", ".csv", ".conf", ".xml", ".zip", ".tar.gz", ".bak", ".py", ".sh", ".yml", ".md"]
PATTERNS = ["ERROR", "FATAL", "Exception", "Timeout", "Unauthorized", "404", "Warning", "ConnectionReset", "NullPointer", "CRITICAL", "OutOfMemory"]
PORTS = [80, 443, 22, 3389, 8080, 5432, 3306, 1433, 9000, 27017, 6379, 8443, 5000, 3000]
HOSTS = ["google.com", "github.com", "microsoft.com", "api.github.com", "192.168.1.1", "10.0.0.1", "127.0.0.1", "smtp.office365.com", "dns.google", "cloudflare.com"]
PROCESSES = ["nginx", "python", "node", "docker", "postgres", "redis", "chrome", "firefox", "sshd", "apache2", "code", "teams", "java", "dotnet"]
SERVICES = ["nginx", "postgresql", "docker", "ssh", "apache2", "redis-server", "cron", "wuauserv", "spooler", "w32time", "bits", "MSSQLSERVER"]
SIZES = [10, 50, 100, 200, 500, 1024, 2048]
DAYS = [1, 2, 7, 14, 30, 90]
COUNTS = [5, 10, 20, 25, 50, 100]
BRANCHES = ["main", "master", "develop", "feature/login", "bugfix/timeout", "release/v1.0", "staging"]
USERS = ["admin", "root", "deploy", "nginx", "appuser", "ubuntu"]
GROUPS = ["www-data", "docker", "admin", "developers", "users"]
VOLUMES = ["/data", "/var/lib/mysql", "/app/data", "/shared"]


# ==============================================================================
# Linux Template Specifications (with designated holdouts for zero leakage)
# ==============================================================================

LINUX_TEMPLATES = [
    # 1. Process kill by port
    (
        [
            "kill process listening on port {port}",
            "terminate the program running on port {port}",
            "force kill process on port {port}",
            "free up port {port} by terminating the process",
            "stop the process using port {port}",
            "how to kill whatever is listening on port {port}",
            "kill whatever is bound to port {port}",
            "free port {port} and stop its process",
        ],
        "kill -9 $(lsof -t -i:{port})",
        False
    ),
    # 2. Kill by name
    (
        [
            "kill all processes named {proc}",
            "terminate all instances of {proc}",
            "forcefully stop all {proc} processes",
            "kill processes matching {proc}",
            "pkill all instances of {proc}",
            "stop every process with name {proc}",
        ],
        "pkill -9 -f {proc}",
        False
    ),
    # 3. Kill by PID
    (
        [
            "kill process with pid {pid}",
            "force terminate process id {pid}",
            "send sigkill to process {pid}",
            "stop process {pid}",
            "kill -9 on pid {pid}",
            "force kill pid {pid}",
        ],
        "kill -9 {pid}",
        False
    ),
    # 4. Find files larger than size
    (
        [
            "find all files larger than {size}MB in {path}",
            "search for files bigger than {size}MB under {path}",
            "list files exceeding {size} megabytes in {path}",
            "find files with size greater than {size}M in {path}",
            "locate files over {size}MB inside {path}",
            "find large files over {size}MB in {path}",
        ],
        "find {path} -type f -size +{size}M",
        False
    ),
    # 5. Find by extension and delete
    (
        [
            "find and delete all {ext} files in {path}",
            "recursively remove all {ext} files under {path}",
            "delete all {ext} files located in {path}",
            "purge all {ext} files in {path}",
            "find all {ext} files in {path} and delete them",
            "cleanup all files with extension {ext} in {path}",
        ],
        "find {path} -type f -name '*{ext}' -delete",
        False
    ),
    # 6. Find modified in last N days
    (
        [
            "find files modified in the last {days} days in {path}",
            "list files changed in the past {days} days under {path}",
            "search for files modified within {days} days in {path}",
            "show all files updated in last {days} days under {path}",
            "find recently modified files in last {days} days in {path}",
        ],
        "find {path} -type f -mtime -{days}",
        False
    ),
    # 7. Grep recursive in directory
    (
        [
            "recursively search for '{pattern}' in {path}",
            "grep for string '{pattern}' in all files under {path}",
            "find text '{pattern}' inside directory {path}",
            "search for pattern '{pattern}' in {path} with line numbers",
            "recursively grep '{pattern}' under {path}",
            "find occurrences of '{pattern}' in {path}",
        ],
        "grep -rn '{pattern}' {path}",
        False
    ),
    # 8. Grep count occurrences
    (
        [
            "count occurrences of '{pattern}' in {path}/app.log",
            "count matching lines for '{pattern}' in {path}/app.log",
            "number of times '{pattern}' appears in {path}/app.log",
            "get match count for '{pattern}' in {path}/app.log",
            "how many times does '{pattern}' occur in {path}/app.log",
        ],
        "grep -c '{pattern}' {path}/app.log",
        False
    ),
    # 9. Grep invert match
    (
        [
            "filter out lines containing '{pattern}' in {path}/app.log",
            "search for lines that do not match '{pattern}' in {path}/app.log",
            "exclude '{pattern}' from {path}/app.log output",
            "print lines in {path}/app.log without '{pattern}'",
            "invert match for '{pattern}' in {path}/app.log",
        ],
        "grep -v '{pattern}' {path}/app.log",
        False
    ),
    # 10. Head command
    (
        [
            "show the first {count} lines of {path}/app{ext}",
            "display top {count} lines of {path}/app{ext}",
            "view beginning {count} lines of {path}/app{ext}",
            "head first {count} lines from {path}/app{ext}",
            "print the first {count} lines of {path}/app{ext}",
        ],
        "head -n {count} {path}/app{ext}",
        False
    ),
    # 11. Tail follow command
    (
        [
            "view the last {count} lines of {path}/app{ext} in real time",
            "tail -f {count} lines of {path}/app{ext}",
            "follow live updates for {path}/app{ext} with {count} lines",
            "stream last {count} lines of {path}/app{ext}",
            "monitor the end of {path}/app{ext} with {count} lines",
        ],
        "tail -n {count} -f {path}/app{ext}",
        False
    ),
    # 12. Disk space
    (
        [
            "check available disk space in human readable format",
            "show free disk space across mounted filesystems",
            "display disk usage summary",
            "view filesystem disk usage in human units",
            "how much disk space is left",
        ],
        "df -h",
        False
    ),
    # 13. Directory size
    (
        [
            "check total disk usage of {path}",
            "calculate size of folder {path}",
            "show disk footprint for {path}",
            "find overall size of directory {path}",
            "du disk summary for {path}",
        ],
        "du -sh {path}",
        False
    ),
    # 14. Memory usage
    (
        [
            "check system memory usage in megabytes",
            "show ram and swap usage in MB",
            "display free memory",
            "check free and used ram in mb",
            "view system memory in megabytes",
        ],
        "free -m",
        False
    ),
    # 15. Listening ports (ss)
    (
        [
            "check all active listening tcp and udp ports",
            "list listening sockets with process ids",
            "show open network ports",
            "display listening ports and associated processes",
            "list all active network listening sockets",
        ],
        "ss -tulpn",
        False
    ),
    # 16. Curl header inspection
    (
        [
            "fetch http response headers from {host}",
            "check status code and headers for {host}",
            "inspect web headers of {host}",
            "curl headers only from https://{host}",
            "get response headers for {host}",
        ],
        "curl -I https://{host}",
        False
    ),
    # 17. Ping test
    (
        [
            "test network connectivity to {host} with {count} packets",
            "ping host {host} {count} times",
            "check reachability of {host}",
            "send {count} ping requests to {host}",
            "verify connection to {host} with {count} pings",
        ],
        "ping -c {count} {host}",
        False
    ),
    # 18. Service management (restart)
    (
        [
            "restart the {service} service",
            "reboot daemon {service}",
            "reload and restart {service}",
            "systemctl restart {service}",
            "restart service {service} using systemctl",
        ],
        "sudo systemctl restart {service}",
        False
    ),
    # 19. Service status
    (
        [
            "check status of {service} service",
            "is {service} running",
            "view service health for {service}",
            "get systemctl status for {service}",
            "inspect {service} daemon status",
        ],
        "systemctl status {service}",
        False
    ),
    # 20. Chmod permissions
    (
        [
            "make script {path}/run.sh executable",
            "grant execute permission to {path}/run.sh",
            "chmod +x on {path}/run.sh",
            "allow execution of script {path}/run.sh",
            "set executable bit on {path}/run.sh",
        ],
        "chmod +x {path}/run.sh",
        False
    ),
    # 21. Tar archive creation
    (
        [
            "create a compressed tar.gz archive of {path}",
            "compress directory {path} into archive.tar.gz",
            "pack folder {path} to archive.tar.gz",
            "create tar gz of {path}",
            "archive {path} with gzip compression",
        ],
        "tar -czvf archive.tar.gz {path}",
        False
    ),
    # 22. Git status & log
    (
        [
            "check git status",
            "view git repository working tree status",
            "show modified files in git",
            "display git repository current status",
            "inspect git working tree",
        ],
        "git status",
        False
    ),
    (
        [
            "show git commit history one line per commit for last {count} commits",
            "git log compact format top {count} commits",
            "view last {count} git commits",
            "display latest {count} git commits on one line",
            "show last {count} commits in oneline format",
        ],
        "git log --oneline -n {count}",
        False
    ),
    # 23. Git branch creation & switch
    (
        [
            "create and switch to new git branch {branch}",
            "create a new branch called {branch} and checkout to it",
            "checkout new git branch named {branch}",
            "git switch to new branch {branch}",
            "make new branch {branch} and switch to it",
        ],
        "git checkout -b {branch}",
        False
    ),
    # 24. Git diff & stash
    (
        [
            "show git diff for the previous commit",
            "compare changes with last commit in git",
            "view git diff against HEAD~1",
            "show changes made in the last commit",
            "diff working tree against HEAD~1",
        ],
        "git diff HEAD~1",
        False
    ),
    (
        [
            "stash current uncommitted git changes",
            "save working directory changes to git stash",
            "temporarily stash modifications in git",
            "git stash uncommitted work",
            "stash unstaged changes",
        ],
        "git stash",
        False
    ),
    # 25. Docker list & logs
    (
        [
            "list all running and stopped docker containers",
            "show all docker containers including exited",
            "docker container list all",
            "view all docker containers with ps -a",
            "list every container on docker",
        ],
        "docker ps -a",
        False
    ),
    (
        [
            "view live logs for container {proc}",
            "stream logs from docker container {proc}",
            "follow docker logs of {proc}",
            "tail and follow logs of container {proc}",
            "monitor docker logs for {proc}",
        ],
        "docker logs -f --tail {count} {proc}",
        False
    ),
    # 26. Docker run container with port mapping
    (
        [
            "run docker container for {proc} with port {port} mapped in background",
            "start container {proc} detached mapping port {port}",
            "run {proc} in background with port {port} exposed",
            "launch docker container {proc} on port {port} detached",
            "docker run detached {proc} binding port {port}",
        ],
        "docker run -d -p {port}:{port} --name {proc} {proc}:latest",
        False
    ),
    # 27. Docker exec interactive shell
    (
        [
            "open an interactive shell inside docker container {proc}",
            "exec into running container {proc} with sh",
            "attach interactive sh shell to container {proc}",
            "enter docker container {proc} terminal",
            "run interactive sh inside {proc} container",
        ],
        "docker exec -it {proc} sh",
        False
    ),
    # 28. Piped: Grep count and sort uniq
    (
        [
            "count and rank unique occurrences of '{pattern}' in {path}/app.log",
            "find unique counts of pattern '{pattern}' in {path}/app.log sorted",
            "grep pattern '{pattern}' in {path}/app.log and count occurrences with uniq",
            "show sorted frequency of '{pattern}' in {path}/app.log",
        ],
        "grep -o '{pattern}' {path}/app.log | sort | uniq -c",
        False
    ),
    # 29. Piped: Find and exec chmod
    (
        [
            "find all {ext} files in {path} and set permissions to 644",
            "recursively chmod 644 all {ext} files under {path}",
            "change permissions of {ext} files in {path} to 644 using find exec",
            "find and change file mode to 644 for all {ext} files in {path}",
        ],
        "find {path} -type f -name '*{ext}' -exec chmod 644 {{}} +",
        False
    ),
    # 30. Ownership: Chown recursive
    (
        [
            "change owner and group of {path} to {user}:{group} recursively",
            "recursively set ownership of {path} to {user}:{group}",
            "chown -R {user}:{group} on {path}",
            "make {user} and group {group} owners of directory {path}",
        ],
        "chown -R {user}:{group} {path}",
        False
    ),
    # 31. Diagnostics: Traceroute
    (
        [
            "trace network route to {host}",
            "run traceroute to remote host {host}",
            "trace network hops to {host}",
            "find network path and latency hops to {host}",
        ],
        "traceroute {host}",
        False
    ),

    # ==================== DESIGNATED HOLDOUTS (Strictly for Validation) ====================
    # Holdout A: Process memory ranking with pipe
    (
        [
            "list top {count} processes sorted by memory consumption",
            "show top {count} memory consuming processes",
            "display processes using the most memory",
            "view top {count} processes by ram usage",
            "rank processes by memory and take top {count}",
        ],
        "ps aux --sort=-%mem | head -n {count}",
        True  # HOLDOUT
    ),
    # Holdout B: Tar extraction with destination flag
    (
        [
            "extract tar.gz archive to destination {path}",
            "uncompress archive.tar.gz into directory {path}",
            "extract tar archive to {path}",
            "unpack archive.tar.gz to directory {path}",
            "untar archive.tar.gz into target {path}",
        ],
        "tar -xzvf archive.tar.gz -C {path}",
        True  # HOLDOUT
    ),
    # Holdout C: Find files by permission 777
    (
        [
            "find all files with 777 permissions in {path}",
            "search for world-writable files with permission 777 in {path}",
            "locate files with mode 0777 inside {path}",
            "find world readable writable files 777 in {path}",
        ],
        "find {path} -type f -perm 0777",
        True  # HOLDOUT
    ),
    # Holdout D: Service enable and start
    (
        [
            "enable {service} service to start on boot and start it now",
            "enable and start {service} immediately",
            "set {service} to start at boot and start it now with systemctl",
            "systemctl enable --now for service {service}",
        ],
        "sudo systemctl enable --now {service}",
        True  # HOLDOUT
    ),
    # Holdout E: Docker prune stopped containers
    (
        [
            "remove all stopped docker containers without prompting",
            "prune unused docker containers forcefully",
            "force cleanup all exited docker containers",
            "docker container prune force without confirmation",
        ],
        "docker container prune -f",
        True  # HOLDOUT
    ),
]


# ==============================================================================
# PowerShell Template Specifications (with designated holdouts for zero leakage)
# ==============================================================================

PWSH_TEMPLATES = [
    # 1. Process on port
    (
        [
            "kill process listening on port {port}",
            "stop the process using port {port}",
            "terminate application on port {port}",
            "kill port {port} in powershell",
            "free up port {port} by terminating owning process",
            "find and kill process bound to port {port}",
            "kill the process listening on port {port} in powershell",
        ],
        "Stop-Process -Id (Get-NetTCPConnection -LocalPort {port}).OwningProcess -Force",
        False
    ),
    # 2. Stop process by name
    (
        [
            "kill all processes named {proc}",
            "force stop process {proc}",
            "terminate application {proc}",
            "stop-process {proc}",
            "kill all running instances of {proc}",
            "stop process with name {proc} forcefully",
        ],
        "Stop-Process -Name '{proc}' -Force",
        False
    ),
    # 3. Stop process by PID
    (
        [
            "terminate process with id {pid}",
            "stop process id {pid} forcefully",
            "kill process pid {pid}",
            "force stop process with pid {pid}",
            "kill process by pid {pid}",
        ],
        "Stop-Process -Id {pid} -Force",
        False
    ),
    # 4. Find files larger than size
    (
        [
            "find all files larger than {size}MB in {path}",
            "search for files bigger than {size}MB under {path}",
            "list files exceeding {size} megabytes in {path}",
            "get files over {size}MB in {path}",
            "find large files greater than {size}MB in {path}",
        ],
        "Get-ChildItem -Path '{path}' -File -Recurse | Where-Object {{ $_.Length -gt {size}MB }}",
        False
    ),
    # 5. Find by extension
    (
        [
            "find all {ext} files in {path}",
            "recursively search for {ext} files in {path}",
            "list {ext} files under {path}",
            "get all files with extension {ext} in {path}",
            "find files matching pattern *{ext} under {path}",
        ],
        "Get-ChildItem -Path '{path}' -Filter '*{ext}' -Recurse",
        False
    ),
    # 6. Delete files matching extension
    (
        [
            "remove all {ext} files from {path}",
            "delete all {ext} files inside {path}",
            "purge {ext} files under {path}",
            "force delete all {ext} files in {path}",
            "clean up {ext} files in {path}",
        ],
        "Remove-Item -Path '{path}\\*{ext}' -Force",
        False
    ),
    # 7. Test net connection (port test)
    (
        [
            "check if port {port} is open on {host}",
            "test connection to {host} on port {port}",
            "verify if {host} is listening on port {port}",
            "test tcp port {port} on {host}",
            "check reachability of port {port} on {host}",
        ],
        "Test-NetConnection -ComputerName '{host}' -Port {port}",
        False
    ),
    # 8. Select string (grep in file)
    (
        [
            "search for pattern '{pattern}' in file {path}\\app.log",
            "grep for '{pattern}' inside {path}\\app.log",
            "find text '{pattern}' in {path}\\app.log",
            "look for pattern '{pattern}' in {path}\\app.log case sensitive",
            "select lines containing '{pattern}' in {path}\\app.log",
        ],
        "Select-String -Path '{path}\\app.log' -Pattern '{pattern}' -CaseSensitive",
        False
    ),
    # 9. Select string across wildcard files
    (
        [
            "search for '{pattern}' in all {ext} files in {path}",
            "find occurrences of '{pattern}' in {path}\\*{ext}",
            "grep for '{pattern}' across all {ext} files under {path}",
            "look for string '{pattern}' in {path}\\*{ext}",
        ],
        "Get-ChildItem -Path '{path}\\*{ext}' | Select-String -Pattern '{pattern}'",
        False
    ),
    # 10. Head content
    (
        [
            "show top {count} lines of {path}\\app{ext}",
            "view first {count} lines of {path}\\app{ext}",
            "read head {count} lines of {path}\\app{ext}",
            "get first {count} lines from {path}\\app{ext}",
            "display beginning {count} lines of {path}\\app{ext}",
        ],
        "Get-Content -Path '{path}\\app{ext}' -TotalCount {count}",
        False
    ),
    # 11. Tail content
    (
        [
            "read the last {count} lines of {path}\\app{ext}",
            "view tail {count} lines of {path}\\app{ext}",
            "get last {count} lines of {path}\\app{ext}",
            "display bottom {count} lines from {path}\\app{ext}",
        ],
        "Get-Content -Path '{path}\\app{ext}' -Tail {count}",
        False
    ),
    # 12. Create directory
    (
        [
            "create a new directory named {folder} in {path}",
            "make directory {path}\\{folder}",
            "create folder {folder} at {path}",
            "new directory {path}\\{folder} in powershell",
        ],
        "New-Item -Path '{path}\\{folder}' -ItemType Directory -Force",
        False
    ),
    # 13. Create file
    (
        [
            "create an empty file named {name}{ext} in {path}",
            "touch file {name}{ext} in {path}",
            "new empty file at {path}\\{name}{ext}",
            "create new file {name}{ext} inside {path}",
        ],
        "New-Item -Path '{path}\\{name}{ext}' -ItemType File -Force",
        False
    ),
    # 14. Copy item
    (
        [
            "copy file {path}\\source{ext} to {path}\\dest{ext}",
            "make a copy of {path}\\source{ext} at {path}\\dest{ext}",
            "copy item {path}\\source{ext} to {path}\\dest{ext} force",
        ],
        "Copy-Item -Path '{path}\\source{ext}' -Destination '{path}\\dest{ext}' -Force",
        False
    ),
    # 15. Service restart
    (
        [
            "restart service {service}",
            "reboot windows service {service}",
            "force restart service named {service}",
            "restart-service for {service}",
            "stop and start service {service}",
        ],
        "Get-Service -Name '{service}' | Restart-Service -Force",
        False
    ),
    # 16. Service status
    (
        [
            "get status of service {service}",
            "check if service {service} is running",
            "view service details for {service}",
            "check windows service state for {service}",
        ],
        "Get-Service -Name '{service}'",
        False
    ),
    # 17. Compress archive (zip)
    (
        [
            "compress folder {path} into a zip archive",
            "zip directory {path} to C:\\Backup\\archive.zip",
            "create zip archive of {path} at C:\\Backup\\archive.zip",
            "compress-archive for folder {path}",
        ],
        "Compress-Archive -Path '{path}\\*' -DestinationPath 'C:\\Backup\\archive.zip' -Force",
        False
    ),
    # 18. Expand archive (unzip)
    (
        [
            "unzip archive C:\\Backup\\archive.zip to {path}",
            "extract zip file C:\\Backup\\archive.zip into {path}",
            "expand zip archive C:\\Backup\\archive.zip to destination {path}",
            "decompress C:\\Backup\\archive.zip into {path}",
        ],
        "Expand-Archive -Path 'C:\\Backup\\archive.zip' -DestinationPath '{path}' -Force",
        False
    ),
    # 19. Test path existence
    (
        [
            "check if file {path}\\config.json exists",
            "test existence of path {path}\\config.json",
            "verify whether {path}\\config.json exists",
            "test-path for {path}\\config.json",
        ],
        "Test-Path -Path '{path}\\config.json'",
        False
    ),
    # 20. Process list
    (
        [
            "list all running processes",
            "get system process list",
            "display active processes",
            "show all running windows processes",
        ],
        "Get-Process",
        False
    ),
    # 21. Piped: Filter services by Status Running
    (
        [
            "list all currently running windows services",
            "show all services with status running",
            "get services that are active and running",
            "filter services where status is running",
        ],
        "Get-Service | Where-Object {{ $_.Status -eq 'Running' }}",
        False
    ),
    # 22. Piped: Content unique sort
    (
        [
            "get unique sorted lines from {path}\\app{ext}",
            "read {path}\\app{ext} and sort unique entries",
            "display distinct lines in {path}\\app{ext}",
            "sort-object unique lines of {path}\\app{ext}",
        ],
        "Get-Content -Path '{path}\\app{ext}' | Sort-Object -Unique",
        False
    ),
    # 23. Git operations in PowerShell
    (
        [
            "create and switch to new git branch {branch}",
            "checkout new branch {branch} in git",
            "create new branch named {branch} and switch to it",
            "git branch checkout {branch}",
        ],
        "git checkout -b {branch}",
        False
    ),
    (
        [
            "check git status",
            "view git repository working tree status",
            "show modified files in git",
            "display current git status",
        ],
        "git status",
        False
    ),
    (
        [
            "show git commit history one line per commit for last {count} commits",
            "git log compact format top {count} commits",
            "view last {count} git commits on one line",
        ],
        "git log --oneline -n {count}",
        False
    ),
    (
        [
            "stash current uncommitted changes in git",
            "save working changes to git stash",
            "stash unstaged modifications",
        ],
        "git stash",
        False
    ),
    # 24. Docker in PowerShell
    (
        [
            "list all running and stopped docker containers",
            "show all docker containers including exited",
            "docker ps all containers",
        ],
        "docker ps -a",
        False
    ),
    (
        [
            "view live logs for container {proc}",
            "stream logs from docker container {proc}",
            "follow docker logs of {proc}",
        ],
        "docker logs -f --tail {count} {proc}",
        False
    ),
    (
        [
            "run docker container {proc} with port {port} mapped in background",
            "start detached docker container {proc} mapping port {port}",
            "docker run detached {proc} on port {port}",
        ],
        "docker run -d -p {port}:{port} --name {proc} {proc}:latest",
        False
    ),
    # 25. Remove directory recursive
    (
        [
            "delete folder {path}\\{folder} and all its contents recursively",
            "recursively remove directory {path}\\{folder}",
            "force purge directory {path}\\{folder}",
            "remove-item recursive on {path}\\{folder}",
        ],
        "Remove-Item -Path '{path}\\{folder}' -Recurse -Force",
        False
    ),
    # 26. DNS resolution
    (
        [
            "resolve dns for domain {host}",
            "lookup dns records for {host}",
            "query dns name {host}",
            "resolve-dnsname for {host}",
        ],
        "Resolve-DnsName -Name '{host}'",
        False
    ),

    # ==================== DESIGNATED HOLDOUTS (Strictly for Validation) ====================
    # Holdout A: Process sorting by memory in PowerShell
    (
        [
            "show top {count} processes sorted by memory working set",
            "get top {count} memory consuming processes",
            "list processes using most memory in powershell top {count}",
            "rank processes by workingset64 descending and take top {count}",
        ],
        "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First {count}",
        True  # HOLDOUT
    ),
    # Holdout B: File hash SHA256 computation
    (
        [
            "calculate sha256 hash of file {path}\\{name}.exe",
            "verify sha256 checksum for {path}\\{name}.exe",
            "get sha256 file hash of {path}\\{name}.exe",
            "compute sha256 hash on {path}\\{name}.exe",
        ],
        "Get-FileHash -Path '{path}\\{name}.exe' -Algorithm SHA256",
        True  # HOLDOUT
    ),
    # Holdout C: Web request / API query
    (
        [
            "send http get request to https://{host} using basic parsing",
            "invoke web request for https://{host}",
            "fetch web content from https://{host} with basic parsing",
            "query url https://{host} using invoke-webrequest",
        ],
        "Invoke-WebRequest -Uri 'https://{host}' -UseBasicParsing",
        True  # HOLDOUT
    ),
    # Holdout D: System event log query
    (
        [
            "get the newest {count} error events from system log",
            "view latest {count} system error event logs",
            "read top {count} error events from system eventlog",
            "query system event log for latest {count} errors",
        ],
        "Get-EventLog -LogName System -EntryType Error -Newest {count}",
        True  # HOLDOUT
    ),
    # Holdout E: Windows volume information
    (
        [
            "get disk volume details and free space",
            "show windows drive volumes formatted as auto-size table",
            "view volume information formatted as table",
            "display drive volumes with format-table autosize",
        ],
        "Get-Volume | Format-Table -AutoSize",
        True  # HOLDOUT
    ),
    # Holdout F: Process CPU filter and sort
    (
        [
            "find processes with cpu greater than {size} seconds sorted descending",
            "get high cpu processes above {size} in powershell",
            "list processes where cpu is greater than {size} sorted",
            "filter processes by cpu over {size} seconds descending",
        ],
        "Get-Process | Where-Object {{ $_.CPU -gt {size} }} | Sort-Object CPU -Descending",
        True  # HOLDOUT
    ),
    # Holdout G: File modification filter by days
    (
        [
            "find files modified in the last {days} days under {path} in powershell",
            "get-childitem for files changed within {days} days under {path}",
            "search files in {path} updated in the past {days} days using where-object",
            "filter files modified in last {days} days in {path}",
        ],
        "Get-ChildItem -Path '{path}' -Recurse | Where-Object {{ $_.LastWriteTime -gt (Get-Date).AddDays(-{days}) }}",
        True  # HOLDOUT
    ),
    # Holdout H: Registry software query
    (
        [
            "read windows current version registry properties",
            "query registry path HKLM Software Microsoft Windows CurrentVersion",
            "get-itemproperty for windows currentversion in registry",
            "inspect registry values for windows currentversion",
        ],
        "Get-ItemProperty -Path 'HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion'",
        True  # HOLDOUT
    ),
    # Holdout I: IP Configuration table
    (
        [
            "display ipv4 addresses formatted as table",
            "get all active ipv4 network addresses in powershell",
            "show ipv4 address table using get-netipaddress",
            "list ipv4 network adapters with format-table",
        ],
        "Get-NetIPAddress -AddressFamily IPv4 | Format-Table",
        True  # HOLDOUT
    ),
    # Holdout J: Operating system details via CimInstance
    (
        [
            "get windows os version and build information using ciminstance",
            "query win32_operatingsystem caption and version in powershell",
            "display operating system name and version via get-ciminstance",
            "view windows operating system version details",
        ],
        "Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object Caption, Version",
        True  # HOLDOUT
    ),
]


# ==============================================================================
# Sample Generation Engine with True Template-Level Splitting
# ==============================================================================

SUBDIRS_LINUX = ["app", "logs", "config", "data", "backup", "src", "bin", "tests", "var", "cache", "build"]
SUBDIRS_PWSH = ["App", "Logs", "Config", "Data", "Backup", "Src", "Bin", "Tests", "Cache", "Temp"]

def generate_samples_from_template(
    prompts: List[str],
    cmd_template: str,
    os_name: str,
    num_samples: int,
    seen_global: Set[str],
) -> List[Dict[str, str]]:
    """Generates synthetic samples by interpolating random variables into templates."""
    results = []
    max_attempts = max(num_samples * 15, 600)

    for _ in range(max_attempts):
        if random.random() < 0.6:
            port = random.choice(PORTS)
        else:
            port = random.randint(1024, 65000)

        host = random.choice(HOSTS)
        proc_base = random.choice(PROCESSES)
        proc_suffix = random.choice(["", "-server", "-worker", "-daemon", "2", "3"]) if random.random() < 0.4 else ""
        proc = f"{proc_base}{proc_suffix}"

        service = random.choice(SERVICES)
        ext = random.choice(EXTENSIONS)
        size = random.choice([5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 750, 1024, 2048, 4096])
        days = random.choice([1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 180, 365])
        count = random.choice([3, 5, 10, 15, 20, 25, 30, 50, 75, 100, 200, 500])
        pid = random.randint(1000, 65535)
        branch = random.choice(BRANCHES)
        user = random.choice(USERS)
        group = random.choice(GROUPS)
        volume = random.choice(VOLUMES)

        folder = random.choice(["app", "build", "dist", "data", "backup", "logs", "temp", "release", "cache"])
        num_suffix = f"_{random.randint(1, 99)}" if random.random() < 0.5 else ""
        name = f"{random.choice(['app', 'server', 'data', 'test', 'config', 'output', 'service'])}{num_suffix}"

        base_path = random.choice(PWSH_PATHS if os_name == "powershell" else LINUX_PATHS)
        if random.random() < 0.6:
            subdir = random.choice(SUBDIRS_PWSH if os_name == "powershell" else SUBDIRS_LINUX)
            sep = "\\" if os_name == "powershell" else "/"
            path = f"{base_path}{sep}{subdir}"
        else:
            path = base_path

        pattern = random.choice(PATTERNS)

        vars_dict = {
            "port": port,
            "host": host,
            "proc": proc,
            "service": service,
            "ext": ext,
            "size": size,
            "days": days,
            "count": count,
            "pid": pid,
            "branch": branch,
            "user": user,
            "group": group,
            "volume": volume,
            "folder": folder,
            "name": name,
            "path": path,
            "pattern": pattern,
        }

        raw_prompt = random.choice(prompts)
        try:
            p_formatted = raw_prompt.format(**vars_dict)
            c_formatted = cmd_template.format(**vars_dict)
        except KeyError:
            continue

        p_clean = clean_text(p_formatted)
        c_clean = clean_command(c_formatted)

        key = f"{p_clean}|{c_clean}"
        if key not in seen_global and is_valid_sample(p_clean, c_clean):
            seen_global.add(key)
            results.append({"os": os_name, "prompt": p_clean, "cmd": c_clean})

        if len(results) >= num_samples:
            break

    return results


def build_scaled_dataset(
    samples_per_os: int = 10000,
    val_ratio: float = 0.10,
    test_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str = "data",
) -> Tuple[str, str, str]:
    """
    Constructs ~20,000 balanced pairs across a clean 3-way partition:
    - Train: Comprehensive coverage across all command families (~16,000 samples).
    - Validation: In-distribution evaluation on unseen prompt phrasings and argument substitutions (~2,000 samples).
    - Held-Out Test: Independent evaluation on unseen compositional formulations across all 9 domains (~1,600 samples).
    """
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"
    test_file = out_path / "test.jsonl"

    train_samples: List[Dict[str, str]] = []
    val_samples: List[Dict[str, str]] = []
    test_samples: List[Dict[str, str]] = []
    seen_all: Set[str] = set()
    train_prompts_seen: Set[str] = set()
    val_prompts_seen: Set[str] = set()

    # Pre-partition templates for each OS
    os_partitioned = {}
    for os_name, templates in [("linux", LINUX_TEMPLATES), ("powershell", PWSH_TEMPLATES)]:
        partitioned = []
        for prompts, cmd_tmpl, is_h in templates:
            n_p = len(prompts)
            if n_p >= 3:
                val_count = max(1, int(n_p * 0.18))
                test_count = max(1, int(n_p * 0.18))
                train_p = prompts[: n_p - val_count - test_count]
                val_p = prompts[n_p - val_count - test_count : n_p - test_count]
                test_p = prompts[n_p - test_count :]
            elif n_p == 2:
                train_p = [prompts[0]]
                val_p = [prompts[1]]
                test_p = [prompts[1]]
            else:
                train_p = prompts
                val_p = prompts
                test_p = prompts
            partitioned.append((train_p, val_p, test_p, cmd_tmpl))
        os_partitioned[os_name] = partitioned

    # 1. Generate ALL Training samples across both OSes
    for os_name, partitioned in os_partitioned.items():
        target_train = samples_per_os - int(samples_per_os * val_ratio) - int(samples_per_os * test_ratio)
        per_tmpl_train = max(10, target_train // len(partitioned))
        train_os_samples = []

        for train_p, _, _, cmd_tmpl in partitioned:
            generated = generate_samples_from_template(train_p, cmd_tmpl, os_name, per_tmpl_train, seen_all)
            for s in generated:
                train_prompts_seen.add(s["prompt"])
            train_os_samples.extend(generated)

        # Top up training if needed to reach target_train
        pass_idx = 0
        while len(train_os_samples) < target_train and pass_idx < 10:
            pass_idx += 1
            needed = target_train - len(train_os_samples)
            top_up_per = max(2, needed // len(partitioned))
            for train_p, _, _, cmd_tmpl in partitioned:
                gen = generate_samples_from_template(train_p, cmd_tmpl, os_name, top_up_per, seen_all)
                for s in gen:
                    train_prompts_seen.add(s["prompt"])
                train_os_samples.extend(gen)
                if len(train_os_samples) >= target_train:
                    break

        train_samples.extend(train_os_samples[:target_train])

    # 2. Generate ALL Validation samples across both OSes (strictly disjoint from train prompts)
    for os_name, partitioned in os_partitioned.items():
        target_val = int(samples_per_os * val_ratio)
        per_tmpl_val = max(3, target_val // len(partitioned))
        val_os_samples = []

        for _, val_p, _, cmd_tmpl in partitioned:
            generated = generate_samples_from_template(val_p, cmd_tmpl, os_name, per_tmpl_val * 3, seen_all)
            unique_val = [s for s in generated if s["prompt"] not in train_prompts_seen and s["prompt"] not in val_prompts_seen]
            for s in unique_val[:per_tmpl_val]:
                val_prompts_seen.add(s["prompt"])
                val_os_samples.append(s)

        val_samples.extend(val_os_samples[:target_val])

    # 3. Generate ALL Held-Out Test samples across both OSes (strictly disjoint from train & val)
    for os_name, partitioned in os_partitioned.items():
        target_test = int(samples_per_os * test_ratio)
        per_tmpl_test = max(2, target_test // len(partitioned))
        test_os_samples = []

        for _, _, test_p, cmd_tmpl in partitioned:
            generated = generate_samples_from_template(test_p, cmd_tmpl, os_name, per_tmpl_test * 3, seen_all)
            unique_test = [
                s for s in generated
                if s["prompt"] not in train_prompts_seen and s["prompt"] not in val_prompts_seen
            ]
            test_os_samples.extend(unique_test[:per_tmpl_test])

        test_samples.extend(test_os_samples[:target_test])

    # Shuffle splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # Write files
    logger.info(f"Writing {len(train_samples):,} training records to {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Writing {len(val_samples):,} validation records to {val_file}...")
    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Writing {len(test_samples):,} held-out test records to {test_file}...")
    with open(test_file, "w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    train_linux_count = sum(1 for x in train_samples if x["os"] == "linux")
    train_pwsh_count = sum(1 for x in train_samples if x["os"] == "powershell")
    val_linux_count = sum(1 for x in val_samples if x["os"] == "linux")
    val_pwsh_count = sum(1 for x in val_samples if x["os"] == "powershell")
    test_linux_count = sum(1 for x in test_samples if x["os"] == "linux")
    test_pwsh_count = sum(1 for x in test_samples if x["os"] == "powershell")

    logger.info("=" * 60)
    logger.info(" SCALED DATASET GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f" Train Total : {len(train_samples):,} (Linux: {train_linux_count:,}, PowerShell: {train_pwsh_count:,})")
    logger.info(f" Val Total   : {len(val_samples):,} (Linux: {val_linux_count:,}, PowerShell: {val_pwsh_count:,})")
    logger.info(f" Test Total  : {len(test_samples):,} (Linux: {test_linux_count:,}, PowerShell: {test_pwsh_count:,})")
    logger.info("=" * 60)

    return str(train_file), str(val_file), str(test_file)


def main():
    parser = argparse.ArgumentParser(description="CommandLLM Scaled Dataset Builder")
    parser.add_argument("--samples-per-os", type=int, default=10000, help="Target samples per OS (default: 10000)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio (default: 0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.08, help="Held-out test ratio (default: 0.08)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_scaled_dataset(
        samples_per_os=args.samples_per_os,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
