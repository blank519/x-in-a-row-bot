import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import {
	appendFile,
	lstat,
	mkdir,
	readdir,
	readFile,
	realpath,
	rename,
	rm,
	stat,
	writeFile,
} from "node:fs/promises";
import { basename, dirname, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import {
	withFileMutationQueue,
	type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";

/** Conservative defaults for the repository's current 15x15 Gomoku stack. */
export const DEFAULT_REQUIRED_FREE_VRAM_MIB = 4096;
export const DEFAULT_STARTUP_TIMEOUT_SECONDS = 600;
export const STARTUP_POLL_INTERVAL_MS = 1_000;
export const DUPLICATE_INTERRUPT_GRACE_MS = 5_000;
export const DUPLICATE_TERMINATE_GRACE_MS = 3_000;
export const DUPLICATE_KILL_GRACE_MS = 3_000;
const LOCK_POLL_INTERVAL_MS = 250;
const LOCK_STALE_AFTER_MS = 30 * 60 * 1_000;
const LOG_READ_LIMIT_BYTES = 512 * 1024;
const LOG_TAIL_BYTES = 16 * 1024;
const PID_RECORD_VERSION = 1;

export const launchRunSchema = Type.Object({
	scriptPath: Type.String({
		description: "Root-level .py training-script copy already configured with the exact run name. A leading @ is accepted; directories and traversal are rejected.",
	}),
	runName: Type.String({
		description: "Exact filesystem-safe MLflow run name encoded by the script (letters, digits, dot, underscore, hyphen; used for deterministic log/PID files and case-insensitive duplicate replacement).",
	}),
	requiredFreeVramMiB: Type.Optional(Type.Integer({
		minimum: 1,
		description: `Minimum free VRAM required on one GPU in MiB (default ${DEFAULT_REQUIRED_FREE_VRAM_MIB}); the selected GPU is bound to the child.`,
	})),
	startupTimeoutSeconds: Type.Optional(Type.Integer({
		minimum: 1,
		maximum: 3600,
		description: `Seconds to wait for real PPO progress or a fresh finite MLflow metric sample (default ${DEFAULT_STARTUP_TIMEOUT_SECONDS}).`,
	})),
}, { additionalProperties: false });

export type LaunchRunInput = Static<typeof launchRunSchema>;
export type StartupEvidenceType = "log_progress" | "mlflow_metric";

export interface GpuEvidence {
	index: number;
	name: string;
	freeMiB: number;
	totalMiB: number;
}

export interface ProcessEvidence {
	pid: number;
	pgid: number;
	sid: number;
	startToken: string;
	marker: string;
	cwd: string;
	cmdline: string[];
}

export interface DuplicateKillEvidence {
	pid: number;
	pgid: number;
	marker: string;
	startToken: string;
	signals: string[];
	escalated: boolean;
	leaderIdentityGone: boolean;
	processGroupGone: boolean;
	markerGone: boolean;
	mlflowRunIds: string[];
	mlflowReconciled: string[];
}

export interface MlflowRunEvidence {
	runId: string;
	runName: string | null;
	status: string | null;
	startTimeMs: number | null;
	runPath: string;
	experimentId: string;
}

export interface LaunchRunDetails {
	outcome: "started";
	pid: number;
	pgid: number;
	runName: string;
	scriptPath: string;
	launchArgv: string[];
	launchCommand: string;
	launchTimestampMs: number;
	launchTimestamp: string;
	logPath: string;
	pidRecordPath: string;
	gpu: GpuEvidence;
	requiredFreeVramMiB: number;
	allGpus: GpuEvidence[];
	duplicates: DuplicateKillEvidence[];
	startupEvidence: {
		type: StartupEvidenceType;
		value: string;
		logOffset: number;
	};
	mlflowRunId: string | null;
	mlflowRunPath: string | null;
	marker: string;
	startToken: string;
}

export interface LaunchRunResult {
	content: Array<{ type: "text"; text: string }>;
	details: LaunchRunDetails;
}

export interface ValidatedPaths {
	repositoryRoot: string;
	scriptPath: string;
	scriptRelative: string;
	logsPath: string;
	logPath: string;
	pidRecordPath: string;
	lockPath: string;
	mlrunsPath: string;
	repositoryWsl: string;
	scriptWsl: string;
	logWsl: string;
	pythonWsl: string;
	mlrunsWsl: string;
}

interface PidRecord {
	version: number;
	pid: number;
	pgid: number;
	startToken: string;
	marker: string;
	markerPrefix: string;
	runName: string;
	normalizedRunName: string;
	scriptPath: string;
	repositoryRoot: string;
	repositoryWsl: string;
	scriptWsl: string;
	launchedAtMs: number;
	launchedAt: string;
	mlflowRunId: string | null;
	mlflowRunPath: string | null;
	gpu: GpuEvidence;
	requiredFreeVramMiB: number;
}

interface LockRecord {
	token: string;
	hostPid: number;
	hostPlatform: string;
	createdAtMs: number;
	runName: string;
}

interface ExecResult {
	stdout: string;
	stderr: string;
	code: number;
}

export interface LaunchClock {
	now(): number;
	sleep(milliseconds: number, signal?: AbortSignal): Promise<void>;
}

/**
 * High-level dependency seams keep the lifecycle deterministic under evaluation.
 * The registered tool always binds PRODUCTION_LAUNCH_RUN_DEPENDENCIES; there is
 * no schema field or public runtime switch that bypasses a safety stage.
 */
export interface LaunchRunDependencies {
	clock: LaunchClock;
	toWslPath(path: string, signal?: AbortSignal): Promise<string>;
	validateRuntime(paths: ValidatedPaths, signal?: AbortSignal): Promise<void>;
	queryGpus(signal?: AbortSignal): Promise<GpuEvidence[]>;
	inspectProcesses(markerPrefix: string, paths: ValidatedPaths, signal?: AbortSignal): Promise<ProcessEvidence[]>;
	signalProcessGroup(process: ProcessEvidence, signalName: "SIGINT" | "SIGTERM" | "SIGKILL", paths: ValidatedPaths, signal?: AbortSignal): Promise<void>;
	processGroupExists(pgid: number, paths: ValidatedPaths, signal?: AbortSignal): Promise<boolean>;
	launchDetached(request: {
		paths: ValidatedPaths;
		marker: string;
		runNameBase64: string;
		gpuIndex: number;
	}, signal?: AbortSignal): Promise<number>;
	reconcileMlflowRun(runId: string, paths: ValidatedPaths, signal?: AbortSignal): Promise<void>;
}

function cancellationError(message = "Run launch was cancelled."): Error {
	const error = new Error(message);
	error.name = "AbortError";
	return error;
}

function abortIfRequested(signal?: AbortSignal): void {
	if (signal?.aborted) throw cancellationError();
}

async function systemSleep(milliseconds: number, signal?: AbortSignal): Promise<void> {
	abortIfRequested(signal);
	if (milliseconds <= 0) return;
	await new Promise<void>((resolvePromise, reject) => {
		const timer = setTimeout(done, milliseconds);
		function done(): void {
			signal?.removeEventListener("abort", cancelled);
			resolvePromise();
		}
		function cancelled(): void {
			clearTimeout(timer);
			signal?.removeEventListener("abort", cancelled);
			reject(cancellationError());
		}
		signal?.addEventListener("abort", cancelled, { once: true });
	});
}

export const SYSTEM_LAUNCH_CLOCK: LaunchClock = {
	now: () => Date.now(),
	sleep: systemSleep,
};

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function quoteForDisplay(value: string): string {
	return /^[A-Za-z0-9_./:\\=-]+$/.test(value) ? value : JSON.stringify(value);
}

function samePath(left: string, right: string): boolean {
	return process.platform === "win32"
		? left.toLocaleLowerCase() === right.toLocaleLowerCase()
		: left === right;
}

function insideRoot(root: string, candidate: string): boolean {
	const rel = relative(root, candidate);
	return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}

function normalizeRunName(runName: string): string {
	return runName.toLocaleLowerCase();
}

export function validateRunName(runName: string): string {
	if (runName !== runName.trim()) throw new Error("runName must not contain leading or trailing whitespace.");
	if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(runName) || runName === "." || runName === ".." || runName.endsWith(".")) {
		throw new Error("runName must be 1-128 filesystem-safe characters: start with a letter/digit, then use only letters, digits, '.', '_', or '-', without a trailing dot.");
	}
	return runName;
}

function normalizeScriptArgument(scriptPath: string): string {
	const normalized = scriptPath.replace(/^@/, "");
	if (!normalized) throw new Error("scriptPath cannot be empty.");
	if (normalized !== basename(normalized) || normalized.includes("/") || normalized.includes("\\") || normalized === "." || normalized === "..") {
		throw new Error("scriptPath must name one root-level Python file; directories, absolute paths, and traversal are not allowed.");
	}
	if (extname(normalized).toLocaleLowerCase() !== ".py") throw new Error("scriptPath must be a root-level .py file.");
	return normalized;
}

async function validatePaths(input: LaunchRunInput, cwd: string, dependencies: LaunchRunDependencies, signal?: AbortSignal): Promise<ValidatedPaths> {
	abortIfRequested(signal);
	const runName = validateRunName(input.runName);
	const scriptRelative = normalizeScriptArgument(input.scriptPath);
	let repositoryRoot: string;
	try {
		repositoryRoot = await realpath(resolve(cwd));
		if (!(await stat(repositoryRoot)).isDirectory()) throw new Error("not a directory");
	} catch (error) {
		throw new Error(`ctx.cwd is not a readable project directory: ${resolve(cwd)} (${errorText(error)})`);
	}
	const unresolvedScript = resolve(repositoryRoot, scriptRelative);
	if (!insideRoot(repositoryRoot, unresolvedScript) || !samePath(dirname(unresolvedScript), repositoryRoot)) {
		throw new Error("scriptPath must resolve directly inside the repository root.");
	}
	try {
		const linkInfo = await lstat(unresolvedScript);
		if (linkInfo.isSymbolicLink()) throw new Error("symbolic links are not accepted");
		if (!linkInfo.isFile()) throw new Error("not a regular file");
	} catch (error) {
		throw new Error(`scriptPath is missing or is not a regular root-level .py file: ${unresolvedScript} (${errorText(error)})`);
	}
	const scriptPath = await realpath(unresolvedScript);
	if (!insideRoot(repositoryRoot, scriptPath) || !samePath(dirname(scriptPath), repositoryRoot)) {
		throw new Error(`scriptPath resolves outside the repository root: ${scriptPath}`);
	}

	const logsPath = join(repositoryRoot, "logs");
	const logPath = join(logsPath, `${runName}.log`);
	const pidRecordPath = join(logsPath, `${runName}.pid.json`);
	const lockHash = createHash("sha256").update(normalizeRunName(runName)).digest("hex").slice(0, 24);
	const lockPath = join(logsPath, `.launch-run-${lockHash}.lock`);
	const mlrunsPath = join(repositoryRoot, "mlruns");
	await mkdir(logsPath, { recursive: true });

	let repositoryWsl: string;
	let scriptWsl: string;
	let logWsl: string;
	let mlrunsWsl: string;
	try {
		[repositoryWsl, scriptWsl, logWsl, mlrunsWsl] = await Promise.all([
			dependencies.toWslPath(repositoryRoot, signal),
			dependencies.toWslPath(scriptPath, signal),
			dependencies.toWslPath(logPath, signal),
			dependencies.toWslPath(mlrunsPath, signal),
		]);
	} catch (error) {
		throw new Error(`Unable to resolve project paths in WSL: ${errorText(error)}`);
	}
	const pythonWsl = `${repositoryWsl.replace(/\/$/, "")}/.venv/bin/python`;
	const paths: ValidatedPaths = {
		repositoryRoot,
		scriptPath,
		scriptRelative,
		logsPath,
		logPath,
		pidRecordPath,
		lockPath,
		mlrunsPath,
		repositoryWsl,
		scriptWsl,
		logWsl,
		pythonWsl,
		mlrunsWsl,
	};
	await dependencies.validateRuntime(paths, signal);
	return paths;
}

function executableForWsl(command: string, args: string[]): { command: string; args: string[] } {
	if (process.platform === "win32") return { command: "wsl.exe", args: ["--exec", command, ...args] };
	return { command, args };
}

async function execCapture(command: string, args: string[], options: { signal?: AbortSignal; timeoutMs?: number } = {}): Promise<ExecResult> {
	abortIfRequested(options.signal);
	const actual = executableForWsl(command, args);
	return new Promise<ExecResult>((resolvePromise, reject) => {
		const child = spawn(actual.command, actual.args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		let settled = false;
		const timeout = options.timeoutMs == null ? undefined : setTimeout(() => {
			child.kill();
			finish(new Error(`${command} timed out after ${options.timeoutMs} ms.`));
		}, options.timeoutMs);
		function cleanup(): void {
			if (timeout) clearTimeout(timeout);
			options.signal?.removeEventListener("abort", cancelled);
		}
		function finish(error?: Error, result?: ExecResult): void {
			if (settled) return;
			settled = true;
			cleanup();
			if (error) reject(error);
			else resolvePromise(result!);
		}
		function cancelled(): void {
			child.kill();
			finish(cancellationError(`WSL command ${command} was cancelled.`));
		}
		options.signal?.addEventListener("abort", cancelled, { once: true });
		child.stdout.on("data", (chunk) => { stdout += String(chunk); });
		child.stderr.on("data", (chunk) => { stderr += String(chunk); });
		child.on("error", (error) => finish(new Error(`Unable to execute ${actual.command}: ${error.message}`)));
		child.on("close", (code) => finish(undefined, { stdout, stderr, code: code ?? -1 }));
	});
}

async function checkedWsl(command: string, args: string[], signal?: AbortSignal, timeoutMs = 30_000): Promise<string> {
	const result = await execCapture(command, args, { signal, timeoutMs });
	if (result.code !== 0) {
		throw new Error(`${command} failed with exit ${result.code}: ${(result.stderr || result.stdout).trim() || "no diagnostic output"}`);
	}
	return result.stdout;
}

export function parseNvidiaSmiCsv(text: string): GpuEvidence[] {
	const devices: GpuEvidence[] = [];
	for (const rawLine of text.split(/\r?\n/)) {
		const line = rawLine.trim();
		if (!line) continue;
		const columns = line.split(",").map((column) => column.trim());
		if (columns.length < 4) throw new Error(`Malformed nvidia-smi row: ${rawLine}`);
		const index = Number(columns.shift());
		const totalMiB = Number(columns.pop());
		const freeMiB = Number(columns.pop());
		const name = columns.join(",").trim();
		if (!Number.isInteger(index) || index < 0 || !name || !Number.isFinite(freeMiB) || freeMiB < 0 || !Number.isFinite(totalMiB) || totalMiB <= 0 || freeMiB > totalMiB) {
			throw new Error(`Malformed nvidia-smi GPU values: ${rawLine}`);
		}
		devices.push({ index, name, freeMiB, totalMiB });
	}
	if (!devices.length) throw new Error("nvidia-smi returned no GPU rows.");
	devices.sort((left, right) => left.index - right.index || left.name.localeCompare(right.name));
	const seen = new Set<number>();
	for (const device of devices) {
		if (seen.has(device.index)) throw new Error(`nvidia-smi returned duplicate GPU index ${device.index}.`);
		seen.add(device.index);
	}
	return devices;
}

const PROCESS_INSPECTOR = String.raw`
import json, os, sys
prefix, expected_cwd = sys.argv[1:3]
out = []
for entry in os.listdir('/proc'):
    if not entry.isdigit():
        continue
    pid = int(entry)
    try:
        env_raw = open(f'/proc/{pid}/environ', 'rb').read().split(b'\0')
        env = {}
        for item in env_raw:
            if b'=' in item:
                key, value = item.split(b'=', 1)
                env[key.decode('utf-8', 'replace')] = value.decode('utf-8', 'replace')
        marker = env.get('PI_LAUNCH_RUN_MARKER', '')
        if not marker.startswith(prefix):
            continue
        stat_text = open(f'/proc/{pid}/stat', encoding='utf-8').read()
        close = stat_text.rfind(')')
        fields = stat_text[close + 2:].split()
        start_token = fields[19]
        cwd = os.readlink(f'/proc/{pid}/cwd')
        cmdline = [part.decode('utf-8', 'replace') for part in open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0') if part]
        if cwd != expected_cwd:
            continue
        out.append({'pid': pid, 'pgid': os.getpgid(pid), 'sid': os.getsid(pid), 'startToken': start_token,
                    'marker': marker, 'cwd': cwd, 'cmdline': cmdline})
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, IndexError):
        pass
print(json.dumps(sorted(out, key=lambda row: (row['pgid'], row['pid']))))
`;

const SIGNAL_GROUP = String.raw`
import json, os, signal, sys
expected = json.loads(sys.argv[1])
sig = getattr(signal, sys.argv[2])
found = False
try:
    pid = int(expected['pid'])
    env = open(f'/proc/{pid}/environ', 'rb').read().split(b'\0')
    marker = None
    for item in env:
        if item.startswith(b'PI_LAUNCH_RUN_MARKER='):
            marker = item.split(b'=', 1)[1].decode('utf-8', 'replace')
            break
    stat_text = open(f'/proc/{pid}/stat', encoding='utf-8').read()
    close = stat_text.rfind(')')
    start_token = stat_text[close + 2:].split()[19]
    found = (marker == expected['marker'] and start_token == expected['startToken']
             and os.getpgid(pid) == expected['pgid'])
except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, IndexError, KeyError, ValueError):
    pass
if not found:
    print(json.dumps({'signalled': False, 'reason': 'verified marker/PID/start-token/process-group not found'}))
    sys.exit(3)
os.killpg(expected['pgid'], sig)
print(json.dumps({'signalled': True, 'signal': sys.argv[2], 'pgid': expected['pgid']}))
`;

const GROUP_EXISTS = String.raw`
import os, sys
pgid = int(sys.argv[1])
try:
    os.killpg(pgid, 0)
except ProcessLookupError:
    print('false')
except PermissionError:
    print('true')
else:
    print('true')
`;

const LAUNCH_SHELL = String.raw`
set -euo pipefail
repo=$1
python=$2
script=$3
log=$4
marker=$5
run_name_b64=$6
gpu=$7
cd -- "$repo"
umask 077
capture="\${log}.launch.$$.pid"
rm -f -- "$capture" "\${capture}.tmp"
nohup setsid bash -c '
set -euo pipefail
capture=$1
python=$2
script=$3
marker=$4
run_name_b64=$5
gpu=$6
printf "%s\n" "$$" >"\${capture}.tmp"
mv -- "\${capture}.tmp" "$capture"
exec env CUDA_VISIBLE_DEVICES="$gpu" PI_LAUNCH_RUN_MARKER="$marker" PI_LAUNCH_RUN_NAME_B64="$run_name_b64" "$python" -u "$script"
' launch-run-child "$capture" "$python" "$script" "$marker" "$run_name_b64" "$gpu" </dev/null >>"$log" 2>&1 &
wrapper=$!
for _ in $(seq 1 100); do
    if [ -s "$capture" ]; then
        pid=$(cat -- "$capture")
        rm -f -- "$capture"
        printf '%s\n' "$pid"
        exit 0
    fi
    if ! kill -0 "$wrapper" 2>/dev/null; then
        wait "$wrapper" || true
        rm -f -- "$capture" "\${capture}.tmp"
        echo "detached child exited before publishing its Linux PID" >&2
        exit 1
    fi
    sleep 0.05
done
rm -f -- "$capture" "\${capture}.tmp"
echo "timed out waiting for detached Linux PID" >&2
exit 1
`;

const RECONCILE_MLFLOW = String.raw`
from pathlib import Path
import sys
from mlflow.tracking import MlflowClient
run_id, root = sys.argv[1:3]
client = MlflowClient(tracking_uri=Path(root).resolve().as_uri())
run = client.get_run(run_id)
status = str(run.info.status).upper()
if status in {'RUNNING', 'SCHEDULED'}:
    client.set_terminated(run_id, status='KILLED')
print(client.get_run(run_id).info.status)
`;

async function productionToWslPath(path: string, signal?: AbortSignal): Promise<string> {
	if (process.platform !== "win32") return path;
	return (await checkedWsl("wslpath", ["-a", path], signal)).trim();
}

async function productionValidateRuntime(paths: ValidatedPaths, signal?: AbortSignal): Promise<void> {
	const script = "set -eu; test -d \"$1\"; test -f \"$2\"; test -x \"$3\"; command -v setsid >/dev/null; command -v nohup >/dev/null; command -v nvidia-smi >/dev/null";
	try {
		await checkedWsl("bash", ["-lc", script, "launch-run-check", paths.repositoryWsl, paths.scriptWsl, paths.pythonWsl], signal);
	} catch (error) {
		throw new Error(`WSL prerequisites are unavailable (project/script, .venv/bin/python, setsid, nohup, and nvidia-smi are required): ${errorText(error)}`);
	}
}

async function productionQueryGpus(signal?: AbortSignal): Promise<GpuEvidence[]> {
	let output: string;
	try {
		output = await checkedWsl("bash", ["-lc", "exec nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits"], signal);
	} catch (error) {
		throw new Error(`VRAM preflight could not run nvidia-smi in WSL: ${errorText(error)}`);
	}
	try { return parseNvidiaSmiCsv(output); }
	catch (error) { throw new Error(`VRAM preflight could not parse nvidia-smi output: ${errorText(error)}`); }
}

async function productionInspectProcesses(markerPrefix: string, paths: ValidatedPaths, signal?: AbortSignal): Promise<ProcessEvidence[]> {
	const output = await checkedWsl(paths.pythonWsl, ["-c", PROCESS_INSPECTOR, markerPrefix, paths.repositoryWsl], signal);
	let parsed: unknown;
	try { parsed = JSON.parse(output); } catch { throw new Error(`Process inspector returned malformed JSON: ${output.trim()}`); }
	if (!Array.isArray(parsed)) throw new Error("Process inspector returned a non-array result.");
	return parsed as ProcessEvidence[];
}

async function productionSignalProcessGroup(processEvidence: ProcessEvidence, signalName: "SIGINT" | "SIGTERM" | "SIGKILL", paths: ValidatedPaths, signal?: AbortSignal): Promise<void> {
	const payload = JSON.stringify({ marker: processEvidence.marker, pid: processEvidence.pid, startToken: processEvidence.startToken, pgid: processEvidence.pgid });
	await checkedWsl(paths.pythonWsl, ["-c", SIGNAL_GROUP, payload, signalName], signal);
}

async function productionProcessGroupExists(pgid: number, paths: ValidatedPaths, signal?: AbortSignal): Promise<boolean> {
	return (await checkedWsl(paths.pythonWsl, ["-c", GROUP_EXISTS, String(pgid)], signal)).trim() === "true";
}

async function productionLaunchDetached(request: { paths: ValidatedPaths; marker: string; runNameBase64: string; gpuIndex: number }, _signal?: AbortSignal): Promise<number> {
	// Do not abort the transient wsl.exe wrapper after the background child may
	// exist: awaiting its PID is what makes cancellation cleanup race-free.
	const launched = await execCapture("bash", [
		"-lc",
		LAUNCH_SHELL,
		"launch-run",
		request.paths.repositoryWsl,
		request.paths.pythonWsl,
		request.paths.scriptWsl,
		request.paths.logWsl,
		request.marker,
		request.runNameBase64,
		String(request.gpuIndex),
	], { timeoutMs: 30_000 });
	// Some WSL builds propagate a detached descendant's later signal as the
	// transient wsl.exe status even after the launcher published its PID. The
	// PID is only a candidate: the core immediately verifies marker, start token,
	// cwd, script argv, PGID, and SID before trusting it.
	const pidText = launched.stdout.trim().split(/\r?\n/).at(-1) ?? "";
	const pid = Number(pidText);
	if (!Number.isInteger(pid) || pid <= 1) {
		throw new Error(`Detached WSL launcher failed (exit ${launched.code}) without a valid Linux PID: ${(launched.stderr || launched.stdout).trim()}`);
	}
	return pid;
}

async function productionReconcileMlflowRun(runId: string, paths: ValidatedPaths, signal?: AbortSignal): Promise<void> {
	const status = (await checkedWsl(paths.pythonWsl, ["-c", RECONCILE_MLFLOW, runId, paths.mlrunsWsl], signal)).trim().toLocaleUpperCase();
	if (status === "RUNNING" || status === "SCHEDULED") throw new Error(`MLflow lifecycle API left run ${runId} active (${status}).`);
}

export const PRODUCTION_LAUNCH_RUN_DEPENDENCIES: LaunchRunDependencies = {
	clock: SYSTEM_LAUNCH_CLOCK,
	toWslPath: productionToWslPath,
	validateRuntime: productionValidateRuntime,
	queryGpus: productionQueryGpus,
	inspectProcesses: productionInspectProcesses,
	signalProcessGroup: productionSignalProcessGroup,
	processGroupExists: productionProcessGroupExists,
	launchDetached: productionLaunchDetached,
	reconcileMlflowRun: productionReconcileMlflowRun,
};

function parseSimpleYaml(text: string): Record<string, string> {
	const result: Record<string, string> = {};
	for (const rawLine of text.split(/\r?\n/)) {
		const line = rawLine.trim();
		if (!line || line.startsWith("#")) continue;
		const colon = line.indexOf(":");
		if (colon <= 0) continue;
		const key = line.slice(0, colon).trim();
		let value = line.slice(colon + 1).trim();
		if (value.length >= 2 && ((value.startsWith("'") && value.endsWith("'")) || (value.startsWith('"') && value.endsWith('"')))) {
			value = value.slice(1, -1);
		}
		result[key] = value;
	}
	return result;
}

async function readText(path: string): Promise<string | undefined> {
	try { return await readFile(path, "utf8"); } catch { return undefined; }
}

function parseStoredTime(value: string | undefined): number | null {
	if (value == null || value.trim() === "") return null;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : null;
}

function normalizedMlflowStatus(status: string | null): "active" | "terminal" | "unknown" {
	if (status == null) return "unknown";
	const value = status.trim().toLocaleLowerCase().replace(/[\s_-]+/g, "");
	if (["1", "2", "running", "active", "scheduled", "pending"].includes(value)) return "active";
	if (["3", "4", "5", "finished", "completed", "succeeded", "failed", "killed", "cancelled", "canceled", "terminated"].includes(value)) return "terminal";
	return "unknown";
}

async function scanMlflowRuns(root: string, signal?: AbortSignal): Promise<MlflowRunEvidence[]> {
	abortIfRequested(signal);
	let experiments;
	try { experiments = await readdir(root, { withFileTypes: true }); } catch { return []; }
	const runs: MlflowRunEvidence[] = [];
	for (const experiment of experiments.sort((a, b) => a.name.localeCompare(b.name))) {
		abortIfRequested(signal);
		if (!experiment.isDirectory()) continue;
		const experimentPath = join(root, experiment.name);
		let children;
		try { children = await readdir(experimentPath, { withFileTypes: true }); } catch { continue; }
		for (const child of children.sort((a, b) => a.name.localeCompare(b.name))) {
			abortIfRequested(signal);
			if (!child.isDirectory()) continue;
			const runPath = join(experimentPath, child.name);
			const metaText = await readText(join(runPath, "meta.yaml"));
			if (metaText === undefined) continue;
			const meta = parseSimpleYaml(metaText);
			const tagName = (await readText(join(runPath, "tags", "mlflow.runName")))?.trim();
			runs.push({
				runId: meta.run_id?.trim() || child.name,
				runName: tagName || meta.run_name?.trim() || null,
				status: meta.status?.trim() || null,
				startTimeMs: parseStoredTime(meta.start_time),
				runPath,
				experimentId: experiment.name,
			});
		}
	}
	return runs;
}

function exactNameRuns(runs: MlflowRunEvidence[], runName: string): MlflowRunEvidence[] {
	const wanted = normalizeRunName(runName);
	return runs.filter((run) => normalizeRunName(run.runName ?? "") === wanted);
}

async function recursiveFiles(root: string, signal?: AbortSignal): Promise<string[]> {
	const files: string[] = [];
	const pending = [root];
	while (pending.length) {
		abortIfRequested(signal);
		const directory = pending.pop()!;
		let entries;
		try { entries = await readdir(directory, { withFileTypes: true }); } catch { continue; }
		for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
			const target = join(directory, entry.name);
			if (entry.isDirectory()) pending.push(target);
			else if (entry.isFile()) files.push(target);
		}
	}
	return files.sort();
}

export function firstFiniteMlflowSample(text: string): string | null {
	for (const line of text.split(/\r?\n/)) {
		const parts = line.trim().split(/\s+/);
		if (parts.length !== 3) continue;
		const timestamp = Number(parts[0]);
		const value = Number(parts[1]);
		const step = Number(parts[2]);
		if (Number.isFinite(timestamp) && Number.isFinite(value) && Number.isFinite(step)) return line.trim();
	}
	return null;
}

async function freshMetricEvidence(run: MlflowRunEvidence, signal?: AbortSignal): Promise<{ file: string; sample: string } | null> {
	const metricsRoot = join(run.runPath, "metrics");
	for (const file of await recursiveFiles(metricsRoot, signal)) {
		const text = await readText(file);
		if (text === undefined) continue;
		const sample = firstFiniteMlflowSample(text);
		if (sample) return { file, sample };
	}
	return null;
}

export function detectTrainingProgress(text: string): string | null {
	for (const rawLine of text.split(/\r?\n/)) {
		const line = rawLine.trim();
		let match = line.match(/(?:^|\|)\s*total_timesteps\s*(?:\||[:=])\s*([0-9]+(?:\.[0-9]+)?)/i);
		if (!match) match = line.match(/(?:^|\|)\s*iterations?\s*(?:\||[:=])\s*([0-9]+(?:\.[0-9]+)?)/i);
		if (match && Number(match[1]) > 0) return line;
	}
	return null;
}

async function logBytesAfter(path: string, offset: number): Promise<string> {
	let data: Buffer;
	try { data = await readFile(path); } catch { return ""; }
	const start = Math.max(0, Math.min(offset, data.length));
	return data.subarray(start, Math.min(data.length, start + LOG_READ_LIMIT_BYTES)).toString("utf8");
}

async function logTail(path: string): Promise<string> {
	let data: Buffer;
	try { data = await readFile(path); } catch { return "(log unavailable)"; }
	return data.subarray(Math.max(0, data.length - LOG_TAIL_BYTES)).toString("utf8").trim() || "(log is empty)";
}

async function atomicWriteJson(path: string, value: unknown): Promise<void> {
	const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
	try {
		await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
		await rename(temporary, path);
	} finally {
		await rm(temporary, { force: true }).catch(() => undefined);
	}
}

async function readPidRecord(path: string): Promise<PidRecord | null> {
	try {
		const parsed = JSON.parse(await readFile(path, "utf8")) as Partial<PidRecord>;
		if (parsed.version !== PID_RECORD_VERSION || !Number.isInteger(parsed.pid) || !Number.isInteger(parsed.pgid) || typeof parsed.marker !== "string" || typeof parsed.startToken !== "string") return null;
		return parsed as PidRecord;
	} catch { return null; }
}

function hostPidAlive(pid: number): boolean {
	try { process.kill(pid, 0); return true; } catch { return false; }
}

async function acquireFilesystemLock(paths: ValidatedPaths, runName: string, dependencies: LaunchRunDependencies, signal?: AbortSignal): Promise<() => Promise<void>> {
	const token = randomUUID();
	while (true) {
		abortIfRequested(signal);
		try {
			await mkdir(paths.lockPath);
			const owner: LockRecord = { token, hostPid: process.pid, hostPlatform: process.platform, createdAtMs: dependencies.clock.now(), runName };
			await writeFile(join(paths.lockPath, "owner.json"), `${JSON.stringify(owner)}\n`, { encoding: "utf8", flag: "wx" });
			return async () => {
				try {
					const current = JSON.parse(await readFile(join(paths.lockPath, "owner.json"), "utf8")) as LockRecord;
					if (current.token === token) await rm(paths.lockPath, { recursive: true, force: true });
				} catch { /* Never remove a lock whose ownership cannot be proved. */ }
			};
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "EEXIST") {
				await rm(paths.lockPath, { recursive: true, force: true }).catch(() => undefined);
				throw new Error(`Unable to acquire launch serialization lock ${paths.lockPath}: ${errorText(error)}`);
			}
			let owner: LockRecord | null = null;
			try { owner = JSON.parse(await readFile(join(paths.lockPath, "owner.json"), "utf8")) as LockRecord; } catch { /* incomplete lock */ }
			let lockAgeMs = 0;
			try { lockAgeMs = dependencies.clock.now() - (await stat(paths.lockPath)).mtimeMs; } catch { /* retried below */ }
			const old = owner
				? dependencies.clock.now() - owner.createdAtMs > LOCK_STALE_AFTER_MS
				: lockAgeMs > LOCK_STALE_AFTER_MS;
			const sameHost = owner?.hostPlatform === process.platform;
			if ((!owner && old) || (owner && old && sameHost && !hostPidAlive(owner.hostPid))) {
				await rm(paths.lockPath, { recursive: true, force: true });
				continue;
			}
			await dependencies.clock.sleep(LOCK_POLL_INTERVAL_MS, signal);
		}
	}
}

function processKey(processEvidence: ProcessEvidence): string {
	return `${processEvidence.marker}\u0000${processEvidence.pgid}`;
}

function groupProcesses(processes: ProcessEvidence[]): ProcessEvidence[] {
	const result = new Map<string, ProcessEvidence>();
	for (const processEvidence of processes) {
		const key = processKey(processEvidence);
		const existing = result.get(key);
		if (!existing || processEvidence.pid === processEvidence.pgid || processEvidence.pid < existing.pid) result.set(key, processEvidence);
	}
	return [...result.values()].sort((a, b) => a.pgid - b.pgid);
}

async function waitForDeath(processEvidence: ProcessEvidence, markerPrefix: string, paths: ValidatedPaths, dependencies: LaunchRunDependencies, milliseconds: number): Promise<{ markerGone: boolean; groupGone: boolean; leaderGone: boolean }> {
	const deadline = dependencies.clock.now() + milliseconds;
	let current: ProcessEvidence[] = [];
	while (true) {
		current = await dependencies.inspectProcesses(markerPrefix, paths, undefined);
		const exact = current.filter((candidate) => candidate.marker === processEvidence.marker && candidate.pgid === processEvidence.pgid);
		const groupGone = !(await dependencies.processGroupExists(processEvidence.pgid, paths, undefined));
		const leaderGone = !exact.some((candidate) => candidate.pid === processEvidence.pid && candidate.startToken === processEvidence.startToken);
		if (!exact.length && groupGone && leaderGone) return { markerGone: true, groupGone: true, leaderGone: true };
		if (dependencies.clock.now() >= deadline) return { markerGone: exact.length === 0, groupGone, leaderGone };
		await dependencies.clock.sleep(Math.min(250, deadline - dependencies.clock.now()), undefined);
	}
}

async function stopVerifiedGroup(processEvidence: ProcessEvidence, markerPrefix: string, paths: ValidatedPaths, dependencies: LaunchRunDependencies): Promise<DuplicateKillEvidence> {
	const evidence: DuplicateKillEvidence = {
		pid: processEvidence.pid,
		pgid: processEvidence.pgid,
		marker: processEvidence.marker,
		startToken: processEvidence.startToken,
		signals: [],
		escalated: false,
		leaderIdentityGone: false,
		processGroupGone: false,
		markerGone: false,
		mlflowRunIds: [],
		mlflowReconciled: [],
	};
	for (const [signalName, grace] of [
		["SIGINT", DUPLICATE_INTERRUPT_GRACE_MS],
		["SIGTERM", DUPLICATE_TERMINATE_GRACE_MS],
		["SIGKILL", DUPLICATE_KILL_GRACE_MS],
	] as const) {
		const liveMembers = (await dependencies.inspectProcesses(markerPrefix, paths, undefined))
			.filter((candidate) => candidate.marker === processEvidence.marker && candidate.pgid === processEvidence.pgid);
		const verifiedMember = liveMembers.find((candidate) => candidate.pid === processEvidence.pid && candidate.startToken === processEvidence.startToken) ?? liveMembers[0];
		if (!verifiedMember) {
			const groupGone = !(await dependencies.processGroupExists(processEvidence.pgid, paths, undefined));
			if (groupGone) {
				evidence.leaderIdentityGone = true;
				evidence.processGroupGone = true;
				evidence.markerGone = true;
				return evidence;
			}
			throw new Error(`Process group ${processEvidence.pgid} remains but no verified marker-bearing member exists; refusing to signal a possibly reused group.`);
		}
		await dependencies.signalProcessGroup(verifiedMember, signalName, paths, undefined);
		evidence.signals.push(signalName);
		if (signalName !== "SIGINT") evidence.escalated = true;
		const death = await waitForDeath(processEvidence, markerPrefix, paths, dependencies, grace);
		evidence.leaderIdentityGone = death.leaderGone;
		evidence.processGroupGone = death.groupGone;
		evidence.markerGone = death.markerGone;
		if (death.leaderGone && death.groupGone && death.markerGone) return evidence;
	}
	throw new Error(`Unable to prove duplicate PID ${processEvidence.pid}/PGID ${processEvidence.pgid} and marker are gone after SIGINT/SIGTERM/SIGKILL; refusing to launch another copy.`);
}

function associatedActiveRunIds(active: MlflowRunEvidence[], record: PidRecord | null, groups: ProcessEvidence[]): string[] {
	if (!active.length) return [];
	if (record?.mlflowRunId && active.some((run) => run.runId.toLocaleLowerCase() === record.mlflowRunId!.toLocaleLowerCase())) return [record.mlflowRunId];
	if (groups.length !== 1) throw new Error(`Found ${active.length} active same-name MLflow run(s) and ${groups.length} verified process groups; ownership is ambiguous.`);
	const processLaunchMs = record?.launchedAtMs ?? 0;
	const candidates = active.filter((run) => run.startTimeMs === null || run.startTimeMs >= processLaunchMs - 60_000);
	if (candidates.length === 1 && active.length === 1) return [candidates[0].runId];
	throw new Error(`Active same-name MLflow identity is unresolved (${active.map((run) => run.runId).join(", ")}); refusing to kill or start a second copy.`);
}

async function reconcileIfActive(runId: string, paths: ValidatedPaths, dependencies: LaunchRunDependencies): Promise<boolean> {
	const before = (await scanMlflowRuns(paths.mlrunsPath)).find((run) => run.runId.toLocaleLowerCase() === runId.toLocaleLowerCase());
	if (!before || normalizedMlflowStatus(before.status) !== "active") return false;
	await dependencies.reconcileMlflowRun(before.runId, paths, undefined);
	const after = (await scanMlflowRuns(paths.mlrunsPath)).find((run) => run.runId.toLocaleLowerCase() === runId.toLocaleLowerCase());
	if (after && normalizedMlflowStatus(after.status) === "active") throw new Error(`MLflow run ${runId} remains active after lifecycle reconciliation.`);
	return true;
}

async function cleanupLaunchedProcess(markerPrefix: string, processEvidence: ProcessEvidence | null, paths: ValidatedPaths, dependencies: LaunchRunDependencies, runName: string, baselineIds: Set<string>): Promise<string[]> {
	const diagnostics: string[] = [];
	try {
		let candidate = processEvidence;
		if (!candidate) {
			const found = groupProcesses(await dependencies.inspectProcesses(markerPrefix, paths, undefined));
			candidate = found[0] ?? null;
		}
		if (candidate) {
			const stopped = await stopVerifiedGroup(candidate, markerPrefix, paths, dependencies);
			diagnostics.push(`cleanup signals=${stopped.signals.join("/")} pid=${stopped.pid} markerGone=${stopped.markerGone} groupGone=${stopped.processGroupGone}`);
		}
	} catch (error) {
		diagnostics.push(`PROCESS CLEANUP ERROR: ${errorText(error)}`);
	}
	try {
		const fresh = exactNameRuns(await scanMlflowRuns(paths.mlrunsPath), runName)
			.filter((run) => !baselineIds.has(run.runId) && normalizedMlflowStatus(run.status) === "active");
		for (const run of fresh) {
			await dependencies.reconcileMlflowRun(run.runId, paths, undefined);
			diagnostics.push(`reconciled MLflow run ${run.runId}`);
		}
	} catch (error) {
		diagnostics.push(`MLFLOW CLEANUP ERROR: ${errorText(error)}`);
	}
	try {
		const record = await readPidRecord(paths.pidRecordPath);
		if (!record || record.marker.startsWith(markerPrefix)) await rm(paths.pidRecordPath, { force: true });
	} catch (error) {
		diagnostics.push(`PID RECORD CLEANUP ERROR: ${errorText(error)}`);
	}
	return diagnostics;
}

function selectGpu(devices: GpuEvidence[], requiredMiB: number): GpuEvidence {
	const selected = devices.find((device) => device.freeMiB >= requiredMiB);
	if (selected) return selected;
	const observed = devices.map((device) => `GPU ${device.index} ${device.name}: ${device.freeMiB}/${device.totalMiB} MiB free`).join("; ");
	throw new Error(`Insufficient free VRAM: required ${requiredMiB} MiB on one GPU; observed ${observed}. No Python process was spawned.`);
}

function markerPrefixFor(paths: ValidatedPaths, runName: string): string {
	const identity = createHash("sha256")
		.update(`${paths.repositoryWsl}\u0000${normalizeRunName(runName)}`)
		.digest("hex")
		.slice(0, 32);
	return `pi-launch-run-v1:${identity}:`;
}

async function updateRecordWithMlflow(paths: ValidatedPaths, marker: string, run: MlflowRunEvidence): Promise<void> {
	const record = await readPidRecord(paths.pidRecordPath);
	if (!record || record.marker !== marker) return;
	await atomicWriteJson(paths.pidRecordPath, { ...record, mlflowRunId: run.runId, mlflowRunPath: run.runPath });
}

async function baselineLogOffset(path: string): Promise<number> {
	try { return (await stat(path)).size; } catch { return 0; }
}

/** Complete launch lifecycle core. The registered tool binds production dependencies. */
export async function launchRun(
	input: LaunchRunInput,
	cwd: string,
	signal?: AbortSignal,
	onUpdate?: (result: { content: Array<{ type: "text"; text: string }>; details?: unknown }) => void,
	dependencies: LaunchRunDependencies = PRODUCTION_LAUNCH_RUN_DEPENDENCIES,
): Promise<LaunchRunResult> {
	abortIfRequested(signal);
	const requiredMiB = input.requiredFreeVramMiB ?? DEFAULT_REQUIRED_FREE_VRAM_MIB;
	const startupTimeoutSeconds = input.startupTimeoutSeconds ?? DEFAULT_STARTUP_TIMEOUT_SECONDS;
	if (!Number.isInteger(requiredMiB) || requiredMiB < 1) throw new Error("requiredFreeVramMiB must be a positive integer.");
	if (!Number.isInteger(startupTimeoutSeconds) || startupTimeoutSeconds < 1 || startupTimeoutSeconds > 3600) throw new Error("startupTimeoutSeconds must be an integer from 1 to 3600.");
	const paths = await validatePaths(input, cwd, dependencies, signal);
	const runName = validateRunName(input.runName);
	const markerPrefix = markerPrefixFor(paths, runName);

	return withFileMutationQueue(paths.pidRecordPath, async () => {
		const releaseLock = await acquireFilesystemLock(paths, runName, dependencies, signal);
		let launchedProcess: ProcessEvidence | null = null;
		let launchAttempted = false;
		let marker = "";
		let baselineIds = new Set<string>();
		try {
			onUpdate?.({ content: [{ type: "text", text: `Checking duplicate ownership for ${runName}...` }] });
			abortIfRequested(signal);
			const record = await readPidRecord(paths.pidRecordPath);
			const discoveredProcesses = await dependencies.inspectProcesses(markerPrefix, paths, signal);
			const discovered = groupProcesses(discoveredProcesses);
			const verifiedRecord = record && discoveredProcesses.some((candidate) =>
				candidate.marker === record.marker
				&& candidate.pid === record.pid
				&& candidate.pgid === record.pgid
				&& candidate.startToken === record.startToken
			) ? record : null;
			const currentRuns = exactNameRuns(await scanMlflowRuns(paths.mlrunsPath, signal), runName);
			const activeRuns = currentRuns.filter((run) => normalizedMlflowStatus(run.status) === "active");
			const associatedIds = associatedActiveRunIds(activeRuns, verifiedRecord, discovered);
			if (!discovered.length && activeRuns.length) {
				throw new Error(`Found active same-name MLflow run(s) ${activeRuns.map((run) => run.runId).join(", ")} but no verified launch marker/process owner; refusing to create an orphaned duplicate.`);
			}

			const duplicates: DuplicateKillEvidence[] = [];
			for (const duplicate of discovered) {
				abortIfRequested(signal);
				const stopped = await stopVerifiedGroup(duplicate, markerPrefix, paths, dependencies);
				stopped.mlflowRunIds = [...associatedIds];
				for (const runId of associatedIds) {
					if (await reconcileIfActive(runId, paths, dependencies)) stopped.mlflowReconciled.push(runId);
				}
				duplicates.push(stopped);
			}
			const remainingProcesses = await dependencies.inspectProcesses(markerPrefix, paths, signal);
			if (remainingProcesses.length) throw new Error("A same-name process marker remains after duplicate cleanup; refusing launch.");
			const remainingActive = exactNameRuns(await scanMlflowRuns(paths.mlrunsPath, signal), runName)
				.filter((run) => normalizedMlflowStatus(run.status) === "active");
			if (remainingActive.length) throw new Error(`Same-name MLflow run(s) remain active after cleanup: ${remainingActive.map((run) => run.runId).join(", ")}.`);

			abortIfRequested(signal);
			onUpdate?.({ content: [{ type: "text", text: `Running WSL VRAM preflight (requires ${requiredMiB} MiB)...` }] });
			const allGpus = await dependencies.queryGpus(signal);
			const gpu = selectGpu(allGpus, requiredMiB);
			abortIfRequested(signal);

			const logOffset = await baselineLogOffset(paths.logPath);
			const baselineRuns = exactNameRuns(await scanMlflowRuns(paths.mlrunsPath, signal), runName);
			baselineIds = new Set(baselineRuns.map((run) => run.runId));
			const launchTimestampMs = dependencies.clock.now();
			const launchTimestamp = new Date(launchTimestampMs).toISOString();
			marker = `${markerPrefix}${randomUUID()}`;
			await appendFile(paths.logPath, `\n[launch_run ${launchTimestamp}] script=${paths.scriptRelative} run=${runName} gpu=${gpu.index}\n`, "utf8");
			onUpdate?.({ content: [{ type: "text", text: `Launching detached WSL process on GPU ${gpu.index} (${gpu.freeMiB} MiB free)...` }] });
			launchAttempted = true;
			const returnedPid = await dependencies.launchDetached({
				paths,
				marker,
				runNameBase64: Buffer.from(runName, "utf8").toString("base64"),
				gpuIndex: gpu.index,
			}, signal);
			abortIfRequested(signal);

			const candidates = (await dependencies.inspectProcesses(markerPrefix, paths, signal))
				.filter((candidate) => candidate.marker === marker && candidate.pid === returnedPid && candidate.cmdline.includes(paths.scriptWsl));
			if (candidates.length !== 1) throw new Error(`Detached launcher returned Linux PID ${returnedPid}, but exactly one expected marked process could not be verified.`);
			launchedProcess = candidates[0];
			if (launchedProcess.pid !== launchedProcess.pgid || launchedProcess.pid !== launchedProcess.sid) {
				throw new Error(`Launched PID ${launchedProcess.pid} is not the detached session/process-group leader (PGID ${launchedProcess.pgid}, SID ${launchedProcess.sid}).`);
			}
			const recordToWrite: PidRecord = {
				version: PID_RECORD_VERSION,
				pid: launchedProcess.pid,
				pgid: launchedProcess.pgid,
				startToken: launchedProcess.startToken,
				marker,
				markerPrefix,
				runName,
				normalizedRunName: normalizeRunName(runName),
				scriptPath: paths.scriptPath,
				repositoryRoot: paths.repositoryRoot,
				repositoryWsl: paths.repositoryWsl,
				scriptWsl: paths.scriptWsl,
				launchedAtMs: launchTimestampMs,
				launchedAt: launchTimestamp,
				mlflowRunId: null,
				mlflowRunPath: null,
				gpu,
				requiredFreeVramMiB: requiredMiB,
			};
			await atomicWriteJson(paths.pidRecordPath, recordToWrite);

			const deadline = launchTimestampMs + startupTimeoutSeconds * 1000;
			let startupEvidence: LaunchRunDetails["startupEvidence"] | null = null;
			let freshMlflowRun: MlflowRunEvidence | null = null;
			while (dependencies.clock.now() <= deadline) {
				abortIfRequested(signal);
				const alive = (await dependencies.inspectProcesses(markerPrefix, paths, signal))
					.some((candidate) => candidate.marker === marker && candidate.pid === launchedProcess!.pid && candidate.startToken === launchedProcess!.startToken && candidate.pgid === launchedProcess!.pgid);
				if (!alive) throw new Error(`Python PID ${launchedProcess.pid} exited before startup progression was confirmed.`);

				const newLogText = await logBytesAfter(paths.logPath, logOffset);
				const progress = detectTrainingProgress(newLogText);
				const freshRuns = exactNameRuns(await scanMlflowRuns(paths.mlrunsPath, signal), runName)
					.filter((run) => !baselineIds.has(run.runId));
				if (freshRuns.length > 1) throw new Error(`Multiple fresh exact-name MLflow runs appeared (${freshRuns.map((run) => run.runId).join(", ")}); startup identity is ambiguous.`);
				freshMlflowRun = freshRuns[0] ?? null;
				if (freshMlflowRun) await updateRecordWithMlflow(paths, marker, freshMlflowRun);
				if (progress) {
					startupEvidence = { type: "log_progress", value: progress, logOffset };
					break;
				}
				if (freshMlflowRun) {
					const metric = await freshMetricEvidence(freshMlflowRun, signal);
					if (metric) {
						startupEvidence = { type: "mlflow_metric", value: `${relative(freshMlflowRun.runPath, metric.file)}: ${metric.sample}`, logOffset };
						break;
					}
				}
				if (dependencies.clock.now() >= deadline) break;
				onUpdate?.({ content: [{ type: "text", text: `PID ${launchedProcess.pid} is alive; waiting for positive PPO progress or a fresh MLflow metric...` }] });
				await dependencies.clock.sleep(Math.min(STARTUP_POLL_INTERVAL_MS, deadline - dependencies.clock.now()), signal);
			}
			if (!startupEvidence) throw new Error(`Startup timed out after ${startupTimeoutSeconds}s without positive PPO progress or a finite metric in a fresh exact-name MLflow run.`);
			const finalAlive = (await dependencies.inspectProcesses(markerPrefix, paths, signal))
				.some((candidate) => candidate.marker === marker && candidate.pid === launchedProcess!.pid && candidate.startToken === launchedProcess!.startToken);
			if (!finalAlive) throw new Error(`Python PID ${launchedProcess.pid} exited while startup evidence was being finalized.`);

			const launchArgv = [paths.pythonWsl, "-u", paths.scriptWsl];
			const details: LaunchRunDetails = {
				outcome: "started",
				pid: launchedProcess.pid,
				pgid: launchedProcess.pgid,
				runName,
				scriptPath: paths.scriptPath,
				launchArgv,
				launchCommand: `CUDA_VISIBLE_DEVICES=${gpu.index} nohup setsid ${launchArgv.map(quoteForDisplay).join(" ")} >> ${quoteForDisplay(paths.logWsl)} 2>&1 < /dev/null`,
				launchTimestampMs,
				launchTimestamp,
				logPath: paths.logPath,
				pidRecordPath: paths.pidRecordPath,
				gpu,
				requiredFreeVramMiB: requiredMiB,
				allGpus,
				duplicates,
				startupEvidence,
				mlflowRunId: freshMlflowRun?.runId ?? null,
				mlflowRunPath: freshMlflowRun?.runPath ?? null,
				marker,
				startToken: launchedProcess.startToken,
			};
			return {
				content: [{ type: "text", text: `Started ${runName} successfully. WSL/Linux PID (and process group) ${details.pid}; startup confirmed by ${startupEvidence.type}. Log: ${paths.logPath}` }],
				details,
			};
		} catch (error) {
			const cleanup = launchAttempted
				? await cleanupLaunchedProcess(marker || markerPrefix, launchedProcess, paths, dependencies, runName, baselineIds)
				: [];
			const tail = await logTail(paths.logPath);
			const message = `${errorText(error)}${cleanup.length ? ` Cleanup: ${cleanup.join("; ")}.` : ""} Log: ${paths.logPath}\nLog tail:\n${tail}`;
			if (error instanceof Error && error.name === "AbortError") throw cancellationError(message);
			throw new Error(message);
		} finally {
			await releaseLock();
		}
	});
}

export default function launchRunExtension(pi: ExtensionAPI): void {
	pi.registerTool({
		name: "launch_run",
		label: "Launch Training Run",
		description: "Safely launch an already-created root-level Python training script in detached WSL. Provide only its path and exact filesystem-safe MLflow run name; launch_run validates paths, serializes same-name calls, replaces verified duplicates and reconciles MLflow, checks VRAM, writes deterministic log/PID metadata, binds the selected GPU, and returns the WSL PID only after real PPO/log or fresh-MLflow metric progression.",
		promptSnippet: "Launch a prepared training run safely with duplicate cleanup, VRAM preflight, durable logs/PID metadata, and objective startup confirmation",
		promptGuidelines: [
			"Use launch_run instead of constructing nohup/WSL shell commands whenever a root-level training-script copy is ready to run.",
			"Before calling launch_run, ensure the script passes its exact filesystem-safe runName to mlflow.start_run; call launch_run with scriptPath and that exact runName, and report its returned WSL PID/log paths.",
		],
		parameters: launchRunSchema,
		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			return launchRun(params, ctx.cwd, signal, onUpdate, PRODUCTION_LAUNCH_RUN_DEPENDENCIES);
		},
	});
}
