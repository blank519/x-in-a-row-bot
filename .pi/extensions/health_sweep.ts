import { spawn } from "node:child_process";
import { open, readdir, readFile, stat } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";

/** Fixed production budget shared by every target in one invocation. */
export const HEALTH_SWEEP_TIMEOUT_MS = 600_000;
export const HEALTH_SWEEP_POLL_INTERVAL_MS = 5_000;
export const MAX_HEALTH_SWEEP_RUNS = 16;
export const MAX_LOG_READ_BYTES = 1_048_576;
export const MAX_EVIDENCE_CHARS = 800;
const LAUNCH_TIME_SLOP_MS = 120_000;

const nonBlankTrimmed = "^\\S(?:.*\\S)?$";
export const healthSweepRunSchema = Type.Object({
	runName: Type.Optional(Type.String({
		minLength: 1,
		maxLength: 128,
		pattern: nonBlankTrimmed,
		description: "Exact filesystem-safe launch/MLflow run name (trimmed; optional when runId or pid is supplied).",
	})),
	runId: Type.Optional(Type.String({
		minLength: 1,
		maxLength: 256,
		pattern: nonBlankTrimmed,
		description: "Full MLflow run ID or unique case-insensitive prefix (trimmed; optional when runName or pid is supplied).",
	})),
	pid: Type.Optional(Type.Integer({
		minimum: 2,
		description: "Positive WSL/Linux training PID greater than 1 (optional when runName or runId is supplied).",
	})),
}, {
	additionalProperties: false,
	anyOf: [{ required: ["runName"] }, { required: ["runId"] }, { required: ["pid"] }],
	description: "One run identity. Multiple supplied identifiers must all describe the same launch.",
});

export const healthSweepSchema = Type.Object({
	runs: Type.Array(healthSweepRunSchema, {
		minItems: 1,
		maxItems: MAX_HEALTH_SWEEP_RUNS,
		description: "Non-empty batch of launched runs to check together under one fixed 10-minute deadline.",
	}),
}, { additionalProperties: false });

export type HealthSweepRunInput = Static<typeof healthSweepRunSchema>;
export type HealthSweepInput = Static<typeof healthSweepSchema>;
export type HealthReasonCode =
	| "process_unresolved"
	| "process_dead"
	| "process_identity_mismatch"
	| "log_missing"
	| "training_banner_missing"
	| "ppo_progress_missing"
	| "fatal_log_error"
	| "mlflow_run_missing"
	| "mlflow_terminal_failure"
	| "identity_ambiguous"
	| "identity_mismatch";

export interface HealthReason {
	code: HealthReasonCode;
	message: string;
	evidence?: string;
}

export interface ProcessObservation {
	pid: number;
	exists: boolean;
	alive: boolean;
	state: string | null;
	pgid: number | null;
	sid: number | null;
	startToken: string | null;
	marker: string | null;
	error?: string;
}

export interface LaunchRecord {
	kind: "current" | "legacy";
	path: string;
	fileRunName: string;
	runName: string;
	pid: number;
	pgid: number | null;
	marker: string | null;
	startToken: string | null;
	launchedAtMs: number | null;
	launchedAt: string | null;
	mlflowRunId: string | null;
	mlflowRunPath: string | null;
}

export interface MlflowRunObservation {
	experimentId: string;
	runId: string;
	runName: string | null;
	status: string | null;
	startTimeMs: number | null;
	runPath: string;
	metaPath: string;
}

export interface RepositoryHealthIndex {
	logsPath: string;
	mlrunsPath: string;
	launchRecords: LaunchRecord[];
	recordErrors: Array<{ path: string; fileRunName: string; message: string }>;
	mlflowRuns: MlflowRunObservation[];
}

export interface LogObservation {
	path: string | null;
	exists: boolean;
	association: "current_launch" | "legacy_unverified" | "manual_unverified" | "none";
	banner: string | null;
	progress: string | null;
	fatal: string | null;
	boundary: string | null;
	bytesRead: number;
	truncated: boolean;
	associationError?: string;
}

export interface HealthSweepClock {
	now(): number;
	sleep(milliseconds: number, signal?: AbortSignal): Promise<void>;
}

/** Filesystem/process seams are deliberately high-level so evaluators can use virtual time and harmless fixtures. */
export interface HealthSweepDependencies {
	clock: HealthSweepClock;
	buildIndex(cwd: string, signal?: AbortSignal): Promise<RepositoryHealthIndex>;
	inspectProcess(pid: number, signal?: AbortSignal): Promise<ProcessObservation>;
	inspectLog(cwd: string, runName: string | null, record: LaunchRecord | null, signal?: AbortSignal): Promise<LogObservation>;
}

export interface RunChecks {
	processAlive: boolean;
	processIdentityVerified: boolean;
	trainingBanner: boolean;
	ppoProgress: boolean;
	mlflowCreated: boolean;
	mlflowIdentityConsistent: boolean;
}

export interface UnhealthyRun {
	inputIndex: number;
	supplied: HealthSweepRunInput;
	resolved: { runName: string | null; runId: string | null; pid: number | null };
	reasons: HealthReason[];
	checks: RunChecks;
	evidence: {
		pidRecordPath: string | null;
		logPath: string | null;
		mlflowRunPath: string | null;
		process: ProcessObservation | null;
		log: LogObservation | null;
	};
	observedAtMs: number;
	observedAt: string;
}

export interface HealthSweepDetails {
	unhealthyRuns: UnhealthyRun[];
	observations: Array<UnhealthyRun & { healthy: boolean }>;
	startedAtMs: number;
	startedAt: string;
	deadlineMs: number;
	deadline: string;
	completedAtMs: number;
	completedAt: string;
	waitedMs: number;
	pollCount: number;
	timedOut: boolean;
}

export interface HealthSweepResult {
	content: Array<{ type: "text"; text: string }>;
	details: HealthSweepDetails;
}

interface TargetObservation extends UnhealthyRun {
	healthy: boolean;
	pending: string[];
	conclusive: boolean;
}

function iso(ms: number): string {
	return new Date(ms).toISOString();
}

function abortError(): Error {
	const error = new Error("Health sweep was cancelled.");
	error.name = "AbortError";
	return error;
}

function abortIfRequested(signal?: AbortSignal): void {
	if (signal?.aborted) throw abortError();
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
			reject(abortError());
		}
		signal?.addEventListener("abort", cancelled, { once: true });
	});
}

export const SYSTEM_HEALTH_SWEEP_CLOCK: HealthSweepClock = {
	now: () => Date.now(),
	sleep: systemSleep,
};

function bounded(value: string): string {
	const normalized = value.replace(/\0/g, "").trim();
	return normalized.length <= MAX_EVIDENCE_CHARS ? normalized : `${normalized.slice(0, MAX_EVIDENCE_CHARS - 1)}…`;
}

function equalFold(left: string | null | undefined, right: string | null | undefined): boolean {
	return (left ?? "").toLocaleLowerCase() === (right ?? "").toLocaleLowerCase();
}

function hasIdPrefix(runId: string, prefix: string): boolean {
	return runId.toLocaleLowerCase().startsWith(prefix.toLocaleLowerCase());
}

function uniqueByPath<T extends { path: string }>(items: T[]): T[] {
	return [...new Map(items.map((item) => [item.path, item])).values()];
}

/** Small YAML reader for the scalar metadata fields written by MLflow. */
export function parseSimpleYaml(text: string): Record<string, string> {
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

export function detectTrainingBanner(text: string): string | null {
	for (const rawLine of text.split(/\r?\n/)) {
		const line = rawLine.trim();
		if (/^\[Train\].*\bdevice\s*=\s*\S+/i.test(line) || /^Using\s+\S+\s+device\s*$/i.test(line)) return bounded(line);
	}
	return null;
}

/** Planned `total_timesteps=...` text is intentionally excluded; only SB3 table records count. */
export function detectPpoProgress(text: string): string | null {
	for (const rawLine of text.split(/\r?\n/)) {
		const line = rawLine.trim();
		const match = line.match(/^\|\s*(?:iterations?|total_timesteps)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*$/i);
		if (match && Number.isFinite(Number(match[1])) && Number(match[1]) > 0) return bounded(line);
	}
	return null;
}

export function detectFatalLogError(text: string): string | null {
	for (const rawLine of text.split(/\r?\n/)) {
		const line = rawLine.trim();
		if (/Traceback \(most recent call last\):/i.test(line)
			|| /CUDA\s+(?:error:\s*)?out of memory/i.test(line)
			|| /(?:uncaught|unhandled)\s+(?:exception|error)/i.test(line)) return bounded(line);
	}
	return null;
}

export function normalizeMlflowStatus(status: string | null): "active" | "successful" | "failed" | "unknown" {
	if (status === null) return "unknown";
	const value = status.trim().toLocaleLowerCase().replace(/[\s_-]+/g, "");
	if (["1", "2", "running", "active", "scheduled", "pending"].includes(value)) return "active";
	if (["3", "finished", "completed", "succeeded", "success"].includes(value)) return "successful";
	if (["4", "5", "failed", "failure", "error", "killed", "cancelled", "canceled", "terminated"].includes(value)) return "failed";
	return "unknown";
}

function validateInput(input: HealthSweepInput): HealthSweepRunInput[] {
	if (!input || typeof input !== "object" || !Array.isArray(input.runs) || input.runs.length < 1 || input.runs.length > MAX_HEALTH_SWEEP_RUNS) {
		throw new Error(`runs must be a non-empty array with at most ${MAX_HEALTH_SWEEP_RUNS} descriptors.`);
	}
	return input.runs.map((raw, index) => {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`runs[${index}] must be an object.`);
		const keys = Object.keys(raw as object);
		if (keys.some((key) => !["runName", "runId", "pid"].includes(key))) throw new Error(`runs[${index}] contains an unknown field.`);
		const item = raw as HealthSweepRunInput;
		if (item.runName === undefined && item.runId === undefined && item.pid === undefined) throw new Error(`runs[${index}] must supply runName, runId, or pid.`);
		if (item.runName !== undefined) {
			if (typeof item.runName !== "string" || item.runName !== item.runName.trim() || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(item.runName) || item.runName.endsWith(".")) {
				throw new Error(`runs[${index}].runName must be a trimmed filesystem-safe launch name.`);
			}
		}
		if (item.runId !== undefined && (typeof item.runId !== "string" || item.runId !== item.runId.trim() || !item.runId)) {
			throw new Error(`runs[${index}].runId must be a non-empty trimmed string.`);
		}
		if (item.pid !== undefined && (!Number.isInteger(item.pid) || item.pid <= 1)) throw new Error(`runs[${index}].pid must be an integer greater than 1.`);
		return { ...(item.runName === undefined ? {} : { runName: item.runName }), ...(item.runId === undefined ? {} : { runId: item.runId }), ...(item.pid === undefined ? {} : { pid: item.pid }) };
	});
}

async function textOrUndefined(path: string, maxBytes = 256 * 1024): Promise<string | undefined> {
	try {
		const info = await stat(path);
		if (!info.isFile() || info.size > maxBytes) return undefined;
		return await readFile(path, "utf8");
	} catch { return undefined; }
}

function parseFiniteTime(value: unknown): number | null {
	if (typeof value === "number") return Number.isFinite(value) ? value : null;
	if (typeof value !== "string" || !value.trim()) return null;
	const numeric = Number(value);
	if (Number.isFinite(numeric)) return numeric;
	const parsed = Date.parse(value);
	return Number.isFinite(parsed) ? parsed : null;
}

function parseCurrentRecord(path: string, fileRunName: string, text: string): LaunchRecord {
	let value: Record<string, unknown>;
	try { value = JSON.parse(text) as Record<string, unknown>; }
	catch (error) { throw new Error(`malformed JSON: ${error instanceof Error ? error.message : String(error)}`); }
	const runName = typeof value.runName === "string" ? value.runName : "";
	const pid = Number(value.pid);
	const pgid = Number(value.pgid);
	const marker = typeof value.marker === "string" ? value.marker : "";
	const startToken = typeof value.startToken === "string" ? value.startToken : "";
	if (!runName || !Number.isInteger(pid) || pid <= 1 || !Number.isInteger(pgid) || pgid <= 1 || !marker || !startToken) throw new Error("missing valid runName/PID/PGID/marker/startToken");
	if (runName !== fileRunName) throw new Error(`record runName ${JSON.stringify(runName)} disagrees with filename ${JSON.stringify(fileRunName)}`);
	const launchedAtMs = parseFiniteTime(value.launchedAtMs ?? value.launchedAt);
	return {
		kind: "current", path, fileRunName, runName, pid, pgid, marker, startToken,
		launchedAtMs,
		launchedAt: typeof value.launchedAt === "string" ? value.launchedAt : launchedAtMs === null ? null : iso(launchedAtMs),
		mlflowRunId: typeof value.mlflowRunId === "string" && value.mlflowRunId.trim() ? value.mlflowRunId.trim() : null,
		mlflowRunPath: typeof value.mlflowRunPath === "string" && value.mlflowRunPath.trim() ? value.mlflowRunPath : null,
	};
}

/** Read-only repository index used by every target in a poll. */
export async function buildRepositoryHealthIndex(cwd: string, signal?: AbortSignal): Promise<RepositoryHealthIndex> {
	abortIfRequested(signal);
	const logsPath = resolve(cwd, "logs");
	const mlrunsPath = resolve(cwd, "mlruns");
	const launchRecords: LaunchRecord[] = [];
	const recordErrors: RepositoryHealthIndex["recordErrors"] = [];
	let logEntries;
	try { logEntries = await readdir(logsPath, { withFileTypes: true }); } catch { logEntries = []; }
	const currentNames = new Set<string>();
	for (const entry of logEntries.sort((a, b) => a.name.localeCompare(b.name))) {
		abortIfRequested(signal);
		if (!entry.isFile() || !entry.name.endsWith(".pid.json")) continue;
		const fileRunName = entry.name.slice(0, -9);
		currentNames.add(fileRunName.toLocaleLowerCase());
		const path = join(logsPath, entry.name);
		const text = await textOrUndefined(path);
		if (text === undefined) { recordErrors.push({ path, fileRunName, message: "PID record is unreadable or oversized." }); continue; }
		try { launchRecords.push(parseCurrentRecord(path, fileRunName, text)); }
		catch (error) { recordErrors.push({ path, fileRunName, message: error instanceof Error ? error.message : String(error) }); }
	}
	for (const entry of logEntries.sort((a, b) => a.name.localeCompare(b.name))) {
		abortIfRequested(signal);
		if (!entry.isFile() || !entry.name.endsWith(".pid") || entry.name.endsWith(".pid.json")) continue;
		const fileRunName = entry.name.slice(0, -4);
		if (currentNames.has(fileRunName.toLocaleLowerCase())) continue;
		const path = join(logsPath, entry.name);
		const text = await textOrUndefined(path, 1024);
		const pid = Number(text?.trim());
		if (!Number.isInteger(pid) || pid <= 1) { recordErrors.push({ path, fileRunName, message: "Legacy PID file does not contain one Linux PID greater than 1." }); continue; }
		launchRecords.push({ kind: "legacy", path, fileRunName, runName: fileRunName, pid, pgid: null, marker: null, startToken: null, launchedAtMs: null, launchedAt: null, mlflowRunId: null, mlflowRunPath: null });
	}

	const mlflowRuns: MlflowRunObservation[] = [];
	let experiments;
	try { experiments = await readdir(mlrunsPath, { withFileTypes: true }); } catch { experiments = []; }
	for (const experiment of experiments.sort((a, b) => a.name.localeCompare(b.name))) {
		abortIfRequested(signal);
		if (!experiment.isDirectory()) continue;
		const experimentPath = join(mlrunsPath, experiment.name);
		let children;
		try { children = await readdir(experimentPath, { withFileTypes: true }); } catch { continue; }
		for (const child of children.sort((a, b) => a.name.localeCompare(b.name))) {
			abortIfRequested(signal);
			if (!child.isDirectory()) continue;
			const runPath = join(experimentPath, child.name);
			const metaPath = join(runPath, "meta.yaml");
			const metaText = await textOrUndefined(metaPath);
			if (metaText === undefined) continue;
			const meta = parseSimpleYaml(metaText);
			const tagName = (await textOrUndefined(join(runPath, "tags", "mlflow.runName"), 16 * 1024))?.trim();
			mlflowRuns.push({
				experimentId: experiment.name,
				runId: meta.run_id?.trim() || child.name,
				runName: tagName || meta.run_name?.trim() || null,
				status: meta.status?.trim() || null,
				startTimeMs: parseFiniteTime(meta.start_time),
				runPath,
				metaPath,
			});
		}
	}
	return { logsPath, mlrunsPath, launchRecords, recordErrors, mlflowRuns };
}

function parseProcStat(text: string): { state: string; pgid: number; sid: number; startToken: string } | null {
	const close = text.lastIndexOf(")");
	if (close < 0) return null;
	const fields = text.slice(close + 2).trim().split(/\s+/);
	if (fields.length < 20) return null;
	const pgid = Number(fields[2]);
	const sid = Number(fields[3]);
	if (!Number.isInteger(pgid) || !Number.isInteger(sid)) return null;
	return { state: fields[0], pgid, sid, startToken: fields[19] };
}

async function inspectPosixProcess(pid: number): Promise<ProcessObservation> {
	try {
		const statText = await readFile(`/proc/${pid}/stat`, "utf8");
		const parsed = parseProcStat(statText);
		if (!parsed) return { pid, exists: true, alive: false, state: null, pgid: null, sid: null, startToken: null, marker: null, error: "Malformed /proc stat record." };
		let marker: string | null = null;
		try {
			const environment = await readFile(`/proc/${pid}/environ`);
			for (const item of environment.toString("utf8").split("\0")) if (item.startsWith("PI_LAUNCH_RUN_MARKER=")) marker = item.slice("PI_LAUNCH_RUN_MARKER=".length);
		} catch { /* A raw/legacy PID can still provide lower-confidence liveness. */ }
		return { pid, exists: true, alive: parsed.state !== "Z" && parsed.state !== "X", state: parsed.state, pgid: parsed.pgid, sid: parsed.sid, startToken: parsed.startToken, marker };
	} catch (error) {
		const code = (error as NodeJS.ErrnoException).code;
		if (code === "ENOENT" || code === "ESRCH") return { pid, exists: false, alive: false, state: null, pgid: null, sid: null, startToken: null, marker: null };
		return { pid, exists: false, alive: false, state: null, pgid: null, sid: null, startToken: null, marker: null, error: error instanceof Error ? error.message : String(error) };
	}
}

const WSL_PROCESS_INSPECTOR = String.raw`
import json, os, sys
pid = int(sys.argv[1])
out = {'pid': pid, 'exists': False, 'alive': False, 'state': None, 'pgid': None, 'sid': None, 'startToken': None, 'marker': None}
try:
    raw = open(f'/proc/{pid}/stat', encoding='utf-8').read()
    close = raw.rfind(')')
    fields = raw[close + 2:].split()
    out.update({'exists': True, 'state': fields[0], 'pgid': int(fields[2]), 'sid': int(fields[3]), 'startToken': fields[19]})
    out['alive'] = fields[0] not in {'Z', 'X'}
    try:
        for item in open(f'/proc/{pid}/environ', 'rb').read().split(b'\0'):
            if item.startswith(b'PI_LAUNCH_RUN_MARKER='):
                out['marker'] = item.split(b'=', 1)[1].decode('utf-8', 'replace')
                break
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        pass
except (FileNotFoundError, ProcessLookupError):
    pass
except Exception as exc:
    out['error'] = str(exc)
print(json.dumps(out))
`;

async function inspectWslProcess(pid: number, signal?: AbortSignal): Promise<ProcessObservation> {
	abortIfRequested(signal);
	return new Promise<ProcessObservation>((resolvePromise, reject) => {
		const child = spawn("wsl.exe", ["--exec", "python3", "-c", WSL_PROCESS_INSPECTOR, String(pid)], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		let settled = false;
		const timer = setTimeout(() => finish(new Error("WSL process inspection timed out.")), 30_000);
		function cleanup(): void { clearTimeout(timer); signal?.removeEventListener("abort", cancelled); }
		function finish(error?: Error, value?: ProcessObservation): void {
			if (settled) return;
			settled = true;
			cleanup();
			if (error) reject(error); else resolvePromise(value!);
		}
		function cancelled(): void { child.kill(); finish(abortError()); }
		signal?.addEventListener("abort", cancelled, { once: true });
		child.stdout.on("data", (chunk) => { stdout += String(chunk); });
		child.stderr.on("data", (chunk) => { stderr += String(chunk); });
		child.on("error", (error) => finish(new Error(`Unable to execute wsl.exe: ${error.message}`)));
		child.on("close", (code) => {
			if (code !== 0) return finish(new Error(`WSL process inspection failed with exit ${code}: ${bounded(stderr || stdout)}`));
			try { finish(undefined, JSON.parse(stdout) as ProcessObservation); }
			catch { finish(new Error(`WSL process inspection returned malformed JSON: ${bounded(stdout)}`)); }
		});
	});
}

export async function inspectProductionProcess(pid: number, signal?: AbortSignal): Promise<ProcessObservation> {
	if (!Number.isInteger(pid) || pid <= 1) return { pid, exists: false, alive: false, state: null, pgid: null, sid: null, startToken: null, marker: null, error: "PID must be an integer greater than 1." };
	return process.platform === "win32" ? inspectWslProcess(pid, signal) : inspectPosixProcess(pid);
}

async function readBoundedTail(path: string): Promise<{ text: string; bytesRead: number; truncated: boolean } | null> {
	let handle;
	try {
		const info = await stat(path);
		if (!info.isFile()) return null;
		const length = Math.min(info.size, MAX_LOG_READ_BYTES);
		const buffer = Buffer.alloc(length);
		handle = await open(path, "r");
		const read = await handle.read(buffer, 0, length, Math.max(0, info.size - length));
		return { text: buffer.subarray(0, read.bytesRead).toString("utf8"), bytesRead: read.bytesRead, truncated: info.size > length };
	} catch { return null; }
	finally { await handle?.close().catch(() => undefined); }
}

export interface LaunchBoundary { index: number; end: number; line: string; timestampMs: number | null; runName: string }

export function findLaunchBoundaries(text: string): LaunchBoundary[] {
	const result: LaunchBoundary[] = [];
	const expression = /^\[launch_run\s+([^\]]+)\].*?\brun=([^\s]+)(?:\s|$).*$/gm;
	for (const match of text.matchAll(expression)) {
		result.push({ index: match.index!, end: match.index! + match[0].length, line: bounded(match[0]), timestampMs: parseFiniteTime(match[1]), runName: match[2] });
	}
	return result;
}

export async function inspectProductionLog(cwd: string, runName: string | null, record: LaunchRecord | null, signal?: AbortSignal): Promise<LogObservation> {
	abortIfRequested(signal);
	const path = runName ? resolve(cwd, "logs", `${runName}.log`) : null;
	const empty: LogObservation = { path, exists: false, association: "none", banner: null, progress: null, fatal: null, boundary: null, bytesRead: 0, truncated: false };
	if (!path) return empty;
	const read = await readBoundedTail(path);
	if (!read) return empty;
	let segment = read.text;
	let association: LogObservation["association"] = record?.kind === "legacy" ? "legacy_unverified" : "manual_unverified";
	let boundary: LaunchBoundary | null = null;
	let associationError: string | undefined;
	if (record?.kind === "current") {
		const matching = findLaunchBoundaries(read.text).filter((candidate) => equalFold(candidate.runName, runName));
		boundary = matching.at(-1) ?? null;
		if (!boundary) associationError = read.truncated ? "Current launch boundary is outside the bounded log tail or missing." : "Current launch boundary is missing.";
		else if (record.launchedAtMs === null || boundary.timestampMs === null || Math.abs(record.launchedAtMs - boundary.timestampMs) > 2_000) associationError = "Latest matching launch boundary disagrees with the PID record launch timestamp.";
		else {
			association = "current_launch";
			segment = read.text.slice(boundary.end);
		}
	}
	return {
		path,
		exists: true,
		association,
		banner: associationError ? null : detectTrainingBanner(segment),
		progress: associationError ? null : detectPpoProgress(segment),
		fatal: associationError ? null : detectFatalLogError(segment),
		boundary: boundary?.line ?? null,
		bytesRead: read.bytesRead,
		truncated: read.truncated,
		...(associationError ? { associationError } : {}),
	};
}

export const PRODUCTION_HEALTH_SWEEP_DEPENDENCIES: HealthSweepDependencies = {
	clock: SYSTEM_HEALTH_SWEEP_CLOCK,
	buildIndex: buildRepositoryHealthIndex,
	inspectProcess: inspectProductionProcess,
	inspectLog: inspectProductionLog,
};

function baseChecks(): RunChecks {
	return { processAlive: false, processIdentityVerified: false, trainingBanner: false, ppoProgress: false, mlflowCreated: false, mlflowIdentityConsistent: false };
}

function reason(code: HealthReasonCode, message: string, evidence?: string): HealthReason {
	return { code, message, ...(evidence ? { evidence: bounded(evidence) } : {}) };
}

function resolveUnique<T>(items: T[], label: string, identify: (item: T) => string): { value: T | null; issue: HealthReason | null } {
	if (items.length <= 1) return { value: items[0] ?? null, issue: null };
	return { value: null, issue: reason("identity_ambiguous", `${label} matched multiple identities.`, items.map(identify).join(", ")) };
}

function compatibleMlflowRuns(runs: MlflowRunObservation[], name: string, record: LaunchRecord | null): MlflowRunObservation[] {
	const exact = runs.filter((run) => equalFold(run.runName, name));
	if (!record || record.launchedAtMs === null) return exact;
	return exact.filter((run) => run.startTimeMs !== null && run.startTimeMs >= record.launchedAtMs! - LAUNCH_TIME_SLOP_MS);
}

async function observeTarget(
	supplied: HealthSweepRunInput,
	inputIndex: number,
	index: RepositoryHealthIndex,
	cwd: string,
	nowMs: number,
	deadlineReached: boolean,
	signal: AbortSignal | undefined,
	dependencies: HealthSweepDependencies,
): Promise<TargetObservation> {
	const checks = baseChecks();
	const reasons: HealthReason[] = [];
	const pending: string[] = [];
	let conclusive = false;

	const idMatches = supplied.runId === undefined ? [] : index.mlflowRuns.filter((run) => hasIdPrefix(run.runId, supplied.runId!));
	const explicitMlflow = resolveUnique(idMatches, `runId ${JSON.stringify(supplied.runId)}`, (run) => run.runId);
	if (explicitMlflow.issue) { reasons.push(explicitMlflow.issue); conclusive = true; }

	const nameRecords = supplied.runName === undefined ? [] : uniqueByPath(index.launchRecords.filter((record) => equalFold(record.runName, supplied.runName)));
	const pidRecords = supplied.pid === undefined ? [] : uniqueByPath(index.launchRecords.filter((record) => record.pid === supplied.pid));
	const idRecords = supplied.runId === undefined ? [] : uniqueByPath(index.launchRecords.filter((record) => record.mlflowRunId !== null && hasIdPrefix(record.mlflowRunId, supplied.runId!)));
	for (const [label, candidates] of [["runName launch metadata", nameRecords], ["PID launch metadata", pidRecords], ["runId launch metadata", idRecords]] as const) {
		if (candidates.length > 1) { reasons.push(reason("identity_ambiguous", `${label} matched duplicate PID records.`, candidates.map((item) => item.path).join(", "))); conclusive = true; }
	}
	const candidateRecords = uniqueByPath([...(nameRecords.length === 1 ? nameRecords : []), ...(pidRecords.length === 1 ? pidRecords : []), ...(idRecords.length === 1 ? idRecords : [])]);
	if (candidateRecords.length > 1) { reasons.push(reason("identity_mismatch", "Supplied selectors point to different launch metadata records.", candidateRecords.map((item) => item.path).join(", "))); conclusive = true; }
	let record = candidateRecords.length === 1 ? candidateRecords[0] : null;

	if (!record && supplied.runName !== undefined && nameRecords.length === 0) {
		const malformed = index.recordErrors.find((item) => equalFold(item.fileRunName, supplied.runName));
		if (malformed) { reasons.push(reason("process_identity_mismatch", "The matching launch PID record is malformed.", `${malformed.path}: ${malformed.message}`)); conclusive = true; }
	}

	let resolvedName = record?.runName ?? explicitMlflow.value?.runName ?? supplied.runName ?? null;
	if (explicitMlflow.value && supplied.runName !== undefined && !equalFold(explicitMlflow.value.runName, supplied.runName)) { reasons.push(reason("identity_mismatch", "Supplied runName disagrees with the explicitly selected MLflow run.", explicitMlflow.value.runName ?? "MLflow name absent")); conclusive = true; }
	if (!record && resolvedName) {
		const derived = uniqueByPath(index.launchRecords.filter((candidate) => equalFold(candidate.runName, resolvedName)));
		if (derived.length === 1) record = derived[0];
		else if (derived.length > 1) { reasons.push(reason("identity_ambiguous", "Resolved name matches duplicate launch metadata records.", derived.map((item) => item.path).join(", "))); conclusive = true; }
	}
	resolvedName = resolvedName ?? record?.runName ?? null;

	// Every supplied selector is an assertion about one launch. In particular, a
	// supplied ID remains authoritative even while its MLflow directory is absent:
	// never replace it with a record/name-selected run from a different identity.
	if (record && supplied.runName !== undefined && !equalFold(record.runName, supplied.runName)) { reasons.push(reason("identity_mismatch", "Supplied runName disagrees with PID launch metadata.")); conclusive = true; }
	if (record && supplied.pid !== undefined && record.pid !== supplied.pid) { reasons.push(reason("identity_mismatch", "Supplied PID disagrees with run-name/run-ID launch metadata.")); conclusive = true; }
	if (record?.mlflowRunId && supplied.runId !== undefined && !hasIdPrefix(record.mlflowRunId, supplied.runId)) {
		reasons.push(reason("identity_mismatch", "Supplied runId does not match the launch metadata MLflow ID.", `${supplied.runId} != ${record.mlflowRunId}`));
		conclusive = true;
	}
	if (record?.mlflowRunId && explicitMlflow.value && !equalFold(record.mlflowRunId, explicitMlflow.value.runId)) {
		reasons.push(reason("identity_mismatch", "Launch metadata MLflow ID disagrees with the explicitly selected MLflow run.", `${record.mlflowRunId} != ${explicitMlflow.value.runId}`));
		conclusive = true;
	}
	const resolvedPid = supplied.pid ?? record?.pid ?? null;

	let mlflow = explicitMlflow.value;
	if (supplied.runId === undefined && !mlflow && record?.mlflowRunId) {
		const matches = index.mlflowRuns.filter((run) => equalFold(run.runId, record!.mlflowRunId));
		const selected = resolveUnique(matches, "PID-record MLflow ID", (run) => run.runId);
		if (selected.issue) { reasons.push(selected.issue); conclusive = true; } else mlflow = selected.value;
	}
	if (supplied.runId === undefined && !mlflow && resolvedName) {
		const matches = compatibleMlflowRuns(index.mlflowRuns, resolvedName, record);
		const selected = resolveUnique(matches, `exact MLflow name ${JSON.stringify(resolvedName)}`, (run) => `${run.runId}@${run.runPath}`);
		if (selected.issue) { reasons.push(selected.issue); conclusive = true; } else mlflow = selected.value;
	}
	if (mlflow && supplied.runId !== undefined && !hasIdPrefix(mlflow.runId, supplied.runId)) { reasons.push(reason("identity_mismatch", "Resolved MLflow run does not match the supplied runId.", `${supplied.runId} != ${mlflow.runId}`)); conclusive = true; }
	if (mlflow && resolvedName && !equalFold(mlflow.runName, resolvedName)) { reasons.push(reason("identity_mismatch", "Resolved launch name disagrees with MLflow run metadata.", mlflow.runName ?? "MLflow name absent")); conclusive = true; }
	if (mlflow?.runName && (!record || equalFold(record.runName, mlflow.runName))) resolvedName = record?.runName ?? mlflow.runName;
	if (record?.mlflowRunPath && mlflow && resolve(record.mlflowRunPath) !== resolve(mlflow.runPath)) { reasons.push(reason("identity_mismatch", "PID record MLflow path disagrees with the resolved run directory.")); conclusive = true; }
	const resolvedRunId = mlflow?.runId ?? (supplied.runId === undefined ? record?.mlflowRunId ?? null : null);

	let processObservation: ProcessObservation | null = null;
	if (resolvedPid === null) pending.push("process identity");
	else {
		try { processObservation = await dependencies.inspectProcess(resolvedPid, signal); }
		catch (error) {
			if (error instanceof Error && error.name === "AbortError") throw error;
			processObservation = { pid: resolvedPid, exists: false, alive: false, state: null, pgid: null, sid: null, startToken: null, marker: null, error: error instanceof Error ? error.message : String(error) };
		}
		checks.processAlive = processObservation.alive;
		if (!processObservation.alive) { reasons.push(reason("process_dead", `WSL/Linux PID ${resolvedPid} is absent or not live${processObservation.state ? ` (state ${processObservation.state})` : ""}.`, processObservation.error)); conclusive = true; }
		else if (record?.kind === "current") {
			const matches = processObservation.startToken === record.startToken
				&& processObservation.marker === record.marker
				&& processObservation.pgid === record.pgid
				&& processObservation.sid === record.pid;
			checks.processIdentityVerified = matches;
			if (!matches) { reasons.push(reason("process_identity_mismatch", `Live PID ${resolvedPid} does not match the current launch marker/start token/process group/session.`, JSON.stringify({ expected: { marker: record.marker, startToken: record.startToken, pgid: record.pgid, sid: record.pid }, observed: processObservation }))); conclusive = true; }
		} else checks.processIdentityVerified = true;
	}

	let logObservation: LogObservation | null = null;
	if (resolvedName) logObservation = await dependencies.inspectLog(cwd, resolvedName, record, signal);
	checks.trainingBanner = Boolean(logObservation?.banner);
	checks.ppoProgress = Boolean(logObservation?.progress);
	if (logObservation?.fatal) { reasons.push(reason("fatal_log_error", "The current launch log contains a fatal startup diagnostic.", logObservation.fatal)); conclusive = true; }
	if (!logObservation?.exists) pending.push("run log");
	else {
		if (logObservation.associationError) pending.push("current-launch log boundary");
		if (!checks.trainingBanner) pending.push("training banner");
		if (!checks.ppoProgress) pending.push("positive PPO progress");
	}

	if (mlflow) {
		checks.mlflowCreated = true;
		checks.mlflowIdentityConsistent = !resolvedName || equalFold(mlflow.runName, resolvedName);
		if (normalizeMlflowStatus(mlflow.status) === "failed" && checks.processAlive) { reasons.push(reason("mlflow_terminal_failure", `Live PID ${resolvedPid} is paired with terminal failed/killed MLflow status ${mlflow.status}.`, mlflow.runPath)); conclusive = true; }
	} else pending.push("MLflow run creation");

	if (deadlineReached && !conclusive) {
		if (resolvedPid === null) reasons.push(reason("process_unresolved", "No WSL/Linux PID could be resolved from the supplied selectors and launch metadata."));
		if (!logObservation?.exists) reasons.push(reason("log_missing", resolvedName ? `The deterministic log logs/${resolvedName}.log is missing or unreadable.` : "No run name could be resolved for deterministic log lookup."));
		else {
			if (!checks.trainingBanner) reasons.push(reason("training_banner_missing", "No canonical [Train] device banner or compatible SB3 device banner appeared for the current launch.", logObservation.associationError));
			if (!checks.ppoProgress) reasons.push(reason("ppo_progress_missing", "No positive SB3 iterations/total_timesteps table record appeared for the current launch.", logObservation.associationError));
		}
		if (!mlflow) reasons.push(reason("mlflow_run_missing", supplied.runId ? `No unique readable MLflow run exists for ID/prefix ${JSON.stringify(supplied.runId)}.` : "No launch-associated readable MLflow run with meta.yaml was created."));
	}

	const healthy = !conclusive && reasons.length === 0 && checks.processAlive && checks.processIdentityVerified && checks.trainingBanner && checks.ppoProgress && checks.mlflowCreated && checks.mlflowIdentityConsistent;
	return {
		inputIndex,
		supplied,
		resolved: { runName: resolvedName, runId: resolvedRunId, pid: resolvedPid },
		reasons,
		checks,
		evidence: { pidRecordPath: record?.path ?? null, logPath: logObservation?.path ?? null, mlflowRunPath: mlflow?.runPath ?? null, process: processObservation, log: logObservation },
		observedAtMs: nowMs,
		observedAt: iso(nowMs),
		healthy,
		pending,
		conclusive,
	};
}

function makeResult(observations: TargetObservation[], startedAtMs: number, completedAtMs: number, pollCount: number, timedOut: boolean): HealthSweepResult {
	const unhealthyRuns: UnhealthyRun[] = observations.filter((item) => !item.healthy).map(({ healthy: _healthy, pending: _pending, conclusive: _conclusive, ...item }) => item);
	const audit = observations.map(({ pending: _pending, conclusive: _conclusive, ...item }) => item);
	const details: HealthSweepDetails = {
		unhealthyRuns,
		observations: audit,
		startedAtMs,
		startedAt: iso(startedAtMs),
		deadlineMs: startedAtMs + HEALTH_SWEEP_TIMEOUT_MS,
		deadline: iso(startedAtMs + HEALTH_SWEEP_TIMEOUT_MS),
		completedAtMs,
		completedAt: iso(completedAtMs),
		waitedMs: Math.max(0, completedAtMs - startedAtMs),
		pollCount,
		timedOut,
	};
	return { content: [{ type: "text", text: JSON.stringify(unhealthyRuns) }], details };
}

/** Shared-deadline polling core. The fixed timeout cannot be overridden by public input. */
export async function healthSweep(
	input: HealthSweepInput,
	cwd: string,
	signal?: AbortSignal,
	onUpdate?: (result: { content: Array<{ type: "text"; text: string }>; details?: unknown }) => void,
	dependencies: HealthSweepDependencies = PRODUCTION_HEALTH_SWEEP_DEPENDENCIES,
): Promise<HealthSweepResult> {
	const runs = validateInput(input);
	abortIfRequested(signal);
	const startedAtMs = dependencies.clock.now();
	const deadlineMs = startedAtMs + HEALTH_SWEEP_TIMEOUT_MS;
	let pollCount = 0;
	let finalizedOnce = false;
	const settled = new Map<number, TargetObservation>();

	while (true) {
		abortIfRequested(signal);
		const nowMs = dependencies.clock.now();
		const deadlineReached = nowMs >= deadlineMs;
		const index = await dependencies.buildIndex(cwd, signal);
		const observations: TargetObservation[] = [];
		for (let inputIndex = 0; inputIndex < runs.length; inputIndex++) {
			abortIfRequested(signal);
			const prior = settled.get(inputIndex);
			if (prior) observations.push(prior);
			else {
				const observation = await observeTarget(runs[inputIndex], inputIndex, index, cwd, nowMs, deadlineReached, signal, dependencies);
				observations.push(observation);
				if (observation.conclusive) settled.set(inputIndex, observation);
			}
		}
		pollCount += 1;
		const allDecided = observations.every((item) => item.healthy || item.conclusive || deadlineReached);
		if (deadlineReached) return makeResult(observations, startedAtMs, nowMs, pollCount, true);
		if (allDecided) {
			// Rebuild the index and recheck every non-conclusive target once immediately
			// before return, preventing an earlier liveness result from being reused.
			if (finalizedOnce) return makeResult(observations, startedAtMs, nowMs, pollCount, false);
			finalizedOnce = true;
			continue;
		}
		finalizedOnce = false;
		const pendingText = observations
			.filter((item) => !item.healthy && !item.conclusive)
			.map((item) => `#${item.inputIndex + 1}: ${[...new Set(item.pending)].join(", ")}`)
			.join("; ");
		onUpdate?.({ content: [{ type: "text", text: `Health sweep pending (${Math.max(0, deadlineMs - nowMs)} ms remain): ${pendingText}` }], details: { pending: observations.filter((item) => !item.healthy && !item.conclusive).map((item) => item.inputIndex) } });
		await dependencies.clock.sleep(Math.min(HEALTH_SWEEP_POLL_INTERVAL_MS, deadlineMs - dependencies.clock.now()), signal);
		// The next iteration is the mandatory final read when sleep lands exactly on the deadline.
	}
}

export default function healthSweepExtension(pi: ExtensionAPI): void {
	pi.registerTool({
		name: "health_sweep",
		label: "Health Sweep After Launch",
		description: "Immediately after launch_run, check a non-empty batch under one fixed 10-minute cancellable deadline. Resolves consistent runName/runId/PID launch metadata, verifies live WSL process identity, the current launch's training banner and positive PPO rollout progress, and readable MLflow creation; returns only the input-ordered unhealthy-run JSON list (literal [] when all are healthy). Use this instead of rebuilding ad hoc ps, grep, or MLflow checks.",
		promptSnippet: "Immediately health-check launched training runs as one batch instead of ad hoc ps/grep/MLflow commands",
		promptGuidelines: [
			"Call health_sweep immediately after launch_run for all newly launched runs in one batch; do not rebuild ad hoc ps, grep, or MLflow health commands.",
			"Pass every known runName, runId, and WSL PID to health_sweep when available; multiple identifiers for one descriptor must identify the same launch.",
			"Treat health_sweep's returned JSON array as the unhealthy runs; literal [] means every requested launch passed process, current-log, PPO-progress, and MLflow checks.",
		],
		parameters: healthSweepSchema,
		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			return healthSweep(params, ctx.cwd, signal, onUpdate, PRODUCTION_HEALTH_SWEEP_DEPENDENCIES);
		},
	});
}
