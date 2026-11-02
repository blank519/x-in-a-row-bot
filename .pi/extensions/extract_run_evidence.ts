import { mkdtemp, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative, resolve, sep } from "node:path";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	truncateHead,
	withFileMutationQueue,
	type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";

export const SCHEMA_VERSION = 1;
export const MAX_RUNS = 32;
export const DEFAULT_RECENT_COUNT = 5;
export const EVIDENCE_FILE_NAME = "evidence.json";

export interface EvidenceSample { timestamp: number; value: number; step: number }

export interface EvidenceWindow {
	steps: number[];
	values: (number | null)[];
	presentCount: number;
	missingSteps: number[];
	mean: number | null;
	min: number | null;
	max: number | null;
}

export interface MetricEvidence {
	rawValidCount: number;
	uniqueStepCount: number;
	invalidLineCount: number;
	duplicateStepCount: number;
	firstSample: EvidenceSample | null;
	latestSample: EvidenceSample | null;
	minStep: number | null;
	maxStep: number | null;
	window: EvidenceWindow;
}

export interface ArtifactEntry { path: string; bytes: number }

export interface RunEvidence {
	label: string;
	experimentId: string;
	experimentName: string | null;
	directoryName: string;
	runId: string;
	tagRunName: string | null;
	metadataRunName: string | null;
	runName: string | null;
	status: string | null;
	startTimeMs: number | null;
	startTime: string | null;
	endTimeMs: number | null;
	endTime: string | null;
	parameterCount: number;
	parameters: Record<string, string>;
	metricFileCount: number;
	metricCount: number;
	metrics: Record<string, MetricEvidence>;
	missingMetricKeys: string[];
	artifactCount: number;
	artifacts: ArtifactEntry[];
	warnings: string[];
}

export interface EvidenceWindowConfig {
	mode: "aligned" | "recent";
	alignedSteps?: number[];
	recentCount?: number;
}

export interface ExtractRunEvidenceResult {
	schemaVersion: number;
	window: EvidenceWindowConfig;
	runs: RunEvidence[];
}

export interface ExtractRunEvidenceDetails {
	evidence: ExtractRunEvidenceResult;
	truncated: boolean;
	fullReportPath?: string;
}

const runDescriptorSchema = Type.Object({
	label: Type.String({ description: "Required unique non-blank label; preserved in the output for baseline/candidate tables." }),
	runId: Type.Optional(Type.String({ description: "Case-insensitive full run ID or unique run-ID prefix." })),
	runName: Type.Optional(Type.String({ description: "Case-insensitive exact run name from tags/mlflow.runName (meta.yaml run_name fallback)." })),
	experimentId: Type.Optional(Type.String({ description: "Exact experiment ID scoping the runId/runName lookup." })),
}, { additionalProperties: false });

export const extractRunEvidenceSchema = Type.Object({
	mlrunsPath: Type.Optional(Type.String({ description: "MLflow file-store root, relative to the current directory (default: mlruns)." })),
	runs: Type.Array(runDescriptorSchema, {
		minItems: 1,
		maxItems: MAX_RUNS,
		description: `Required non-empty batch of labeled run descriptors (at most ${MAX_RUNS}); each needs at least one of runId or runName.`,
	}),
	metricKeys: Type.Optional(Type.Array(Type.String(), { description: "Slash-normalized metric keys or unambiguous aliases; absent extracts every readable metric, empty array extracts none." })),
	alignedSteps: Type.Optional(Type.Array(Type.Integer({ minimum: 0 }), { description: "Explicit strictly increasing non-negative steps for aligned metric windows; overrides recentCount." })),
	recentCount: Type.Optional(Type.Integer({ minimum: 1, description: `Window size used when alignedSteps is absent (default ${DEFAULT_RECENT_COUNT}); must not be combined with alignedSteps.` })),
}, { additionalProperties: false });

export type RunDescriptor = Static<typeof runDescriptorSchema>;
export type ExtractRunEvidenceInput = Static<typeof extractRunEvidenceSchema>;

type WindowSpec = { mode: "aligned"; alignedSteps: number[] } | { mode: "recent"; recentCount: number };

interface ParsedMetricFile {
	rawValid: EvidenceSample[];
	unique: EvidenceSample[];
	invalidLineCount: number;
	duplicateStepCount: number;
}

interface RunCandidate {
	experimentId: string;
	experimentName: string | null;
	directoryName: string;
	runId: string;
	tagRunName: string | null;
	metadataRunName: string | null;
	resolvedRunName: string | null;
	status: string | null;
	startTimeMs: number | null;
	endTimeMs: number | null;
	warnings: string[];
}

function abortIfRequested(signal?: AbortSignal): void {
	if (signal?.aborted) {
		const error = new Error("Run evidence extraction was cancelled.");
		error.name = "AbortError";
		throw error;
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function rejectUnknownKeys(value: Record<string, unknown>, allowed: readonly string[], where: string): void {
	const unknown = Object.keys(value).filter((key) => !allowed.includes(key)).sort();
	if (unknown.length) {
		throw new Error(`${where} has unknown field(s): ${unknown.join(", ")}. Allowed fields: ${allowed.join(", ")}.`);
	}
}

function requireNonBlankString(value: unknown, field: string): string {
	if (typeof value !== "string" || value.trim() === "") {
		throw new Error(`${field} must be a non-blank string.`);
	}
	return value;
}

function validateInput(input: unknown): ExtractRunEvidenceInput {
	if (!isRecord(input)) throw new Error("Input must be an object.");
	rejectUnknownKeys(input, ["mlrunsPath", "runs", "metricKeys", "alignedSteps", "recentCount"], "Input");
	if (input.mlrunsPath !== undefined) {
		requireNonBlankString(input.mlrunsPath, "mlrunsPath");
	}
	const rawRuns = input.runs;
	if (!Array.isArray(rawRuns) || rawRuns.length === 0) {
		throw new Error("runs must be a non-empty array of labeled run descriptors.");
	}
	if (rawRuns.length > MAX_RUNS) {
		throw new Error(`runs accepts at most ${MAX_RUNS} descriptors; received ${rawRuns.length}.`);
	}
	const seenLabels = new Set<string>();
	const runs: RunDescriptor[] = rawRuns.map((raw, index) => {
		const where = `runs[${index}]`;
		if (!isRecord(raw)) throw new Error(`${where} must be an object.`);
		rejectUnknownKeys(raw, ["label", "runId", "runName", "experimentId"], where);
		const label = requireNonBlankString(raw.label, `${where}.label`);
		const labelKey = label.toLocaleLowerCase();
		if (seenLabels.has(labelKey)) {
			throw new Error(`Duplicate label "${label}" in runs; labels must be unique (case-insensitive).`);
		}
		seenLabels.add(labelKey);
		const hasRunId = raw.runId !== undefined;
		const hasRunName = raw.runName !== undefined;
		if (!hasRunId && !hasRunName) {
			throw new Error(`${where} needs at least one of runId or runName.`);
		}
		if (hasRunId) requireNonBlankString(raw.runId, `${where}.runId`);
		if (hasRunName) requireNonBlankString(raw.runName, `${where}.runName`);
		if (raw.experimentId !== undefined) requireNonBlankString(raw.experimentId, `${where}.experimentId`);
		return raw as unknown as RunDescriptor;
	});

	let metricKeys: string[] | undefined;
	if (input.metricKeys !== undefined) {
		if (!Array.isArray(input.metricKeys)) throw new Error("metricKeys must be an array of slash-normalized metric keys.");
		metricKeys = input.metricKeys.map((key, index) => requireNonBlankString(key, `metricKeys[${index}]`));
	}

	let alignedSteps: number[] | undefined;
	if (input.alignedSteps !== undefined) {
		if (!Array.isArray(input.alignedSteps) || input.alignedSteps.length === 0) {
			throw new Error("alignedSteps must be a non-empty array of non-negative integer steps.");
		}
		alignedSteps = input.alignedSteps.map((step, index) => {
			if (!Number.isInteger(step) || step < 0) {
				throw new Error(`alignedSteps[${index}] must be a non-negative integer; received ${String(step)}.`);
			}
			return step;
		});
		for (let index = 1; index < alignedSteps.length; index++) {
			if (alignedSteps[index] <= alignedSteps[index - 1]) {
				throw new Error("alignedSteps must be strictly increasing and duplicate-free.");
			}
		}
	}

	let recentCount: number | undefined;
	if (input.recentCount !== undefined) {
		if (!Number.isInteger(input.recentCount) || input.recentCount < 1) {
			throw new Error(`recentCount must be a positive integer; received ${String(input.recentCount)}.`);
		}
		recentCount = input.recentCount;
	}
	if (alignedSteps !== undefined && recentCount !== undefined) {
		throw new Error('recentCount must not be supplied together with alignedSteps; supply exactly one window mode.');
	}

	return { mlrunsPath: input.mlrunsPath as string | undefined, runs, metricKeys, alignedSteps, recentCount };
}

function parseSimpleYaml(text: string): Record<string, string> {
	const result: Record<string, string> = {};
	for (const raw of text.split(/\r?\n/)) {
		const line = raw.trim();
		if (!line || line.startsWith("#")) continue;
		const colon = line.indexOf(":");
		if (colon <= 0) continue;
		const key = line.slice(0, colon).trim();
		let value = line.slice(colon + 1).trim();
		if (value.length >= 2 && ((value.startsWith("'") && value.endsWith("'")) || (value.startsWith('"') && value.endsWith('"')))) {
			const quote = value[0];
			value = value.slice(1, -1).replace(quote === "'" ? /''/g : /\\"/g, quote === "'" ? "'" : '"');
		}
		result[key] = value;
	}
	return result;
}

function parseStoredTime(value: string | undefined): number | null {
	if (value == null || value.trim() === "") return null;
	const number = Number(value);
	return Number.isFinite(number) ? number : null;
}

function iso(ms: number | null): string | null {
	if (ms == null) return null;
	const date = new Date(ms);
	return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

async function readText(file: string): Promise<string | undefined> {
	try { return await readFile(file, "utf8"); } catch { return undefined; }
}

async function recursiveFiles(root: string, signal?: AbortSignal): Promise<string[]> {
	const files: string[] = [];
	const pending = [root];
	while (pending.length) {
		abortIfRequested(signal);
		const directory = pending.pop()!;
		let entries;
		try { entries = await readdir(directory, { withFileTypes: true }); } catch { continue; }
		entries.sort((a, b) => a.name.localeCompare(b.name));
		for (const entry of entries) {
			abortIfRequested(signal);
			const full = join(directory, entry.name);
			if (entry.isDirectory()) pending.push(full);
			else if (entry.isFile()) files.push(full);
		}
	}
	return files.sort();
}

function relativeKey(root: string, file: string): string {
	return relative(root, file).split(sep).join("/");
}

function normalizedSelector(value: string): string {
	return value.replace(/\\/g, "/").replace(/^\.\//, "");
}

function alias(value: string): string {
	return normalizedSelector(value).toLocaleLowerCase().replace(/[^a-z0-9]/g, "");
}

function parseMetricFile(text: string, signal?: AbortSignal): ParsedMetricFile {
	const rawValid: EvidenceSample[] = [];
	const byStep = new Map<number, EvidenceSample>();
	let invalidLineCount = 0;
	let duplicateStepCount = 0;
	for (const rawLine of text.split(/\r?\n/)) {
		abortIfRequested(signal);
		const line = rawLine.trim();
		if (!line) continue;
		const parts = line.split(/\s+/);
		if (parts.length !== 3) { invalidLineCount++; continue; }
		const timestamp = Number(parts[0]);
		const value = Number(parts[1]);
		const step = Number(parts[2]);
		if (!Number.isFinite(timestamp) || !Number.isFinite(value) || !Number.isFinite(step) || !Number.isInteger(step) || step < 0) {
			invalidLineCount++;
			continue;
		}
		const sample: EvidenceSample = { timestamp, value, step };
		rawValid.push(sample);
		if (byStep.has(step)) duplicateStepCount++;
		// Duplicate valid steps keep the last appended valid record.
		byStep.set(step, sample);
	}
	const unique = [...byStep.values()].sort((a, b) => a.step - b.step);
	return { rawValid, unique, invalidLineCount, duplicateStepCount };
}

function summaryOf(values: number[]): { mean: number | null; min: number | null; max: number | null } {
	if (!values.length) return { mean: null, min: null, max: null };
	let sum = 0;
	let min = values[0];
	let max = values[0];
	for (const value of values) {
		sum += value;
		if (value < min) min = value;
		if (value > max) max = value;
	}
	return { mean: sum / values.length, min, max };
}

function buildWindow(steps: number[], valueByStep: Map<number, number>): EvidenceWindow {
	const values: (number | null)[] = [];
	const present: number[] = [];
	const missingSteps: number[] = [];
	for (const step of steps) {
		const value = valueByStep.get(step);
		if (value === undefined) {
			values.push(null);
			missingSteps.push(step);
		} else {
			values.push(value);
			present.push(value);
		}
	}
	return { steps: [...steps], values, presentCount: present.length, missingSteps, ...summaryOf(present) };
}

function metricEvidence(parsed: ParsedMetricFile, window: EvidenceWindow): MetricEvidence {
	const unique = parsed.unique;
	return {
		rawValidCount: parsed.rawValid.length,
		uniqueStepCount: unique.length,
		invalidLineCount: parsed.invalidLineCount,
		duplicateStepCount: parsed.duplicateStepCount,
		firstSample: unique[0] ?? null,
		latestSample: unique[unique.length - 1] ?? null,
		minStep: unique[0]?.step ?? null,
		maxStep: unique[unique.length - 1]?.step ?? null,
		window,
	};
}

function resolveMetricKey(discoveredKeys: string[], selector: string, runId: string): string | undefined {
	const exact = normalizedSelector(selector);
	if (discoveredKeys.includes(exact)) return exact;
	const wanted = alias(exact);
	const matches = discoveredKeys.filter((key) => alias(key) === wanted);
	if (matches.length > 1) {
		throw new Error(`Ambiguous metric key "${selector}" in run ${runId}; matches: ${matches.join(", ")}. Use one exact slash-normalized key.`);
	}
	return matches[0];
}

async function scanCandidates(root: string, signal?: AbortSignal): Promise<RunCandidate[]> {
	const candidates: RunCandidate[] = [];
	let experimentEntries;
	try { experimentEntries = await readdir(root, { withFileTypes: true }); }
	catch (error) { throw new Error(`Unable to read MLflow store ${root}: ${String(error)}`); }
	for (const experimentEntry of experimentEntries.sort((a, b) => a.name.localeCompare(b.name))) {
		abortIfRequested(signal);
		if (!experimentEntry.isDirectory() || experimentEntry.name.startsWith(".")) continue;
		const experimentId = experimentEntry.name;
		const experimentDirectory = join(root, experimentId);
		const experimentMetaText = await readText(join(experimentDirectory, "meta.yaml"));
		const experimentName = experimentMetaText ? parseSimpleYaml(experimentMetaText).name?.trim() || null : null;
		let children;
		try { children = await readdir(experimentDirectory, { withFileTypes: true }); } catch { continue; }
		for (const child of children.sort((a, b) => a.name.localeCompare(b.name))) {
			abortIfRequested(signal);
			if (!child.isDirectory() || child.name.startsWith(".")) continue;
			const runDirectory = join(experimentDirectory, child.name);
			const warnings: string[] = [];
			if (!experimentMetaText) warnings.push(`Experiment ${experimentId} has no readable meta.yaml; experiment name is unknown.`);
			const metaText = await readText(join(runDirectory, "meta.yaml"));
			const meta = metaText ? parseSimpleYaml(metaText) : {};
			if (metaText === undefined) warnings.push("Run meta.yaml is missing or unreadable.");
			const tagRunName = (await readText(join(runDirectory, "tags", "mlflow.runName")))?.trim() || null;
			const metadataRunName = meta.run_name?.trim() || null;
			const startTimeMs = parseStoredTime(meta.start_time);
			const endTimeMs = parseStoredTime(meta.end_time);
			if (meta.start_time !== undefined && startTimeMs === null) warnings.push("Malformed start_time in meta.yaml.");
			if (meta.end_time !== undefined && endTimeMs === null) warnings.push("Malformed end_time in meta.yaml.");
			if (meta.experiment_id !== undefined && meta.experiment_id.trim() !== experimentId) {
				warnings.push(`meta.yaml experiment_id "${meta.experiment_id.trim()}" does not match directory "${experimentId}".`);
			}
			candidates.push({
				experimentId,
				experimentName,
				directoryName: child.name,
				runId: meta.run_id?.trim() || child.name,
				tagRunName,
				metadataRunName,
				resolvedRunName: tagRunName || metadataRunName,
				status: meta.status?.trim() || null,
				startTimeMs,
				endTimeMs,
				warnings,
			});
		}
	}
	return candidates;
}

function describeCandidate(candidate: RunCandidate): string {
	return `${candidate.experimentId}/${candidate.directoryName} (run_id=${candidate.runId}, run_name=${candidate.resolvedRunName ?? "absent"})`;
}

function filterByExperiment(candidates: RunCandidate[], experimentId: string | undefined, root: string, where: string): RunCandidate[] {
	if (experimentId === undefined) return candidates;
	const matches = candidates.filter((candidate) => candidate.experimentId === experimentId);
	if (!matches.length) {
		const known = [...new Set(candidates.map((candidate) => candidate.experimentId))].sort();
		throw new Error(`${where}: experimentId "${experimentId}" matched no experiment in ${root}; known experiment IDs: ${known.join(", ") || "(none)"}.`);
	}
	return matches;
}

function resolveCandidate(candidates: RunCandidate[], descriptor: RunDescriptor, index: number, root: string): RunCandidate {
	const where = `runs[${index}] ("${descriptor.label}")`;
	const scoped = filterByExperiment(candidates, descriptor.experimentId, root, where);
	let byId: RunCandidate | undefined;
	if (descriptor.runId !== undefined) {
		const wanted = descriptor.runId.toLocaleLowerCase();
		const exact = scoped.filter((candidate) => candidate.runId.toLocaleLowerCase() === wanted);
		const pool = exact.length ? exact : scoped.filter((candidate) => candidate.runId.toLocaleLowerCase().startsWith(wanted));
		if (!pool.length) {
			throw new Error(`${where}: no run matches runId "${descriptor.runId}"${descriptor.experimentId !== undefined ? ` in experiment ${descriptor.experimentId}` : ""} in ${root}.`);
		}
		if (pool.length > 1) {
			throw new Error(`${where}: runId "${descriptor.runId}" is ambiguous; ${pool.length} runs match. Candidates: ${pool.map(describeCandidate).join("; ")}. Use the full run ID.`);
		}
		byId = pool[0];
	}
	let byName: RunCandidate | undefined;
	if (descriptor.runName !== undefined) {
		const wanted = descriptor.runName.toLocaleLowerCase();
		const pool = scoped.filter((candidate) => (candidate.resolvedRunName ?? "").toLocaleLowerCase() === wanted);
		if (!pool.length) {
			throw new Error(`${where}: no run matches runName "${descriptor.runName}"${descriptor.experimentId !== undefined ? ` in experiment ${descriptor.experimentId}` : ""} in ${root}.`);
		}
		if (pool.length > 1) {
			throw new Error(`${where}: runName "${descriptor.runName}" is ambiguous; ${pool.length} runs share that exact name. Candidates: ${pool.map(describeCandidate).join("; ")}. Add experimentId or select by runId instead.`);
		}
		byName = pool[0];
	}
	if (byId && byName && (byId.experimentId !== byName.experimentId || byId.directoryName !== byName.directoryName)) {
		throw new Error(`${where}: runId and runName selectors resolve to different runs. runId "${descriptor.runId}" -> ${describeCandidate(byId)}; runName "${descriptor.runName}" -> ${describeCandidate(byName)}.`);
	}
	return byId ?? byName!;
}

async function collectArtifacts(runDirectory: string, signal?: AbortSignal): Promise<{ entries: ArtifactEntry[]; warnings: string[] }> {
	const entries: ArtifactEntry[] = [];
	const warnings: string[] = [];
	const artifactsRoot = join(runDirectory, "artifacts");
	try {
		if (!(await stat(artifactsRoot)).isDirectory()) throw new Error("not a directory");
	} catch {
		warnings.push("Missing or unreadable artifacts directory.");
		return { entries, warnings };
	}
	for (const file of await recursiveFiles(artifactsRoot, signal)) {
		abortIfRequested(signal);
		try {
			const info = await stat(file);
			if (!info.isFile()) continue;
			entries.push({ path: relativeKey(artifactsRoot, file), bytes: info.size });
		} catch {
			warnings.push(`Unreadable artifact file: ${relativeKey(artifactsRoot, file)}`);
		}
	}
	entries.sort((a, b) => a.path.localeCompare(b.path));
	return { entries, warnings };
}

async function loadRunEvidence(
	root: string,
	candidate: RunCandidate,
	descriptor: RunDescriptor,
	metricKeys: string[] | undefined,
	spec: WindowSpec,
	signal?: AbortSignal,
): Promise<RunEvidence> {
	abortIfRequested(signal);
	const runDirectory = join(root, candidate.experimentId, candidate.directoryName);
	const warnings = [...candidate.warnings];

	const parameterMap = new Map<string, string>();
	const paramsRoot = join(runDirectory, "params");
	try {
		if (!(await stat(paramsRoot)).isDirectory()) throw new Error("not a directory");
	} catch {
		warnings.push("Missing or unreadable params directory.");
	}
	for (const file of await recursiveFiles(paramsRoot, signal)) {
		const key = relativeKey(paramsRoot, file);
		const value = await readText(file);
		if (value === undefined) warnings.push(`Unreadable parameter: ${key}`);
		else parameterMap.set(key, value.trim());
	}

	const metricMap = new Map<string, ParsedMetricFile>();
	const metricsRoot = join(runDirectory, "metrics");
	let metricFileCount = 0;
	try {
		if (!(await stat(metricsRoot)).isDirectory()) throw new Error("not a directory");
	} catch {
		warnings.push("Missing or unreadable metrics directory.");
	}
	const metricFiles = await recursiveFiles(metricsRoot, signal);
	metricFileCount = metricFiles.length;
	for (const file of metricFiles) {
		abortIfRequested(signal);
		const key = relativeKey(metricsRoot, file);
		const text = await readText(file);
		if (text === undefined) { warnings.push(`Unreadable metric: ${key}`); continue; }
		metricMap.set(key, parseMetricFile(text, signal));
	}

	const artifactCollection = await collectArtifacts(runDirectory, signal);
	warnings.push(...artifactCollection.warnings);

	const discoveredKeys = [...metricMap.keys()].sort();
	const selectedKeys: string[] = [];
	const missingMetricKeys = new Set<string>();
	if (metricKeys === undefined) {
		selectedKeys.push(...discoveredKeys);
	} else {
		for (const selector of metricKeys) {
			const key = resolveMetricKey(discoveredKeys, selector, candidate.runId);
			if (key === undefined) missingMetricKeys.add(selector);
			else if (!selectedKeys.includes(key)) selectedKeys.push(key);
		}
	}
	selectedKeys.sort();

	const metrics: Record<string, MetricEvidence> = {};
	for (const key of selectedKeys) {
		const parsed = metricMap.get(key)!;
		const valueByStep = new Map<number, number>(parsed.unique.map((sample) => [sample.step, sample.value] as const));
		const steps = spec.mode === "aligned"
			? spec.alignedSteps
			: parsed.unique.slice(Math.max(0, parsed.unique.length - spec.recentCount)).map((sample) => sample.step);
		metrics[key] = metricEvidence(parsed, buildWindow(steps, valueByStep));
	}

	const parameters: Record<string, string> = {};
	for (const key of [...parameterMap.keys()].sort()) parameters[key] = parameterMap.get(key)!;

	return {
		label: descriptor.label,
		experimentId: candidate.experimentId,
		experimentName: candidate.experimentName,
		directoryName: candidate.directoryName,
		runId: candidate.runId,
		tagRunName: candidate.tagRunName,
		metadataRunName: candidate.metadataRunName,
		runName: candidate.resolvedRunName,
		status: candidate.status,
		startTimeMs: candidate.startTimeMs,
		startTime: iso(candidate.startTimeMs),
		endTimeMs: candidate.endTimeMs,
		endTime: iso(candidate.endTimeMs),
		parameterCount: parameterMap.size,
		parameters,
		metricFileCount,
		metricCount: selectedKeys.length,
		metrics,
		missingMetricKeys: [...missingMetricKeys].sort(),
		artifactCount: artifactCollection.entries.length,
		artifacts: artifactCollection.entries,
		warnings,
	};
}

export function formatEvidenceJson(evidence: ExtractRunEvidenceResult): string {
	return `${JSON.stringify(evidence, null, 2)}\n`;
}

export async function extractRunEvidence(input: unknown, cwd: string, signal?: AbortSignal): Promise<ExtractRunEvidenceResult> {
	abortIfRequested(signal);
	const validated = validateInput(input);
	const spec: WindowSpec = validated.alignedSteps !== undefined
		? { mode: "aligned", alignedSteps: validated.alignedSteps }
		: { mode: "recent", recentCount: validated.recentCount ?? DEFAULT_RECENT_COUNT };
	const root = resolve(cwd, (validated.mlrunsPath ?? "mlruns").replace(/^@/, ""));
	try { if (!(await stat(root)).isDirectory()) throw new Error("not a directory"); }
	catch { throw new Error(`MLflow store is not a readable directory: ${root}`); }
	const candidates = await scanCandidates(root, signal);
	// Resolve every requested identity before extracting any evidence.
	const resolved = validated.runs.map((descriptor, index) => resolveCandidate(candidates, descriptor, index, root));
	abortIfRequested(signal);
	const runs: RunEvidence[] = [];
	for (const [index, candidate] of resolved.entries()) {
		runs.push(await loadRunEvidence(root, candidate, validated.runs[index], validated.metricKeys, spec, signal));
	}
	const window: EvidenceWindowConfig = spec.mode === "aligned"
		? { mode: "aligned", alignedSteps: [...spec.alignedSteps] }
		: { mode: "recent", recentCount: spec.recentCount };
	return { schemaVersion: SCHEMA_VERSION, window, runs };
}

export default function extractRunEvidenceExtension(pi: ExtensionAPI): void {
	pi.registerTool({
		name: "extract_run_evidence",
		label: "Extract Run Evidence",
		description: `Deterministic, read-only, multi-run MLflow evidence extractor for the local file store (no MLflow API). One call resolves and extracts one or many labeled runs (up to ${MAX_RUNS}) selected by case-insensitive run ID (full or unique prefix), case-insensitive exact run name, and optional exact experiment scope, preserving caller order for baseline/candidate tables. Each run record carries metadata, sorted parameters, per-metric raw/unique/invalid/duplicate counts with first/latest samples and step bounds, a recent-N or explicit aligned-step evidence window with null-filled missing steps and mean/min/max, an MLflow artifact inventory with byte sizes, missing requested metric keys, and warnings. Use this instead of writing one-off MLflow evidence parsers. Output is versioned pretty JSON truncated at ${DEFAULT_MAX_LINES} lines or ${formatSize(DEFAULT_MAX_BYTES)}; when truncated the complete JSON is saved to an OS-temporary evidence.json whose path is returned in the text and details.fullReportPath while details.evidence stays complete.`,
		promptSnippet: "Extract deterministic versioned JSON evidence for one or many MLflow runs without the MLflow API",
		promptGuidelines: [
			"Use extract_run_evidence instead of writing one-off MLflow evidence parsers whenever run evidence is needed for analysis or comparison.",
			"Batch every labeled run for one comparison into a single extract_run_evidence call; the returned runs array preserves caller order, so label the baseline first.",
			"Use extract_run_evidence's alignedSteps for exact checkpoint comparisons and rely on the default recent-five window otherwise; read details.evidence for the complete JSON when the text is truncated.",
		],
		parameters: extractRunEvidenceSchema,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const evidence = await extractRunEvidence(params, ctx.cwd, signal);
			abortIfRequested(signal);
			const jsonText = formatEvidenceJson(evidence);
			const truncation = truncateHead(jsonText, { maxLines: DEFAULT_MAX_LINES, maxBytes: DEFAULT_MAX_BYTES });
			const details: ExtractRunEvidenceDetails = { evidence, truncated: truncation.truncated };
			let text = truncation.content;
			if (truncation.truncated) {
				abortIfRequested(signal);
				const directory = await mkdtemp(join(tmpdir(), "pi-extract-run-evidence-"));
				const fullReportPath = join(directory, EVIDENCE_FILE_NAME);
				await withFileMutationQueue(fullReportPath, () => writeFile(fullReportPath, jsonText, { encoding: "utf8", signal }));
				details.fullReportPath = fullReportPath;
				const notice = `[Evidence truncated from ${truncation.totalLines} lines (${formatSize(truncation.totalBytes)}). Complete deterministic evidence JSON saved to: ${fullReportPath}]`;
				const separator = "\n\n";
				const readable = truncateHead(jsonText, {
					maxLines: DEFAULT_MAX_LINES - 2,
					maxBytes: DEFAULT_MAX_BYTES - Buffer.byteLength(separator + notice, "utf8"),
				});
				text = readable.content + separator + notice;
			}
			return { content: [{ type: "text" as const, text }], details };
		},
	});
}
