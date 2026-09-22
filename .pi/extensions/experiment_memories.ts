import { constants } from "node:fs";
import {
	lstat,
	mkdir,
	open,
	readdir,
	readFile,
	realpath,
	rename,
	rm,
	stat,
} from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { randomUUID } from "node:crypto";
import { StringEnum } from "@earendil-works/pi-ai";
import { withFileMutationQueue, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";

export const SCHEMA_VERSION = 3;
export const TREND_SECTIONS = [
	"breakthroughs_and_dead_ends",
	"correlations_and_patterns",
	"champion",
	"promising_directions",
] as const;

type JsonScalar = string | number | boolean | null;
type JsonValue = JsonScalar | JsonValue[] | { [key: string]: JsonValue };
type TrendSection = (typeof TREND_SECTIONS)[number];

const jsonScalarSchema = Type.Union([Type.String(), Type.Number(), Type.Boolean(), Type.Null()]);
const resultInsightSchema = Type.Object({
	metric: Type.String({ description: "Exact metric key supporting this result." }),
	outcome: Type.String({ description: "Structured outcome label or value." }),
	insight: Type.String({ description: "Explanation of what the metric/outcome implies." }),
});
export const runEntrySchema = Type.Object({
	run_name: Type.String(),
	timestamp: Type.String({
		format: "date-time",
		pattern: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3}Z$",
		description: "Canonical UTC run-start timestamp derived from MLflow start_time (new Date(start_time).toISOString()).",
	}),
	mlruns_run_id: Type.Optional(Type.String()),
	mlruns_path: Type.Optional(Type.String()),
	modified_params: Type.Record(Type.String(), jsonScalarSchema),
	goal_hypothesis: Type.String(),
	results_insights: Type.Array(resultInsightSchema, { minItems: 1 }),
	reasoning: Type.Array(Type.String(), { minItems: 3, description: "Candidate explanations ordered by likelihood." }),
	tags: Type.Optional(Type.Array(Type.String(), { description: "Structured tags used for exact lookup; omitted input is stored as an empty array." })),
});

export const writeExperimentMemorySchema = Type.Object({
	operation: StringEnum(["append_run", "update_experiment_conclusion", "update_trends_section"] as const),
	experiment: Type.Optional(Type.String({ description: "Safe experiment/ticket filename stem; required for append_run and update_experiment_conclusion." })),
	conclusion: Type.Optional(Type.String({ description: "Non-empty curated experiment conclusion; required for append_run and update_experiment_conclusion." })),
	run: Type.Optional(runEntrySchema),
	section: Type.Optional(StringEnum(TREND_SECTIONS)),
	content: Type.Optional(Type.Any({ description: "Non-null JSON-compatible section content; strings must be non-empty." })),
});

export const queryExperimentMemoriesSchema = Type.Object({
	mode: StringEnum(["runs", "experiment_conclusion", "trends_section"] as const),
	experiment: Type.Optional(Type.String({ description: "Case-insensitive exact experiment name." })),
	run_name: Type.Optional(Type.String({ description: "Case-insensitive exact run name." })),
	modified_param: Type.Optional(Type.String({ description: "Case-insensitive exact modified parameter name." })),
	modified_param_value: Type.Optional(jsonScalarSchema),
	metric: Type.Optional(Type.String({ description: "Case-insensitive exact results metric." })),
	outcome: Type.Optional(Type.String({ description: "Case-insensitive exact results outcome." })),
	tag: Type.Optional(Type.String({ description: "Case-insensitive exact tag." })),
	section: Type.Optional(StringEnum(TREND_SECTIONS)),
});

export type RunEntry = Static<typeof runEntrySchema>;
export type WriteExperimentMemoryInput = Static<typeof writeExperimentMemorySchema>;
export type QueryExperimentMemoriesInput = Static<typeof queryExperimentMemoriesSchema>;
export interface ExperimentDocument { schema_version: number; experiment: string; conclusion: string; runs: RunEntry[] }
export interface TrendsDocument { schema_version: number; sections: Record<TrendSection, JsonValue> }
export interface RunMatch { experiment: string; entry: RunEntry }
export interface CompactEvidencePointer { timestamp: string; run_id?: string; experiment_name?: string }

function abortIfRequested(signal?: AbortSignal): void {
	if (signal?.aborted) {
		const error = new Error("Experiment memory operation was cancelled.");
		error.name = "AbortError";
		throw error;
	}
}

function nonEmpty(value: unknown): value is string {
	return typeof value === "string" && value.trim().length > 0;
}

function validateExperimentName(value: unknown, field = "experiment"): asserts value is string {
	if (!nonEmpty(value)) throw new Error(`${field} must be a non-empty string.`);
	if (value === "." || value === ".." || value.includes("/") || value.includes("\\") || value.includes("\0")) {
		throw new Error(`${field} must be a filesystem-safe filename stem without traversal or path separators.`);
	}
	if (value.toLocaleLowerCase() === "_trends") throw new Error(`${field} name _TRENDS is reserved for the canonical trends memory.`);
	if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)) {
		throw new Error(`${field} may contain only letters, numbers, dot, underscore, and hyphen, and must start with a letter or number.`);
	}
}

function isJsonScalar(value: unknown): value is JsonScalar {
	return value === null || typeof value === "string" || typeof value === "boolean" || (typeof value === "number" && Number.isFinite(value));
}

function validateCanonicalTimestamp(value: unknown, path: string): asserts value is string {
	if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)) {
		throw new Error(`${path} must be a canonical UTC ISO-8601 timestamp with milliseconds (for example, 2026-09-12T16:43:19.363Z).`);
	}
	const milliseconds = Date.parse(value);
	if (!Number.isFinite(milliseconds) || new Date(milliseconds).toISOString() !== value) {
		throw new Error(`${path} must be a valid canonical UTC ISO-8601 timestamp with milliseconds.`);
	}
}

const FORBIDDEN_TREND_KEYS = new Set(["qualification", "support", "terminal_evidence"]);

function validateCompactEvidencePointer(value: unknown, path: string): asserts value is CompactEvidencePointer {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${path} must be a compact evidence pointer object.`);
	const prototype = Object.getPrototypeOf(value);
	if (prototype !== Object.prototype && prototype !== null) throw new Error(`${path} must be a plain compact evidence pointer object.`);
	const pointer = value as Record<string, unknown>;
	const keys = Object.keys(pointer).sort();
	const hasRunId = Object.prototype.hasOwnProperty.call(pointer, "run_id");
	const hasExperimentName = Object.prototype.hasOwnProperty.call(pointer, "experiment_name");
	if (hasRunId === hasExperimentName) throw new Error(`${path} must contain exactly one evidence locator: run_id or experiment_name.`);
	const expectedKeys = (hasRunId ? ["run_id", "timestamp"] : ["experiment_name", "timestamp"]).sort();
	if (keys.length !== expectedKeys.length || keys.some((key, index) => key !== expectedKeys[index])) {
		throw new Error(`${path} may contain only one locator (run_id or experiment_name) and timestamp; extra evidence keys are not allowed.`);
	}
	validateCanonicalTimestamp(pointer.timestamp, `${path}.timestamp`);
	if (hasRunId) {
		if (!nonEmpty(pointer.run_id)) throw new Error(`${path}.run_id must be a non-empty string.`);
	} else {
		validateExperimentName(pointer.experiment_name, `${path}.experiment_name`);
	}
}

function validateTrendJsonValue(value: unknown, path: string, seen = new Set<object>()): asserts value is JsonValue {
	if (isJsonScalar(value)) return;
	if (typeof value !== "object" || value === null) throw new Error(`${path} must be JSON-compatible (objects, arrays, or scalar values).`);
	if (seen.has(value)) throw new Error(`${path} must not contain cyclic data.`);
	seen.add(value);
	if (Array.isArray(value)) {
		value.forEach((item, index) => validateTrendJsonValue(item, `${path}[${index}]`, seen));
	} else {
		const prototype = Object.getPrototypeOf(value);
		if (prototype !== Object.prototype && prototype !== null) throw new Error(`${path} must contain plain JSON objects only.`);
		const object = value as Record<string, unknown>;
		for (const [key, item] of Object.entries(object)) {
			if (FORBIDDEN_TREND_KEYS.has(key)) throw new Error(`${path}.${key} is forbidden in schema 3 trends; migrate its reasoning to an experiment conclusion.`);
			if (key === "evidence") {
				if (!Array.isArray(item) || item.length === 0) throw new Error(`${path}.evidence must be a non-empty list of compact timestamped evidence pointers.`);
				item.forEach((pointer, index) => validateCompactEvidencePointer(pointer, `${path}.evidence[${index}]`));
				continue;
			}
			if (key === "historical_summaries") {
				if (!Array.isArray(item)) throw new Error(`${path}.historical_summaries must be a list of evidence-free one-line strings.`);
				item.forEach((summary, index) => {
					if (!nonEmpty(summary) || /[\r\n]/.test(summary)) throw new Error(`${path}.historical_summaries[${index}] must be a non-empty single-line string.`);
				});
				continue;
			}
			validateTrendJsonValue(item, `${path}.${key}`, seen);
		}
	}
	seen.delete(value);
}

function validateRunEntry(value: unknown, path = "run", requireStoredTags = false): asserts value is RunEntry {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${path} must be an object.`);
	const run = value as Record<string, unknown>;
	validateCanonicalTimestamp(run.timestamp, `${path}.timestamp`);
	for (const field of ["run_name", "goal_hypothesis"] as const) {
		if (!nonEmpty(run[field])) throw new Error(`${path}.${field} must be a non-empty string.`);
	}
	if (!nonEmpty(run.mlruns_run_id) && !nonEmpty(run.mlruns_path)) {
		throw new Error(`${path} must include at least one non-empty evidence locator: mlruns_run_id or mlruns_path.`);
	}
	for (const field of ["mlruns_run_id", "mlruns_path"] as const) {
		if (run[field] !== undefined && !nonEmpty(run[field])) throw new Error(`${path}.${field}, when supplied, must be a non-empty string.`);
	}
	if (typeof run.modified_params !== "object" || run.modified_params === null || Array.isArray(run.modified_params)) {
		throw new Error(`${path}.modified_params must be a non-empty parameter-name-to-JSON-scalar map.`);
	}
	const parameters = Object.entries(run.modified_params as Record<string, unknown>);
	if (!parameters.length) throw new Error(`${path}.modified_params must contain at least one changed parameter.`);
	for (const [key, item] of parameters) {
		if (!nonEmpty(key)) throw new Error(`${path}.modified_params keys must be non-empty.`);
		if (!isJsonScalar(item)) throw new Error(`${path}.modified_params.${key} must be a finite JSON scalar.`);
	}
	if (!Array.isArray(run.results_insights) || !run.results_insights.length) {
		throw new Error(`${path}.results_insights must be a non-empty structured list.`);
	}
	run.results_insights.forEach((item, index) => {
		if (typeof item !== "object" || item === null || Array.isArray(item)) throw new Error(`${path}.results_insights[${index}] must be an object.`);
		for (const field of ["metric", "outcome", "insight"] as const) {
			if (!nonEmpty((item as Record<string, unknown>)[field])) throw new Error(`${path}.results_insights[${index}].${field} must be a non-empty string.`);
		}
	});
	if (!Array.isArray(run.reasoning) || run.reasoning.length < 3) throw new Error(`${path}.reasoning must contain at least 3 candidate explanations ordered by likelihood.`);
	run.reasoning.forEach((reason, index) => {
		if (!nonEmpty(reason)) throw new Error(`${path}.reasoning[${index}] must be a non-empty string.`);
	});
	if (run.tags === undefined && !requireStoredTags) return;
	if (!Array.isArray(run.tags)) throw new Error(`${path}.tags must be a string array (it may be empty).`);
	run.tags.forEach((tag, index) => {
		if (!nonEmpty(tag)) throw new Error(`${path}.tags[${index}] must be a non-empty string.`);
	});
}

function schemaMigrationGuidance(version: unknown): string {
	return version === 1 || version === 2
		? ` Schema version ${version} is controlled migration input only; migrate to schema 3 across the complete experiment memory store before reading or writing.`
		: "";
}

function validateExperimentDocument(value: unknown, expectedExperiment?: string): asserts value is ExperimentDocument {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("experiment memory must be a JSON object.");
	const document = value as Record<string, unknown>;
	if (document.schema_version !== SCHEMA_VERSION) {
		throw new Error(`unsupported experiment memory schema_version; expected ${SCHEMA_VERSION}.${schemaMigrationGuidance(document.schema_version)}`);
	}
	const keys = Object.keys(document).sort();
	const expectedKeys = ["conclusion", "experiment", "runs", "schema_version"].sort();
	if (keys.length !== expectedKeys.length || keys.some((key, index) => key !== expectedKeys[index])) {
		throw new Error("experiment memory must contain exactly schema_version, experiment, conclusion, and runs.");
	}
	validateExperimentName(document.experiment, "stored experiment");
	if (expectedExperiment !== undefined && document.experiment !== expectedExperiment) throw new Error(`stored experiment identity ${JSON.stringify(document.experiment)} does not match filename ${JSON.stringify(expectedExperiment)}.`);
	if (!nonEmpty(document.conclusion)) throw new Error("experiment memory conclusion must be a non-empty string.");
	if (!Array.isArray(document.runs)) throw new Error("experiment memory runs must be an array.");
	document.runs.forEach((run, index) => validateRunEntry(run, `runs[${index}]`, true));
}

function isTrendSection(value: unknown): value is TrendSection {
	return typeof value === "string" && (TREND_SECTIONS as readonly string[]).includes(value);
}

function emptySections(): Record<TrendSection, JsonValue> {
	return {
		breakthroughs_and_dead_ends: [],
		correlations_and_patterns: [],
		champion: {},
		promising_directions: [],
	};
}

function validateTrendsDocument(value: unknown): asserts value is TrendsDocument {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("trends memory must be a JSON object.");
	const document = value as Record<string, unknown>;
	if (document.schema_version !== SCHEMA_VERSION) {
		throw new Error(`unsupported trends memory schema_version; expected ${SCHEMA_VERSION}.${schemaMigrationGuidance(document.schema_version)}`);
	}
	const documentKeys = Object.keys(document).sort();
	if (documentKeys.length !== 2 || documentKeys[0] !== "schema_version" || documentKeys[1] !== "sections") {
		throw new Error("trends memory must contain exactly schema_version and sections.");
	}
	if (typeof document.sections !== "object" || document.sections === null || Array.isArray(document.sections)) throw new Error("trends memory sections must be an object.");
	const sections = document.sections as Record<string, unknown>;
	const keys = Object.keys(sections).sort();
	const expected = [...TREND_SECTIONS].sort();
	if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) throw new Error(`trends memory must contain exactly these sections: ${TREND_SECTIONS.join(", ")}.`);
	for (const section of TREND_SECTIONS) validateTrendJsonValue(sections[section], `sections.${section}`);
}

function containedBy(parent: string, child: string): boolean {
	const rel = relative(parent, child);
	return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}

async function ensurePlainDirectory(path: string): Promise<void> {
	try {
		const info = await lstat(path);
		if (info.isSymbolicLink()) throw new Error(`Refusing symlinked memory directory: ${path}`);
		if (!info.isDirectory()) throw new Error(`Memory store component is not a directory: ${path}`);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
		await mkdir(path);
	}
}

async function storeRootForWrite(cwd: string): Promise<string> {
	const canonicalCwd = await realpath(cwd);
	const memories = join(canonicalCwd, "memories");
	const store = join(memories, "experiments");
	await ensurePlainDirectory(memories);
	await ensurePlainDirectory(store);
	const canonicalStore = await realpath(store);
	if (!containedBy(canonicalCwd, canonicalStore)) throw new Error(`Memory store escapes the working directory: ${canonicalStore}`);
	return canonicalStore;
}

async function storeRootForRead(cwd: string): Promise<string | undefined> {
	const canonicalCwd = await realpath(cwd);
	const memories = join(canonicalCwd, "memories");
	const store = join(memories, "experiments");
	try {
		const memoriesInfo = await lstat(memories);
		if (memoriesInfo.isSymbolicLink() || !memoriesInfo.isDirectory()) throw new Error(`Memory store parent must be a real directory, not a symlink: ${memories}`);
		const info = await lstat(store);
		if (info.isSymbolicLink() || !info.isDirectory()) throw new Error(`Memory store must be a real directory, not a symlink: ${store}`);
		const canonicalStore = await realpath(store);
		if (!containedBy(canonicalCwd, canonicalStore)) throw new Error(`Memory store escapes the working directory: ${canonicalStore}`);
		return canonicalStore;
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
		throw error;
	}
}

function targetPath(root: string, filename: string): string {
	const target = resolve(root, filename);
	if (dirname(target) !== root || !containedBy(root, target)) throw new Error("Resolved memory path escapes the memory store.");
	return target;
}

async function rejectSymlinkTarget(target: string): Promise<void> {
	try {
		const info = await lstat(target);
		if (info.isSymbolicLink()) throw new Error(`Refusing symlinked memory file: ${target}`);
		if (!info.isFile()) throw new Error(`Memory target is not a regular file: ${target}`);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
	}
}

async function readJson(target: string, label: string): Promise<unknown> {
	let text: string;
	try { text = await readFile(target, "utf8"); }
	catch (error) { throw new Error(`Unable to read ${label} ${target}: ${String(error)}`); }
	try { return JSON.parse(text); }
	catch (error) { throw new Error(`Malformed JSON in ${label} ${target}: ${String(error)}`); }
}

async function atomicWrite(target: string, value: unknown, signal?: AbortSignal): Promise<void> {
	abortIfRequested(signal);
	const serialized = `${JSON.stringify(value, null, 2)}\n`;
	const temporary = join(dirname(target), `.${basename(target)}.${process.pid}.${randomUUID()}.tmp`);
	let created = false;
	try {
		const handle = await open(temporary, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY, 0o600);
		created = true;
		try { await handle.writeFile(serialized, { encoding: "utf8", signal }); }
		finally { await handle.close(); }
		abortIfRequested(signal);
		await rename(temporary, target);
		created = false;
	} catch (error) {
		let cleanupError: unknown;
		if (created) {
			try { await rm(temporary, { force: true }); } catch (cleanup) { cleanupError = cleanup; }
		}
		if (cleanupError) throw new Error(`Atomic memory write failed (${String(error)}); temporary-file cleanup also failed for ${temporary}: ${String(cleanupError)}`);
		throw error;
	}
}

function canonicalJson(value: JsonValue): JsonValue {
	if (Array.isArray(value)) return value.map(canonicalJson);
	if (value !== null && typeof value === "object") {
		return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalJson(value[key])])) as JsonValue;
	}
	return value;
}

function cloneRun(run: RunEntry): RunEntry {
	const entry: RunEntry = {
		run_name: run.run_name,
		timestamp: run.timestamp,
		modified_params: Object.fromEntries(Object.keys(run.modified_params).sort().map((key) => [key, run.modified_params[key]])),
		goal_hypothesis: run.goal_hypothesis,
		results_insights: run.results_insights.map(({ metric, outcome, insight }) => ({ metric, outcome, insight })),
		reasoning: [...run.reasoning],
		tags: [...(run.tags ?? [])],
	};
	if (run.mlruns_run_id !== undefined) entry.mlruns_run_id = run.mlruns_run_id;
	if (run.mlruns_path !== undefined) entry.mlruns_path = run.mlruns_path;
	return entry;
}

export async function writeExperimentMemory(input: WriteExperimentMemoryInput, cwd: string, signal?: AbortSignal): Promise<{ path: string; operation: string; experiment?: string; conclusion?: string; entry?: RunEntry; section?: TrendSection; sectionContent?: JsonValue }> {
	abortIfRequested(signal);
	if (input.operation === "append_run") {
		if (input.section !== undefined || input.content !== undefined) throw new Error("append_run accepts only experiment, conclusion, and run payload fields.");
		validateExperimentName(input.experiment);
		if (!nonEmpty(input.conclusion)) throw new Error("append_run requires a non-empty conclusion.");
		validateRunEntry(input.run);
		const root = await storeRootForWrite(cwd);
		const target = targetPath(root, `${input.experiment}.json`);
		return withFileMutationQueue(target, async () => {
			abortIfRequested(signal);
			await rejectSymlinkTarget(target);
			let current: ExperimentDocument = { schema_version: SCHEMA_VERSION, experiment: input.experiment!, conclusion: input.conclusion!, runs: [] };
			try {
				await stat(target);
				const parsed = await readJson(target, "experiment memory");
				validateExperimentDocument(parsed, input.experiment);
				current = parsed;
			} catch (error) {
				if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
			}
			const entry = cloneRun(input.run!);
			const next: ExperimentDocument = { schema_version: SCHEMA_VERSION, experiment: current.experiment, conclusion: input.conclusion!, runs: [...current.runs, entry] };
			validateExperimentDocument(next, input.experiment);
			await atomicWrite(target, next, signal);
			return { path: target, operation: input.operation, experiment: input.experiment, conclusion: input.conclusion, entry };
		});
	}
	if (input.operation === "update_experiment_conclusion") {
		if (input.run !== undefined || input.section !== undefined || input.content !== undefined) throw new Error("update_experiment_conclusion accepts only experiment and conclusion payload fields.");
		validateExperimentName(input.experiment);
		if (!nonEmpty(input.conclusion)) throw new Error("update_experiment_conclusion requires a non-empty conclusion.");
		const root = await storeRootForWrite(cwd);
		const target = targetPath(root, `${input.experiment}.json`);
		return withFileMutationQueue(target, async () => {
			abortIfRequested(signal);
			await rejectSymlinkTarget(target);
			const parsed = await readJson(target, "experiment memory");
			validateExperimentDocument(parsed, input.experiment);
			const next: ExperimentDocument = { schema_version: SCHEMA_VERSION, experiment: parsed.experiment, conclusion: input.conclusion!, runs: parsed.runs };
			validateExperimentDocument(next, input.experiment);
			await atomicWrite(target, next, signal);
			return { path: target, operation: input.operation, experiment: input.experiment, conclusion: input.conclusion };
		});
	}
	if (input.operation === "update_trends_section") {
		if (input.experiment !== undefined || input.conclusion !== undefined || input.run !== undefined) throw new Error("update_trends_section accepts only section and content payload fields.");
		if (!isTrendSection(input.section)) throw new Error(`section must be one of: ${TREND_SECTIONS.join(", ")}.`);
		if (input.content === undefined || input.content === null || (typeof input.content === "string" && !nonEmpty(input.content))) throw new Error("content must be non-null JSON-compatible data; string content must be non-empty.");
		validateTrendJsonValue(input.content, "content");
		const root = await storeRootForWrite(cwd);
		const target = targetPath(root, "_TRENDS.json");
		return withFileMutationQueue(target, async () => {
			abortIfRequested(signal);
			await rejectSymlinkTarget(target);
			let current: TrendsDocument = { schema_version: SCHEMA_VERSION, sections: emptySections() };
			try {
				await stat(target);
				const parsed = await readJson(target, "trends memory");
				validateTrendsDocument(parsed);
				current = parsed;
			} catch (error) {
				if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
			}
			const sectionContent = canonicalJson(input.content);
			const next: TrendsDocument = { schema_version: SCHEMA_VERSION, sections: { ...current.sections, [input.section]: sectionContent } };
			validateTrendsDocument(next);
			await atomicWrite(target, next, signal);
			return { path: target, operation: input.operation, section: input.section, sectionContent };
		});
	}
	throw new Error("operation must be append_run, update_experiment_conclusion, or update_trends_section.");
}

function equalName(left: string, right: string): boolean {
	return left.toLocaleLowerCase() === right.toLocaleLowerCase();
}

function equalScalar(left: JsonScalar, right: JsonScalar): boolean {
	return left === right;
}

function validateRunQuery(input: QueryExperimentMemoriesInput): void {
	if (input.section !== undefined) throw new Error("runs mode does not accept a trends section.");
	if (input.experiment !== undefined) validateExperimentName(input.experiment);
	for (const field of ["run_name", "modified_param", "metric", "outcome", "tag"] as const) {
		if (input[field] !== undefined && !nonEmpty(input[field])) throw new Error(`${field} must be a non-empty string when supplied.`);
	}
	if (input.modified_param_value !== undefined && input.modified_param === undefined) throw new Error("modified_param_value requires modified_param.");
	if (input.modified_param_value !== undefined && !isJsonScalar(input.modified_param_value)) throw new Error("modified_param_value must be a finite JSON scalar.");
}

function matchesRun(input: QueryExperimentMemoriesInput, experiment: string, run: RunEntry): boolean {
	if (input.experiment !== undefined && !equalName(experiment, input.experiment)) return false;
	if (input.run_name !== undefined && !equalName(run.run_name, input.run_name)) return false;
	if (input.modified_param !== undefined) {
		const key = Object.keys(run.modified_params).find((candidate) => equalName(candidate, input.modified_param!));
		if (key === undefined) return false;
		if (input.modified_param_value !== undefined && !equalScalar(run.modified_params[key] as JsonScalar, input.modified_param_value as JsonScalar)) return false;
	}
	if (input.metric !== undefined && !run.results_insights.some((item) => equalName(item.metric, input.metric!))) return false;
	if (input.outcome !== undefined && !run.results_insights.some((item) => equalName(item.outcome, input.outcome!))) return false;
	if (input.tag !== undefined && !(run.tags ?? []).some((tag) => equalName(tag, input.tag!))) return false;
	return true;
}

export async function queryExperimentMemories(input: QueryExperimentMemoriesInput, cwd: string, signal?: AbortSignal): Promise<{ mode: "runs"; root: string; matches: RunMatch[] } | { mode: "experiment_conclusion"; path: string; experiment: string; conclusion: string } | { mode: "trends_section"; path: string; section: TrendSection; content: JsonValue }> {
	abortIfRequested(signal);
	if (input.mode === "experiment_conclusion") {
		for (const field of ["run_name", "modified_param", "modified_param_value", "metric", "outcome", "tag", "section"] as const) {
			if (input[field] !== undefined) throw new Error(`experiment_conclusion mode does not accept ${field}.`);
		}
		validateExperimentName(input.experiment);
		const root = await storeRootForRead(cwd);
		if (!root) throw new Error(`Experiment memory ${JSON.stringify(input.experiment)} is missing because the experiment memory store does not exist.`);
		const entries = await readdir(root, { withFileTypes: true });
		const candidates = entries.filter((entry) => entry.name.endsWith(".json") && entry.name.toLocaleLowerCase() === `${input.experiment}.json`.toLocaleLowerCase());
		if (candidates.length === 0) throw new Error(`Experiment memory ${JSON.stringify(input.experiment)} does not exist.`);
		if (candidates.length > 1) throw new Error(`Experiment memory name ${JSON.stringify(input.experiment)} is ambiguous under case-insensitive lookup.`);
		const filenameExperiment = candidates[0].name.slice(0, -5);
		const target = targetPath(root, candidates[0].name);
		await rejectSymlinkTarget(target);
		const parsed = await readJson(target, "experiment memory");
		validateExperimentDocument(parsed, filenameExperiment);
		return { mode: "experiment_conclusion", path: target, experiment: parsed.experiment, conclusion: parsed.conclusion };
	}
	if (input.mode === "trends_section") {
		for (const field of ["experiment", "run_name", "modified_param", "modified_param_value", "metric", "outcome", "tag"] as const) {
			if (input[field] !== undefined) throw new Error(`trends_section mode does not accept ${field}.`);
		}
		if (!isTrendSection(input.section)) throw new Error(`section must be one of: ${TREND_SECTIONS.join(", ")}.`);
		const root = await storeRootForRead(cwd);
		if (!root) throw new Error("Canonical trends memory is missing because the experiment memory store does not exist.");
		const target = targetPath(root, "_TRENDS.json");
		await rejectSymlinkTarget(target);
		const parsed = await readJson(target, "canonical trends memory");
		validateTrendsDocument(parsed);
		return { mode: "trends_section", path: target, section: input.section, content: parsed.sections[input.section] };
	}
	if (input.mode !== "runs") throw new Error("mode must be runs, experiment_conclusion, or trends_section.");
	validateRunQuery(input);
	const root = await storeRootForRead(cwd);
	if (!root) return { mode: "runs", root: resolve(cwd, "memories", "experiments"), matches: [] };
	const matches: RunMatch[] = [];
	const entries = await readdir(root, { withFileTypes: true });
	for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
		abortIfRequested(signal);
		if (!entry.isFile() || !entry.name.endsWith(".json") || entry.name.toLocaleLowerCase() === "_trends.json") continue;
		const experiment = entry.name.slice(0, -5);
		try { validateExperimentName(experiment, "memory filename"); }
		catch (error) { throw new Error(`Malformed experiment memory filename ${entry.name}: ${String(error)}`); }
		if (input.experiment !== undefined && !equalName(experiment, input.experiment)) continue;
		const target = targetPath(root, entry.name);
		await rejectSymlinkTarget(target);
		const parsed = await readJson(target, "experiment memory");
		try { validateExperimentDocument(parsed, experiment); }
		catch (error) { throw new Error(`Malformed experiment memory ${target}: ${String(error)}`); }
		for (const run of parsed.runs) if (matchesRun(input, experiment, run)) matches.push({ experiment, entry: run });
	}
	return { mode: "runs", root, matches };
}

export default function experimentMemoriesExtension(pi: ExtensionAPI): void {
	pi.registerTool({
		name: "write_experiment_memory",
		label: "Write Experiment Memory",
		description: "Append a validated evidence-linked run analysis with its experiment conclusion, update an existing conclusion, or transactionally replace one canonical trends section. Strict JSON schema version 3 is stored under memories/experiments.",
		promptSnippet: "Append validated run analysis, update an experiment conclusion, or update one canonical trends section",
		parameters: writeExperimentMemorySchema,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const details = await writeExperimentMemory(params, ctx.cwd, signal);
			const text = details.operation === "append_run"
				? `Appended run ${JSON.stringify(details.entry!.run_name)} and refreshed the conclusion for experiment ${JSON.stringify(details.experiment)} at ${details.path}.`
				: details.operation === "update_experiment_conclusion"
					? `Updated the conclusion for experiment ${JSON.stringify(details.experiment)} at ${details.path}; existing runs were preserved.`
					: `Updated trends section ${JSON.stringify(details.section)} at ${details.path}; all other sections were preserved.`;
			return { content: [{ type: "text" as const, text }], details };
		},
	});

	pi.registerTool({
		name: "query_experiment_memories",
		label: "Query Experiment Memories",
		description: "Query validated experiment run entries using case-insensitive exact AND-composed filters, retrieve one experiment conclusion without its runs, or retrieve exactly one canonical trends section. Read-only queries never create the store.",
		promptSnippet: "Query structured experiment runs, one conclusion, or one canonical trends section",
		parameters: queryExperimentMemoriesSchema,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const details = await queryExperimentMemories(params, ctx.cwd, signal);
			const text = details.mode === "runs"
				? `Found ${details.matches.length} matching experiment run entr${details.matches.length === 1 ? "y" : "ies"}.`
				: details.mode === "experiment_conclusion"
					? `Retrieved the conclusion for experiment ${JSON.stringify(details.experiment)} from ${details.path}.`
					: `Retrieved trends section ${JSON.stringify(details.section)} from ${details.path}.`;
			return { content: [{ type: "text" as const, text }], details };
		},
	});
}
