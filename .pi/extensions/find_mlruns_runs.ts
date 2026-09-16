import { mkdtemp, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative, resolve, sep } from "node:path";
import { StringEnum } from "@earendil-works/pi-ai";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	truncateHead,
	withFileMutationQueue,
	type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";

const PARAMETER_OPERATORS = ["eq", "ne", "contains", "gt", "gte", "lt", "lte"] as const;
const METRIC_OPERATORS = ["eq", "ne", "gt", "gte", "lt", "lte"] as const;
const SORT_FIELDS = ["start_time", "end_time", "run_name", "run_id"] as const;
const SORT_ORDERS = ["asc", "desc"] as const;
export const MAX_RESULTS = 100;
export const DEFAULT_RESULTS = 10;

type ParameterOperator = (typeof PARAMETER_OPERATORS)[number];
type MetricOperator = (typeof METRIC_OPERATORS)[number];
type SortOrder = (typeof SORT_ORDERS)[number];
type SortField = (typeof SORT_FIELDS)[number];

const parameterFilterSchema = Type.Object({
	key: Type.String({ description: "Slash-normalized parameter key, or an unambiguous normalized alias." }),
	operator: Type.Optional(StringEnum(PARAMETER_OPERATORS, { description: "Comparison operator (default: eq)." })),
	value: Type.Union([Type.String(), Type.Number()]),
});

const metricFilterSchema = Type.Object({
	key: Type.String({ description: "Slash-normalized metric key, or an unambiguous normalized alias." }),
	operator: Type.Optional(StringEnum(METRIC_OPERATORS, { description: "Numeric comparison operator (default: eq)." })),
	value: Type.Number(),
});

export const findMlrunsRunsSchema = Type.Object({
	mlrunsPath: Type.Optional(Type.String({ description: "MLflow file-store root, relative to the current directory (default: mlruns)." })),
	experimentId: Type.Optional(Type.String({ description: "Exact experiment ID." })),
	experimentName: Type.Optional(Type.String({ description: "Exact experiment name (case-insensitive)." })),
	runName: Type.Optional(Type.String({ description: "Case-insensitive run-name substring." })),
	runId: Type.Optional(Type.String({ description: "Full run ID or run-ID prefix." })),
	startedAfter: Type.Optional(Type.Union([Type.String(), Type.Number()], { description: "Inclusive ISO/date or Unix-seconds/milliseconds lower bound." })),
	startedBefore: Type.Optional(Type.Union([Type.String(), Type.Number()], { description: "Inclusive ISO/date or Unix-seconds/milliseconds upper bound." })),
	endedAfter: Type.Optional(Type.Union([Type.String(), Type.Number()], { description: "Inclusive ISO/date or Unix-seconds/milliseconds lower bound." })),
	endedBefore: Type.Optional(Type.Union([Type.String(), Type.Number()], { description: "Inclusive ISO/date or Unix-seconds/milliseconds upper bound." })),
	parameterFilters: Type.Optional(Type.Array(parameterFilterSchema, { description: "AND-composed parameter predicates." })),
	metricFilters: Type.Optional(Type.Array(metricFilterSchema, { description: "AND-composed predicates on each metric file's latest valid sample." })),
	rankByMetric: Type.Optional(Type.String({ description: "Metric key to rank by; runs missing it are excluded." })),
	rankOrder: Type.Optional(StringEnum(SORT_ORDERS, { description: "Metric ranking order (default: desc)." })),
	latest: Type.Optional(Type.Integer({ minimum: 1, maximum: MAX_RESULTS, description: "Preselect this many newest matching runs before optional ranking/sorting." })),
	sort: Type.Optional(Type.Object({
		by: StringEnum(SORT_FIELDS, { description: "Metadata sort field (ignored when rankByMetric is set)." }),
		order: Type.Optional(StringEnum(SORT_ORDERS, { description: "Sort order (default: desc)." })),
	})),
	limit: Type.Optional(Type.Integer({ minimum: 1, maximum: MAX_RESULTS, description: `Maximum returned runs (default ${DEFAULT_RESULTS}, maximum ${MAX_RESULTS}).` })),
});

export type FindMlrunsRunsInput = Static<typeof findMlrunsRunsSchema>;
export interface MetricSample { value: number; timestamp: number | null; step: number | null }
export interface MlrunsRunResult {
	experimentId: string;
	experimentName: string | null;
	runId: string;
	runName: string | null;
	status: string | null;
	startTimeMs: number | null;
	startTime: string | null;
	endTimeMs: number | null;
	endTime: string | null;
	parameters: Record<string, string>;
	metrics: Record<string, MetricSample>;
	warnings: string[];
}
export interface FindMlrunsRunsDetails {
	count: number;
	mlrunsPath: string;
	query: FindMlrunsRunsInput;
	runs: MlrunsRunResult[];
	truncated: boolean;
	fullReportPath?: string;
}

interface LoadedRun extends MlrunsRunResult {
	parameterMap: Map<string, string>;
	metricMap: Map<string, MetricSample>;
}

function abortIfRequested(signal?: AbortSignal): void {
	if (signal?.aborted) {
		const error = new Error("MLflow run search was cancelled.");
		error.name = "AbortError";
		throw error;
	}
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

function parseQueryTime(value: string | number | undefined, field: string): number | undefined {
	if (value === undefined) return undefined;
	if (typeof value === "number") {
		if (!Number.isFinite(value)) throw new Error(`${field} must be a finite Unix timestamp or valid date.`);
		return Math.abs(value) < 1_000_000_000_000 ? value * 1000 : value;
	}
	const trimmed = value.trim();
	if (!trimmed) throw new Error(`${field} cannot be empty; use an ISO date/time or Unix seconds/milliseconds.`);
	if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(trimmed)) {
		const number = Number(trimmed);
		if (!Number.isFinite(number)) throw new Error(`${field} is not a finite timestamp: ${value}`);
		return Math.abs(number) < 1_000_000_000_000 ? number * 1000 : number;
	}
	const parsed = Date.parse(trimmed);
	if (!Number.isFinite(parsed)) throw new Error(`${field} is not a valid date/time: ${value}`);
	return parsed;
}

function iso(ms: number | null): string | null {
	if (ms == null) return null;
	try { return new Date(ms).toISOString(); } catch { return null; }
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

function parseMetricHistory(text: string): MetricSample | undefined {
	const lines = text.split(/\r?\n/);
	for (let index = lines.length - 1; index >= 0; index--) {
		const line = lines[index].trim();
		if (!line) continue;
		const parts = line.split(/\s+/);
		if (parts.length !== 3) continue;
		const timestamp = Number(parts[0]);
		const value = Number(parts[1]);
		const step = Number(parts[2]);
		if (Number.isFinite(timestamp) && Number.isFinite(value) && Number.isFinite(step)) {
			return { value, timestamp, step };
		}
	}
	return undefined;
}

function sortedObject<T>(map: Map<string, T>): Record<string, T> {
	return Object.fromEntries([...map.entries()].sort(([a], [b]) => a.localeCompare(b)));
}

async function loadRun(runDirectory: string, directoryRunId: string, experimentId: string, experimentName: string | null, signal?: AbortSignal): Promise<LoadedRun | undefined> {
	abortIfRequested(signal);
	const metaText = await readText(join(runDirectory, "meta.yaml"));
	if (metaText === undefined) return undefined;
	const meta = parseSimpleYaml(metaText);
	const warnings: string[] = [];
	const parameterMap = new Map<string, string>();
	const metricMap = new Map<string, MetricSample>();

	const paramsRoot = join(runDirectory, "params");
	for (const file of await recursiveFiles(paramsRoot, signal)) {
		const key = relativeKey(paramsRoot, file);
		const value = await readText(file);
		if (value === undefined) warnings.push(`Unreadable parameter: ${key}`);
		else parameterMap.set(key, value.trim());
	}
	const metricsRoot = join(runDirectory, "metrics");
	for (const file of await recursiveFiles(metricsRoot, signal)) {
		const key = relativeKey(metricsRoot, file);
		const value = await readText(file);
		if (value === undefined) { warnings.push(`Unreadable metric: ${key}`); continue; }
		const sample = parseMetricHistory(value);
		if (sample) metricMap.set(key, sample);
		else warnings.push(`Metric has no valid samples: ${key}`);
	}

	const tagName = (await readText(join(runDirectory, "tags", "mlflow.runName")))?.trim();
	const startTimeMs = parseStoredTime(meta.start_time);
	const endTimeMs = parseStoredTime(meta.end_time);
	if (meta.start_time && startTimeMs === null) warnings.push("Malformed start_time in meta.yaml");
	if (meta.end_time && endTimeMs === null) warnings.push("Malformed end_time in meta.yaml");
	const runId = meta.run_id?.trim() || directoryRunId;
	return {
		experimentId,
		experimentName,
		runId,
		runName: tagName || meta.run_name?.trim() || null,
		status: meta.status?.trim() || null,
		startTimeMs,
		startTime: iso(startTimeMs),
		endTimeMs,
		endTime: iso(endTimeMs),
		parameters: sortedObject(parameterMap),
		metrics: sortedObject(metricMap),
		warnings,
		parameterMap,
		metricMap,
	};
}

function normalizedSelector(value: string): string {
	return value.replace(/\\/g, "/").replace(/^\.\//, "");
}

function alias(value: string): string {
	return normalizedSelector(value).toLocaleLowerCase().replace(/[^a-z0-9]/g, "");
}

function resolveKey<T>(map: Map<string, T>, selector: string, kind: string, runId: string): string | undefined {
	const exact = normalizedSelector(selector);
	if (map.has(exact)) return exact;
	const wanted = alias(exact);
	const matches = [...map.keys()].filter((key) => alias(key) === wanted).sort();
	if (matches.length > 1) throw new Error(`Ambiguous ${kind} key "${selector}" in run ${runId}; use one of: ${matches.join(", ")}`);
	return matches[0];
}

function compareNumber(left: number, operator: MetricOperator | Exclude<ParameterOperator, "contains">, right: number): boolean {
	switch (operator) {
		case "eq": return left === right;
		case "ne": return left !== right;
		case "gt": return left > right;
		case "gte": return left >= right;
		case "lt": return left < right;
		case "lte": return left <= right;
	}
}

function passesParameter(run: LoadedRun, filter: Static<typeof parameterFilterSchema>): boolean {
	const key = resolveKey(run.parameterMap, filter.key, "parameter", run.runId);
	if (!key) return false;
	const actual = run.parameterMap.get(key)!;
	const operator = (filter.operator ?? "eq") as ParameterOperator;
	if (operator === "contains") return actual.toLocaleLowerCase().includes(String(filter.value).toLocaleLowerCase());
	if (operator === "eq" || operator === "ne") {
		const equal = actual === String(filter.value);
		return operator === "eq" ? equal : !equal;
	}
	const left = Number(actual);
	const right = Number(filter.value);
	return Number.isFinite(left) && Number.isFinite(right) && compareNumber(left, operator, right);
}

function passesMetric(run: LoadedRun, filter: Static<typeof metricFilterSchema>): boolean {
	const key = resolveKey(run.metricMap, filter.key, "metric", run.runId);
	if (!key) return false;
	return compareNumber(run.metricMap.get(key)!.value, (filter.operator ?? "eq") as MetricOperator, filter.value);
}

function metricFor(run: LoadedRun, selector: string): MetricSample | undefined {
	const key = resolveKey(run.metricMap, selector, "metric", run.runId);
	return key ? run.metricMap.get(key) : undefined;
}

function deterministicCompare(a: LoadedRun, b: LoadedRun, field: SortField, order: SortOrder): number {
	let comparison = 0;
	if (field === "start_time") comparison = (a.startTimeMs ?? -Infinity) - (b.startTimeMs ?? -Infinity);
	else if (field === "end_time") comparison = (a.endTimeMs ?? -Infinity) - (b.endTimeMs ?? -Infinity);
	else if (field === "run_name") comparison = (a.runName ?? "").localeCompare(b.runName ?? "");
	else comparison = a.runId.localeCompare(b.runId);
	if (comparison) return order === "asc" ? comparison : -comparison;
	const startTie = (b.startTimeMs ?? -Infinity) - (a.startTimeMs ?? -Infinity);
	return startTie || a.experimentId.localeCompare(b.experimentId) || a.runId.localeCompare(b.runId);
}

export async function queryMlrunsRuns(input: FindMlrunsRunsInput, cwd: string, signal?: AbortSignal): Promise<{ mlrunsPath: string; runs: MlrunsRunResult[] }> {
	abortIfRequested(signal);
	const limit = input.limit ?? input.latest ?? DEFAULT_RESULTS;
	if (!Number.isInteger(limit) || limit < 1 || limit > MAX_RESULTS) throw new Error(`limit must be an integer from 1 to ${MAX_RESULTS}.`);
	if (input.latest !== undefined && (!Number.isInteger(input.latest) || input.latest < 1 || input.latest > MAX_RESULTS)) throw new Error(`latest must be an integer from 1 to ${MAX_RESULTS}.`);
	const times = {
		startedAfter: parseQueryTime(input.startedAfter, "startedAfter"),
		startedBefore: parseQueryTime(input.startedBefore, "startedBefore"),
		endedAfter: parseQueryTime(input.endedAfter, "endedAfter"),
		endedBefore: parseQueryTime(input.endedBefore, "endedBefore"),
	};
	if (times.startedAfter !== undefined && times.startedBefore !== undefined && times.startedAfter > times.startedBefore) throw new Error("startedAfter must not be later than startedBefore.");
	if (times.endedAfter !== undefined && times.endedBefore !== undefined && times.endedAfter > times.endedBefore) throw new Error("endedAfter must not be later than endedBefore.");

	const root = resolve(cwd, (input.mlrunsPath ?? "mlruns").replace(/^@/, ""));
	try { if (!(await stat(root)).isDirectory()) throw new Error("not a directory"); }
	catch { throw new Error(`MLflow store is not a readable directory: ${root}`); }
	let experiments;
	try { experiments = await readdir(root, { withFileTypes: true }); }
	catch (error) { throw new Error(`Unable to read MLflow store ${root}: ${String(error)}`); }
	const loaded: LoadedRun[] = [];
	for (const experimentEntry of experiments.sort((a, b) => a.name.localeCompare(b.name))) {
		abortIfRequested(signal);
		if (!experimentEntry.isDirectory()) continue;
		const experimentId = experimentEntry.name;
		if (input.experimentId && experimentId !== input.experimentId) continue;
		const experimentDirectory = join(root, experimentId);
		const experimentMetaText = await readText(join(experimentDirectory, "meta.yaml"));
		const experimentName = experimentMetaText ? parseSimpleYaml(experimentMetaText).name?.trim() || null : null;
		if (input.experimentName && (experimentName ?? "").toLocaleLowerCase() !== input.experimentName.toLocaleLowerCase()) continue;
		let children;
		try { children = await readdir(experimentDirectory, { withFileTypes: true }); } catch { continue; }
		for (const child of children.sort((a, b) => a.name.localeCompare(b.name))) {
			abortIfRequested(signal);
			if (!child.isDirectory()) continue;
			const run = await loadRun(join(experimentDirectory, child.name), child.name, experimentId, experimentName, signal);
			if (run) loaded.push(run);
		}
	}

	let matches = loaded.filter((run) => {
		if (input.runId && !run.runId.toLocaleLowerCase().startsWith(input.runId.toLocaleLowerCase())) return false;
		if (input.runName && !(run.runName ?? "").toLocaleLowerCase().includes(input.runName.toLocaleLowerCase())) return false;
		if (times.startedAfter !== undefined && (run.startTimeMs === null || run.startTimeMs < times.startedAfter)) return false;
		if (times.startedBefore !== undefined && (run.startTimeMs === null || run.startTimeMs > times.startedBefore)) return false;
		if (times.endedAfter !== undefined && (run.endTimeMs === null || run.endTimeMs < times.endedAfter)) return false;
		if (times.endedBefore !== undefined && (run.endTimeMs === null || run.endTimeMs > times.endedBefore)) return false;
		if (input.parameterFilters && !input.parameterFilters.every((filter) => passesParameter(run, filter))) return false;
		if (input.metricFilters && !input.metricFilters.every((filter) => passesMetric(run, filter))) return false;
		return true;
	});

	matches.sort((a, b) => deterministicCompare(a, b, "start_time", "desc"));
	if (input.latest !== undefined) matches = matches.slice(0, input.latest);
	if (input.rankByMetric) {
		matches = matches.filter((run) => metricFor(run, input.rankByMetric!) !== undefined);
		const order = input.rankOrder ?? "desc";
		matches.sort((a, b) => {
			const comparison = metricFor(a, input.rankByMetric!)!.value - metricFor(b, input.rankByMetric!)!.value;
			if (comparison) return order === "asc" ? comparison : -comparison;
			return deterministicCompare(a, b, "start_time", "desc");
		});
	} else if (input.sort) {
		matches.sort((a, b) => deterministicCompare(a, b, input.sort!.by, input.sort!.order ?? "desc"));
	}
	return { mlrunsPath: root, runs: matches.slice(0, limit).map(({ parameterMap: _p, metricMap: _m, ...run }) => run) };
}

function formatRun(run: MlrunsRunResult, index: number): string {
	const lines = [
		`=== Run ${index + 1} ===`,
		`Run ID: ${run.runId}`,
		`Run name: ${run.runName ?? "(absent)"}`,
		`Experiment: ${run.experimentName ?? "(absent)"} (${run.experimentId})`,
		`Status: ${run.status ?? "(absent)"}`,
		`Start: ${run.startTime ?? "(absent)"}`,
		`End: ${run.endTime ?? "(absent)"}`,
		"Parameters:",
	];
	const parameters = Object.entries(run.parameters);
	if (!parameters.length) lines.push("  (none)");
	else for (const [key, value] of parameters) lines.push(`  - ${key}: ${value === "" ? "(empty)" : value}`);
	lines.push("Metrics:");
	const metrics = Object.entries(run.metrics);
	if (!metrics.length) lines.push("  (none)");
	else for (const [key, sample] of metrics) lines.push(`  - ${key}: ${sample.value} (timestamp=${sample.timestamp ?? "absent"}, step=${sample.step ?? "absent"})`);
	if (run.warnings.length) {
		lines.push("Warnings:");
		for (const warning of run.warnings) lines.push(`  - ${warning}`);
	}
	return lines.join("\n");
}

export function formatMlrunsReport(runs: MlrunsRunResult[], mlrunsPath: string): string {
	if (!runs.length) return `Found 0 matching MLflow runs in ${mlrunsPath}.`;
	return [`Found ${runs.length} matching MLflow run(s) in ${mlrunsPath}.`, ...runs.map(formatRun)].join("\n\n");
}

export default function findMlrunsRunsExtension(pi: ExtensionAPI): void {
	pi.registerTool({
		name: "find_mlruns_runs",
		label: "Find MLflow Runs",
		description: `Directly search an on-disk MLflow file tree. Criteria compose with AND semantics; names are case-insensitive substrings, run IDs are prefixes, bounds are inclusive, and metrics use their latest valid history sample. Returns every readable parameter and metric for up to ${MAX_RESULTS} runs. Output is truncated at ${DEFAULT_MAX_LINES} lines or ${formatSize(DEFAULT_MAX_BYTES)} with a full-report path.`,
		promptSnippet: "Search and rank local MLflow runs without the MLflow API",
		parameters: findMlrunsRunsSchema,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const result = await queryMlrunsRuns(params, ctx.cwd, signal);
			abortIfRequested(signal);
			const fullReport = formatMlrunsReport(result.runs, result.mlrunsPath);
			const truncation = truncateHead(fullReport, { maxLines: DEFAULT_MAX_LINES, maxBytes: DEFAULT_MAX_BYTES });
			let text = truncation.content;
			const details: FindMlrunsRunsDetails = { count: result.runs.length, mlrunsPath: result.mlrunsPath, query: params, runs: result.runs, truncated: truncation.truncated };
			if (truncation.truncated) {
				abortIfRequested(signal);
				const directory = await mkdtemp(join(tmpdir(), "pi-find-mlruns-"));
				const fullReportPath = join(directory, "report.txt");
				await withFileMutationQueue(fullReportPath, () => writeFile(fullReportPath, fullReport, { encoding: "utf8", signal }));
				details.fullReportPath = fullReportPath;
				const notice = `[Output truncated from ${truncation.totalLines} lines (${formatSize(truncation.totalBytes)}). Full report saved to: ${fullReportPath}]`;
				const separator = "\n\n";
				const readable = truncateHead(fullReport, {
					maxLines: DEFAULT_MAX_LINES - 2,
					maxBytes: DEFAULT_MAX_BYTES - Buffer.byteLength(separator + notice, "utf8"),
				});
				text = readable.content + separator + notice;
			}
			return { content: [{ type: "text" as const, text }], details };
		},
	});
}
