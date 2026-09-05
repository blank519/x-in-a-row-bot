import * as fs from "node:fs";
import * as path from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

type SortOrder = "asc" | "desc";
type ParamOp = "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "contains";

interface ParameterFilter {
	name: string;
	op?: ParamOp;
	value: string | number;
}

interface MetricQuery {
	name: string;
	sort?: SortOrder;
	min?: number;
	max?: number;
}

interface RunRecord {
	experimentId: string;
	experimentName?: string;
	runId: string;
	runName?: string;
	status?: string;
	startTimeMs?: number;
	endTimeMs?: number;
	params: Map<string, string>;
	paramNumeric: Map<string, number>;
	metrics: Map<string, number>;
}

interface QueryInput {
	mlrunsPath?: string;
	experimentId?: string;
	experimentName?: string;
	latestBy?: "start_time" | "end_time";
	startAfter?: string | number;
	startBefore?: string | number;
	endAfter?: string | number;
	endBefore?: string | number;
	metric?: MetricQuery;
	paramFilters?: ParameterFilter[];
	sort?: {
		by: "start_time" | "end_time" | "run_name" | "metric";
		order?: SortOrder;
	};
	limit?: number;
}

const MAX_LIMIT = 100;

export function normalizeAlias(value: string): string {
	return value.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function parseSimpleYaml(content: string): Record<string, string> {
	const parsed: Record<string, string> = {};
	for (const rawLine of content.split(/\r?\n/)) {
		const line = rawLine.trim();
		if (!line || line.startsWith("#")) continue;
		const idx = line.indexOf(":");
		if (idx <= 0) continue;
		const key = line.slice(0, idx).trim();
		let value = line.slice(idx + 1).trim();
		if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
			value = value.slice(1, -1);
		}
		parsed[key] = value;
	}
	return parsed;
}

function parseMaybeNumber(value?: string): number | undefined {
	if (value == null || value === "") return undefined;
	const num = Number(value);
	return Number.isFinite(num) ? num : undefined;
}

function parseTime(value: unknown): number | undefined {
	if (value == null) return undefined;
	if (typeof value === "number" && Number.isFinite(value)) {
		return value > 1_000_000_000_000 ? value : value * 1000;
	}
	if (typeof value === "string") {
		const trimmed = value.trim();
		if (!trimmed) return undefined;
		const numeric = Number(trimmed);
		if (Number.isFinite(numeric)) {
			return numeric > 1_000_000_000_000 ? numeric : numeric * 1000;
		}
		const parsed = Date.parse(trimmed);
		return Number.isFinite(parsed) ? parsed : undefined;
	}
	return undefined;
}

function formatTime(ms?: number): string {
	if (!ms) return "n/a";
	const d = new Date(ms);
	return Number.isNaN(d.getTime()) ? `${ms}` : d.toISOString();
}

function readTextSafe(filePath: string): string | undefined {
	try {
		return fs.readFileSync(filePath, "utf-8");
	} catch {
		return undefined;
	}
}

function listFilesRecursive(root: string): string[] {
	const files: string[] = [];
	const stack = [root];
	while (stack.length > 0) {
		const dir = stack.pop()!;
		let entries: fs.Dirent[] = [];
		try {
			entries = fs.readdirSync(dir, { withFileTypes: true });
		} catch {
			continue;
		}
		for (const entry of entries) {
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) {
				stack.push(full);
			} else if (entry.isFile()) {
				files.push(full);
			}
		}
	}
	return files;
}

function parseMetricFile(metricFile: string): number | undefined {
	const content = readTextSafe(metricFile);
	if (!content) return undefined;
	const lines = content
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter(Boolean);
	if (lines.length === 0) return undefined;
	for (let i = lines.length - 1; i >= 0; i -= 1) {
		const parts = lines[i].split(/\s+/);
		if (parts.length >= 2) {
			const value = Number(parts[1]);
			if (Number.isFinite(value)) return value;
		}
		const solo = Number(lines[i]);
		if (Number.isFinite(solo)) return solo;
	}
	return undefined;
}

function parseRun(runDir: string, experimentId: string, experimentName?: string): RunRecord | undefined {
	const runId = path.basename(runDir);
	const metaPath = path.join(runDir, "meta.yaml");
	const metaRaw = readTextSafe(metaPath);
	if (!metaRaw) return undefined;

	const meta = parseSimpleYaml(metaRaw);
	const params = new Map<string, string>();
	const paramNumeric = new Map<string, number>();
	const metrics = new Map<string, number>();

	const paramsDir = path.join(runDir, "params");
	if (fs.existsSync(paramsDir)) {
		for (const paramFile of listFilesRecursive(paramsDir)) {
			const rel = path.relative(paramsDir, paramFile).split(path.sep).join("/");
			const value = (readTextSafe(paramFile) ?? "").trim();
			params.set(rel, value);
			const numeric = parseMaybeNumber(value);
			if (numeric != null) paramNumeric.set(rel, numeric);
		}
	}

	const metricsDir = path.join(runDir, "metrics");
	if (fs.existsSync(metricsDir)) {
		for (const metricFile of listFilesRecursive(metricsDir)) {
			const rel = path.relative(metricsDir, metricFile).split(path.sep).join("/");
			const value = parseMetricFile(metricFile);
			if (value != null) {
				metrics.set(rel, value);
			}
		}
	}

	const runName = (readTextSafe(path.join(runDir, "tags", "mlflow.runName")) ?? "").trim() || meta.run_name;

	return {
		experimentId,
		experimentName,
		runId: meta.run_id || runId,
		runName,
		status: meta.status,
		startTimeMs: parseTime(meta.start_time),
		endTimeMs: parseTime(meta.end_time),
		params,
		paramNumeric,
		metrics,
	};
}

function findMatchingKey(map: Map<string, unknown>, selector: string): string | undefined {
	if (map.has(selector)) return selector;
	const selectorNorm = normalizeAlias(selector);
	const candidates = [...map.keys()].filter((key) => normalizeAlias(key) === selectorNorm);
	if (candidates.length === 0) return undefined;
	if (candidates.length === 1) return candidates[0];

	const slashNormalizedSelector = selector.replace(/\\/g, "/").toLowerCase();
	const exactPathish = candidates.find((key) => key.toLowerCase() === slashNormalizedSelector);
	if (exactPathish) return exactPathish;

	const endingMatch = candidates.find((key) => key.toLowerCase().endsWith(`/${slashNormalizedSelector}`));
	if (endingMatch) return endingMatch;

	return candidates.sort((a, b) => a.length - b.length)[0];
}

function passesParamFilter(run: RunRecord, filter: ParameterFilter): boolean {
	const op: ParamOp = filter.op ?? "eq";
	const key = findMatchingKey(run.params as Map<string, unknown>, filter.name);
	if (!key) return false;
	const rawValue = run.params.get(key) ?? "";
	const filterValueRaw = String(filter.value);

	if (op === "contains") {
		return rawValue.toLowerCase().includes(filterValueRaw.toLowerCase());
	}

	if (op === "eq") return rawValue === filterValueRaw;
	if (op === "ne") return rawValue !== filterValueRaw;

	const left = run.paramNumeric.get(key) ?? Number(rawValue);
	const right = typeof filter.value === "number" ? filter.value : Number(filter.value);
	if (!Number.isFinite(left) || !Number.isFinite(right)) return false;
	if (op === "gt") return left > right;
	if (op === "gte") return left >= right;
	if (op === "lt") return left < right;
	if (op === "lte") return left <= right;
	return false;
}

function metricValueForRun(run: RunRecord, metricName?: string): number | undefined {
	if (!metricName) return undefined;
	const key = findMatchingKey(run.metrics as Map<string, unknown>, metricName);
	if (!key) return undefined;
	return run.metrics.get(key);
}

function loadRuns(mlrunsPath: string): RunRecord[] {
	let experimentDirs: fs.Dirent[] = [];
	try {
		experimentDirs = fs.readdirSync(mlrunsPath, { withFileTypes: true });
	} catch {
		return [];
	}

	const runs: RunRecord[] = [];
	for (const expEntry of experimentDirs) {
		if (!expEntry.isDirectory()) continue;
		const experimentId = expEntry.name;
		const experimentPath = path.join(mlrunsPath, experimentId);
		const experimentMetaRaw = readTextSafe(path.join(experimentPath, "meta.yaml"));
		const experimentMeta = experimentMetaRaw ? parseSimpleYaml(experimentMetaRaw) : {};
		const experimentName = experimentMeta.name;

		let children: fs.Dirent[] = [];
		try {
			children = fs.readdirSync(experimentPath, { withFileTypes: true });
		} catch {
			continue;
		}

		for (const child of children) {
			if (!child.isDirectory()) continue;
			const runDir = path.join(experimentPath, child.name);
			const parsed = parseRun(runDir, experimentId, experimentName);
			if (parsed) runs.push(parsed);
		}
	}

	return runs;
}

export function queryRuns(input: QueryInput, cwd: string): RunRecord[] {
	const mlrunsPath = path.resolve(cwd, input.mlrunsPath ?? "mlruns");
	const allRuns = loadRuns(mlrunsPath);

	let runs = allRuns.filter((run) => {
		if (input.experimentId && run.experimentId !== input.experimentId) return false;
		if (input.experimentName && run.experimentName !== input.experimentName) return false;
		return true;
	});

	const startAfter = parseTime(input.startAfter);
	const startBefore = parseTime(input.startBefore);
	const endAfter = parseTime(input.endAfter);
	const endBefore = parseTime(input.endBefore);

	if (startAfter != null) runs = runs.filter((run) => (run.startTimeMs ?? -Infinity) >= startAfter);
	if (startBefore != null) runs = runs.filter((run) => (run.startTimeMs ?? Infinity) <= startBefore);
	if (endAfter != null) runs = runs.filter((run) => (run.endTimeMs ?? -Infinity) >= endAfter);
	if (endBefore != null) runs = runs.filter((run) => (run.endTimeMs ?? Infinity) <= endBefore);

	if (input.paramFilters?.length) {
		runs = runs.filter((run) => input.paramFilters!.every((filter) => passesParamFilter(run, filter)));
	}

	if (input.metric?.name) {
		const min = input.metric.min;
		const max = input.metric.max;
		runs = runs.filter((run) => {
			const value = metricValueForRun(run, input.metric?.name);
			if (value == null) return false;
			if (min != null && value < min) return false;
			if (max != null && value > max) return false;
			return true;
		});
	}

	const requestedSortBy = input.sort?.by ?? (input.metric?.name ? "metric" : input.latestBy === "end_time" ? "end_time" : "start_time");
	const requestedOrder: SortOrder = input.sort?.order ?? input.metric?.sort ?? "desc";

	runs.sort((a, b) => {
		let cmp = 0;
		if (requestedSortBy === "metric") {
			const av = metricValueForRun(a, input.metric?.name ?? "") ?? -Infinity;
			const bv = metricValueForRun(b, input.metric?.name ?? "") ?? -Infinity;
			cmp = av - bv;
		} else if (requestedSortBy === "run_name") {
			cmp = (a.runName ?? "").localeCompare(b.runName ?? "");
		} else if (requestedSortBy === "end_time") {
			cmp = (a.endTimeMs ?? -Infinity) - (b.endTimeMs ?? -Infinity);
		} else {
			cmp = (a.startTimeMs ?? -Infinity) - (b.startTimeMs ?? -Infinity);
		}

		if (cmp === 0) {
			cmp = (a.startTimeMs ?? -Infinity) - (b.startTimeMs ?? -Infinity);
		}
		return requestedOrder === "asc" ? cmp : -cmp;
	});

	const limit = Math.min(Math.max(input.limit ?? 10, 1), MAX_LIMIT);
	return runs.slice(0, limit);
}

function formatRun(run: RunRecord, requestedMetric?: string): string {
	const lines: string[] = [];
	lines.push(`Run ID: ${run.runId}`);
	lines.push(`Run name: ${run.runName || "n/a"}`);
	lines.push(`Experiment: ${run.experimentName || "n/a"} (${run.experimentId})`);
	lines.push(`Status: ${run.status || "n/a"}`);
	lines.push(`Start: ${formatTime(run.startTimeMs)}`);
	lines.push(`End: ${formatTime(run.endTimeMs)}`);
	if (requestedMetric) {
		const mv = metricValueForRun(run, requestedMetric);
		lines.push(`Requested metric (${requestedMetric}): ${mv == null ? "n/a" : mv}`);
	}

	lines.push("Parameters:");
	if (run.params.size === 0) {
		lines.push("  (none)");
	} else {
		for (const [key, value] of [...run.params.entries()].sort(([a], [b]) => a.localeCompare(b))) {
			lines.push(`  - ${key}: ${value}`);
		}
	}

	lines.push("Metrics:");
	if (run.metrics.size === 0) {
		lines.push("  (none)");
	} else {
		for (const [key, value] of [...run.metrics.entries()].sort(([a], [b]) => a.localeCompare(b))) {
			lines.push(`  - ${key}: ${value}`);
		}
	}

	return lines.join("\n");
}

const findMlrunsRunsTool = defineTool({
	name: "find_mlruns_runs",
	label: "Find MLflow Runs",
	description:
		"Find runs in local mlruns/ by time, metric, and parameter filters, then return readable multi-run reports with full params and metrics.",
	parameters: Type.Object({
		mlrunsPath: Type.Optional(Type.String({ description: "Path to mlruns directory (default: ./mlruns)" })),
		experimentId: Type.Optional(Type.String({ description: "Only include this MLflow experiment id" })),
		experimentName: Type.Optional(Type.String({ description: "Only include this experiment name" })),
		latestBy: Type.Optional(Type.Union([Type.Literal("start_time"), Type.Literal("end_time")])),
		startAfter: Type.Optional(Type.Union([Type.String(), Type.Number()])),
		startBefore: Type.Optional(Type.Union([Type.String(), Type.Number()])),
		endAfter: Type.Optional(Type.Union([Type.String(), Type.Number()])),
		endBefore: Type.Optional(Type.Union([Type.String(), Type.Number()])),
		metric: Type.Optional(
			Type.Object({
				name: Type.String({ description: "Metric key to rank/filter by (alias tolerant)" }),
				sort: Type.Optional(Type.Union([Type.Literal("asc"), Type.Literal("desc")], { default: "desc" })),
				min: Type.Optional(Type.Number()),
				max: Type.Optional(Type.Number()),
			}),
		),
		paramFilters: Type.Optional(
			Type.Array(
				Type.Object({
					name: Type.String({ description: "Parameter key (alias tolerant)" }),
					op: Type.Optional(
						Type.Union(
							[
								Type.Literal("eq"),
								Type.Literal("ne"),
								Type.Literal("gt"),
								Type.Literal("gte"),
								Type.Literal("lt"),
								Type.Literal("lte"),
								Type.Literal("contains"),
							],
							{ default: "eq" },
						),
					),
					value: Type.Union([Type.String(), Type.Number()]),
				}),
			),
		),
		sort: Type.Optional(
			Type.Object({
				by: Type.Union([
					Type.Literal("start_time"),
					Type.Literal("end_time"),
					Type.Literal("run_name"),
					Type.Literal("metric"),
				]),
				order: Type.Optional(Type.Union([Type.Literal("asc"), Type.Literal("desc")], { default: "desc" })),
			}),
		),
		limit: Type.Optional(Type.Number({ minimum: 1, maximum: MAX_LIMIT, description: `Max runs to return (1-${MAX_LIMIT})` })),
	}),

	async execute(_toolCallId, params: QueryInput, _signal, _onUpdate, ctx) {
		const runs = queryRuns(params, ctx.cwd);

		if (runs.length === 0) {
			return {
				content: [{ type: "text", text: "No matching MLflow runs were found in the requested scope." }],
				details: { count: 0, query: params },
			};
		}

		const rendered = [
			`Found ${runs.length} matching run(s).`,
			...runs.map((run, index) => `\n=== Run ${index + 1} ===\n${formatRun(run, params.metric?.name)}`),
		].join("\n");

		return {
			content: [{ type: "text", text: rendered }],
			details: {
				count: runs.length,
				runs: runs.map((run) => ({
					runId: run.runId,
					runName: run.runName,
					experimentId: run.experimentId,
					experimentName: run.experimentName,
					status: run.status,
					startTimeMs: run.startTimeMs,
					endTimeMs: run.endTimeMs,
					params: Object.fromEntries(run.params),
					metrics: Object.fromEntries(run.metrics),
				})),
				query: params,
			},
		};
	},
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(findMlrunsRunsTool);
}
