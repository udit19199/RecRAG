"use client";

import { useReducer } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import type { Requirements } from "@/lib/api/orchestrator";
import { createRun } from "@/lib/api/orchestrator";

const STEP_COUNT = 5;

type WizardFields = {
	use_case?: string;
	audience?: string;
	document_modality?: string;
	budget_monthly_usd?: number;
	citations_required?: boolean;
};

type State = WizardFields & { step: number };

const INITIAL: State = { step: 0 };

function toRequirements(state: State, skipCurrentStep: boolean): Requirements {
	const { step, ...fields } = state;
	const requirements: Requirements = {
		use_case: fields.use_case?.trim() || "Unspecified",
	};
	if (fields.audience?.trim()) {
		requirements.audience = fields.audience.trim();
	}
	if (fields.document_modality) {
		requirements.document_modality = fields.document_modality;
	}
	if (
		fields.budget_monthly_usd != null &&
		!Number.isNaN(fields.budget_monthly_usd)
	) {
		requirements.budget_monthly_usd = fields.budget_monthly_usd;
	}
	if (step === 4 && !skipCurrentStep) {
		requirements.citations_required = fields.citations_required ?? false;
	}
	return requirements;
}

export function RequirementsWizard() {
	const router = useRouter();
	const [state, dispatch] = useReducer(
		(s: State, patch: Partial<State>) => ({ ...s, ...patch }),
		INITIAL,
	);

	const isLastStep = state.step === STEP_COUNT - 1;
	const next = () => dispatch({ step: state.step + 1 });
	const prev = () => dispatch({ step: Math.max(0, state.step - 1) });

	const finish = async (skipCurrentStep = false) => {
		const { run_id } = await createRun(toRequirements(state, skipCurrentStep));
		router.push(`/recommend?run_id=${run_id}`);
	};

	return (
		<div className="flex flex-col gap-6">
			<p className="text-sm text-muted-foreground">
				Step {state.step + 1} of {STEP_COUNT}
			</p>

			{state.step === 0 && (
				<FieldGroup>
					<Field>
						<FieldLabel>Use case</FieldLabel>
						<p className="mb-2 text-sm text-muted-foreground">
							What will this RAG system be used for?
						</p>
						<Textarea
							value={state.use_case ?? ""}
							onChange={(e) => dispatch({ use_case: e.target.value })}
							placeholder="e.g. Internal policy search for a law firm"
						/>
					</Field>
				</FieldGroup>
			)}
			{state.step === 1 && (
				<FieldGroup>
					<Field>
						<FieldLabel>Audience</FieldLabel>
						<p className="mb-2 text-sm text-muted-foreground">
							Who will use it — role, expertise, or team size?
						</p>
						<Input
							value={state.audience ?? ""}
							onChange={(e) => dispatch({ audience: e.target.value })}
							placeholder="e.g. Junior legal staff"
						/>
					</Field>
				</FieldGroup>
			)}
			{state.step === 2 && (
				<FieldGroup>
					<Field>
						<FieldLabel>Document modality</FieldLabel>
						<p className="mb-2 text-sm text-muted-foreground">
							What kind of content will be queried?
						</p>
						<select
							className="w-full rounded-md border px-3 py-2 text-sm"
							value={state.document_modality ?? ""}
							onChange={(e) =>
								dispatch({
									document_modality: e.target.value || undefined,
								})
							}
						>
							<option value="">Not specified</option>
							<option value="text_heavy">Text-heavy PDFs</option>
							<option value="scanned">Scanned documents</option>
							<option value="image_heavy">Image / chart heavy</option>
							<option value="mixed">Mixed</option>
						</select>
					</Field>
				</FieldGroup>
			)}
			{state.step === 3 && (
				<FieldGroup>
					<Field>
						<FieldLabel>Monthly budget (USD)</FieldLabel>
						<p className="mb-2 text-sm text-muted-foreground">
							Monthly spend ceiling for tokens, cloud, and storage.
						</p>
						<Input
							type="number"
							min={0}
							value={state.budget_monthly_usd ?? ""}
							onChange={(e) =>
								dispatch({
									budget_monthly_usd: e.target.value
										? Number(e.target.value)
										: undefined,
								})
							}
							placeholder="e.g. 1500"
						/>
					</Field>
				</FieldGroup>
			)}
			{state.step === 4 && (
				<FieldGroup>
					<Field className="flex flex-row items-center justify-between gap-4">
						<div>
							<FieldLabel>Citations required</FieldLabel>
							<p className="text-sm text-muted-foreground">
								Answers must cite source documents.
							</p>
						</div>
						<Switch
							checked={state.citations_required ?? false}
							onCheckedChange={(v) => dispatch({ citations_required: v })}
						/>
					</Field>
				</FieldGroup>
			)}

			<div className="flex items-center justify-between gap-3">
				<Button
					type="button"
					variant="ghost"
					onClick={prev}
					disabled={state.step === 0}
				>
					Back
				</Button>
				<div className="flex gap-2">
					{isLastStep ? (
						<>
							<Button
								type="button"
								variant="ghost"
								onClick={() => finish(true)}
							>
								Skip
							</Button>
							<Button type="button" onClick={() => finish(false)}>
								Get recommendation
							</Button>
						</>
					) : (
						<>
							<Button type="button" variant="ghost" onClick={next}>
								Skip
							</Button>
							<Button type="button" onClick={next}>
								Next
							</Button>
						</>
					)}
				</div>
			</div>
		</div>
	);
}
