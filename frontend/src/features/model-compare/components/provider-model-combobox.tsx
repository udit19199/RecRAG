import {
	Combobox,
	ComboboxCollection,
	ComboboxContent,
	ComboboxEmpty,
	ComboboxGroup,
	ComboboxInput,
	ComboboxItem,
	ComboboxLabel,
	ComboboxList,
	ComboboxSeparator,
} from "@/components/ui/combobox";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import {
	type ProviderModelGroup,
	modelValueLabel,
	parseValue,
} from "@/features/model-compare/lib";

interface ProviderModelComboboxProps {
	groups: ProviderModelGroup[];
	value: string | null | undefined;
	onValueChange: (value: string | null) => void;
	allowClear?: boolean;
	showUnavailable?: boolean;
	disabled?: boolean;
	placeholder?: string;
	inputClassName?: string;
	readOnly?: boolean;
}

const CLEAR_VALUE = "";

export function ProviderModelCombobox({
	groups,
	value,
	onValueChange,
	allowClear = false,
	showUnavailable = false,
	disabled = false,
	placeholder = "Select model",
	inputClassName,
	readOnly = false,
}: ProviderModelComboboxProps) {
	const handleValueChange = (
		nextValue: string | (string | null)[] | null | undefined,
	) => {
		if (typeof nextValue === "string") {
			onValueChange(nextValue || null);
			return;
		}
		if (nextValue === null) {
			onValueChange(null);
		}
	};

	if (allowClear) {
		const flatItems = [
			CLEAR_VALUE,
			...groups.flatMap((group) => (group.available ? group.items : [])),
		];

		return (
			<Combobox
				items={flatItems}
				value={value ?? null}
				onValueChange={handleValueChange}
				itemToStringLabel={modelValueLabel}
				disabled={disabled}
			>
				<ComboboxInput
					placeholder={placeholder}
					readOnly={readOnly}
					className={inputClassName}
				/>
				<ComboboxContent>
					<ComboboxEmpty>No matching models.</ComboboxEmpty>
					<ComboboxList>
						{(item) => (
							<ComboboxItem
								key={item || "__none__"}
								value={item}
								className={
									item === CLEAR_VALUE ? "text-muted-foreground" : undefined
								}
							>
								{modelValueLabel(item)}
							</ComboboxItem>
						)}
					</ComboboxList>
				</ComboboxContent>
			</Combobox>
		);
	}

	return (
		<Combobox
			items={groups}
			value={value ?? null}
			onValueChange={handleValueChange}
			itemToStringLabel={(item) =>
				typeof item === "string" ? modelValueLabel(item) : ""
			}
			disabled={disabled}
		>
			<ComboboxInput
				placeholder={placeholder}
				readOnly={readOnly}
				className={inputClassName}
			/>
			<ComboboxContent>
				<ComboboxEmpty>No matching models.</ComboboxEmpty>
				<ComboboxList>
					{(group, index) => {
						if (!group.available) {
							if (!showUnavailable) return null;
							return (
								<ComboboxGroup key={group.key}>
									<ComboboxLabel className="flex items-center justify-between">
										<span>{group.label}</span>
										<Tooltip>
											<TooltipTrigger asChild>
												<span className="ml-2 cursor-default text-xs text-muted-foreground">
													unavailable
												</span>
											</TooltipTrigger>
											<TooltipContent side="right">
												<p className="max-w-xs text-xs">
													{group.reason || "Provider not available"}
												</p>
											</TooltipContent>
										</Tooltip>
									</ComboboxLabel>
								</ComboboxGroup>
							);
						}

						if (group.items.length === 0) return null;

						return (
							<ComboboxGroup key={group.key} items={group.items}>
								<ComboboxLabel>{group.label}</ComboboxLabel>
								<ComboboxCollection>
									{(item) => (
										<ComboboxItem key={item} value={item}>
											{parseValue(item)?.model ?? item}
										</ComboboxItem>
									)}
								</ComboboxCollection>
								{index < groups.length - 1 ? <ComboboxSeparator /> : null}
							</ComboboxGroup>
						);
					}}
				</ComboboxList>
			</ComboboxContent>
		</Combobox>
	);
}
