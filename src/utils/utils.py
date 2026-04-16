def check_for_missing_and_extra_elements(required: set, provided: set) -> tuple[set, set]:
    missing = required - provided
    extra = provided - required
    return missing, extra
