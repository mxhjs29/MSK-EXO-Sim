## Summary

Describe the change and why it is needed.

## Validation

- [ ] `make lint`
- [ ] `make test`
- [ ] `make ci` or equivalent targeted checks

List the commands you actually ran:

```bash
# Paste validation commands here
```

## Checklist

- [ ] Tests were added or updated when behavior changed
- [ ] Docs were updated when user-facing workflows changed
- [ ] Large formatting-only changes were kept separate from functional changes
- [ ] `python scripts/check_repository_scope.py` passes; no datasets or outputs included
- [ ] Task rewards and physics changes are identified separately from shared PPO code
- [ ] The `.pre-commit-config.yaml` allowlist was left unchanged, or the scope change is isolated and justified

## Related Issues

Link issues, design notes, or discussion threads here.
