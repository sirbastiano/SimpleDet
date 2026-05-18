# Docs and CI checklist

- [ ] Run `make docs-audit` locally (or confirm no docs changes were made).
- [ ] Verify all added/modified command examples still match implementation.
- [ ] Confirm documentation links resolve: internal `.html` paths, anchors, local media, and nav links.
- [ ] Confirm any updated command output snippets were regenerated from a current run.
- [ ] Confirm image assets in docs have non-empty `alt` attributes (where applicable).
