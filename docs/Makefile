# Minimal Makefile for Sphinx documentation

SPHINXOPTS ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = _build
APIGENDIR = api/generated

.PHONY: help clean html html-strict rebuild

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" "$(APIGENDIR)"

html:
	@mkdir -p "$(APIGENDIR)"
	@$(SPHINXBUILD) -E -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

html-strict:
	@mkdir -p "$(APIGENDIR)"
	@$(SPHINXBUILD) -E -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

rebuild: clean html
