TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))
CHECK_RERUN = grep Rerun ms.log

default: rvhmc.pdf

rvhmc.pdf: rvhmc.tex
	tectonic rvhmc.tex
	#pdflatex ms
	#bibtex ms
	#pdflatex ms
	#pdflatex ms

clean:
	$(RM_TMP)
