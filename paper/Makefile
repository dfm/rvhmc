TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))
CHECK_RERUN = grep Rerun ms.log

NAME = rvhmc

default: ${NAME}.pdf

${NAME}.aux: ${NAME}.tex
	pdflatex ${NAME}

${NAME}.bbl: ${NAME}.aux ${NAME}.bib
	bibtex ${NAME}

${NAME}.pdf: ${NAME}.bbl
	pdflatex ${NAME}
	pdflatex ${NAME}

#rvhmc.pdf: rvhmc.tex
#    #tectonic rvhmc.tex
#    pdflatex rvhmc
#    bibtex rvhmc
#    pdflatex rvhmc
#    pdflatex rvhmc

clean:
	$(RM_TMP)
