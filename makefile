projfiles=$(shell ls *.tex)
mainfile=gpu_programming_bpg.tex

all: clean $(projfiles)
	latexmk -pdflatex='pdflatex -file-line-error -synctex=1' -pdf -f $(mainfile)

clean:
	@echo cleaning...
	- rm *.aux
	- rm *.bbl
	- rm *.blg
	- rm *.fdb_latexmk
	- rm *.fls
	- rm *.log
	- rm *.out
	- rm *.pdf
	- rm *.dvi
	- rm *.synctex.gz
	- rm *.toc
	- rm *Notes.bib
