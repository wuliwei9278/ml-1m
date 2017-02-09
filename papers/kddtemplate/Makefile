.SUFFIXES: .tex .in .dvi .ps .pdf

all:  kdd17.pdf kdd17.pdf

.tex.pdf: 
	pdflatex $<
	-bibtex $*
	pdflatex $<
	pdflatex $<
	pdflatex $<
	
clean:
	rm -f  *.aux *.dvi *.ps *.bbl *.blg *.out *.pdf *.toc *.nav *.snm *.bak

