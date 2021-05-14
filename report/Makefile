
# ---------------------------
#          Makefile 
# ---------------------------

TARGET := main

all : $(TARGET).pdf

clean : 
	rm *.log
	rm *.lot
	rm *.lof
	rm *.toc
	rm *.blg
	rm *.bbl
	rm *.aux
	rm *.dvi 

bib : 
	bibtex $(TARGET)	
#	for i in 1 2; do touch $(TARGET).tex; make -j10; done

bibc : 
	touch $(TARGET).tex
	make -j10

%.pdf : %.dvi
	dvipdfmx $<
	open $(TARGET).pdf

%.dvi : %.tex ./section/*.tex 
	uplatex $<

