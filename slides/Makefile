LATEX="rubber"
CFLAGS="--pdf"
RMFLAGS="--clean"

all: slides 

clean: slides.tex
	$(LATEX) $(RMFLAGS) $<

slides: slides.tex
	$(LATEX) $(CFLAGS) $<
