# how to make these figures


```
xelatex SYMPTOM-CHORD.tex
xelatex SYMPTOM-TERNARY.tex
xelatex FIGURE-5.tex
magick -density 600 FIGURE-5.pdf -background white -alpha remove -alpha off FIGURE-5.png
```

