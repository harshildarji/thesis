echo "- Running PdfLaTeX..."
pdflatex --shell-escape thesis.tex > /dev/null 2>&1

echo "- Executing Biber..."
biber thesis > /dev/null 2>&1

echo "- Running PdfLaTeX (again"'!'")..."
pdflatex --shell-escape thesis.tex > /dev/null 2>&1

echo "- Running PdfLaTeX (last time, I promise"'!'")..."
pdflatex --shell-escape thesis.tex > /dev/null 2>&1

xdg-open thesis.pdf
