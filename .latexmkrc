$pdf_mode = 5;            # 5 = xelatex
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;
$biber = 'biber %O %S';
$out_dir = 'build';
$clean_ext = 'bbl run.xml bcf synctex.gz';
