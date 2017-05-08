#!/usr/bin/Rscript

# library(knitr)
library(rmarkdown)
#knit('StatsNotes.Rmd')
#pandoc('StatsNotes.md',format='latex')
#markdownToHTML('StatsNotes.md','StatsNotes.html')
# render('StatsNotes.Rmd',output_format='all')
render('Titanic.Rmd',output_format='all')
# knit('MachineLearning.Rmd')
# pandoc('MachineLearning.md',format='latex')
# markdownToHTML('MachineLearning.md','MachineLearning.html')
# knit2html('PA1_template.Rmd','PA1_template.html')


