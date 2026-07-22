lualatex -interaction=nonstopmode -output-directory ./docs/assets/ docs/tex/dryml_definition_graph.tex
pdf2svg docs/assets/dryml_definition_graph.pdf docs/assets/dryml_definition_graph.svg

lualatex -interaction=nonstopmode -output-directory ./docs/assets/ docs/tex/dryml_experiment_concrete_definition.tex
pdf2svg docs/assets/dryml_experiment_concrete_definition.pdf docs/assets/dryml_experiment_concrete_definition.svg
