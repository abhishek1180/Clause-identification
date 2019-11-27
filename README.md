# Clause identification

## Import nltk for tokenization, POS tagging etc.

## Download and install ghostscript
1.To visualize each word's POS tags in the form of a tree or to draw a parse tree of the sentence, We need to install ghostscript in case nltk throws an error.

2.We are modifying environment variable to use ghostscript, we have to add to the path manually (for windows)
> os.environ['PATH']+=os.pathsep+"C:\\Program Files\\gs\\gs9.50\\bin" 
