# Clause identification

1. Import nltk for tokenization, POS tagging etc.

2. Download and install ghostscript

   i) To visualize each word's POS tags in the form of a tree or to draw a parse tree of the sentence, We need to install ghostscript in 
   case nltk throws an error.
   
   ii) We are modifying environment variable to use ghostscript, we have to add to the path manually (for windows)
   > os.environ['PATH']+=os.pathsep+"C:\\Program Files\\gs\\gs9.50\\bin" 
   
3. We define tree representation of syntactic structure of sentence. For that we use constituency Parser.
   
   i) A constituency parser has been built based on such grammars/rules, which are usually collectively available as context-free grammar      (CFG). This parser processes input sentences according to these rules and builds a parse tree. We used nltk to generate parse tree.
   
   ii) Constituent-based grammars analyze and determine the constituents of a sentence. It represent the internal structure of sentences         in terms of a hierarchically ordered structure of their constituents.
   
   iii) We can create general CFG(context free grammar) by adding more POS tags further to analyze sentence structure but here we have        created grammar with three postags NNP, VBD/VBN or NN for sentences having postags only these one.
   
4. If your tree whether it is in string format or not. To avoid doubt, we use below method to parse a tree from a string tree with parentheses.
> Tree.fromstring(str(tree)) 

5. Using this constructed CFG and parse tree, we identify clauses from a sentence or written text and we print them out.
