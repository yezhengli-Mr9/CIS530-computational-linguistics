# CIS530-HW3-Vector-space-models-Computational-Linguistics

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Homework 3 "Vector Space Models"</title>
  </head>
<body>
    <a class="sr-only" href="#content">Skip to main content</a>

<div class="alert alert-info">
This assignment is due before 11:00AM on Wednesday, January 31, 2018.
</div>

<h1 id="vector-space-models--assignment-3">Vector Space Models <span class="text-muted">: Assignment 3</span></h1>

<p>In this assignment you will implement many of the things you learned in <a href="https://web.stanford.edu/~jurafsky/slp3/15.pdf">Chapter 15 of the textbook</a>. If you haven’t read it yet, now would be a good time to do that.  We’ll wait.  Done?  Great, let’s move on.</p>

<p>We will provide a corpus of Shakespeare plays, which you will use to create a term-document matrix and a term-context matrix. You’ll implement a selection of the weighting methods and similarity metrics defined in the textbook. Ultimately, your goal is to use the resulting vectors to measure how similar Shakespeare plays are to each other, and to find words that are used in a similar fashion. All (or almost all) of the code you write will be direct implementations of concepts and equations described in <a href="https://web.stanford.edu/~jurafsky/slp3/15.pdf">Chapter 15</a>.</p>

<p><em>All difficulties are easy when they are known.</em></p>

<div class="alert alert-info">
  <p>Here are the materials that you should download for this assignment:</p>
  <ul>
    <li><a href="/downloads/hw3/main.py">Skeleton python code</a></li>
    <li><a href="/downloads/hw3/will_play_text.csv">Data - csv of the complete works of Shakespeare</a></li>
    <li><a href="/downloads/hw3/vocab.txt">Data - vocab the complete works of Shakespeare</a></li>
    <li><a href="/downloads/hw3/play_names.txt">Data - list of all plays in dataset</a></li>
  </ul>
</div>

<h1 id="term-document-matrix">Term-Document Matrix</h1>

<p>You will write code to compile a term-document matrix for Shakespeare’s plays, following the description in section 15.1.1 in textbook.</p>

<blockquote>
  <p>In a <em>term-document matrix</em>, each row represents a word in the vocabulary and each column represents a document from some collection. The figure below shows a small selection from a term-document matrix showing the occurrence of four words in four plays by Shakespeare. Each cell in this matrix represents the number of times a particular word (defined by the row) occurs in a particular document (defined by the column). Thus <em>clown</em> appeared 117 times in *Twelfth Night</p>
</blockquote>

<table>
  <thead>
    <tr>
      <th style="text-align: center"> </th>
      <th style="text-align: center">As You Like It</th>
      <th style="text-align: center">Twelfth Night</th>
      <th style="text-align: center">Julias Caesar</th>
      <th style="text-align: center">Henry V</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><strong>battle</strong></td>
      <td style="text-align: center">1</td>
      <td style="text-align: center">1</td>
      <td style="text-align: center">8</td>
      <td style="text-align: center">15</td>
    </tr>
    <tr>
      <td style="text-align: center"><strong>soldier</strong></td>
      <td style="text-align: center">2</td>
      <td style="text-align: center">2</td>
      <td style="text-align: center">12</td>
      <td style="text-align: center">36</td>
    </tr>
    <tr>
      <td style="text-align: center"><strong>fool</strong></td>
      <td style="text-align: center">37</td>
      <td style="text-align: center">58</td>
      <td style="text-align: center">1</td>
      <td style="text-align: center">5</td>
    </tr>
    <tr>
      <td style="text-align: center"><strong>crown</strong></td>
      <td style="text-align: center">5</td>
      <td style="text-align: center">117</td>
      <td style="text-align: center">0</td>
      <td style="text-align: center">0</td>
    </tr>
  </tbody>
</table>

<p>The dimensions of your term-document matrix will be the number of documents $D$ (in this case, the number of Shakespeare’s plays that we give you in the corpus by the number of unique word types $|V|$ in that collection.   The columns represent the documents, and the rows represent the words, and each cell represents the frequency of that word in that document.</p>

<p>In your code you will write a function to <code class="highlighter-rouge">create_term_document_matrix</code>.  This will let you be the hit of your next dinner party by being able to answer trivia questions like <em>how many words did Shakespeare use?</em>, which may give us a hint to the answer to <em>[How many words did Shakespeare know?]</em>  The table will also tell you how many words Shakespeare used only once.  Did you know that there’s a technical term for that?  In corpus linguistics they are called <a href="https://en.wikipedia.org/wiki/Hapax_legomenon"><em>hapax legomena</em></a>, but I prefer the term <em>singleton</em>, because I don’t like snooty Greek or Latin terms.</p>

<h2 id="comparing-plays">Comparing plays</h2>

<p>The term-document matrix will also let us do cool things like figure out which plays are most similar to each other, by comparing the column vectors.  We could even look for outliers to see if some plays are so dissimilar from the rest of the canon that <a href="https://en.wikipedia.org/wiki/Shakespeare_authorship_question">maybe they weren’t authored by Shakespeare after all</a>.</p>

<p>Let’s begin by considering the column representing each play.  Each column is a $|V|$-dimensional vector.  Let’s use some math to define the similarity of these vectors.   By far the most common similarity metric is the cosine of the angle between the vectors.  The cosine similarity metric is defined in Section 15.3 of the textbook.</p>

<blockquote>
  <p>The cosine, like most measures for vector similarity used in NLP, is based on the dot product operator from linear algebra, also called the inner product:</p>
</blockquote>

<blockquote>
  <p>dot-product($\vec{v}, \vec{w}) = \vec{v} \cdot \vec{w} = \sum_{i=1}^{N}{v_iw_i} = v_1w_1 +v_2w_2 +…+v_Nw_N$</p>
</blockquote>

<blockquote>
  <p>The dot product acts as a similarity metric because it will tend to be high just when the two vectors have large values in the same dimensions. Alternatively, vectors that have zeros in different dimensions (orthogonal vectors) will have a dot product of 0, representing their strong dissimilarity.</p>
</blockquote>

<blockquote>
  <p>This raw dot-product, however, has a problem as a similarity metric: it favors long vectors. The vector length is defined as</p>
</blockquote>

<blockquote>
  <p>$|\vec{v}| = \sqrt{\sum_{i=1}^{N}{v_i^2}}$</p>
</blockquote>

<blockquote>
  <p>The dot product is higher if a vector is longer, with higher values in each dimension. More frequent words have longer vectors, since they tend to co-occur with more words and have higher co-occurrence values with each of them. The raw dot product thus will be higher for frequent words. But this is a problem; we would like a similarity metric that tells us how similar two words are regardless of their frequency.</p>
</blockquote>

<blockquote>
  <p>The simplest way to modify the dot product to normalize for the vector length is to divide the dot product by the lengths of each of the two vectors. This normalized dot product turns out to be the same as the cosine of the angle between the two vectors, following from the definition of the dot product between two vectors $\vec{v}$ and $\vec{w}$ as:</p>
</blockquote>

<blockquote>
  <p>$\vec{v} \cdot \vec{w} = |\vec{v}||\vec{w}| cos \Theta$</p>
</blockquote>

<blockquote>
  <p>$\frac{\vec{v} \cdot \vec{w}}{|\vec{v}||\vec{w}|} =  cos \Theta$</p>
</blockquote>

<blockquote>
  <p>The cosine similarity metric between two vectors $\vec{v}$ and $\vec{w}$ thus can be computed</p>
</blockquote>

<blockquote>
  <p>$cosine(\vec{v}, \vec{w}) = \frac{\vec{v} \cdot \vec{w}}{|\vec{v}| |\vec{w}|} = \frac{\sum_{i=1}^{N}{v_iw_i}}{\sqrt{\sum_{i=1}^{N}{v_i^2}} \sqrt{\sum_{i=1}^{N}{w_i^2}}} $</p>
</blockquote>

<p>The cosine value ranges from 1 for vectors pointing in the same direction, through 0 for vectors that are orthogonal, to -1 for vectors pointing in opposite directions. Since our term-document matrix contains raw frequency counts, it is non-negative, so the cosine for its vectors will range from 0 to 1.  1 means that the vectors are identical, 0 means that they are totally dissimilar.</p>

<p>Please implement <code class="highlighter-rouge">compute_cosine_similarity</code>, and for each play in the corpus, score how similar each other play is to it.  Which plays are the closet to each other in vector space (ignoring self similarity)?  Which plays are the most distant from each other?</p>

<h2 id="how-do-i-know-if-my-rankings-are-good">How do I know if my rankings are good?</h2>

<p>First, read all of the plays. Then perform at least three of them. Now that you are a true thespian, you should have a good intuition for the central themes in the plays.   Alternately, take a look at <a href="https://en.wikipedia.org/wiki/Shakespeare%27s_plays#Canonical_plays">this grouping of Shakespeare’s plays into Tragedies, Comedies and Histories</a>. Do plays that are thematically similar to the one that you’re ranking appear among its most similar plays, according to cosine similarity? Another clue that you’re doing the right thing is if a play has a cosine of 1 with itself.  If that’s not the case, then you’ve messed something up. Another good hint, is that there are a ton of plays about Henry.  They’ll probably be similar to each other.</p>

<h1 id="measuring-word-similarity">Measuring word similarity</h1>

<p>Next, we’re going to see how we can represent words as vectors in vector space.  This will give us a way of representing some aspects of the <em>meaning</em> of words, by measuring the similarity of their vectors.</p>

<p>In our term-document matrix, the rows are word vectors.  Instead of a $|V|$-dimensional vector, these row vectors only have $D$ dimensions.  Do you think that’s enough to represent the meaning of words?  Try it out.  In the same way that you computed the similarity of the plays, you can compute the similarity of the words in the matrix.  Pick some words and compute 10 words with the highest cosine similarity between their row vector representations.  Are those 10 words good synonyms?</p>

<h2 id="term-context-matrix">Term-Context Matrix</h2>

<p>Instead of using a term-document matrix, a more common way of computing word similarity is by constructing a term-context matrix (also called a word-word matrix), where columns are labeled by words rather than documents.  The dimensionality of this kind of a matrix is $|V|$ by $|V|$.  Each cell represents how often the word in the row (the target word) co-occurs with the word in the column (the context) in a training corpus.</p>

<p>For this part of the assignment, you should write the <code class="highlighter-rouge">create_term_context_matrix</code> function.  This function specifies the size word window around the target word that you will use to gather its contexts.  For instance, if you set that variable to be 4, then you will use 4 words to the left of the target word, and 4 words to its right for the context.  In this case, the cell represents the number of times in Shakespeare’s plays the column word occurs in +/-4 word window around the row word.</p>

<p>You can now re-compute the most similar words for your test words using the row vectors in your term-context matrix instead of your term-document matrix.  What is the dimensionality of your word vectors now?  Do the most similar words make more sense than before?</p>

<h1 id="weighting-terms">Weighting terms</h1>

<p>Your term-context matrix contains the raw frequency of the co-occurrence of two words in each cell.  Raw frequency turns out not to be the best way of measuring the association between words.  There are several methods for weighting words so that we get better results.  You should implement two weighting schemes:</p>

<ul>
  <li>Positive pointwise mutual information (PPMI)</li>
  <li>Term frequency inverse document frequency (tf-idf)</li>
</ul>

<p>These are defined in Section 15.2 of the textbook.</p>

<p><em>Warning, calculating PPMI for your whole $|V|$-by-$|V|$ matrix might be slow. Our intrepid TA’s implementation for PPMI takes about 10 minutes to compute all values. She always writes perfectly optimized code on her first try. You may improve performance by using matrix operations a la MATLAB.</em></p>

<h1 id="weighting-terms-1">Weighting terms</h1>

<p>There are several ways of computing the similarity between two vectors.  In addition to writing a function to compute cosine similarity, you should also write functions to <code class="highlighter-rouge">compute_jaccard_similarity</code> and <code class="highlighter-rouge">compute_dice_similarity</code>.  Check out section 15.3.1. of the textbook for the defintions of the Jaccard and Dice measures.</p>

<h1 id="your-tasks">Your Tasks</h1>

<p>All of the following are function stubs in the python code. You just need to fill them out.</p>

<p>Create matrices:</p>
<ul>
  <li>fill out <code class="highlighter-rouge">create_term_document_matrix</code></li>
  <li>fill out <code class="highlighter-rouge">create_term_context_matrix</code></li>
  <li>fill out <code class="highlighter-rouge">create_PPMI_matrix</code></li>
  <li>fill out <code class="highlighter-rouge">compute_tf_idf_matrix</code></li>
</ul>

<p>Compute similarities:</p>
<ul>
  <li>fill out <code class="highlighter-rouge">compute_cosine_similarity</code></li>
  <li>fill out <code class="highlighter-rouge">compute_jaccard_similarity</code></li>
  <li>fill out <code class="highlighter-rouge">compute_dice_similarity</code></li>
</ul>

<p>Do some ranking:</p>
<ul>
  <li>fill out <code class="highlighter-rouge">rank_plays</code></li>
  <li>fill out <code class="highlighter-rouge">rank_words</code></li>
</ul>

<h1 id="report">Report</h1>

<p>In the ranking tasks, play with different vector representations and different similarity functions. Does one combination appear to work better than another? Do any interesting patterns emerge? Include this discussion in your writeup.</p>

<p>Some patterns you could touch upon:</p>
<ul>
  <li>The fourth column of <code class="highlighter-rouge">will_play_text.csv</code> contains the name of the character who spoke each line. Using the methods described above, which characters are most similar? Least similar?</li>
  <li>Shakespeare’s plays are traditionally classified into <a href="https://en.wikipedia.org/wiki/Shakespeare%27s_plays">comedies, histories, and tragedies</a>. Can you use these vector representations to cluster the plays?</li>
  <li>Do the vector representations of <a href="https://en.wikipedia.org/wiki/Category:Female_Shakespearean_characters">female characters</a> differ distinguishably from <a href="https://en.wikipedia.org/wiki/Category:Male_Shakespearean_characters">male ones</a>?</li>
</ul>

<h1 id="extra-credit">Extra credit</h1>

<p>Quantifying the goodness of one vector space representation over another can be very difficult to do.  It might ultimately require testing how the different vector representations change the performance when used in a downstream task like question answering. A common way of quantifying the goodness of word vectors is to use them to compare the similarity of words with human similarity judgments, and then calculate the correlation of the two rankings.</p>

<p>If you would like extra credit on this assignment, you can quantify the goodness of each of the different vector space models that you produced (for instance by varying the size of the context window, picking PPMI or tf-idf, and selecting among cosine, Jaccard, and Dice).  You can calculate their scores on the <a href="https://www.cl.cam.ac.uk/~fh295/simlex.html">SimLex999 data set</a>, and compute their correlation with human judgments using <a href="https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient">Kendall’s Tau</a>.</p>

<p>Add a section to your writeup explaining what experiments you ran, and which setting had the highest correlation with human judgments.</p>

<h1 id="more-optional-fun-extra-credit-options">More Optional Fun Extra Credit options</h1>

<p>So you’ve built some machinery that can measure similarity between words and documents. We gave you a Shakespeare corpus, but you can use any body of text you like. For example, check out <a href="https://www.gutenberg.org/">Project Gutenberg</a> for public domain texts. The sky’s the limit on what you can do, but here are some ideas:</p>

<ul>
  <li><em>Term-Character Matrix</em>.  Our data set.</li>
  <li><em>Novel recommender system</em>. Maybe you enjoyed reading <em>Sense and Sensibility</em> and <em>War and Peace</em>. Can you suggest some similar novels? Or maybe you need some variety in your consumption. Find novels that are really different.</li>
  <li><em>Other languages</em>. Do these techniques work in other languages? Project Gutenberg has texts in a variety of languages. Maybe you could use this to measure language similarity?</li>
  <li><em>Modernizing Shakespeare</em>.  When I read Shakespeare in high school, I had the dickens of a time trying to understand all the weird words in the play.  Some people have re-written Shakespeare’s plays into contemporary English.  An <a href="https://cocoxu.github.io">awesome NLP researcher</a> has <a href="https://github.com/cocoxu/Shakespeare">compiled that data</a>.  Use her data and your vector space models to find contemporary words that mean similar things to the Shakespearean English.</li>
</ul>

<h2 id="deliverables">Deliverables</h2>
<div class="alert alert-warning">
  <p>Here are the deliverables that you will need to submit:</p>
  <ul>
    <li>writeup.pdf</li>
    <li>code (.zip). It should be written in Python 3.</li>
  </ul>
</div>

<h2 id="recommended-readings">Recommended readings</h2>

<table>
   
    <tr>
      <td>
	
		<a href="https://web.stanford.edu/~jurafsky/slp3/15.pdf">Vector Semantics.</a>
        
	Dan Jurafsky and James H. Martin.
	Speech and Language Processing (3rd edition draft)  .

	
		
</td></tr>
  
    <tr>
      <td>
	
		<a href="https://www.jair.org/media/2934/live-2934-4846-jair.pdf">From Frequency to Meaning&colon; Vector Space Models of Semantics.</a>
        
	Peter D. Turney and Patrick Pantel.
	Journal of Artificial Intelligence Research  2010.

	
	<!-- abstract button -->
	<a data-toggle="modal" href="#from-frequency-to-meaning-abstract" class="label label-success">Abstract</a>
	<!-- /.abstract button -->
	<!-- abstract content -->
	<div id="from-frequency-to-meaning-abstract" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="from-frequency-to-meaning">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="from-frequency-to-meaning">From Frequency to Meaning&colon; Vector Space Models of Semantics</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
        Computers understand very little of the meaning of human language. This profoundly limits our ability to give instructions to computers, the ability of computers to explain their actions to us, and the ability of computers to analyse and process text. Vector space models (VSMs) of semantics are beginning to address these limits. This paper surveys the use of VSMs for semantic processing of text. We organize the literature on VSMs according to the structure of the matrix in a VSM. There are currently three broad classes of VSMs, based on term–document, word–context, and pair–pattern matrices, yielding three classes of applications. We survey a broad range of applications in these three categories and we take a detailed look at a specific open source project in each category. Our goal in this survey is to show the breadth of applications of VSMs for semantics, to provide a new perspective on VSMs for those who are already familiar with the area, and to provide pointers into the literature for those who are less familiar with the field.
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.abstract-content -->
	
		
	<!-- bibtex button -->
	<a data-toggle="modal" href="#from-frequency-to-meaning-bibtex" class="label label-default">BibTex</a>
	<!-- /.bibtex button -->
	<!-- bibtex content -->
	<div id="from-frequency-to-meaning-bibtex" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="from-frequency-to-meaning">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="from-frequency-to-meaning">From Frequency to Meaning&colon; Vector Space Models of Semantics</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
 	   <pre>@article{turney2010frequency,
  title={From Frequency to Meaning: Vector Space Models of Semantics},
  author={Turney, Peter D and Pantel, Patrick},
  journal={Journal of Artificial Intelligence Research},
  volume={37},
  pages={141--188},
  year={2010}
}

           </pre>
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.bibtex-content -->
	
</td></tr>
  
    <tr>
      <td>
	
		<a href="http://www.aclweb.org/anthology/C12-1177">Paraphrasing for Style.</a>
        
	Wei Xu, Alan Ritter, Bill Dolan, Ralph Grisman, and Colin Cherry.
	Coling  2012.

	
	<!-- abstract button -->
	<a data-toggle="modal" href="#paraphrasing-for-style-abstract" class="label label-success">Abstract</a>
	<!-- /.abstract button -->
	<!-- abstract content -->
	<div id="paraphrasing-for-style-abstract" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="paraphrasing-for-style">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="paraphrasing-for-style">Paraphrasing for Style</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
        We present initial investigation into the task of paraphrasing language while targeting a particular writing style. The plays of William Shakespeare and their modern translations are used as a testbed for evaluating paraphrase systems targeting a specific style of writing. We show that even with a relatively small amount of parallel training data, it is possible to learn paraphrase models which capture stylistic phenomena, and these models outperform baselines based on dictionaries and out-of-domain parallel text. In addition we present an initial investigation into automatic evaluation metrics for paraphrasing writing style. To the best of our knowledge this is the first work to investigate the task of paraphrasing text with the goal of targeting a specific style of writing.
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.abstract-content -->
	
		
	<!-- bibtex button -->
	<a data-toggle="modal" href="#paraphrasing-for-style-bibtex" class="label label-default">BibTex</a>
	<!-- /.bibtex button -->
	<!-- bibtex content -->
	<div id="paraphrasing-for-style-bibtex" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="paraphrasing-for-style">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="paraphrasing-for-style">Paraphrasing for Style</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
 	   <pre>@inproceedings{xu2012paraphrasing,
  title={Paraphrasing for Style},
  author={Xu, Wei and Ritter, Alan and Dolan, Bill and Grishman, Ralph and Cherry, Colin},
  booktitle={COLING},
  pages={2899--2914},
  year={2012}
}

           </pre>
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.bibtex-content -->
	
</td></tr>
  
    <tr>
      <td>
	
		<a href="http://www.aclweb.org/anthology/D15-1036">Evaluation methods for unsupervised word embeddings.</a>
        
	Tobias Schnabel, Igor Labutov, David Mimno, Thorsten Joachims.
	EMNLP  2015.

	
	<!-- abstract button -->
	<a data-toggle="modal" href="#evaluation-methods-for-word-embeddings-abstract" class="label label-success">Abstract</a>
	<!-- /.abstract button -->
	<!-- abstract content -->
	<div id="evaluation-methods-for-word-embeddings-abstract" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="evaluation-methods-for-word-embeddings">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="evaluation-methods-for-word-embeddings">Evaluation methods for unsupervised word embeddings</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
        We present a comprehensive study of evaluation methods for unsupervised embedding techniques that obtain meaningful representations of words from text. Different evaluations result in different orderings of embedding methods, calling into question the common assumption that there is one single optimal vector representation. We present new evaluation techniques that directly compare embeddings with respect to specific queries. These methods reduce bias, provide greater insight, and allow us to solicit data-driven relevance judgments rapidly and accurately through crowdsourcing.
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.abstract-content -->
	
		
	<!-- bibtex button -->
	<a data-toggle="modal" href="#evaluation-methods-for-word-embeddings-bibtex" class="label label-default">BibTex</a>
	<!-- /.bibtex button -->
	<!-- bibtex content -->
	<div id="evaluation-methods-for-word-embeddings-bibtex" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="evaluation-methods-for-word-embeddings">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="evaluation-methods-for-word-embeddings">Evaluation methods for unsupervised word embeddings</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
 	   <pre>@InProceedings{schnabel-EtAl:2015:EMNLP,
  author    = {Schnabel, Tobias  and  Labutov, Igor  and  Mimno, David  and  Joachims, Thorsten},
  title     = {Evaluation methods for unsupervised word embeddings},
  booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2015},
  address   = {Lisbon, Portugal},
  publisher = {Association for Computational Linguistics},
  pages     = {298--307},
  url       = {http://aclweb.org/anthology/D15-1036}
}

           </pre>
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.bibtex-content -->
	
</td></tr>
  
    <tr>
      <td>
	
		<a href="http://www.aclweb.org/anthology/P14-5004">Community Evaluation and Exchange of Word Vectors at wordvectors.org.</a>
        
	Manaal Faruqui and Chris Dyer.
	ACL demos  2014.

	
	<!-- abstract button -->
	<a data-toggle="modal" href="#wordvectors-abstract" class="label label-success">Abstract</a>
	<!-- /.abstract button -->
	<!-- abstract content -->
	<div id="wordvectors-abstract" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="wordvectors">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="wordvectors">Community Evaluation and Exchange of Word Vectors at wordvectors.org</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
        Vector space word representations are useful for many natural language processing applications. The diversity of techniques for computing vector representations and the large number of evaluation benchmarks makes reliable comparison a tedious task both for researchers developing new vector space models and for those wishing to use them. We present a website and suite of offline tools that that facilitate evaluation of word vectors on standard lexical semantics benchmarks and permit exchange and archival by users who wish to find good vectors for their applications. The system is accessible at www.wordvectors.org.
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.abstract-content -->
	
		
	<!-- bibtex button -->
	<a data-toggle="modal" href="#wordvectors-bibtex" class="label label-default">BibTex</a>
	<!-- /.bibtex button -->
	<!-- bibtex content -->
	<div id="wordvectors-bibtex" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="wordvectors">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="wordvectors">Community Evaluation and Exchange of Word Vectors at wordvectors.org</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
 	   <pre>@InProceedings{faruqui-dyer:2014:P14-5,
  author    = {Faruqui, Manaal  and  Dyer, Chris},
  title     = {Community Evaluation and Exchange of Word Vectors at wordvectors.org},
  booktitle = {Proceedings of 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  month     = {June},
  year      = {2014},
  address   = {Baltimore, Maryland},
  publisher = {Association for Computational Linguistics},
  pages     = {19--24},
  url       = {http://www.aclweb.org/anthology/P14-5004}
}

           </pre>
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.bibtex-content -->
	
</td></tr>
  
</table>

<div class="panel panel-danger">
<div class="panel-heading">
    <h2 id="grading-rubric">Grading Rubric</h2>
  </div>
<div class="panel-body">

    <p>This assignment was worth 60 points total (30 code, 30 writeup).   The rubic used for grading this homework is below. The code we used to test your <code class="highlighter-rouge">main.py</code> scripts locally is available <a href="/downloads/hw3/hw3-localtest.py">here</a>, and the solution code is <a href="/downloads/hw3/main_solutions.py">here</a>.</p>

    <h4 id="code-30-points-total">Code (30 points total)</h4>

    <p>1.1  (3) Function <code class="highlighter-rouge">create_term_document_matrix</code> correct</p>

    <ul>
      <li>-2 incorrect counts in matrix</li>
      <li>-1 dimensions flipped</li>
    </ul>

    <p>1.2  (3) Function <code class="highlighter-rouge">create_term_context_matrix</code> correct</p>

    <ul>
      <li>-2 incorrect counts in matrix</li>
    </ul>

    <p>1.3  (3) Function <code class="highlighter-rouge">create_tf_idf_matrix</code> correct</p>

    <ul>
      <li>-2 TF-IDF for frequent word ‘run’, play ‘Julius Caesar’ is greater than TF-IDF for rare word ‘dagger’, play ‘Julius Caesar’; should be less.</li>
    </ul>

    <p>1.4  (3) Function <code class="highlighter-rouge">create_PPMI_matrix</code> correct</p>

    <ul>
      <li>-2 PPMI for frequent context ‘the’, word ‘sword’ is greater than PPMI for rare context ‘bloody’, word ‘sword’; should be less.</li>
    </ul>

    <p>1.5  (4) Function <code class="highlighter-rouge">compute_cosine_similarity</code> correct</p>

    <ul>
      <li>-2 Incorrect value returned for random test vectors</li>
    </ul>

    <p>1.6  (4) Function <code class="highlighter-rouge">compute_jaccard_similarity</code> correct</p>

    <ul>
      <li>-2 Incorrect value returned for random test vectors</li>
    </ul>

    <p>1.7  (4) Function <code class="highlighter-rouge">compute_dice_similarity</code> correct</p>

    <ul>
      <li>-2 Incorrect value returned for random test vectors</li>
    </ul>

    <p>1.8  (3) Function <code class="highlighter-rouge">rank_plays</code> correct</p>

    <ul>
      <li>-2 Incorrect ranking returned for standardized matrix and similarity function input</li>
      <li>-1 Function re-reads Shakespeare data from disk</li>
    </ul>

    <p>1.9  (3) Function <code class="highlighter-rouge">rank_words</code> correct</p>

    <ul>
      <li>-2 Incorrect ranking returned for standardized matrix and similarity function input</li>
      <li>-1 Function re-reads Shakespeare data from disk</li>
    </ul>

    <h4 id="writeup-30-points-total">Writeup (30 points total)</h4>

    <p>2.1  (15) Analysis of similarity between play vectors</p>

    <ul>
      <li>-5 does not include comparison of nearest plays in terms of thematic similarity, or other general assessment of vector quality</li>
      <li>-5 does not analyze different similarity metrics</li>
      <li>-5 does not analyze different matrix weighting schemes</li>
      <li>-10 No analysis of similarity between play vectors, or results reported without any analysis</li>
    </ul>

    <p>2.2  (15) Analysis of similarity between word vectors</p>

    <ul>
      <li>-5 does not analyze different matrix weighting schemes</li>
      <li>-5 does not analyze different similarity metrics</li>
      <li>-5 No analysis of similarity between word vectors</li>
      <li>-10 No analysis of similarity between word vectors, or results reported without any analysis</li>
    </ul>

    <h4 id="extra-credit-10-points-max">Extra Credit (10 points max)</h4>

    <ul>
      <li>+2 Includes analysis of character similarity based on term-character matrix</li>
      <li>+5 Includes comparison of vector similarity with human judgements</li>
      <li>+5 Other additional substantive analysis</li>
    </ul>

  </div>
</div>

        </div>
      </div>
      
      </div>

    <footer class="text-center text-muted">
      <hr>
      
      <p>Daphne Ippolito, Anne Cocos, Stephen Mayhew, and Chris Callison-Burch developed this homework assignment for UPenn’s CIS 530 class in Spring 2018.</p>
<br>
      
    </footer>

    <script src="//ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>
    <script src="//maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
    <script type="text/javascript">
      $(document).ready(function(){
        $("#homework").addClass("active");
        
      });
    </script>
  </body>
</html>
