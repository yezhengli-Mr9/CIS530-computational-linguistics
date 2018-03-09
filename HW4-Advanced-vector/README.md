# CIS530-HW4-Advanced-vector-models-Computational-Linguistics


<!DOCTYPE html>
<html lang="en">
  
<!-- End of check whether the assignment is up to date -->

<div class="alert alert-info">
This assignment is due before 11:00AM on Wednesday, February 7, 2018.
</div>

<h1 id="advanced-vector-space-models--assignment-4">Advanced Vector Space Models <span class="text-muted">: Assignment 4</span></h1>

<p>In this assignment, we will examine some advanced uses of vector representations of words. We are going to look at two different problems:</p>
<ol>
  <li>Solving word relation problems like analogies using word embeddings.</li>
  <li>Discovering the different senses of a “polysemous” word by clustering together its synonyms. 
You will use an open source Python package for creating and manipulating word vectors called <em>gensim.</em>  Gensim lets you easily train word embedding models like word2vec.</li>
</ol>

<div class="alert alert-warning">
  <p>In order to use the gensim package, you’ll have to be using Python version 3.6 or higher.  On my Mac, I did the following:</p>
  <ul>
    <li><code class="highlighter-rouge">brew install python3</code></li>
    <li><code class="highlighter-rouge">pip3 install gensim</code></li>
    <li>Then when I ran python, I used the command <code class="highlighter-rouge">python3</code> instead of just <code class="highlighter-rouge">python</code></li>
  </ul>
</div>

<div class="alert alert-info">
  <p>Here are the materials that you should download for this assignment:</p>
  <ul>
    <li><a href="/downloads/hw4/question1.txt"><code class="highlighter-rouge">question1.txt</code></a> A template for answering question 1.</li>
    <li><a href="/downloads/hw4/data.zip"><code class="highlighter-rouge">data.zip</code></a> Contains all the data</li>
    <li><a href="/downloads/hw4/vectorcluster.py"><code class="highlighter-rouge">vectorcluster.py</code></a> Main code stub</li>
    <li><a href="/downloads/hw4/evaluate.py"><code class="highlighter-rouge">evaluate.py</code></a> Evaluation script</li>
    <li><a href="/downloads/hw4/writeup.tex"><code class="highlighter-rouge">writeup.tex</code></a> Report template.</li>
    <li><a href="/downloads/hw4/makecooccurrences.py"><code class="highlighter-rouge">makecooccurrences.py</code></a> Script to make cooccurrences (optional use)</li>
    <li><a href="http://www.cis.upenn.edu/~cis530/18sp/data/reuters.rcv1.tokenized.gz">Tokenized Reuters RCV1 Corpus</a></li>
    <li><a href="https://code.google.com/archive/p/word2vec/">Google’s pretrained word2vec vectors</a>, under the heading “Pretrained word and phrase vectors”</li>
  </ul>
</div>

<h1 id="part-1-exploring-analogies-and-other-word-pair-relationships">Part 1: Exploring Analogies and Other Word Pair Relationships</h1>

<p>Word2vec is a very cool word embedding method that was developed by Thomas Mikolov and his collaborators.  One of the noteworthy things about the method is that it can be used to solve word analogy problems like
  man is to king as woman is to [blank]
The way that it works is to perform vector math.  They take the vectors representing <em>king</em>, <em>man</em> and <em>woman</em> and perform some vector arithmetic to produce a vector that is close to the expected answer. 
$king−man+woman \approx queen$
So We can find the nearest a vector in the vocabulary by looking for $argmax \ cos(x, king-man+woman)$.  Omar Levy has a nice explnation of the method in <a href="https://www.quora.com/How-does-Mikolovs-word-analogy-for-word-embedding-work-How-can-I-code-such-a-function">this Quora post</a>, and in his paper <a href="http://www.aclweb.org/anthology/W14-1618">Linguistic Regularities in Sparse and Explicit Word Representations</a>.</p>

<p>In addition to solving this sort of analogy problem, the same sort of vector arithmetic was used with word2vec embeddings to find relationships between pairs of words like the following:</p>

<p><img src="/assets/img/word2vec_word_pair_relationships.jpg" alt="Examples of five types of semantic and nine types of syntactic questions in the Semantic- Syntactic Word Relationship test set" style="width: 50%;" /></p>

<p>In the first part of this homework, you will play around with the <a href="https://radimrehurek.com/gensim/index.html">gensim library</a>  library.  You will use <code class="highlighter-rouge">gensim</code>  load a dense vector model trained using <code class="highlighter-rouge">word2vec</code>, and use it to manipulate and analyze the vectors.<br />
 You can start by experimenting on your own, or reading through  <a href="https://rare-technologies.com/word2vec-tutorial/">this tutorial on using word2vec with gensim</a>. You should familiarize yourself with the <a href="https://radimrehurek.com/gensim/models/keyedvectors.html">KeyedVectors documentation</a>.</p>

<p>The questions below are designed to familiarize you with the <code class="highlighter-rouge">gensim</code> Word2Vec package, and get you thinking about what type of semantic information word embeddings can encode.  You’ll submit your answers to these questions when you submit your other homework materials.</p>

<p>Load the word vectors using the following Python commands:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="kn">from</span> <span class="nn">gensim.models</span> <span class="kn">import</span> <span class="n">KeyedVectors</span>
<span class="n">vecfile</span> <span class="o">=</span> <span class="s">'GoogleNews-vectors-negative300.bin'</span>
<span class="n">vecs</span> <span class="o">=</span> <span class="n">KeyedVectors</span><span class="o">.</span><span class="n">load_word2vec_format</span><span class="p">(</span><span class="n">vecfile</span><span class="p">,</span> <span class="n">binary</span><span class="o">=</span><span class="bp">True</span><span class="p">)</span></code></pre></figure>

<ul>
  <li>What is the dimensionality of these word embeddings? Provide an integer answer.</li>
  <li>What are the top-5 most similar words to <code class="highlighter-rouge">picnic</code> (not including <code class="highlighter-rouge">picnic</code> itself)? (Use the function <code class="highlighter-rouge">gensim.models.KeyedVectors.wv.most_similar</code>)</li>
  <li>According to the word embeddings, which of these words is not like the others?
<code class="highlighter-rouge">['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette']</code>
(Use the function <code class="highlighter-rouge">gensim.models.KeyedVectors.wv.doesnt_match</code>)</li>
  <li>Solve the following analogy: “leg” is to “jump” as X is to “throw”.
(Use the function <code class="highlighter-rouge">gensim.models.KeyedVectors.wv.most_similar</code> with <code class="highlighter-rouge">positive</code> and <code class="highlighter-rouge">negative</code> arguments.)</li>
</ul>

<p>We have provided a file called <code class="highlighter-rouge">question1.txt</code> for you to submit answers to the questions above.</p>

<h1 id="part-2-creating-word-sense-clusters">Part 2: Creating Word Sense Clusters</h1>

<p>Many natural language processing (NLP) tasks require knowing the sense of polysemous words, which are words with multiple meanings. For example, the word <em>bug</em> can mean</p>
<ol>
  <li>a creepy crawly thing</li>
  <li>an error in your computer code</li>
  <li>a virus or bacteria that makes you sick</li>
  <li>a listening device planted by the FBI</li>
</ol>

<p>In past research my PhD students and I have looked into automatically deriving the different meaning of polysemous words like bug by clustering their paraphrases.  We have developed a resource called <a href="http://paraphrase.org/">the paraphrase database (PPDB)</a> that contains of paraphrases for  tens of millions words and phrases.  For the target word <em>bug</em>, we have an unordered list of paraphrases including: <em>insect, glitch, beetle, error, microbe, wire, cockroach, malfunction, microphone, mosquito, virus, tracker, pest, informer, snitch, parasite, bacterium, fault, mistake, failure</em> and many others.  We used automatic clustering group those into sets like:</p>

<p><img src="/assets/img/bug_clusters.jpg" alt="Bug Clusters" style="width: 50%;" /></p>

<p>These clusters approximate the different word senses of <em>bug</em>.  You will explore the main idea underlying our word sense clustering method: which measure the similarity between each pair of paraphrases for a target word and then group together the paraphrases that are most similar to each other.   This affinity matrix gives an example of one of the methods for measuring similarity that we tried in <a href="https://www.cis.upenn.edu/~ccb/publications/clustering-paraphrases-by-word-sense.pdf">our paper</a>:</p>

<p><img src="/assets/img/affinity_matrix.jpg" alt="Similarity of paraphrses" style="width: 50%;" /></p>

<p>Here the darkness values give an indication of how similar paraprhases are to each other.  For instance <em>sim(insect, pest) &gt; sim(insect, error)</em>.</p>

<p>In this assignment, we will use vector representations in order to measure their similarities of pairs of paraprhases.  You will play with different vector space representations of words to create clusters of word senses.</p>

<p>In this image, we have a target word “bug”, and a list of all synonyms (taken from WordNet). The 4 circles are the 4 senses of “bug.” The input to the problem is all the synonyms in a single list, and the task is to separate them correctly. As humans, this is pretty intuitive, but computers aren’t that smart. We will use this task to explore different types of word representations.</p>

<p>You can read more about this task in <a href="https://www.cis.upenn.edu/~ccb/publications/clustering-paraphrases-by-word-sense.pdf">these</a> <a href="https://cs.uwaterloo.ca/~cdimarco/pdf/cs886/Pantel+Lin02.pdf">papers</a>.</p>

<h1 id="clustering-with-word-vectors">Clustering with Word Vectors</h1>

<p>We expect that you have read Jurafsky and Martin, chapters <a href="https://web.stanford.edu/~jurafsky/slp3/15.pdf">15</a> and <a href="https://web.stanford.edu/~jurafsky/slp3/16.pdf">16</a>. Word vectors, also known as word embeddings, can be thought of simply as points in some high-dimensional space. Remember in geometry class when you learned about the Euclidean plane, and 2-dimensional points in that plane? It’s not hard to understand distance between those points – you can even measure it with a ruler. Then you learned about 3-dimensional points, and how to calculate the distance between these. These 3-dimensional points can be thought of as positions in physical space.</p>

<p>Now, do your best to stop thinking about physical space, and generalize this idea in your mind: you can calculate a distance between 2-dimensional and 3-dimensional points, now imagine a point with 300 dimensions. The dimensions don’t necessarily have meaning in the same way as the X,Y, and Z dimensions in physical space, but we can calculate distances all the same.</p>

<p>This is how we will use word vectors in this assignment: as points in some high-dimensional space, where distances between points are meaningful. The interpretation of distance between word vectors depends entirely on how they were made, but for our purposes, we will consider distance to measure semantic similarity. Word vectors that are close together should have meanings that are similar.</p>

<p>With this framework, we can see how to solve our synonym clustering problem. Imagine in the image below that each point is a (2-dimensional) word vector. Using the distance between points, we can separate them into 3 clusters. This is our task.</p>

<p><img src="/assets/img/kmeans.svg" alt="kmeans" />
(Image taken from <a href="https://en.wikipedia.org/wiki/K-means_clustering">Wikipedia</a>)</p>

<h2 id="the-data">The Data</h2>

<p>The data to be used for this assignment consists of sets of paraphrases corresponding to one of 56 polysemous target words, e.g.</p>

<table class="table">
  <thead>
    <tr>
      <th scope="col">Target</th>
      <th scope="col">Paraphrase set</th>
    </tr>
  </thead>
  <tbody>
    <tr>      
      <td>note.v</td>
      <td>comment mark tell observe state notice say remark mention</td>
    </tr>
    <tr>
      <td>hot.a</td>
      <td>raging spicy blistering red-hot live</td>
    </tr>
  </tbody>
</table>

<p>(Here the <code class="highlighter-rouge">.v</code> following the target <code class="highlighter-rouge">note</code> indicates the part of speech.)</p>

<p>Your objective is to automatically cluster each paraphrase set such that each cluster contains words pertaining to a single <em>sense</em>, or meaning, of the target word. Note that a single word from the paraphrase set might belong to one or more clusters.</p>

<p>For evaluation, we take the set of ground truth senses from <a href="http://wordnet.princeton.edu">WordNet</a>.</p>

<h3 id="development-data">Development data</h3>

<p>The development data consists of two files – a words file (the input), and a clusters file (to evaluate your output). The words file <code class="highlighter-rouge">dev_input.txt</code> is formatted such that each line contains one target, its paraphrase set, and the number of ground truth clusters <em>k</em>, separated by a <code class="highlighter-rouge">::</code> symbol:</p>

<div class="highlighter-rouge"><div class="highlight"><pre class="highlight"><code>target.pos :: k :: paraphrase1 paraphrase2 paraphrase3 ...
</code></pre></div></div>

<p>You can use <em>k</em> as input to your clustering algorithm.</p>

<p>The clusters file <code class="highlighter-rouge">dev_output.txt</code> contains the ground truth clusters for each target word’s paraphrase set, split over <em>k</em> lines:</p>

<div class="highlighter-rouge"><div class="highlight"><pre class="highlight"><code>target.pos :: 1 :: paraphrase2 paraphrase6
target.pos :: 2 :: paraphrase3 paraphrase4 paraphrase5
...
target.pos :: k :: paraphrase1 paraphrase9
</code></pre></div></div>

<h3 id="test-data">Test data</h3>

<p>For testing, you will receive only words file <code class="highlighter-rouge">test_input.txt</code> containing the test target words and their paraphrase sets. Your job is to create an output file, formatted in the same way as <code class="highlighter-rouge">dev_output.txt</code>, containing the clusters produced by your system. Neither order of senses, nor order of words in a cluster matter.</p>

<h2 id="evaluation">Evaluation</h2>

<p>There are many possible ways to evaluate clustering solutions. For this homework we will rely on the paired F-score, which you can read more about in <a href="https://www.cs.york.ac.uk/semeval2010_WSI/paper/semevaltask14.pdf">this paper</a>.</p>

<p>The general idea behind paired F-score is to treat clustering prediction like a classification problem; given a target word and its paraphrase set, we call a <em>positive instance</em> any pair of paraphrases that appear together in a ground-truth cluster. Once we predict a clustering solution for the paraphrase set, we similarly generate the set of word pairs such that both words in the pair appear in the same predicted cluster. We can then evaluate our set of predicted pairs against the ground truth pairs using precision, recall, and F-score.</p>

<p>We have provided an evaluation script that you can use when developing your own system. You can run it as follows:</p>

<div class="highlighter-rouge"><div class="highlight"><pre class="highlight"><code>python evaluate.py &lt;GROUND-TRUTH-FILE&gt; &lt;PREDICTED-CLUSTERS-FILE&gt;
</code></pre></div></div>

<h2 id="baselines">Baselines</h2>

<p>On the dev data, a random baseline gets about 20%, the word cooccurrence matrix gets about 36%, and the word2vec vectors get about 30%.</p>

<h3 id="1-sparse-representations">1. Sparse Representations</h3>

<p>Your next task is to generate clusters for the target words in <code class="highlighter-rouge">test_input.txt</code> based on a feature-based (not dense) vector space representation. In this type of VSM, each dimension of the vector space corresponds to a specific feature, such as a context word (see, for example, the term-context matrix described in <a href="https://web.stanford.edu/~jurafsky/slp3/15.pdf">Chapter 15.1.2 of Jurafsky &amp; Martin</a>).</p>

<p>You will calculate cooccurrence vectors on the Reuters RCV1 corpus. Download a <a href="http://www.cis.upenn.edu/~cis530/18sp/data/reuters.rcv1.tokenized.gz">tokenized and cleaned version here</a>. The original is <a href="https://archive.ics.uci.edu/ml/datasets/Reuters+RCV1+RCV2+Multilingual,+Multiview+Text+Categorization+Test+collection">here</a>. Use the provided script, <code class="highlighter-rouge">makecooccurrences.py</code>, to build these vectors. Be sure to set D and W to what you want.</p>

<p>It can take a long time to build cooccurrence vectors, so we have pre-built a set, included in the data.zip, called <code class="highlighter-rouge">coocvec-500mostfreq-window-3.vec.filter</code>. To save on space, these include only the words used in the given files.</p>

<p>You will add K-means clustering to <code class="highlighter-rouge">vectorcluster.py</code>. Here is an example of the K-means code:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="kn">from</span> <span class="nn">sklearn.cluster</span> <span class="kn">import</span> <span class="n">KMeans</span>
<span class="n">kmeans</span> <span class="o">=</span> <span class="n">KMeans</span><span class="p">(</span><span class="n">n_clusters</span><span class="o">=</span><span class="n">k</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">kmeans</span><span class="o">.</span><span class="n">labels_</span><span class="p">)</span></code></pre></figure>

<!--- Baseline description is a placeholder --->
<p>The baseline system for this section represents words using a term-context matrix <code class="highlighter-rouge">M</code> of size <code class="highlighter-rouge">|V| x D</code>, where <code class="highlighter-rouge">|V|</code> is the size of the vocabulary and D=500. Each feature corresponds to one of the top 500 most-frequent words in the corpus. The value of matrix entry <code class="highlighter-rouge">M[i][j]</code> gives the number of times the context word represented by column <code class="highlighter-rouge">j</code> appeared within W=3 words to the left or right of the word represented by row <code class="highlighter-rouge">i</code> in the corpus. Using this representation, the baseline system clusters each paraphrase set using K-means.</p>

<p>While experimenting, write out clusters for the dev input to <code class="highlighter-rouge">dev_output_features.txt</code> and use the <code class="highlighter-rouge">evaluate.py</code> script to compare against the provided <code class="highlighter-rouge">dev_output.txt</code>.</p>

<p>Implementing the baseline will score you a B, but why not try and see if you can do better? You might try experimenting with different features, for example:</p>

<ul>
  <li>What if you reduce or increase <code class="highlighter-rouge">D</code> in the baseline implementation?</li>
  <li>Does it help to change the window <code class="highlighter-rouge">W</code> used to extract contexts?</li>
  <li>Play around with the feature weighting – instead of raw counts, would it help to use PPMI?</li>
  <li>Try a different clustering algorithm that’s included with the <a href="http://scikit-learn.org/stable/modules/clustering.html">scikit-learn clustering package</a>, or implement your own.</li>
  <li>What if you include additional types of features, like paraphrases in the <a href="http://www.paraphrase.org">Paraphrase Database</a> or the part-of-speech of context words?</li>
</ul>

<p>The only feature types that are off-limits are WordNet features.</p>

<p>Turn in the predicted clusters that your VSM generates in the file <code class="highlighter-rouge">test_output_features.txt</code>. Also provide a brief description of your method in <code class="highlighter-rouge">writeup.pdf</code>, making sure to describe the vector space model you chose, the clustering algorithm you used, and the results of any preliminary experiments you might have run on the dev set. We have provided a LaTeX file shell, <code class="highlighter-rouge">writeup.tex</code>, which you can use to guide your writeup.</p>

<h3 id="2-dense-representations">2. Dense Representations</h3>
<p>Finally, we’d like to see if dense word embeddings are better for clustering the words in our test set. Run the word clustering task again, but this time use a dense word representation.</p>

<p>For this task, use files:</p>

<ul>
  <li><a href="https://code.google.com/archive/p/word2vec/">Google’s pretrained word2vec vectors</a>, under the heading “Pretrained word and phrase vectors”</li>
  <li>The Google file is very large (~3.4GB), so we have also included in the data.zip a file called <code class="highlighter-rouge">GoogleNews-vectors-negative300.filter</code>, which is filtered to contain only the words in the dev/test splits.</li>
  <li>Modify <code class="highlighter-rouge">vectorcluster.py</code> to load dense vectors.</li>
</ul>

<p>The baseline system for this section uses the provided word vectors to represent words, and K-means for clustering.</p>

<p>As before, achieving the baseline score will get you a B, but you might try to see if you can do better. Here are some ideas:</p>

<ul>
  <li>Try downloading a different dense vector space model from the web, like <a href="http://www.cs.cmu.edu/~jwieting/">Paragram</a> or <a href="https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md">fastText</a>.</li>
  <li>Train your own word vectors, either on the provided corpus or something you find online. You can use the <code class="highlighter-rouge">gensim.models.Word2Vec</code> package for the skip-gram or CBOW models, or <a href="https://nlp.stanford.edu/projects/glove/">GLOVE</a>. Try experimenting with the dimensionality.</li>
  <li><a href="https://www.cs.cmu.edu/~hovy/papers/15HLT-retrofitting-word-vectors.pdf">Retrofitting</a> is a simple way to add additional semantic knowledge to pre-trained vectors. The retrofitting code is available <a href="https://github.com/mfaruqui/retrofitting">here</a>. Experiment with different lexicons, or even try <a href="http://www.aclweb.org/anthology/N16-1018">counter-fitting</a>.</li>
</ul>

<p>As in question 2, turn in the predicted clusters that your dense vector representation generates in the file <code class="highlighter-rouge">test_output_dense.txt</code>. Also provide a brief description of your method in <code class="highlighter-rouge">writeup.pdf</code> that includes the vectors you used, and any experimental results you have from running your model on the dev set.</p>

<p>In addition, do an analysis of different errors made by each system – i.e. look at instances that the word-context matrix representation gets wrong and dense gets right, and vice versa, and see if there are any interesting patterns. There is no right answer for this.</p>

<h3 id="3-the-leaderboard">3. The Leaderboard</h3>
<p>In order to stir up some friendly competition, we would also like you to submit the clustering from your best model to a leaderboard. Copy the output file from your best model to a file called <code class="highlighter-rouge">test_output_leaderboard.txt</code>, and include it with your submission.</p>

<h3 id="extra-credit">Extra Credit</h3>
<p>We made the clustering problem deliberately easier by providing you with <code class="highlighter-rouge">k</code>, the number of clusters, as an input. But in most clustering situations the best <code class="highlighter-rouge">k</code> isn’t obvious.
To take this assignment one step further, see if you can come up with a way to automatically choose <code class="highlighter-rouge">k</code>. We have provided an additional test set, <code class="highlighter-rouge">test_nok_input.txt</code>, where the <code class="highlighter-rouge">k</code> field has been zeroed out. See if you can come up with a method that clusters words by sense, and chooses the best <code class="highlighter-rouge">k</code> on its own. (Don’t look at the number of WordNet synsets for this, as that would ruin all the fun.) The baseline system for this portion always chooses <code class="highlighter-rouge">k=5</code>.
You can submit your output to this part in a file called <code class="highlighter-rouge">test_nok_output_leaderboard.txt</code>. Be sure to describe your method in <code class="highlighter-rouge">writeup.pdf</code>.</p>

<h2 id="deliverables">Deliverables</h2>
<div class="alert alert-warning">
  <p>Here are the deliverables that you will need to submit:</p>
  <ul>
    <li><code class="highlighter-rouge">question1.txt</code> file with answers to questions from Exploration</li>
    <li>simple VSM clustering output <code class="highlighter-rouge">test_output_features.txt</code></li>
    <li>dense model clustering output <code class="highlighter-rouge">test_output_dense.txt</code></li>
    <li>your favorite clustering output for the leaderboard, <code class="highlighter-rouge">test_output_leaderboard.txt</code> (this will probably be a copy of either <code class="highlighter-rouge">test_output_features.txt</code> or <code class="highlighter-rouge">test_output_dense.txt</code>)</li>
    <li><code class="highlighter-rouge">writeup.pdf</code> (compiled from <code class="highlighter-rouge">writeup.tex</code>)</li>
    <li>your code (.zip). It should be written in Python 3.</li>
    <li>(optional) the output of your model that automatically chooses the number of clusters, <code class="highlighter-rouge">test_nok_output_leaderboard.txt</code> (submit this to the Gradescope assignment ‘Homework 4 EXTRA CREDIT’)</li>
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
	
		<a href="https://web.stanford.edu/~jurafsky/slp3/16.pdf">Semantics with Dense Vectors.</a>
        
	Dan Jurafsky and James H. Martin.
	Speech and Language Processing (3rd edition draft)  .

	
		
</td></tr>
  
    <tr>
      <td>
	
		<a href="https://arxiv.org/pdf/1301.3781.pdf?">Efficient Estimation of Word Representations in Vector Space.</a>
        
	Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean.
	ArXiV  2013.

	
	<!-- abstract button -->
	<a data-toggle="modal" href="#efficient-estimation-of-word-representations-abstract" class="label label-success">Abstract</a>
	<!-- /.abstract button -->
	<!-- abstract content -->
	<div id="efficient-estimation-of-word-representations-abstract" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="efficient-estimation-of-word-representations">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="efficient-estimation-of-word-representations">Efficient Estimation of Word Representations in Vector Space</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
        We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.abstract-content -->
	
		
</td></tr>
  
    <tr>
      <td>
	
		<a href="https://www.aclweb.org/anthology/N13-1090">Linguistic Regularities in Continuous Space Word Representations.</a>
        
	Tomas Mikolov, Wen-tau Yih, Geoffrey Zweig.
	NAACL  2013.

	
	<!-- abstract button -->
	<a data-toggle="modal" href="#linguistic-regularities-in-continous-space-word-representations-abstract" class="label label-success">Abstract</a>
	<!-- /.abstract button -->
	<!-- abstract content -->
	<div id="linguistic-regularities-in-continous-space-word-representations-abstract" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="linguistic-regularities-in-continous-space-word-representations">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="linguistic-regularities-in-continous-space-word-representations">Linguistic Regularities in Continuous Space Word Representations</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
        Continuous space language models have recently demonstrated outstanding results across a variety of tasks. In this paper, we examine the vector-space word representations that are implicitly learned by the input-layer weights. We find that these representations are surprisingly good at capturing syntactic and semantic regularities in language, and that each relationship is characterized by a relation-specific vector offset. This allows vector-oriented reasoning based on the offsets between words. For example, the male/female relationship is automatically learned, and with the induced vector representations, “King Man + Woman” results in a vector very close to “Queen.” We demonstrate that the word vectors capture syntactic regularities by means of syntactic analogy questions (provided with this paper), and are able to correctly answer almost 40% of the questions. We demonstrate that the word vectors capture semantic regularities by using the vector offset method to answer SemEval-2012 Task 2 questions. Remarkably, this method outperforms the best previous systems.
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.abstract-content -->
	
		
	<!-- bibtex button -->
	<a data-toggle="modal" href="#linguistic-regularities-in-continous-space-word-representations-bibtex" class="label label-default">BibTex</a>
	<!-- /.bibtex button -->
	<!-- bibtex content -->
	<div id="linguistic-regularities-in-continous-space-word-representations-bibtex" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="linguistic-regularities-in-continous-space-word-representations">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="linguistic-regularities-in-continous-space-word-representations">Linguistic Regularities in Continuous Space Word Representations</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
 	   <pre>@InProceedings{mikolov-yih-zweig:2013:NAACL-HLT,
  author    = {Mikolov, Tomas  and  Yih, Wen-tau  and  Zweig, Geoffrey},
  title     = {Linguistic Regularities in Continuous Space Word Representations},
  booktitle = {Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2013},
  address   = {Atlanta, Georgia},
  publisher = {Association for Computational Linguistics},
  pages     = {746--751},
  url       = {http://www.aclweb.org/anthology/N13-1090}
}

           </pre>
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.bibtex-content -->
	
</td></tr>
  
    <tr>
      <td>
	
		<a href="https://www.semanticscholar.org/paper/Discovering-word-senses-from-text-Pantel-Lin/">Discovering Word Senses from Text.</a>
        
	Patrick Pangel and Dekang Ling.
	KDD  2002.

	
	<!-- abstract button -->
	<a data-toggle="modal" href="#discovering-word-senses-from-text-abstract" class="label label-success">Abstract</a>
	<!-- /.abstract button -->
	<!-- abstract content -->
	<div id="discovering-word-senses-from-text-abstract" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="discovering-word-senses-from-text">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="discovering-word-senses-from-text">Discovering Word Senses from Text</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
        Inventories of manually compiled dictionaries usually serve as a source for word senses. However, they often include many rare senses while missing corpus/domain-specific senses. We present a clustering algorithm called CBC (Clustering By Committee) that automatically discovers word senses from text. It initially discovers a set of tight clusters called committees that are well scattered in the similarity space. The centroid of the members of a committee is used as the feature vector of the cluster. We proceed by assigning words to their most similar clusters. After assigning an element to a cluster, we remove their overlapping features from the element. This allows CBC to discover the less frequent senses of a word and to avoid discovering duplicate senses. Each cluster that a word belongs to represents one of its senses. We also present an evaluation methodology for automatically measuring the precision and recall of discovered senses.
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.abstract-content -->
	
		
	<!-- bibtex button -->
	<a data-toggle="modal" href="#discovering-word-senses-from-text-bibtex" class="label label-default">BibTex</a>
	<!-- /.bibtex button -->
	<!-- bibtex content -->
	<div id="discovering-word-senses-from-text-bibtex" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="discovering-word-senses-from-text">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="discovering-word-senses-from-text">Discovering Word Senses from Text</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
 	   <pre>@inproceedings{Pantel2002DiscoveringWS,
  title={Discovering word senses from text},
  author={Patrick Pantel and Dekang Lin},
  booktitle={KDD},
  year={2002}
}

           </pre>
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.bibtex-content -->
	
</td></tr>
  
    <tr>
      <td>
	
		<a href="https://www.cis.upenn.edu/~ccb/publications.html">Clustering Paraphrases by Word Sense.</a>
        
	Anne Cocos and Chris Callison-Burch.
	NAACL  2016.

	
	<!-- abstract button -->
	<a data-toggle="modal" href="#clustering-paraphrases-by-word-sense-abstract" class="label label-success">Abstract</a>
	<!-- /.abstract button -->
	<!-- abstract content -->
	<div id="clustering-paraphrases-by-word-sense-abstract" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="clustering-paraphrases-by-word-sense">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="clustering-paraphrases-by-word-sense">Clustering Paraphrases by Word Sense</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
        Automatically generated databases of English paraphrases have the drawback that they return a single list of paraphrases for an input word or phrase. This means that all senses of polysemous words are grouped together, unlike WordNet which partitions different senses into separate synsets. We present a new method for clustering paraphrases by word sense, and apply it to the Paraphrase Database (PPDB). We investigate the performance of hierarchical and spectral clustering algorithms, and systematically explore different ways of defining the similarity matrix that they use as input. Our method produces sense clusters that are qualitatively and quantitatively good, and that represent a substantial improvement to the PPDB resource.
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.abstract-content -->
	
		
	<!-- bibtex button -->
	<a data-toggle="modal" href="#clustering-paraphrases-by-word-sense-bibtex" class="label label-default">BibTex</a>
	<!-- /.bibtex button -->
	<!-- bibtex content -->
	<div id="clustering-paraphrases-by-word-sense-bibtex" class="modal fade" tabindex="-1" role="dialog" aria-labelledby="clustering-paraphrases-by-word-sense">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
          <h4 class="modal-title" id="clustering-paraphrases-by-word-sense">Clustering Paraphrases by Word Sense</h4>
        </div><!-- /.modal-header -->
        <div class="modal-body">
 	   <pre>@inproceedings{Cocos-Callison-Burch:2016:NAACL,
 author = {Anne Cocos and Chris Callison-Burch},
 title = {Clustering Paraphrases by Word Sense},
 booktitle = {The 2016 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2016)},
 month = {June},
 year = {2016},
 address = {San Diego, California},
 url = {http://www.cis.upenn.edu/~ccb/publications/clustering-paraphrases-by-word-sense.pdf}
 } 

           </pre>
        </div><!-- /.modal-body -->
	</div><!-- /.modal-content -->
	</div><!-- /.modal-dialog -->
	</div><!-- /.bibtex-content -->
	
</td></tr>
  
</table>

        </div>
      </div>
      
      </div>

    <footer class="text-center text-muted">
      <hr>
      
      <p>Stephen Mayhew, Anne Cocos and Chris Callison-Burch developed this homework assignment for UPenn’s CIS 530 class in Spring 2018.</p>
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
