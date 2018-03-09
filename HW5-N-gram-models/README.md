# CIS530-HW5-N-gram-models-Computational-Linguistics

<!DOCTYPE html>
<html lang="en">
 
<!-- End of check whether the assignment is up to date -->

<div class="alert alert-info">
This assignment is due before 11:00AM on Wednesday, February 14, 2018.
</div>

<h1 id="character-based-language-models--assignment-5">Character-based Language Models <span class="text-muted">: Assignment 5</span></h1>

<p>In the textbook, language modeling was defined as the task of predicting the next word in a sequence given the previous words. In this assignment, we will focus on the related problem of predicting the next <em>character</em> in a sequence given the previous characters.</p>

<p>The learning goals of this assignment are to:</p>
<ul>
  <li>Understand how to compute language model probabilities using maximum likelihood estimation</li>
  <li>Implement basic smoothing, back-off and interpolation.</li>
  <li>Have fun using a language model to probabilistically generate texts.</li>
  <li>Use a set of language models to perform text classification.</li>
</ul>

<div class="alert alert-info">
  <p>Here are the materials that you should download for this assignment:</p>
  <ul>
    <li><a href="/downloads/hw5/language_model.py">Skeleton python code</a>.</li>
    <li><a href="/downloads/hw5/cities_train.zip">training data for text classification task</a>.</li>
    <li><a href="/downloads/hw5/cities_val.zip">dev data for text classification task</a>.</li>
    <li><a href="/downloads/hw5/cities_test.txt">test file for leaderboard</a></li>
  </ul>
</div>

<h2 id="part-1-unsmoothed-maximum-likelihood-character-level-language-models">Part 1: Unsmoothed Maximum Likelihood Character-Level Language Models</h2>

<p>We’re going to be starting with some <a href="http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139">nice, compact code for character-level language models</a>. that was written by <a href="http://u.cs.biu.ac.il/~yogo/">Yoav Goldberg</a>.  Below is Yoav’s code for training a language model.</p>

<div class="alert alert-warning">
  <p>Note: all of this code is included in the provided code stub called <code class="highlighter-rouge">language_model.py</code>. No need to copy and paste.</p>
</div>

<h3 id="train-a-language-model">Train a language model</h3>

<p>Note: we provide you this code in the Python <a href="/downloads/hw5/language_model.py">stub file</a>.</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="k">def</span> <span class="nf">train_char_lm</span><span class="p">(</span><span class="n">fname</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">add_k</span><span class="o">=</span><span class="mi">1</span><span class="p">):</span>
  <span class="s">''' Trains a language model.
  This code was borrowed from 
  http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139
  Inputs:
    fname: Path to a text corpus.
    order: The length of the n-grams.
    add_k: k value for add-k smoothing. NOT YET IMPLMENTED
  Returns:
    A dictionary mapping from n-grams of length n to a list of tuples.
    Each tuple consists of a possible net character and its probability.
  '''</span>

  <span class="c"># TODO: Add your implementation of add-k smoothing.</span>

  <span class="n">data</span> <span class="o">=</span> <span class="nb">open</span><span class="p">(</span><span class="n">fname</span><span class="p">)</span><span class="o">.</span><span class="n">read</span><span class="p">()</span>
  <span class="n">lm</span> <span class="o">=</span> <span class="n">defaultdict</span><span class="p">(</span><span class="n">Counter</span><span class="p">)</span>
  <span class="n">pad</span> <span class="o">=</span> <span class="s">"~"</span> <span class="o">*</span> <span class="n">order</span>
  <span class="n">data</span> <span class="o">=</span> <span class="n">pad</span> <span class="o">+</span> <span class="n">data</span>
  <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">data</span><span class="p">)</span><span class="o">-</span><span class="n">order</span><span class="p">):</span>
    <span class="n">history</span><span class="p">,</span> <span class="n">char</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="n">order</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="n">i</span><span class="o">+</span><span class="n">order</span><span class="p">]</span>
    <span class="n">lm</span><span class="p">[</span><span class="n">history</span><span class="p">][</span><span class="n">char</span><span class="p">]</span><span class="o">+=</span><span class="mi">1</span>
  <span class="k">def</span> <span class="nf">normalize</span><span class="p">(</span><span class="n">counter</span><span class="p">):</span>
    <span class="n">s</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span><span class="nb">sum</span><span class="p">(</span><span class="n">counter</span><span class="o">.</span><span class="n">values</span><span class="p">()))</span>
    <span class="k">return</span> <span class="p">[(</span><span class="n">c</span><span class="p">,</span><span class="n">cnt</span><span class="o">/</span><span class="n">s</span><span class="p">)</span> <span class="k">for</span> <span class="n">c</span><span class="p">,</span><span class="n">cnt</span> <span class="ow">in</span> <span class="n">counter</span><span class="o">.</span><span class="n">items</span><span class="p">()]</span>
  <span class="n">outlm</span> <span class="o">=</span> <span class="p">{</span><span class="n">hist</span><span class="p">:</span><span class="n">normalize</span><span class="p">(</span><span class="n">chars</span><span class="p">)</span> <span class="k">for</span> <span class="n">hist</span><span class="p">,</span> <span class="n">chars</span> <span class="ow">in</span> <span class="n">lm</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span>
  <span class="k">return</span> <span class="n">outlm</span></code></pre></figure>

<p><code class="highlighter-rouge">fname</code> is a file to read the characters from. <code class="highlighter-rouge">order</code> is the history size to consult. Note that we pad the data with leading <code class="highlighter-rouge">~</code> so that we also learn how to start.</p>

<p>Now you can train a language model.  First grab some text like this corpus of Shakespeare:</p>

<div class="language-bash highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="nv">$ </span>wget http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt<span class="sb">`</span>
</code></pre></div></div>

<p>Now train the model:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="n">lm</span> <span class="o">=</span> <span class="n">train_char_lm</span><span class="p">(</span><span class="s">"shakespeare_input.txt"</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">4</span><span class="p">)</span></code></pre></figure>

<h3 id="phello-world">P(hello world)</h3>

<p>Ok. Now we can look-up the probability of the next letter given some history.  Here are some example queries:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="n">lm</span><span class="p">[</span><span class="s">'hell'</span><span class="p">]</span>
<span class="p">[(</span><span class="s">'!'</span><span class="p">,</span> <span class="mf">0.06912442396313365</span><span class="p">),</span> <span class="p">(</span><span class="s">' '</span><span class="p">,</span> <span class="mf">0.22119815668202766</span><span class="p">),</span> <span class="p">(</span><span class="s">"'"</span><span class="p">,</span> <span class="mf">0.018433179723502304</span><span class="p">),</span> <span class="p">(</span><span class="s">'i'</span><span class="p">,</span> <span class="mf">0.03225806451612903</span><span class="p">),</span> <span class="p">(</span><span class="s">'</span><span class="se">\n</span><span class="s">'</span><span class="p">,</span> <span class="mf">0.018433179723502304</span><span class="p">),</span> <span class="p">(</span><span class="s">'-'</span><span class="p">,</span> <span class="mf">0.059907834101382486</span><span class="p">),</span> <span class="p">(</span><span class="s">','</span><span class="p">,</span> <span class="mf">0.20276497695852536</span><span class="p">),</span> <span class="p">(</span><span class="s">'o'</span><span class="p">,</span> <span class="mf">0.15668202764976957</span><span class="p">),</span> <span class="p">(</span><span class="s">'.'</span><span class="p">,</span> <span class="mf">0.1336405529953917</span><span class="p">),</span> <span class="p">(</span><span class="s">'s'</span><span class="p">,</span> <span class="mf">0.009216589861751152</span><span class="p">),</span> <span class="p">(</span><span class="s">';'</span><span class="p">,</span> <span class="mf">0.027649769585253458</span><span class="p">),</span> <span class="p">(</span><span class="s">':'</span><span class="p">,</span> <span class="mf">0.018433179723502304</span><span class="p">),</span> <span class="p">(</span><span class="s">'?'</span><span class="p">,</span> <span class="mf">0.03225806451612903</span><span class="p">)]</span></code></pre></figure>

<p>Actually, let’s pretty print the output, and sort the letters based on their probabilities.</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="kn">import</span> <span class="nn">pprint</span>
<span class="kn">import</span> <span class="nn">operator</span>

<span class="k">def</span> <span class="nf">print_probs</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="n">history</span><span class="p">):</span>
    <span class="n">probs</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">lm</span><span class="p">[</span><span class="n">history</span><span class="p">],</span><span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:(</span><span class="o">-</span><span class="n">x</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="n">x</span><span class="p">[</span><span class="mi">0</span><span class="p">]))</span>
    <span class="n">pp</span> <span class="o">=</span> <span class="n">pprint</span><span class="o">.</span><span class="n">PrettyPrinter</span><span class="p">()</span>
    <span class="n">pp</span><span class="o">.</span><span class="n">pprint</span><span class="p">(</span><span class="n">probs</span><span class="p">)</span></code></pre></figure>

<p>OK, print again:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="n">print_probs</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="s">"hell"</span><span class="p">)</span>
<span class="p">[(</span><span class="s">' '</span><span class="p">,</span> <span class="mf">0.22119815668202766</span><span class="p">),</span>
 <span class="p">(</span><span class="s">','</span><span class="p">,</span> <span class="mf">0.20276497695852536</span><span class="p">),</span>
 <span class="p">(</span><span class="s">'o'</span><span class="p">,</span> <span class="mf">0.15668202764976957</span><span class="p">),</span>
 <span class="p">(</span><span class="s">'.'</span><span class="p">,</span> <span class="mf">0.1336405529953917</span><span class="p">),</span>
 <span class="p">(</span><span class="s">'!'</span><span class="p">,</span> <span class="mf">0.06912442396313365</span><span class="p">),</span>
 <span class="p">(</span><span class="s">'-'</span><span class="p">,</span> <span class="mf">0.059907834101382486</span><span class="p">),</span>
 <span class="p">(</span><span class="s">'?'</span><span class="p">,</span> <span class="mf">0.03225806451612903</span><span class="p">),</span>
 <span class="p">(</span><span class="s">'i'</span><span class="p">,</span> <span class="mf">0.03225806451612903</span><span class="p">),</span>
 <span class="p">(</span><span class="s">';'</span><span class="p">,</span> <span class="mf">0.027649769585253458</span><span class="p">),</span>
 <span class="p">(</span><span class="s">'</span><span class="se">\n</span><span class="s">'</span><span class="p">,</span> <span class="mf">0.018433179723502304</span><span class="p">),</span>
 <span class="p">(</span><span class="s">"'"</span><span class="p">,</span> <span class="mf">0.018433179723502304</span><span class="p">),</span>
 <span class="p">(</span><span class="s">':'</span><span class="p">,</span> <span class="mf">0.018433179723502304</span><span class="p">),</span>
 <span class="p">(</span><span class="s">'s'</span><span class="p">,</span> <span class="mf">0.009216589861751152</span><span class="p">)]</span></code></pre></figure>

<p>This means that <code class="highlighter-rouge">hell</code> can be followed by any of these dozen characters:</p>

<p><code class="highlighter-rouge"> ,o.!-?i;\n':s</code></p>

<p>and that the probability of <code class="highlighter-rouge">o</code> given <code class="highlighter-rouge">hell</code> is 15.7%, <script type="math/tex">p(o \mid hell)=0.157</script>.  The most probable character to see after <code class="highlighter-rouge">hell</code> is a space, <script type="math/tex">p(o \mid hell)=0.221</script>.</p>

<p>The distribution of letters that occur after <code class="highlighter-rouge">worl</code> is different than the distribution of letters that occur after <code class="highlighter-rouge">hell</code>.  Here is that distribution:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="n">print_probs</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="s">"worl"</span><span class="p">)</span>
<span class="p">[(</span><span class="s">'d'</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">)]</span></code></pre></figure>

<p>What does that mean?  It means that in our corpus, the only possible continuation that we observed for <code class="highlighter-rouge">worl</code> was the letter <code class="highlighter-rouge">d</code>, and we assign 100% of probability mass to it, <script type="math/tex">p(d \mid worl)=1.0</script>.</p>

<h3 id="lets-write-some-shakespeare">Let’s write some Shakespeare!</h3>

<p>Generating text with the model is simple. To generate a letter, we will look up the last <code class="highlighter-rouge">n</code> characters, and then sample a random letter based on the probability distribution for those letters.   Here’s Yoav’s code for that:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="k">def</span> <span class="nf">generate_letter</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="n">history</span><span class="p">,</span> <span class="n">order</span><span class="p">):</span>
  <span class="s">''' Randomly chooses the next letter using the language model.
  
  Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of text at least 'order' long.
    order: The length of the n-grams in the language model.
    
  Returns: 
    A letter
  '''</span>
  
  <span class="n">history</span> <span class="o">=</span> <span class="n">history</span><span class="p">[</span><span class="o">-</span><span class="n">order</span><span class="p">:]</span>
  <span class="n">dist</span> <span class="o">=</span> <span class="n">lm</span><span class="p">[</span><span class="n">history</span><span class="p">]</span>
  <span class="n">x</span> <span class="o">=</span> <span class="n">random</span><span class="p">()</span>
  <span class="k">for</span> <span class="n">c</span><span class="p">,</span><span class="n">v</span> <span class="ow">in</span> <span class="n">dist</span><span class="p">:</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">x</span> <span class="o">-</span> <span class="n">v</span>
    <span class="k">if</span> <span class="n">x</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">:</span> <span class="k">return</span> <span class="n">c</span></code></pre></figure>

<p>To generate a passage of text, we just seed it with the initial history and run letter generation in a loop, updating the history at each turn.  We’ll stop generating after a specified number of letters.</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="k">def</span> <span class="nf">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="n">order</span><span class="p">,</span> <span class="n">nletters</span><span class="o">=</span><span class="mi">500</span><span class="p">):</span>
  <span class="s">'''Generates a bunch of random text based on the language model.
  
  Inputs:
  lm: The output from calling train_char_lm.
  order: The length of the n-grams in the language model.
  nletters: the number of characters worth of text to generate
  
  Returns: 
    A letter  
  '''</span>
  <span class="n">history</span> <span class="o">=</span> <span class="s">"~"</span> <span class="o">*</span> <span class="n">order</span>
  <span class="n">out</span> <span class="o">=</span> <span class="p">[]</span>
  <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">nletters</span><span class="p">):</span>
    <span class="n">c</span> <span class="o">=</span> <span class="n">generate_letter</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="n">history</span><span class="p">,</span> <span class="n">order</span><span class="p">)</span>
    <span class="n">history</span> <span class="o">=</span> <span class="n">history</span><span class="p">[</span><span class="o">-</span><span class="n">order</span><span class="p">:]</span> <span class="o">+</span> <span class="n">c</span>
    <span class="n">out</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">c</span><span class="p">)</span>
  <span class="k">return</span> <span class="s">""</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">out</span><span class="p">)</span></code></pre></figure>

<p>Now, try generating some Shakespeare with different order n-gram models.  You should try running the following commands.</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="n">lm</span> <span class="o">=</span> <span class="n">train_char_lm</span><span class="p">(</span><span class="s">"shakespeare_input.txt"</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">2</span><span class="p">))</span>


<span class="o">&gt;&gt;&gt;</span> <span class="n">lm</span> <span class="o">=</span> <span class="n">train_char_lm</span><span class="p">(</span><span class="s">"shakespeare_input.txt"</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">3</span><span class="p">))</span>


<span class="o">&gt;&gt;&gt;</span> <span class="n">lm</span> <span class="o">=</span> <span class="n">train_char_lm</span><span class="p">(</span><span class="s">"shakespeare_input.txt"</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">4</span><span class="p">)</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">4</span><span class="p">))</span>


<span class="o">&gt;&gt;&gt;</span> <span class="n">lm</span> <span class="o">=</span> <span class="n">train_char_lm</span><span class="p">(</span><span class="s">"shakespeare_input.txt"</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">7</span><span class="p">)</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">7</span><span class="p">))</span></code></pre></figure>

<p>What do you think?  Is it as good as <a href="https://www.youtube.com/watch?v=no_elVGGgW8">1000 monkeys working at 1000 typewriters</a>?</p>

<h3 id="what-the-f">What the F?</h3>

<p>Try generating a bunch of short passages:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">First</span><span class="p">,</span> <span class="ow">and</span> <span class="n">quence</span>
<span class="n">Shall</span> <span class="n">we</span> <span class="n">gave</span> <span class="n">it</span><span class="o">.</span> <span class="n">Now</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">First</span> <span class="n">die</span><span class="o">.</span>

<span class="n">KING</span> <span class="n">OF</span> <span class="n">FRANCE</span><span class="p">:</span>
<span class="n">I</span> <span class="n">prithee</span><span class="p">,</span> 
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">First</span> <span class="n">marriage</span><span class="p">,</span>
<span class="n">And</span> <span class="n">scarce</span> <span class="n">it</span><span class="p">:</span> <span class="n">wretches</span> 
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">First</span><span class="p">,</span> <span class="n">the</span> <span class="n">midsummer</span><span class="p">;</span>
<span class="n">We</span> <span class="n">make</span> <span class="n">us</span> <span class="n">allia</span> <span class="n">s</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">First</span> <span class="n">choose</span>
<span class="n">Which</span> <span class="n">now</span><span class="p">,</span>
<span class="n">Where</span> <span class="n">like</span> <span class="n">thee</span><span class="o">.</span>

<span class="o">&gt;&gt;&gt;</span> <span class="n">lm</span> <span class="o">=</span> <span class="n">train_char_lm</span><span class="p">(</span><span class="s">"shakespeare_input.txt"</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">4</span><span class="p">)</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">First</span> <span class="n">blood</span>
<span class="n">assurance</span>
<span class="n">To</span> <span class="n">grace</span> <span class="ow">and</span> <span class="n">leade</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">First</span><span class="p">,</span> <span class="n">are</span> <span class="n">almightly</span><span class="p">,</span>
<span class="n">Am</span> <span class="n">I</span> <span class="n">to</span> <span class="n">bedew</span> <span class="n">the</span> 
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">First</span> <span class="n">Senato</span><span class="p">,</span> <span class="n">come</span> <span class="n">unexamination</span> <span class="n">hast</span> <span class="n">br</span>

<span class="o">&gt;&gt;&gt;</span> <span class="n">lm</span> <span class="o">=</span> <span class="n">train_char_lm</span><span class="p">(</span><span class="s">"shakespeare_input.txt"</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">Firm</span>
<span class="n">Histed</span> <span class="n">mor</span> <span class="n">ituffe</span> <span class="n">bonguis</span> <span class="n">hon</span> <span class="n">tract</span>
<span class="o">&gt;&gt;&gt;</span> <span class="k">print</span><span class="p">(</span><span class="n">generate_text</span><span class="p">(</span><span class="n">lm</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">40</span><span class="p">))</span>
<span class="n">Fir</span> <span class="n">my</span> <span class="n">fat</span><span class="p">,</span>
<span class="n">Forromfor</span> <span class="n">intre</span> <span class="n">You</span> <span class="n">to</span> <span class="n">lor</span> <span class="n">c</span></code></pre></figure>

<p>Do you notice anything?  <em>They all start with F!</em>  In fact, after we hit a certain order, the first word is always <em>First</em>?  Why is that?  Is the model trying to be clever?  First, generate the word <em>First</em>. Explain what is going on in your writeup.</p>

<h2 id="part-2-perplexity-smoothing-back-off-and-interpolation">Part 2: Perplexity, smoothing, back-off and interpolation</h2>

<p>In this part of the assignment, you’ll adapt Yoav’s code in order to implement several of the  techniques described in <a href="https://web.stanford.edu/~jurafsky/slp3/4.pdf">Section 4.2 of the Jurafsky and Martin textbook</a>.</p>

<h3 id="perplexity">Perplexity</h3>

<p>How do we know whether a LM is good? There are two basic approaches:</p>
<ol>
  <li>Task-based evaluation (also known as extrinsic evaluation), where we use the LM as part of some other task, like automatic speech recognition, or spelling correcktion, or an OCR system that tries to covert a professor’s messy handwriting into text.</li>
  <li>Intrinsic evaluation.  Intrinsic evaluation tries to directly evalute the goodness of the language model by seeing how well the probability distributions that it estimates are able to explain some previously unseen test set.</li>
</ol>

<p>Here’s what the textbook says:</p>

<blockquote>
  <p>For an intrinsic evaluation of a language model we need a test set. As with many of the statistical models in our field, the probabilities of an N-gram model come from the corpus it is trained on, the training set or training corpus. We can then measure the quality of an N-gram model by its performance on some unseen data called the test set or test corpus. We will also sometimes call test sets and other datasets that are not in our training sets held out corpora because we hold them out from the training data.</p>
</blockquote>

<blockquote>
  <p>So if we are given a corpus of text and want to compare two different N-gram models, we divide the data into training and test sets, train the parameters of both models on the training set, and then compare how well the two trained models fit the test set.</p>
</blockquote>

<blockquote>
  <p>But what does it mean to “fit the test set”? The answer is simple: whichever model assigns a higher probability to the test set is a better model.</p>
</blockquote>

<p>We’ll implement the most common method for intrinsic metric of language models: <em>perplexity</em>.  The perplexity of a language model on a test set is the inverse probability of the test set, normalized by the number of words. For a test set <script type="math/tex">W = w_1 w_2 ... w_N</script>:</p>

<script type="math/tex; mode=display">Perplexity(W) = P(w_1 w_2 ... w_N)^{-\frac{1}{N}}</script>

<script type="math/tex; mode=display">= \sqrt[N]{\frac{1}{P(w_1 w_2 ... w_N)}}</script>

<script type="math/tex; mode=display">= \sqrt[N]{\prod_{i=1}^{N}{\frac{1}{P(w_i \mid w_1 ... w_{i-1})}}}</script>

<p>OK - let’s implement it. Here’s a possible function signature for perplexity.  (We might update it during class on Wednesday).  Give it a go.</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="k">def</span> <span class="nf">perplexity</span><span class="p">(</span><span class="n">test_filename</span><span class="p">,</span> <span class="n">lm</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">4</span><span class="p">):</span>
  <span class="s">'''Computes the perplexity of a text file given the language model.
  
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model.
  '''</span>
  <span class="n">test</span> <span class="o">=</span> <span class="nb">open</span><span class="p">(</span><span class="n">test_filename</span><span class="p">)</span><span class="o">.</span><span class="n">read</span><span class="p">()</span>
  <span class="n">pad</span> <span class="o">=</span> <span class="s">"~"</span> <span class="o">*</span> <span class="n">order</span>
  <span class="n">test</span> <span class="o">=</span> <span class="n">pad</span> <span class="o">+</span> <span class="n">data</span>
  
  <span class="c"># TODO: YOUR CODE HERE</span></code></pre></figure>

<p>A couple of things to keep in mind:</p>

<ol>
  <li>Remember to pad the front of the file</li>
  <li>Numeric underflow is going to be a problem, so consider using logs.</li>
  <li>Perplexity is undefined if LM assigns any zero probabilities to the test set. In that case your code should return positive infinity - <code class="highlighter-rouge">float("inf")</code>.</li>
  <li>On your unsmoothed models, you’ll definitely get some zero probabilities for the test set.  To test you code, you should try computing perplexity on the trianing set, and you should compute perplexity for your LMs that use smoothing and interpolation.</li>
</ol>

<h4 id="in-your-report">In your report:</h4>
<p>Discuss the perplexity for text that is similar and different from Shakespeare’s plays. We provide you <a href="/downloads/hw5/test_data.zip">two dev text files</a>, a New York Times article and several of Shakespeare’s sonnets, but feel free to experiment with your own text.</p>

<p>Note: you may want to create a smoothed language model before calculating perplexity, otherwise you will get a perplexity of 0.</p>

<h3 id="laplace-smoothing-and-add-k-smoothing">Laplace Smoothing and Add-k Smoothing</h3>

<p>Laplace Smoothing is described in section 4.4.1.  Laplace smoothing  adds one to each count (hence its alternate name <em>add-one smoothing</em>).   Since there are <em>V</em> words in the vocabulary and each one was incremented, we also need to adjust the denominator to take into account the extra V observations.</p>

<script type="math/tex; mode=display">P_{Laplace}(w_i) = \frac{count_i + 1}{N+V}</script>

<p>A variant of Laplace smoothing is called <em>Add-k smoothing</em> or <em>Add-epsilon smoothing</em>.  This is described in section Add-k 4.4.2.  Let’s change the function definition of <code class="highlighter-rouge">train_char_lm</code> so that it takes a new argument, <code class="highlighter-rouge">add_k</code>, which specifies how much to add.  By default, we’ll set it to one, so that it acts like Laplace smoothing:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="k">def</span> <span class="nf">train_char_lm</span><span class="p">(</span><span class="n">fname</span><span class="p">,</span> <span class="n">order</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">add_k</span><span class="o">=</span><span class="mi">1</span><span class="p">):</span>
    <span class="c"># Your code here...</span></code></pre></figure>

<h3 id="interpolation">Interpolation</h3>

<p>Next, let’s implement interpolation.  The idea here is to calculate the higher order n-gram probabilities also combining the probabilities for lower-order n-gram models.  Like smoothing, this helps us avoid the problem of zeros if we haven’t observed the longer sequence in our training data.  Here’s the math:</p>

<script type="math/tex; mode=display">P_{backoff}(w_i|w_{i−2} w_{i−1}) = \lambda_1 P(w_i|w_{i−2} w_{i−1}) + \lambda_2 P(w_i|w_{i−1}) + \lambda_3 P(w_i)</script>

<p>where $\lambda_1 + \lambda_2 + \lambda_3 = 1$.</p>

<p>Now, write a back-off function:</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="k">def</span> <span class="nf">calculate_prob_with_backoff</span><span class="p">(</span><span class="n">char</span><span class="p">,</span> <span class="n">history</span><span class="p">,</span> <span class="n">lms</span><span class="p">,</span> <span class="n">lambdas</span><span class="p">):</span>
  <span class="s">'''Uses interpolation to compute the probability of char given a series of 
     language models trained with different length n-grams.

   Inputs:
     char: Character to compute the probability of.
     history: A sequence of previous text.
     lms: A list of language models, outputted by calling train_char_lm.
     lambdas: A list of weights for each lambda model. These should sum to 1.
    
  Returns:
    Probability of char appearing next in the sequence.
  '''</span> 
  <span class="c"># TODO: YOUR CODE HERE</span>
  <span class="k">pass</span></code></pre></figure>

<p>You should also write a helper function to set the lambdas.  Here’s a function definition that gives you access to a development set.  You can also experiment with setting them manually.</p>

<figure class="highlight"><pre><code class="language-python" data-lang="python"><span class="c"># returns a list of lambda values that weight the contribution of n-gram model</span>
<span class="k">def</span> <span class="nf">set_lambdas</span><span class="p">(</span><span class="n">lms</span><span class="p">,</span> <span class="n">dev_filename</span><span class="p">):</span>
  <span class="s">'''Returns a list of lambda values that weight the contribution of each n-gram model

  This can either be done heuristically or by using a development set.

  Inputs:
    lms: A list of language models, outputted by calling train_char_lm.
    dev_filename: Path to a development text file to optionally use for tuning the lmabdas. 

  Returns:
    Probability of char appearing next in the sequence.
  '''</span>
  <span class="c"># TODO: YOUR CODE HERE</span>
  <span class="k">pass</span></code></pre></figure>

<h4 id="in-your-report-1">In your report:</h4>
<p>Experiment with a couple different lambdas and values of k, and discuss their effects.</p>

<h2 id="part-3-text-classification-using-lms">Part 3: Text Classification using LMs</h2>

<p>Language models can be applied to text classification. If we want to classify a text <script type="math/tex">D</script> into a category <script type="math/tex">c \in C={c_1, ..., c_N}</script>. We can pick the category <script type="math/tex">c</script> that has the largest posterior probability given the text. That is,</p>

<script type="math/tex; mode=display">c^* = arg max_{c \in C} P(c|D)</script>

<p>Using Bayes rule, this can be rewritten as:</p>

<script type="math/tex; mode=display">c^* = arg max_{c \in C} P(D|c) P(c)</script>

<p>If we assume that all classes are equally likely, then we can just drop the <script type="math/tex">P(c)</script> term:</p>

<script type="math/tex; mode=display">= arg max_{c \in C} P(D|c)</script>

<p>Here <script type="math/tex">P(D \mid c)</script> is the likelihood of <script type="math/tex">D</script> under category <script type="math/tex">c</script>, which can be computed by training language models for all texts associated with category <script type="math/tex">c</script>.  This technique of text classification is drawn from <a href="http://www.aclweb.org/anthology/E/E03/E03-1053.pdf">literature on authorship identification</a>, where the approach is to learn a separate language model for each author, by training on a data set from that author. Then, to categorize a new text D, they use each language model to calculate the likelihood of D under that model, and pick the  category that assigns the highest probability to D.</p>

<p>Try it!  We have provided you training and validation datsets consisting of the names of cities. The task is to predict the country a city is in. The following countries are including in the dataset.</p>
<div class="highlighter-rouge"><div class="highlight"><pre class="highlight"><code>af	Afghanistan
cn	China
de	Germany
fi	Finland
fr	France
in	India
ir	Iran
pk	Pakistan
za	South Africa
</code></pre></div></div>

<p><strong>Leaderboard</strong></p>

<p>We’ll set up a leaderboard for the text classification task.  Your job is to configure a set of language models that perform the best on the text classification task. We will use the city names dataset, which you should have already downloaded. The test set has one unlabeled city name per line. Your code should output a file <code class="highlighter-rouge">labels.txt</code> with one two-letter country code per line.</p>

<p>In next week’s assignment, you will use a recurrent neural network on the same dataset in order to compare performance.</p>

<h4 id="in-your-report-2">In your report:</h4>
<p>Describe the parameters of your final leaderboard model and any experimentation you did before settling on it.</p>

<h2 id="deliverables">Deliverables</h2>
<div class="alert alert-warning">
  <p>Here are the deliverables that you will need to submit:</p>
  <ul>
    <li>writeup.pdf</li>
    <li>code (.zip). It should be written in Python 3 and include a README.txt briefly explaining how to run it.</li>
    <li><code class="highlighter-rouge">labels.txt</code> predictions for leaderboard.</li>
  </ul>
</div>

<h2 id="recommended-readings">Recommended readings</h2>

<table>
   
    <tr>
      <td>
	
		<a href="https://web.stanford.edu/~jurafsky/slp3/4.pdf">Language Modeling with N-grams.</a>
        
	Dan Jurafsky and James H. Martin.
	Speech and Language Processing (3rd edition draft)  .

	
		
</td></tr>
  
    <tr>
      <td>
	
		<a href="http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139">The Unreasonable Effectiveness of Character-level Language Models.</a>
        
	Yoav Goldberg.
	Response to Andrej Karpathy's blog post.  2015.

	
		
</td></tr>
  
    <tr>
      <td>
	
		<a href="http://www.aclweb.org/anthology/E/E03/E03-1053.pdf">Language Independent Authorship Attribution using Character Level Language Models.</a>
        
	Fuchun Pen, Dale Schuurmans, Vlado Keselj, Shaojun Wan.
	EACL  2003.

	
		
</td></tr>
  
</table>

        </div>
      </div>
      
      </div>

    <footer class="text-center text-muted">
      <hr>
      
      <p>This assignment is based on <a href="http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139">The Unreasonable Effectiveness of Character-level Language Models</a> by Yoav Goldberg. Daphne Ippolito, John Hewitt, and Chris Callison-Burch adapted their work into a homework assignment for UPenn’s CIS 530 class in Spring 2018.</p>
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
