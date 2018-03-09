<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Homework 8</title>
        <link rel="stylesheet" type="text/css" href="resources/homework.css">
        <link rel="stylesheet" type="text/css" href="resources/prism.css">
        <script type="text/javascript" src="resources/prism.js"></script>
        <script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
        <script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});</script>
    </head>
    <body>
        <div class="content language-markup">
            <h1>CIS 521: Homework 8<span class="points" data-value="50"></span></h1>
            <table class="dates">
                <tr>
                    <td>Release Date</td>
                    <td>Tuesday, November 21, 2017</td>
                </tr>
                <tr>
                    <td>Due Date</td>
                    <td>11:59 pm on Thursday, November 30, 2017</td>
                </tr>
            </table>
            <section>
                <h3>Instructions</h3>
                <p>In this assignment, you will gain experience working with hidden Markov models for part-of-speech tagging.</p>
                <p>A skeleton file <code>homework8.py</code> containing empty definitions for each question has been provided. Since portions of this assignment will be graded automatically, none of the names or function signatures in this file should be modified. However, you are free to introduce additional variables or functions if needed.</p>
                <p>You may import definitions from any standard Python library, and are encouraged to do so in case you find yourself reinventing the wheel.</p>
                <p>You will find that in addition to a problem specification, most programming questions also include a pair of examples from the Python interpreter. These are meant to illustrate typical use cases, and should not be taken as comprehensive test suites.</p>
                <p>You are strongly encouraged to follow the Python style guidelines set forth in <a href="http://legacy.python.org/dev/peps/pep-0008/">PEP 8</a>, which was written in part by the creator of Python. However, your code will not be graded for style.</p>
                <p>Once you have completed the assignment, you should submit your file on Eniac using the following <code>turnin</code> command, where the flags <code>-c</code> and <code>-p</code> stand for "course" and "project", respectively.</p>
                <pre><code>turnin -c cis521 -p hw8 homework8.py</code></pre>
                <p>You may submit as many times as you would like before the deadline, but only the last submission will be saved. To view a detailed listing of the contents of your most recent submission, you can use the following command, where the flag <code>-v</code> stands for "verbose".</p>
                <pre><code>turnin -c cis521 -p hw8 -v</code></pre>
            </section>
            <section class="numbered">
                <h3>Hidden Markov Models<span class="points" data-value="45"></span></h3>
                <p>In this section, you will develop a hidden Markov model for part-of-speech (POS) tagging, using the Brown corpus as training data. The tag set used in this assignment will be the <a href="http://www.petrovi.de/data/universal.pdf">universal POS tag set</a>, which is composed of the twelve POS tags <span class="small-caps">Noun</span> (noun), <span class="small-caps">Verb</span> (verb), <span class="small-caps">Adj</span> (adjective), <span class="small-caps">Adv</span> (adverb), <span class="small-caps">Pron</span> (pronoun), <span class="small-caps">Det</span> (determiner or article), <span class="small-caps">Adp</span> (preposition or postposition), <span class="small-caps">Num</span> (numeral), <span class="small-caps">Conj</span> (conjunction), <span class="small-caps">Prt</span> (particle), '<span class="small-caps">.</span>' (punctuation mark), and <span class="small-caps">X</span> (other).</p>
                <p>As in previous assignments, your use of external code should be limited to built-in Python modules, which excludes packages such as NumPy and NLTK.</p>
                <ol>
                    <li>
                        <p><span class="points" data-value="5"></span>Write a function <code class="language-python">load_corpus(path)</code> that loads the corpus at the given path and returns it as a list of POS-tagged sentences. Each line in the file should be treated as a separate sentence, where sentences consist of sequences of whitespace-separated strings of the form <code class="language-python">"token=POS"</code>. Your function should return a list of lists, with individual entries being $2$-tuples of the form (token,&nbsp;POS).</p>
                        <table class="codeGroup">
                            <tr>
                                <td>
                                    <pre><code class="language-python">
&gt;&gt;&gt; c = load_corpus("brown_corpus.txt")
&gt;&gt;&gt; c[1402]
[('It', 'PRON'), ('made', 'VERB'),
 ('him', 'PRON'), ('human', 'NOUN'),
 ('.', '.')]
</code></pre>
                                </td>
                                <td>
                                    <pre><code class="language-python">
&gt;&gt;&gt; c = load_corpus("brown_corpus.txt")
&gt;&gt;&gt; c[1799]
[('The', 'DET'), ('prospects', 'NOUN'),
 ('look', 'VERB'), ('great', 'ADJ'),
 ('.', '.')]
</code></pre>
                                </td>
                            </tr>
                        </table>
                    </li>
                    <li>
                        <span class="points" data-value="10"></span>In the <code class="language-python">Tagger</code> class, write an initialization method <code class="language-python">__init__(self, sentences)</code> which takes a list of sentences in the form produced by <code class="language-python">load_corpus(path)</code> as input and initializes the internal variables needed for the POS tagger. In particular, if $\{ t_1, t_2, \cdots, t_n \}$ denotes the set of tags and $\{ w_1, w_2, \cdots, w_m \}$ denotes the set of tokens found in the input sentences, you should at minimum compute:
                        <ul>
                            <li>
                                <p>The initial tag probabilities $\pi(t_i)$ for $1 \le i \le n$, where $\pi(t_i)$ is the probability that a sentence begins with tag $t_i$.</p>
                            </li>
                            <li>
                                <p>The transition probabilities $a(t_i \to t_j)$ for $1 \le i, j \le n$, where $a(t_i \to t_j)$ is the probability that tag $t_j$ occurs after tag $t_i$.</p>
                            </li>
                            <li>
                                <p>The emission probabilities $b(t_i \to w_j)$ for $1 \le i \le n$ and $1 \le j \le m$, where $b(t_i \to w_j)$ is the probability that token $w_j$ is generated given tag $t_i$.</p>
                            </li>
                        </ul>
                        It is imperative that you use Laplace smoothing where appropriate to ensure that your system can handle novel inputs, but the exact manner in which this is done is left up to you as a design decision. Your initialization method should take no more than a few seconds to complete when given the full Brown corpus as input.
                    <li>
                        <p><span class="points" data-value="10"></span>In the <code class="language-python">Tagger</code> class, write a method <code class="language-python">most_probable_tags(self, tokens)</code> which returns the list of the most probable tags corresponding to each input token. In particular, the most probable tag for a token $w_j$ is defined to be the tag with index $i^* = \mathrm{argmax}_i \, b(t_i \to w_j)$.</p>
                        <table class="codeGroup">
                            <tr>
                                <td>
                                    <pre><code class="language-python">
&gt;&gt;&gt; c = load_corpus("brown_corpus.txt")
&gt;&gt;&gt; t = Tagger(c)
&gt;&gt;&gt; t.most_probable_tags(
...   ["The", "man", "walks", "."])
['DET', 'NOUN', 'VERB', '.']
</code></pre>
                                </td>
                                <td>
                                    <pre><code class="language-python">
&gt;&gt;&gt; c = load_corpus("brown_corpus.txt")
&gt;&gt;&gt; t = Tagger(c)
&gt;&gt;&gt; t.most_probable_tags(
...   ["The", "blue", "bird", "sings"])
['DET', 'ADJ', 'NOUN', 'VERB']
</code></pre>
                                </td>
                            </tr>
                        </table>
                    </li>
                    <li>
                        <p><span class="points" data-value="20"></span>In the <code class="language-python">Tagger</code> class, write a method <code class="language-python">viterbi_tags(self, tokens)</code> which returns the most probable tag sequence as found by Viterbi decoding. Recall from lecture that Viterbi decoding is a modification of the Forward algorithm, adapted to find the path of highest probability through the trellis graph containing all possible tag sequences. Computation will likely proceed in two stages: you will first compute the probability of the most likely tag sequence, and will then reconstruct the sequence which achieves that probability from end to beginning by tracing backpointers.</p>
                        <table class="codeGroup">
                            <tr>
                                <td>
                                    <pre><code class="language-python">
&gt;&gt;&gt; c = load_corpus("brown_corpus.txt")
&gt;&gt;&gt; t = Tagger(c)
&gt;&gt;&gt; s = "I am waiting to reply".split()
&gt;&gt;&gt; t.most_probable_tags(s)
['PRON', 'VERB', 'VERB', 'PRT', 'NOUN']
&gt;&gt;&gt; t.viterbi_tags(s)
['PRON', 'VERB', 'VERB', 'PRT', 'VERB']
</code></pre>
                                </td>
                                <td>
                                    <pre><code class="language-python">
&gt;&gt;&gt; c = load_corpus("brown_corpus.txt")
&gt;&gt;&gt; t = Tagger(c)
&gt;&gt;&gt; s = "I saw the play".split()
&gt;&gt;&gt; t.most_probable_tags(s)
['PRON', 'VERB', 'DET', 'VERB']
&gt;&gt;&gt; t.viterbi_tags(s)
['PRON', 'VERB', 'DET', 'NOUN']
</code></pre>
                                </td>
                            </tr>
                        </table>
                    </li>
                </ol>
            </section>
            <section class="numbered">
                <h3>Feedback<span class="points" data-value="5"></span></h3>
                <ol>
                    <li>
                        <p><span class="points" data-value="1"></span>Approximately how long did you spend on this assignment?</p>
                    </li>
                    <li>
                        <p><span class="points" data-value="2"></span>Which aspects of this assignment did you find most challenging? Were there any significant stumbling blocks?</p>
                    </li>
                    <li>
                        <p><span class="points" data-value="2"></span>Which aspects of this assignment did you like? Is there anything you would have changed?</p>
                    </li>
                </ol>
            </section>
        </div>
    </body>
</html>
