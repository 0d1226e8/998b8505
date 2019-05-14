## Bayesian Deep Learning <a class="anchor" id="bayesian-deep-learning"></a>

Deep learning is continuously transforming intelligent technology
across fields, from advancing medical diagnostics with complex data, to
enabling autonomous driving, to deciding high-stakes economic actions [[3]](../docs/Citations.md#LeCun-2015).
However, **deep learning models** struggle to inform their
users when they "don't know" -- in other words, these models
**fail to communicate their uncertainty in their predictions**. The implications
for deep models entrusted with life-or-death decisions are far-reaching:
experts in medical domains cannot know whether to trust the their
auto-diagnostics system, and passengers in self-driving vehicles cannot
be alerted to take control when the car does not know how to proceed.

Bayesian Deep Learning (BDL) offers a pragmatic approach to combining
Bayesian probability theory with modern deep learning.
BDL is concerned with the development of techniques and tools for
quantifying when deep models become uncertain, a process known as _inference_
in probabilistic modelling. BDL has already been demonstrated to play
a crucial role in applications such as
medical diagnostics [[4](../docs/Citations.md#Leibig-2017),[5](../docs/Citations.md#Kamnitsas-2017),[6](../docs/Citations.md#Ching-2017),[7](../docs/Citations.md#Worrall-2016)],
computer vision [[8](../docs/Citations.md#Kendall-2015), [9](../docs/Citations.md#Kendall-2016), [10](../docs/Citations.md#Kampffmeyer-2016)],
in the sciences [[11](../docs/Citations.md#Levasseur-2017), [12](../docs/Citations.md#McGibbon-2017)],
and autonomous driving [[13](../docs/Citations.md#Amodei-2016), [14](../docs/Citations.md#Kahn-2017), [8](../docs/Citations.md#Kendall-2015), [9](../docs/Citations.md#Kendall-2016), [15](../docs/Citations.md#Kendall-2017)].

### Importance of Benchmarks <a class="anchor" id="importance-of-benchmarks"></a>

Despite BDL's impact on a range of real-world applications and
the recent ideas and inference techniques suggested,
**the development of the field itself is impeded by the lack of realistic benchmarks to guide research**.
Evaluating new inference techniques on real-world applications often
requires expert domain knowledge, and current benchmarks used for the
development of new inference tools lack consideration for cost of development,
or for scalability to real world applications.

For example, benchmarks such as the toy UCI datasets [[16]](../docs/Citations.md#Hernández-2015)
consist of only evaluating root mean square error (RMSE) and log likelihood on
simple datasets with only a few hundred or thousand data
points, each with low input and output dimensionality. Such
evaluations are akin to the MNIST [[17]](../docs/Citations.md#LeCun-1998) evaluations in the early
days of deep learning. In current BDL research it is common
for researchers developing new inference techniques to evaluate their methods
with such toy benchmarks alone, ignoring the demands and constraints
of the real-world applications which make use of such tools [[18]](../docs/Citations.md#Mukhoti-2018).