---
Layout: default
title: Another page
---

[a](#settings)

[b](#jekyll-themes)

this is another page

[Go back](https://zhaoph2008.github.io/)
## Welcome to My Pages

<script>
            paper_count = 0

            function add_paper(title, authors, conference, link, bib, abstract, arxiv_link, code, data, slides, talk, msg) {
                list_entry = "<li style=\"font-size:18px\">"
                if (link != null)
                    list_entry += "<a href=\"" + link + "\">"
                list_entry += "<b>" + title + "</b>"
                if (link != null)
                    list_entry += "</a>"
                list_entry += "<br>" + authors + ".<br>" + conference + ".</li>"

                if (bib != null) {
                    list_entry += "<div id=\"bib" + paper_count + "\" style=\"display:none\">" + bib + "</div>"
                    list_entry += "<a href=\"javascript:copy(div" + paper_count + ",bib" + paper_count + ")\"> <span class=\"label label-success\">bib</span></a>"
                }

                if (abstract != null) {
                    list_entry += "<div id=\"abstract" + paper_count + "\" style=\"display:none\">" + abstract + "</div>"
                    list_entry += "<a href=\"javascript:copy(div" + paper_count + ",abstract" + paper_count + ")\"> <span class=\"label label-warning\">abstract</span></a>"
                }
                if (arxiv_link != null)
                    list_entry += " <a href=\"" + arxiv_link + "\"><span class=\"label label-primary\">arXiv</span></a>"

                if (code != null)
                    list_entry += " <a href=\"" + code + "\"><span class=\"label label-danger\">code/models</span></a>"

                if (data != null)
                    list_entry += " <a href=\"" + data + "\"><span class=\"label label-default\">data</span></a>"

                if (slides != null)
                    list_entry += " <a href=\"" + slides + "\"><span class=\"label label-info\">slides/poster</span></a>"

                if (talk != null)
                    list_entry += " <a href=\"" + talk + "\"><span class=\"label label-success\">talk</span></a>"

                list_entry += "<br>"

                if (msg != null)
                    list_entry += "<i>" + msg + "</i>"

                list_entry += "<div id=\"div" + paper_count + "\" style=\"font-size:15px\"></div><br>"

                document.write(list_entry)

                paper_count += 1
            }

            document.write("<h2>2021</h2>")
            document.write("<ul>")
            add_paper("SimCSE: Simple Contrastive Learning of Sentence Embeddings",
                "Tianyu Gao*, Xingcheng Yao*, <b>Danqi Chen</b>",
                "arXiv:2104.08821",
                "https://arxiv.org/pdf/2104.08821.pdf",
                "@article{gao2021simcse,<br>" +
                "&nbsp;&nbsp;&nbsp;title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},<br>" +
                "&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:2104.08821},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2021}<br>}",
                "This paper presents SimCSE, a simple contrastive learning framework that greatly advances the state-of-the-art sentence embeddings. We first describe an unsupervised approach, which takes an input sentence and predicts itself in a contrastive objective, with only standard dropout used as noise. This simple method works surprisingly well, performing on par with previous supervised counterparts. We hypothesize that dropout acts as minimal data augmentation and removing it leads to a representation collapse. Then, we draw inspiration from the recent success of learning sentence embeddings from natural language inference (NLI) datasets and incorporate annotated pairs from NLI datasets into contrastive learning by using \"entailment\" pairs as positives and \"contradiction\" pairs as hard negatives. We evaluate SimCSE on standard semantic textual similarity (STS) tasks, and our unsupervised and supervised models using BERT-base achieve an average of 74.5% and 81.6% Spearman's correlation respectively, a 7.9 and 4.6 points improvement compared to previous best results. We also show that contrastive learning theoretically regularizes pre-trained embeddings' anisotropic space to be more uniform, and it better aligns positive pairs when supervised signals are available.",
                "https://arxiv.org/abs/2104.08821",
                "https://github.com/princeton-nlp/SimCSE"
            )

            add_paper("Making Pre-trained Language Models Better Few-shot Learners",
                "Tianyu Gao*, Adam Fisch*, <b>Danqi Chen</b>",
                "In ACL 2021",
                "https://arxiv.org/pdf/2012.15723.pdf",
                "@inproceedings{gao2021making,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Making Pre-trained Language Models Better Few-shot Learners},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Gao, Tianyu and Fisch, Adam and Chen, Danqi},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Association for Computational Linguistics (ACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2021}<br>}",
                "The recent GPT-3 model (Brown et al., 2020) achieves remarkable few-shot performance solely by leveraging a natural-language prompt and a few task demonstrations as input context. Inspired by their findings, we study few-shot learning in a more practical scenario, where we use smaller language models for which fine-tuning is computationally efficient. We present LM-BFF--better few-shot fine-tuning of language models--a suite of simple and complementary techniques for fine-tuning language models on a small number of annotated examples. Our approach includes (1) prompt-based fine-tuning together with a novel pipeline for automating prompt generation; and (2) a refined strategy for dynamically and selectively incorporating demonstrations into each context. Finally, we present a systematic evaluation for analyzing few-shot performance on a range of NLP tasks, including classification and regression. Our experiments demonstrate that our methods combine to dramatically outperform standard fine-tuning procedures in this low resource setting, achieving up to 30% absolute improvement, and 11% on average across all tasks. Our approach makes minimal assumptions on task resources and domain expertise, and hence constitutes a strong task-agnostic method for few-shot learning.",
                "https://arxiv.org/abs/2012.15723",
                "https://github.com/princeton-nlp/LM-BFF",
                null,
                null,
                null,
                "Check out Tianyu's <a href=\"https://gaotianyu.xyz/prompting/\" style=\"color: #8C1515\"> blog post</a> on prompting and LM-BFF."
            )

            add_paper("Learning Dense Representations of Phrases at Scale",
                "Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, <b>Danqi Chen</b>",
                "In ACL 2021",
                "https://arxiv.org/pdf/2012.12624.pdf",
                "@inproceedings{lee2021learning,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Learning Dense Representations of Phrases at Scale},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Lee, Jinhyuk and Sung, Mujeen and Kang, Jaewoo and Chen, Danqi},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Association for Computational Linguistics (ACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2021}<br>}",
                "Open-domain question answering can be reformulated as a phrase retrieval problem, without the need for processing documents on-demand during inference (Seo et al., 2019). However, current phrase retrieval models heavily depend on their sparse representations while still underperforming retriever-reader approaches. In this work, we show for the first time that we can learn dense phrase representations alone that achieve much stronger performance in open-domain QA. Our approach includes (1) learning query-agnostic phrase representations via question generation and distillation; (2) novel negative-sampling methods for global normalization; (3) query-side fine-tuning for transfer learning. On five popular QA datasets, our model DensePhrases improves previous phrase retrieval models by 15%-25% absolute accuracy and matches the performance of state-of-the-art retriever-reader models. Our model is easy to parallelize due to pure dense representations and processes more than 10 questions per second on CPUs. Finally, we directly use our pre-indexed dense phrase representations for two slot filling tasks, showing the promise of utilizing DensePhrases as a dense knowledge base for downstream tasks.",
                "https://arxiv.org/abs/2012.12624",
                "https://github.com/princeton-nlp/DensePhrases",
                null,
                null,
                null,
                "You can try out the <a href=\"http://densephrases.korea.ac.kr/\" style=\"color: #8C1515\">demo</a> of DensePhrases!"
            )

            add_paper("A Frustratingly Easy Approach for Entity and Relation Extraction",
                "Zexuan Zhong, <b>Danqi Chen</b>",
                "In NAACL 2021",
                "https://arxiv.org/pdf/2010.12812.pdf",
                "@inproceedings{zhong2021frustratingly,<br>" +
                "&nbsp;&nbsp;&nbsp;title={A Frustratingly Easy Approach for Entity and Relation Extraction},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Zhong, Zexuan and Chen, Danqi},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={North American Association for Computational Linguistics (NAACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2021}<br>}",
                "End-to-end relation extraction aims to identify named entities and extract relations between them. Most recent work models these two subtasks jointly, either by casting them in one structured prediction framework, or performing multi-task learning through shared representations. In this work, we present a simple pipelined approach for entity and relation extraction, and establish the new state-of-the-art on standard benchmarks (ACE04, ACE05 and SciERC), obtaining a 1.7%-2.8% absolute improvement in relation F1 over previous joint models with the same pre-trained encoders. Our approach essentially builds on two independent encoders and merely uses the entity model to construct the input for the relation model. Through a series of careful examinations, we validate the importance of learning distinct contextual representations for entities and relations, fusing entity information early in the relation model, and incorporating global context. Finally, we also present an efficient approximation to our approach which requires only one pass of both entity and relation encoders at inference time, achieving an 8-16× speedup with a slight reduction in accuracy.",
                "https://arxiv.org/abs/2010.12812",
                "https://github.com/princeton-nlp/PURE",
                null,
                "https://github.com/princeton-nlp/PURE/blob/main/slides/slides.pdf"
            )

            add_paper("Factual Probing Is [MASK]: Learning vs. Learning to Recall",
                "Zexuan Zhong*, Dan Friedman*, <b>Danqi Chen</b>",
                "In NAACL 2021",
                "https://arxiv.org/pdf/2104.05240.pdf",
                "@inproceedings{zhong2021factual,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Factual Probing Is[MASK]: Learning vs. Learning to Recall},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Zhong, Zexuan and Friedman, Dan and Chen, Danqi},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={North American Association for Computational Linguistics (NAACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2021}<br>}",
                "Petroni et al. (2019) demonstrated that it is possible to retrieve world facts from a pre-trained language model by expressing them as cloze-style prompts and interpret the model's prediction accuracy as a lower bound on the amount of factual information it encodes. Subsequent work has attempted to tighten the estimate by searching for better prompts, using a disjoint set of facts as training data. In this work, we make two complementary contributions to better understand these factual probing techniques. First, we propose OptiPrompt, a novel and efficient method which directly optimizes in continuous embedding space. We find this simple method is able to predict an additional 6.4% of facts in the LAMA benchmark. Second, we raise a more important question: Can we really interpret these probing results as a lower bound? Is it possible that these prompt-search methods learn from the training data too? We find, somewhat surprisingly, that the training data used by these methods contains certain regularities of the underlying fact distribution, and all the existing prompt methods, including ours, are able to exploit them for better fact prediction. We conduct a set of control experiments to disentangle \"learning\" from \"learning to recall\", providing a more detailed picture of what different prompts can reveal about pre-trained language models.",
                "https://arxiv.org/abs/2104.05240",
                "https://github.com/princeton-nlp/OptiPrompt",
                null,
                "https://github.com/princeton-nlp/OptiPrompt/blob/main/slides/slides.pdf"
            )

            add_paper("Non-Parametric Few-Shot Learning for Word Sense Disambiguation",
                "Howard Chen, Mengzhou Xia, <b>Danqi Chen</b>",
                "In NAACL 2021 (short)",
                "https://arxiv.org/pdf/2104.12677.pdf",
                "@inproceedings{chen2021nonparametric,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Non-Parametric Few-Shot Learning for Word Sense Disambiguation},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Chen, Howard and Xia, Mengzhou and Chen, Danqi},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={North American Association for Computational Linguistics (NAACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2021}<br>}",
                "Word sense disambiguation (WSD) is a long-standing problem in natural language processing. One significant challenge in supervised all-words WSD is to classify among senses for a majority of words that lie in the long-tail distribution. For instance, 84% of the annotated words have less than 10 examples in the SemCor training data. This issue is more pronounced as the imbalance occurs in both word and sense distributions. In this work, we propose MetricWSD, a non-parametric few-shot learning approach to mitigate this data imbalance issue. By learning to compute distances among the senses of a given word through episodic training, MetricWSD transfers knowledge (a learned metric space) from high-frequency words to infrequent ones. MetricWSD constructs the training episodes tailored to word frequencies and explicitly addresses the problem of the skewed distribution, as opposed to mixing all the words trained with parametric models in previous work. Without resorting to any lexical resources, MetricWSD obtains strong performance against parametric alternatives, achieving a 75.1 F1 score on the unified WSD evaluation benchmark (Raganato et al., 2017b). Our analysis further validates that infrequent words and senses enjoy significant improvement.",
                "https://arxiv.org/abs/2104.12677",
                "https://github.com/princeton-nlp/metric-wsd"
            )

            add_paper("NeurIPS 2020 EfficientQA Competition: Systems, Analyses and Lessons Learned",
                "Sewon Min, Jordan Boyd-Graber, Chris Alberti, <b>Danqi Chen</b>, Eunsol Choi, Michael Collins, Kelvin Guu, Hannaneh Hajishirzi, Kenton Lee, Jennimaria Palomaki, Colin Raffel, Adam Roberts, Tom Kwiatkowski and EfficientQA participants",
                "Proceedings of Machine Learning Research",
                "https://arxiv.org/pdf/2101.00133.pdf",
                "@article{min2021neurips,<br>" +
                "&nbsp;&nbsp;&nbsp;title={NeurIPS 2020 EfficientQA Competition: Systems, Analyses and Lessons Learned},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Sewon Min and Jordan Boyd-Graber and Chris Alberti and Danqi Chen and Eunsol Choi and Michael Collins and Kelvin Guu and Hannaneh Hajishirzi and Kenton Lee and Jennimaria Palomaki and Colin Raffel and Adam Roberts and Tom Kwiatkowski and Patrick Lewis and Yuxiang Wu and Heinrich Küttler and Linqing Liu and Pasquale Minervini and Pontus Stenetorp and Sebastian Riedel and Sohee Yang and Minjoon Seo and Gautier Izacard and Fabio Petroni and Lucas Hosseini and Nicola De Cao and Edouard Grave and Ikuya Yamada and Sonse Shimaoka and Masatoshi Suzuki and Shumpei Miyawaki and Shun Sato and Ryo Takahashi and Jun Suzuki and Martin Fajcik and Martin Docekal and Karel Ondrej and Pavel Smrz and Hao Cheng and Yelong Shen and Xiaodong Liu and Pengcheng He and Weizhu Chen and Jianfeng Gao and Barlas Oguz and Xilun Chen and Vladimir Karpukhin and Stan Peshterliev and Dmytro Okhonko and Michael Schlichtkrull and Sonal Gupta and Yashar Mehdad and Wen-tau Yih},<br>" +
                "&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:2101.00133},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2021}<br>}",
                "We review the EfficientQA competition from NeurIPS 2020. The competition focused on open-domain question answering (QA), where systems take natural language questions as input and return natural language answers. The aim of the competition was to build systems that can predict correct answers while also satisfying strict on-disk memory budgets. These memory budgets were designed to encourage contestants to explore the trade-off between storing large, redundant, retrieval corpora or the parameters of large learned models. In this report, we describe the motivation and organization of the competition, review the best submissions, and analyze system predictions to inform a discussion of evaluation for open-domain QA.",
                "https://arxiv.org/abs/2101.00133",
                "https://github.com/efficientqa/retrieval-based-baselines",
                "https://github.com/google-research-datasets/natural-questions/tree/master/nq_open",
                null,
                "https://www.youtube.com/watch?v=3tdWV4vAf2I&ab_channel=SewonMin",
                "<a href=\"http://efficientqa.github.io/\" style=\"color: #8C1515\">http://efficientqa.github.io/</a>"
            )
            document.write("</ul>")

            document.write("<h2>2020</h2>")
            document.write("<ul>")


            add_paper("Dense Passage Retrieval for Open-Domain Question Answering",
                "Vladimir Karpukhin*, Barlas Oğuz*, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, <b>Danqi Chen</b>, Wen-tau Yih",
                "In EMNLP 2020",
                "papers/emnlp2020a.pdf",
                "@inproceedings{karpukhin2020dense,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Dense Passage Retrieval for Open-Domain Question Answering},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Karpukhin, Vladimir and Oğuz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Empirical Methods in Natural Language Processing (EMNLP)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2020}<br>}",
                "Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks.",
                "https://arxiv.org/abs/2004.04906",
                "https://github.com/facebookresearch/DPR",
                null,
                null,
                "https://slideslive.com/38939151/dense-passage-retrieval-for-opendomain-question-answering",
                "You can try out the <a href=\"http://qa.cs.washington.edu:2020/\" style=\"color: #8C1515\">demo</a> of DPR!"
            )

            add_paper("TextHide: Tackling Data Privacy in Language Understanding Tasks",
                "Yangsibo Huang, Zhao Song, <b>Danqi Chen</b>, Kai Li, Sanjeev Arora",
                "In EMNLP 2020 (Findings)",
                "papers/emnlp2020b.pdf",
                "@inproceedings{huang2020texthide,<br>" +
                "&nbsp;&nbsp;&nbsp;title={{TextHide}: Tackling Data Privacy in Language Understanding Tasks},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Huang, Yangsibo and Song, Zhao and Chen, Danqi and Li, Kai and Arora, Sanjeev},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Findings of Empirical Methods in Natural Language Processing (EMNLP)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2020}<br>}",
                "An unsolved challenge in distributed or federated learning is to effectively mitigate privacy risks without slowing down training or reducing accuracy.  In this paper, we propose TextHide aiming at addressing this challenge for natural language understanding tasks.  It requires all participants to add a simple encryption step to prevent an eavesdropping attacker from recovering private text data.  Such an encryption step is efficient and only affects the task performance slightly.  In addition, TextHide fits well with the popular framework of fine-tuning pre-trained language models (e.g., BERT) for any sentence or sentence-pair task. We evaluate TextHide on the GLUE benchmark, and our experiments show that TextHide can effectively defend attacks on shared gradients or representations and the averaged accuracy reduction is only 1.9%. We also present an analysis of the security of TextHide using a conjecture about the computational intractability of a mathematical problem.",
                "https://arxiv.org/abs/2010.06053",
                "https://github.com/Hazelsuko07/TextHide",
                null,
                null,
                "https://slideslive.com/38939771/texthide-tackling-data-privacy-in-language-understanding-tasks"
            )

            add_paper("SpanBERT: Improving Pre-training by Representing and Predicting Spans",
                "Mandar Joshi*, <b>Danqi Chen</b>*, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy",
                "In TACL 2020 (presented at ACL 2020)",
                "papers/tacl2020.pdf",
                "@article{joshi2020spanbert,<br>" +
                "&nbsp;&nbsp;&nbsp;title={{SpanBERT}: Improving Pre-training by Representing and Predicting Spans},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Joshi, Mandar and Chen, Danqi and Liu, Yinhan and Weld, Daniel S and Zettlemoyer, Luke and Levy, Omer},<br>" +
                "&nbsp;&nbsp;&nbsp;journal={Transactions of the Association of Computational Linguistics (TACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2020}<br>}",
                "We present SpanBERT, a pre-training method that is designed to better represent and predict spans of text. Our approach extends BERT by (1) masking contiguous random spans, rather than random tokens, and (2) training the span boundary representations to predict the entire content of the masked span, without relying on the individual token representations within it. SpanBERT consistently outperforms BERT and our better-tuned baselines, with substantial gains on span selection tasks such as question answering and coreference resolution. In particular, with the same training data and model size as BERT-large, our single model obtains 94.6% and 88.7% F1 on SQuAD 1.1 and 2.0, respectively. We also achieve a new state of the art on the OntoNotes coreference resolution task (79.6\% F1), strong performance on the TACRED relation extraction benchmark, and even show gains on GLUE.",
                "https://arxiv.org/abs/1907.10529",
                "https://github.com/facebookresearch/SpanBERT",
                null,
                null,
                "https://slideslive.com/38929502/spanbert-improving-pretraining-by-representing-and-predicting-spans"
            )

            add_paper("Open-Domain Question Answering",
                "<b>Danqi Chen</b>, Wen-tau Yih",
                "ACL 2020 (Tutorial)",
                "papers/acl2020tutorial.pdf",
                "@inproceedings{chen2020open,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Open-Domain Question Answering},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Chen, Danqi and Yih, Wen-tau},<br>" +
                "&nbsp;&nbsp;&nbsp;journal={Association for Computational Linguistics (ACL): Tutorial Abstracts},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2020},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={34--37}<br>}",
                "This tutorial provides a comprehensive and coherent overview of cutting-edge research in open-domain question answering (QA), the task of answering questions using a large collection of documents of diversified topics. We will start by first giving a brief historical background, discussing the basic setup and core technical challenges of the research problem, and then describe modern datasets with the common evaluation metrics and benchmarks. The focus will then shift to cutting-edge models proposed for open-domain QA, including two-stage retriever-reader approaches, dense retriever and end-to-end training, and retriever-free methods. Finally, we will cover some hybrid approaches using both text and large knowledge bases and conclude the tutorial with important open questions. We hope that the tutorial will not only help the audience to acquire up-to-date knowledge but also provide new perspectives to stimulate the advances of open-domain QA research in the next phase.",
                null,
                null,
                null,
                "https://github.com/danqi/acl2020-openqa-tutorial/tree/master/slides",
                "https://slideslive.com/38931668/t8-opendomain-question-answering"
            )
            document.write("</ul>")

            document.write("<h2>2019</h2>")
            document.write("<ul>")
            add_paper("RoBERTa: A Robustly Optimized BERT Pretraining Approach",
                "Yinhan Liu*, Myle Ott*, Naman Goyal*, Jingfei Du*, Mandar Joshi, <b>Danqi Chen</b>, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov",
                "arXiv 1907.11692",
                "papers/roberta_paper.pdf",
                "@article{liu2019roberta,<br>" +
                "&nbsp;&nbsp;&nbsp;title={{RoBERTa}: {A} Robustly Optimized {BERT} Pretraining Approach},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin},<br>" +
                "&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:1907.11692},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2019}<br>}",
                "Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices, and raise questions about the source of recently reported improvements. We release our models and code.",
                "https://arxiv.org/abs/1907.11692",
                "https://github.com/pytorch/fairseq/tree/master/examples/roberta"
            )

            add_paper("Knowledge Guided Text Retrieval and Reading for Open Domain Question Answering",
                "Sewon Min, <b>Danqi Chen</b>, Luke Zettlemoyer, Hannaneh Hajishirzi",
                "arXiv 1911.03868",
                "papers/graphqa_paper.pdf",
                "@article{min2019knowledge,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Knowledge Guided Text Retrieval and Reading for Open Domain Question Answering},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Min, Sewon and Chen, Danqi and Zettlemoyer, Luke and Hajishirzi, Hannaneh},<br>" +
                "&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:1911.03868},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2019}<br>}",
                "We introduce an approach for open-domain question answering (QA) that retrieves and reads a passage graph, where vertices are passages of text and edges represent relationships that are derived from an external knowledge base or co-occurrence in the same article. Our goals are to boost coverage by using knowledge-guided retrieval to find more relevant passages than text-matching methods, and to improve accuracy by allowing for better knowledge-guided fusion of information across related passages. Our graph retrieval method expands a set of seed keyword-retrieved passages by traversing the graph structure of the knowledge base. Our reader extends a BERT-based architecture and updates passage representations by propagating information from related passages and their relations, instead of reading each passage in isolation. Experiments on three open-domain QA datasets, WebQuestions, Natural Questions and TriviaQA, show improved performance over non-graph baselines by 2-11% absolute. Our approach also matches or exceeds the state-of-the-art in every case, without using an expensive end-to-end training regime.",
                "https://arxiv.org/abs/1911.03868"
            )

            add_paper("A Discrete Hard EM Approach for Weakly Supervised Question Answering",
                "Sewon Min, <b>Danqi Chen</b>, Hannaneh Hajishirzi, Luke Zettlemoyer",
                "In EMNLP 2019",
                "papers/emnlp2019.pdf",
                "@article{min2019discrete,<br>" +
                "&nbsp;&nbsp;&nbsp;title={A Discrete Hard {EM} Approach for Weakly Supervised Question Answering},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Min, Sewon and Chen, Danqi and Hajishirzi, Hannaneh and Zettlemoyer, Luke},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Empirical Methods in Natural Language Processing (EMNLP)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2019},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={2851--2864}<br>}",
                "Many question answering (QA) tasks only provide weak supervision for how the answer should be computed. For example, TriviaQA answers are entities that can be mentioned multiple times in supporting documents, while DROP answers can be computed by deriving many different equations from numbers in the reference text. In this paper, we show it is possible to convert such tasks into discrete latent variable learning problems with a precomputed, task-specific set of possible \"solutions\" (e.g. different mentions or equations) that contains one correct option. We then develop a hard EM learning scheme that computes gradients relative to the most likely solution at each update. Despite its simplicity, we show that this approach significantly outperforms previous methods on six QA tasks, including absolute gains of 2--10%, and achieves the state-of-the-art on five of them. Using hard updates instead of maximizing marginal likelihood is key to these results as it encourages the model to find the one correct answer, which we show through detailed qualitative analysis.",
                "https://arxiv.org/abs/1909.04849",
                "https://github.com/shmsw25/qa-hard-em",
                null,
                "presentations/emnlp2019_slides.pdf",
                "https://vimeo.com/426355627"
            )

            add_paper("CoQA: A Conversational Question Answering Challenge",
                "Siva Reddy*, <b>Danqi Chen</b>*, Christopher D. Manning",
                "In TACL 2019 (presented at NAACL 2019)",
                "papers/tacl2019.pdf",
                "@article{reddy2019coqa,<br>" +
                "&nbsp;&nbsp;&nbsp;title={{CoQA}: A Conversational Question Answering Challenge},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Reddy, Siva and Chen, Danqi and Manning, Christopher D},<br>" +
                "&nbsp;&nbsp;&nbsp;journal={Transactions of the Association of Computational Linguistics (TACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2019}<br>}",
                "Humans gather information by engaging in conversations involving a series of interconnected questions and answers. For machines to assist in information gathering, it is therefore essential to enable them to answer conversational questions. We introduce CoQA, a novel dataset for building Conversational Question Answering systems. Our dataset contains 127k questions with answers, obtained from 8k conversations about text passages from seven diverse domains. The questions are conversational, and the answers are free-form text with their corresponding evidence highlighted in the passage. We analyze CoQA in depth and show that conversational questions have challenging phenomena not present in existing reading comprehension datasets, e.g., coreference and pragmatic reasoning. We evaluate strong conversational and reading comprehension models on CoQA. The best system obtains an F1 score of 65.4%, which is 23.4 points behind human performance (88.8%), indicating there is ample room for improvement. We launch CoQA as a challenge to the community at https://stanfordnlp.github.io/coqa/.",
                "https://arxiv.org/abs/1808.07042",
                "https://github.com/stanfordnlp/coqa-baselines",
                "https://stanfordnlp.github.io/coqa/",
                null,
                "https://vimeo.com/356110589",
                "The dataset and leaderboard are at <a href=\"https://stanfordnlp.github.io/coqa/\" style=\"color: #8C1515\">https://stanfordnlp.github.io/coqa/</a>."
            )

            add_paper("MRQA 2019 Shared Task: Evaluating Generalization in Reading Comprehension",
                "Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eunsol Choi, <b>Danqi Chen</b>",
                "In MRQA 2019",
                "papers/mrqa2019.pdf",
                "@inproceedings{fisch2019mrqa,<br>" +
                "&nbsp;&nbsp;&nbsp;title={{MRQA} 2019 Shared Task: Evaluating Generalization in Reading Comprehension},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Fisch, Adam and Talmor, Alon and Jia, Robin and Seo, Minjoon and Choi, Eunsol and Chen, Danqi},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Proceedings of 2nd Machine Reading for Reading Comprehension (MRQA) Workshop at EMNLP},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2019},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={1--13}<br>}",
                "We present the results of the Machine Reading for Question Answering (MRQA) 2019 shared task on evaluating the generalization capabilities of reading comprehension systems. In this task, we adapted and unified 18 distinct question answering datasets into the same format. Among them, six datasets were made available for training, six datasets were made available for development, and the final six were hidden for final evaluation. Ten teams submitted systems, which explored various ideas including data sampling, multi-task learning, adversarial training and ensembling. The best system achieved an average F1 score of 72.5 on the 12 held-out datasets, 10.7 absolute points higher than our initial baseline based on BERT.",
                "https://arxiv.org/abs/1910.09753",
                "https://github.com/mrqa/MRQA-Shared-Task-2019/tree/master/baseline",
                "https://github.com/mrqa/MRQA-Shared-Task-2019#datasets"
            )
            document.write("</ul>")

            document.write("<h2>2018 and before</h2>")
            document.write("<ul>")
            add_paper("Neural Reading Comprehension and Beyond",
                "<b>Danqi Chen</b>",
                "PhD thesis, Stanford University, 2018<br> (<b><font color=\"red\">Arthur Samuel Best Doctoral Thesis Award</font></b>)",
                "papers/thesis.pdf",
                "@phdthesis{chen2018neural,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Neural Reading Comprehension and Beyond},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Chen, Danqi},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2018},<br>" +
                "   school={Stanford University}<br>}",
                "Teaching machines to understand human language documents is one of the most elusive and long-standing challenges in Artificial Intelligence. This thesis tackles the problem of reading comprehension: how to build computer systems to read a passage of text and answer  comprehension questions. On the one hand, we think that reading comprehension is an important task for evaluating how well computer systems understand human language. On the other hand, if we can build high-performing reading comprehension systems, they would be a crucial technology for applications such as question answering and dialogue systems. <br><br>In this thesis, we focus on neural reading comprehension: a class of reading comprehension models built on top of deep neural networks. Compared to traditional sparse, hand-designed feature-based models, these end-to-end neural models have proven to be more effective in learning rich linguistic phenomena and improved performance on all the modern reading comprehension benchmarks by a large margin. <br><br>This thesis consists of two parts. In the first part, we aim to cover the essence of neural reading comprehension and present our efforts at building effective neural reading comprehension models, and more importantly, understanding what neural reading comprehension models have actually learned, and what depth of language understanding is needed to solve current tasks. We also summarize recent advances and discuss future directions and open questions in this field. <br><br>In the second part of this thesis, we investigate how we can build practical applications based on the recent success of neural reading comprehension. In particular, we pioneered two new research directions: 1) how we can combine information retrieval techniques with neural reading comprehension to tackle large-scale open-domain question answering; and 2) how we can build conversational question answering systems from current single-turn, span-based reading comprehension models. We implemented these ideas in the DrQA and CoQA projects and we demonstrate the effectiveness of these approaches. We believe that they hold great promise for future language technologies.",
                null,
                "https://github.com/danqi/thesis"
            )

            add_paper("Position-aware Attention and Supervised Data Improve Slot Filling",
                "Yuhao Zhang, Victor Zhong, <b>Danqi Chen</b>, Gabor Angeli, and Christopher D. Manning",
                "In EMNLP 2017 <br>(<b><font color=\"red\">Outstanding Paper Award</font></b>)",
                "papers/emnlp2017.pdf",
                "@inproceedings{zhang2017tacred,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Position-aware Attention and Supervised Data Improve Slot Filling},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Zhang, Yuhao and Zhong, Victor and Chen, Danqi and Angeli, Gabor and Manning, Christopher D.},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Empirical Methods in Natural Language Processing (EMNLP)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2017},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={35--45}<br>}",
                "Organized relational knowledge in the form of “knowledge graphs” is important for many applications. However, the ability to populate knowledge bases with facts automatically extracted from documents has improved frustratingly slowly. This paper simultaneously addresses two issues that have held back prior work. We first propose an effective new model, which combines an LSTM sequence model with a form of entity position-aware attention that is better suited to relation extraction. Then we build TACRED, a large (119,474 examples) supervised relation extraction dataset obtained via crowdsourcing and targeted towards TAC KBP relations. The combination of better supervised data and a more appropriate high-capacity model enables much better relation extraction performance. When the model trained on this new dataset replaces the previous relation extraction component of the best TAC KBP 2015 slot filling system, its F1 score increases markedly from 22.2% to 26.7%.",
                null,
                "https://github.com/yuhaozhang/tacred-relation",
                "https://catalog.ldc.upenn.edu/LDC2018T24",
                "presentations/emnlp2017_slides.pdf",
                "https://vimeo.com/238230211",
                "The TACRED project page is at <a href=\"https://nlp.stanford.edu/projects/tacred/\" style=\"color: #8C1515\">https://nlp.stanford.edu/projects/tacred/</a>."
            )

            add_paper("Reading Wikipedia to Answer Open-Domain Questions",
                "<b>Danqi Chen</b>, Adam Fisch, Jason Weston, Antoine Bordes",
                "In ACL 2017",
                "papers/acl2017.pdf",
                "@inproceedings{chen2017reading,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Reading {Wikipedia} to Answer Open-Domain Questions},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Chen, Danqi and Fisch, Adam and Weston, Jason and Bordes, Antoine},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Association for Computational Linguistics (ACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2017},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={1870--1879}<br>}",
                "This paper proposes to tackle open-domain question answering using Wikipedia as the unique knowledge source: the answer to any factoid question is a text span in a Wikipedia article. This task of machine reading at scale combines the challenges of document retrieval (finding the relevant articles) with that of machine comprehension of text (identifying the answer spans from those articles). Our approach combines a search component based on bigram hashing and TF-IDF matching with a multi-layer recurrent neural network model trained to detect answers in Wikipedia paragraphs. Our experiments on multiple existing QA datasets indicate that (1) both modules are highly competitive with respect to existing counterparts and (2) multitask learning using distant supervision on their combination is an effective complete system on this challenging task.",
                "https://arxiv.org/abs/1704.00051",
                "https://github.com/facebookresearch/DrQA",
                "https://github.com/danqi/drqa-datasets",
                "presentations/acl2017_poster.pdf",
                null,
                "The DrQA project page is at <a href=\"https://github.com/facebookresearch/DrQA\" style=\"color: #8C1515\">https://github.com/facebookresearch/DrQA</a>."
            )

            add_paper("A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task",
                "<b>Danqi Chen</b>, Jason Bolton, Christopher D. Manning",
                "In ACL 2016 <br>(<b><font color=\"red\">Outstanding Paper Award</font></b>)",
                "papers/acl2016.pdf",
                "@inproceedings{chen2016thorough,<br>" +
                "&nbsp;&nbsp;&nbsp;title={A Thorough Examination of the {CNN/Daily Mail} Reading Comprehension Task},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Chen, Danqi and Bolton, Jason and Manning, Christopher D.},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Association for Computational Linguistics (ACL)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2016},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={2358--2367}<br>}",
                "Enabling a computer to understand a document so that it can answer comprehension questions is a central, yet unsolved goal of NLP. A key factor impeding its solution by machine learned systems is the limited availability of human-annotated data. Hermann et al. (2015) seek to solve this problem by creating over a million training examples by pairing CNN and Daily Mail news articles with their summarized bullet points, and show that a neural network can then be trained to give good performance on this task. In this paper, we conduct a thorough examination of this new reading comprehension task. Our primary aim is to understand what depth of language understanding is required to do well on this task. We approach this from one side by doing a careful hand-analysis of a small subset of the problems and from the other by showing that simple, carefully designed systems can obtain accuracies of 73.6% and 76.6% on these two datasets, exceeding current state-of-the-art results by 7-10% and approaching what we believe is the ceiling for performance on this task.",
                "https://arxiv.org/abs/1606.02858",
                "https://github.com/danqi/rc-cnn-dailymail",
                null,
                "presentations/acl2016_slides.pdf",
                "http://techtalks.tv/talks/a-thorough-examination-of-the-cnndaily-mail-reading-comprehension-task/63222/"
            )

            add_paper("Stanford at TAC KBP 2016: Sealing Pipeline Leaks and Understanding Chinese",
                "Yuhao Zhang*, Arun Chaganty*, Ashwin Paranjape*, <b>Danqi Chen</b>*, Jason Bolton*, Peng Qi, Christopher D. Manning",
                "In TAC 2016",
                "papers/tac2016.pdf",
                "@inproceedings{zhang2016stanford,<br>" +
                "&nbsp;&nbsp;&nbsp;title={{Stanford} at {TAC} {KBP} 2016: Sealing Pipeline Leaks and Understanding Chinese},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Zhang, Yuhao and Chaganty, Arun and Paranjape, Ashwin and Chen, Danqi and Bolton, Jason and Qi, Peng and Manning, Christopher D},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Text Analysis Conference (TAC)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2016},<br>}",
                "We describe Stanford’s entries in the TAC KBP 2016 Cold Start Slot Filling and Knowledge Base Population challenge. Our biggest contribution is an entirely new Chinese entity detection and relation extraction system for the new Chinese and cross-lingual relation extraction tracks. This new system consists of several ruled-based relation extractors and a distantly supervised extractor. We also analyze errors produced by our existing mature English KBP system, which leads to several fixes, notably improvements to our patternsbased extractor and neural network model, support for nested mentions and inferred relations. Stanford’s 2016 English, Chinese and cross-lingual submissions achieved an overall (macro-averaged LDC-MEAN) F1 of 22.0, 14.2, and 11.2 respectively on the 2016 evaluation data, performing well above the median entries, at 7.5, 13.2 and 8.3 respectively."
            )

            add_paper("Representing Text for Joint Embedding of Text and Knowledge Bases",
                "Kristina Toutanova, <b>Danqi Chen</b>, Patrick Pantel, Hoifung Poon, Pallavi Choudhury, Michael Gamon",
                "In EMNLP 2015",
                "papers/emnlp2015.pdf",
                "@inproceedings{toutanova2015representing,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Representing Text for Joint Embedding of Text and Knowledge Bases},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Toutanova, Kristina and Chen, Danqi and Pantel, Patrick and Poon, Hoifung and Choudhury, Pallavi and Gamon, Michael},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Empirical Methods in Natural Language Processing (EMNLP)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2015},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={1499--1509}<br>}",
                "Models that learn to represent textual and knowledge base relations in the same continuous latent space are able to perform joint inferences among the two kinds of relations and obtain high accuracy on knowledge base completion (Riedel et al., 2013). In this paper we propose a model that captures the compositional structure of textual relations, and jointly optimizes entity, knowledge base, and textual relation representations. The proposed model significantly improves performance over a model that does not share parameters among textual relations with common sub-structure.",
                null,
                null,
                "data/fb15k-237.zip",
                null,
                "https://vimeo.com/163292987"
            )

            add_paper("Bootstrapped Self Training for Knowledge Base Population",
                "Gabor Angeli, Victor Zhong, <b>Danqi Chen</b>, Arun Chaganty, Jason Bolton, <br> Melvin Johnson Premkumar, Panupong Pasupat, Sonal Gupta, Christopher D. Manning",
                "In TAC 2015",
                "papers/tac2015.pdf",
                "@inproceedings{angeli2015bootstrapped,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Bootstrapped self training for knowledge base population},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Angeli, Gabor and Zhong, Victor and Chen, Danqi and Chaganty, Arun and Bolton, Jason and Premkumar, Melvin Johnson and Pasupat, Panupong and Gupta, Sonal and Manning, Christopher D},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Text Analysis Conference (TAC)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2015},<br>}",
                "A central challenge in relation extraction is the lack of supervised training data. Pattern-based relation extractors suffer from low recall, whereas distant supervision yields noisy data which hurts precision. We propose bootstrapped selftraining to capture the benefits of both systems: the precision of patterns and the generalizability of trained models. We show that training on the output of patterns drastically improves performance over the patterns. We propose self-training for further improvement: recall can be improved by incorporating the predictions from previous iterations; precision by filtering the assumed negatives based previous predictions. We show that even our patternbased model achieves good performance on the task, and the self-trained models rank among the top systems.",
            )

            add_paper("Observed Versus Latent Features for Knowledge Base and Text Inference",
                "Observed Versus Latent Features for Knowledge Base and Text Inference",
                "In Workshop on Continuous Vector Space Models and Their Compositionality (CVSC) 2015",
                "papers/cvsc2015.pdf",
                "@inproceedings{toutanova2015observed,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Observed Versus Latent Features for Knowledge Base and Text Inference},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Kristina Toutanova and Danqi Chen},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Workshop on Continuous Vector Space Models and Their Compositionality (CVSC)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2015},<br>}",
                "In this paper we show the surprising effectiveness of a simple observed features model in comparison to latent feature models on two benchmark knowledge base completion datasets, FB15K and WN18. We also compare latent and observed feature models on a more challenging dataset derived from FB15K, and additionally coupled with textual mentions from a web-scale corpus. We show that the observed features model is most effective at capturing the information present for entity pairs with textual relations, and a combination of the two combines the strengths of both model types.",
                null,
                null,
                "data/fb15k-237.zip",
                null
            )

            add_paper("A Fast and Accurate Dependency Parser using Neural Networks",
                "<b>Danqi Chen</b>, Christopher D. Manning",
                "In EMNLP 2014",
                "papers/emnlp2014.pdf",
                "@inproceedings{chen2014fast,<br>" +
                "&nbsp;&nbsp;&nbsp;title={A Fast and Accurate Dependency Parser using Neural Networks},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Chen, Danqi and Manning, Christopher D},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Empirical Methods in Natural Language Processing (EMNLP)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2014},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={740--750}<br>}",
                "Almost all current dependency parsers classify based on millions of sparse indicator features. Not only do these features generalize poorly, but the cost of feature computation restricts parsing speed significantly. In this work, we propose a novel way of learning a neural network classifier for use in a greedy, transition-based dependency parser. Because this classifier learns and uses just a small number of dense features, it can work very fast, while achieving an about 2% improvement in unlabeled and labeled attachment scores on both English and Chinese datasets. Concretely, our parser is able to parse more than 1000 sentences per second at 92.2% unlabeled attachment score on the English Penn Treebank.",
                null,
                "https://stanfordnlp.github.io/CoreNLP/depparse.html",
                null,
                "presentations/emnlp2014_slides.pdf",
                "https://www.youtube.com/watch?v=MLAcBv5dLEs",
                "The neural dependency parser is included in the <a href=\"https://nlp.stanford.edu/software/corenlp.shtml\" style=\"color: #8C1515\">Stanford CoreNLP</a> software (since v3.5.0)"
            )

            add_paper("Reasoning With Neural Tensor Networks for Knowledge Base Completion",
                "Richard Socher*, <b>Danqi Chen</b>*, Christopher D. Manning, Andrew Ng",
                "In NIPS 2013",
                "papers/nips2013.pdf",
                "@inproceedings{socher2013reasoning,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Reasoning With Neural Tensor Networks for Knowledge Base Completion},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Socher, Richard and Chen, Danqi and Manning, Christopher D and Ng, Andrew},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={Advances in Neural Information Processing Systems (NIPS)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2013},<br>" +
                "&nbsp;&nbsp;&nbsp;pages={926--934}<br>}",
                "Knowledge bases are an important resource for question answering and other tasks but often suffer from incompleteness and lack of ability to reason over their discrete entities and relationships. In this paper we introduce an expressive neural tensor network suitable for reasoning over relationships between two entities. Previous work represented entities as either discrete atomic units or with a single entity vector representation. We show that performance can be improved when entities are represented as an average of their constituting word vectors. This allows sharing of statistical strength between, for instance, facts involving the “Sumatran tiger” and “Bengal tiger.” Lastly, we demonstrate that all models improve when these word vectors are initialized with vectors learned from unsupervised large corpora. We assess the model by considering the problem of predicting additional true relations between entities given a subset of the knowledge base. Our model outperforms previous models and can classify unseen relationships in WordNet and FreeBase with an accuracy of 86.2% and 90.0%, respectively.",
                null,
                null,
                "data/nips13-dataset.tar.bz2",
                "presentations/nips2013_poster.pdf"
            )

            add_paper("Learning New Facts From Knowledge Bases With Neural Tensor Networks and Semantic Word Vectors",
                "<b>Danqi Chen</b>, Richard Socher, Christopher D. Manning, Andrew Ng",
                "In ICLR 2013 (workshop track)",
                "papers/iclr2013.pdf",
                "@inproceedings{chen2013learning,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Learning new facts from knowledge bases with neural tensor networks and semantic word vectors},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Socher, Richard and Chen, Danqi and Manning, Christopher D and Ng, Andrew},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={International Conference on Learning Representations (ICLR)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2013},<br>}",
                "Knowledge bases provide applications with the benefit of easily accessible, systematic relational knowledge but often suffer in practice from their incompleteness and lack of knowledge of new entities and relations. Much work has focused on building or extending them by finding patterns in large unannotated text corpora. In contrast, here we mainly aim to complete a knowledge base by predicting additional true relationships between entities, based on generalizations that can be discerned in the given knowledgebase. We introduce a neural tensor network (NTN) model which predicts new relationship entries that can be added to the database. This model can be improved by initializing entity representations with word vectors learned in an unsupervised fashion from text, and when doing this, existing relations can even be queried for entities that were not present in the database. Our model generalizes and outperforms existing models for this problem, and can classify unseen relationships in WordNet with an accuracy of 75.8%.",
                "https://arxiv.org/abs/1301.3618"
            )

            add_paper("Beyond Ten Blue Links: Enabling User Click Modeling in Federated Web Search",
                "Danqi Chen</b>, Weizhu Chen, Haixun Wang, Zheng Chen, Qiang Yang",
                "In WSDM 2012",
                "papers/wsdm2012.pdf",
                "@inproceedings{chen2012beyond,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Beyond ten blue links: enabling user click modeling in federated web search},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Chen, Danqi and Chen, Weizhu and Wang, Haixun and Chen, Zheng and Yang, Qiang},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={International Conference on Web Search and Data Mining (WSDM)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2012},<br>}",
                "Click model has been positioned as an effective approach to interpret user click behavior in search engines. Existing advances in click models mostly focus on traditional Web search which contains only ten homogeneous Web HTML documents. However, in modern commercial search engines, more and more Web search results are federated from multiple sources and contain non-HTML results returned by other heterogeneous vertical engines, such as video or image search engines. In this paper, we study user click behavior in federated search results. In order to investigate this problem, we put forward an observation that user click behavior in federated search is highly different from that in traditional Web search, making it difficult to interpret using existing click models. Thus, we propose a novel federated click model (FCM) to interpret user click behavior in federated search. In particular, we introduce two new biases in FCM. The first indicates that users tend to be attracted by vertical results and their visual attention on them may increase the examination probability of other nearby web results. The other illustrates that user click behavior on vertical results may lead to more indication of relevance due to their presentation style in federated search. With these biases and an effective model to correct them, FCM is more accurate in characterizing user click behavior in federated search. Our extensive experimental results show that FCM can outperform other click models in interpreting user click behavior in federated search and achieve significant improvements in terms of both perplexity and log-likelihood.",
                null,
                null,
                null,
                "presentations/wsdm2012_slides.pptx"
            )

            add_paper("Characterizing Inverse Time Dependency in Multi-class Learning",
                "<b>Danqi Chen</b>, Weizhu Chen, Qiang Yang",
                "In ICDM 2011",
                "papers/icdm2011.pdf",
                "@inproceedings{chen2011characterizing,<br>" +
                "&nbsp;&nbsp;&nbsp;title={Characterizing Inverse Time Dependency in Multi-class Learning},<br>" +
                "&nbsp;&nbsp;&nbsp;author={Chen, Danqi and Chen, Weizhu and Yang, Qiang},<br>" +
                "&nbsp;&nbsp;&nbsp;booktitle={International Conference on Data Mining (ICDM)},<br>" +
                "&nbsp;&nbsp;&nbsp;year={2011}<br>}",
                "The training time of most learning algorithms increases as the size of training data increases. Yet, recent advances in linear binary SVM and LR challenge this commonsense by proposing an inverse dependency property, where the training time decreases as the size of training data increases. In this paper, we study the inverse dependency property of multi-class classification problem. We describe a general framework for multi-class classification problem with a single objective to achieve inverse dependency and extend it to three popular multi-class algorithms. We present theoretical results demonstrating its convergence and inverse dependency guarantee. We conduct experiments to empirically verify the inverse dependency of all the three algorithms on large-scale datasets as well as to ensure the accuracy."
            )
            document.write("</ul>")
</script>

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
