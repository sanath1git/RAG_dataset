Option B — Golden Dataset for RAG
A RAG system is only as good as the questions you test it against. Your job is to build a small but thoughtful
evaluation set from the four videos below, all on neural networks and deep learning.
Videos:
1. 3Blue1Brown — But what is a Neural Network? — youtube.com/watch?v=aircAruvnKk
2. 3Blue1Brown — Transformers, the tech behind LLMs — youtube.com/watch?v=wjZofJX0v4M
3. CampusX — What is Deep Learning? (Hindi) — youtube.com/watch?v=fHF22Wxuyw4
4. CodeWithHarry — All About ML & Deep Learning (Hindi) — youtube.com/watch?v=C6YtPJxNULA
You can pull transcripts using youtubetranscript.com or the youtube-transcript-api Python library.
Submit 5 question-answer pairs. Each one needs:

question
answer
source — which video, with timestamp or section
Along with the QA pairs, include a short methodology note (bullet points is fine) covering:
How did you decide which questions made the cut?
How did you actually pull them from the material?
What are these questions testing — what would a wrong retrieval look like?