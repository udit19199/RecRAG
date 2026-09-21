# Dataset choices and source records

RecRAG uses four question-answer datasets. A **record** is one dataset item. It
contains one question, its answer, and text from one or more Wikipedia pages.

The datasets store that text in different shapes:

| Dataset | Text stored in each record |
| --- | --- |
| HotpotQA | Sentences grouped under page titles |
| 2WikiMultiHopQA | Paragraphs grouped under page titles |
| TriviaQA | Evidence passages grouped under page titles |
| Natural Questions | A complete Wikipedia page as HTML and tokens |

The examples below show the same records in a readable form.

## HotpotQA

**Record ID:** `5a8b57f25542995d1e6f1371`

**Question:** Were Scott Derrickson and Ed Wood of the same nationality?

**Answer:** Yes.

**Text stored in the record:**

**Scott Derrickson**

> Scott Derrickson (born July 16, 1966) is an American director, screenwriter,
> and producer.
>
> He lives in Los Angeles, California.
>
> He is best known for directing horror films such as "Sinister", "The Exorcism
> of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel
> Cinematic Universe installment, "Doctor Strange."

**Ed Wood**

> Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American
> filmmaker, actor, writer, producer, and director.

The record marks the first sentence for both pages as supporting evidence. Both
people are described as American, so the answer is "yes".

## 2WikiMultiHopQA

**Record ID:** `8813f87c0bdd11eba7f7acde48001122`

**Question:** Who is the mother of the director of the film *Polish-Russian War*?

**Answer:** Małgorzata Braunek.

**Text stored in the record:**

**Polish-Russian War (film)**

> Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by
> Xawery Żuławski based on the novel Polish-Russian War under the white-red flag
> by Dorota Masłowska.

**Xawery Żuławski**

> Xawery Żuławski (born 22 December 1971 in Warsaw) is a Polish film director.
> He is the son of actress Małgorzata Braunek and director Andrzej Żuławski.

**How the answer is found:**

1. The film's director is Xawery Żuławski.
2. Xawery Żuławski's mother is Małgorzata Braunek.

Here, `context` only groups the text by page. The `evidences` field stores the
same two links in a compact form: film → director, then director → mother.

## TriviaQA

**Question ID:** `qw_4481`

**Question:** In which country is the port of Incheon?

**Answer:** South Korea.

**Page:** Port of Incheon

> The Port of Incheon is the main port in South Korea, located in Incheon.

The record also stores other accepted answer forms, such as "Korea, South" and
"South Korean". The evidence passage is already in the record. No page fetch is
needed.

## Natural Questions

**Example ID:** `4549465242785278785`

**Question:** When is the last episode of season 8 of *The Walking Dead*?

**Answer:** March 18, 2018.

**Page:** *The Walking Dead (season 8)*

**Text around the answer:**

| Episode | Title | Date |
| ---: | --- | --- |
| 12 | "The Key" | March 18, 2018 |

The raw record stores the complete Wikipedia page in `document_html`. It also
stores a token list and the answer's location in that page. The `document_url`
identifies the Wikipedia revision used for the record. The page text is already
local, so the URL does not need to be fetched to read this example.

See the [Natural Questions data format](https://github.com/google-research-datasets/natural-questions#data-format).
