Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 1 items.

=== ITEM id=3b74a42a8678dd45 ===
Gold sentences and facts:
Sentence 0: The feature is currently available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.
  Fact 0.0: The feature is currently available in France.
  Fact 0.1: The feature is currently available in Germany.
  Fact 0.2: The feature is currently available in Japan.
  Fact 0.3: The feature is currently available in the U.K.
  Fact 0.4: The feature is currently available in the U.S.
  Fact 0.5: There are plans for further expansion of the feature.
Sentence 1: LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
  Fact 1.0: LinkedIn leans into AI tools.
  Fact 1.1: LinkedIn this week unveiled a string of new AI features.
  Fact 1.2: The new LinkedIn AI features span its job hunting products.
  Fact 1.3: The new LinkedIn AI features span its marketing products.
  Fact 1.4: The new LinkedIn AI features span its sales products.
  Fact 1.5: Ingrid writes about LinkedIn's new AI features.

Retrieved items:
<item 1>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 1>

<item 2>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
He gave Alan Eustace, Google’s senior vice president of engineering, what amounted to a blank check to hire any AI expertise he needed. Eustace and Jeff Dean,
who led the Brain lab, flew to Lake Tahoe and took Hinton and his students out to dinner at a steakhouse inside the hotel the night before the auction. The
smell of old cigarettes was overpowering, Dean recalled. They made the case for coming to work at Google. The next day, Hinton ran the auction from his hotel
room. Because of an old back injury, he rarely sat down. He turned a trash can upside down on a table, put his laptop on top and watched the bids roll in over
the next two days.

Google made an offer. So did Microsoft. DeepMind quickly bowed out as the price went up. The industry giants pushed the bids to $US20 million and then $US25
million, according to documents detailing the auction.
</item 2>

<item 3>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
That in and of itself is probably still largely unexplored territory that will see a lot of change and maturing in the next few years. One key aspect there
might be cost monitoring as well since companies like OpenAI charge largely “by the token,” which is a very different metric than most cloud computing.

What can be done to ensure industry-specific startups that tune generative AI models for a specific niche will prove defensible?
</item 3>

<item 4>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Incumbents like Datadog, New Relic and Splunk have all produced LLM observability tools and do appear to be putting a lot of R&D dollars behind this, which may
curtail the market area in the short-term.

However, as we have seen before with the inceptions of the internet and cloud computing, incumbents tend to innovate until innovation becomes stagnant. With AI
becoming a household name that finds use cases in every vertical, startups have the chance to come in with innovative solutions that disrupt and reimagine the
work of incumbents. It’s still too early to say with certainty who the winners will be, as every day reveals new gaps in existing AI frameworks. Therein lie
major opportunities for startups.

How much room in the market do the largest tech companies’ services leave for smaller companies and startups tooling for LLM deployment?
</item 4>

<item 5>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
On the surface, Microsoft is currently looking like the biggest winner amidst the OpenAI turmoil, as it was already apparently looking to reduce its reliance on
OpenAI even though it remains once of its major shareholders. But Facebook’s parent Meta could also stand to benefit, as businesses pursue multi-modal
strategies or models with a more “open” ethos embedded.

“Open source today offers a wide variety of models for companies to essentially diversify,” Ceze added. “By doing so, these startups can quickly pivot and
minimize risk. There is also a major upside — many of these models already outperform the likes of OpenAI’s in terms [of] price-performance and speed.”

A leaked internal memo from Google earlier this year seemed to express fears that despite the huge advances made by proprietary LLM models from the likes of
OpenAI, open source AI would ultimately trump them all. “We have no moat, and neither does OpenAI,” the document noted.
</item 5>

<item 6>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q2: Electric Boog-AI-loo

Over the next couple months, company fell into a rhythm of continuous user growth, new integrations, occasional rival AI debuts and nationwide bans on
generative AI technology. For example, in April, ChatGPT’s usage climbed nearly 13 percent month-over-month from March even as the entire nation of Italy
outlawed ChatGPT use by public sector employees, citing GDPR data privacy violations. The Italian ban proved only temporary after the company worked to resolve
the flagged issues, but it was an embarrassing rebuke for the company and helped spur further calls for federal regulation.

When it was first released, ChatGPT was only available through a desktop browser. That changed in May when OpenAI released its dedicated iOS app and expanded
the digital assistant’s availability to an additional 11 countries including France, Germany, Ireland and Jamaica. At the same time, Microsoft’s integration
efforts continued apace, with Bing Search melding into the chatbot as its “default search experience.”
</item 6>

<item 7>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI promises to defend business customers against copyright claims

Bowing to peer pressure, OpenAI it will pay legal costs incurred by customers who face lawsuits over IP claims against work generated by an OpenAI tool. The
protections seemingly don’t extend to all OpenAI products, like the free and Plus tiers of ChatGPT.

As OpenAI’s multimodal API launches broadly, research shows it’s still flawed

OpenAI announced that GPT-4 with vision will become available alongside the upcoming launch of GPT-4 Turbo API. But some researchers found that the model
remains flawed in several significant and problematic ways.

OpenAI launches API, letting developers build ‘assistants’ into their apps

At its OpenAI DevDay, OpenAI announced the Assistants API to help developers build “agent-like experiences” within their apps. Use cases range from a natural
language-based data analysis app to a coding assistant or even an AI-powered vacation planner.
</item 7>

<item 8>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q3: [Pushback intensifies]

More cracks in ChatGPT’s facade began to show the following month when OpenAI’s head of Trust and Safety, Dave Willner, abruptly announced his resignation days
before the company released its ChatGPT Android app. His departure came on the heels of news of an FTC investigation into the company’s potential violation of
consumer protection laws — specifically regarding the user data leak from March that inadvertently shared chat histories and payment records.

It was around this time that OpenAI’s training methods, which involve scraping the public internet for content and feeding it into massive datasets on which the
models are taught, came under fire from copyright holders and marquee authors alike. Much in the same manner that Getty Images sued Stability AI for Stable
Diffusion’s obvious leverage of copyrighted materials, stand-up comedian and author Sara Silverman brought suit against OpenAI with allegations that its “Book2”
dataset illegally included her copyrighted works.
</item 8>

<item 9>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
“The OpenAI / Microsoft drama underlines one of the big near-term risks with AI — that this next wave of technology is controlled by the same tiny set of
players who have shaped that last era of the internet,” Mark Surman, president and executive director at the Mozilla Foundation, told TechCrunch. “We might have
a chance of avoiding this if GPT-X were responsibly open sourced, giving researchers and startups a shot at making this technology safer, more useful and more
trustworthy for people everywhere.”

Open and shut

In an open letter published by Mozilla a few weeks back, Meta’s chief AI scientist Yann LeCun joined some 70 other signatories in calling for more openness in
AI development, though that letter has since garnered more than 1,700 signatures. The backdrop stems from Big Tech companies such as OpenAI and Google’s
DeepMind calling for more regulation, warning of catastrophic consequences if the AI levers were to meet the wrong hands — in other words, they argued that
proprietary AI is safer than open source.
</item 9>

<item 10>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
(Spoiler alert: Yes.)

Rumors are swilling that OpenAI may be considering developing its own AI chips. That’s going to get interesting, especially if you’ll recall that Nvidia’s
ongoing stock market rally is likely driven by the current boom in AI. The company’s ChatGPT’s mobile app hit a record $4.6 million in revenue last month, but
growth is slowing. Oh, and there’s no need to shed tears for Nvidia quite yet; Brian’s piece breaking down how Nvidia became a major player in robotics is super
interesting.

Adobe has doubled down on its Firefly generative AI models. This week, Frederic covered how the software can now generate more realistic images and can help
artists create vector graphics in Illustrator. Neato.

More on the AI front

If AI can’t go to the mountain, the mountain will come to AI: Dutch startup Tidalflow exits stealth with backing from Google’s Gradient Ventures. It is aiming
to help any software play nice with ChatGPT and other LLM ecosystems.
</item 10>

<item 11>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
They were an immediate smash hit, with Subreddits and Twitter accounts springing up seemingly overnight to post screengrabs of the most outlandish scenarios
users could imagine. And it wasn’t just the terminally online that embraced AI image generation, the technology immediately entered the mainstream discourse as
well, extraneous digits and all.

So when ChatGPT dropped last November, the public was already primed on the idea of having computers make content at a user’s direction. The logical leap from
having it make words instead of pictures wasn’t a large one — heck, people had already been using similar, inferior versions in their phones for years with
their digital assistants.

Q1: [Hyping intensifies]

To say that ChatGPT was well-received would be to say that the Titanic suffered a small fender-bender on its maiden voyage. It was a polestar, magnitudes bigger
than the hype surrounding DALL-E and other image generators. People flat out lost their minds over the new AI and its CEO, Sam Altman.
</item 11>

<item 12>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
In addition to academic partners such as the University of Tokyo, UC Berkeley, the University of Illinois, Cornell and the aforementioned Imperial College
London and Yale, Sony, ServiceNow, the National Science Foundation, NASA, Oracle, the Cleveland Clinic and Dell have pledged their participation in some form.

MLCommons, the engineering consortium behind MLPerf, the benchmarking suite used by major chip manufacturers to evaluate their hardware’s AI performance, is
also a founding AI Alliance member. So are LangChain and LlamaIndex, two creators behind some of the more widely used tools and frameworks for building apps
powered by text-generating AI models.

But without the participation of so many major AI industry players — and lacking deadlines or even concrete objectives — can the AI Alliance succeed? What would
success look like, even?

Beats me.

The vast number of competing interests — from healthcare networks (Cleveland Clinic) to insurance providers (Roadzen) — won’t make it easy for the Alliance’s
members to coalesce around a single, united front.
</item 12>

<item 13>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
It’s a co-founder with Meta after all. I’d venture to guess more exposure for its burgeoning generative AI platform. IBM’s most recent earnings were boosted by
enterprises’ interest in generative AI, but the company has stiff competition in Microsoft and OpenAI (and to a lesser extent Google), which are jointly
developing enterprise-focused AI services that directly compete with IBM’s.

I’ve asked IBM’s PR, which first informed me of the AI Alliance’s founding, about the curious omissions from the early membership, like Stanford (which has a
prominent AI research lab, Stanford HAI), MIT (which is at the forefront of robotics research) and high-profile AI startups like Anthropic, Cohere and Adept. A
press rep didn’t respond as of publication time. But the same philosophical differences that kept Google and Microsoft away likely were at play; I’d wager it’s
no accident that Anthropic, Cohere and Adept have relatively few open source AI projects to their names.
</item 13>

<item 14>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Do you expect startups to build striated solutions for individual layers of the LLM stack, or pursue a more vertical approach?

While there are startups specializing in parts of the stack (like Pinecone), Valor’s focus is on applied AI, which we define as AI that is solving a customer
problem. Saile.ai is a good example — it uses AI to generate closeable leads for the Fortune 500. Or Funding U using its own trained dataset to create a more
useful credit risk score. Or Allelica, using AI on treatment solutions applied to individual DNA to find the best medical treatment for you personally in a
given situation.

Companies like Datadog are building products to support the expanding AI market, including releasing an LLM observability tool. Will efforts like what Datadog
has built (and similar output from large/incumbent tech powers) curtail the market area where startups can build and compete?

Tools like Datadog can only help the acceptance of AI tools if they succeed in monitoring AI performance bottlenecks.
</item 14>

<item 15>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
By the spring of 2010, they were building their dream machine. They wholeheartedly believed that because they understood the risks, they were uniquely
positioned to protect the world. “I don’t see this as a contradictory position,” said Mustafa Suleyman, one of the three DeepMind founders. “There are huge
benefits to come from these technologies. The goal is not to eliminate them or pause their development. The goal is to mitigate the downsides.” Loading Having
won over Thiel, Hassabis worked his way into Musk’s orbit. About two years later, they met at a conference organised by Thiel’s investment fund, which had also
put money into Musk’s company SpaceX. Hassabis secured a tour of SpaceX headquarters. Afterward, with rocket hulls hanging from the ceiling, the two men lunched
in the cafeteria and talked. Musk explained that his plan was to colonise Mars to escape overpopulation and other dangers on Earth.
</item 15>

<item 16>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Instead, users will automatically be switched to models based on the prompt.

ChatGPT officially gets web search

OpenAI has formally launched its internet-browsing feature to ChatGPT, some three weeks after re-introducing the feature in beta after several months in hiatus.
The AI chatbot that has historically been limited to data up to September, 2021.

OpenAI integrates DALL-E 3 into ChatGPT

The integration means users don’t have to think so carefully about their text-prompts when asking DALL-E to create an image. Users will also now be able to
receive images as part of their text-based queries without having to switch between apps.

Microsoft-affiliated research finds flaws in GPT-4

A Microsoft-affiliated scientific paper looked at the “trustworthiness” — and toxicity — of LLMs, including GPT-4.
</item 16>

<item 17>
The restaurant nearest Google (The Verge, 2023-10-26T13:00:00+00:00)
Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence. To improve local
ranking, Google encourages businesses to provide a trove of information on their profile. Owners should have basics like updated hours of operation and accurate
location details, but they should also respond to reviews, add photos, and even post individual items in stock to their Google page.

“I don’t think they’re really great marketing campaigns”

The prominence measure in Maps includes data from the web — links to the business, how the business ranks in traditional search, and press coverage, for
example. In this way, SEO impacts how a business appears in local search results, and news articles and links are one way Google tries to determine the
relevance of a page. (It’s why reporters’ inboxes are overrun with spammy requests to buy “guest posts” on our sites from marketers.)
</item 17>

<item 18>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
April 2023

ChatGPT parent company OpenAI closes $300M share sale at $27B-29B valuation

VC firms including Sequoia Capital, Andreessen Horowitz, Thrive and K2 Global are picking up new shares, according to documents seen by TechCrunch. A source
tells us Founders Fund is also investing. Altogether the VCs have put in just over $300 million at a valuation of $27 billion to $29 billion. This is separate
to a big investment from Microsoft announced earlier this year, a person familiar with the development told TechCrunch, which closed in January. The size of
Microsoft’s investment is believed to be around $10 billion, a figure we confirmed with our source.

OpenAI previews new subscription tier, ChatGPT Business

Called ChatGPT Business, OpenAI describes the forthcoming offering as “for professionals who need more control over their data as well as enterprises seeking to
manage their end users.”
</item 18>

<item 19>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Within the enterprise sector, there’s a clear recognition of the value of AI. However, many lack the internal capabilities to develop AI solutions. This gap
presents a significant opportunity for startups specializing in AI to engage with enterprise clients. As the business landscape matures, proficiency in
leveraging AI is becoming a strategic imperative.

McKinsey reports that generative AI alone can add up to $4.4 trillion in value across industries through writing code, analyzing consumer trends, personalizing
customer service, improving operating efficiencies, and more. Ninety-four percent of business leaders agree AI will be critical to all businesses’ success over
the next five years, and total global spending on AI is expected to reach $154 billion by the end of this year, a 27% increase from 2022. The next three years
are also expected to see a compound annual growth rate of 27% — the annual AI spending in 2026 will be over $300 billion.
</item 19>

<item 20>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
This standard, known as Digital Imaging and Communications in Medicine (DICOM), is the internationally recognized format for medical imaging. But as discovered
by Aplite, a Germany-based cybersecurity consultancy, security shortcomings in DICOM mean many medical facilities have unintentionally made private data
accessible to the open web.

Meta generates images: Not to be outdone by Google’s Gemini launch, Meta rolled out a new, stand-alone generative AI experience on the web, Imagine with Meta
AI, that allows users to create images by describing them in natural language. Similar to OpenAI’s DALL-E, Midjourney and Stable Diffusion, Imagine with Meta
AI, which is powered by Meta’s existing Emu image-generation model, creates high-resolution images from text prompts.

Spotify makes cuts: Spotify is eliminating about 1,500 jobs, or roughly 17% of its workforce, in its third round of layoffs this year as the music streaming
giant looks to become “both productive and efficient.”
</item 20>

<item 21>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
And for all their talk of openness, IBM and Meta aren’t exactly the poster children for the future that the Alliance’s release depicts — casting doubt on their
sincerity.

Perhaps I’m wrong and the AI Alliance will be a smash success. Or perhaps it’ll crumble under mistrust and its own bureaucracy. We’ll see; time will tell.
</item 21>

<item 22>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Legal cases build and break

We saw a fair number of lawsuits filed in 2023, but few saw any real movement, let alone success. Most suits over copyright and other missteps in the AI
industry are still pending. 2024 will see a lot of them fall by the wayside, as companies stonewall critical information like training data and methods, making
allegations like the use of thousands of copyrighted books difficult to prove in court.

This was only the beginning, however, and many of these lawsuits were filed essentially on principle. Though they may not succeed, they may crack the process
open far enough during testimony and discovery that companies would rather settle than have certain information come to light. 2024 will bring new lawsuits as
well, ones pertaining to misuse and abuse of AI, such as wrongful termination, bias in hiring and lending, and other areas where AI is being put to work without
a lot of thought.
</item 22>

<item 23>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Despite cloud computing remaining critical, AI budgets are now more than double that of cloud computing. Eighty-two percent of business leaders believe the
integration of AI solutions will increase their employee performance and job satisfaction, and startups should expect a high level of desire for and experience
with AI solutions in their future customers.

Finally, we’ve seen consumption, or usage-based priced tech products’ growth slow in recent quarters. Will that fact lead startups building modern AI tools to
pursue more traditional SaaS pricing? (The OpenAI pricing schema based on tokens and usage led us to this question.)

The trajectory of usage-based pricing has organically aligned with the needs of large language models, given that there is significant variation in
prompt/output sizes and resource utilization per user. OpenAI itself racks upward of $700,000 per day on compute, so to achieve profitability, these operation
costs need to be allocated effectively.

Nevertheless, we’ve seen the sentiment that tying all costs to volume is generally unpopular with end users, who prefer predictable systems that allow them to
budget more effectively.
</item 23>

<item 24>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
Well — confusingly — the Partnership on AI is in fact a member of the AI Alliance. The Alliance says that it plans to “utilize pre-existing collaborations”
(including the Partnership on AI’s, presumably) to “identify opportunities that develop open AI resources that meet the needs of business and society equally
and responsibly,” a press release shared last week with TechCrunch reads.

The AI Alliance’s members will first form working groups, a governing board and a technical oversight committee dedicated to advancing areas like AI “trust and
validation” metrics, hardware and infrastructure that supports AI training and open source AI models and frameworks. They’ll also establish project standards
and guidelines, and then partner with “important existing initiatives” — initiatives conspicuously not named in the press release — from government, nonprofit
and civil society organizations “who are doing valuable and aligned work in the AI space.”

If that sounds a lot like what the inaugural members of the Alliance were already doing independently, you’re not wrong.
</item 24>

<item 25>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In a post on X, CEO Sam Altman looked back on the night before its launch: “what a year it’s been…”

a year ago tonight we were probably just sitting around the office putting the finishing touches on chatgpt before the next morning’s launch. what a year it’s
been… — Sam Altman (@sama) November 30, 2023

Apple and Google avoid naming ChatGPT as their ‘app of the year’

Neither Apple nor Google chose an AI app as its app of the year for 2023, despite the success of ChatGPT’s mobile app, which became the fastest-growing consumer
application in history before the record was broken by Meta’s Threads.

An attack from researchers prompts ChatGPT to reveal training data

A test led by researchers at Google DeepMind found that there is a significant amount of privately identifiable information in OpenAI’s LLMs.
</item 25>

<item 26>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
This part of the stack includes cutting-edge capabilities, encompassing model fine-tuning, prompt engineering and agile model orchestration. It’s here that we
anticipate the rise of entities akin to Databricks. Yet, the competitive dynamics of this layer present a unique challenge. Primarily, the emergence of
foundation model providers expanding into middle-layer tools heightens commoditization risks. Additionally, established market leaders venturing into this space
further intensify the competition. Consequently, despite a surge in startups within this domain, clear winners still need to be discovered.

Companies like Datadog are building products to support the expanding AI market, including releasing an LLM observability tool. Will efforts like what Datadog
has built (and similar output from large/incumbent tech powers) curtail the market area where startups can build and compete?

LLM observability falls within the “middle layer” category, acting as a catalyst for specialized business applications to use foundational models.
</item 26>

<item 27>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Nadella told Scott to get it done. The deal closed in 2019. Altman and OpenAI had formed a for-profit company under the original nonprofit, they had $US1
billion in fresh capital, and Microsoft had a new way to build AI into its vast cloud computing service. Not everyone inside OpenAI was happy.

Dario Amodei, a researcher with ties to the effective altruist community, had been on hand at the Rosewood hotel when OpenAI was born. Amodei, who endlessly
twisted his curls between his fingers as he talked, was leading the lab’s efforts to build a neural network called a large language model that could learn from
enormous amounts of digital text. Seeking the path to artificial general intelligence, AGI. Credit: iStock By analysing countless Wikipedia articles, digital
books and message boards, it could generate text on its own. It also had the unfortunate habit of making things up. It was called GPT-3, and it was released in
the summer of 2020.
</item 27>

<item 28>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Hinton turned Baidu down, but the money got his attention. The Cambridge-educated British expatriate had spent most of his career in academia, except for
occasional stints at Microsoft and Google, and was not especially driven by money. But he had a neurodivergent child, and the money would mean financial
security. “We did not know how much we were worth,” Hinton said. He consulted lawyers and experts on acquisitions and came up with a plan: “We would organise an
auction, and we would sell ourselves.” The auction would take place during an annual AI conference at the Harrah’s hotel and casino on Lake Tahoe. Big Tech took
notice.

Google, Microsoft, Baidu and other companies were beginning to believe that neural networks were a path to machines that could not only see but also hear,
write, talk and — eventually — think. Loading Page had seen similar technology at Google Brain, his company’s AI lab, and he thought Hinton’s research could
elevate his scientists’ work.
</item 28>

<item 29>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
As the price passed $US30 million, Microsoft quit, but it rejoined the bidding at $US37 million. “We felt like we were in a movie,” Hinton said. Then Microsoft
dropped out a second time. Only Baidu and Google were left, and they pushed the bidding to $US42 million, $US43 million. Finally, at $US44 million, Hinton and
his students stopped the auction. The bids were still climbing, but they wanted to work for Google. And the money was staggering. It was an unmistakable sign
that deep-pocketed companies were determined to buy the most talented AI researchers, which was not lost on Hassabis at DeepMind. He had always told his
employees that DeepMind would remain an independent company. That was, he believed, the best way to ensure its technology didn’t turn into something dangerous.
But as Big Tech entered the talent race, he decided he had no choice: It was time to sell.
</item 29>

<item 30>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
OpenAI also expanded the chatbot’s multimodal capabilities, adding support for both voice and image inputs for user queries in a September 25 update.

Q4: Starring Sam Altman as “Lazarus”

The fourth quarter of 2023 has been a hell of a decade for OpenAI. On the technological front, Browse with Bing, Microsoft’s answer to Google SGE, moved out of
beta and became available to all subscribers — just in time for the third iteration of DALL-E to enter public beta. Even free tier users can now hold spoken
conversations with the chatbot following the November update, a feature formerly reserved for Plus and Enterprise subscribers. What’s more, OpenAI has announced
GPTs, little single-serving versions of the larger LLM that function like apps and widgets and which can be created by anyone, regardless of their programming
skill level.

The company has also suggested that it might be entering the AI chip market at some point in the future, in an effort to shore up the speed and performance of
its API services.
</item 30>

<item 31>
Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation (TechCrunch, 2023-10-28T13:04:25+00:00)
Why don’t we stitch those together in a meaningful way?

To help determine the best method for picking.

Exactly. That ultimately helps our designers determine and algorithmically prove that was the best method. The theme here is that generative AI has a lot of
promise, particularly in influencing our designers to make a better system.

I was recently speaking with Daniela Rus, and she was excited by the concept of using generative AI to literally design robots.

The dynamics of the robots, to literally move the robots — path planning to actually figure out how to get the right angles — generative AI is incredible at
that. We’re seeing a lot of promise with that today.

What about real-world problem-solving?

It’s another good example. I want to be careful on generative AI versus the machine learning systems that we have. We have what we call “flow” inside the
building. We have machine learning systems that understand what line needs what at what time and can help divert the right material flow to the right stations,
for example.
</item 31>

<item 32>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
“This is another indication that AI is progressing faster than even many experts anticipated,” Victoria Krakovna, who would soon join DeepMind as an “AI safety”
researcher, wrote in a blog post. DeepMind’s founders were increasingly worried about what Google would do with their inventions. In 2017, they tried to break
away from the company. Google responded by increasing the salaries and stock award packages of the DeepMind founders and their staff. They stayed put. The
ethics board never had a second meeting. The Breakup

Convinced that Page’s optimistic view of AI was dead wrong, and angry at his loss of DeepMind, Musk built his own lab. OpenAI was founded in late 2015, just a
few months after he met with Altman at the Rosewood hotel in Silicon Valley. Sam Altman Credit: Bloomberg Businessweek Musk pumped money into the lab, and his
former PayPal buddies – Hoffman and Thiel – came along for the ride.
</item 32>

<item 33>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
First, the application must address a real enterprise pain point prioritized by executives. Second, to provide tangible benefits and long-term differentiation,
the application should be composed of cutting-edge models that fit the specific and unique needs of the software. It’s not enough to simply plug into OpenAI;
rather, applications should choose their models intentionally while balancing cost, compute, and performance.

Third, the application is only as sophisticated as the data that it is fed. Proprietary data is necessary for specific and relevant insights and to ensure
others cannot replicate the final product. To this end, in-house middle-layer capabilities provide a competitive edge while harnessing the power of foundational
models. Finally, due to the inevitable margin of error of generative AI, the niche market must tolerate imprecision, which is inherently found in subjective and
ambiguous content, like sales or marketing.

How much technical competence can startups presume that their future enterprise AI customers will have in-house, and how much does that presumed expertise guide
startup product selection and go-to-market motion?
</item 33>

<item 34>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In its privacy policy, the ChatGPT maker makes a passing acknowledgement of the objection requirements attached to relying on “legitimate interest” (LI),
pointing users towards more information about requesting an opt out — when it writes: “See here for instructions on how you can opt out of our use of your
information to train our models.”

What controversies have surrounded ChatGPT?

Recently, Discord announced that it had integrated OpenAI’s technology into its bot named Clyde where two users tricked Clyde into providing them with
instructions for making the illegal drug methamphetamine (meth) and the incendiary mixture napalm.

An Australian mayor has publicly announced he may sue OpenAI for defamation due to ChatGPT’s false claims that he had served time in prison for bribery. This
would be the first defamation lawsuit against the text-generating service.

CNET found itself in the midst of controversy after Futurism reported the publication was publishing articles under a mysterious byline completely generated by
AI.
</item 34>

<item 35>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
This got the attention of another Silicon Valley powerhouse, Google, and specifically Page. He saw a demonstration of DeepMind’s machine playing Atari games. He
wanted in. The talent auction In the spring of 2012, Geoffrey Hinton, a 64-year-old professor at the University of Toronto, and two graduate students published
a research paper that showed the world what AI could do. They trained a neural network to recognise common objects such as flowers, dogs and cars.

Scientists were surprised by the accuracy of the technology built by Hinton and his students. One who took particular notice was Yu Kai, an AI researcher who
had met Hinton at a research conference and had recently started working for Baidu, a giant Chinese internet company. Baidu offered Hinton and his students
$US12 million to join the company in Beijing, according to three people familiar with the offer. Hinton turned Baidu down, but the money got his attention.
</item 35>

<item 36>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
The Bengaluru-headquartered startup employed about 150 people at peak and raised more than $130 million over its eight-year journey.

Audio

TechCrunch’s roster of podcast episodes keeps growing — just in time for weekend listening.

Equity featured a throwback conversation from TechCrunch Disrupt 2023, when Alex sat down with Serhii Bohoslovskyi, the founder of a no-code app builder,
Trible, that helps people construct online courses. The pair caught up on the state of the creator economy, the use of no-code tooling today (and how it’s
received by nontechnical creators) and the security of startups with roots in Ukraine.

Over on Found, the crew talked to David Rogier, the CEO and founder of MasterClass, a streaming platform where you can learn from the world’s experts on a range
of topics.
</item 36>

<item 37>
Robotics Q&A with Boston Dynamics’ Aaron Saunders (TechCrunch, 2023-12-09T16:05:51+00:00)
For the next few weeks, TechCrunch’s robotics newsletter Actuator will be running Q&As with some of the top minds in robotics. Subscribe here for future
updates.

Part 1: CMU’s Matthew Johnson-Roberson

Part 2: Toyota Research Institute’s Max Bajracharya and Russ Tedrake

Part 3: Meta’s Dhruv Batra

This time it’s Boston Dynamics CTO, Aaron Saunders. He has been with the company for more than 20 years, most recently serving as its vice president of
Engineering.

What role(s) will generative AI play in the future of robotics?

The current rate of change makes it hard to predict very far into the future. Foundation models represent a major shift in how the best machine learning models
are created, and we are already seeing some impressive near-term accelerations in natural language interfaces. They offer opportunities to create conversational
interfaces to our robots, improve the quality of existing computer vision functions and potentially enable new customer-facing capabilities such as visual
question answering.
</item 37>

<item 38>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
You cannot escape conversations about AI no matter how far or fast you run. Hyperbole abounds around what current AI tech will be able to do (revolutionize
every industry!) and what current AI tech will be able to do (take over the world!). Closer to the ground, TechCrunch+ is working to understand where startups
might find footholds in the market by levering large language models (LLMs), a recent and impactful new method of creating artificially intelligent software.

How AI will play in startup land is not a new topic of conversation. A few years back, one venture firm asked how AI-focused startups would monetize and whether
they would suffer from impaired margins due to costs relating to running models on behalf of customers. That conversation died down, only to come roaring back
in recent quarters as it became clear that while LLM technology is quickly advancing, it’s hardly cheap to run in its present form.

But costs are only one area where we have unanswered questions.
</item 38>

<item 39>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
I’ll note that Nvidia isn’t a member of the AI Alliance, either — a suspect absence given that the company is by far the dominant provider of AI chips and a
maintainer of many open source models in its own right. Perhaps the chipmaker perceived a conflict of interest in collaborating with Intel and AMD. Or perhaps
it decided to cast its lot with Microsoft, Google and the rest of the tech giants opting out of the Alliance for strategic reasons. Who can say?

Sriram Raghavan, VP of IBM’s research AI division, told me via email that the Alliance is, for now, focused on “members that are strongly committed to open
innovation and open source AI” — implying that those who aren’t participating aren’t as strongly committed. I’m not sure they’d agree.

“This of course is just the starting point,” he added. “We welcome and expect more organizations to join in the future.”
</item 39>

<item 40>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
For sure, Meta isn’t the only company flaunting its “open” approach to AI development — notably, the likes of Hugging Face, Mistral AI and 01.AI, which have all
raised sizeable sums at lofty valuations with similar goals in mind. But as a $900 billion juggernaut with a long history of courting developers through open
source endeavors, Meta is perhaps best positioned to capitalize on the mess that OpenAI has created for itself. Its decision to pursue “openness” over
“closedness” seems to be well vindicated right now, and regardless of whether Llama is or isn’t really open source, it’s likely “open enough” for most people.

It’s still too early to make any surefire claims on what impact the OpenAI fallout will have on LLM development and uptake in the future. Altman and Brockman
are undoubtedly steady hands for a commercial AI startup, and they may even return to steward OpenAI.
</item 40>

<item 41>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
But costs are only one area where we have unanswered questions. We are also incredibly curious about how startups should approach building tools for AI
technologies, how defensible startup-focused AI work will prove, and how upstart tech companies should charge for AI-powered tooling.

With the amount of capital flowing to startups working with and building AI today, it’s critical that we understand the market as best we can. So we asked a
number of venture capitalists who are active in the AI investing space to walk us through what they are seeing in the market today.

What we learned from the investing side of the house was useful. Rick Grinnell, founder and managing partner at Glasswing Ventures, said that within the new AI
tech stack, “most of the opportunity lies in the application layer,” where “the best applications will harness their in-house expertise to build specialized
middle-layer tooling and blend them with the appropriate foundational models.”
</item 41>

<item 42>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
Now, those companies have a clear horse in the race and perhaps regulatory capture on the mind… but they’re not wrong entirely. Meta continues to take
calculated open sourcing risks (within the bounds of regulators’ tolerances), releasing text-generating models like Llama that bad actors have gone on to abuse
but which plenty of developers have built useful apps upon.

“The platform that will win will be the open one,” Yann LeCun, Meta’s chief AI scientist, was quoted as saying in an interview with The New York Times — and
who’s among the more than 70 influential signers of a letter calling for more openness in AI development. LeCun has a point; according to one estimate,
Stability AI’s open source AI-powered image generator, Stable Diffusion, released last August, is now responsible for 80% of all AI-generated imagery.

But wait, you might say — what does IBM gain from the AI Alliance? It’s a co-founder with Meta after all.
</item 42>

<item 43>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
8 predictions for AI in 2024 How will AI impact the US primary elections? What's next for OpenAI? Here are our predictions for AI in 2024.

This last year was a banger for AI as the technology went from niche to mainstream about as fast as anything ever has. 2024, however, will be the year when the
hype runs full-steam into reality as people reckon with the capabilities and limitations of AI at large. Here are a few ways we think that’s going to play out.

OpenAI becomes a product company

After the leadership shake-up in November, OpenAI is going to be a changed company — perhaps not outwardly, but the trickle-down effect of Sam Altman being more
fully in charge will be felt at every level. And one of the ways we expect that to manifest is in “ship it” mindset.

We’ll see that with the GPT store, originally planned for launch in December but understandably delayed due to the C-suite fracas.
</item 43>

<item 44>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
But 2024 will be the year something gives. Where exactly it is impossible to predict, as research is active at the frontiers of this field.

The seemingly magical “emergent” capabilities of LLMs will be better studied and understood in 2024, and things like their inability to multiply large numbers
will make more sense.

In parallel, we will begin to see diminishing returns on parameter counts, to the point where training a 500-billion-parameter model may technically produce
better results, but the compute required to do so could provably be deployed more effectively. A single monolithic model is unwieldy and expensive, while a
mixture of experts — a collection of smaller, more specific models and likely multimodal ones — may prove almost as effective while being much easier to update
piecemeal.

Marketing meets reality

The simple fact is that the hype built up in 2023 is going to be very hard for companies to follow through on.
</item 44>

<item 45>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
The “app store for AI” will be pushed hard as the platform to get your AI toys and tools from, and never mind Hugging Face or any open source models. They have
an excellent model to work from, Apple’s, and will follow it all the way to the bank.

Expect more moves like that from 2024’s OpenAI as the caution and academic reserve that the previous board exerted gives way to an unseemly lust for markets and
customers.

Other major companies with AI efforts will also follow this trend (for instance, expect Gemini/Bard to horn in on a ton of Google products), but I suspect it
will be more pronounced in this case.

Agents, generated video and generated music graduate from quaint to experimental

Some niche applications of AI models will grow beyond “eh” status in 2024, including agent-based models and generative multimedia.

If AI is going to help you do more than summarize or make lists of things, it’ll need access to things like your spreadsheets, ticket buying interfaces,
transportation apps and so on.
</item 45>

<item 46>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 46>

<item 47>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 47>

<item 48>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
The specifics of the process and technologies are only important as they pertain to how well they solve the problem, the costs to do so, and the overall end
user experience.

Product innovation is experimental and should not always be expected to yield productive results. It requires a learning curve and patience, as the outcomes are
often ambiguous and unknown. Business leadership can struggle with this perspective because it is indefinite (in terms of results and timelines), and it is
challenging to translate pure technology innovation into value creation. A gap opens between technology and product teams wherein technology teams strain to
articulate the capabilities and value of a technology innovation, leading to unfilled promises and the perception of “technology for the sake of technology” or
quips like “a solution looking for a problem.”

The current hype cycle in AI serves as a great concrete example. For technology or product executives, the challenge is how to do more than check the AI box —
it’s about how to meaningfully incorporate AI into a product. Instead of forcing a technology onto a product, the design process flows to the technologies.
</item 48>

<item 49>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
“The co-founders themselves came to the conclusion that they wished to depart OpenAI to start their own company, made this known to OpenAI’s leadership, and
over several weeks negotiated an exit on mutually agreeable terms.” Anthropic accepted a $US4 billion investment from Amazon and another $US2 billion from
Google two years later. The Reveal After OpenAI received another $US2 billion from Microsoft, Altman and another senior executive, Greg Brockman, visited Bill
Gates at his sprawling mansion on the shores of Lake Washington, outside Seattle. The Microsoft founder was no longer involved in the company day to day but
kept in regular touch with its executives.

Over dinner, Gates told them he doubted that large language models could work. He would stay sceptical, he said, until the technology performed a task that
required critical thinking – passing an Advanced Placement biology test, for instance. Five months later, on August 24, 2022, Altman and Brockman returned and
brought along an OpenAI researcher named Chelsea Voss.
</item 49>

<item 50>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
Meta, on an open source tear, wants to spread its influence in the ongoing battle for AI mindshare.

This morning, the social network announced that it’s teaming up with IBM, whose audience is decidedly more corporate and enterprise, to launch the AI Alliance,
an industry body to support “open innovation” and “open science” in AI.

So what will the AI Alliance do exactly — and how will its work differ from the quite similar (at least in terms of its overarching mission, members and tenets)
Partnership on AI? The Partnership on AI years ago promised to publish research using open source licenses and minutes from its meetings to, as the AI Alliance
purportedly seeks to do, educate the public on pressing AI issues of the day.

Well — confusingly — the Partnership on AI is in fact a member of the AI Alliance.
</item 50>

<item 51>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
2023 saw a few tentative attempts at this “agent” approach, but none really caught on. We don’t really expect any to really take off in 2024, either, but
agent-based models will show their stuff a little more convincingly than they did last year, and a few clutch use cases will show up for famously tedious
processes like submitting insurance claims.

Video and audio will also find niches where their shortcomings aren’t quite so visible. In the hands of skilled creators, a lack of photorealism isn’t a
problem, and we’ll see AI video used in fun and interesting ways. Likewise, generative music models will likely make it into a few major productions like games,
again where professional musicians can leverage the tools to create an unending soundtrack.

The limits of monolithic LLMs become clearer

So far there has been great optimism about the capabilities of large language models, which have indeed proved more capable than anyone expected, and have grown
correspondingly more so as more compute is added. But 2024 will be the year something gives.
</item 51>

<item 52>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 52>

<item 53>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
It is aiming to help any software play nice with ChatGPT and other LLM ecosystems.

What big eyes you have: Adobe continues to push for easier image editing, showing off its Project Stardust as a sneak preview of its next-gen AI photo editing
engine.

Holding back the tide: Creatives across industries are strategizing in a campaign that targets potential corporate abuse of AI technology. The conglomeration is
realistic about the ways that musicians and some other creatives could benefit on an individual level from automating parts of their work. The goal is that AI
tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

Tech you can touch

A while back I argued that Apple’s new AR headset is a game-changer for startups. It seems that’s likely the case at the high end. But more in the realm of
affordability, Meta Quest 3 takes a step closer to mainstream AR/VR, Brian reports.
</item 53>

<item 54>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The report also claims Microsoft has set up a “robust user reporting and appeal process to review and respond to user concerns of harmful or misleading
content”.

Over the next six months, the report does not commit Bing Search to any specific additional steps to address risk attached to the use of generative AI —
Microsoft just says it’s keeping a watching brief, writing: “Bing is regularly reviewing and evaluating its policies and practices related to existing and new
Bing features and adjusts and updates policies as needed.”

TikTok

In its report, TikTok focuses on AI-generated content in the context of ensuring the “integrity” of its services — flagging a recent update to its community
guidelines which also saw it modify its synthetic media policy “to address the use of content created or modified by AI technology on our platform”.

“While we welcome the creativity that new AI may unlock, in line with our updated policy, users must proactively disclose when their content is AI-generated or
manipulated but shows realistic scenes,” it also writes.
</item 54>

<item 55>
There’s something going on with AI startups in France (TechCrunch, 2023-11-09T14:51:44+00:00)
They sometimes move to the U.S. for postdoctoral research and then move back to France to join a research lab.

For instance, Facebook (now Meta) created its Paris research lab back in 2015 with Yann LeCun at the helm of the AI initiative — this research group is called
FAIR, for Facebook Artificial Intelligence Research. Google followed suit with its own AI research center dedicated to AI in Paris.

“The FAIR team is extremely impressive. Just look at what they’ve done with Llama,” an entrepreneur told me yesterday. He recently had a private meeting with
some of the engineers and scientists working on AI projects.

And if you look at the founders of Mistral AI, the French AI startup working on open source large language models that raised a €105 million seed round ($112
million at today’s exchange rate), all three co-founders (Arthur Mensch, Timothée Lacroix and Guillaume Lample) previously worked at Google’s DeepMind or FAIR.
</item 55>

<item 56>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 56>

<item 57>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
But while a few egregious examples of misuse will be punished, a lack of relevant laws specific to it means that it will necessarily only haphazardly be brought
to court. On that note…

Early adopters take new rules by the horns

Big moves like the EU’s AI Act could change how the industry works, but they tend to be slow to take effect. That’s by design, so companies don’t have to adjust
to new rules overnight, but it also means that we won’t see the effect of these big laws for a good while except among those willing to make changes
preemptively and voluntarily. There will be a lot of “we are beginning the process of…” talk. (Also expect a few quiet lawsuits challenging various parts of
laws.)

To that end we can expect a newly flourishing AI compliance industry as the billions going into the technology prompt matching investments (at a smaller scale,
but still considerable) in making sure the tools and processes meet international and local standards.
</item 57>

<item 58>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The latest funding round brings QI Tech’s total capital raised to $262 million.

New Web Summit CEO appointed: Former Wikimedia Foundation CEO Katherine Maher has been appointed as the new CEO of Web Summit, following the controversial
departure of its previous leader, Paddy Cosgrave. Maher’s appointment comes as the tech conference faces a challenge to rebuild relationships and trust within
the tech community.

Top reads on TechCrunch this week

Shinier, smarter maps: Google is rolling out a variety of updates for Maps, incorporating AI technology to enhance user experience. The feature is currently
available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.

From stop-and-start to charging ahead: Over on TC+, Tim has had it with U.S. automakers hesitating on EVs, as Ford and GM are showing signs of backing away from
their commitment to electric vehicles. Cowards.
</item 58>

<item 59>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
But in the release, the AI Alliance stresses that its work — whatever form it ultimately takes — is intended to be complementary and additive rather than
needlessly duplicative.

“[M]ore collaboration and information sharing will help the community innovate faster and more inclusively, and identify specific risks and mitigate those risks
before putting a product into the world,” the release reads. “This stands in contrast to a vision that aims to relegate AI innovation and value creation to a
small number of companies with a closed, proprietary vision for the AI industry.”

Key subtext

That jab at the end says a lot about Meta’s ulterior motives, here.

Google, OpenAI and Microsoft, a close OpenAI partner and investor, have been among the chief critics of Meta’s open source AI approach, arguing that it’s
potentially dangerous and disinformation-encouraging. (Unsurprisingly, none are members of the AI Alliance despite being longtime members of the Partnership on
AI.)
</item 59>

<item 60>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
It has been a whirlwind four days for OpenAI, the generative AI poster child behind the smash hit ChatGPT.

Seemingly out of nowhere, the OpenAI board ousted CEO and co-founder Sam Altman and demoted president and co-founder Greg Brockman, who subsequently resigned,
paving the way for what looked like a mutiny by staff insisting the founders be reinstated post-haste. By then, Microsoft had already hired Altman and Brockman
to head up a new internal AI unit, though, as things transpired, nothing had actually been signed yet, with rumors suggesting that the ousted leaders might
actually return to OpenAI after all — in some capacity, at least.

The situation remains fluid, and any number of potential outcomes still remain on the table. But the whole debacle has shone a spotlight on the forces that
control the burgeoning AI revolution, leading many to question what happens if you go all-in on a centralized proprietary player, and what happens if things
then go belly-up?
</item 60>

<item 61>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
That’s at the root of the research we shared from the institute last week. Noted senior research scientist Benjamin Burchfiel:

We’ve seen some big progress with the advent of [large language models], using them to impart this high level of cognitive intelligence into robots. If you have
a robot that picks up a thing, now instead of having to specify an object, you can tell it to pick up the can of Coke. Or you can tell it to pick up the shiny
object, or you can do the same thing and do it in French. That’s really great, but if you want a robot to plug in a USB device or pick up a tissue, those models
just don’t work. They’re really useful, but they don’t solve that part of the problem. We’re focused on filling in that missing piece, and the thing we’re
really excited about now is that we actually have a system and that the fundamentals are correct.
</item 61>

<item 62>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
The article is well worth a read, to figure out in which circumstances you can expect a decent return on investment on your event-going antics.

Today’s newsletter is going to be a bit more to the point than usual: I’m laid up at home with pneumonia (yes, I really am an 86-year-old grandmother.
Surprise!), so forgive the antibiotics-addled ramblings this week. I hope I’ll be back with non-pharmaceutically-enhanced ramblings next week.

Artificial intelligence: Y’all just can’t get enough

Our most-read stories consistently continue to be about AI. No big surprise, perhaps, the AI hype cycle continues apace. This week there’s been a bunch of
stories about the seedy underbelly of AI, including how humans are part of the problem, continuing to trick AI systems into generating boobs and 9/11 memes. Oh,
humans. We spoke with investors to figure out whether the future of AI has hope for us beyond daft memes. (Spoiler alert: Yes.)
</item 62>

<item 63>
When it comes to generative AI in the enterprise, CIOs are taking it slow (TechCrunch, 2023-12-15T18:45:09+00:00)
A July Morgan Stanley survey of large company CIOs found that many were proceeding cautiously, with 56% of respondents reporting that generative AI was having
an impact on their investment priorities, but only 4% had actually launched significant projects. In fact, most were still in the evaluation or proof of concept
phase. This may be a fast moving area, but it fits with what we’re hearing in conversations with CIOs as well.

That said, much like the consumerization of IT a decade ago, CIOs are under pressure to deliver the kind of experiences people are seeing when they play with
ChatGPT online, says Jon Turow, a partner at Madrona Ventures.

“I think it’s undeniable that enterprise employees, who are the internal customers of the CIO or CTO, have all tried ChatGPT and they know what amazing looks
like. They know where it’s early, and they know where it’s inspiring, and for lack of a better word, where they see greatness.
</item 63>

<item 64>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
“We welcome and expect more organizations to join in the future.”

A broad assembly

Counting around 45 organizations among its membership, including AMD and Intel, the research lab CERN, universities like Yale and the Imperial College London
and AI startups Stability AI and Hugging Face, the AI Alliance will focus on fostering an “open” community and enabling developers and researchers to
“accelerate responsible innovation in AI” while “ensuring scientific rigor, trust, safety, security, diversity and economic competitiveness,” according to the
release.

“By bringing together leading developers, scientists, academic institutions, companies and other innovators, we’ll pool resources and knowledge to address
safety concerns while providing a platform for sharing and developing solutions that fit the needs of researchers, developers and adopters around the world,”
the release reads.

The AI Alliance’s initial cohort is exceptionally broad — sitting at the intersection of not just AI and enterprise but healthcare, silicon and
software-as-a-service as well.
</item 64>

<item 65>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
AI was once laughed off. Now the richest men in Silicon Valley were shelling out billions to keep from being left behind. The lost ethics board

When Musk invested in DeepMind, he broke his own informal rule – that he would not invest in any company he didn’t run himself. The downsides of his decision
were already apparent when, only a month or so after his birthday spat with Page, he again found himself face to face with his former friend and fellow
billionaire. Larry Page, co-founder of Google. Credit: Bloomberg The occasion was the first meeting of DeepMind’s ethics board, on August 14, 2015. The board
had been set up at the insistence of the startup’s founders to ensure that their technology did no harm after the sale. The members convened in a conference
room just outside Musk’s office at SpaceX. But that’s where Musk’s control ended. When Google bought DeepMind, it bought the whole thing. Musk was out.
</item 65>

<item 66>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
When considering the landscape of foundational layer model providers like Alphabet/Google’s Bard, Microsoft/OpenAI’s GPT-4, and Anthropic’s Claude, it’s evident
that the more significant players possess inherent advantages regarding data accessibility, talent pool and computational resources. We expect this layer to
settle into an oligopolistic structure like the cloud provider market, albeit with the addition of a strong open source contingency that will drive considerable
third-party adoption.

As we look at the generative AI tech stack, the largest market opportunity lies above the model itself. Companies that introduce AI-powered APIs and operational
layers for specific industries will create brand-new use cases and transform workflows. By embracing this technology to revolutionize workflows, these companies
stand to unlock substantial value.

However, it’s essential to recognize that the market is still far from being crystallized. LLMs are still in their infancy, with adoption at large corporations
and startups lacking full maturity and refinement. We need robust tools and platforms to enable broader utilization among businesses and individuals.
</item 66>

<item 67>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
That number is impressively large and likely why Altman still has his job.

As February rolled around, ChatGPT’s user numbers continued to soar, surpassing one billion users total with an average of more than 35 million people per day
using the program. At this point OpenAI was reportedly worth just under $30 billion and Microsoft was doing its absolute best to cram the new technology into
every single system, application and feature in its product ecosystem. ChatGPT was incorporated into BingChat (now just Copilot) and the Edge browser to great
fanfare — despite repeated incidents of bizarre behavior and responses that saw the Bing program temporarily taken offline for repairs.

Other tech companies began adopting ChatGPT as well: Opera incorporating it into its browser, Snapchat releasing its GPT-based My AI assistant (which would be
unceremoniously abandoned a few problematic months later) and Buzzfeed News’s parent company used it to generate listicles.
</item 67>

<item 68>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
The three men and others pledged to put $US1 billion into the project, which Altman, who was 30 at the time, would help run. To get them started, they poached
Ilya Sutskever from Google. (Sutskever was one of the graduate students Google “bought” in Hinton’s auction.) Initially, Musk wanted to operate OpenAI as a
nonprofit, free from the economic incentives that were driving Google and other corporations. But by the time Google wowed the tech community with its Go stunt,
Musk was changing his mind about how it should be run. He desperately wanted OpenAI to invent something that would capture the world’s imagination and close the
gap with Google, but it wasn’t getting the job done as a nonprofit.

In late 2017, Musk hatched a plan to wrest control of the lab from Altman and the other founders and transform it into a commercial operation that would join
forces with Tesla and rely on supercomputers the car company was developing, according to four people familiar with the matter.
</item 68>

<item 69>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“These partnerships are part of a larger effort to empower Microsoft users to better understand the information they consume across our platforms and products,”
it suggests, also citing efforts undertaken in media literacy campaigns and “cyber-skilling” which it says are “not designed to tell individuals what to believe
or how to think; rather, they are about equipping people to think critically and make informed decisions about what information they consume”.

On Bing Search, where Microsoft was quick to embed generative AI features — leading to some embarrassing early reviews which demonstrated the tool producing
dubious content — the report claims it has taken a raft of measures to mitigate risks including applying its AI principles during development and consulting
with experts; engaging in pre-launch testing and a limited preview period and phased release; the use of classifiers and metaprompting, defensive search
interventions, enhanced reporting functionality, and increased operations and incident response; as well as updating Bing’s terms of use to include a Code of
Conduct for users.
</item 69>

<item 70>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Startups, he added, can use speed to their advantage as they work to “innovate, iterate and deploy solutions” to customers.

Will that work prove defensible in the long run? Edward Tsai, a managing partner at Alumni Ventures, told us that he had a potentially “controversial opinion
that VCs and startups may want to temporarily reduce their focus on defensibility and increase their focus on products that deliver compelling value and
focusing on speed to market.” Presuming massive TAM, that could work!

Read on for answers to all our questions from:

Rick Grinnell, founder and managing partner, Glasswing Ventures

There are several layers to the emerging LLM stack, including models, pre-training solutions and fine-tuning tools. Do you expect startups to build striated
solutions for individual layers of the LLM stack, or pursue a more vertical approach?
</item 70>

<item 71>
Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation (TechCrunch, 2023-10-28T13:04:25+00:00)
It’s taken us more than 10 years to reach this scale. During that time, Amazon has hired hundreds of thousands of employees to work in our operations. We take a
purpose-driven approach to how we design and deploy technology at our facilities and we consistently prioritize using robots to support safety and ease everyday
tasks for our employees.

One other bit before we move on to the interview. At the top of the second day, an Amazon rep noted, “Every one of our teams is working on building generative
AI applications.” That jumped out at me, for obvious reasons, but as the event pressed on into specifics around drone and robotics plans, the topic largely fell
away.

I kicked off my conversation with Tye Brady with a few questions on the subject.

The subject of generative AI came up earlier in the day, but it was largely absent from the robotics conversations. How is your team thinking about the subject?

I’ll talk about machine learning and then generative AI.
</item 71>

<item 72>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 72>

<item 73>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Before Rogier launched MasterClass, he worked as a VC, and — through his connections — he received a $500,000 seed round before he even had an idea for a
company.

And on Chain Reaction, Jacquelyn interviewed David Pakman, managing partner and head of venture investments at CoinFund. Before CoinFund, David spent 14 years
at the venture capital firm Venrock. He also led the Series A and B rounds at Dollar Shave Club, which was acquired by Unilever for $1 billion. And, in 1991,
David co-created Apple Music when he was part of Apple’s system software product marketing group.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
</item 73>

<item 74>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
By the end of 2012, Google and Facebook were angling to acquire the London lab, according to three people familiar with the matter. Hassabis and his co-founders
insisted on two conditions: No DeepMind technology could be used for military purposes, and its AGI technology must be overseen by an independent board of
technologists and ethicists.

Google offered $US650 million. Zuckerberg of Facebook offered a bigger payout to DeepMind’s founders but would not agree to the conditions. DeepMind sold to
Google. Zuckerberg was determined to build an AI lab of his own. He hired Yann LeCun, a French computer scientist who had also done pioneering AI research, to
run it. A year after Hinton’s auction, Zuckerberg and LeCun flew to Lake Tahoe for the same AI conference. While padding around a suite at the Harrah’s casino
in his socks, Zuckerberg personally interviewed top researchers, who were soon offered millions of dollars in salary and stock. AI was once laughed off.
</item 74>

<item 75>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Hey, folks, welcome to Week in Review (WiR), TechCrunch’s regular newsletter that recaps the past few days in tech. AI stole the headlines once again, with tech
giants from Google to X (formerly Twitter) heading off against OpenAI for chatbot supremacy. But plenty happened besides.

In this edition of WiR, we cover Google faking a demo of its new AI model (and giving out offensive notebooks to Black summit attendees), defense startup
Anduril unveiling a fighter jet weapon, the continued fallout from the 23andMe hack, and the release of the Grand Theft Auto VI trailer. Also on the roster are
stories about patient scans and health records spilling online, Meta’s new AI-powered image generator, Spotify cutting jobs and an autonomous truck startup
leaving the U.S.

It’s a lot to get to, so we won’t delay. But first, a reminder to sign up here to receive WiR in your inbox every Saturday if you haven’t already done so.
</item 75>

<item 76>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
The goal is that AI tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

“It’s really interesting from a music perspective, specifically, because . . . musicians are perhaps more familiar with the idea of AI,” Holland said.
“Musicians in general are more familiar with things like music production software, and AI tools like MIDI drum loops . . . so I think that there is a certain
amount of more progressive learning from them, when it comes to technology, and its ability to make their music better.”

When it comes to art and AI, the conversation is complicated, to say the least. Musicians are nervous about industry giants copyrighting AI music and cutting
them out of the process. Major record labels are worried about AI models training on their catalogues and stealing a slice of their considerable pie.
</item 76>

<item 77>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
LeCun et al. disagree.

“Yes, openly available models come with risks and vulnerabilities — AI models can be abused by malicious actors or deployed by ill-equipped developers,” the
letter acknowledged. “However, we have seen time and time again that the same holds true for proprietary technologies — and that increasing public access and
scrutiny makes technology safer, not more dangerous. The idea that tight and proprietary control of foundational AI models is the only path to protecting us
from society-scale harm is naive at best, dangerous at worst.”

On a personal level, LeCun has accused the big-name AI players of trying to secure “regulatory capture of the AI industry” by lobbying against open AI R&D. And
on a company level, Meta is doing all it can to encourage collaboration and “openness,” recently partnering with Hugging Face to launch a new startup
accelerator designed to spur adoption of open source AI models.

But OpenAI was — up until last week, at least — still the AI darling everyone wanted to dance with.
</item 77>

<item 78>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Marketing claims made for machine learning systems that companies adopted in order to not fall behind will receive their quarterly and yearly reviews… and it’s
very likely they will be found wanting.

Expect a considerable customer withdrawal from AI tools as the benefits fail to justify the costs and risks. On the far end of this spectrum, we are likely to
see lawsuits and regulatory action with AI service providers that failed to back up their claims.

While capabilities will continue to grow and advance, 2023’s products will not all survive by a long shot, and there will be a round of consolidation as the
wobblier riders of the wave fall and are consumed.

Apple jumps in

Apple has an established pattern of waiting, watching and learning from other companies’ failures, then blowing in with a refined and polished take that puts
others to shame. The timing is right for Apple to do this in AI, not just because if it waits too long its competition may eat up the market, but because the
tech is ripe for their kind of improvement.
</item 78>

<item 79>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
Musk agreed. But it was pretty clear that his Google guests were not prepared to embark on a redistribution of (their) wealth. Schmidt said he thought the
worries were completely overblown. In his usual whisper, Page agreed. AI would create more jobs than it took away, he argued. Eight months later, DeepMind had a
breakthrough that stunned the AI community and the world. A DeepMind machine called AlphaGo beat one of the world’s best players at the ancient game of Go. The
game, streamed over the internet, was watched by 200 million people across the globe. Most researchers had assumed that AI needed another 10 years to muster the
ingenuity to do that. Demis Hassabis, right, co-founder of DeepMind, with South Korean professional Lee Se-dol. Lee played Go against the DeepMind machine
AlphaGo in March 2016. Credit: EPA

Rationalists, effective altruists and others who worried about the risks of AI claimed the computer’s win validated their fears.
</item 79>

<item 80>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
The idea, O’Hara said, is to rethink social networking from the ground up.

Ford shutters SaaS app for field work: Ford has shut down VIIZR, a software-as-a-service company that, along with Salesforce, built an app to help tradespeople
like plumbers, locksmiths and electricians to schedule field appointments, send invoices and manage customers, Kirsten reports. VIIZR, which was announced in
December 2021, was a separate company majority owned by Ford, with Salesforce as a minority investor.

Apple bets on generative AI: Apple CEO Tim Cook pushed back against the notion that the company was behind in AI on Apple’s Q4 earnings call with investors, as
he highlighted technology developments that Apple had made recently that “would not be possible without AI.” Cook also said that Apple was working on generative
AI technologies, lending credence to reports suggesting the company is on track to spend $1 billion per year on developing generative AI products.
</item 80>

<item 81>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 81>

<item 82>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
To help address misinformation, Google has also announced that it will soon be integrating new innovations in watermarking, metadata, and other techniques into
its latest generative models.

“Google also recently joined other leading AI companies to jointly commit to advancing responsible practices in the development of artificial intelligence which
will support efforts by the G7, the OECD, and national governments. Going forward we will continue to report and expand upon Google developed AI tools and are
committed to advance bold and responsible AI, to maximise AI’s benefits and minimise its risks.”

Over the next six months Google’s report states it has no additional measures planned for YouTube. But, with generative image capabilities rolling out
internally over the next year, it commits Google Search to leveraging IPTC Photo Metadata Standard to add metadata tags to images that are generated by Google
AI.

“Creators and publishers will be able to add a similar markup to their own images, so a label can be displayed in Search to indicate the images as AI
generated,” Google’s report further notes.
</item 82>

<item 83>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
In our proprietary view of the GenAI tech stack, we categorize the landscape into four distinct layers: foundation model providers, middle-tier companies,
end-market or top-layer applications, and full stack or end-to-end vertical companies.

We think that most of the opportunity lies in the application layer, and within that layer, we believe that in the near future, the best applications will
harness their in-house expertise to build specialized middle-layer tooling and blend them with the appropriate foundational models. These are “vertically
integrated” or “full-stack” applications. For startups, this approach means a shorter time-to-market. Without negotiating or integrating with external entities,
startups can innovate, iterate and deploy solutions at an accelerated pace. This speed and agility can often be the differentiating factor in capturing market
share or meeting a critical market need before competitors.

On the other hand, we view the middle layer as a conduit, connecting the foundational aspects of AI with the refined specialized application layer.
</item 83>

<item 84>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
2. Do they disagree on long-term strategy?

Despite being the hottest tech company in the world right now and everyone talking about ChatGPT, OpenAI isn’t exactly a sound business. It’s shoveling money
into the furnace as fast as it can by serving, by all accounts, a fantastically expensive product at bargain-bin prices.

That’s all well and good for a year or two, but at some point that strategy changes from a growth hack to an existential liability. Could Altman and the board
have had irreconcilable differences on where that point lies?

This doesn’t seem so likely. The company has been very deliberately pursuing this very publicly, confidently and on a long-term basis. Altman and the board seem
to be in sync on this, at least for the present.

3. Do the numbers not add up?

On the other hand, could OpenAI be losing even more money than Altman admitted or projected? It seems impossible, but the costs of running this operation have
no precedent, nor really does the operation itself.
</item 84>

<item 85>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.
</item 85>

<item 86>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Most read

AI, faked: Google unveiled a new flagship AI model this week called Gemini. But it didn’t release the full model, Gemini Ultra — only a “lite” version called
Gemini Pro. In a press briefing and blog posts, Google touted Gemini’s coding capabilities and multimodal prowess, claiming that the model can understand
images, audio and videos just as well as text. But Gemini Pro — which is strictly text-in, text-out — has proven to be mistake-prone. And in a worse look for
Google, the company was caught faking a Gemini demo by tuning text prompts with still images off camera.

Offensive notebooks: In another Google PR blunder, people who attended the company’s K&I Black Summit in August were given third-party notebooks containing
highly insensitive language. My colleague Dominic-Madori writes that the inside of the notebooks were printed with the phrase “I was just cotton the moment, but
I came back to take your notes” (emphasis ours).
</item 86>

<item 87>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The EU’s voluntary anti-disinformation Code has 44 signatories in all — covering not just major social media and search platforms such as the aforementioned
giants but entities from across the ad industry and civil society organizations involved in fact-checking.

Google

On generative AI, Google’s report discusses “recent progress in large-scale AI models” which it suggests has “sparked additional discussion about the social
impacts of AI and raised concerns on topics such as misinformation”. The tech giant is an early adopter of generative AI in search — via its Bard chatbot.

“Google is committed to developing technology responsibly and has published AI Principles to guide our work, including application areas we will not pursue,” it
writes in summary on the topic, adding: “We have also established a governance team to put them into action by conducting ethical reviews of new systems,
avoiding bias and incorporating privacy, security and safety.

“Google Search has published guidance on AI-generated content, outlining its approach to maintaining a high standard of information quality and the overall
helpfulness of content on Search.
</item 87>

<item 88>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
The Boston Dynamics AI Institute is still in its infancy, and therefore doesn’t have much in the way of public facing research to show just yet. It is, however,
doing a lot of hiring. This week it announced that MIT’s Kate Darling will lead research around “ethics and societal impact of robotics and AI.” From the
institute:

Darling’s team will explore immediate as well as long term questions on the implementation and use of robotics, impact on the workplace, infrastructure and
other topics. The team will perform studies and experiments designed to generate data needed for others to make informed ethics and policy decisions, and will
also develop a series of talks and workshops at the intersection of ethics, law, economics and robotics that will offer a platform for broad discussion.

“I’ve heard Gill say that making a car is taking small bits of metal and banging them up, welding them together and attaching them,” Raibert says. “That’s a
little on the primitive side. Now we have software, robotics, AI and all that stuff, and the car companies need to embrace that.
</item 88>

<item 89>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
But eight years later, the argument between the two men seems prescient. The question of whether AI will elevate the world or destroy it – or at least inflict
grave damage – has framed an ongoing debate among Silicon Valley founders, chatbot users, academics, legislators and regulators about whether the technology
should be controlled or set free. Loading That debate has pitted some of the world’s richest men against one another: Musk, Page, Mark Zuckerberg of Meta, tech
investor Peter Thiel, Satya Nadella of Microsoft and Sam Altman of OpenAI. All have fought for a piece of the business – which one day could be worth trillions
of dollars – and the power to shape it. At the heart of this competition is a brain-stretching paradox. The people who say they are most worried about AI are
among the most determined to create it and enjoy its riches. They have justified their ambition with their strong belief that they alone can keep AI from
endangering Earth. Musk and Page stopped speaking soon after the party that summer.
</item 89>

<item 90>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
His strategy was to arrange another meeting. There was a deep tension between the bishop and the knight, he told Thiel. The two pieces carried the same value,
but the best players understood that their strengths were vastly different. It worked. Charmed, Thiel invited the group back the next day, where they gathered
in the kitchen. Their host had just finished his morning workout and was still sweating in a shiny tracksuit. A butler handed him a Diet Coke. The three made
their pitch, and soon Thiel and his venture capital firm agreed to put £1.4 million ($A2.7 million) into their startup. He was their first major investor.

They named their company DeepMind, a nod to “deep learning,” a way for AI systems to learn skills by analysing large amounts of data; to neuroscience; and to
the Deep Thought supercomputer from the sci-fi novel The Hitchhiker’s Guide to the Galaxy. By the spring of 2010, they were building their dream machine.
</item 90>

<item 91>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
I would expect an AI that focuses on practical applications of users’ own data, using Apple’s increasingly central position in their lives to integrate the many
signals and ecosystems the company is privy to. There will likely also be a clever and elegant way to handle problematic or dangerous prompts, and although it
will almost certainly have multimodal understanding (primarily to handle user images), I imagine they’ll totally skip media generation. Expect some narrowly
tailored but impressive agent capabilities as well: “Siri, get a table for 4 at a sushi place downtown around 7 and book a car to take us” sort of thing.

What’s hard to say is whether they will bill it as an improved Siri or as a whole new service, Apple AI, with a name you can choose yourself. They may feel the
old brand is freighted with years of being comparatively incapable, but millions already say “hey Siri” every 10 seconds so it’s more likely they’ll opt to keep
that momentum.
</item 91>

<item 92>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT: Everything you need to know about the AI-powered chatbot

ChatGPT, OpenAI’s text-generating AI chatbot, has taken the world by storm. What started as a tool to hyper-charge productivity through writing essays and code
with short text prompts has evolved into a behemoth used by more than 92% of Fortune 500 companies for more wide-ranging needs. And that growth has propelled
OpenAI itself into becoming one of the most-hyped companies in recent memory, even if CEO and co-founder Sam Altman’s firing and swift return raised concerns
about its direction and opened the door for competitors.

What does that mean for OpenAI, ChatGPT and its other ambitions? The fallout is still settling, but it might empower competitors like Meta and its LLaMA family
of large language models, or help other AI startups get attention and funding as the industry watches OpenAI implode and put itself back together.
</item 92>

<item 93>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Released March 14, GPT-4 is available for paying ChatGPT Plus users and through a public API. Developers can sign up on a waitlist to access the API.

ChatGPT is available in Azure OpenAI service

ChatGPT is generally available through the Azure OpenAI Service, Microsoft’s fully managed, corporate-focused offering. Customers, who must already be
“Microsoft managed customers and partners,” can apply here for special access.

OpenAI launches an API for ChatGPT

OpenAI makes another move toward monetization by launching a paid API for ChatGPT. Instacart, Snap (Snapchat’s parent company) and Quizlet are among its initial
customers.

February 2023

Microsoft launches the new Bing, with ChatGPT built in

At a press event in Redmond, Washington, Microsoft announced its long-rumored integration of OpenAI’s GPT-4 model into Bing, providing a ChatGPT-like experience
within the search engine.
</item 93>

<item 94>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
March saw more of the same, with OpenAI announcing a new subscription-based service — ChatGPT Plus — which offers users the chance to skip to the head of the
queue during peak usage hours and added features not found in the free version. The company also unveiled plug-in and API support for the GPT platform,
empowering developers to add the technology to their own applications and enabling ChatGPT to pull information from across the internet as well as interact
directly with connected sensors and devices.

ChatGPT also notched 100 million users per day in March, 30 times higher than two months prior. Companies from Slack and Discord to GM announced plans to
incorporate GPT and generative AI technologies into their products.

Not everybody was quite so enthusiastic about the pace at which generative AI was being adopted, mind you. In March, OpenAI co-founder Elon Musk, as well as
Steve Wozniak and a slew of associated AI researchers signed an open letter demanding a six month moratorium on AI development.
</item 94>

<item 95>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Google bought DeepMind, it bought the whole thing. Musk was out. Financially, he had come out ahead, but he was unhappy. Three Google executives now firmly
in control of DeepMind were there: Page; Sergey Brin, a Google co-founder and Tesla investor; and Eric Schmidt, Google’s chair. Among the other attendees were
Reid Hoffman, another PayPal founder; and Toby Ord, an Australian philosopher studying “existential risk”.

The DeepMind founders reported that they were pushing ahead with their work but that they were aware the technology carried serious risks. Suleyman, the
DeepMind co-founder, gave a presentation called “The Pitchforkers Are Coming.” AI could lead to an explosion in disinformation, he told the board. He fretted
that as the technology replaced countless jobs in the coming years, the public would accuse Google of stealing their livelihoods. Google would need to share its
wealth with the millions who could no longer find work and provide a “universal basic income,” he argued. Musk agreed.
</item 95>

<item 96>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
That’s effectively making robots that are smarter, better at mobile manipulation and move more dynamically. Raibert admits that “Boston Dynamics AI Institute”
might not be the most instructive name, both with regard to confusion between itself and Boston Dynamics the company, as well as a seeming focus on AI over
robotics. A name change is apparently coming, but nothing has been decided just yet.
</item 96>

<item 97>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 97>

<item 98>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
I think the leadership at Hyundai — the chairman visited me last week, and we’re in close touch with him and the people who work with him — think that getting
into the 22nd century is an important thing to do and we’re getting started on it now.”

Pratt adds, “There’s actually this alignment that’s happening between cars and robotics, as well. Rod Brooks was the one who a few years ago said that modern
cars are eldercare robots. I think that’s really true. You think about the amount of computers in them, the amount of software that’s in them. TRI has different
divisions inside of it, one of which works on very advanced things in cars that is completely overlapping with some of the software and some of the concepts
that we’re using in the robotics space.”

For its part, the Boston Dynamics AI Institute lists three key pillars of its research: intelligence, dexterity and mobility. That’s effectively making robots
that are smarter, better at mobile manipulation and move more dynamically.
</item 98>

<item 99>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Altman and others pushed back, Musk quit and said he would focus on his own AI work at Tesla. In February 2018, he announced his departure to OpenAI’s
staff on the top floor of the startup’s offices in a converted truck factory, three people who attended the meeting said. When he said that OpenAI needed to
move faster, one researcher retorted at the meeting that Musk was being reckless. Musk called the researcher a “jackass” and stormed out, taking his deep
pockets with him. OpenAI suddenly needed new financing in a hurry. Altman flew to Sun Valley for a conference and ran into Satya Nadella, Microsoft’s CEO. A
tie-up seemed natural. Altman knew Microsoft’s chief technology officer, Kevin Scott. Microsoft had bought LinkedIn from Hoffman, an OpenAI board member.
Nadella told Scott to get it done. The deal closed in 2019.
</item 99>

<item 100>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
We need robust tools and platforms to enable broader utilization among businesses and individuals. Startups have the opportunity here to act quickly, find novel
solutions to emerging problems, and define new categories.

Interestingly, even large tech companies recognize the gaps in their services and have begun investing heavily in startups alongside VCs. These companies apply
AI to their internal processes and thus see the value startups bring to LLM deployment and integration. Consider the recent investments from Microsoft, Nvidia,
and Salesforce into companies like Inflection AI and Cohere.

What can be done to ensure industry-specific startups that tune generative AI models for a specific niche will prove defensible?

To ensure industry-specific startups will prove defensible in the rising climate of AI integration, startups must prioritize collecting proprietary data,
integrating a sophisticated application layer and assuring output accuracy.

We have established a framework to assess the defensibility of application layers of AI companies. First, the application must address a real enterprise pain
point prioritized by executives.
</item 100>
=== END ITEM id=3b74a42a8678dd45 ===
