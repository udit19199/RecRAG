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

=== ITEM id=5d6ed2c972ddcaaa ===
Gold sentences and facts:
Sentence 0: The guidance lays out ways to responsibly use AI in the courts, including being aware of potential bias and upholding privacy.
  Fact 0.0: The guidance lays out ways to use AI responsibly in the courts.
  Fact 0.1: The guidance says courts should be aware of potential bias when using AI.
  Fact 0.2: The guidance says courts should uphold privacy when using AI.
Sentence 1: They promised “invest[ment] in the AI-powered technology” that, amongst other things, would help YouTube “protect our community of viewers, creators, artists and songwriters… [from] trademark and copyright abuse” .
  Fact 1.0: The party referred to as 'they' promised investment in the AI-powered technology.
  Fact 1.1: The AI-powered technology would, among other things, help YouTube protect its community of viewers, creators, artists and songwriters.
  Fact 1.2: The AI-powered technology would help YouTube protect its community from trademark and copyright abuse.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
The list of strictly prohibited use-cases of AI will apply sooner, just six months after the law enters into force — which could, potentially, mean bans on
certain “unacceptable risk” uses of AI, such as social scoring or Clearview AI-style selfie scraping for facial recognition databases, will get up and running
in the second half of 2024, assuming no last minute opposition to the regulation springs up within the Council or Parliament. (For the full list of banned AI
uses, read our earlier post.)
</item 2>

<item 3>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
As creative industries grapple with AI’s explosion into every artistic medium at once, separate calls from artists warning the world to take action before it’s
too late are starting to converge. From fake Drake songs to stylized Instagram profile pictures, art conjured with newly sophisticated AI tools is suddenly
ubiquitous — and so are conversations about how to rein in the technology before it does irrevocable harm to creative communities.

This week, digital rights organization Fight for the Future partnered with music industry labor group United Musicians and Allied Workers to launch
#AIdayofaction, a campaign that calls on Congress to block corporations from obtaining copyrights on music and other art made with AI.

The idea is that by preventing industry behemoths like major record labels, for example, from copyrighting music made with the assistance of AI, those companies
will be forced to keep looping humans into the creative process. But those same concerns — and the same potential strategies for pushing back against the
onslaught of AI — exist across creative industries.
</item 3>

<item 4>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
On iPad, ChatGPT now runs in full-screen mode, optimized for the tablet’s interface.

May 2023

Texas judge orders all AI-generated content must be declared and checked

The Texas federal judge has added a requirement that any attorney appearing in his court must attest that “no portion of the filing was drafted by generative
artificial intelligence,” or if it was, that it was checked “by a human being.”

ChatGPT app expanded to more than 30 countries

The list of new countries includes Algeria, Argentina, Azerbaijan, Bolivia, Brazil, Canada, Chile, Costa Rica, Ecuador, Estonia, Ghana, India, Iraq, Israel,
Japan, Jordan, Kazakhstan, Kuwait, Lebanon, Lithuania, Mauritania, Mauritius, Mexico, Morocco, Namibia, Nauru, Oman, Pakistan, Peru, Poland, Qatar, Slovenia,
Tunisia and the United Arab Emirates.
</item 4>

<item 5>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Legal cases build and break

We saw a fair number of lawsuits filed in 2023, but few saw any real movement, let alone success. Most suits over copyright and other missteps in the AI
industry are still pending. 2024 will see a lot of them fall by the wayside, as companies stonewall critical information like training data and methods, making
allegations like the use of thousands of copyrighted books difficult to prove in court.

This was only the beginning, however, and many of these lawsuits were filed essentially on principle. Though they may not succeed, they may crack the process
open far enough during testimony and discovery that companies would rather settle than have certain information come to light. 2024 will bring new lawsuits as
well, ones pertaining to misuse and abuse of AI, such as wrongful termination, bias in hiring and lending, and other areas where AI is being put to work without
a lot of thought.
</item 5>

<item 6>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
GPAIs that fall in the AI Act’s high risk tier will face ex ante-style regulatory requirements to assess and mitigate systemic risks — meaning they must
proactively test model outputs to shrink risks of actual (or “reasonably foreseeable”) negative effects on public health, safety, public security, fundamental
rights, or for society as a whole.

While “low tier” GPAIs will only face lighter transparency requirements, including obligations to apply watermarking to generative AI outputs.

The watermarking requirement for GPAIs falls in an article that was in the original Commission version of the risk-based framework, presented all the way back
in April 2021, which focused on transparency requirements for technologies such as AI chatbots and deepfakes — but which will now also apply generally to
general purpose AI systems.

“There is an obligation to try to watermark [generative AI-produced] text on the basis of the latest state of the art technology that is available,” the
Commission official said, fleshing out details of the agreed watermarking obligations.
</item 6>

<item 7>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Even if generative-Al-assisted outputs do compete with existing works that were used in training, or with future works by the authors of those works, the
pro-competitive nature of copying for the purpose of “knowledge harvesting” has traditionally been a reason to favor a holding of fair use , not a reason to
reject it.”

the output of that model might … Even if generative-Al-assisted outputs do compete with existing works that were used in training, or with future works by the
authors of those works, the pro-competitive nature of copying for the purpose of “knowledge harvesting” has traditionally been a reason to , not a reason to
reject it.” And then the kicker: “Any prohibition or limitation on the use of copyrighted materials for purposes of Al training would therefore undermine the
purpose of copyright and foreclose the many opportunities that come with this technology.”
</item 7>

<item 8>
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
</item 8>

<item 9>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
Robert Kyncl, Warner Music Group

In Kyncl’s view, the solutions put in place to address UGC form a “blueprint” than can be used to address AI-generated content on platforms as well.

“When YouTube was formed… people started to upload content, including copyrighted material, which, obviously, put YouTube into hot water with lots of different
copyright holders. And I had the privilege of working through a lot of that and fix it up.

“But we made we made a very important decision, which was to go above and beyond the law, and build a fingerprinting software that allowed us to track the
copyright on our platform, and then have commercial relationship[s] with copyright holders to send them the money. Out of that we built a multi-billion-dollar
business, which now is a multi-billion-dollar business per year. And it was an incredible new revenue stream for everyone. AI is that with new super tools.”
</item 9>

<item 10>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
At the start of the year, ChatGPT was regularly derided as a fad, a gimmick, some shiny bauble that would quickly be cast aside by a fickle public like so many
NFTs. Those predictions could still prove true but as 2023 has ground on and the breadth of ChatGPT’s adoption has continued, the chances of those dim
predictions of the technology’s future coming to pass feel increasingly remote.

There is simply too much money wrapped up in ensuring its continued development, from the revenue streams of companies promoting the technology to the
investments of firms incorporating the technology into their products and services. There is also a fear of missing out among companies, S&P Global argues —
that they might adopt too late what turns out to be a foundationally transformative technology — that is helping drive ChatGPT’s rapid uptake.

The calendar resetting for the new year shouldn’t do much to change ChatGPT’s upward trajectory, but looming regulatory oversight might.
</item 10>

<item 11>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Senior political figures in Washington are becoming increasingly interested in the impact of AI on the law.

On Wednesday (November 29), US Senate Majority Leader Chuck Schumer held an Artificial Intelligence (AI) Insight Forum to discuss the importance and role of
creative copyright and intellectual property in the development of AI.

Dennis Kooker, Sony Music Entertainment‘s President of Global Digital Business & US Sales, delivered a speech at the Forum, outlining the major music company’s
position on AI and copyright law.

Kooker’s public statements on the topic arrive a week after MBW pointed out that there was a notable lack of endorsement from SME, and its artists, of YouTube‘s
new experimental AI project called ‘Dream Track’ that lets creators clone the vocals of successful musicians.

The initial cohort of stars involved in the project, as we noted last week, include recording artists signed to Warner Music Group and Universal Music Group –
or labels within each of those two majors, but no Sony Music artists.
</item 11>

<item 12>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Who owns the copyright on ChatGPT-created content or media?

The user who requested the input from ChatGPT is the copyright owner.

What lawsuits are there surrounding ChatGPT?

None specifically targeting ChatGPT. But OpenAI is involved in at least one lawsuit that has implications for AI systems trained on publicly available data,
which would touch on ChatGPT.

Are there issues regarding plagiarism with ChatGPT?

Yes. Text-generating AI models like ChatGPT have a tendency to regurgitate content from their training data.
</item 12>

<item 13>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
As reported by MBW last week, Universal Music Group submitted its response to the USCO. Unsurprisingly, amongst the highlights of UMG’s submission was the
rejection of the notion that using copyrighted material to train LLMs is “fair use”.

In his speech on Wednesday, Kooker commented directly on the flurry of submissions to the USCO from the tech industry at the end of October.

In what may have been a nod to Google’s filing and “fair use” position, Kooker told Senators that “based on recent Copyright Office filings it is clear that the
technology industry and speculative financial investors would like governments to believe in a very distorted view of copyright”.

He added: “[That view is] one in which music is considered fair use for training purposes and in which certain companies are permitted to appropriate the entire
value produced by the creative sector without permission, and to build huge businesses based on it without paying anything to the creators concerned.”
</item 13>

<item 14>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Confirm That Copying Music to Train AI Models is Not Fair Use. “Even worse are those that argue that copyrighted content should automatically be considered fair
use so that protected works are never compensated for usage and creators have no say in the products or business models that are developed around them and their
work. Congress should assure and agencies should presume that reproducing music to train AI models, in itself, is not a fair use.” Prevent the Cloning of
Artists’ Voices and Likenesses Without Express Permission. “We cannot allow an artist’s voice or likeness to be cloned for use without the express permission of
the artist. This is a very personal decision for the artist. Congress should pass into law effective federal protections for name, image, and likeness.”
Incentivize Accurate Recordkeeping. “Correct attribution will be a critical element to artists being paid fairly and correctly for new works that are created.
</item 14>

<item 15>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
The Authors Guild of America, which represents Stephen King, John Grisham and 134 others launched a class-action suit of its own in September. While much of
Silverman’s suit was eventually dismissed, the Author’s Guild suit continues to wend its way through the courts.

Select news outlets, on the other hand, proved far more amenable. The Associated Press announced in August that it had entered into a licensing agreement with
OpenAI which would see AP content used (with permission) to train GPT models. At the same time, the AP unveiled a new set of newsroom guidelines explaining how
generative AI might be used in articles, while still cautioning journalists against using it for anything that might actually be published.

ChatGPT itself didn’t seem too inclined to follow the rules. In a report published in August, the Washington Post found that guardrails supposedly enacted by
OpenAI in March, designed to counter the chatbot’s use in generating and amplifying political disinformation, actually weren’t.
</item 15>

<item 16>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
“At the moment, technologies are much better at watermarking videos and audio than watermarking text. But what we ask is the fact that this watermarking takes
place on the basis of state of the art technology — and then we expect, of course, that over time the technology will mature and will be as [good] as possible.”

GPAI model makers must also commit to respecting EU copyright rules, including complying with an existing machine readable opt-out from text and data mining
contained in the EU Copyright Directive — and a carve-out of the Act’s transparency requirements for open source GPAIs does not extend to cutting them loose
from the copyright obligations, with the Commission confirming the Copyright Directive will still apply on open source GPAIs.

As regards the AI Office, which will play a key role in setting risk classification thresholds for GPAIs, the Commission confirmed there’s no budget nor
headcount defined for the expert body as yet.
</item 16>

<item 17>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
And that’s all some people need.

Likely there will be some half-hearted efforts to block generated content from being used in this way, but these posts can’t be taken down fast enough by the
likes of Meta and Google, and the idea that X can (or will) effectively monitor and take down such content is implausible. It’s gonna be a bad time!
</item 17>

<item 18>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
ChatGPT can also show bias in its responses, answering in sexist and racist, overtly Anglocentric ways — or regurgitating portions of the data that it was
trained on.

Lawyers have been sanctioned after using ChatGPT to assist in writing motions, discovering — too late — that ChatGPT invented fake lawsuit citations. And scores
of authors have sued OpenAI over the chatbot regurgitating portions of their work — and not receiving compensation for it.

So what comes next? What might ChatGPT’s second year hold, if not more of the same?

Interestingly — and fortunately — some of the more dire predictions about ChatGPT didn’t come to pass. Some researchers feared the chatbot would be used to
generate disinformation on a massive scale, while others sounded the alarm over ChatGPT’s phishing email-, spam- and malware-generating potential.
</item 18>

<item 19>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
He added that “platforms are quick to point to the loopholes in the law as an excuse to drag their feet or to not take the deepfakes down when requested.”

Kooker commended the No FAKES Act policy drafted by senators earlier this year, which he explained “would create a federal property right in one’s voice or
visual likeness and protect against unauthorized AI-generated replicas”.

2. Sony has ‘roughly 200 active conversations taking place’ with AI startups that “include potential equity investments”

Kooker explained that despite the negative impacts of AI, including unauthorized clones and low-quality AI music, there are still “many positive and optimistic
developments to highlight”.

He highlighted legitimate AI startups and established firms “with music ideas that want to partner with the industry”.

According to Kooker, Sony Music has “roughly 200 active conversations taking place with startups and established players” right now about building new
AI-related products and tools.
</item 19>

<item 20>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.
</item 20>

<item 21>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
In the meantime, music emerged out of [digitization] better and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll
likely be first.”

Asked when that sea change will occur, Kyncl said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is
increasing quality at a very fast pace.”

Kyncl drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.

“I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”
</item 21>

<item 22>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
A person can also specify their family size, so the text-generating AI can give responses about meals, grocery and vacation planning accordingly.

The FTC is reportedly investigating OpenAI

The FTC is reportedly in at least the exploratory phase of investigation over whether OpenAI’s flagship ChatGPT conversational AI made “false, misleading,
disparaging or harmful” statements about people.

TechCrunch Reporter Devin Coldewey reports:

This kind of investigation doesn’t just appear out of thin air — the FTC doesn’t look around and say “That looks suspicious.” Generally a lawsuit or formal
complaint is brought to their attention and the practices described by it imply that regulations are being ignored. For example, a person may sue a supplement
company because the pills made them sick, and the FTC will launch an investigation on the back of that because there’s evidence the company lied about the side
effects.

OpenAI announced the general availability of GPT-4

Starting July 6, all existing OpenAI developers “with a history of successful payments” can access GPT-4.
</item 22>

<item 23>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
“About the time that we started the discussion with this artist, we had begun informal talks with a generative AI start-up company whose business model focused
on working with IP rightsholders the ‘right way’.

“In other words, they respect intellectual property rights and want to work with rights holders in ways that enhance and protect the copyrighted works.”

3. SME has asked Congress to embrace a set of principles around generative AI

Kooker told the lawmakers during his speech on Wednesday that “if copyrights are protected and enforced appropriately, we are at the beginning of a multi-decade
marathon that will change the creative and commercial landscape for music”.

With that in mind, Kooker explained that Sony Music has established a set of principles to guide the company’s decision-making around generative AI.

“If copyrights are protected and enforced appropriately, we are at the beginning of a multi-decade marathon that will change the creative and commercial
landscape for music.”
</item 23>

<item 24>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists;

and jointly announced that the video platform was launching an – a program by which new tools and innovations will be developed at YouTube in close conjunction
with artists and the music biz. To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists; Also in
August, YouTube publicly committed to three principles/pledges behind its development of music-based generative AI tools . These pledges promised “appropriate
protections… for music partners who decide to participate” in said tools. They promised “invest[ment] in the AI-powered technology” that, amongst other things,
would help YouTube “protect our community of viewers, creators, artists and songwriters… [from] trademark and copyright abuse” .
</item 24>

<item 25>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
“ChatGPT Business will follow our API’s data usage policies, which means that end users’ data won’t be used to train our models by default,” OpenAI wrote in a
blog post. “We plan to make ChatGPT Business available in the coming months.”

OpenAI wants to trademark “GPT”

OpenAI applied for a trademark for “GPT,” which stands for “Generative Pre-trained Transformer,” last December. Last month, the company petitioned the USPTO to
speed up the process, citing the “myriad infringements and counterfeit apps” beginning to spring into existence.

Unfortunately for OpenAI, its petition was dismissed last week. According to the agency, OpenAI’s attorneys neglected to pay an associated fee as well as
provide “appropriate documentary evidence supporting the justification of special action.”

That means a decision could take up to five more months.
</item 25>

<item 26>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
But WhatsApp, other encrypted messaging services, and digital rights groups say this scanning would require breaking apps’ encryption systems and invading user
privacy. Whitehead says that Ofcom plans to consult on this next year, leaving its full impact on encrypted messaging uncertain.

“We’re not regulating the technology, we’re regulating the context.”

There’s another technology not emphasized in today’s consultation: artificial intelligence. But that doesn’t mean AI-generated content won’t fall under the
rules. The Online Safety Act attempts to address online harms in a “technology neutral” way, Whitehead says, regardless of how they’ve been created. So
AI-generated CSAM would be in scope by virtue of it being CSAM, and a deepfake used to conduct fraud would be in scope by virtue of the fraud. “We’re not
regulating the technology, we’re regulating the context,” Whitehead says.
</item 26>

<item 27>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
There was also agreement on a “two-tier” system of guardrails to be applied to “general” AI systems, such as the so-called foundational models underpinning the
viral boom in generative AI applications like ChatGPT.

As we reported earlier, the deal reached on foundational models/general purpose AIs (GPAIs) includes some transparency requirements for what co-legislators
referred to as “low tier” AIs — meaning model makers must draw up technical documentation and produce (and publish) detailed summaries about the content used
for training in order to support compliance with EU copyright law. For “high-impact” GPAIs (defined as the cumulative amount of compute used for their training
measured in floating point operations is greater than 10^25) with so-called “systemic risk” there are more stringent obligations.

“If these models meet certain criteria they will have to conduct model evaluations, assess and mitigate systemic risks, conduct adversarial testing, report to
the Commission on serious incidents, ensure cybersecurity and report on their energy efficiency,” the parliament wrote.
</item 27>

<item 28>
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
</item 28>

<item 29>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
In a roundtable hosted by the FTC this week, the agency brought together figures from across creative industries — from voice acting and science fiction to
screenwriting, music, illustration and even fashion — to delve into how generative AI is affecting creatives.

“I know that generative AI in particular poses a unique set of opportunities and challenges to creative industries,” FTC chair Lina Khan said. “We’ve already
heard significant concerns about how these technologies could virtually overnight significantly disempower creators and artists who may watch their life’s
creation be appropriated into models over which they have no control.”

In the comments, representatives from myriad creative communities expressed concerns around opt-out requirements that by default train AI models on artists’
original work, and how existing copyright law could be a useful if not comprehensive tool for setting out regulatory guardrails.

In the conversation, a representative with the WGA emphasized that while striking writers obtained their own protections in a newly won agreement, the fight for
artists’ livelihoods “doesn’t stop at the bargaining table.”
</item 29>

<item 30>
Sweeping White House AI executive order takes aim at the technology's toughest challenges (Engadget, 2023-10-30T09:00:08+00:00)
It remains to be seen whether those systems will be able to address the concerns of misbehaving models that SEC head Gary Gensler recently raised.

AI watermarking and cryptographic validation

We're already seeing the normalization of deepfake trickery and AI-empowered disinformation on the campaign trail. So, the White House is taking steps to ensure
that the public can trust the text, audio and video content that it publishes on its official channels. The public must be able to easily validate whether the
content they see is AI-generated or not, argued White House officials on the press call.

Adobe

The Department of Commerce is in charge of the latter effort and is expected to work closely with existing industry advocacy groups like the C2PA and its sister
organization, the CAI, to develop and implement a watermarking system for federal agencies. “We aim to support and facilitate and help standardize that work [by
the C2PA],” administration officials said. “We see ourselves as plugging into that ecosystem.”
</item 30>

<item 31>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
The concerns pushed policymakers in Europe to mandate security assessments for any products using generative AI systems like ChatGPT, and over 20,000
signatories — including Elon Musk and Apple co-founder Steve Wozniak — to sign an open letter calling for the immediate pause of large-scale AI experiments like
ChatGPT.

But examples of ChatGPT abuse in the wild have been few and far between — so far.

With the launch of GPTs, OpenAI’s tool for building custom conversational, action-taking AI systems powered by OpenAI’s models, including the models
underpinning ChatGPT, ChatGPT could become more a gateway to a broader ecosystem of AI-powered chatbots than the end-all-be-all.

With GPTs, a user can train a model on a cookbook collection, for example, so that it can answer questions about ingredients for a specific recipe. Or they can
give a model their company’s proprietary codebases so that developers can check their style or generate code in line with best practices.
</item 31>

<item 32>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“I therefore urge platforms to be vigilant and provide efficient safeguards for this in the context of elections.”

The EU commissioner noted she’s meeting representatives of ChatGPT maker, OpenAI, later today to discuss the issue.

The AI giant is not a signatory to the bloc’s anti-disinformation Code — as yet — so is likely to be facing pressure to get on board with the effort. (We’ve
reached out to OpenAI with questions about its meeting with the Jourova.)

The commissioner’s remarks today on generative AI follow initial pressure applied to platforms this summer, when she urged signatories to label deepfakes and
other AI-generated content — calling on Code signatories to create a dedicated and separate track to tackle “AI production”, and quipping that machines should
not have free speech.

An incoming pan-EU AI regulation (aka, the EU AI Act) is expected to make user disclosures a legal requirement on makers of generative AI technologies like AI
chatbots. Although the still draft legislation remains the subject of negotiations by EU co-legislators.
</item 32>

<item 33>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
Commitment 15 in TikTok’s report signs the platform up to “tak[ing] into consideration transparency obligations and the list of manipulative practices
prohibited under the proposal for Artificial Intelligence Act” — and here it lists being a launch partner of the Partnership on AI’s (PAI) “Responsible
Practices for Synthetic Media” (and contributing to the development of “relevant practices”); and joining “new relevant groups”, such as the Generative AI
working group which started work this month as implemented measures towards this pledge.

In the next six months it says it wants to further strengthen its enforcement of its synthetic media policy — and explore “new products and initiatives to help
enhance our detection and enforcement capabilities” in this area, including in the area of user education.

Meta

Facebook and Instagram parent Meta’s report also includes a recognition that “widespread availability and adoption of generative AI tools may have implications
for how we identify, and address disinformation on our platforms”.
</item 33>

<item 34>
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
</item 34>

<item 35>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Italy’s data protection authority has just put out a timely reminder that some countries do have laws that already apply to cutting edge AI: it has ordered
OpenAI to stop processing people’s data locally with immediate effect. The Italian DPA said it’s concerned that the ChatGPT maker is breaching the European
Union’s General Data Protection Regulation (GDPR), and is opening an investigation.

March 2023

1,100+ signatories signed an open letter asking all ‘AI labs to immediately pause for 6 months’

The letter’s signatories include Elon Musk, Steve Wozniak and Tristan Harris of the Center for Humane Technology, among others. The letter calls on “all AI labs
to immediately pause for at least 6 months the training of AI systems more powerful than GPT-4.”

The letter reads:

Contemporary AI systems are now becoming human-competitive at general tasks,[3] and we must ask ourselves: Should we let machines flood our information channels
with propaganda and untruth?
</item 35>

<item 36>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
YouTube says it paid out over USD $6 billion to music rightsholders in the year to end of June 2022, with around $2 billion of that coming just from ads on
user-generated content.)

However, the October US Copyright Office filing from Google has definitely put the cat amongst the pigeons at Sony Music HQ. As we’re sure it has at the other
two major record companies.

“Any prohibition or limitation on the use of copyrighted materials for purposes of Al training would undermine the purpose of copyright and foreclose the many
opportunities that come with this technology.” Google filing with the US Copyright Office, October 30

Witness Universal Music Group’s own filing with the US Copyright Office on the topic of AI. It couldn’t offer a starker contrast to Google’s assertion that
“prohibition or limitation on the use of copyrighted materials for purposes of Al training would undermine the purpose of copyright and foreclose the many
opportunities that come with this technology”.
</item 36>

<item 37>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
The Council’s PR further stipulates that the higher sanction (7%) would apply for violations of the banned AI applications, while penalties of 1.5% would be
levied for the supply of incorrect information. Additionally, it says sanctions of 3% could be imposed for violations of other AI Act obligations but also notes
that the provisional agreement allows for “more proportionate caps” on administrative fines for SMEs and start-ups in case of infringements. So there looks to
be some scope for AI startups to face smaller penalties for infringements than AI giants may invite.

The deal agreed today also allows for a phased entry into force after the law is adopted — with six months allowed until rules on prohibited use cases kick in;
12 months for transparency and governance requirements; and 24 months for all other requirements. So the full force of the EU’s AI Act may not be felt until
2026.
</item 37>

<item 38>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
And it was an incredible new revenue stream for everyone. AI is that with new super tools.”

Kyncl was referring to YouTube’s Content ID system, which scours uploaded videos for copyrighted content (video and audio), then offers the copyright owner the
option to monetize that video, or to request that it be taken down. The innovative system has essentially allowed YouTube users to upload content without paying
much attention to copyright, while ensuring that copyright owners are paid.

When it comes to the proliferation of AI, “we need to approach it with the same thoughtfulness, and we have to make sure that artists have a choice,” Kyncl
added.

And he made it clear that, in his view, simply rejecting AI and fighting against it is not an option.

“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.”
</item 38>

<item 39>
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
</item 39>

<item 40>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
are now allowed to use ChatGPT in legal rulings

The U.K. Judicial Office issued guidance that permits judges to use ChatGPT, along with other AI tools, to write legal rulings and perform court duties. The
guidance lays out ways to responsibly use AI in the courts, including being aware of potential bias and upholding privacy.

OpenAI makes repeating words “forever” a violation of its terms of service after Google DeepMind test

Following an experiment by Google DeepMind researchers that led ChatGPT to repeat portions of its training data, OpenAI has flagged asking ChatGPT to repeat
specific words “forever” as a violation of its terms of service.

Lawmakers in Brazil enact an ordinance written by ChatGPT

City lawmakers in Brazil enacted a piece of legislation written entirely by ChatGPT without even knowing. Weeks after the bill was passed, Porto Alegre
councilman Ramiro Rosário admitted that he used ChatGPT to write the proposal, and did not tell fellow council members until after the fact.
</item 40>

<item 41>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Human authorship is a bedrock requirement of copyright.”

2. AI developers should be required to license materials and keep records of the materials they use for training

Many key players in the music industry and beyond have been adamant that using copyrighted works to train AI algorithms without authorization is a copyright
infringement, and a number of lawsuits have been launched against AI companies on these grounds.

One such lawsuit was filed by Universal Music Group, Concord Music Group and ABKCO against AI developer Anthropic, seeking potentially tens of millions of
dollars for the alleged “systematic and widespread infringement of their copyrighted song lyrics.”

But that doesn’t mean key music industry players don’t want their materials ever used to train AI – they just want AI developers to license that music, not
unlike how streaming services, bars and restaurants or movie producers license music.
</item 41>

<item 42>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
In addition, rights can only be enforced around the training of AI when there are accurate records about what is being copied. Otherwise, the inability to
enforce rights in the AI marketplace equates to a lack of rights at all, producing a dangerous imbalance that prevents a thriving ecosystem. This requires
strong and accurate record keeping by the generative AI platforms, a requirement that urgently needs legislative support to ensure incentives are in place so
that it happens consistently and correctly.” Assure Transparency for Consumers and Artists. “Transparency is necessary to clearly distinguish human-created
works from AI-created works. The public should know, when they are listening to music, whether that music was created by a human being or a machine.”

Added Kooker: “While these principles are simple and basic, they require a new level of commitment and investment from generative AI platforms.
</item 42>

<item 43>
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
</item 43>

<item 44>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
But last month something else happened that, we hear, effectively scuppered any chance that Sony Music would get involved with ‘Dream Track’ at this juncture.

On October 30, Google – the parent of YouTube within Alphabet Inc. – issued a filing with the US Copyright Office that outlined its core position on ‘Artificial
Intelligence and Copyright’.

The contents of this document have caused some alarm at Sony Music HQ.

What was in the Google document?

You may have read on MBW last week that Ed Newton-Rex – a generative AI pioneer, but also a published composer – decided to quit his role at Stability AI over
concerns over the company’s position on “fair use” within its own recent US Copyright Office filing.

‘Fair use’ is, as it sounds, the argument that there are cases where copyrighted material can be used and even replicated that are, well, fair enough.
</item 44>

<item 45>
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
</item 45>

<item 46>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
To be fair to Google, its submission does make note of the importance of US copyright law, including when applied to generative AI, striking “the right balance
between the legitimate interests of rightsholders and the equally legitimate interests of the public and succeeding generations of creators”.

Yet it could clearly be argued that Google’s take on copyright “harvesting” by generative AI models sits in odd contrast to the copyright-related reassurances
that YouTube has been careful to make to the music industry during the pre-development, development, and trial launch of ‘Dream Track’.

A final thought…

MBW’s sources close to Sony Music were keen to point out that the company’s general ongoing relationship with YouTube and YouTube Music is a harmonious one.

(YouTube is, after all, now the second largest commercial partner of the major record companies – with a stated ambition to catch up with Spotify in the years
ahead.
</item 46>

<item 47>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
So it will be worth scrutinizing the text that emerges, likely in January or February.

Additionally, while the full regulation won’t be up and running for a few years the EU will be pushing for GPAIs to abide by codes of practice in the meanwhile
— so AI giants will be under pressure to stick as close to the hard regulations coming down the pipe as possible, via the bloc’s AI Pact.

The EU AI Act itself likely won’t be in full force until some time in 2026 — given the final text must, once compiled (and translated into Member States’
languages), be affirmed by final votes in the parliament and Council, after which there’s a short period before the text of the law is published in the EU’s
Official Journal and another before it comes into force.

EU lawmakers have also agreed a phased approach to the Act’s compliance demands, with 24 months allowed before the high risk rules will apply for GPAIs.
</item 47>

<item 48>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
On that front, the European Union may have gone some – if not all – of the way in making that a reality. Its proposed AI Act – currently the subject of
back-and-forth negotiations between the European Commission and the European Parliament – contains a rule that would require AI developers to disclose whether
their AI had been trained on copyrighted materials.

Transparency in the use of AI is also becoming a rule on social media sites where AI-generated content is likely to appear: Both TikTok and YouTube have now
passed rules requiring AI-generated content to be labeled as such.

3. The NMPA is opposed to an ‘opt-out’ regime for use of copyrighted works in AI

One of the questions the US Copyright Office posed in its call for submissions was “should copyright owners have to affirmatively consent (opt in) to the use of
their works for training [AI], or should they be provided with the means to object (opt out)?”

The NMPA made it clear it’s on the side of “opt in.”
</item 48>

<item 49>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
The goal is that AI tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

“It’s really interesting from a music perspective, specifically, because . . . musicians are perhaps more familiar with the idea of AI,” Holland said.
“Musicians in general are more familiar with things like music production software, and AI tools like MIDI drum loops . . . so I think that there is a certain
amount of more progressive learning from them, when it comes to technology, and its ability to make their music better.”

When it comes to art and AI, the conversation is complicated, to say the least. Musicians are nervous about industry giants copyrighting AI music and cutting
them out of the process. Major record labels are worried about AI models training on their catalogues and stealing a slice of their considerable pie.
</item 49>

<item 50>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

Legislators and regulators around the world are grappling with how to address the many issues that have emerged since AI technology “went mainstream” about a
year ago.

In the US, one of the key efforts on that front is being spearheaded by the US Copyright Office, which this past summer issued a call for submissions on the
issue of AI and copyright. The goal is for the USCO to put together a study to ”help assess whether legislative or regulatory steps in this area are warranted.”

Numerous businesses involved in AI, and those with substantial copyright holdings, have submitted their thoughts on the issue, including Universal Music Group
and AI developer Anthropic.
</item 50>
=== END ITEM id=5d6ed2c972ddcaaa ===
