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

This file has 2 items.

=== ITEM id=0848788caad8a0ca ===
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
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
As reported by MBW last week, Universal Music Group submitted its response to the USCO. Unsurprisingly, amongst the highlights of UMG’s submission was the
rejection of the notion that using copyrighted material to train LLMs is “fair use”.

In his speech on Wednesday, Kooker commented directly on the flurry of submissions to the USCO from the tech industry at the end of October.

In what may have been a nod to Google’s filing and “fair use” position, Kooker told Senators that “based on recent Copyright Office filings it is clear that the
technology industry and speculative financial investors would like governments to believe in a very distorted view of copyright”.

He added: “[That view is] one in which music is considered fair use for training purposes and in which certain companies are permitted to appropriate the entire
value produced by the creative sector without permission, and to build huge businesses based on it without paying anything to the creators concerned.”
</item 1>

<item 2>
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
</item 2>

<item 3>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Kooker also openly discussed Sony Music’s efforts to get unauthorized AI content removed from platforms online – including the fact that the firm has already
issued close to 10,000 separate takedowns.

Kooker further discussed Sony Music’s optimism about AI’s potential for enhancing creativity, as well as key principles the company has adopted around
generative AI.

Here are three things that stood out from Kooker’s speech…

1. Sony Music has issued close to 10,000 takedowns for unauthorized deep fakes of artists

Kooker argued in his speech that, while “the music industry sees great potential with various forms of AI technology in this early phase of AI,” so far, “the
available generative AI products are not delivering on the expected promise of new products in creative industries”.

He clarified that point by explaining that some generative AI platforms “are not expanding the business model or enhancing human creativity”.
</item 3>

<item 4>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
The goal is that AI tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

“It’s really interesting from a music perspective, specifically, because . . . musicians are perhaps more familiar with the idea of AI,” Holland said.
“Musicians in general are more familiar with things like music production software, and AI tools like MIDI drum loops . . . so I think that there is a certain
amount of more progressive learning from them, when it comes to technology, and its ability to make their music better.”

When it comes to art and AI, the conversation is complicated, to say the least. Musicians are nervous about industry giants copyrighting AI music and cutting
them out of the process. Major record labels are worried about AI models training on their catalogues and stealing a slice of their considerable pie.
</item 4>

<item 5>
California’s privacy watchdog eyes AI rules with opt-out and access rights (TechCrunch, 2023-11-27T17:30:24+00:00)
And who knows how far developments in AI will have moved on by then.

* The CPPA’s proposed definition for ADMT in the draft framework is “any system, software, or process — including one derived from machine-learning, statistics,
other data-processing or artificial intelligence — that processes personal information and uses computation as whole or part of a system to make or execute a
decision or facilitate human decisionmaking”. Its definition also affirms “ADMT includes profiling” — which is defined as “any form of automated processing of
personal information to evaluate certain personal aspects relating to a natural person and in particular to analyze or predict aspects concerning that natural
person’s performance at work, economic situation, health, personal preferences, interests, reliability, behavior, location, or movements”
</item 5>

<item 6>
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
</item 6>

<item 7>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q3: [Pushback intensifies]

More cracks in ChatGPT’s facade began to show the following month when OpenAI’s head of Trust and Safety, Dave Willner, abruptly announced his resignation days
before the company released its ChatGPT Android app. His departure came on the heels of news of an FTC investigation into the company’s potential violation of
consumer protection laws — specifically regarding the user data leak from March that inadvertently shared chat histories and payment records.

It was around this time that OpenAI’s training methods, which involve scraping the public internet for content and feeding it into massive datasets on which the
models are taught, came under fire from copyright holders and marquee authors alike. Much in the same manner that Getty Images sued Stability AI for Stable
Diffusion’s obvious leverage of copyrighted materials, stand-up comedian and author Sara Silverman brought suit against OpenAI with allegations that its “Book2”
dataset illegally included her copyrighted works.
</item 7>

<item 8>
California’s privacy watchdog eyes AI rules with opt-out and access rights (TechCrunch, 2023-11-27T17:30:24+00:00)
Of course a law that lets consumers opt-out of privacy-hostile data processing is only going to work if the people involved are aware how their information is
being used. Hence the planned framework also sets out a requirement that businesses wanting to apply ADMT must provide so-called “pre-use notices” to affected
consumers — so they can decide whether to opt-out of their data being used (or not); or indeed whether to exercise their access right to get more info about the
intended use of automation/AI.

This too looks broadly similar to provisions in the EU’s GDPR which put transparency (and fairness) obligations on entities processing personal data — in
addition to requiring a valid lawful basis for them to use personal data.

Although the European regulation contains some exceptions — such as where info was not directly collected from individuals and fulfilling their right to be
informed would be “unreasonably expensive” or “impossible” — which may have undermined EU lawmakers’ intent that data subjects should be kept informed.
</item 8>

<item 9>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT can now browse the internet to provide you with current and authoritative information, complete with direct links to sources. It is no longer limited to
data before September 2021. pic.twitter.com/pyj8a9HWkB — OpenAI (@OpenAI) September 27, 2023

ChatGPT now has a voice

OpenAI announced that it’s adding a new voice for verbal conversations and image-based smarts to the AI-powered chatbot.

Poland opens an investigation against OpenAI

The Polish authority publically announced it has opened an investigation regarding ChatGPT — accusing the company of a string of breaches of the EU’s General
Data Protection Regulation (GDPR).

OpenAI unveils DALL-E 3

The upgraded text-to-image tool, DALL-E 3, uses ChatGPT to help fill in prompts.
</item 9>

<item 10>
California’s privacy watchdog eyes AI rules with opt-out and access rights (TechCrunch, 2023-11-27T17:30:24+00:00)
The current draft further stipulates that behavioral advertising use-cases cannot make use of a number of exemptions to the opt-out right that may apply in
other scenarios (such as if ADMT is being used for security or fraud prevention purposes, for example).

The CPPA’s approach to regulating ADMT is risk-based, per Soltani. This echoes another piece of in-train EU legislation: the AI Act — a dedicated risk-based
framework for regulating applications of artificial intelligence which has been on the table in draft form since 2021 but is now at a delicate stage of
co-legislation, with the bloc’s lawmakers clashing over the not-so-tiny-detail of how (or even whether) to regulate Big AI, among several other policy disputes
on the file.

Given the discord around the EU’s AI Act, as well as the ongoing failure of US lawmakers to pass a comprehensive federal privacy law — since there’s only so
much presidential Executive Orders can do — there’s a plausible prospect of California ending up as one of the top global rulemakers on AI.
</item 10>

<item 11>
The UK just laid out new rules for the internet — it only gets harder from here (The Verge, 2023-11-09T00:01:00+00:00)
But WhatsApp, other encrypted messaging services, and digital rights groups say this scanning would require breaking apps’ encryption systems and invading user
privacy. Whitehead says that Ofcom plans to consult on this next year, leaving its full impact on encrypted messaging uncertain.

“We’re not regulating the technology, we’re regulating the context.”

There’s another technology not emphasized in today’s consultation: artificial intelligence. But that doesn’t mean AI-generated content won’t fall under the
rules. The Online Safety Act attempts to address online harms in a “technology neutral” way, Whitehead says, regardless of how they’ve been created. So
AI-generated CSAM would be in scope by virtue of it being CSAM, and a deepfake used to conduct fraud would be in scope by virtue of the fraud. “We’re not
regulating the technology, we’re regulating the context,” Whitehead says.
</item 11>

<item 12>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

Legislators and regulators around the world are grappling with how to address the many issues that have emerged since AI technology “went mainstream” about a
year ago.

In the US, one of the key efforts on that front is being spearheaded by the US Copyright Office, which this past summer issued a call for submissions on the
issue of AI and copyright. The goal is for the USCO to put together a study to ”help assess whether legislative or regulatory steps in this area are warranted.”

Numerous businesses involved in AI, and those with substantial copyright holdings, have submitted their thoughts on the issue, including Universal Music Group
and AI developer Anthropic.
</item 12>

<item 13>
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
</item 13>

<item 14>
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
</item 14>

<item 15>
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
</item 15>

<item 16>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
The concerns pushed policymakers in Europe to mandate security assessments for any products using generative AI systems like ChatGPT, and over 20,000
signatories — including Elon Musk and Apple co-founder Steve Wozniak — to sign an open letter calling for the immediate pause of large-scale AI experiments like
ChatGPT.

But examples of ChatGPT abuse in the wild have been few and far between — so far.

With the launch of GPTs, OpenAI’s tool for building custom conversational, action-taking AI systems powered by OpenAI’s models, including the models
underpinning ChatGPT, ChatGPT could become more a gateway to a broader ecosystem of AI-powered chatbots than the end-all-be-all.

With GPTs, a user can train a model on a cookbook collection, for example, so that it can answer questions about ingredients for a specific recipe. Or they can
give a model their company’s proprietary codebases so that developers can check their style or generate code in line with best practices.
</item 16>

<item 17>
California’s privacy watchdog eyes AI rules with opt-out and access rights (TechCrunch, 2023-11-27T17:30:24+00:00)
But, it being California, the CCPA can probably expect a lot of pushback from local giants like Meta and OpenAI, to name two.

The draft proposal marks the start of the CPPA’s rulemaking process, with the aforementioned consultation process — which will include a public component — set
to kick off in the coming weeks. So it’s still a ways off a final text. A spokeswoman for the CPPA said it’s unable to comment on a possible timeline for the
rulemaking but she noted this is something that will be discussed at the upcoming board meeting, on December 8.

If the Agency is able to move quickly it’s possible it could have a regulation finalized in the second half of next year. Although there would obviously need to
be a grace period before compliance kicks in for in-scope companies — so 2025 looks like the very earliest for a law to be up and running. And who knows how far
developments in AI will have moved on by then.
</item 17>

<item 18>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
NMPA

In this regard, US legal precedent is – at least so far – on the NMPA’s side.

In a recent court case, an individual by the name of Stephen Thaler, who owns a computer system he calls the “Creativity Machine,” had a copyright application
for an AI-generated “painting” rejected by the USCO on the grounds that the Creativity Machine had created it.

In a ruling this past August, a US District Court in Washington, D.C., sided with the Copyright Office.

“Copyright has never stretched so far… as to protect works generated by new forms of technology operating absent any guiding human hand, as plaintiff urges
here,” Judge Beryl A. Howell commented. “Human authorship is a bedrock requirement of copyright.”

2.
</item 18>

<item 19>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Apple is developing AI tools to challenge OpenAI, Google and others. The tech giant created a chatbot that some engineers are internally referring to as “Apple
GPT,” but Apple has yet to determine a strategy for releasing the AI to consumers.

How does ChatGPT handle data privacy?

OpenAI has said that individuals in “certain jurisdictions” (such as the EU) can object to the processing of their personal information by its AI models by
filling out this form. This includes the ability to make requests for deletion of AI-generated references about you. Although OpenAI notes it may not grant
every request since it must balance privacy requests against freedom of expression “in accordance with applicable laws”.

The web form for making a deletion of data about you request is entitled “OpenAI Personal Data Removal Request”.
</item 19>

<item 20>
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
</item 20>

<item 21>
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
</item 21>

<item 22>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q2: Electric Boog-AI-loo

Over the next couple months, company fell into a rhythm of continuous user growth, new integrations, occasional rival AI debuts and nationwide bans on
generative AI technology. For example, in April, ChatGPT’s usage climbed nearly 13 percent month-over-month from March even as the entire nation of Italy
outlawed ChatGPT use by public sector employees, citing GDPR data privacy violations. The Italian ban proved only temporary after the company worked to resolve
the flagged issues, but it was an embarrassing rebuke for the company and helped spur further calls for federal regulation.

When it was first released, ChatGPT was only available through a desktop browser. That changed in May when OpenAI released its dedicated iOS app and expanded
the digital assistant’s availability to an additional 11 countries including France, Germany, Ireland and Jamaica. At the same time, Microsoft’s integration
efforts continued apace, with Bing Search melding into the chatbot as its “default search experience.”
</item 22>

<item 23>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
The 🇪🇺 AI Act is a global first. A unique legal framework for the development of AI you can trust. And for the safety and fundamental rights of people and
businesses. A commitment we took in our political guidelines – and we delivered. I welcome today's political agreement. — Ursula von der Leyen (@vonderleyen)
December 8, 2023

Prohibitions

Full details of what’s been agreed won’t be entirely confirmed until a final text is compiled and made public, which may take some weeks. But a press release
put out by the European Parliament confirms the deal reached with the Council includes a total prohibition on the use of AI for:

biometric categorisation systems that use sensitive characteristics (e.g.
</item 23>

<item 24>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“These partnerships are part of a larger effort to empower Microsoft users to better understand the information they consume across our platforms and products,”
it suggests, also citing efforts undertaken in media literacy campaigns and “cyber-skilling” which it says are “not designed to tell individuals what to believe
or how to think; rather, they are about equipping people to think critically and make informed decisions about what information they consume”.

On Bing Search, where Microsoft was quick to embed generative AI features — leading to some embarrassing early reviews which demonstrated the tool producing
dubious content — the report claims it has taken a raft of measures to mitigate risks including applying its AI principles during development and consulting
with experts; engaging in pre-launch testing and a limited preview period and phased release; the use of classifiers and metaprompting, defensive search
interventions, enhanced reporting functionality, and increased operations and incident response; as well as updating Bing’s terms of use to include a Code of
Conduct for users.
</item 24>

<item 25>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
This all seemed to hint at a desire to build a ‘Content ID’-style system on YouTube to control and police content made by cloning artists’ voices;

publicly committed to three principles/pledges behind its development of music-based . These pledges promised in said tools. They promised that, amongst other
things, would help YouTube . This all seemed to hint at a desire to build a on YouTube to control and police content made by cloning artists’ voices; Then,
earlier this month, YouTube announced it was developing a specific system for music partners that allowed them to request the removal of content on its platform
that “mimics an artist’s unique singing or rapping voice”.

Evidently, all of this was enough for Universal Music Group and Warner Music Group to dip their toe into the ‘Dream Track’ experiment – though not without
caution.
</item 25>

<item 26>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“An opt-out system is based on the erroneous premise that training on copyrighted works without permission is by default lawful unless each copyright owner
objects,” UMG stated in its submission.

“That philosophy does violence to basic principles of copyright law, imposes undue burdens on copyright owners, creates the wrong incentives for AI developers,
and is neither practicable nor effective for protecting the rights of copyright owners or ensuring the sensible use of copyrighted works for training purposes.”

4. Training AI on copyrighted works is not “fair use” under almost any circumstance

Another area in which the NMPA and UMG are aligned is on the issue of whether using copyrighted material constitutes “fair use.”

Fair use is the carve-out to copyright law that allows copyrighted materials to be used without permission in certain limited circumstances, such as for
educational purposes or for reporting news.
</item 26>

<item 27>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
The Council’s PR further stipulates that the higher sanction (7%) would apply for violations of the banned AI applications, while penalties of 1.5% would be
levied for the supply of incorrect information. Additionally, it says sanctions of 3% could be imposed for violations of other AI Act obligations but also notes
that the provisional agreement allows for “more proportionate caps” on administrative fines for SMEs and start-ups in case of infringements. So there looks to
be some scope for AI startups to face smaller penalties for infringements than AI giants may invite.

The deal agreed today also allows for a phased entry into force after the law is adopted — with six months allowed until rules on prohibited use cases kick in;
12 months for transparency and governance requirements; and 24 months for all other requirements. So the full force of the EU’s AI Act may not be felt until
2026.
</item 27>

<item 28>
California’s privacy watchdog eyes AI rules with opt-out and access rights (TechCrunch, 2023-11-27T17:30:24+00:00)
California’s Privacy Protection Agency (CPPA) is preparing for its next trick: Putting guardrails on AI.

The state privacy regulator, which has an important role in setting rules of the road for digital giants given how much of Big Tech (and Big AI) is
headquartered on its sun-kissed soil, has today published draft regulations for how people’s data can be used for what it refers to as automated decisionmaking
technology (ADMT*). Aka AI.

The draft represents “by far the most comprehensive and detailed set of rules in the ‘AI space'”, Ashkan Soltani, the CPPA’s exec director, told TechCrunch. The
approach takes inspiration from existing rules in the European Union, where the bloc’s General Data Protection Regulation (GDPR) has given individuals rights
over automated decisions with a legal or significant impact on them since coming into force back in May 2018 — but aims to build on it with more specific
provisions that may be harder for tech giants to wiggle away from.
</item 28>

<item 29>
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
</item 29>

<item 30>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Yet something conspicuous is not on there: any endorsement whatsoever from the world’s second-biggest recording company, Sony Music Entertainment, or its
artists.

Why is Sony missing?

MBW understands that Sony Music (the recorded music arm of Sony Music Group) has, like Universal and Warner, been in discussions for months with YouTube over
potentially licensing the ‘Dream Track’ experiment.

During this period of discussion, YouTube has made a number of positive announcements regarding the biggest issue for any rightsholder regarding AI-driven voice
replication of artists: their ability to police it.

In August, YouTube and Universal Music Group jointly announced that the video platform was launching an ‘AI Music Incubator’ – a program by which new tools and
innovations will be developed at YouTube in close conjunction with artists and the music biz.
</item 30>

<item 31>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Dennis Kooker, Sony Music

Kooker added later in the speech: “Music is a tremendous driver for AI technology, and AI technology presents a tremendous opportunity for the creative
development of music.

“But these opportunities must be grounded by the human creators’ vision with the machine assisting, not with the machine replacing the human creator.”

To achieve this, Kooker said that SME wants Congress to embrace the following principals:

Assure Consent, Compensation, and Credit. “New products and businesses built with music must be developed with the consent of the owner and appropriate
compensation and credit. It is essential to understand why the training of AI models is being done, what products will be developed as a result, and what the
business model is that will monetize the use of the artist’s work. Congress and the agencies should assure that creators’ rights are recognized and respected.”
Confirm That Copying Music to Train AI Models is Not Fair Use.
</item 31>

<item 32>
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
</item 32>

<item 33>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
You can read the full Google USCO submission here, but below are a few of the choicest sections on this topic:

“The doctrine of fair use [within existing US copyright law] provides that copying for a new and different purpose is permitted without authorization where — as
with training Al systems — the secondary use is transformative and does not substitute for the copyrighted work .”

[within existing US copyright law] provides that copying for a new and different purpose is permitted without authorization where — as with — the secondary use
is transformative and .” “If [AI] training could be accomplished without the creation of copies, there would be no copyright questions here. Indeed that act of
“knowledge harvesting”… like the act of reading a book and learning the facts and ideas within it , would not only be non-infringing , it would further the very
purpose of copyright law.
</item 33>

<item 34>
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
</item 34>

<item 35>
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
</item 35>

<item 36>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
And that’s all some people need.

Likely there will be some half-hearted efforts to block generated content from being used in this way, but these posts can’t be taken down fast enough by the
likes of Meta and Google, and the idea that X can (or will) effectively monitor and take down such content is implausible. It’s gonna be a bad time!
</item 36>

<item 37>
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
</item 37>

<item 38>
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
</item 38>

<item 39>
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
</item 39>

<item 40>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
There was also agreement on a “two-tier” system of guardrails to be applied to “general” AI systems, such as the so-called foundational models underpinning the
viral boom in generative AI applications like ChatGPT.

As we reported earlier, the deal reached on foundational models/general purpose AIs (GPAIs) includes some transparency requirements for what co-legislators
referred to as “low tier” AIs — meaning model makers must draw up technical documentation and produce (and publish) detailed summaries about the content used
for training in order to support compliance with EU copyright law. For “high-impact” GPAIs (defined as the cumulative amount of compute used for their training
measured in floating point operations is greater than 10^25) with so-called “systemic risk” there are more stringent obligations.

“If these models meet certain criteria they will have to conduct model evaluations, assess and mitigate systemic risks, conduct adversarial testing, report to
the Commission on serious incidents, ensure cybersecurity and report on their energy efficiency,” the parliament wrote.
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
What’s at stake in the Supreme Court’s landmark social media case (TechCrunch, 2023-10-04T20:07:42+00:00)
Aside from forcing platforms to allow some forms of content that would otherwise be disallowed, these laws also seek to force social media companies to provide
users with individualized explanations when their content is removed or restricted. Because this process is largely conducted algorithmically now — generally
with light human intervention or oversight — social media companies might need to reimagine their content moderation systems or hire way more humans to respond
to these incidents. Those kind of adjustments would likely be costly and difficult to scale.

“As you can imagine, that can get kind of onerous when you’re taking down millions and millions of pieces of content a day, and much of that activity of the
vast majority of it is currently being done automatically,” Barrett said. “The idea that a human being would have to go back and explain each time something
came down would be quite a challenge.”
</item 42>

<item 43>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
“ChatGPT Business will follow our API’s data usage policies, which means that end users’ data won’t be used to train our models by default,” OpenAI wrote in a
blog post. “We plan to make ChatGPT Business available in the coming months.”

OpenAI wants to trademark “GPT”

OpenAI applied for a trademark for “GPT,” which stands for “Generative Pre-trained Transformer,” last December. Last month, the company petitioned the USPTO to
speed up the process, citing the “myriad infringements and counterfeit apps” beginning to spring into existence.

Unfortunately for OpenAI, its petition was dismissed last week. According to the agency, OpenAI’s attorneys neglected to pay an associated fee as well as
provide “appropriate documentary evidence supporting the justification of special action.”

That means a decision could take up to five more months.
</item 43>

<item 44>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“Recognizing that there is an important role for government, academia and civil society to play in the responsible deployment of AI, we also created a roadmap
for the governance of AI across the world as well as creating a vision for the responsible advancement of AI, both inside Microsoft and throughout the world,
including specifically in Europe,” Microsoft goes on, committing to continue building on efforts — including by developing new tools (such as Project Providence
with Truepic) and inking partnerships (examples it gives include the Coalition for Content Provenance and Authenticity (C2PA), to combat the rise of manipulated
or AI created media; with EFE Verifica to track false narratives spreading in Spain, Latin America, and Spanish speaking populations; and Reporters Sans
Frontières to use their Journalism Trust Initiative dataset in Microsoft products).
</item 44>

<item 45>
California’s privacy watchdog eyes AI rules with opt-out and access rights (TechCrunch, 2023-11-27T17:30:24+00:00)
Not every use of ADMT will be in-scope of the CCPA’s proposed rules. The draft regulation proposes to set a threshold as follows:

For a decision that produces legal or similarly significant effects concerning a consumer (e.g., decisions to provide or deny employment opportunities).
Profiling a consumer who is acting in their capacity as an employee, independent contractor, job applicant, or student. Profiling a consumer while they are in a
publicly accessible place.

The Agency also says the upcoming consultation will discuss whether the rules should also apply to: profiling a consumer for behavioral advertising; profiling a
consumer the business has “actual knowledge is under the age of 16” (i.e. profiling children); and processing the personal information of consumers to train
ADMT — indicating it’s not yet confirmed how much of the planned regime will apply to (and potentially limit the modus operandi of) adtech and data-scraping
generative AI giants.

The more expansive list of proposed thresholds would clearly make the law bite down harder on adtech giants and Big AI.
</item 45>

<item 46>
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
</item 46>

<item 47>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
Spotify erased thousands of AI-crafted songs from its platform but also recently globally launched an AI-powered DJ that curates music for listeners while
talking to them in a synthetic voice.

“The training of generative AI using our artists’ music . . . begs the question as to which side of history all stakeholders in the music ecosystem want to be
on: the side of artists, fans and human creative expression, or on the side of deep fakes, fraud and denying artists their due compensation,” Universal Music
Group said after a song using AI to imitate Drake and The Weeknd, two of its artists, went viral.

These same conversations and contradictions are manifesting across creative industries, but artists themselves don’t always have a seat at the table.
Independent artists in particular are learning that their voices resonate louder when coming together across disciplines to push back against what Holland
describes as an “extraordinary spectrum of exploitation” that leverages their work.
</item 47>

<item 48>
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
</item 48>

<item 49>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
if you want a sustainable career, with repeatability and success.”

Here are three other things we learned from Kyncl’s talk at the Code Conference…

Credit: Tatiana Shepeleva / Shutterstock 1) Music will feel the impact of AI before other industries do – ‘within the next year’

Kyncl predicted that the music business will be out ahead of other industries when it comes to exploiting the potential of AI, and it will also – out of
necessity– be ahead on the issue of how to regulate and monetize the use of AI by non-rights-holders.

“Music – because it’s so broadly distributed and it’s so well aligned with the internet, because it’s short format, it lends itself to recommendations, it’s on
all platforms – is generally first in most transformations and most innovations. So it digitizes first,” he said.

“Whatever happened to the music industry 20 years ago is starting to happen in the movies and TV shows now.
</item 49>

<item 50>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
A clumsy example: I read a great book, then I meet you for breakfast, where I tell you about the story and quote a couple of my favorite passages to you.

To suggest I’ve just committed copyright infringement wouldn’t only be silly – it would mean that copyright protections were getting in the way of the
commercial furtherment of the author/rightsholder (i.e. you might go and buy the book on my recommendation).

Google’s submission to the USCO contains a number of passages that posit a similar position… but for generative AI models. The document argues that the
ingestion of copyrighted material for the training of generative AI platforms shouldn’t be hampered by copyright law.
</item 50>

<item 51>
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
</item 51>

<item 52>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
NMPA

Furthermore, it contends that any argument that an AI algorithm was trained on copyrighted material for “noncommercial” or “research” purposes should be “viewed
skeptically.”

Any business could claim it’s training AI for research or non-commercial purposes, “then shift entirely to commercial exploitation, leaving the creators of the
copied works with no compensation,” the submission argued.

AI training also fails the fair use test on the second factor, as generative AI can be used to create “expressive” works of music, and on the third, given that
AI training means the entirety of a copyrighted work is ingested, “although there is no inherent need for that,” the NMPA states.

AI training also fails the final test because “AI can be used to generate works that compete in the marketplace with the copied works, thereby reducing revenue
from existing licensing markets as well,” the NMPA stated.

The issue of fair use could prove to be one of the key battlegrounds between AI developers and copyright holders.
</item 52>

<item 53>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
The list of strictly prohibited use-cases of AI will apply sooner, just six months after the law enters into force — which could, potentially, mean bans on
certain “unacceptable risk” uses of AI, such as social scoring or Clearview AI-style selfie scraping for facial recognition databases, will get up and running
in the second half of 2024, assuming no last minute opposition to the regulation springs up within the Council or Parliament. (For the full list of banned AI
uses, read our earlier post.)
</item 53>

<item 54>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
“Flooding the zone” has always been an effective tactic and now AI acts as a labor multiplier, allowing more voluminous yet also targeted campaigns. Expect both
false positives and false negatives in a concerted effort to confuse the narrative and make people distrust everything they see and read. That’s a win state for
those politicians who thrive in chaos.

Organizations will tout “AI-powered” analyses to back up purges of voter rolls, challenges to vote counts and other efforts to suppress or interfere with
existing processes.

Generated video and audio will join the fray, and though neither are perfect, they’re good enough to be believable given a bit of fuzzing: The clip doesn’t have
to be perfect, because it will be presented as a grainy zoomed-in cellphone capture in a dark room, or a hot mic at a private event, or what have you. Then it
becomes a matter of “who are you going to believe, me or him?” And that’s all some people need.
</item 54>

<item 55>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
Robert Kyncl, Warner Music Group

In Kyncl’s view, the solutions put in place to address UGC form a “blueprint” than can be used to address AI-generated content on platforms as well.

“When YouTube was formed… people started to upload content, including copyrighted material, which, obviously, put YouTube into hot water with lots of different
copyright holders. And I had the privilege of working through a lot of that and fix it up.

“But we made we made a very important decision, which was to go above and beyond the law, and build a fingerprinting software that allowed us to track the
copyright on our platform, and then have commercial relationship[s] with copyright holders to send them the money. Out of that we built a multi-billion-dollar
business, which now is a multi-billion-dollar business per year. And it was an incredible new revenue stream for everyone. AI is that with new super tools.”
</item 55>

<item 56>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
While real-time use of this intrusive AI tech will be limited in time and location, and can only be used for the following purposes:

targeted searches of victims (abduction, trafficking, sexual exploitation),

prevention of a specific and present terrorist threat, or

the localisation or identification of a person suspected of having committed one of the specific crimes mentioned in the regulation (e.g. terrorism,
trafficking, sexual exploitation, murder, kidnapping, rape, armed robbery, participation in a criminal organisation, environmental crime).

The Council’s press release on the deal emphasizes that the provisional agreement “clarifies that the regulation does not apply to areas outside the scope of EU
law and should not, in any case, affect member states’ competences in national security or any entity entrusted with tasks in this area”. It also confirms the
AI act will not apply to systems exclusively for military or defence purposes.
</item 56>

<item 57>
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
</item 57>

<item 58>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
While there is a more…nefarious side to ChatGPT, it’s clear that AI tools are not going away anytime soon. Since its initial launch nearly a year ago, ChatGPT
has hit 100 million weekly active users, and OpenAI is heavily investing in it.

Prior to the leadership chaos, on November 6, OpenAI held its first developer conference: OpenAI DevDay. During the conference, it announced a slew of updates
coming to GPT, including GPT-4 Turbo (super-charged versions of GPT-4, its latest language-writing model), a multimodal API and a GPT store where users can
create and monetize their own custom versions of GPT.

GPT-4, which can write more naturally and fluently than previous models, remains largely exclusive to paying ChatGPT users. But you can access GPT-4 for free
through Microsoft’s Bing Chat in Microsoft Edge, Google Chrome and Safari web browsers.
</item 58>

<item 59>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
That argument could yet come into play in Anthropic’s legal battle with Universal.

And given the high stakes involved – for the music industry, for the tech industry and for the shape of culture going forward – that case, along with the US
Copyright Office’s study of AI and copyright, are worth keeping a close eye on.

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 59>

<item 60>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In a post on X, CEO Sam Altman looked back on the night before its launch: “what a year it’s been…”

a year ago tonight we were probably just sitting around the office putting the finishing touches on chatgpt before the next morning’s launch. what a year it’s
been… — Sam Altman (@sama) November 30, 2023

Apple and Google avoid naming ChatGPT as their ‘app of the year’

Neither Apple nor Google chose an AI app as its app of the year for 2023, despite the success of ChatGPT’s mobile app, which became the fastest-growing consumer
application in history before the record was broken by Meta’s Threads.

An attack from researchers prompts ChatGPT to reveal training data

A test led by researchers at Google DeepMind found that there is a significant amount of privately identifiable information in OpenAI’s LLMs.
</item 60>

<item 61>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
They’re all chasing after a customer service software market that’ll be worth $58.1 billion by 2023, assuming the rather optimistic prediction from Acumen
Research comes true.

Here are the YC-backed startups that caught our eye:

Yuma, whose customer demographic is primarily Shopify merchants, provides ChatGPT-like AI systems that integrate with help desk software, suggesting drafts of
replies to customer tickets.

Baselit, which uses one of OpenAI’s text-understanding models to allow businesses to embed chatbot-style analytics for their customers.

Lasso customers send descriptions or videos of the processes they’d like to automate and the company combines ChatGPT-like interface with robotic process
automation (RPA) and a Chrome extension to build out those automations.

BerriAI, whose platform is designed to help developers spin up ChatGPT apps for their organization data through various data connectors.

Italy orders ChatGPT to be blocked

OpenAI has started geoblocking access to its generative AI chatbot, ChatGPT, in Italy.
</item 61>

<item 62>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
In addition, rights can only be enforced around the training of AI when there are accurate records about what is being copied. Otherwise, the inability to
enforce rights in the AI marketplace equates to a lack of rights at all, producing a dangerous imbalance that prevents a thriving ecosystem. This requires
strong and accurate record keeping by the generative AI platforms, a requirement that urgently needs legislative support to ensure incentives are in place so
that it happens consistently and correctly.” Assure Transparency for Consumers and Artists. “Transparency is necessary to clearly distinguish human-created
works from AI-created works. The public should know, when they are listening to music, whether that music was created by a human being or a machine.”

Added Kooker: “While these principles are simple and basic, they require a new level of commitment and investment from generative AI platforms.
</item 62>

<item 63>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
However, what makes the app different from the default experience or the dozens of generic AI chat apps now available are the characters offered which you can
use to engage with SuperChat’s AI features.

Italy gives OpenAI to-do list for lifting ChatGPT suspension order

Italy’s data protection watchdog has laid out what OpenAI needs to do for it to lift an order against ChatGPT issued at the end of last month — when it said it
suspected the AI chatbot service was in breach of the EU’s GSPR and ordered the U.S.-based company to stop processing locals’ data.

The DPA has given OpenAI a deadline — of April 30 — to get the regulator’s compliance demands done. (The local radio, TV and internet awareness campaign has a
slightly more generous timeline of May 15 to be actioned.)
</item 63>

<item 64>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
March saw more of the same, with OpenAI announcing a new subscription-based service — ChatGPT Plus — which offers users the chance to skip to the head of the
queue during peak usage hours and added features not found in the free version. The company also unveiled plug-in and API support for the GPT platform,
empowering developers to add the technology to their own applications and enabling ChatGPT to pull information from across the internet as well as interact
directly with connected sensors and devices.

ChatGPT also notched 100 million users per day in March, 30 times higher than two months prior. Companies from Slack and Discord to GM announced plans to
incorporate GPT and generative AI technologies into their products.

Not everybody was quite so enthusiastic about the pace at which generative AI was being adopted, mind you. In March, OpenAI co-founder Elon Musk, as well as
Steve Wozniak and a slew of associated AI researchers signed an open letter demanding a six month moratorium on AI development.
</item 64>

<item 65>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
“We have identified an important role for a scientific panel to be set up where the scientific panel can effectively help the Artificial Intelligence Office in
understanding whether there are new risks that have not been yet identified,” the official noted. “And, for example, also flag some alerts about the models that
are not captured by the FLOP threshold that for certain reasons could actually give rise to important risks that governments should should look at.”

While the EU’s executive seems keen to ensure key details of the incoming law are put out there in spite of there being no final text yet — because work to
consolidate what was agreed by co-legislators during the marathon 38 hour talks that ended on Friday night is the next task facing the bloc over the coming
weeks — there could still be some devils lurking in that detail. So it will be worth scrutinizing the text that emerges, likely in January or February.
</item 65>

<item 66>
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
</item 66>

<item 67>
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
</item 67>

<item 68>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Sir Lucian Grainge, Chairman and CEO of Universal Music Group, said of UMG’s decision to offer a limited license to ‘Dream Track’: “We have a fundamental
responsibility to our artists to ensure the digital ecosystem evolves to protect them and their work against unauthorized exploitation, including by generative
AI platforms.

“At the same time, we must help artists achieve their greatest creative and commercial potential – in part by providing them access to the kind of opportunities
and cutting-edge creative tools made possible by AI.”

“We have a fundamental responsibility to our artists to ensure the digital ecosystem evolves to protect them and their work against unauthorized exploitation,
including by generative AI platforms.
</item 68>

<item 69>
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
</item 69>

<item 70>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.
</item 70>

<item 71>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Licensing musical works before training use is both required and practicable,” the NMPA’s submission states, adding that “there are well developed processes in
place for technology ventures to obtain free market licenses on a large scale.”

And indeed it would have to be on a “large scale,” given that AI algorithms such as large language models are trained on countless millions or even billions of
pieces of digital data.

“Transparency and recordkeeping requirements are needed to disincentivize infringing activity and to support enforcement activities by copyright owners.” NMPA

The NMPA notes that work on licensing “is already underway,” pointing to news reports earlier this year that UMG, along with Warner Music Group, are in talks
with Google to secure licenses for artists’ vocals and musical melodies to create AI-generated songs.
</item 71>

<item 72>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
As creative industries grapple with AI’s explosion into every artistic medium at once, separate calls from artists warning the world to take action before it’s
too late are starting to converge. From fake Drake songs to stylized Instagram profile pictures, art conjured with newly sophisticated AI tools is suddenly
ubiquitous — and so are conversations about how to rein in the technology before it does irrevocable harm to creative communities.

This week, digital rights organization Fight for the Future partnered with music industry labor group United Musicians and Allied Workers to launch
#AIdayofaction, a campaign that calls on Congress to block corporations from obtaining copyrights on music and other art made with AI.

The idea is that by preventing industry behemoths like major record labels, for example, from copyrighting music made with the assistance of AI, those companies
will be forced to keep looping humans into the creative process. But those same concerns — and the same potential strategies for pushing back against the
onslaught of AI — exist across creative industries.
</item 72>

<item 73>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Confirm That Copying Music to Train AI Models is Not Fair Use. “Even worse are those that argue that copyrighted content should automatically be considered fair
use so that protected works are never compensated for usage and creators have no say in the products or business models that are developed around them and their
work. Congress should assure and agencies should presume that reproducing music to train AI models, in itself, is not a fair use.” Prevent the Cloning of
Artists’ Voices and Likenesses Without Express Permission. “We cannot allow an artist’s voice or likeness to be cloned for use without the express permission of
the artist. This is a very personal decision for the artist. Congress should pass into law effective federal protections for name, image, and likeness.”
Incentivize Accurate Recordkeeping. “Correct attribution will be a critical element to artists being paid fairly and correctly for new works that are created.
</item 73>

<item 74>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
Microsoft

Microsoft — a major investor in OpenAI which has also baked generative AI capabilities into its own search engine — claims it’s taking “a cross product whole of
company approach to ensure the responsible implementation of AI”.

Its report flags its “Responsible AI Principles” which it says it’s developed into a Responsible AI standard v.2 and Information Integrity Principles “to help
set baseline standards and guidance across product teams”.
</item 74>

<item 75>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
“Foundation models, of course, are part of the general purpose AI models. These are models that can be used for a very large variety of tasks, they can also be
integrated in systems. To give you a concrete example, the general purpose AI model would be GPT-4 and the general purpose AI system would be ChatGPT — where
GPT-4 is integrated in ChatGPT.”

As we reported earlier, the deal agreed by the bloc’s co-legislators includes a low risk tier and a high risk tier for regulating so-called general purpose AIs
(GPAIs) — such as models behind the viral boom in generative AI tools like OpenAI’s ChatGPT. The trigger for high risk rules to apply on generative AI
technologies is determined by an initial threshold set out in the law.
</item 75>

<item 76>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“Today, this is a multi-million euro weapon of mass manipulation aimed both internally at the Russians as well as at Europeans and the rest of the world. We
must address this risk. The very large platforms must address this risk. Especially that we have to expect that the Kremlin and others will be active before
elections. I expect signatories to adjust their actions to reflect that there is a war in the information space waged against us and that there are upcoming
elections where malicious actors will try to use the design features of the platforms to manipulate.”

Per the Commission’s early analysis of Big Tech’s Code reports, YouTube shut down more than 400 channels between January and April 2023 which were involved in
coordinated influence operations linked to the Russian-state sponsored Internet Research Agency (IRA). It also removed ads from almost 300 sites linked to
state-funded propaganda sites.

While the EU highlighted that TikTok’s fact-checking efforts now cover Russian, Ukrainian, Belarusian and 17 European languages, including through a new
partnership with Reuters.
</item 76>

<item 77>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists;

and jointly announced that the video platform was launching an – a program by which new tools and innovations will be developed at YouTube in close conjunction
with artists and the music biz. To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists; Also in
August, YouTube publicly committed to three principles/pledges behind its development of music-based generative AI tools . These pledges promised “appropriate
protections… for music partners who decide to participate” in said tools. They promised “invest[ment] in the AI-powered technology” that, amongst other things,
would help YouTube “protect our community of viewers, creators, artists and songwriters… [from] trademark and copyright abuse” .
</item 77>

<item 78>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
These products range from tools for “creative or marketing assistance”, to tools that “potentially give us the ability to better protect artist content or find
it when used in an unauthorized fashion” as well as “brand new products that have never been launched before”.

Kooker said that some of these conversations “also include potential equity investments which would accelerate the development of these companies”.

The Sony Music exec cited, as an example, a recent generative AI-powered project around a reissue and remix of an album. Although Kooker didn’t name it
specifically, SME did recently announce a generative AI project with The Orb and David Gilmour, which was a partnership between Sony Music Entertainment, Legacy
Recordings AI company Vermillio

“These artists are known for their cutting-edge experimentation in music,” said Kooker.
</item 78>

<item 79>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
The political deal clinched by European Union lawmakers late Friday over what the bloc is billing as world’s first comprehensive law for regulating artificial
intelligence includes powers for the Commission to adapt the pan-EU AI rulebook to keep pace with developments in the cutting edge field, it has confirmed.

Lawmakers’ choice of term for regulating the most powerful models behind the current boom in generative AI tools — which the EU Act refers to as “general
purpose” AI models and systems, rather than using industry terms of choice, like “foundational” or “frontier” models — was also selected with an eye on
futureproofing the incoming law, per the Commission, with co-legislators favoring a generic term to avoid a classification that could be chained to use of a
specific technology (i.e. transformer based machine learning).

“In the future, we may have different technical approaches. And so we were looking for a more generic term,” a Commission official suggested today. “Foundation
models, of course, are part of the general purpose AI models.
</item 79>

<item 80>
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
</item 80>

<item 81>
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
</item 81>

<item 82>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
Whether this is capturing GPT-4 or Gemini or others we are not here now to assert — because also, in our framework, it is the companies that would have to come
and self assess what the amount of FLOPs or the computing capacity they have used. But, of course, if you read the scientific literature, many will point to
these numbers as being very much the most advanced models at the moment. We will see what the companies will assess because they’re the best placed to make this
assessment.”

“The rules have not been written keeping in mind certain companies,” they added. “They’ve really been written with the idea of defining the threshold — which,
by the way, may change because we have the possibility to be empowered to change this threshold on the basis of technological evolution. It could go up, it
could go down and we could also develop other benchmarks that in the future will be the more appropriate to benchmark the different moments.”
</item 82>

<item 83>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
“If established early on, they will result in an even playing field for all participants so there will not be an unfair competitive advantage for a few at the
expense of future innovators.

“The principles will ensure that intellectual property businesses can succeed in this new world alongside tech partners, further advancing investment and
expanding economic opportunities. “Music Business Worldwide
</item 83>

<item 84>
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
</item 84>

<item 85>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
In the meantime, music emerged out of [digitization] better and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll
likely be first.”

Asked when that sea change will occur, Kyncl said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is
increasing quality at a very fast pace.”

Kyncl drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.

“I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”
</item 85>

<item 86>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
So it will be worth scrutinizing the text that emerges, likely in January or February.

Additionally, while the full regulation won’t be up and running for a few years the EU will be pushing for GPAIs to abide by codes of practice in the meanwhile
— so AI giants will be under pressure to stick as close to the hard regulations coming down the pipe as possible, via the bloc’s AI Pact.

The EU AI Act itself likely won’t be in full force until some time in 2026 — given the final text must, once compiled (and translated into Member States’
languages), be affirmed by final votes in the parliament and Council, after which there’s a short period before the text of the law is published in the EU’s
Official Journal and another before it comes into force.

EU lawmakers have also agreed a phased approach to the Act’s compliance demands, with 24 months allowed before the high risk rules will apply for GPAIs.
</item 86>

<item 87>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Who owns the copyright on ChatGPT-created content or media?

The user who requested the input from ChatGPT is the copyright owner.

What lawsuits are there surrounding ChatGPT?

None specifically targeting ChatGPT. But OpenAI is involved in at least one lawsuit that has implications for AI systems trained on publicly available data,
which would touch on ChatGPT.

Are there issues regarding plagiarism with ChatGPT?

Yes. Text-generating AI models like ChatGPT have a tendency to regurgitate content from their training data.
</item 87>

<item 88>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
ChatGPT can also show bias in its responses, answering in sexist and racist, overtly Anglocentric ways — or regurgitating portions of the data that it was
trained on.

Lawyers have been sanctioned after using ChatGPT to assist in writing motions, discovering — too late — that ChatGPT invented fake lawsuit citations. And scores
of authors have sued OpenAI over the chatbot regurgitating portions of their work — and not receiving compensation for it.

So what comes next? What might ChatGPT’s second year hold, if not more of the same?

Interestingly — and fortunately — some of the more dire predictions about ChatGPT didn’t come to pass. Some researchers feared the chatbot would be used to
generate disinformation on a massive scale, while others sounded the alarm over ChatGPT’s phishing email-, spam- and malware-generating potential.
</item 88>

<item 89>
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
</item 89>

<item 90>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Legal cases build and break

We saw a fair number of lawsuits filed in 2023, but few saw any real movement, let alone success. Most suits over copyright and other missteps in the AI
industry are still pending. 2024 will see a lot of them fall by the wayside, as companies stonewall critical information like training data and methods, making
allegations like the use of thousands of copyrighted books difficult to prove in court.

This was only the beginning, however, and many of these lawsuits were filed essentially on principle. Though they may not succeed, they may crack the process
open far enough during testimony and discovery that companies would rather settle than have certain information come to light. 2024 will bring new lawsuits as
well, ones pertaining to misuse and abuse of AI, such as wrongful termination, bias in hiring and lending, and other areas where AI is being put to work without
a lot of thought.
</item 90>

<item 91>
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
</item 91>

<item 92>
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
</item 92>

<item 93>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
“At the moment, technologies are much better at watermarking videos and audio than watermarking text. But what we ask is the fact that this watermarking takes
place on the basis of state of the art technology — and then we expect, of course, that over time the technology will mature and will be as [good] as possible.”

GPAI model makers must also commit to respecting EU copyright rules, including complying with an existing machine readable opt-out from text and data mining
contained in the EU Copyright Directive — and a carve-out of the Act’s transparency requirements for open source GPAIs does not extend to cutting them loose
from the copyright obligations, with the Commission confirming the Copyright Directive will still apply on open source GPAIs.

As regards the AI Office, which will play a key role in setting risk classification thresholds for GPAIs, the Commission confirmed there’s no budget nor
headcount defined for the expert body as yet.
</item 93>

<item 94>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
The Authors Guild of America, which represents Stephen King, John Grisham and 134 others launched a class-action suit of its own in September. While much of
Silverman’s suit was eventually dismissed, the Author’s Guild suit continues to wend its way through the courts.

Select news outlets, on the other hand, proved far more amenable. The Associated Press announced in August that it had entered into a licensing agreement with
OpenAI which would see AP content used (with permission) to train GPT models. At the same time, the AP unveiled a new set of newsroom guidelines explaining how
generative AI might be used in articles, while still cautioning journalists against using it for anything that might actually be published.

ChatGPT itself didn’t seem too inclined to follow the rules. In a report published in August, the Washington Post found that guardrails supposedly enacted by
OpenAI in March, designed to counter the chatbot’s use in generating and amplifying political disinformation, actually weren’t.
</item 94>

<item 95>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Even if generative-Al-assisted outputs do compete with existing works that were used in training, or with future works by the authors of those works, the
pro-competitive nature of copying for the purpose of “knowledge harvesting” has traditionally been a reason to favor a holding of fair use , not a reason to
reject it.”

the output of that model might … Even if generative-Al-assisted outputs do compete with existing works that were used in training, or with future works by the
authors of those works, the pro-competitive nature of copying for the purpose of “knowledge harvesting” has traditionally been a reason to , not a reason to
reject it.” And then the kicker: “Any prohibition or limitation on the use of copyrighted materials for purposes of Al training would therefore undermine the
purpose of copyright and foreclose the many opportunities that come with this technology.”
</item 95>

<item 96>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
Whether Congress mobilizes in time to address mounting concerns around AI and creative industries or not, for its part the FTC does appear to be very tuned into
the technology’s risks — and the power of bringing voices together across industries.

“Art is fundamentally human,” FTC commissioner Rebecca Slaughter said.

“Humans may use technology to assist in creating art, but something cannot be art without human input. Technology is, by definition, not human . . . humans may
endeavor to make generative AI that is ever more intelligent, [but] it cannot and will not replace human creativity.”
</item 96>

<item 97>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
Although the still draft legislation remains the subject of negotiations by EU co-legislators. Add to that, once adopted the law is not expected to apply for a
couple of years so the Commission has turned to the Code to act as a stop-gap vehicle to encourage signatories to be proactive about deepfake disclosures it
expects to be mandatory in the future.

Following efforts to beef up the anti-disinformation Code last year the Commission also made it clear it would treat adherence to the non-legally binding Code
as a favorable signal for compliance with (hard legal) requirements hitting larger platforms which are subject to the Digital Services Act (DSA) — another major
piece of pan-EU digital regulation that obliges so called very-large-online-platforms (VLOPs) and search engines (VLOSEs) to assess and mitigate societal risks
attached to their algorithms (such as disinformation).
</item 97>

<item 98>
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
</item 98>

<item 99>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
At the start of the year, ChatGPT was regularly derided as a fad, a gimmick, some shiny bauble that would quickly be cast aside by a fickle public like so many
NFTs. Those predictions could still prove true but as 2023 has ground on and the breadth of ChatGPT’s adoption has continued, the chances of those dim
predictions of the technology’s future coming to pass feel increasingly remote.

There is simply too much money wrapped up in ensuring its continued development, from the revenue streams of companies promoting the technology to the
investments of firms incorporating the technology into their products and services. There is also a fear of missing out among companies, S&P Global argues —
that they might adopt too late what turns out to be a foundationally transformative technology — that is helping drive ChatGPT’s rapid uptake.

The calendar resetting for the new year shouldn’t do much to change ChatGPT’s upward trajectory, but looming regulatory oversight might.
</item 99>

<item 100>
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
</item 100>
=== END ITEM id=0848788caad8a0ca ===

=== ITEM id=0c6786d636b3b92c ===
Gold sentences and facts:
Sentence 0: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 0.0: Swift has shared that she has nothing to hide.
  Fact 0.1: Swift is in a new relationship with Kelce.
  Fact 0.2: Swift has shared that she has nothing to hide in her new relationship with Kelce.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: The podcast host, referred to as 'he', later invited the 'Anti-Hero' singer to one of his games at Arrowhead Stadium.
  Fact 1.1: The invitation was to a game at Arrowhead Stadium.
  Fact 1.2: The 'Anti-Hero' singer accepted the invitation.
  Fact 1.3: The 'Anti-Hero' singer famously accepted the invitation on 24 September.

Retrieved items:
<item 1>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 1>

<item 2>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 4>

<item 5>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Wang said he trusted Bankman-Fried’s judgment and didn’t pursue the issue further. Later, though, Alameda’s balance was more negative than FTT and the trading
revenue, and Wang had more conversations with Bankman-Fried about it. Wang said he knew at the time that Alameda was using FTX’s customers’ funds and that he
knew it was wrong.

We heard yesterday about the line of credit Alameda had on FTX, a staggering $65 billion. Today we saw it in the database. The line of credit didn’t start that
big — Wang testified that Bankman-Fried had asked him to increase it a few times because Alameda kept running into its limits. First it was “a few hundred
million,” then a billion, then even that wasn’t enough. It wasn’t clear how Wang arrived at $65 billion specifically, but he said he’d discussed the number with
Bankman-Fried. Other customers didn’t have the same privileges.
</item 5>

<item 6>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Additionally, he was promised that Alameda had no preferential treatment on the FTX platform, even though the hedge fund was one of its top traders. Huang said
he was concerned about FTX's lack of a board of directors, but he eventually invested anyway. During cross-examination, Huang said Paradigm pressed
Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts. — CNBC's Dawn Giel
contributed to this report.
</item 6>

<item 7>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
In June 2022, he submitted a report to Bankman-Fried on Signal that showed $8 billion in customer money held in an internal database tracking the cash wired to
an Alameda account called "fiat at ftx.com" was missing. Yedidia said he and Bankman-Fried spoke about it at the pickleball court at the resort in Nassau,
Bahamas. He asked his boss if things were OK. He was concerned because it "seemed like a lot of money" from FTX customers was at risk. "Sam said, we were
bulletproof last year. We aren't bulletproof this year," Yedidia testified. Yedidia said he asked when they would be bulletproof again. Bankman-Fried said he
wasn't sure, but it may be six months to three years. Yedidia said Bankman-Fried appeared "worried or nervous," which he said was atypical.
</item 8>

<item 9>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 9>

<item 10>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 10>
=== END ITEM id=0c6786d636b3b92c ===
