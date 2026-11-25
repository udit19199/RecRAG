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

=== ITEM id=10ab74fc1bdb970c ===
Gold sentences and facts:
Sentence 0: Kooker added later in the speech: “Music is a tremendous driver for AI technology, and AI technology presents a tremendous opportunity for the creative development of music.
  Fact 0.0: Kooker added later in the speech.
  Fact 0.1: Kooker said music is a tremendous driver for AI technology.
  Fact 0.2: Kooker said AI technology presents a tremendous opportunity for the creative development of music.
Sentence 1: Robert Kyncl, CEO of Warner Music Group – and an ex-high flyer at YouTube – commented: “With each major leap in technology, the music industry navigates a fresh set of challenges and opportunities.
  Fact 1.0: Robert Kyncl is the CEO of Warner Music Group.
  Fact 1.1: Robert Kyncl is an ex-high flyer at YouTube.
  Fact 1.2: Robert Kyncl commented that with each major leap in technology, the music industry navigates a fresh set of challenges and opportunities.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
So yeah, we are in that business.

“We are in the business of administering [rights], which means collecting revenue from thousands of platforms around the world; very complicated matters. But
we’re also in the business of ownership.”Music Business Worldwide
</item 2>

<item 3>
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
</item 3>

<item 4>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Sir Lucian Grainge, Chairman and CEO of Universal Music Group, said of UMG’s decision to offer a limited license to ‘Dream Track’: “We have a fundamental
responsibility to our artists to ensure the digital ecosystem evolves to protect them and their work against unauthorized exploitation, including by generative
AI platforms.

“At the same time, we must help artists achieve their greatest creative and commercial potential – in part by providing them access to the kind of opportunities
and cutting-edge creative tools made possible by AI.”

“We have a fundamental responsibility to our artists to ensure the digital ecosystem evolves to protect them and their work against unauthorized exploitation,
including by generative AI platforms.
</item 4>

<item 5>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
This all seemed to hint at a desire to build a ‘Content ID’-style system on YouTube to control and police content made by cloning artists’ voices;

publicly committed to three principles/pledges behind its development of music-based . These pledges promised in said tools. They promised that, amongst other
things, would help YouTube . This all seemed to hint at a desire to build a on YouTube to control and police content made by cloning artists’ voices; Then,
earlier this month, YouTube announced it was developing a specific system for music partners that allowed them to request the removal of content on its platform
that “mimics an artist’s unique singing or rapping voice”.

Evidently, all of this was enough for Universal Music Group and Warner Music Group to dip their toe into the ‘Dream Track’ experiment – though not without
caution.
</item 5>

<item 6>
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
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
In its own submission to the Copyright Office, Anthropic argued that using copyrighted works to train AI is, indeed, fair use.

The copying of copyrighted works to feed it into the AI algorithm “is merely an intermediate step, extracting unprotectable elements about the entire corpus of
works, in order to create new outputs,” Anthropic stated.

“In this way, the use of the original copyrighted work is non-expressive; that is, it is not re-using the copyrighted expression to communicate it to users… To
the extent copyrighted works are used in training data, it is for analysis (of statistical relationships between words and concepts) that is unrelated to any
expressive purpose of the work.”

“This sort of transformative use has been recognized as lawful in the past and should continue to be considered lawful in this case.”

That argument could yet come into play in Anthropic’s legal battle with Universal.
</item 8>

<item 9>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
Whether Congress mobilizes in time to address mounting concerns around AI and creative industries or not, for its part the FTC does appear to be very tuned into
the technology’s risks — and the power of bringing voices together across industries.

“Art is fundamentally human,” FTC commissioner Rebecca Slaughter said.

“Humans may use technology to assist in creating art, but something cannot be art without human input. Technology is, by definition, not human . . . humans may
endeavor to make generative AI that is ever more intelligent, [but] it cannot and will not replace human creativity.”
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 10>

<item 11>
‘Latin artists have been paving the way for deals that offer more control, transparency, and better economics.’ (Music Business Worldwide,
2023-12-07T20:32:11+00:00)
We are in a world where music discovery and consumption are now primarily digital, and artists have now understood that. They look for partners who get it too
and whose expertise will allow them to leverage digital platforms to reach new and existing audiences wherever they are.

“Artists today are also much more conscious of their career trajectory and look for better deals, with more freedom, more transparency, and more options.”

This is especially the case in Spanish-speaking markets, where music fans are amongst the most engaged audiences in the world, whether on digital platforms or
on social media. Music content is consumed with more frequency, in different formats, across multiple platforms and through different aspects related to the
track.

Beyond that, artists today are also much more conscious of their career trajectory and look for better deals, with more freedom, more transparency, and more
options. They want to keep ownership of their masters, and get better economics.
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
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
On the surface, Microsoft is currently looking like the biggest winner amidst the OpenAI turmoil, as it was already apparently looking to reduce its reliance on
OpenAI even though it remains once of its major shareholders. But Facebook’s parent Meta could also stand to benefit, as businesses pursue multi-modal
strategies or models with a more “open” ethos embedded.

“Open source today offers a wide variety of models for companies to essentially diversify,” Ceze added. “By doing so, these startups can quickly pivot and
minimize risk. There is also a major upside — many of these models already outperform the likes of OpenAI’s in terms [of] price-performance and speed.”

A leaked internal memo from Google earlier this year seemed to express fears that despite the huge advances made by proprietary LLM models from the likes of
OpenAI, open source AI would ultimately trump them all. “We have no moat, and neither does OpenAI,” the document noted.
</item 13>

<item 14>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
NMPA

The submission adds: “Even more alarming is that we do not know how long the window is to act before it is too late. We therefore implore the [US Copyright]
Office to support proactive protections for human creators and, where there is uncertainty, to err on the side of protecting human creators.”

That position might not come as a complete surprise from the NMPA, which earlier this year joined the fledgling Human Artistry Campaign, whose goal – in
broadest terms – is to ensure that AI doesn’t replace or “erode” human culture.

Yet the NMPA’s submission is hardly a Luddite diatribe against high tech; rather, it advocates for regulations and principles that it sees as the right approach
to ensuring that the interests of musical artists – and music rights holders – don’t end up subjugated in the frenzy to build our brave new AI-powered world.

Here are some of the key arguments the MNPA made in its submission…

1.
</item 14>

<item 15>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
(Example from UMG’s filing: “The wholesale appropriation of UMG’s enormous catalog of copyright-protected sound recordings and musical compositions to build
multibillion commercial enterprises [in AI] is anything but fair use.”)

As well as part-explaining Sony Music’s refusal to be included in ‘Dream Track’ thus far, Google’s USCO filing also perhaps explains the careful words selected
by Robert Kyncl and especially Sir Lucian Grainge in their respective comments endorsing YouTube’s AI experiment.

(Grainge’s statement, remember, began: “We have a fundamental responsibility to our artists to ensure the digital ecosystem evolves to protect them and their
work against unauthorized exploitation, including by generative AI platforms…”)

One interesting final observation on the clutch of artists who agreed, with their record companies, to enter the YouTube ‘Dream Track’ trial?

They’re stars, for sure.
</item 15>

<item 16>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
The genie is not going back.”

But he cautioned that the technology to develop a Content ID-style system for is “not yet developed, but… people are working on that.”

{Believe CEO Denis Ladegallerie might disagree. In comments earlier this year, Ladegallerie said that the tools to recognize and flag AI-generated content are
just about ready to go, and he expects to see them implemented in the coming few quarters.)

2) The streaming audio business won’t experience the same problems currently hitting streaming video

The streaming video business is going through tough times.

Netflix experienced its first-ever decline in subscriptions last year (though it has since roared back to health); the subscriber count of Disney+ appears to be
in freefall; and talk is growing about ongoing consolidation of the many streaming video services that have come online over the past several years.

Asked if the audio streaming business is likely to go through something similar, Kyncl’s answer was unequivocal.

“No.
</item 16>

<item 17>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
You can read the full Google USCO submission here, but below are a few of the choicest sections on this topic:

“The doctrine of fair use [within existing US copyright law] provides that copying for a new and different purpose is permitted without authorization where — as
with training Al systems — the secondary use is transformative and does not substitute for the copyrighted work .”

[within existing US copyright law] provides that copying for a new and different purpose is permitted without authorization where — as with — the secondary use
is transformative and .” “If [AI] training could be accomplished without the creation of copies, there would be no copyright questions here. Indeed that act of
“knowledge harvesting”… like the act of reading a book and learning the facts and ideas within it , would not only be non-infringing , it would further the very
purpose of copyright law.
</item 17>

<item 18>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
That argument could yet come into play in Anthropic’s legal battle with Universal.

And given the high stakes involved – for the music industry, for the tech industry and for the shape of culture going forward – that case, along with the US
Copyright Office’s study of AI and copyright, are worth keeping a close eye on.

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 18>

<item 19>
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
</item 19>

<item 20>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
These products range from tools for “creative or marketing assistance”, to tools that “potentially give us the ability to better protect artist content or find
it when used in an unauthorized fashion” as well as “brand new products that have never been launched before”.

Kooker said that some of these conversations “also include potential equity investments which would accelerate the development of these companies”.

The Sony Music exec cited, as an example, a recent generative AI-powered project around a reissue and remix of an album. Although Kooker didn’t name it
specifically, SME did recently announce a generative AI project with The Orb and David Gilmour, which was a partnership between Sony Music Entertainment, Legacy
Recordings AI company Vermillio

“These artists are known for their cutting-edge experimentation in music,” said Kooker.
</item 20>

<item 21>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
Here are some of the key arguments the MNPA made in its submission…

1. Works created principally through AI should not be copyrightable

In the view of the NMPA, simply giving a prompt to a AI music generator – “compose a power ballad about falling in love” – shouldn’t be enough to create a
copyrightable work.

“Policies surrounding copyrightability of AI-generated content will either serve to incentivize continued investment and effort into human creativity or to
disincentivize it,” states the submission, which can be read in full here.

“The law should never be such that human creators stand to gain more from repeatedly clicking a button to generate massive amounts of AI-produced materials than
from putting their hearts, souls, experiences, skills, talents, and emotions into expressive works of art.”
</item 21>

<item 22>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
A Final thought…

As mentioned in the opening sentence of this article, there is plenty of drama surrounding HSF right now – from the ‘discontinuation vote’ we saw at the company
in October, through to a recent lawsuit launched against Hipgnosis Songs Fund by a former business partner of Merck Mercuriadis.

Yet, amongst all of the noise, what will likely most interest HSF’s investors (including the likes of Investec Wealth & Investment, Aviva Investors, and
BlackRock) is the ongoing underlying commercial performance of the firm’s copyrights.

How are Hipgnosis Songs Fund’s assets performing vs. the market? How is the company’s ‘premium-only’ acquisition strategy holding up against its promise of
growth “uncorrelated” to other industries? And what does all of this suggest about the ongoing growth potential of the HSF portfolio vs. the multiples that
Merck Mercuriadis paid to acquire it?
</item 22>

<item 23>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
The goal is that AI tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

“It’s really interesting from a music perspective, specifically, because . . . musicians are perhaps more familiar with the idea of AI,” Holland said.
“Musicians in general are more familiar with things like music production software, and AI tools like MIDI drum loops . . . so I think that there is a certain
amount of more progressive learning from them, when it comes to technology, and its ability to make their music better.”

When it comes to art and AI, the conversation is complicated, to say the least. Musicians are nervous about industry giants copyrighting AI music and cutting
them out of the process. Major record labels are worried about AI models training on their catalogues and stealing a slice of their considerable pie.
</item 23>

<item 24>
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
</item 24>

<item 25>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists;

and jointly announced that the video platform was launching an – a program by which new tools and innovations will be developed at YouTube in close conjunction
with artists and the music biz. To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists; Also in
August, YouTube publicly committed to three principles/pledges behind its development of music-based generative AI tools . These pledges promised “appropriate
protections… for music partners who decide to participate” in said tools. They promised “invest[ment] in the AI-powered technology” that, amongst other things,
would help YouTube “protect our community of viewers, creators, artists and songwriters… [from] trademark and copyright abuse” .
</item 25>

<item 26>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
The law should treat a musical work created entirely by AI the same way it treats works that are in the public domain, the NMPA argues – not copyrightable, and
free for anyone to use as they see fit, whether for commercial or other purposes.

But the submission draws a clear distinction between music generated by AI, and music generated with the help of AI.

“Where creators use AI technology as a tool in the creative process to make works that represent their original authorship, their works should be entitled to
protection under copyright law,” the NMPA states.

“Creators that use AI to refine, recast, or modify, or to create new derivative works based on their preexisting works, may also have legitimate claims of
authorship over the resulting work in some circumstances.”

“The law should never be such that human creators stand to gain more from repeatedly clicking a button to generate massive amounts of AI-produced materials than
from putting their hearts, souls, experiences, skills, talents, and emotions into expressive works of art.”
</item 26>

<item 27>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
“If established early on, they will result in an even playing field for all participants so there will not be an unfair competitive advantage for a few at the
expense of future innovators.

“The principles will ensure that intellectual property businesses can succeed in this new world alongside tech partners, further advancing investment and
expanding economic opportunities. “Music Business Worldwide
</item 27>

<item 28>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Human authorship is a bedrock requirement of copyright.”

2. AI developers should be required to license materials and keep records of the materials they use for training

Many key players in the music industry and beyond have been adamant that using copyrighted works to train AI algorithms without authorization is a copyright
infringement, and a number of lawsuits have been launched against AI companies on these grounds.

One such lawsuit was filed by Universal Music Group, Concord Music Group and ABKCO against AI developer Anthropic, seeking potentially tens of millions of
dollars for the alleged “systematic and widespread infringement of their copyrighted song lyrics.”

But that doesn’t mean key music industry players don’t want their materials ever used to train AI – they just want AI developers to license that music, not
unlike how streaming services, bars and restaurants or movie producers license music.
</item 28>

<item 29>
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
</item 29>

<item 30>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Licensing musical works before training use is both required and practicable,” the NMPA’s submission states, adding that “there are well developed processes in
place for technology ventures to obtain free market licenses on a large scale.”

And indeed it would have to be on a “large scale,” given that AI algorithms such as large language models are trained on countless millions or even billions of
pieces of digital data.

“Transparency and recordkeeping requirements are needed to disincentivize infringing activity and to support enforcement activities by copyright owners.” NMPA

The NMPA notes that work on licensing “is already underway,” pointing to news reports earlier this year that UMG, along with Warner Music Group, are in talks
with Google to secure licenses for artists’ vocals and musical melodies to create AI-generated songs.
</item 30>

<item 31>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Do you expect startups to build striated solutions for individual layers of the LLM stack, or pursue a more vertical approach?

While there are startups specializing in parts of the stack (like Pinecone), Valor’s focus is on applied AI, which we define as AI that is solving a customer
problem. Saile.ai is a good example — it uses AI to generate closeable leads for the Fortune 500. Or Funding U using its own trained dataset to create a more
useful credit risk score. Or Allelica, using AI on treatment solutions applied to individual DNA to find the best medical treatment for you personally in a
given situation.

Companies like Datadog are building products to support the expanding AI market, including releasing an LLM observability tool. Will efforts like what Datadog
has built (and similar output from large/incumbent tech powers) curtail the market area where startups can build and compete?

Tools like Datadog can only help the acceptance of AI tools if they succeed in monitoring AI performance bottlenecks.
</item 31>

<item 32>
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
</item 32>

<item 33>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
Robert Kyncl, Warner Music Group

In Kyncl’s view, the solutions put in place to address UGC form a “blueprint” than can be used to address AI-generated content on platforms as well.

“When YouTube was formed… people started to upload content, including copyrighted material, which, obviously, put YouTube into hot water with lots of different
copyright holders. And I had the privilege of working through a lot of that and fix it up.

“But we made we made a very important decision, which was to go above and beyond the law, and build a fingerprinting software that allowed us to track the
copyright on our platform, and then have commercial relationship[s] with copyright holders to send them the money. Out of that we built a multi-billion-dollar
business, which now is a multi-billion-dollar business per year. And it was an incredible new revenue stream for everyone. AI is that with new super tools.”
</item 33>

<item 34>
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
</item 34>

<item 35>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
As creative industries grapple with AI’s explosion into every artistic medium at once, separate calls from artists warning the world to take action before it’s
too late are starting to converge. From fake Drake songs to stylized Instagram profile pictures, art conjured with newly sophisticated AI tools is suddenly
ubiquitous — and so are conversations about how to rein in the technology before it does irrevocable harm to creative communities.

This week, digital rights organization Fight for the Future partnered with music industry labor group United Musicians and Allied Workers to launch
#AIdayofaction, a campaign that calls on Congress to block corporations from obtaining copyrights on music and other art made with AI.

The idea is that by preventing industry behemoths like major record labels, for example, from copyrighting music made with the assistance of AI, those companies
will be forced to keep looping humans into the creative process. But those same concerns — and the same potential strategies for pushing back against the
onslaught of AI — exist across creative industries.
</item 35>

<item 36>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
They’re stars, for sure. But in both the case of Universal Music Group and Warner Music Group, they’re – currently anyway – not the biggest megastars on either
company’s books.

You might not expect, at this early stage, for the likes of UMG to put forward Drake, Taylor Swift, or The Beatles – or for WMG to put forward Ed Sheeran or Dua
Lipa – to be guinea pigs in any early AI music experiments, even if they’re run by ‘friends’ of the music industry like YouTube.

You might, then, also understandably wonder if there have been significant licensing payments made by YouTube to UMG and WMG for the right to play with the AI
vocals of artists who have signed off on ‘Dream Track’ thus far. And, if so, if these payments are being placed against any as-yet-unrecouped advances that some
of these artists may have on their label accounts.
</item 36>

<item 37>
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
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
“No. Music is much more resilient. So first, I think what happened in the last 15 years is incredible. They had literally zero people in the subscription model
and now we have 700 million people in the world in the premium experience… which is incredible.

“And I think really credit goes to Daniel Ek for forging the path for everybody. And then companies like Apple and YouTube and Amazon following and building up
the [music] business. It’s pretty incredible what has happened.”

“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.” Robert Kyncl, Warner Music Group

Kyncl added: “I think the opportunity ahead of us is twofold. One, the continued growth in emerging markets, and also the GDPs of those countries will be rising
at the same time. So there’s lots of growth there.
</item 39>

<item 40>
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
</item 40>

<item 41>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
At the same time, we must help artists achieve their greatest creative and commercial potential…” Sir Lucian Grainge on UMG’s decision to license YouTube’s
‘Dream Track’ experiment

Robert Kyncl, CEO of Warner Music Group – and an ex-high flyer at YouTube – commented: “With each major leap in technology, the music industry navigates a fresh
set of challenges and opportunities.

“It’s not always the case that, from the outset, tech platforms partner with artists, songwriters, labels, and publishers to experiment, iterate, and find
possible solutions.

“YouTube is taking a collaborative approach with this Beta. These artists are being offered the choice to lean in, and we’re pleased to experiment and find out
what the creators come up with.”

Why, then, did Sony Music not join its rivals in granting YouTube and ‘Dream Track’ access to any of its active artists’ licensed voices?
</item 41>

<item 42>
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
</item 42>

<item 43>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
“At the moment, technologies are much better at watermarking videos and audio than watermarking text. But what we ask is the fact that this watermarking takes
place on the basis of state of the art technology — and then we expect, of course, that over time the technology will mature and will be as [good] as possible.”

GPAI model makers must also commit to respecting EU copyright rules, including complying with an existing machine readable opt-out from text and data mining
contained in the EU Copyright Directive — and a carve-out of the Act’s transparency requirements for open source GPAIs does not extend to cutting them loose
from the copyright obligations, with the Commission confirming the Copyright Directive will still apply on open source GPAIs.

As regards the AI Office, which will play a key role in setting risk classification thresholds for GPAIs, the Commission confirmed there’s no budget nor
headcount defined for the expert body as yet.
</item 43>

<item 44>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
It’s a co-founder with Meta after all. I’d venture to guess more exposure for its burgeoning generative AI platform. IBM’s most recent earnings were boosted by
enterprises’ interest in generative AI, but the company has stiff competition in Microsoft and OpenAI (and to a lesser extent Google), which are jointly
developing enterprise-focused AI services that directly compete with IBM’s.

I’ve asked IBM’s PR, which first informed me of the AI Alliance’s founding, about the curious omissions from the early membership, like Stanford (which has a
prominent AI research lab, Stanford HAI), MIT (which is at the forefront of robotics research) and high-profile AI startups like Anthropic, Cohere and Adept. A
press rep didn’t respond as of publication time. But the same philosophical differences that kept Google and Microsoft away likely were at play; I’d wager it’s
no accident that Anthropic, Cohere and Adept have relatively few open source AI projects to their names.
</item 44>

<item 45>
Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn (Music Business Worldwide,
2023-11-29T20:55:14+00:00)
Their stance is costing the world music economy hundreds of millions a year.” Helen Smith, IMPALA

In its 2020 ruling, the CJEU made it clear that the EU legislature has the power to change this by rewriting the directive governing copyright payments.

“It is for the EU legislature alone, which has exclusive external competence in the matter, to decide on such a limitation,” the court stated following the
ruling.

The artists’ groups noted in their statement on Tuesday that the European Commission “has on several occasions acknowledged concerns about the impact of the
ruling and made clear its intention to find a balanced solution. But a solution has yet to be proposed.”

“It is the EU’s responsibility to prevent European artists and producers losing millions every year to the USA, which has chosen not to protect these rights. A
proposal is needed now to restore legal certainty, safeguard cultural diversity and European sovereignty,” Smith said in the statement.
</item 45>

<item 46>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Within the enterprise sector, there’s a clear recognition of the value of AI. However, many lack the internal capabilities to develop AI solutions. This gap
presents a significant opportunity for startups specializing in AI to engage with enterprise clients. As the business landscape matures, proficiency in
leveraging AI is becoming a strategic imperative.

McKinsey reports that generative AI alone can add up to $4.4 trillion in value across industries through writing code, analyzing consumer trends, personalizing
customer service, improving operating efficiencies, and more. Ninety-four percent of business leaders agree AI will be critical to all businesses’ success over
the next five years, and total global spending on AI is expected to reach $154 billion by the end of this year, a 27% increase from 2022. The next three years
are also expected to see a compound annual growth rate of 27% — the annual AI spending in 2026 will be over $300 billion.
</item 46>

<item 47>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
“It’s funny because if you’ve talked to musicians who have these concerns, they say, ‘Well, authors have been very quiet.’ If you talk to others about these
concerns, they say, ‘Well, musicians and photographers don’t seem to care at all,’” Fight for the Future Campaigns and Communications director Lia Holland told
TechCrunch. “So part of it also is that the different creative fields, when it comes to this sort of work, are a little bit siloed.”

“That was another intent with our launching this effort with the day of action, to try to illustrate how these are common concerns that are shared across
artistic mediums. And to create an organizing point . . . because when artists of different mediums move together they have a lot more power.”

The campaign targets potential corporate abuse of AI technology, but it’s realistic about the ways that musicians and some other creatives could benefit on an
individual level from automating parts of their work.
</item 47>

<item 48>
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
</item 48>

<item 49>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
NMPA

In this regard, US legal precedent is – at least so far – on the NMPA’s side.

In a recent court case, an individual by the name of Stephen Thaler, who owns a computer system he calls the “Creativity Machine,” had a copyright application
for an AI-generated “painting” rejected by the USCO on the grounds that the Creativity Machine had created it.

In a ruling this past August, a US District Court in Washington, D.C., sided with the Copyright Office.

“Copyright has never stretched so far… as to protect works generated by new forms of technology operating absent any guiding human hand, as plaintiff urges
here,” Judge Beryl A. Howell commented. “Human authorship is a bedrock requirement of copyright.”

2.
</item 49>

<item 50>
The new, ‘efficient’ Spotify has a very different approach to podcasting (The Verge, 2023-10-25T16:10:29+00:00)
It doesn’t mean that Spotify’s contributions to podcasting will be insignificant — as one audio industry professional said to me, the translation thing could be
a total game-changer for the medium (if it’s not a complete dud). But we should not expect those contributions to come in the form of original content.

The main question I am left with after these results is where Spotify’s licensing deals fit in. I wish during the call that someone had asked Ek about the plan
for big-name talent like Joe Rogan, Alex Cooper, and Dax Shepard as their deals come up for renewal. On one hand, they have the scale (Rogan especially). On the
other, they cost a chunk of cash (... Rogan especially). I will have more on this later this week, but it will be interesting to see how much Spotify and its
investors can stomach spending on the biggest names in podcasting.

Wondery puts its podcasts on TV

That’s certainly one way to get podcasts to the masses.
</item 50>

<item 51>
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
</item 51>

<item 52>
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
</item 52>

<item 53>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
[Motown was spun out in early 2021, then returned to CMG in early 2023].

Second, we took a hard look at every aspect of our business. We revamped our entire organizational structure, reset our value system, and even replaced certain
terminology that was outdated. All to meet the needs of the modern music business. Arjun was key to this process; his expertise really drove a new dynamic in
how marketing, creative, international, digital and commerce now interact.

“Today, literally every plan is tailored for a specific artist or project — not pulled from a well-worn playbook — and our focus is on finding and creating
opportunities that will have an impact, no matter where an artist is in their development.”
</item 53>

<item 54>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
So there’s lots of growth there. And then there is the price elasticity optimization in mature markets.”

That last point is a reference to the price hikes seen among all the major music streaming services over the past year or so, and the growing belief within the
music industry that streaming price hikes will be sustained – as evidenced recently by Deezer’s second price hike in a year.

Credit: QuiteSimplyStock/Shutterstock 3) The hot music rights acquisitions space is ‘likely slowing down a bit’

The second half of 2022 saw a notable slowdown in music rights acquisitions, a phenomenon that some blamed on higher interest rates and the reduced liquidity
that came with them.

Although there has been something of a rebound this year, in Kyncl’s view, there is still a slowdown in this space – though that might be good for Warner Music
Group.

“I think what happened was there was an opening where artists and songwriters were suddenly open to selling catalogs.
</item 54>

<item 55>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
However, enforcing copyright when it comes to AI algorithms could be exceedingly difficult for rights holders, a point the NMPA makes repeatedly in its
submission: “Reverse-engineering” an AI algorithm to prove it was trained on content to which the developer had no right is no easy task, save for when that AI
produces a work that is substantially similar to the original (as in the case of the lawsuits currently pending against AI companies).

So the NMPA is arguing for a new regulation: That AI developers be required to keep records of what data their algorithms ingested.

“Transparency and recordkeeping requirements are needed to disincentivize infringing activity and to support enforcement activities by copyright owners,” the
NMPA’s submission states.

“The primary parties who should be required to maintain records under a transparency and recordkeeping scheme are developers of AI models, those who use
existing AI models to develop new AI tools and those who broker datasets for use in AI training.”
</item 55>

<item 56>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
What’s happened?

MBW Explains is a series of analytical features in which we explore the context behind major music industry talking points – and suggest what might happen next.
MBW Explains is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

As we said last week: Welcome to the future.

Seven days ago (November 16), YouTube unveiled a revolutionary new experiment – ‘Dream Track’ – enabling creators to clone the vocals, via AI tech and with
official consent, of well-known stars.

Charlie Puth’s voice is on there. John Legend’s voice is on there. Demi Lovato’s voice is on there.

In fact, the voices of a number of well-known recording artists signed to Warner Music Group and Universal Music Group – or labels within each major – are on
there.
</item 56>

<item 57>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
But eight years later, the argument between the two men seems prescient. The question of whether AI will elevate the world or destroy it – or at least inflict
grave damage – has framed an ongoing debate among Silicon Valley founders, chatbot users, academics, legislators and regulators about whether the technology
should be controlled or set free. Loading That debate has pitted some of the world’s richest men against one another: Musk, Page, Mark Zuckerberg of Meta, tech
investor Peter Thiel, Satya Nadella of Microsoft and Sam Altman of OpenAI. All have fought for a piece of the business – which one day could be worth trillions
of dollars – and the power to shape it. At the heart of this competition is a brain-stretching paradox. The people who say they are most worried about AI are
among the most determined to create it and enjoy its riches. They have justified their ambition with their strong belief that they alone can keep AI from
endangering Earth. Musk and Page stopped speaking soon after the party that summer.
</item 57>

<item 58>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Kooker suggested that the “more dire outputs of early generative AI technology are deep fakes and unauthorized voice clones” of artists and that in the United
States, “artists are not adequately protected from these deepfakes”.

“An artist literally makes their livelihood from their voice,” he said. “Deep fakes intentionally exploit an artist’s talent and reputation to steal that income
stream.

“Every stream of a deep fake takes streams and royalty payments away from the legitimate artist. Deepfakes are also misleading and confusing to consumers and
music fans who are not typically interested in supporting fake versions of their favorite artist.”

“Every stream of a deep fake takes streams and royalty payments away from the legitimate artist.” Dennis Kooker, Sony Music

Kooker reported that to date, Sony Music Entertainment has sent “close to 10,000 takedowns to a variety of platforms hosting unauthorized deepfakes that SME
artists asked us to take down.”
</item 58>

<item 59>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
Spotify erased thousands of AI-crafted songs from its platform but also recently globally launched an AI-powered DJ that curates music for listeners while
talking to them in a synthetic voice.

“The training of generative AI using our artists’ music . . . begs the question as to which side of history all stakeholders in the music ecosystem want to be
on: the side of artists, fans and human creative expression, or on the side of deep fakes, fraud and denying artists their due compensation,” Universal Music
Group said after a song using AI to imitate Drake and The Weeknd, two of its artists, went viral.

These same conversations and contradictions are manifesting across creative industries, but artists themselves don’t always have a seat at the table.
Independent artists in particular are learning that their voices resonate louder when coming together across disciplines to push back against what Holland
describes as an “extraordinary spectrum of exploitation” that leverages their work.
</item 59>

<item 60>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
At that time, he said, many predicted ad agencies were headed for the scrap-heap thanks to the arrival of Google’s AdWords, which allowed any business to set up
a customized ad campaign on Google services.

Yet ad agencies didn’t go extinct. Today, “they’re the largest customers of companies like Google,” Kyncl told the audience at the Code Conference.

“In the year 2000, when AdWords came out, everybody was predicting their deaths because all the brands [could] just go direct to AdWords and buy ads… However,
what happened was lots of different platforms emerged. Lots of ad tech emerged, complexity increased exponentially, and brands needed help with that, and the
agencies provided that value.

“And I see a very close analogy [between that and] our business. Music is incredibly broadly distributed. Everybody on earth listens to music. We’re on every
platform. And the complexity is high.
</item 60>

<item 61>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
You cannot escape conversations about AI no matter how far or fast you run. Hyperbole abounds around what current AI tech will be able to do (revolutionize
every industry!) and what current AI tech will be able to do (take over the world!). Closer to the ground, TechCrunch+ is working to understand where startups
might find footholds in the market by levering large language models (LLMs), a recent and impactful new method of creating artificially intelligent software.

How AI will play in startup land is not a new topic of conversation. A few years back, one venture firm asked how AI-focused startups would monetize and whether
they would suffer from impaired margins due to costs relating to running models on behalf of customers. That conversation died down, only to come roaring back
in recent quarters as it became clear that while LLM technology is quickly advancing, it’s hardly cheap to run in its present form.

But costs are only one area where we have unanswered questions.
</item 61>

<item 62>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.
</item 62>

<item 63>
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
</item 63>

<item 64>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
First, a quick reminder on the test US courts use to determine fair use, which has four factors:

The purpose and character of the use – is the use of the copyrighted work for educational purposes or for commercial purposes?

The nature of the copyrighted work – whether or not the work is particularly creative and original.

The amount and substantiality of the portion taken – just how much of a copyrighted work was used without permission?

The effect of the use on the potential market for, or value of, the copyrighted work.

The NMPA’s submission argues that the way AI developers use materials to train AI would fail every one of these tests for determining fair use.

AI algorithms are not used for non-profit or educational purposes, as “the training of AI models is fundamentally a commercial endeavor, especially in the case
of generative AI,” the NMPA stated.

“AI can be used to generate works that compete in the marketplace with the copied works, thereby reducing revenue from existing licensing markets.”
</item 64>

<item 65>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

Legislators and regulators around the world are grappling with how to address the many issues that have emerged since AI technology “went mainstream” about a
year ago.

In the US, one of the key efforts on that front is being spearheaded by the US Copyright Office, which this past summer issued a call for submissions on the
issue of AI and copyright. The goal is for the USCO to put together a study to ”help assess whether legislative or regulatory steps in this area are warranted.”

Numerous businesses involved in AI, and those with substantial copyright holdings, have submitted their thoughts on the issue, including Universal Music Group
and AI developer Anthropic.
</item 65>

<item 66>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
But what was maybe more surprising – for a techie like Kyncl, anyway – was his rational, level-headed explanation for why traditional music labels continue to
be relevant, and continue to thrive, in the current music ecosystem.

With the arrival of digital platforms, including open ones like YouTube and SoundCloud – and with the proliferation of indie music distribution platforms like
DistroKid and TikTok’s SoundOn – it’s inevitable that music biz observers from the world of tech like to speculating about the possibility that music labels
could, in the long-term, be headed for irrelevance.

In a world where anyone can record music using a desktop audio workstation (DAW) and upload it direct to YouTube, or to Spotify via a distribution platform,
labels have little left to do, the argument goes.

Kyncl offered strong a counterpoint to this argument at Code Media, drawing a parallel between music labels today and the advertising industry a decade or two
ago.
</item 66>

<item 67>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Diversity is a superpower. It’s an ongoing process that I’ve put a lot of focus on, along with doing what I can to increase the number of women at our company.
There just aren’t enough of us, particularly within the most senior ranks at the labels.

UMG has [Nashville Chair & CEO] Cindy Mabe and me running standalone label groups, and I know that Lucian agrees with me that there needs to be more.

I will continue to do what I can to increase representation at Capitol Music Group and within the music business at large.Music Business Worldwide
</item 67>

<item 68>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
As reported by MBW last week, Universal Music Group submitted its response to the USCO. Unsurprisingly, amongst the highlights of UMG’s submission was the
rejection of the notion that using copyrighted material to train LLMs is “fair use”.

In his speech on Wednesday, Kooker commented directly on the flurry of submissions to the USCO from the tech industry at the end of October.

In what may have been a nod to Google’s filing and “fair use” position, Kooker told Senators that “based on recent Copyright Office filings it is clear that the
technology industry and speculative financial investors would like governments to believe in a very distorted view of copyright”.

He added: “[That view is] one in which music is considered fair use for training purposes and in which certain companies are permitted to appropriate the entire
value produced by the creative sector without permission, and to build huge businesses based on it without paying anything to the creators concerned.”
</item 68>

<item 69>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
Sony has been subtly expanding its roster — more subtly than Microsoft, at least — over the past three years, and it’s also made heavy investments in studios
like Epic Games and FromSoftware.

With this lineup, Sony is betting heavily on ongoing games, and it has 12 live-service titles in production right now, on top of Bungie’s Destiny franchise.
These include Haven’s Fairgame$ and a multiplayer Horizon title from Guerilla.

“By expanding to PC and mobile, and… also to live services, we have the opportunity to move from a situation of being present in a very narrow segment of the
overall gaming software market, to being present pretty much everywhere," Sony Interactive Entertainment president and CEO Jim Ryan said in 2022.

For the companies at the top, total domination is the goal.

Even still, Tencent is bigger than both Sony and Microsoft.
</item 69>

<item 70>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
More broadly, you might ponder a bigger question.

When the day comes that YouTube asks the world’s biggest superstars to wholeheartedly embrace ‘Dream Track’, will Google’s recent US Copyright Office filing
give said superstars – and their record companies – pause for thought?

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 70>

<item 71>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Today, literally every plan at CMG is tailored for a specific artist or project — not pulled from a well-worn playbook — and our focus is on finding and
creating opportunities that will have an impact, no matter where an artist is in their development. The artist’s vision still lies at the heart of every plan,
but we really listen to the audience, which is smarter and savvier than ever, to determine if our marketing direction is working and if our artists are properly
engaged with their audience.

We strive to be ideas-driven and have honest constructive conversations with our artists and each other. We run our business efficiently, and are constantly
thinking about where we find real value for our artists.

Were there any other things at the top of your ‘to-do’ list?

Prioritization.
</item 71>

<item 72>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
which they were not open to before. And they just created this incredible tidal wave. And these things happen in waves, ebb and flow.

“So I think it likely is slowing down a little bit, but that’s okay. Because it makes the multiples come down a little bit more.”

“I think what happened in the last 15 years is incredible. They had literally zero people in the subscription model and now we have 700 million people in the
world in the premium experience.” Robert Kyncl, Warner Music Group

In other words, Warner Music can go out and buy more if the multiples are low enough?

“Of course. We are in the business of buying catalogs, obviously. We have the publishing catalog of David Bowie, for instance, which we bought a few years ago.
So yeah, we are in that business.
</item 72>

<item 73>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Music is music, and genres shouldn’t be defined by color. In fact, this business really needs to change its definition of “pop” music. There seems to be real
confusion there; we should be calling any music that streams in volume, “pop” music.

Our philosophy is that great ideas transcend genre. We have experts in every “genre” but we all share, collaborate and respectfully challenge each other. In
fact, I don’t even like the word, “genre.” Let’s change it!

Be honest: When you first heard “Unholy” – from that well-known smooth soul singer Sam Smith – what was your reaction? Did you know it was a smash hit?

Sam drove this record, from its creation through the decision for it to be the first single.
</item 73>

<item 74>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
We’re on every platform. And the complexity is high. And the more people can upload content, and the more people can be heard, the greater the noise, which
means it’s harder to cut through the noise and sustain a career.”

But a label’s value is not just about building an artist’s career – it’s about sustaining it once that career is established as well, Kyncl explained.

“When they become established, it is also hard to remain at the top,” Kyncl said. “So their goals change. And again, we’re there to help with that.”

He added later that if you’re a music artist, “you need a team. You need an army behind you…. if you want a sustainable career, with repeatability and success.”
</item 74>

<item 75>
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
</item 75>

<item 76>
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
</item 76>

<item 77>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
And for all their talk of openness, IBM and Meta aren’t exactly the poster children for the future that the Alliance’s release depicts — casting doubt on their
sincerity.

Perhaps I’m wrong and the AI Alliance will be a smash success. Or perhaps it’ll crumble under mistrust and its own bureaucracy. We’ll see; time will tell.
</item 77>

<item 78>
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
</item 78>

<item 79>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
That in and of itself is probably still largely unexplored territory that will see a lot of change and maturing in the next few years. One key aspect there
might be cost monitoring as well since companies like OpenAI charge largely “by the token,” which is a very different metric than most cloud computing.

What can be done to ensure industry-specific startups that tune generative AI models for a specific niche will prove defensible?
</item 79>

<item 80>
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
</item 80>

<item 81>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Especially when SME has shown its own willingness to experiment with generative AI music platforms this week via this announcement about a project with David
Gilmour, The Orb, and AI company Vermillio?

Sources close to Sony Music’s HQ in New York have suggested to MBW that the company is taking an “artist-led” approach to its experiments with any AI platform
that can manipulate the work (or voice) of its roster.

In the case of YouTube’s ‘Dream Track‘, MBW understands, SME first offered the opportunity to participate to a number of its artists, but this group of acts did
not show enthusiasm about participating.

(If one of these artists was Bad Bunny, signed to Sony’s The Orchard, you can see why: The Puerto Rican superstar expressed fury the other week about a popular
track on TikTok that had cloned his vocal stylings via AI.)
</item 81>

<item 82>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

In recent months, Warner Music Group CEO Robert Kyncl has been keeping a close eye on AI in the music industry – not surprising, given that he came to the music
biz from ‘Big Tech’, namely YouTube, where he was Chief Business Officer.

Nor is it surprising, given his current role, that he has been pushing for price hikes at music streaming services.

Both of these topics came up during Kyncl’s Q&A session at the Code Conference in Laguna Niguel, California, on Tuesday (September 26).
</item 82>

<item 83>
‘Latin artists have been paving the way for deals that offer more control, transparency, and better economics.’ (Music Business Worldwide,
2023-12-07T20:32:11+00:00)
They want to keep ownership of their masters, and get better economics.

This level of agility is at the very core of Believe’s structuring: from self-releasing services with TuneCore, to more premium services with Believe Label and
Artist Solutions or Artist Services Solutions, we pride ourselves for being able to accompany talents like Iñigo at all stages of their development, from being
emerging acts, all the way to becoming top charting artists.

Can we read anything into the fact that Inigo is Spanish, and how the trend identified above (i.e. Latin American artists having hits without being on major
record company deals) is starting to spread to different corners of the world and different genres? We just had RAYE enjoying gobal success independently with
Escapism, and in the US earlier this year there was of course Oliver Anthony going to No.1 on the Billboard Hot 100…

For many years now, Latin artists have been paving the way for deals that bring them more control, transparency, and better economics.
</item 83>

<item 84>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Confirm That Copying Music to Train AI Models is Not Fair Use. “Even worse are those that argue that copyrighted content should automatically be considered fair
use so that protected works are never compensated for usage and creators have no say in the products or business models that are developed around them and their
work. Congress should assure and agencies should presume that reproducing music to train AI models, in itself, is not a fair use.” Prevent the Cloning of
Artists’ Voices and Likenesses Without Express Permission. “We cannot allow an artist’s voice or likeness to be cloned for use without the express permission of
the artist. This is a very personal decision for the artist. Congress should pass into law effective federal protections for name, image, and likeness.”
Incentivize Accurate Recordkeeping. “Correct attribution will be a critical element to artists being paid fairly and correctly for new works that are created.
</item 84>

<item 85>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
Meta, on an open source tear, wants to spread its influence in the ongoing battle for AI mindshare.

This morning, the social network announced that it’s teaming up with IBM, whose audience is decidedly more corporate and enterprise, to launch the AI Alliance,
an industry body to support “open innovation” and “open science” in AI.

So what will the AI Alliance do exactly — and how will its work differ from the quite similar (at least in terms of its overarching mission, members and tenets)
Partnership on AI? The Partnership on AI years ago promised to publish research using open source licenses and minutes from its meetings to, as the AI Alliance
purportedly seeks to do, educate the public on pressing AI issues of the day.

Well — confusingly — the Partnership on AI is in fact a member of the AI Alliance.
</item 85>

<item 86>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
In addition, rights can only be enforced around the training of AI when there are accurate records about what is being copied. Otherwise, the inability to
enforce rights in the AI marketplace equates to a lack of rights at all, producing a dangerous imbalance that prevents a thriving ecosystem. This requires
strong and accurate record keeping by the generative AI platforms, a requirement that urgently needs legislative support to ensure incentives are in place so
that it happens consistently and correctly.” Assure Transparency for Consumers and Artists. “Transparency is necessary to clearly distinguish human-created
works from AI-created works. The public should know, when they are listening to music, whether that music was created by a human being or a machine.”

Added Kooker: “While these principles are simple and basic, they require a new level of commitment and investment from generative AI platforms.
</item 86>

<item 87>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
The NMPA made it clear it’s on the side of “opt in.”

“US copyright law is an opt-in system,” the NMPA said. “Opt-out regimes fundamentally undermine copyright protections by shifting the burden to obtain a license
away from users.”

The submission continued: “An opt-out scheme that requires rights holders to opt out on an AI company-by-AI company or application-by-application basis would
not be feasible, given the sheer volume of AI companies and applications; it is nearly a full-time job to keep up with developments in the AI marketplace…
Copyright owners, particularly individual creators and small businesses could not possibly meet such a burden.

“NMPA strongly opposes consideration of such a measure.”

“Opt-out regimes fundamentally undermine copyright protections by shifting the burden to obtain a license away from users.” NMPA

On this issue, the NMPA is fully aligned with UMG, which made a similar argument in its own submission to the Copyright Office.
</item 87>

<item 88>
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
</item 88>

<item 89>
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
</item 89>

<item 90>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
The mere fact that, as a technological matter, copies need to be made to extract those ideas and facts from copyrighted works should not alter that result .”

could be accomplished without the creation of copies, there would be no copyright questions here. Indeed that act of “knowledge harvesting”… like the , would
not only be , it would further the very purpose of copyright law. The mere fact that, as a technological matter, .” “Some might object to this logic in the
context of generative Al systems, arguing that even if such a system produces content that is not substantially similar to any of the content it was trained on,
the output of that model might compete in the marketplace with works used for training or, more broadly, with the authors of those works … This argument
misunderstands both the nature of the fair use inquiry and the creative markets that copyright is intended to protect.
</item 90>

<item 91>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
In the meantime, music emerged out of [digitization] better and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll
likely be first.”

Asked when that sea change will occur, Kyncl said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is
increasing quality at a very fast pace.”

Kyncl drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.

“I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”
</item 91>

<item 92>
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
</item 92>

<item 93>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“An opt-out system is based on the erroneous premise that training on copyrighted works without permission is by default lawful unless each copyright owner
objects,” UMG stated in its submission.

“That philosophy does violence to basic principles of copyright law, imposes undue burdens on copyright owners, creates the wrong incentives for AI developers,
and is neither practicable nor effective for protecting the rights of copyright owners or ensuring the sensible use of copyrighted works for training purposes.”

4. Training AI on copyrighted works is not “fair use” under almost any circumstance

Another area in which the NMPA and UMG are aligned is on the issue of whether using copyrighted material constitutes “fair use.”

Fair use is the carve-out to copyright law that allows copyrighted materials to be used without permission in certain limited circumstances, such as for
educational purposes or for reporting news.
</item 93>

<item 94>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
A clumsy example: I read a great book, then I meet you for breakfast, where I tell you about the story and quote a couple of my favorite passages to you.

To suggest I’ve just committed copyright infringement wouldn’t only be silly – it would mean that copyright protections were getting in the way of the
commercial furtherment of the author/rightsholder (i.e. you might go and buy the book on my recommendation).

Google’s submission to the USCO contains a number of passages that posit a similar position… but for generative AI models. The document argues that the
ingestion of copyrighted material for the training of generative AI platforms shouldn’t be hampered by copyright law.
</item 94>

<item 95>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Being involved in this has been a real career highlight for me, without a doubt.

If I could give you our patented MBW magic wand to change something about today’s business right here and now, what would it be and why?

We could be talking for hours about this, Tim, and there are a few things out there I’d like to change with that wand! But I’ll spare you an exhaustive rant and
focus on one issue.

We need more diversity at all levels and within all facets of the music industry, particularly on the label side.

“along with doing what I can to increase the number of women at our company. There just aren’t enough of us, particularly within the most senior ranks at the
labels.”

Diversity is a superpower.
</item 95>

<item 96>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
However, to get the clearest view of the direction the music industry would like to see things head, it might help to read the submission from the National
Music Publishers Association (NMPA).

The NMPA’s submission, dated October 30, 2023, pulls no punches.

It starts off by stressing that its membership – US music publishers major and independent – are “not opposed” to AI.

“There is widespread belief in the music industry that great benefits could come from generative AI systems that can assist human creators,” the NMPA states.

But then it goes for the jugular: “However, the development of the generative AI marketplace is marked by breathtaking speed, size and complexity. Hindsight may
well prove that there is no hyperbole in saying that generative AI is the greatest risk to the human creative class that has ever existed.”

“Hindsight may well prove that there is no hyperbole in saying that generative AI is the greatest risk to the human creative class that has ever existed.”
</item 96>

<item 97>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Prioritization. I think companies often use that word to mean, “we’re going to focus on one group of artists and ignore all the others.” For CMG, it means
defining what success can uniquely mean for each artist, and then delivering for them.

We identify which artists will benefit from, say, the close attention of a small group of people who will bring passion and nuance to marketing and development.
Others will clearly need the full power of our organization firing on all cylinders from the get-go.

What’s particularly changed in how you approach artist campaigns both domestically and internationally?

We engage directly with international partners earlier than ever before, and don’t unfairly rely on our ex-US affiliates within UMG to do the heavy lifting
early on. They have their own local repertoire to work on.
</item 97>

<item 98>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
Now, those companies have a clear horse in the race and perhaps regulatory capture on the mind… but they’re not wrong entirely. Meta continues to take
calculated open sourcing risks (within the bounds of regulators’ tolerances), releasing text-generating models like Llama that bad actors have gone on to abuse
but which plenty of developers have built useful apps upon.

“The platform that will win will be the open one,” Yann LeCun, Meta’s chief AI scientist, was quoted as saying in an interview with The New York Times — and
who’s among the more than 70 influential signers of a letter calling for more openness in AI development. LeCun has a point; according to one estimate,
Stability AI’s open source AI-powered image generator, Stable Diffusion, released last August, is now responsible for 80% of all AI-generated imagery.

But wait, you might say — what does IBM gain from the AI Alliance? It’s a co-founder with Meta after all.
</item 98>

<item 99>
The new, ‘efficient’ Spotify has a very different approach to podcasting (The Verge, 2023-10-25T16:10:29+00:00)
He pointed to the automatic translation tool as a way of seamlessly scaling podcasts and increasing the amount of content in non-English speaking markets. He
also expects AI will be a boon for podcast advertising, as well.

“Creating a great audio ad is something that’s quite costly and quite expensive for marketers to do,” he said. “What generative AI has the promise to do is
allow for that creative cost to come down… It [also] allows you to scale that creative in unimaginable ways. You can translate whatever creative you had into
lots of different languages; you can use the same voice actor; but instead of producing one or two ads, you can have 1,000 or 10,000 or even 100,000 ads that
are individually created to each user.”

I would take that with a grain of salt (we know Ek loves to dream big!!), but it does underscore the newest iteration of Spotify’s podcasting business — leaner,
less splash, and more scale.
</item 99>

<item 100>
Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn (Music Business Worldwide,
2023-11-29T20:55:14+00:00)
As a result, the US “is the only significant music economy in the world that does not have these full rights in their domestic legislation,” the artists’ groups
said.Music Business Worldwide
</item 100>
=== END ITEM id=10ab74fc1bdb970c ===
