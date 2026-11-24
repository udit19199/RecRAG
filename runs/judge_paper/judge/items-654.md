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

=== ITEM id=eb2819d8723a2fe2 ===
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
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
“No. Music is much more resilient. So first, I think what happened in the last 15 years is incredible. They had literally zero people in the subscription model
and now we have 700 million people in the world in the premium experience… which is incredible.

“And I think really credit goes to Daniel Ek for forging the path for everybody. And then companies like Apple and YouTube and Amazon following and building up
the [music] business. It’s pretty incredible what has happened.”

“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.” Robert Kyncl, Warner Music Group

Kyncl added: “I think the opportunity ahead of us is twofold. One, the continued growth in emerging markets, and also the GDPs of those countries will be rising
at the same time. So there’s lots of growth there.
</item 1>

<item 2>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
These products range from tools for “creative or marketing assistance”, to tools that “potentially give us the ability to better protect artist content or find
it when used in an unauthorized fashion” as well as “brand new products that have never been launched before”.

Kooker said that some of these conversations “also include potential equity investments which would accelerate the development of these companies”.

The Sony Music exec cited, as an example, a recent generative AI-powered project around a reissue and remix of an album. Although Kooker didn’t name it
specifically, SME did recently announce a generative AI project with The Orb and David Gilmour, which was a partnership between Sony Music Entertainment, Legacy
Recordings AI company Vermillio

“These artists are known for their cutting-edge experimentation in music,” said Kooker.
</item 2>

<item 3>
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
</item 3>

<item 4>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
So yeah, we are in that business.

“We are in the business of administering [rights], which means collecting revenue from thousands of platforms around the world; very complicated matters. But
we’re also in the business of ownership.”Music Business Worldwide
</item 4>

<item 5>
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
</item 5>

<item 6>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
“If established early on, they will result in an even playing field for all participants so there will not be an unfair competitive advantage for a few at the
expense of future innovators.

“The principles will ensure that intellectual property businesses can succeed in this new world alongside tech partners, further advancing investment and
expanding economic opportunities. “Music Business Worldwide
</item 6>

<item 7>
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
</item 7>

<item 8>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
The mere fact that, as a technological matter, copies need to be made to extract those ideas and facts from copyrighted works should not alter that result .”

could be accomplished without the creation of copies, there would be no copyright questions here. Indeed that act of “knowledge harvesting”… like the , would
not only be , it would further the very purpose of copyright law. The mere fact that, as a technological matter, .” “Some might object to this logic in the
context of generative Al systems, arguing that even if such a system produces content that is not substantially similar to any of the content it was trained on,
the output of that model might compete in the marketplace with works used for training or, more broadly, with the authors of those works … This argument
misunderstands both the nature of the fair use inquiry and the creative markets that copyright is intended to protect.
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>

<item 11>
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
</item 11>

<item 12>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
However, to get the clearest view of the direction the music industry would like to see things head, it might help to read the submission from the National
Music Publishers Association (NMPA).

The NMPA’s submission, dated October 30, 2023, pulls no punches.

It starts off by stressing that its membership – US music publishers major and independent – are “not opposed” to AI.

“There is widespread belief in the music industry that great benefits could come from generative AI systems that can assist human creators,” the NMPA states.

But then it goes for the jugular: “However, the development of the generative AI marketplace is marked by breathtaking speed, size and complexity. Hindsight may
well prove that there is no hyperbole in saying that generative AI is the greatest risk to the human creative class that has ever existed.”

“Hindsight may well prove that there is no hyperbole in saying that generative AI is the greatest risk to the human creative class that has ever existed.”
</item 12>

<item 13>
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
</item 13>

<item 14>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
The goal is that AI tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

“It’s really interesting from a music perspective, specifically, because . . . musicians are perhaps more familiar with the idea of AI,” Holland said.
“Musicians in general are more familiar with things like music production software, and AI tools like MIDI drum loops . . . so I think that there is a certain
amount of more progressive learning from them, when it comes to technology, and its ability to make their music better.”

When it comes to art and AI, the conversation is complicated, to say the least. Musicians are nervous about industry giants copyrighting AI music and cutting
them out of the process. Major record labels are worried about AI models training on their catalogues and stealing a slice of their considerable pie.
</item 14>

<item 15>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
Spotify erased thousands of AI-crafted songs from its platform but also recently globally launched an AI-powered DJ that curates music for listeners while
talking to them in a synthetic voice.

“The training of generative AI using our artists’ music . . . begs the question as to which side of history all stakeholders in the music ecosystem want to be
on: the side of artists, fans and human creative expression, or on the side of deep fakes, fraud and denying artists their due compensation,” Universal Music
Group said after a song using AI to imitate Drake and The Weeknd, two of its artists, went viral.

These same conversations and contradictions are manifesting across creative industries, but artists themselves don’t always have a seat at the table.
Independent artists in particular are learning that their voices resonate louder when coming together across disciplines to push back against what Holland
describes as an “extraordinary spectrum of exploitation” that leverages their work.
</item 15>

<item 16>
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
</item 16>

<item 17>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.
</item 17>

<item 18>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
which they were not open to before. And they just created this incredible tidal wave. And these things happen in waves, ebb and flow.

“So I think it likely is slowing down a little bit, but that’s okay. Because it makes the multiples come down a little bit more.”

“I think what happened in the last 15 years is incredible. They had literally zero people in the subscription model and now we have 700 million people in the
world in the premium experience.” Robert Kyncl, Warner Music Group

In other words, Warner Music can go out and buy more if the multiples are low enough?

“Of course. We are in the business of buying catalogs, obviously. We have the publishing catalog of David Bowie, for instance, which we bought a few years ago.
So yeah, we are in that business.
</item 18>

<item 19>
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
</item 19>

<item 20>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

Legislators and regulators around the world are grappling with how to address the many issues that have emerged since AI technology “went mainstream” about a
year ago.

In the US, one of the key efforts on that front is being spearheaded by the US Copyright Office, which this past summer issued a call for submissions on the
issue of AI and copyright. The goal is for the USCO to put together a study to ”help assess whether legislative or regulatory steps in this area are warranted.”

Numerous businesses involved in AI, and those with substantial copyright holdings, have submitted their thoughts on the issue, including Universal Music Group
and AI developer Anthropic.
</item 20>

<item 21>
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
</item 21>

<item 22>
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
</item 22>

<item 23>
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
</item 23>

<item 24>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
Here are some of the key arguments the MNPA made in its submission…

1. Works created principally through AI should not be copyrightable

In the view of the NMPA, simply giving a prompt to a AI music generator – “compose a power ballad about falling in love” – shouldn’t be enough to create a
copyrightable work.

“Policies surrounding copyrightability of AI-generated content will either serve to incentivize continued investment and effort into human creativity or to
disincentivize it,” states the submission, which can be read in full here.

“The law should never be such that human creators stand to gain more from repeatedly clicking a button to generate massive amounts of AI-produced materials than
from putting their hearts, souls, experiences, skills, talents, and emotions into expressive works of art.”
</item 24>

<item 25>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
They’re stars, for sure. But in both the case of Universal Music Group and Warner Music Group, they’re – currently anyway – not the biggest megastars on either
company’s books.

You might not expect, at this early stage, for the likes of UMG to put forward Drake, Taylor Swift, or The Beatles – or for WMG to put forward Ed Sheeran or Dua
Lipa – to be guinea pigs in any early AI music experiments, even if they’re run by ‘friends’ of the music industry like YouTube.

You might, then, also understandably wonder if there have been significant licensing payments made by YouTube to UMG and WMG for the right to play with the AI
vocals of artists who have signed off on ‘Dream Track’ thus far. And, if so, if these payments are being placed against any as-yet-unrecouped advances that some
of these artists may have on their label accounts.
</item 25>

<item 26>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
That argument could yet come into play in Anthropic’s legal battle with Universal.

And given the high stakes involved – for the music industry, for the tech industry and for the shape of culture going forward – that case, along with the US
Copyright Office’s study of AI and copyright, are worth keeping a close eye on.

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 26>

<item 27>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
In addition, rights can only be enforced around the training of AI when there are accurate records about what is being copied. Otherwise, the inability to
enforce rights in the AI marketplace equates to a lack of rights at all, producing a dangerous imbalance that prevents a thriving ecosystem. This requires
strong and accurate record keeping by the generative AI platforms, a requirement that urgently needs legislative support to ensure incentives are in place so
that it happens consistently and correctly.” Assure Transparency for Consumers and Artists. “Transparency is necessary to clearly distinguish human-created
works from AI-created works. The public should know, when they are listening to music, whether that music was created by a human being or a machine.”

Added Kooker: “While these principles are simple and basic, they require a new level of commitment and investment from generative AI platforms.
</item 27>

<item 28>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
Robert Kyncl, Warner Music Group

In Kyncl’s view, the solutions put in place to address UGC form a “blueprint” than can be used to address AI-generated content on platforms as well.

“When YouTube was formed… people started to upload content, including copyrighted material, which, obviously, put YouTube into hot water with lots of different
copyright holders. And I had the privilege of working through a lot of that and fix it up.

“But we made we made a very important decision, which was to go above and beyond the law, and build a fingerprinting software that allowed us to track the
copyright on our platform, and then have commercial relationship[s] with copyright holders to send them the money. Out of that we built a multi-billion-dollar
business, which now is a multi-billion-dollar business per year. And it was an incredible new revenue stream for everyone. AI is that with new super tools.”
</item 28>

<item 29>
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
</item 29>

<item 30>
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
</item 30>

<item 31>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Licensing musical works before training use is both required and practicable,” the NMPA’s submission states, adding that “there are well developed processes in
place for technology ventures to obtain free market licenses on a large scale.”

And indeed it would have to be on a “large scale,” given that AI algorithms such as large language models are trained on countless millions or even billions of
pieces of digital data.

“Transparency and recordkeeping requirements are needed to disincentivize infringing activity and to support enforcement activities by copyright owners.” NMPA

The NMPA notes that work on licensing “is already underway,” pointing to news reports earlier this year that UMG, along with Warner Music Group, are in talks
with Google to secure licenses for artists’ vocals and musical melodies to create AI-generated songs.
</item 31>

<item 32>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

In recent months, Warner Music Group CEO Robert Kyncl has been keeping a close eye on AI in the music industry – not surprising, given that he came to the music
biz from ‘Big Tech’, namely YouTube, where he was Chief Business Officer.

Nor is it surprising, given his current role, that he has been pushing for price hikes at music streaming services.

Both of these topics came up during Kyncl’s Q&A session at the Code Conference in Laguna Niguel, California, on Tuesday (September 26).
</item 32>

<item 33>
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
</item 33>

<item 34>
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
</item 34>

<item 35>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
Sir Lucian Grainge, Chairman and CEO of Universal Music Group, said of UMG’s decision to offer a limited license to ‘Dream Track’: “We have a fundamental
responsibility to our artists to ensure the digital ecosystem evolves to protect them and their work against unauthorized exploitation, including by generative
AI platforms.

“At the same time, we must help artists achieve their greatest creative and commercial potential – in part by providing them access to the kind of opportunities
and cutting-edge creative tools made possible by AI.”

“We have a fundamental responsibility to our artists to ensure the digital ecosystem evolves to protect them and their work against unauthorized exploitation,
including by generative AI platforms.
</item 35>

<item 36>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
More broadly, you might ponder a bigger question.

When the day comes that YouTube asks the world’s biggest superstars to wholeheartedly embrace ‘Dream Track’, will Google’s recent US Copyright Office filing
give said superstars – and their record companies – pause for thought?

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 36>

<item 37>
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
</item 37>

<item 38>
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
</item 38>

<item 39>
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
</item 39>

<item 40>
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
</item 40>

<item 41>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
In the meantime, music emerged out of [digitization] better and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll
likely be first.”

Asked when that sea change will occur, Kyncl said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is
increasing quality at a very fast pace.”

Kyncl drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.

“I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”
</item 41>

<item 42>
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
</item 42>

<item 43>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
This all seemed to hint at a desire to build a ‘Content ID’-style system on YouTube to control and police content made by cloning artists’ voices;

publicly committed to three principles/pledges behind its development of music-based . These pledges promised in said tools. They promised that, amongst other
things, would help YouTube . This all seemed to hint at a desire to build a on YouTube to control and police content made by cloning artists’ voices; Then,
earlier this month, YouTube announced it was developing a specific system for music partners that allowed them to request the removal of content on its platform
that “mimics an artist’s unique singing or rapping voice”.

Evidently, all of this was enough for Universal Music Group and Warner Music Group to dip their toe into the ‘Dream Track’ experiment – though not without
caution.
</item 43>

<item 44>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Human authorship is a bedrock requirement of copyright.”

2. AI developers should be required to license materials and keep records of the materials they use for training

Many key players in the music industry and beyond have been adamant that using copyrighted works to train AI algorithms without authorization is a copyright
infringement, and a number of lawsuits have been launched against AI companies on these grounds.

One such lawsuit was filed by Universal Music Group, Concord Music Group and ABKCO against AI developer Anthropic, seeking potentially tens of millions of
dollars for the alleged “systematic and widespread infringement of their copyrighted song lyrics.”

But that doesn’t mean key music industry players don’t want their materials ever used to train AI – they just want AI developers to license that music, not
unlike how streaming services, bars and restaurants or movie producers license music.
</item 44>

<item 45>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
Confirm That Copying Music to Train AI Models is Not Fair Use. “Even worse are those that argue that copyrighted content should automatically be considered fair
use so that protected works are never compensated for usage and creators have no say in the products or business models that are developed around them and their
work. Congress should assure and agencies should presume that reproducing music to train AI models, in itself, is not a fair use.” Prevent the Cloning of
Artists’ Voices and Likenesses Without Express Permission. “We cannot allow an artist’s voice or likeness to be cloned for use without the express permission of
the artist. This is a very personal decision for the artist. Congress should pass into law effective federal protections for name, image, and likeness.”
Incentivize Accurate Recordkeeping. “Correct attribution will be a critical element to artists being paid fairly and correctly for new works that are created.
</item 45>

<item 46>
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
</item 46>

<item 47>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
As creative industries grapple with AI’s explosion into every artistic medium at once, separate calls from artists warning the world to take action before it’s
too late are starting to converge. From fake Drake songs to stylized Instagram profile pictures, art conjured with newly sophisticated AI tools is suddenly
ubiquitous — and so are conversations about how to rein in the technology before it does irrevocable harm to creative communities.

This week, digital rights organization Fight for the Future partnered with music industry labor group United Musicians and Allied Workers to launch
#AIdayofaction, a campaign that calls on Congress to block corporations from obtaining copyrights on music and other art made with AI.

The idea is that by preventing industry behemoths like major record labels, for example, from copyrighting music made with the assistance of AI, those companies
will be forced to keep looping humans into the creative process. But those same concerns — and the same potential strategies for pushing back against the
onslaught of AI — exist across creative industries.
</item 47>

<item 48>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists;

and jointly announced that the video platform was launching an – a program by which new tools and innovations will be developed at YouTube in close conjunction
with artists and the music biz. To progress this ‘incubator’, YouTube said it was working behind the scenes with a number of UMG-affiliated artists; Also in
August, YouTube publicly committed to three principles/pledges behind its development of music-based generative AI tools . These pledges promised “appropriate
protections… for music partners who decide to participate” in said tools. They promised “invest[ment] in the AI-powered technology” that, amongst other things,
would help YouTube “protect our community of viewers, creators, artists and songwriters… [from] trademark and copyright abuse” .
</item 48>

<item 49>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
As reported by MBW last week, Universal Music Group submitted its response to the USCO. Unsurprisingly, amongst the highlights of UMG’s submission was the
rejection of the notion that using copyrighted material to train LLMs is “fair use”.

In his speech on Wednesday, Kooker commented directly on the flurry of submissions to the USCO from the tech industry at the end of October.

In what may have been a nod to Google’s filing and “fair use” position, Kooker told Senators that “based on recent Copyright Office filings it is clear that the
technology industry and speculative financial investors would like governments to believe in a very distorted view of copyright”.

He added: “[That view is] one in which music is considered fair use for training purposes and in which certain companies are permitted to appropriate the entire
value produced by the creative sector without permission, and to build huge businesses based on it without paying anything to the creators concerned.”
</item 49>

<item 50>
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
</item 50>
=== END ITEM id=eb2819d8723a2fe2 ===

=== ITEM id=33cdc94e91826f63 ===
Gold sentences and facts:
Sentence 0: Likely with the sole intent of keeping Messi fit in that time off, and the added bonus of raking in significant cash, Inter Miami scheduled a two-match tour of China in early November after their regular season concluded.
  Fact 0.0: Inter Miami scheduled a two-match tour of China in early November.
  Fact 0.1: The China tour was scheduled after Inter Miami's regular season concluded.
  Fact 0.2: The likely sole intent of the China tour was keeping Messi fit during that time off.
  Fact 0.3: The China tour had the added bonus of raking in significant cash.
Sentence 1: Eventually, Alvaro Barreal’s 78th-minute goal ended any hope that Inter Miami had to complete a stunning playoff charge, and the defeat coupled with CF Montreal’s big 4-1 victory over Portland means Inter Miami are officially out of the postseason running.
  Fact 1.0: Alvaro Barreal scored a goal in the 78th minute.
  Fact 1.1: Barreal's 78th-minute goal ended any hope Inter Miami had of completing a stunning playoff charge.
  Fact 1.2: Inter Miami's defeat, together with CF Montreal's 4-1 victory over Portland, means Inter Miami are officially out of the postseason running.
Sentence 2: Lionel Messi's arrival at Inter Miami has brought with it a craze for fans to get a glimpse of one of the sport's most talented and legendary players of all time.
  Fact 2.0: Lionel Messi's arrival at Inter Miami has brought a craze for fans to get a glimpse of him.
  Fact 2.1: Lionel Messi is one of the sport's most talented players of all time.
  Fact 2.2: Lionel Messi is one of the sport's most legendary players of all time.

Retrieved items:
<item 1>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
65th min: Chance, FC Cincinnati! Inter Miami have hit the post three times, but now it's FC Cincinnati who have struck the woodwork! Brandon Vazquez gets his
head to a corner but it comes off the post.

64th min: Inter Miami have to remain wary of the vicious FC Cincinnati counter-attack, and here a lovely Luciano Acosta switch laterally across the top of the
penalty area finds Alvaro Barreal in space, but his hit is blocked.

Pat Noonan makes his second change of the match, bringing Yuya Kubo on for Junior Moreno in midfield.

58th min: Lionel Messi's influence has been immediate, and he earns a dangerous free-kick just outside the penalty area, well within his range. Messi appears to
be struggling to get good footing on this field, but it's the shirt pull from Miazga which brings him down.

Messi takes it but puts it way over the crossbar. A bit rusty?
</item 1>

<item 2>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
PT

8:30 p.m. ET / 5:30 p.m. PT TV Channel: — None —

— None — Streaming: Apple TV+ (MLS Season Pass)

Chicago Fire vs. Inter Miami betting odds & lines

Inter Miami have been mired in a slump without Lionel Messi, and with the Argentine likely to miss out again today, the Herons are heavy underdogs on the road
at Chicago.

Goals are expected in this match as both sides have conceded in bunches for the most part this year.
</item 2>

<item 3>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
Tomas Aviles is suspended for yellow card accumulation, while Ian Fray and Corentin Jean are both out for the season with ACL injuries.

Inter Miami starting lineup (4-3-3): Callender (GK) — Yedlin, Kryvtsov, Miller, Allen — Arroyo, Busquets, Cremaschi — Farias, Martinez, Taylor.

Inter Miami subs (9): Dos Santos (GK), McVey, Sailor, Ulloa, J. Mota, D. Ruiz, Sunderland, Stefanelli.

How to watch Chicago Fire vs. Inter Miami

Date: Wednesday, Oct. 4, 2023

Wednesday, Oct. 4, 2023 Time: 8:30 p.m. ET / 5:30 p.m. PT

8:30 p.m. ET / 5:30 p.m.
</item 3>

<item 4>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
Without Lionel Messi, Inter Miami have looked exceedingly mortal. They put in a 90-minute performance littered with individual mistakes, and the Chicago Fire
made them pay, winning 4-1 via braces from Xherdan Shaqiri and substitute Maren Haile-Selassie.

The result boosts Chicago’s postseason hopes significantly, moving them into a playoff position in the Eastern Conference standings.

Inter Miami’s playoff hopes, conversely, are now on life support, needing to win out in their final three games and get outside help. The club is winless in
four games with Messi out injured, and it has seen their chances at a postseason berth nearly evaporate.

Shaqiri’s brace was the biggest talking point for the hosts, as the former Liverpool midfielder has not always been at his best since joining MLS, but in front
of a record Chicago Fire crowd at Soldier Field, he rose to the occasion.

The Sporting News followed the Chicago Fire vs. Inter Miami match live, providing score updates, commentary and highlights as they happened.
</item 4>

<item 5>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
MORE: Details on Lionel Messi's injury as he battles "fatigue" and a "scar tissue issue".

La idea es que Leo Messi vuelva a sumar minutos mañana cuando Inter Miami juegue contra Cincinnati. pic.twitter.com/ePNWHb4vSs — Gastón Edul (@gastonedul)
October 6, 2023

Inter Miami vs FC Cincinnati lineups & team news

There was a report that Lionel Messi could finally make his return from injury, and while it felt unlikely that Messi would start three days after missing out
entirely, he appears on the bench today and could be available for a second half appearance if required.

Jordi Alba remains sidelined with a hamstring injury, missing since his injury just minutes before Messi, and without a publicly known timeline for a return.
Also missing is striker Leonardo Campana who was absent from the midweek defeat to the Chicago Fire with a muscle injury, although Gerardo Martino described his
issue as "minor."
</item 5>

<item 6>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
20, 2024 TBD (rumored) Inter Miami vs. River Plate AT&T Stadium

(Dallas, TX, USA)

Messi vs Ronaldo set as Inter Miami schedule Al Nassr friendly

Lionel Messi and Cristiano Ronaldo will meet in a club friendly between Inter Miami and Al Nassr in Riyadh, Saudi Arabia on Thursday, February 1 with kickoff at
1 p.m. ET (9 p.m. local time in Saudi Arabia).

The match comes as the second of a double-header for Inter Miami in Saudi Arabia as part of the annual Riyadh Season Cup showcase. Three days prior, Inter Miami
will also meet fellow Saudi club Al Hilal.

Messi and Ronaldo have squared off 35 times across their careers, with Messi winning 16 times to Ronaldo's 10, plus nine draws. Messi claimed the most recent
face-off as PSG won their friendly matchup 5-4 in Riyadh last time around.
</item 6>

<item 7>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Messi joined Inter Miami with the club bottom of the Eastern Conference, and the gap was just too big to overcome. Messi's injury certainly played a role, but
even so, there was just not enough room for error.

It's official: Inter Miami has been mathematically eliminated from MLS playoff contention. pic.twitter.com/u39ZgeXI9M — Tom Bogert (@tombogert) October 8, 2023

Inter Miami vs FC Cincinnati: Second Half

90+3 min: With five minutes of added time on the board, Lionel Messi draws another foul in a very dangerous position, this one much more obvious for the
official to make. It's just outside the penalty area from a central position, and Messi stands over it yet again.

His effort on goal is close to finding the top-left corner, but it's just high, clipping off the very top of the crossbar.
</item 7>

<item 8>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
47th min: Luciano Acosta is fouled on the edge of the attacking third, and it gives FC Cincinnati a free-kick early in the second half. Alvaro Barreal's
delivery is floated way high in the air and falls to the far post, but after a nervy moment it's cleared.

Lionel Messi begins to warm up on the touchline, and he may be brought on soon to help Inter Miami save their postseason chances.

Kickoff: The second half is underway, and there are no changes from Inter Miami, which means Lionel Messi remains on the bench for now. With their playoff hopes
now squarely on the line, it's a huge call from Gerardo Martino not to bring Messi on now, and it could speak to how fit he truly is.

FC Cincinnati make one change, as Sergio Santos comes off, replaced by Aaron Boupendza up front.
</item 8>

<item 9>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
MORE: A full rundown of how each game in the Messi vs Ronaldo matchup has played out

Inter Miami to play El Salvador in January

Inter Miami's first confirmed friendly of the 2024 preseason will see the club travel to San Sebastian to take on the El Salvador national team on January 19.

The match does not fall under an official FIFA international window, meaning the national team will likely not feature any players from Europe or other leagues
currently in season.

"We look forward to facing the El Salvador national team in what will be a good first test for our squad against the best players from the country,” said Chief
Soccer Officer and Sporting Director Chris Henderson.

Ticketing and broadcast details will be available at a later date.

MORE: All the details regarding Lionel Messi's Inter Miami contract and salary

Inter Miami visit Hong Kong in February tour of Asia

The club announced in early December that Inter Miami would take a tour of Asia in early February as part of their 2024 preseason slate.
</item 9>

<item 10>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Arias, 4. Hagglund, 21. Miazga, 32. Murphy, 31. Barreal (Halsey, 81') — 5. Nwobodo, 10. Acosta (Badji, 88'), 93. Moreno (Kubo, 65') — 19. Vazquez (Pinto, 88'),
17. Santos (Boupendza, 46').

Inter Miami vs FC Cincinnati live updates, highlights, commentary

FULLTIME: Inter Miami 0-1 FC Cincinnati

Inter Miami are made to rue their early missed chances, as Alvaro Barreal's 78th-minute goal is enough to secure victory for FC Cincinnati.

In the process, Inter Miami are officially OUT of the MLS playoff race, as tonight's defeat coupled with CF Montreal's 4-0 victory over Portland are enough to
see them eliminated from contention.
</item 10>

<item 11>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Important midfielder Gregore, who has missed most of the season with a foot injury, is back in training and should return to the field at some point, although
when is not known. Defender Tomas Aviles is back from suspension and starts in the three center-back system.

Inter Miami starting lineup (5-3-2): Callender (GK) — Yedlin, Aviles, Kryvtsov, Miller, Allen — Ruiz, Busquets, Cremaschi — Farias, Martinez.

Inter Miami subs (12): Dos Santos (GK), Sailor, Taylor, Mota, Arroyo, Ulloa, Robinson, Messi, Stefanelli.

Defender Alvas Powell is suspended for the match after being sent off against the New York Red Bulls last time out, with Santiago Arias stepping in at
right-back.
</item 11>

<item 12>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Lionel Messi's arrival at Inter Miami has brought with it a craze for fans to get a glimpse of one of the sport's most talented and legendary players of all
time.

Along with that skyrocketing attention comes a leap in prices, and sadly the club seems to be passing that cost on to the supporters.

In 2024, Inter Miami will sport some of the world's most expensive tickets, rivaling the prices of Champions League giants such as Barcelona, Arsenal, and other
massive clubs around Europe, while dwarfing other MLS ticket packages.

While Fox Sports reported the club has an extensive waiting list for 2024 season tickets, the massive increase in price could push out many of the team's most
dedicated supporters who have been with the team since its inception in 2020, causing massive turnover in the fanbase.

A reminder that for most clubs, season-ticket packages include only home league matches, and while members usually get first access to purchase tickets for
knockout competitions, those all come at an additional cost.
</item 12>

<item 13>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Code Section Price 2 Mid Loge $45,900 3 North Loge $42,840 4 South Loge $42,840 5 Midfield Club $7,650 6 East Field Seats $13,005 A Northwest Prime $5,440 B
Northwest Midfield $4,420 C West Sideline $2,720 D West Reserved $2,380 E East Midfield $2,635 F East Upper Midfield $2,125 G East Sideline $1,955 H East
Reserved $1,360 I North Goaline $1,054 J South Goaline $1,020 K South Stand $867 L North Stand Reserved $884

Inter Miami season tickets among most expensive in world for 2024

With the Inter Miami season-ticket cost increasing by a huge margin over its own pricing model from 2023 to 2024, the club now boasts some of the most expensive
seats in world football, also rivaling that of other major sports in the U.S.
</item 13>

<item 14>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
🤔 pic.twitter.com/Svq79fvbBG — Major League Soccer (@MLS) October 4, 2023

1 hour to kickoff: With lineups not yet released, there is finally a report regarding Lionel Messi's availability. According to Michelle Kaufman of the Miami
Herald, Messi "did not arrive with the team" to Soldier Field. While that's not outright confirmation that he will not take part in the match, it's a very
strong indication that he will not be involved tonight.

1 hour 15 mins to kick: With lineups due imminently, and nothing to indicate if Lionel Messi is involved tonight or not, the Chicago Fire media team has jokes.
</item 14>

<item 15>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
The most expensive premium packages at Inter Miami in 2024, which will run supporters over $45,000, dwarf any of the most luxurious tickets in the Premier
League. The most expensive season ticket in the EPL can be found at Fulham, where the Craven Cottage luxury booths are valued at $3,620. Barcelona's most
expensive season ticket is $1,826 for the full year.

The National Football League (NFL), the most popular sport in the United States by a massive margin, boasted an average ticket price of $112 in 2022, while the
average NBA ticket price was $94 that year.

These prices also dwarf many other clubs in Major League Soccer. Orlando City SC, one of Inter Miami's chief in-state rivals, have their cheapest season ticket
available for just $324 in 2024, over half the cost of any Inter Miami season ticket, while their most expensive premium ticket costs $2,898.
</item 15>

<item 16>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
ET / 4:39 p.m. PT

Location: DRV PNK Stadium (Ft. Lauderdale, FL)

Referee: Rubiel Vazquez

Starting lineups:

Inter Miami (5-3-2, right to left): 1. Callender (GK) — 2. Yedlin (Taylor, 76'), 6. Aviles (Messi, 55'), 27. Kryvtsov, 31. Miller, 32. Allen — 41. Ruiz (Mota,
55'), 5. Busquets, 30. Cremaschi — 11. Farias, 17. Martinez (Stefanelli, 76').

FC Cincinnati (5-4-1, right to left): 18. Celentano (GK) — 13. Arias, 4. Hagglund, 21.
</item 16>

<item 17>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Inter Miami's home field advantage tonight will be significantly hampered.

Not sure how much can be appreciated via the photo, but La Familia — Inter Miami’s supporters’ groups — are mostly sitting here before kickoff.



No drums, chanting, or enthusiasm in what is a protest to the increased price of season tickets for 2024.#InterMiamiCF #Messi𓃵 pic.twitter.com/YaP4ivlX0P —
Franco Panizo (@FrancoPanizo) October 7, 2023

15 mins to kickoff: Interestingly, while tonight's match should kick off without any negative attention, it seems the club just managed to get off the ground
without a hitch. According to Michelle Kaufman of the Miami Herald, the supporter groups had planned a protest regarding 2024 season ticket prices, but the club
offered to meet with them this coming week so they called it off until that can take place.

A reminder, this match officially kicks off at 7:39 p.m.
</item 17>

<item 18>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
“As a player, you want these kinds of final games and to play for something. We put ourselves in this position, and we have a chance to do it, so now we have to
perform.”

Also, Shaqiri confirmed he's Team Messi over Team Ronaldo.

Q: Messi or Ronaldo?



🗣️Xherdan Shaqiri (Chicago Fire): “I’m always a big Messi admirer, if I have to choose now I will go with Messi. He won the WC which was the only thing he
missed. So for sure Messi.”#Messi #InterMiamiCF pic.twitter.com/Tv4rVXddrU — Inter Miami FC Hub (@Intermiamicfhub) October 3, 2023

35 mins to kickoff: Finally lineups have been released, and Lionel Messi is not part of the Inter Miami squad. Otherwise, it's a regular starting XI for Gerardo
Martino, with Josef Martinez up front as Leonardo Campana is out with a knock.
</item 18>

<item 19>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
A reminder, this match officially kicks off at 7:39 p.m. ET so there will be a slight delay from the advertised 7:30 p.m. time.

Inter Miami fan news: the 5 supporters’ groups known as La Familia threatened to stage a protest this game re: ticket price hike for 2024, but team offered to
meet w/them next week, so protest off at least until next game. @MiamiHerald @HeraldSports @InterMiamiCF #InterMiamiCF — Michelle Kaufman (@kaufsports) October
7, 2023

30 mins to kickoff: Lionel Messi's availability for Inter Miami is a welcome sight for the club, as they take the field today with their backs against the wall.
Inter Miami must win tonight to ensure they remain alive in the playoff race. A draw or a loss, however, would see them vulnerable to outside results.
</item 19>

<item 20>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
Chicago are into a playoff position, while Inter Miami's playoff chances take a massive hit with the defeat. Lionel Messi's club no longer controls its own
destiny, needing outside help to make the postseason.

Via @IanHest, #InterMiamiCF would be on the brink of playoff elimination if this result holds. pic.twitter.com/xqyyY5ifzP — Kyle Bonn (@the_bonnfire) October 5,
2023

Chicago Fire vs. Inter Miami: Second Half

90th min: Facundo Farias is booked for a hip check on Miguel Navarro, which might have been a retaliation foul for Navarro's jawing at Sergio Busquets.

Three minutes of added time are shown on the board.

83rd min: Referee Joe Dickerson has to step between Sergio Busquets and Miguel Navarro as the two are jawing at each other. It wasn't hard to read Navarro's
lips as he walked away, and his words aren't repeatable here.
</item 20>

<item 21>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
The trip will see Inter Miami play a collection of top players from the Hong Kong First Division. The match will be played on February 4 at Hong Kong Stadium,
which seats approximately 40,000 fans, while the club will hold an open training session the day prior. Tickets go on sale December 15 on ticket exchange site
Klook.

While the match is the only one in Asia confirmed as of yet, Inter Miami's reference of the trip as a "tour of Asia" suggests more matches could be announced in
due time. It's notable that Inter Miami will indeed travel to Asia after their attempts to play matches in China back in November fell through.

"We're very excited to be visiting Hong Kong and indeed Asia for the very first time," explained Jorge Mas, Inter Miami CF Managing Owner. "From the very
beginning we set out to be a global club at Inter Miami.
</item 21>

<item 22>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
The team from Argentina hold a special place in the heart of Messi and Martino. Not only are both men from the city of Rosario, where Newell's are based, but
Messi developed at the club's youth academy and Martino played more than 400 senior games for the club before becoming head coach in 2012.

"I am delighted to welcome my beloved Newell's to our home here in Miami. It will be a special match due to everything Newell's Old Boys means to me," said the
Inter Miami coach. "It will also be a good opportunity to prepare for what will surely be an exciting season."

Messi signed for Barcelona after spending six years as a youth player at Newell's, during which time he reportedly scored 234 goals.

Will Inter Miami face River Plate in Dallas?

In late November, it was reported that Inter Miami have arranged to contest a friendly against Argentine giants River Plate in Dallas, Texas in late January.
</item 22>

<item 23>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Inter Miami fans chant "Mess, Messi" as they have a free kick! pic.twitter.com/VkyvULPUHp — Mundo Albiceleste ⭐🌟⭐🇦🇷 (@MundoAlbicelest) October 8, 2023

24th min: Chance, Inter Miami! THEY'VE HIT THE WOODWORK AGAIN! Benja Cremaschi is fed through on a one-two with Josef Martinez, and with the outside of his
right foot, he bends it around Matt Miazga but clatters the near post. The THIRD time Inter Miami have hit the bar in under a half-hour! How unlucky??

22nd min: Chance, Inter Miami! Inter Miami hit the woodwork again! Josef Martinez, who has been electric so far, rips a shot from a tight angle about 15 yards
out, and while it beats Roman Celentano, it also clatters the far post and deflects out!
</item 23>

<item 24>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
Inter Miami no va a viajar a China. La gira que incluía dos amistosos se cae por temas burocráticos/comerciales. pic.twitter.com/8I0jBxg4Zz — Gastón Edul
(@gastonedul) October 30, 2023

Instead, Inter Miami met NYCFC in a hastily arranged friendly at DRV PNK Stadium on November 10 to honor Lionel Messi's most recent Ballon d'Or. The friendly is
being dubbed the "Noche d'Or" or "Night of Gold" as a nod to his latest achievement.
</item 24>

<item 25>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Elsewhere, Inter Miami will be nervous about other results from around the league. Most painful is the 2-0 lead for CF Montreal over Portland, which would be a
big blow to Inter Miami, meaning they would be eliminated with anything less than a win tonight.

Inter Miami vs FC Cincinnati: First Half

43rd min: There's a goal mouth scrap that sees Roman Celentano scrambling wildly on the doorstep as the ball bounces around awkwardly, but nobody can get on the
end of it for Inter Miami.

38th min: While Inter Miami haven't generated any clear-cut chances since the Celentano save on the half-hour mark, they've remained in complete control of this
match. The broadcast shows Lionel Messi on the bench, and he looks very antsy, like he wants to enter the game badly.

Some Inter Miami players are called off the touchline to warm up, but Messi isn't one of them.

30th min: Chance, Inter Miami!
</item 25>

<item 26>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
How to watch Inter Miami vs FC Cincinnati

MORE: Watch Premier League matches live with Fubo in USA

TV channel Streaming Global — Apple TV+ (MLS Season Pass)

Inter Miami vs FC Cincinnati betting odds & lines

In Canada, bet MLS at Sports Interaction: See the latest odds for Inter Miami vs FC Cincinnati here
</item 26>

<item 27>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
1 hour 30 mins to kick: As if Inter Miami's predicament tonight wasn't difficult enough, requiring a victory to keep their playoff hopes alive, it will have to
come against the best team in Major League Soccer through the 2023 regular season.

FC Cincinnati won the Supporters' Shield as the regular season champion, confirmed on Wednesday despite suffering defeat. It's been an exceptional season for
the club, and they won't stop here just because they have nothing tangible left to gain over the final few matches.

Campeones!!! 🏆⚽️🏆 2023 supporters’ shield winners 🟠🔵 pic.twitter.com/96ckfHkKEp — Santiago Arias (@santiagoarias13) October 7, 2023

1 hour 45 mins to kick: It's Fan Appreciation Night at DRV PNK Stadium, where they're giving out Inter Miami bucket hats to the first few fans through the
gates.
</item 27>

<item 28>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
With the 2023 MLS season closing earlier than expected for Inter Miami, eliminated from playoff contention, Lionel Messi will have significant time off before
the new campaign begins in February 2024.

With every new season come preseason friendlies, as the players hope to ramp up their match fitness prior to the start of the coming campaign. Players need game
time to build their bodies to the point where they can play full 90-minute matches, especially when they are contesting multiple games a week.

Inter Miami's preseason will be watched with extra observation as Messi draws considerable attention wherever he goes, home or abroad. The Argentina star had
looked like being part of a tour of China, although that's no longer the case, while the prospect of facing a familiar side from home is a tantalising one.

The Sporting News has compiled all the known preseason matches either confirmed or reported prior to the start of Inter Miami's 2024 season.
</item 28>

<item 29>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
According to Argentine journalist German Balcarce of River Plate publication RMG, the match will take place on January 20 and will be played at AT&T Stadium,
the home of the Dallas Cowboys. However, with the El Salvador friendly, which is confirmed by the club, happening literally the day before in El Salvador, it
would be impossible for Inter Miami to play this match as reported unless they split the squad.

The match would supposedly be the second leg of a two-part trip for River Plate, who will also meet Liga MX club Deportivo Guadalajara (formerly Chivas) on
January 17 at the same venue. This will be updated with any additional information on whether this match will be moved to another day or nixed entirely.
</item 29>

<item 30>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Still, Ryan's follow-up indicated that his error quote was doubled in price over the actual cost, which still leaves his seats increasing from around $6,800 to
a whopping $15,300 for two seats, an increase of 125% year-on-year.
</item 30>

<item 31>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
30th min: Chance, Inter Miami! Facundo Farias forces Roman Celentano into a full-stretch save! A right-footed blast sees the FC Cincinnati goalkeeper make his
first big stop. It's an utter siege from Inter Miami through the opening half-hour! Supporters' Shield winners FC Cincinnati have barely survived thus far!

28th min: Obinna Nwobodo is booked for a very late tackle on Facundo Farias in transition. He has to watch himself, as he gives the referee a sarcastic
applause, which is a good way to draw the official's ire and get sent off.

26th min: Inter Miami have another good set-piece opportunity as the rain begins to fall hard at Inter Miami. It's really coming down in South Florida, and more
could be on the way. Sergio Busquets takes it and blasts it straight into the wall.

Inter Miami fans chant "Mess, Messi" as they have a free kick!
</item 31>

<item 32>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Inter Miami were in simply too big of a hole in the MLS playoff race, one which even Lionel Messi couldn’t drag them out of.

With two matches remaining in the season, Inter Miami have been officially eliminated from the MLS playoff race, with Supporters’ Shield winners FC Cincinnati
dealing the deathblow with a 1-0 victory at DRV PNK Stadium.

The match had every hallmark of what made Inter Miami a flawed side when Lionel Messi arrived. Inter Miami utterly dominated the first half, but could not find
the breakthrough, instead hitting the woodwork three separate times in the opening half-hour and four overall in the match. On the other end, Drake Callender
was nothing short of sensational in goal, but even his heroics were just not enough to stymie the electric counter-attack that FC Cincinnati possess.
</item 32>

<item 33>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
@MiamiHerald @HeraldSports @InterMiamiCF #InterMiamiCF #Messi𓃵 #cf97 #MLS pic.twitter.com/hp8KHzkwNB — Michelle Kaufman (@kaufsports) October 4, 2023

2 hours to kickoff: Tata Martino has played his cards extremely close to his chest when it comes to the fitness of Lionel Messi. Each week we see how
tight-lipped the Inter Miami boss has been regarding Messi's injury.

Prior to tonight's match against the Chicago Fire, Martino was asked about the return of Jordi Alba, and he stated that there is a timeline in place for Alba's
return. When asked if there's a timeline for Messi's return, Martino refused to acknowledge if there is.

Tata Martino when I asked about Lionel Messi's return timeline:



"I understand your desire to know the exact timelines, but if I had the exact timelines I would share them.



"I just don't have them."
</item 33>

<item 34>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
"I just don't have them."



🤔🧐🤨#InterMiamiCF #Messi𓃵 pic.twitter.com/U7doZYX491 — Franco Panizo (@FrancoPanizo) October 4, 2023

Chicago Fire vs. Inter Miami lineups & team news

Frank Klopas has rotated his two young attackers, Brian Gutierrez and Jairo Torres along with famed Swiss international Xherdan Shaqiri in the attack. He has
also done his best to keep his striker stop rolling along with Kei Kamara and Georgios Koutsias alternating up front and MLS veteran Kacper Przybylko coming off
the bench.

Injuries to defenders Mauricio Pineda and Miguel Navarro, plus striker Victor Bezerra, will keep them out for today's match, while Chris Mueller is done for the
year with a hip problem.
</item 34>

<item 35>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
The cheapest season ticket for Inter Miami in 2024 is $867, which works out to $51 per match across the 17 home games in the upcoming MLS season. Meanwhile, a
seat in the West Reserved section, which puts a fan in the upper deck near the middle of the pitch, is now priced at a whopping $140 per match.

According to Telemundo, this makes Inter Miami one of the most expensive tickets in the entire world. The cheapest Barcelona season ticket, for example, can be
had for $377 for the season. Even while their prices are muted thanks to the next few seasons being played outside the under-construction Camp Nou, they are
still just 25% cheaper than tickets to Barcelona's famed home ground.

One of Europe's most expensive tickets can be found at Tottenham, thanks in part to the brand new state-of-the-art Tottenham Hotspur Stadium, and even then its
cheapest ticket is around $974, while its most expensive package comes out to $2,498 for the year.
</item 35>

<item 36>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
HALFTIME: Inter Miami 0-0 FC Cincinnati

Needing victory in this match to ensure they remain in the MLS playoff picture, Inter Miami completely dominated the confirmed 2023 regular season champs but
somehow did not manage to capitalize on their overwhelming pressure. Inter Miami were so unlucky not to cash in on multiple occasions, hitting the woodwork an
insane three times, all in the first 30 minutes. Lionel Messi may yet have a job to do tonight, currently available off the bench.

FC Cincinnati had one really good look that they missed as well but otherwise were on their heels the entire 45 minutes. Head coach Pat Noonan tells the
broadcast he feels FC Cincinnati had "a lack of courage on the ball" that half, "as bad as its been all year."

Elsewhere, Inter Miami will be nervous about other results from around the league.
</item 36>

<item 37>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
MORE: Read all the details regarding Inter Miami vs Al Nassr featuring Messi vs Ronaldo

Inter Miami friendly matches for 2024 preseason

Below are all the known fixtures that have either been confirmed by the club directly or widely reported by media.

It was anticipated that Inter Miami would embark on a two-match tour of China, but that has now been cancelled, with the trip proving too difficult logistically
to arrange. It was replaced by a friendly against fellow MLS side NYCFC to honor Lionel Messi's Ballon d'Or.

The biggest confirmed match is the meeting between Messi and rival Cristiano Ronaldo as Inter Miami announced a pair of fixtures in Saudi Arabia, including
against Ronaldo's Al Nassr on February 1.

MORE: When does Inter Miami's 2024 season begin? Lionel Messi to retake field in February

Confirmed Inter Miami preseason fixtures

Unconfirmed Inter Miami preseason fixtures reported

Date Kickoff (ET) Match Location Sat, Jan.
</item 37>

<item 38>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Inter Miami season tickets: Prices for 2024

The pricing structure for 2024 represents an enormous increase over the 2023 costs. The cheapest season ticket in 2023 was $485, but that will jump all the way
to $884, which is an increase of 78.8%. There is a discount for fans who renew by October 7.

Meanwhile, a season ticket in the lower-level midfield club area, which includes upscale food and beverage options, cost $3,600 in 2023 but will now run $7,650
for the 2024 season, an increase of 113%, more than doubling year-over-year.

Inter Miami 2024 season ticket prices

Prices below courtesy of Inter Miami season ticketing release via the Miami Herald on September 28, 2023. Sections 2 through B are considered "premium" seating,
while C through L are considered "standard."
</item 38>

<item 39>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
@fccincinnati goalkeeper @roman_413 had nothing but praise for @LuchoAcosta94 and Pat Noonan 🟠🔵 pic.twitter.com/Xyp2ozr7n5 — CBS Sports Golazo ⚽️
(@CBSSportsGolazo) October 3, 2023

1 hour 15 mins to kick: Lionel Messi was nominated for not only MLS Newcomer of the Year for the 2023 season, but also 2023 MLS MVP.

Laughable doesn't even begin to describe that blatant PR move, considering that most of the damage Messi has done for Inter Miami has come in Cup competitions.
He has one goal and two assists in 247 minutes of play in actual MLS competition. Lionel Messi has been fun, but let's hope the league doesn't make a mockery of
itself by actually giving Messi either honor.
</item 39>

<item 40>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Messi, who had missed the last four games due to injury, returned to play 35 minutes off the bench, but by the time he stepped on the pitch, Inter Miami’s early
dominance in the match had come and gone. Messi played a part, but he was unable to fully change the tide of the game.

Eventually, Alvaro Barreal’s 78th-minute goal ended any hope that Inter Miami had to complete a stunning playoff charge, and the defeat coupled with CF
Montreal’s big 4-1 victory over Portland means Inter Miami are officially out of the postseason running.

The Sporting News followed the Inter Miami vs. FC Cincinnati match live, providing score updates, commentary and highlights as they happened.

MORE: Inter Miami playoff scenarios before potential do-or-die match against FC Cincinnati

Inter Miami vs FC Cincinnati final score

Score Goal scorers Inter Miami 0 – FC Cincinnati 1 Barreal (78')

Kickoff: 7:39 p.m.
</item 40>

<item 41>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
Spain's World Cup-winning midfielder spent five years with Vissel Kobe after spending his entire club career until that point with Barcelona, where he played
alongside Messi, Sergio Busquets and Jordi Alba during an era of astonishing success in Catalonia.

"Vissel Kobe is a powerhouse club in Asia coming off of a historic season after winning the J1 League, so we're thrilled about this opportunity to sharpen our
team in a match like this one. We continue to look for the best ways to prepare for 2024, and feel confident this match and this tour will help us do that,"
said Chief Soccer Officer and Sporting Director Chris Henderson.

Boyhood Messi club to face Inter Miami in USA

After their Asia trip, Inter Miami will host a friendly at DRV PNK Stadium against Newell's Old Boys.

The team from Argentina hold a special place in the heart of Messi and Martino.
</item 41>

<item 42>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
He's pressured while making a one-on-one run, and the delivery from the right flashes wide of the near post as he slides to meet it.

Both teams will believe they should be in front!

9' | Toto creates a chance early off the crossbar 😳🤯#MIAvCIN | 0-0 pic.twitter.com/aogoOmlhCU — Inter Miami CF (@InterMiamiCF) October 7, 2023

8th min: Inter Miami have held the attacking possession in the opening stage of the match, looking to break down FC Cincinnati early, but Matt Miazga holds firm
before he's fouled as Josef Martinez undercuts him in the air.

3rd min: An early free-kick for Inter Miami from about 35 yards away shaded way right, and with Lionel Messi on the bench, Noah Allen stands over the ball.
</item 42>

<item 43>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
MORE: Inter Miami season tickets for 2024 season see unprecedented jump in price

(Also, you'll notice a photo of Lionel Messi's jersey hanging in his locker in that tweet below, a pretty good indication that he'll return tonight)

Scene is set for Fan Appreciation Night 🏟️#MIAvCIN | 7:30 PM ET pic.twitter.com/9I07uq9hGW — Inter Miami CF (@InterMiamiCF) October 7, 2023

2 hours to kickoff: A report from well-connected Argentine journalist Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for
this match in some capacity. Edul reported that "the idea is for Leo Messi to get minutes when Inter Miami plays FC Cincinnati".

That makes it sound like Messi will only be available off the bench, but it's better than nothing to have a game-changer like him available at all.
</item 43>

<item 44>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Messi takes it but puts it way over the crossbar. A bit rusty?

57th min: Jean Mota has a shot from distance, and it looks to be a vicious knuckler with some dip, but it goes just wide.

55th min: HERE COMES LIONEL MESSI! He will enter with around 35 minutes to save Inter Miami's season, replacing Tomas Aviles in a very attacking change. Jean
Mota also comes on to replace David Ruiz, which is harsh as the young midfielder had been very good.

50th min: Chance, FC Cincinnati! DRAKE CALLENDER MAKES A MASSIVE DOUBLE SAVE! FC Cincinnati are RIGHT on the doorstep, but he comes up huge twice! Aaron
Boupendza is the most aggrieved, as his follow-up came from directly in front of the goal but he was stymied by Callender's outstretched right foot!
</item 44>

<item 45>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
A draw or a loss, however, would see them vulnerable to outside results.

If Inter Miami do not win, two of the following three possible outcomes would trigger their elimination from playoff contention: CF Montreal beats Portland
Timbers; NYCFC wins versus D.C. United; and Chicago Fire picks up points against Charlotte FC.

Those are all conceivable results, and therefore Inter Miami will wish to keep it in their own hands just a little longer. They'll eventually need other results
to go their way to climb the table, but for now, their survival is solely up to them.

45 mins to kickoff: Lineups are in, and Lionel Messi is on the bench for Inter Miami.

For FC Cincinnati, Luciano Acosta starts despite the match being meaningless from a tangible standpoint, understandable considering he is chasing the Golden
Boot. Ian Murphy comes in for Yerson Mosquera at the back, while Aaron Boupendza is rotated out up front with Brandon Vazquez and Sergio Santos starting
together up front.
</item 45>

<item 46>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
Chicago Fire starting lineup (4-2-3-1): Brady (GK) — Dean, Omsberg, Teran, Navarro — Doumbia, G. Gimenez — B. Gutierrez, Shaqiri, J. Torres — Kamara.

Chicago Fire subs (9): Richey (GK), Teran, Haile-Selassie, Casas Jr, Aceves, Burks, Souquet, Przybylko, Kamara.

The status of Lionel Messi is uncertain, but it feels unlikely that he will be involved given recent matches. He could, however, be more ready for action over
the weekend when Inter Miami face FC Cincinnati.

Inter Miami will definitely be without the services of both Jordi Alba and Leonardo Campana who are both out with muscle injuries. Midfielder Gregore has
returned to training after months on the sideline with a foot injury, and could be involved in the coming matches as he ramps up his fitness.
</item 46>

<item 47>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
That's nice and all, but the best way to appreciate the fans would be to not price out the loyal supporters from season tickets next year. Instead, Inter Miami
has jacked 2024 membership rates to put the club amongst some of the most expensive tickets in not just MLS, but the world.

While there will certainly be folks who can afford those high-demand tickets, many of those individuals won't be the same fans who have supported the club from
its inception in 2020. This means Inter Miami have chosen to secure maximum profit at the expense of its fanbase, and a high turnover amongst the regular
in-person support from this season to next is likely.
</item 47>

<item 48>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Meanwhile, another nearby rival, Atlanta United, will feature its cheapest season ticket for $605 in 2024, or just $525 for founding members.

Inter Miami season-ticket glitch causes scare

Already, price increases have Inter Miami season ticket holders on edge, but a computer glitch saw supporters fed an even bigger scare.

Mike Ryan, a Miami local who regularly appears on the radio program The Dan LeBatard Show, posted on social media his shock when he received a 2024 renewal for
$30,600 for his two seats, which in 2023 cost just $6,802. Ryan then posted a follow-up a few hours later that stated he was contacted by Inter Miami
representatives who indicated that pricing was incorrect thanks to a computer glitch that sent supporters quotes for additional seats they did not actually
have.
</item 48>

<item 49>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
The play is off the training ground, as his delivery goes vertically to try and create a lateral cross opportunity for DeAndre Yedlin, but it's overcooked and
completely wasted over the end line. Yikes.

Kickoff: The match is under way at DRV PNK Stadium! One club's status for the postseason is as assured as possible, while the other team's status is far from
certain.

Inter Miami vs FC Cincinnati: Pre-match commentary, analysis, stats, and more

5 mins to kickoff: While there is no active protest happening from the Inter Miami supporter section, they are also taking more muted measures to make their
feelings known. They have made a collective effort not to be their usual cheerful selves before kickoff, sitting quietly instead of waving flags and making
noise.

With such a massive game on hand, this is a very clear sign of frustration with the enormous increase in ticket prices next season. Inter Miami's home field
advantage tonight will be significantly hampered.
</item 49>

<item 50>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
Inter Miami play NYCFC in Noche d'Or to replace China trip

With the club's elimination from 2023 MLS playoff contention, Lionel Messi is facing nearly a full month off from competitive fixtures before he takes the field
for Argentina against Uruguay and Brazil in CONMEBOL World Cup qualifying.

Likely with the sole intent of keeping Messi fit in that time off, and the added bonus of raking in significant cash, Inter Miami scheduled a two-match tour of
China in early November after their regular season concluded.

They were expected to visit two Chinese Super League sides, starting with a match against Qingdao Hainiu on November 5, followed by a game against Chengdu
Rongcheng three days later However, it was reported on October 30 the planned tour was off. Gaston Edu of TyC Sports reported the visit had been cancelled due
to "commercial and bureaucratic" reasons.

Inter Miami no va a viajar a China.
</item 50>
=== END ITEM id=33cdc94e91826f63 ===
