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

=== ITEM id=89e0c441908b7d47 ===
Gold sentences and facts:
Sentence 0: Kooker added later in the speech: “Music is a tremendous driver for AI technology, and AI technology presents a tremendous opportunity for the creative development of music.
  Fact 0.0: Kooker added later in the speech.
  Fact 0.1: Kooker said music is a tremendous driver for AI technology.
  Fact 0.2: Kooker said AI technology presents a tremendous opportunity for the creative development of music.
Sentence 1: Robert Kyncl, CEO of Warner Music Group – and an ex-high flyer at YouTube – commented: “With each major leap in technology, the music industry navigates a fresh set of challenges and opportunities.
  Fact 1.0: Robert Kyncl is CEO of Warner Music Group.
  Fact 1.1: Robert Kyncl is a former high flyer at YouTube.
  Fact 1.2: Robert Kyncl commented that with each major leap in technology, the music industry navigates a fresh set of challenges and opportunities.

Retrieved items:
<item 1>
Graph query row:
sony_music_article: Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI
sony_music_passages: ['Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business
Worldwide, 2023-11-30T21:28:03+00:00)\nSenior political figures in Washington are becoming increasingly interested in the impact of AI on the law.\n\nOn
Wednesday (November 29), US Senate Majority Leader Chuck Schumer held an Artificial Intelligence (AI) Insight Forum to discuss the importance and role of
creative copyright and intellectual property in the development of AI.\n\nDennis Kooker, Sony Music Entertainment‘s President of Global Digital Business & US
Sales, delivered a speech at the Forum, outlining the major music company’s position on AI and copyright law.\n\nKooker’s public statements on the topic arrive
a week after MBW pointed out that there was a notable lack of endorsement from SME, and its artists, of YouTube‘s new experimental AI project called ‘Dream
Track’ that lets creators clone the vocals of successful musicians.\n\nThe initial cohort of stars involved in the project, as we noted last week, include
recording artists signed to Warner Music Group and Universal Music Group – or labels within each of those two majors, but no Sony Music artists.', 'Sony Music
has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)\n(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in
response to a request for written submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s
absence from Dream Track.)\n\nGoogle was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and
Stability AI.\n\nAnthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair
use.\n\nStability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of
prominent generative AI executive Ed Newton-Rex from the company.\n\nAs reported by MBW last week, Universal Music Group submitted its response to the USCO.',
'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)\nAs reported by MBW last week, Universal Music Group submitted its response to the USCO. Unsurprisingly, amongst the highlights of
UMG’s submission was the rejection of the notion that using copyrighted material to train LLMs is “fair use”.\n\nIn his speech on Wednesday, Kooker commented
directly on the flurry of submissions to the USCO from the tech industry at the end of October.\n\nIn what may have been a nod to Google’s filing and “fair use”
position, Kooker told Senators that “based on recent Copyright Office filings it is clear that the technology industry and speculative financial investors would
like governments to believe in a very distorted view of copyright”.\n\nHe added: “[That view is] one in which music is considered fair use for training purposes
and in which certain companies are permitted to appropriate the entire value produced by the creative sector without permission, and to build huge businesses
based on it without paying anything to the creators concerned.”', 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from
Dennis Kooker’s speech about AI (Music Business Worldwide, 2023-11-30T21:28:03+00:00)\nKooker also openly discussed Sony Music’s efforts to get unauthorized AI
content removed from platforms online – including the fact that the firm has already issued close to 10,000 separate takedowns.\n\nKooker further discussed Sony
Music’s optimism about AI’s potential for enhancing creativity, as well as key principles the company has adopted around generative AI.\n\nHere are three things
that stood out from Kooker’s speech…\n\n1. Sony Music has issued close to 10,000 takedowns for unauthorized deep fakes of artists\n\nKooker argued in his speech
that, while “the music industry sees great potential with various forms of AI technology in this early phase of AI,” so far, “the available generative AI
products are not delivering on the expected promise of new products in creative industries”.\n\nHe clarified that point by explaining that some generative AI
platforms “are not expanding the business model or enhancing human creativity”.', 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we
learned from Dennis Kooker’s speech about AI (Music Business Worldwide, 2023-11-30T21:28:03+00:00)\nKooker suggested that the “more dire outputs of early
generative AI technology are deep fakes and unauthorized voice clones” of artists and that in the United States, “artists are not adequately protected from
these deepfakes”.\n\n“An artist literally makes their livelihood from their voice,” he said. “Deep fakes intentionally exploit an artist’s talent and reputation
to steal that income stream.\n\n“Every stream of a deep fake takes streams and royalty payments away from the legitimate artist. Deepfakes are also misleading
and confusing to consumers and music fans who are not typically interested in supporting fake versions of their favorite artist.”\n\n“Every stream of a deep
fake takes streams and royalty payments away from the legitimate artist.” Dennis Kooker, Sony Music\n\nKooker reported that to date, Sony Music Entertainment
has sent “close to 10,000 takedowns to a variety of platforms hosting unauthorized deepfakes that SME artists asked us to take down.”', 'Sony Music has issued
nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide, 2023-11-30T21:28:03+00:00)\nHe
added that “platforms are quick to point to the loopholes in the law as an excuse to drag their feet or to not take the deepfakes down when
requested.”\n\nKooker commended the No FAKES Act policy drafted by senators earlier this year, which he explained “would create a federal property right in
one’s voice or visual likeness and protect against unauthorized AI-generated replicas”.\n\n2. Sony has ‘roughly 200 active conversations taking place’ with AI
startups that “include potential equity investments”\n\nKooker explained that despite the negative impacts of AI, including unauthorized clones and low-quality
AI music, there are still “many positive and optimistic developments to highlight”.\n\nHe highlighted legitimate AI startups and established firms “with music
ideas that want to partner with the industry”.\n\nAccording to Kooker, Sony Music has “roughly 200 active conversations taking place with startups and
established players” right now about building new AI-related products and tools.', 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we
learned from Dennis Kooker’s speech about AI (Music Business Worldwide, 2023-11-30T21:28:03+00:00)\nThese products range from tools for “creative or marketing
assistance”, to tools that “potentially give us the ability to better protect artist content or find it when used in an unauthorized fashion” as well as “brand
new products that have never been launched before”.\n\nKooker said that some of these conversations “also include potential equity investments which would
accelerate the development of these companies”.\n\nThe Sony Music exec cited, as an example, a recent generative AI-powered project around a reissue and remix
of an album. Although Kooker didn’t name it specifically, SME did recently announce a generative AI project with The Orb and David Gilmour, which was a
partnership between Sony Music Entertainment, Legacy Recordings AI company Vermillio\n\n“These artists are known for their cutting-edge experimentation in
music,” said Kooker.', 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business
Worldwide, 2023-11-30T21:28:03+00:00)\n“About the time that we started the discussion with this artist, we had begun informal talks with a generative AI
start-up company whose business model focused on working with IP rightsholders the ‘right way’.\n\n“In other words, they respect intellectual property rights
and want to work with rights holders in ways that enhance and protect the copyrighted works.”\n\n3. SME has asked Congress to embrace a set of principles around
generative AI\n\nKooker told the lawmakers during his speech on Wednesday that “if copyrights are protected and enforced appropriately, we are at the beginning
of a multi-decade marathon that will change the creative and commercial landscape for music”.\n\nWith that in mind, Kooker explained that Sony Music has
established a set of principles to guide the company’s decision-making around generative AI.\n\n“If copyrights are protected and enforced appropriately, we are
at the beginning of a multi-decade marathon that will change the creative and commercial landscape for music.”', 'Sony Music has issued nearly 10,000 deepfake
takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide, 2023-11-30T21:28:03+00:00)\nDennis Kooker, Sony
Music\n\nKooker added later in the speech: “Music is a tremendous driver for AI technology, and AI technology presents a tremendous opportunity for the creative
development of music.\n\n“But these opportunities must be grounded by the human creators’ vision with the machine assisting, not with the machine replacing the
human creator.”\n\nTo achieve this, Kooker said that SME wants Congress to embrace the following principals:\n\nAssure Consent, Compensation, and Credit. “New
products and businesses built with music must be developed with the consent of the owner and appropriate compensation and credit. It is essential to understand
why the training of AI models is being done, what products will be developed as a result, and what the business model is that will monetize the use of the
artist’s work. Congress and the agencies should assure that creators’ rights are recognized and respected.” Confirm That Copying Music to Train AI Models is Not
Fair Use.', 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)\nConfirm That Copying Music to Train AI Models is Not Fair Use. “Even worse are those that argue that copyrighted content should
automatically be considered fair use so that protected works are never compensated for usage and creators have no say in the products or business models that
are developed around them and their work. Congress should assure and agencies should presume that reproducing music to train AI models, in itself, is not a fair
use.” Prevent the Cloning of Artists’ Voices and Likenesses Without Express Permission. “We cannot allow an artist’s voice or likeness to be cloned for use
without the express permission of the artist. This is a very personal decision for the artist. Congress should pass into law effective federal protections for
name, image, and likeness.” Incentivize Accurate Recordkeeping. “Correct attribution will be a critical element to artists being paid fairly and correctly for
new works that are created.', 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music
Business Worldwide, 2023-11-30T21:28:03+00:00)\nIn addition, rights can only be enforced around the training of AI when there are accurate records about what is
being copied. Otherwise, the inability to enforce rights in the AI marketplace equates to a lack of rights at all, producing a dangerous imbalance that prevents
a thriving ecosystem. This requires strong and accurate record keeping by the generative AI platforms, a requirement that urgently needs legislative support to
ensure incentives are in place so that it happens consistently and correctly.” Assure Transparency for Consumers and Artists. “Transparency is necessary to
clearly distinguish human-created works from AI-created works. The public should know, when they are listening to music, whether that music was created by a
human being or a machine.”\n\nAdded Kooker: “While these principles are simple and basic, they require a new level of commitment and investment from generative
AI platforms.', 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business
Worldwide, 2023-11-30T21:28:03+00:00)\n“If established early on, they will result in an even playing field for all participants so there will not be an unfair
competitive advantage for a few at the expense of future innovators.\n\n“The principles will ensure that intellectual property businesses can succeed in this
new world alongside tech partners, further advancing investment and expanding economic opportunities. “Music Business Worldwide']
warner_music_article: Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference
warner_music_passages: ['Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference
(Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nMBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to
major recent entertainment events or news stories. MBW Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an
asset class.\n\nIn recent months, Warner Music Group CEO Robert Kyncl has been keeping a close eye on AI in the music industry – not surprising, given that he
came to the music biz from ‘Big Tech’, namely YouTube, where he was Chief Business Officer.\n\nNor is it surprising, given his current role, that he has been
pushing for price hikes at music streaming services.\n\nBoth of these topics came up during Kyncl’s Q&A session at the Code Conference in Laguna Niguel,
California, on Tuesday (September 26).', 'Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the
Code Conference (Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nBut what was maybe more surprising – for a techie like Kyncl, anyway – was his rational,
level-headed explanation for why traditional music labels continue to be relevant, and continue to thrive, in the current music ecosystem.\n\nWith the arrival
of digital platforms, including open ones like YouTube and SoundCloud – and with the proliferation of indie music distribution platforms like DistroKid and
TikTok’s SoundOn – it’s inevitable that music biz observers from the world of tech like to speculating about the possibility that music labels could, in the
long-term, be headed for irrelevance.\n\nIn a world where anyone can record music using a desktop audio workstation (DAW) and upload it direct to YouTube, or to
Spotify via a distribution platform, labels have little left to do, the argument goes.\n\nKyncl offered strong a counterpoint to this argument at Code Media,
drawing a parallel between music labels today and the advertising industry a decade or two ago.', 'Warner Music Group boss Robert Kyncl on AI, why labels still
exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nAt that time, he said, many
predicted ad agencies were headed for the scrap-heap thanks to the arrival of Google’s AdWords, which allowed any business to set up a customized ad campaign on
Google services.\n\nYet ad agencies didn’t go extinct. Today, “they’re the largest customers of companies like Google,” Kyncl told the audience at the Code
Conference.\n\n“In the year 2000, when AdWords came out, everybody was predicting their deaths because all the brands [could] just go direct to AdWords and buy
ads… However, what happened was lots of different platforms emerged. Lots of ad tech emerged, complexity increased exponentially, and brands needed help with
that, and the agencies provided that value.\n\n“And I see a very close analogy [between that and] our business. Music is incredibly broadly distributed.
Everybody on earth listens to music. We’re on every platform. And the complexity is high.', 'Warner Music Group boss Robert Kyncl on AI, why labels still exist,
and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nWe’re on every platform. And the
complexity is high. And the more people can upload content, and the more people can be heard, the greater the noise, which means it’s harder to cut through the
noise and sustain a career.”\n\nBut a label’s value is not just about building an artist’s career – it’s about sustaining it once that career is established as
well, Kyncl explained.\n\n“When they become established, it is also hard to remain at the top,” Kyncl said. “So their goals change. And again, we’re there to
help with that.”\n\nHe added later that if you’re a music artist, “you need a team. You need an army behind you…. if you want a sustainable career, with
repeatability and success.”', 'Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code
Conference (Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nif you want a sustainable career, with repeatability and success.”\n\nHere are three other
things we learned from Kyncl’s talk at the Code Conference…\n\nCredit: Tatiana Shepeleva / Shutterstock 1) Music will feel the impact of AI before other
industries do – ‘within the next year’\n\nKyncl predicted that the music business will be out ahead of other industries when it comes to exploiting the
potential of AI, and it will also – out of necessity– be ahead on the issue of how to regulate and monetize the use of AI by non-rights-holders.\n\n“Music –
because it’s so broadly distributed and it’s so well aligned with the internet, because it’s short format, it lends itself to recommendations, it’s on all
platforms – is generally first in most transformations and most innovations. So it digitizes first,” he said.\n\n“Whatever happened to the music industry 20
years ago is starting to happen in the movies and TV shows now.', 'Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we
learned from his Q&A at the Code Conference (Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nIn the meantime, music emerged out of [digitization] better
and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll likely be first.”\n\nAsked when that sea change will occur, Kyncl
said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”\n\nKyncl
drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.\n\n“I would say within the next
year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”', 'Warner Music Group boss Robert Kyncl on
AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nRobert
Kyncl, Warner Music Group\n\nIn Kyncl’s view, the solutions put in place to address UGC form a “blueprint” than can be used to address AI-generated content on
platforms as well.\n\n“When YouTube was formed… people started to upload content, including copyrighted material, which, obviously, put YouTube into hot water
with lots of different copyright holders. And I had the privilege of working through a lot of that and fix it up.\n\n“But we made we made a very important
decision, which was to go above and beyond the law, and build a fingerprinting software that allowed us to track the copyright on our platform, and then have
commercial relationship[s] with copyright holders to send them the money. Out of that we built a multi-billion-dollar business, which now is a
multi-billion-dollar business per year. And it was an incredible new revenue stream for everyone. AI is that with new super tools.”', 'Warner Music Group boss
Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)\nAnd it was an incredible new revenue stream for everyone. AI is that with new super tools.”\n\nKyncl was referring to YouTube’s
Content ID system, which scours uploaded videos for copyrighted content (video and audio), then offers the copyright owner the option to monetize that video, or
to request that it be taken down. The innovative system has essentially allowed YouTube users to upload content without paying much attention to copyright,
while ensuring that copyright owners are paid.\n\nWhen it comes to the proliferation of AI, “we need to approach it with the same thoughtfulness, and we have to
make sure that artists have a choice,” Kyncl added.\n\nAnd he made it clear that, in his view, simply rejecting AI and fighting against it is not an
option.\n\n“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.”', 'Warner Music Group boss
Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)\nThe genie is not going back.”\n\nBut he cautioned that the technology to develop a Content ID-style system for is “not yet
developed, but… people are working on that.”\n\n{Believe CEO Denis Ladegallerie might disagree. In comments earlier this year, Ladegallerie said that the tools
to recognize and flag AI-generated content are just about ready to go, and he expects to see them implemented in the coming few quarters.)\n\n2) The streaming
audio business won’t experience the same problems currently hitting streaming video\n\nThe streaming video business is going through tough times.\n\nNetflix
experienced its first-ever decline in subscriptions last year (though it has since roared back to health); the subscriber count of Disney+ appears to be in
freefall; and talk is growing about ongoing consolidation of the many streaming video services that have come online over the past several years.\n\nAsked if
the audio streaming business is likely to go through something similar, Kyncl’s answer was unequivocal.\n\n“No.', 'Warner Music Group boss Robert Kyncl on AI,
why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide, 2023-09-28T23:37:22+00:00)\n“No. Music is
much more resilient. So first, I think what happened in the last 15 years is incredible. They had literally zero people in the subscription model and now we
have 700 million people in the world in the premium experience… which is incredible.\n\n“And I think really credit goes to Daniel Ek for forging the path for
everybody. And then companies like Apple and YouTube and Amazon following and building up the [music] business. It’s pretty incredible what has
happened.”\n\n“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.” Robert Kyncl, Warner
Music Group\n\nKyncl added: “I think the opportunity ahead of us is twofold. One, the continued growth in emerging markets, and also the GDPs of those countries
will be rising at the same time. So there’s lots of growth there.', 'Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we
learned from his Q&A at the Code Conference (Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nSo there’s lots of growth there. And then there is the price
elasticity optimization in mature markets.”\n\nThat last point is a reference to the price hikes seen among all the major music streaming services over the past
year or so, and the growing belief within the music industry that streaming price hikes will be sustained – as evidenced recently by Deezer’s second price hike
in a year.\n\nCredit: QuiteSimplyStock/Shutterstock 3) The hot music rights acquisitions space is ‘likely slowing down a bit’\n\nThe second half of 2022 saw a
notable slowdown in music rights acquisitions, a phenomenon that some blamed on higher interest rates and the reduced liquidity that came with them.\n\nAlthough
there has been something of a rebound this year, in Kyncl’s view, there is still a slowdown in this space – though that might be good for Warner Music
Group.\n\n“I think what happened was there was an opening where artists and songwriters were suddenly open to selling catalogs.', 'Warner Music Group boss
Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)\nwhich they were not open to before. And they just created this incredible tidal wave. And these things happen in waves, ebb and
flow.\n\n“So I think it likely is slowing down a little bit, but that’s okay. Because it makes the multiples come down a little bit more.”\n\n“I think what
happened in the last 15 years is incredible. They had literally zero people in the subscription model and now we have 700 million people in the world in the
premium experience.” Robert Kyncl, Warner Music Group\n\nIn other words, Warner Music can go out and buy more if the multiples are low enough?\n\n“Of course. We
are in the business of buying catalogs, obviously. We have the publishing catalog of David Bowie, for instance, which we bought a few years ago. So yeah, we are
in that business.', 'Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference
(Music Business Worldwide, 2023-09-28T23:37:22+00:00)\nSo yeah, we are in that business.\n\n“We are in the business of administering [rights], which means
collecting revenue from thousands of platforms around the world; very complicated matters. But we’re also in the business of ownership.”Music Business
Worldwide']
direct_relationships: []
shared_connections: ['copyright', 'AI', 'Warner Music Group', 'Google', 'Music Business Worldwide', 'YouTube', 'music industry', 'rights', 'music']
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

Graph facts:
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -PUBLISHED_BY-> Music Business Worldwide
- Latin artists have been paving the way for deals that offer more control, transparency, and better economics. -PUBLISHED_BY-> Music Business Worldwide
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group -PUBLISHED_BY-> Music Business Worldwide
- Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as
quickly as my team and I have done in less than two years.’ -PUBLISHED_BY-> Music Business Worldwide
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -PUBLISHED_BY-> Music Business Worldwide

Entities: Music Business Worldwide, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI,
Dennis Kooker, Sony Music Entertainment, The Orb, David Gilmour, Legacy Recordings, Vermillio, generative AI project around a reissue and remix of an album,
potential equity investments, generative AI
</item 2>

<item 3>
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

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> AI
- There’s something going on with AI startups in France -MENTIONS-> AI
- Founders, are events useful? -MENTIONS-> AI
- One year later, ChatGPT is still alive and kicking -MENTIONS-> AI
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> AI

Entities: AI, Music Business Worldwide, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about
AI, Dennis Kooker, deepfakes, No FAKES Act, AI startups, AI-generated music, Sony Music
</item 3>

<item 4>
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

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> AI
- There’s something going on with AI startups in France -MENTIONS-> AI
- Founders, are events useful? -MENTIONS-> AI
- One year later, ChatGPT is still alive and kicking -MENTIONS-> AI
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> AI

Entities: AI, human creativity, Music Business Worldwide, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis
Kooker’s speech about AI, Dennis Kooker’s speech about AI, deep fakes, Dennis Kooker, Sony Music, music industry, generative AI
</item 4>

<item 5>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
(In our report, MBW suggested that YouTube parent Google‘s recent submission to The United States Copyright Office (USCO) – in response to a request for written
submissions as part of a study around copyright law and policy issues raised by AI systems – may have influenced Sony’s absence from Dream Track.)

Google was just one of various tech giants and AI companies that submitted responses to the USCO, amongst them, Anthropic and Stability AI.

Anthropic, in its submission to the USCO, also argued that training (large language models) LLM’s on copyrighted material is fair use.

Stability AI made a similar argument. Its position on the issue of “Fair Use” concerning AI and copyrighted content resulted in the resignation of prominent
generative AI executive Ed Newton-Rex from the company.

As reported by MBW last week, Universal Music Group submitted its response to the USCO.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> AI systems
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> AI systems
- Biden’s AI EO -MENTIONS-> AI systems
- Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment -MENTIONS-> AI systems
- Sam Altman ousted as OpenAI’s CEO -MENTIONS-> AI systems

Entities: AI systems, large language models, AI, Google, Universal Music Group, Anthropic, Music Business Worldwide, Dream Track, Sony Music has issued nearly
10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI, United States Copyright Office, copyright law and policy issues,
Stability AI, fair use, copyrighted material, Ed Newton-Rex
</item 5>

<item 6>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
In the meantime, music emerged out of [digitization] better and stronger and more resilient. So, I would imagine the same will happen here, which is we’ll
likely be first.”

Asked when that sea change will occur, Kyncl said: “I would say within the next year you will see lots of evolution around AI… What you will likely see is
increasing quality at a very fast pace.”

Kyncl drew an analogy between the explosion of AI use among members of the public and the explosion of user-generated content (UGC) a decade or two ago. Just as
user-generated content often infringed on copyright, so too does AI threaten the rights of artists and music rights holders.

“I would say within the next year you will see lots of evolution around AI… What you will likely see is increasing quality at a very fast pace.”

Graph facts:
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> copyright
- NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ -MENTIONS-> copyright
- 8 predictions for AI in 2024 -MENTIONS-> copyright
- Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI -MENTIONS-> copyright
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> copyright

Entities: copyright, AI, Warner Music Group, Music Business Worldwide, Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we
learned from his Q&A at the Code Conference, Robert Kyncl, digitization, user-generated content (UGC), Code Conference
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

Graph facts:
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> Washington
- La desigualdad del estrés por calor -MENTIONS-> Washington
- NFL roundup: Pats hit new low as Lions reach 9-3 for first time since 1962 -MENTIONS-> Washington
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Washington
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Washington

Entities: Washington, AI, Warner Music Group, copyright law, Universal Music Group, US Senate, Chuck Schumer, Music Business Worldwide, Artificial Intelligence
(AI) Insight Forum, Dream Track, creative copyright, intellectual property, Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned
from Dennis Kooker’s speech about AI, Dennis Kooker, Sony Music Entertainment
</item 7>

<item 8>
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

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> AI
- There’s something going on with AI startups in France -MENTIONS-> AI
- Founders, are events useful? -MENTIONS-> AI
- One year later, ChatGPT is still alive and kicking -MENTIONS-> AI
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> AI

Entities: AI, Warner Music Group, Music Business Worldwide, Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned
from his Q&A at the Code Conference, Robert Kyncl, music industry, Code Conference
</item 8>

<item 9>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

Legislators and regulators around the world are grappling with how to address the many issues that have emerged since AI technology “went mainstream” about a
year ago.

In the US, one of the key efforts on that front is being spearheaded by the US Copyright Office, which this past summer issued a call for submissions on the
issue of AI and copyright. The goal is for the USCO to put together a study to ”help assess whether legislative or regulatory steps in this area are warranted.”

Numerous businesses involved in AI, and those with substantial copyright holdings, have submitted their thoughts on the issue, including Universal Music Group
and AI developer Anthropic.

Graph facts:
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> copyright
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> copyright
- 8 predictions for AI in 2024 -MENTIONS-> copyright
- Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI -MENTIONS-> copyright
- Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment -MENTIONS-> copyright

Entities: copyright, US, MBW Reacts, music royalties, NMPA, Universal Music Group, NMPA: Generative AI is ‘the greatest risk to the human creative class that
has ever existed’, Anthropic, US Copyright Office, Music Business Worldwide, JKBX, generative AI
</item 9>

<item 10>
Creatives across industries are strategizing together around AI concerns (TechCrunch, 2023-10-06T23:04:57+00:00)
The goal is that AI tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

“It’s really interesting from a music perspective, specifically, because . . . musicians are perhaps more familiar with the idea of AI,” Holland said.
“Musicians in general are more familiar with things like music production software, and AI tools like MIDI drum loops . . . so I think that there is a certain
amount of more progressive learning from them, when it comes to technology, and its ability to make their music better.”

When it comes to art and AI, the conversation is complicated, to say the least. Musicians are nervous about industry giants copyrighting AI music and cutting
them out of the process. Major record labels are worried about AI models training on their catalogues and stealing a slice of their considerable pie.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> AI tools
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> AI tools
- NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ -MENTIONS-> AI tools
- Founders, are events useful? -MENTIONS-> AI tools
- Sweeping White House AI executive order takes aim at the technology's toughest challenges -MENTIONS-> AI tools

Entities: AI tools, TechCrunch, AI, AI models, major record labels, corporations, Holland, copyrighting, music production software, MIDI drum loops, Creatives
across industries are strategizing together around AI concerns, art, music
</item 10>
=== END ITEM id=89e0c441908b7d47 ===

=== ITEM id=b4921df713686c57 ===
Gold sentences and facts:
Sentence 0: “It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game in junior — it indicated he was going to be able to do that at some point in the NHL.
  Fact 0.0: Davidson said the player was not playing in the NHL during the last few years.
  Fact 0.1: Davidson said the player's level of performance in junior showed how he could control and dominate a game.
  Fact 0.2: Davidson said the junior performance indicated the player was going to be able to do that at some point in the NHL.
Sentence 1: There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala, among many others.
  Fact 1.0: The player helped guide more than a few future NHL stalwarts at that place.
  Fact 1.1: Mattias Ekholm was one of the future NHL stalwarts the player helped guide.
  Fact 1.2: Ryan Ellis was one of the future NHL stalwarts the player helped guide.
  Fact 1.3: Filip Forsberg was one of the future NHL stalwarts the player helped guide.
  Fact 1.4: Calle Jarnkrok was one of the future NHL stalwarts the player helped guide.
  Fact 1.5: Viktor Arvidsson was one of the future NHL stalwarts the player helped guide.
  Fact 1.6: Kevin Fiala was one of the future NHL stalwarts the player helped guide.
  Fact 1.7: The player helped guide many other future NHL stalwarts besides those six.

Retrieved items:
<item 1>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“My name’s Connor,” said the other, standing shorter and extending his.

“All right, well we’re just going to go out and I’m going to do an evaluation on you and do some shooting and have some fun, and I’ll make suggestions but I’m
not here to change anything,” Turk said, giving the two boys his usual spiel and thinking nothing of it.

That evaluation is what he calls an “NHL protocol observation assessment,” which runs new clients through specific drills that allow him to visualize their
shooting actions from a technical standpoint (he calls each player’s shot “like a fingerprint”).

That starts with Turk throwing some pucks down in the slot and simply asking them to shoot them stationary so he can watch their body formation, hip and
shoulder positions, spine angles and puck preparation and positioning, paying particular attention to where the puck starts in their stance versus where it
finishes.

Advertisement

On that day, Nate (Danielson, the Red Wings’ 2023 No.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Connor, Nate Danielson, NHL protocol observation
assessment, Turk, NHL, Red Wings, Hockey
</item 1>

<item 2>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“So many players pull the puck in but can’t get the puck to release from under their body. His footwork is so underrated in that aspect. If you watch his front
leg, at times it’s literally in the air at release — most have their back leg in the air at release. When he transfers his weight, he actually clears space for
his hands to get tighter because he’s not afraid to actually lunge into the shot.”

A fourth shooting coach, who hasn’t worked with Bedard and also requested anonymity because he works for another NHL club, expressed a little more hesitancy.

Advertisement

“I’m in the minority I’m sure, but I am very curious to see how well his trademark shooting style works at this level,” the coach said. “He really likes to pull
pucks right into his feet.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., fourth shooting coach, Bedard, NHL, Hockey
</item 2>

<item 3>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
What is it, really, about Connor Bedard’s shot — a shot that, though he’s just 18 years old, has for years been talked about as if he has patented it — that
makes it so dangerous and unique?

What’s the nerdy science of it, from his feet up to his knees, hips, hands and head? How does he prep, shape and let it go? What does it look like to goalies,
and to professional shooting coaches, those who’ve taught him — or, more accurately, watched him after he taught himself — and had to stop him? The Athletic
spoke with five NHL shooting and skills coaches, his teammates, past opponents and him to try to take apart, piece by piece, “The Bedard.”

Advertisement

Their intel and stories help frame the NHL’s brightest young star’s signature skill.

If there’s one thing Tim Turk knows, it’s shooting.

Graph facts:
- How Did the Houston Astros Cheat in 2017? Everything to Know About a Dark Moment in MLB History -MENTIONS-> The Athletic
- Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti -MENTIONS-> The Athletic
- Browns survive Bears Hail Mary on wild, near-catch by Darnell Mooney: 'Damn near s— myself' -MENTIONS-> The Athletic
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> The Athletic
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> The Athletic

Entities: The Athletic, The New York Times, The Bedard, shooting, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Tim Turk, NHL,
Connor Bedard, Hockey
</item 3>

<item 4>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Bedard wasn’t so reliant on teammates on those goals, either. They were more about the work he’s putting in on the other side of the puck.

“Even in practice, we’ve talked to him about maybe tracking harder and attacking pucks on the forecheck and showing him a couple clips,” Richardson said. “Ten,
12 games in, he’s really figuring things out and realizes — why sit back and let things come to him? Go get it.”

None of this is surprising to anyone involved. Bedard doesn’t show it if he impresses himself. And as grateful as the Blackhawks are to have drafted Bedard,
this is what they expected. This is why teams lined up to take losses last season.

“It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game
in junior — it indicated he was going to be able to do that at some point in the NHL.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Bedard, Richardson, Davidson, NHL, Blackhawks, Hockey
</item 4>

<item 5>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“To me, when he takes a shot, it looks like he just bends down, picks it up with his hand and places it where he wants to.

“He believes that ‘Hey, f— it, the puck is only two and a half inches wide, I can put it wherever I want.'”

Today, Turk would take Bedard’s shot against anyone’s.

Advertisement

“On a shot release basis only, if I had to bet on who could get the puck off the quickest, with the most deception, with optimal speed, power and accuracy based
on a starting point to a finishing point, I’m picking Connor Bedard over Auston Matthews,” Turk said. “And I’m not taking anything away from Auston. It’s just a
little bit different because one’s a righty, one’s a lefty and one’s got a little bit of a higher-angle pull-in change.”

If there’s one thing Nick Quinn wants people to know about “The Bedard,” it’s that it isn’t just some natural gift.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Turk, Auston Matthews, Nick Quinn, The Bedard, Connor
Bedard, Hockey
</item 5>

<item 6>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
If there’s one thing Tim Turk knows, it’s shooting. A self-described “shooting and scoring coach,” Turk has more than two decades of experience working with NHL
stars, national federations and half a dozen NHL teams.

Turk had no idea who Bedard was when he fielded a call from agent Greg Landry of Newport Sports asking him what he was doing tomorrow.

“Tomorrow?” he said. “I’m booked up a year in advance.”

“We have these two players coming in from out west and we want you to see them,” Landry said.

“Listen, I’m booked up,” Turk said again.

“You have to see them,” Landry insisted.

After moving around his schedule, Turk found himself at Gary Roberts’ personal rink in Uxbridge, Ontario.

“Hey guys, I’m Tim Turk,” he said as he stepped into the small locker room.

“My name’s Nate,” said the redhead, extending a hand.

“My name’s Connor,” said the other, standing shorter and extending his.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Greg Landry, Newport Sports, Nate, Connor, Gary
Roberts, national federations, NHL teams, Gary Roberts’ personal rink, Uxbridge, Ontario, Tim Turk, Bedard, NHL
</item 6>

<item 7>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
9 pick) went first, and Turk could tell he was a talented and hard-working player with a good shot and real promise.

Then, on Connor’s turn, Turk placed him in a position and watched how he got the puck ready to shoot. Only Connor didn’t stickhandle and prepare for the shot,
he just took the puck and shot it right away.

After watching him run through the same number of shots as Nate did, Turk spoke for the first time.

“Timeout one second, just give me a second here, I’m just trying to see what you’re doing so I’m going to have you take an extra few shots so that I can analyze
where the puck’s starting and where it’s finishing,” he said.

Another round later, he knew that this Connor kid, whose last name he still didn’t know and who was then 16, was very different from even his high-end peers
simply by his proficiency in shooting from a “frontal position.”
</item 7>

<item 8>
Graph query row:
article: Wild HC Dean Evason, hire John Hynes as replacement
publisher: Yardbarker
passage: Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The Wild have relieved head coach Dean Evason and assistant coach Bob Woods of their duties effective immediately, the team said in a statement Monday evening.
The Athletic’s Michael Russo reports the Wild will be naming former Devils and Predators bench boss John Hynes as Evason’s successor.

Wild GM Bill Guerin gave the following comment:

"Dean did an excellent job during his tenure with the Minnesota Wild, especially as Head Coach of our team. I am very thankful for his hard work and dedication
to our organization. I would also like to thank Bob for his hard work during his time as an Assistant Coach with the Wild. I wish Dean, Bob and their families
all the best in the future."

Evason had been a member of the Wild organization since he was brought on as an assistant in 2018, while Woods joined the Wild bench one season prior.
relationship: MENTIONS
connection: The Athletic
</item 8>

<item 9>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“It’s a lot of different angles he can give you.”

Sabres first-round pick Zach Benson, who has played on teams with and against Bedard and skates with him in the summers, says it’s Bedard’s ability to set up
the shot with his agility pre-shot that stands out to him.

“He can move left to right like no one I’ve seen before,” Benson said.

“It’s just trying to get out, face him, and hope it hits ya,” said Scott Ratzlaff, Seattle Thunderbirds goalie and Buffalo Sabres prospect, with a smile. “He
can shoot from anywhere and he’s lethal from anywhere. You’ve just always got to be ready just in case he shoots it. And then he’s got a really good toe drag
release, so it’s watching for that and making sure you’re lined up.”

A number of Chicago Blackhawks over the last few decades liked to put in extra time shooting.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Zach Benson, Bedard, Scott Ratzlaff, Buffalo Sabres,
Seattle Thunderbirds, Chicago Blackhawks, Hockey
</item 9>

<item 10>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“I can tell you firsthand it didn’t happen by accident. Connor’s worked on this since he was little,” said Quinn, an NHL skills coach who has worked with Bedard
each summer for several years.

If there’s another, it’s how hard Bedard’s shot is to defend because he doesn’t show you he’s preparing for it.

“As replicated in so many other areas of Connor’s game, it’s the deception and elite multi-tasking that catches opponents off guard,” Quinn said. “Connor’s
ability to create deception and change the shot angle at top speed is like very few I’ve ever seen. The multi-tasking involved with attempting these shots at
top speed is far beyond most player’s capabilities.”

Another shooting coach, who requested anonymity for this story because he works for another NHL club, pointed to Bedard’s hands and legs.

“What’s really interesting is how high and left he can get his top hand,” the coach said.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special., Quinn, anonymous shooting coach, NHL, Connor Bedard,
Hockey
</item 10>
=== END ITEM id=b4921df713686c57 ===
