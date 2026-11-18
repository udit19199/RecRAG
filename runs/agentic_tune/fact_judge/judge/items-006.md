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

This file has 5 items.

=== ITEM id=4460e797628b512d ===
Gold sentences and facts:
Sentence 0: His career has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early age, Mitchell is clear-eyed.
  Fact 0.0: Mitchell's career has only blossomed in the years since he took that trip.
  Fact 0.1: Some artists might feel trepidation at attaining so much success at such an early age.
  Fact 0.2: Mitchell is clear-eyed.
Sentence 1: Since then he’s been linked to club jobs in Europe but opted to stick around with the national team.
  Fact 1.0: Since then, he has been linked to club jobs in Europe.
  Fact 1.1: He opted to stick around with the national team.

Retrieved items:
<item 1>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
He got to answer critics like Mark Bosnich and Craig Foster as well – bluntly rejecting their negativity with the dismissive “who cares?”. That would have
tasted sweet.

On Tuesday, FA CEO James Johnson, who held firm on Arnold 20 months ago, got to congratulate the coach on his impending record-breaking match against Bangladesh
as 2026 World Cup qualifying starts on Thursday.

Advertisement

“Graham is one of the greatest contributors to the Socceroos,” Johnson said in a statement.

“He’s been a constant in the Australian football landscape for more than 30 years as both a player and coach, being a part of some of the Socceroos most
memorable moments.

“Anyone who knows Arnie will tell you he bleeds green and gold. His love for the Socceroos and football in Australia is both infectious and inspiring.
</item 1>

<item 2>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
“I’m super grateful that I’ve got a little chant like that, and hopefully, it’s a little legacy I can continue, and I’m just happy the Magpie army are looking
after me. I feel like they’re always on my side, no matter what mistake I make ... if anything, they motivate me to continue to take the game on and just be my
true self.” Markov felt like he was cheating on Collingwood when he began training with the Blues, after finding out the Pies overlooked him for what was then
their final list spot.

That followed Gold Coast delisting the 27-year-old, who also played 23 games for Richmond. There was interest in him at the time, but he feared his career might
be over. “I went through the national draft and no one looked into me, so there were technically 17 clubs that didn’t really want me,” he said.
</item 2>

<item 3>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
In March last year Graham Arnold seemed a dead man walking – the Socceroos World Cup qualification campaign was lurching towards failure and the manager was
being battered from all sides.

Arnold’s team had finished seven points off automatic qualification for Qatar after a shocking run of one win in seven games left them facing a Asian playoff
against United Arab Emirates and then another against Peru to make the finals.

While rumours swirled and ex-players and fans bayed for his blood, Football Australia took the pragmatic course and stuck by Arnold. It likely had more to do
with the lateness of the hour and financial considerations than any overwhelming faith in Arnie to run around the fortunes.

Facebook Twitter Whatsapp Reddit Email Share

But he did just that with qualification – and an excellent run at the finals – bowing out to the eventual champions Argentina and after nearly taking them to
extra time.
</item 3>

<item 4>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
His love for the Socceroos and football in Australia is both infectious and inspiring. That energy and passion is a huge element to the way he’s able to lead
and inspire our nation’s best male footballers in some of our sport’s most challenging arenas.

“On behalf of Football Australia, I’d like to congratulate Arnie on everything he’s achieved in reaching this milestone and we look forward to him continuing to
build on what is already an exceptional legacy.”

In a remarkable few weeks last year Arnold went from on the nose to a man revived. Since then he’s been linked to club jobs in Europe but opted to stick around
with the national team.

While national team coaches make important calls every game, Arnold’s record-breaking achievement is conceivably down to one big ballsy call in the playoff
against Peru.
</item 4>

<item 5>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
None of those targets were achieved and Jones changed his tune about building for the future when he named a young team for the doomed Cup campaign – where his
team became the first Australians to bow out of a RWC in the pool stages.

He finished the year with a 2-7 win loss record – the only two victories coming against lowly Georgia and Portugal.

Advertisement

On Tuesday he said he no regrets with leaving experienced players such as Quade Cooper and Michael Hooper out of the side for the tournament.

“Everyone’s got their own judgment and they’re allowed to have that judgment,” he said. “My choice was that I decided to go with youth. And I stand by that
because I think we’ve got a a lot of good young players who are really going to benefit from that experience at the World Cup, and going forward to the 2027
World Cup.”
</item 5>

<item 6>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
However, names such as Alli, Heung-Min Son, Toby Alderweireld, Eric Dier and Kieran Trippier set the side up for a bright future, before chairman Daniel Levy
got in Mitchell’s way.

Leaving what he called his ‘dream job’ after less than two years, it was reportedly due to Levy’s over-involvement in negotiations and the recruitment process,
something which wasn’t an issue at his next club – Leipzig.

Ralf Rangnick landed Mitchell and more success followed. Signings such as Dani Olmo, Christopher Nkunku and Amadou Haidara again showcased what Mitchell can do
with the shackles off, and again the results followed.

6 Getting Nkunku from PSG was one of Mitchell's biggest coups Credit: Getty

Many of his signings starred in a run to the Champions League semi-finals in 2020, and in 2022 won the club’s first ever trophy with a DFB Pokal.
</item 6>

<item 7>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
With moments left in extra time against the South Americans, Arnold dragged Mat Ryan and replaced him with Andrew Redmayne – who saved the decisive spot kick
(and also threw his opponents water bottle with cheat sheet attached away from the goal).

Advertisement

Ryan might well have gotten the Socceroos to Qatar, but if Australia had not advanced there’s little chance Arnold would still be their coach today.

And there’s virtually no chance he’ll have to endure a similar level of stress through to the next finals – with FIFA expanding the tournament from 36 to 48
teams meaning eight direct Asian confederation qualifiers will proceed directly instead of four.

Arnold, in his second stint as national team manager, will prowl the technical area for a 59th time on Thursday, taking him clear of Frank Farina.
</item 7>

<item 8>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
Arnold – who also represented the Socceroos 56 times as a player – has been involved in national team coaching ranks since 2000, first as an assistant to Farina
and then Guus Hiddink, before a brief period as caretaker coach in 2006.

“It’s certainly a proud moment and an achievement that I’ll reflect on when the time is right,” Arnold said.

“I look at some of the names on that list – Farina, Arok, Thompson, Rasic – guys I played under or assisted in the early part of my coaching career, and it
gives you some perspective on what a journey it’s been so far.

Advertisement

“I think back to that caretaker period and how much I learnt about coaching; it was a challenging experience and one that on reflection I probably wasn’t ready
for. I’m grateful for the opportunity I was given and what it was able to teach me ahead of my pathway into club coaching.
</item 8>

<item 9>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Advertisement

After a year in where the Wallabies’ detail and nous was clearly absent on the field, the highly detailed Schmidt would help bring rugby smarts to the
Australian program.

Indeed, the manipulation and ability to play both sides of the ruck was clearly visible by Ireland during his Irish tenure.

What’s more, should RA go down a path of bringing Schmidt into the fold, his own relationship with Ireland’s long-time director of high-performance David
Nucifora could see the Australian reconsider his position with the governing body.

Last year Nucifora told The Roar he would unlikely consider returning to RA when he leaves his post with Ireland following next year’s Olympics.

“I don’t think it’d be wise to revisit,” Nucifora said.

“I think that you have a go at something once, you do your best and then if it’s not good enough or it didn’t work you move on, do something else.
</item 9>

<item 10>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
“I was able to put lessons from that time into practice, develop and eventually return to the Socceroos much more prepared for the role. It’s been a privilege
to be part of the national team again with the benefit of that experience, and I’m immensely proud of what this group has been able to achieve in the last four
years.

“In my opinion, there’s no greater honour than representing your nation on the international stage. I look at the young players coming through the Socceroos
system now full of pride in the work they’ve put in to get where they are, and the journey that’s ahead of them.

“I’ve always loved being a part of the Socceroos culture and my desire to see Australia fulfil its footballing potential is still driving me to take this group
to further success.”

Australia starts the road to north America in Melbourne at AAMI Park on Thursday night, before travelling to the Middle East to play Palestine next Wednesday
(1.00am AEDT).
</item 10>
=== END ITEM id=4460e797628b512d ===

=== ITEM id=d28ed82bc0c77ab5 ===
Gold sentences and facts:
Sentence 0: The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.
  Fact 0.0: The present Board of Trustees of the Fund consists of the Secretary of Education as Chairman.
  Fact 0.1: The present Board of Trustees of the Fund includes the Director-General of the National Economic and Development Authority.
  Fact 0.2: The present Board of Trustees of the Fund includes the President of the Association of Christian Schools, Colleges and Universities (ACSCU).
  Fact 0.3: The present Board of Trustees of the Fund includes the President of the Catholic Educational Association of the Philippines (CEAP).
  Fact 0.4: The present Board of Trustees of the Fund includes the President of the Philippine Association of Colleges and Universities (PACU).
  Fact 0.5: The present Board of Trustees of the Fund shall also constitute the proposed Board of Trustees of the PhilEd.
Sentence 1: In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five directors.
  Fact 1.0: Andrew Wilkinson is 72 years old.
  Fact 1.1: Andrew Wilkinson has informed the HSF board of his intention to retire as a board director.
  Fact 1.2: Andrew Wilkinson's retirement will reduce the HSF board to five directors.

Retrieved items:
<item 1>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In response, Hipgnosis Song Management issued its own statement today: “Hipgnosis Song Management has fulfilled its duties to [HSF] with respect to both the
independent valuation and preparation of the interim results in a timely and efficient manner.

“Notwithstanding the Board’s decision to delay publication of the interim financial statements, [we] will continue to work in a constructive manner to support
the interests of the Company and its shareholders.”

The oddest thing about all of this? Citrin Cooperman’s valuation of Hipgnosis Songs Fund is, according to HSF’s own company documentation, appointed and
overseen… by the HSF board.

See this line, in every annual and interim report from HSF so far (bolding MBW’s own): “Portfolio Independent Valuer… Citrin Cooperman Advisors LLC, formerly
Massarsky Consulting, Inc., [is] appointed by the Board to independently value the Company’s Catalogues within the Portfolio.”
</item 1>

<item 2>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
This has been evidenced by the 44% total return, including right to income, transaction fees and expected taxes, realised since acquisition on the [$440
million] Transaction.

“Furthermore, the Board and the Investment Adviser are committed to ensuring that this value is achieved for shareholders by a re-rating of the share price and
are determined to deliver on the ongoing opportunity of the Company.”Music Business Worldwide
</item 2>

<item 3>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
HSF also confirmed a number of other new measures in a ‘circular’ to its shareholders today.

These measures included the fact that, if Hipgnosis Songs Fund shareholders vote to remain a publicly-traded fund (via that 2023 ‘continuation vote’) at the
firm’s next AGM, then another ‘continuation vote’ will be put in front of HSF shareholders at an Extraordinary General Meeting in January 2026.

(The ‘continuation vote’ that will take place in Q4 2023 has been scheduled ever since HSF went public in 2018 – i.e. five years after it floated. The 2026
‘Continuation Vote’, as newly-agreed, means another one is now being scheduled for around two years’ time.)

In addition, HSF today confirmed some new amendments to its ‘Investment Advisory Agreement’ (IAA) with HSM.
</item 3>

<item 4>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
The $440 million price also reflects a +51% premium on the value of the 29 catalogs implied by Hipgnosis Songs Fund’s public market cap for the 30 days ending
September 13.

What happened today?

Amongst the news confirmed by HSF today (September 28) is that the firm’s Chair of its board since 2018, Andrew Sutch, 73, is stepping down, both as HSF board
Chair and as a director of the company.

HSF says it’s begun a process to recruit a new Chair; Sutch (pictured inset) will leave his post as and when this person is hired, or in any case before HSF’s
AGM next year (in Q4 2024).

In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five
directors.
</item 4>

<item 5>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.
</item 5>

<item 6>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
WHAT’S HAPPENED?

MBW Explains is a series of analytical features in which we explore the context behind major music industry talking points – and suggest what might happen next.
MBW Explains is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

The drama at Hipgnosis Songs Fund isn’t letting up for the Holidays, it seems.

This morning (December 19), the company’s board – now led by Chairman, Rob Naylor – issued a statement announcing that HSF would no longer, as previously
expected, be issuing its latest financial results today.

Instead, the results (covering the six months to end of September 2023) are now expected to be published sometime before the close of December 31.

The reason for the delay?
</item 6>

<item 7>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Hipgnosis Songs Fund (HSF), the UK-listed music rights fund, looks likely to hold its all-important 2023 ‘continuation vote’ on Thursday, October 26, according
to an announcement from the firm today.

Why’s that ‘likely’ rather than ‘definitely’? Because there’s still a possibility it might get bumped into November – the reasons for which we’ll get into later
in this article.

The ‘continuation vote’ will take place at HSF’s Annual General Meeting (AGM), and will see the firm’s shareholders vote on whether to keep the entity going as
a closed-ended investment entity i.e. a publicly-traded bundle of music rights.

How will that vote go? Obviously we don’t know yet – and there’s the small matter of a $440 million transaction looming that could potentially influence things.
</item 7>

<item 8>
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
</item 8>

<item 9>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and
Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of
the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.

The management team of PEAC will be the incoming management team of PhilEd.

Interestingly, PhilHealth, in addition to premiums collected from members as well as subsidies from local governments who have enrolled indigents in the
program, receives a share of the “sin taxes” collected from Tobacco and Sugar Sweetened Beverage. The justification for this is that tobacco smokers and sugar
sweetened beverage drinkers impose an additional burden on the healthcare system of the Filipinos and so must contribute to their health services. In Fiscal
Year 2022, this amounted to P74 billion.
</item 9>

<item 10>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
If a ‘superior proposal’ (versus the $440 million offer from HSC) is received during the ‘go-shop’ period, said HSF today – whether or not HSC uses its
‘matching right’ to beat it – it may be necessary to adjourn the HSF AGM (and therefore the 2023 ‘continuation vote’) into November.

In its note to shareholders today, the HSF board reiterated its belief – and Hipgnosis Song Management’s belief – that HSF owns “a unique portfolio of iconic,
culturally significant songs that will deliver strong long-term value as they benefit from the structural tailwinds in the music industry”.

The note added: “Furthermore, the [HSF] Board believes that the Investment Adviser’s approach to Song Management should enable the Company to outperform the
wider music market.
</item 10>
=== END ITEM id=d28ed82bc0c77ab5 ===

=== ITEM id=dfffec4516b9c66d ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: They get other benefits from Android.
  Fact 0.1: Google could solve this problem today if Google wanted to.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will also suggest that Google hid items from discovery.
  Fact 1.1: Epic will suggest Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 1.2: Google will counter this as well.
  Fact 1.3: Google will say it has provided the court with thousands of chat logs.
  Fact 1.4: Google will say it has provided the court with millions of documents.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 1>

<item 2>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
That sounds fascinating, but if I don’t care about Fortnite or rich companies suing each other, what’s in it for me?

The future of Google’s app store could depend on this trial — both Epic and Google agree on that. Epic wants to break up Google’s alleged monopoly on Android
app stores and payment methods, so developers aren’t stuck paying the “Google Tax” or passing that fee along to you.

But if Epic wins — according to Google, anyhow — it could make Android phones less safe by dismantling basic protections against sideloaded apps, and damage
Android’s ability to compete with the iPhone because it (arguably) can’t run a competitive app store by giving it away for free.

Or maybe you just want juicy dirt on both companies? More on that later!

What exactly is Epic claiming?

Epic argues that Google makes it so difficult for developers and users to get around its Android app store, and that app store’s standard fees, that it’s
created an illegal monopoly that unjustly enriches Google while artificially driving up app prices because other stores can’t compete.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 3>

<item 4>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 7>

<item 8>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 8>

<item 9>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Google spent its 45 minutes attempting to explain that its dominance over the Android app market isn’t anything nefarious but simply the natural outcome of
Google fiercely competing with the iPhone and its iOS App Store, where Google would like the court to believe that competition truly lies.

If Google can convince the jury of that, it could be a winning argument in the case — because obviously, Google doesn’t have a monopoly on app stores or phones
in general. “You cannot separate the quality of a phone from the quality of the apps in its app store, and that means Google and Apple compete against each
other,” began Google lead attorney Glenn Pomerantz.

But Google wound up spending much of its opening statement attempting to explain away its seemingly bad behavior as normal business practices and didn’t always
succeed out of the gate.
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 10>
=== END ITEM id=dfffec4516b9c66d ===

=== ITEM id=34c05cddc3e8d721 ===
Gold sentences and facts:
Sentence 0: One of the major sources of Gen Z’s distress, of course, is climate change.
  Fact 0.0: Climate change is one of the major sources of Gen Z's distress.
Sentence 1: In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers.
  Fact 1.0: We need to improve our understanding of drivers of sea surface temperature patterns, and this is the first step.
  Fact 1.1: One driver is decadal coupled variability.
  Fact 1.2: We need to improve our understanding of the pace of ocean heat uptake.
  Fact 1.3: We need to improve our understanding of the spatial structure of ocean heat uptake.
  Fact 1.4: We need to improve our understanding of forcing by aerosols.
  Fact 1.5: We need to improve our understanding of forcing by greenhouse gases.
  Fact 1.6: We need to improve our understanding of forcing by volcanoes.
  Fact 1.7: We need to improve our understanding of the relative timing of these drivers.

Retrieved items:
<item 1>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
That’s why Gen Z is more concerned with sustainability than any generation before them. Just look at Greta Thunberg: The 20-year-old has become one of the
best-known environmental activists, famously speaking at the United Nations in 2019 with scathing words for world leaders:

“You have stolen my dreams and my childhood with your empty words. And yet I’m one of the lucky ones,” Thunberg said. “People are suffering. People are dying.
Entire ecosystems are collapsing. We are in the beginning of a mass extinction, and all you can talk about is money and fairy tales of eternal economic growth.
How dare you!”

Gen Z in the workplace

By 2030, Gen Z will account for nearly one-third of the U.S. workforce, but they’re already radically redefining the meaning of work.
</item 1>

<item 2>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
The inflation bill is responsible for about a quarter of that projected decrease.)

“He brags about how he thinks we should consider him a climate champion because he reentered the Paris accord. That Paris Agreement will take us to about 3
degrees Celsius of global heating,” Kalmus said. “I don’t think we’ll have a civilization at 3 degrees Celsius.”

(A 2022 United Nations report estimated global temperatures would rise between 2.1 and 2.9 degrees Celsius by 2100 if countries held to their climate
commitments. Many countries remain off that pace.)

Climate change is weighing on scientists, but also everyday Americans.

A 2022 poll found almost two-thirds of Americans say they have been affected by extreme weather they believe was at least partially due to climate change.

About 27% of Americans say they are “very worried” about climate change; another 27% just avoid the subject as best they can. One in 10 reported feeling
symptoms of anxiety or depression over climate change.
</item 2>

<item 3>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
6, 2021, following the election defeat of former President Donald Trump. Then there are school shootings, which have only increased in frequency since Columbine
in 1999, with more students documenting the terror on their phones and sharing it online. One of them was the Parkland shooting of 2018, which tragically
created the first spokespeople of the post-millennial generation.

Gen Z’s mental health

Gen Z has the worst reported mental health of any generation—45% of young people report having “excellent” or “very good” mental health, according to a 2018
report by the American Psychological Association.

One of the major sources of Gen Z’s distress, of course, is climate change. Nearly seven in 10 Gen Zers say they experience anxiety when viewing climate change
content on social media, according to a 2021 Pew Research report.
</item 3>

<item 4>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
But they’re not just reading about the detrimental effects of human-caused global warming, they’re living through the consequences themselves.

This summer reached record-breaking temperatures, with July being the hottest month the planet has seen in over 100,000 years. As a result, Arizona experienced
a monthlong heat wave with temperatures at or above 110 degrees every day. Deadly fires broke out across the Mediterranean; suffocating smoke from Canadian
wildfires blanketed New York City and the Northeast for days; and ice melt in the Arctic accelerated.

And extreme heat is likely here to stay—and get worse—unless countries can rapidly reduce their carbon emissions. That’s why Gen Z is more concerned with
sustainability than any generation before them.
</item 4>

<item 5>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
Roughly 60% of Gen Zers say they are stressed about money this year more than last year, according to a Bankrate survey from July. It’s no surprise either—85%
of Gen Zers say that they couldn’t afford one month’s expenses if they lost their job today.

And since young people are typically affected by inflation the most, as they are the most likely to work part-time or low-paying jobs, this economic climate may
have left Gen Z with permanent “psychological scars,” one expert says.

“How can young people build careers or wealth if they don’t have jobs, and prices of goods and services continue to increase?” Dayo Abinusawa, founder of
London’s Awa Business School and a former lecturer at Cambridge University’s Judge Business School, previously told Fortune.

One Fidelity survey backs this argument: 45% of 18- to 35-year-olds “don’t see a point in saving until things return to normal.”
</item 5>

<item 6>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
6 insurrection—an event unseen for centuries in American politics. Then there’s the first major European ground war since World War II in Ukraine, not to
mention market crashes in 2008 and 2020 that recall the Great Depression itself. As the first digitally native generation, Gen Z is experiencing it all through
videos, images, and articles online, which is shaping their mental health, workplace attitudes, and financial habits in visible ways.

It’s no wonder, this Gen Z reporter notes, that 46% of young workers ages 18 to 26 say that they are regularly so distraught over what is happening in the news
that they are unable to function at work, according to a 2023 Edelman report. By comparison, 38% of millennials, 24% of Gen Xers, and 19% of baby boomers and
older generations say the same.

Everything about their behavior communicates that Gen Z is just not okay with it.
</item 6>

<item 7>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
BELFAST, Maine — How does it feel to be a climate scientist in 2023?

Let’s set the scene. Hundreds of picnickers are sprawled across a lawn speckled with colorful blankets. Yachts bob in the bay. Guitars blare, children wrestle
and dancers twirl across a concert lawn in the golden glow of a dying summer night.

It’s a picture-perfect August evening for the seaside town of Belfast’s “Summer Nights” concert series.

Off to the side, Peter Kalmus sits alone. His eyes are closed. His toes are tucked beneath his body. He’s catching an occasional side-eye from onlookers.

“When I’m meditating, I don’t feel anxious,” Kalmus later says, explaining that he strives for two hours of meditation a day to keep his climate anxiety at bay.
Otherwise, “it’s completely overwhelming.”

Many of us avoid thinking too deeply about the worst scenarios of climate change. He is fixated on them.
</item 7>

<item 8>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
Everything about their behavior communicates that Gen Z is just not okay with it. This ranges from their widespread, hell-bent determination to find purpose in
work and pushing their employers to have a social conscience, to a sense of despair over their own and the world’s future finances. They have largely given up
on saving money and instead dish out for little “treats” as a way to cope with the larger absurdity of 21st-century life.

Consider the lifetime that was three years ago, as Gen Z emerged into young adulthood, when online videos of the murder of George Floyd shook the country in May
2020, resulting in a summer of violent Black Lives Matter protests and riots. A year later, people watched as armed right-wing extremists stormed the Capitol on
Jan. 6, 2021, following the election defeat of former President Donald Trump.
</item 8>

<item 9>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He has been on a 15-year quest to halt climate change and has already oriented most everything in his life to minimize his own climate footprint and maximize
what he can do to push society to change.

Nothing has been enough.

And so this summer, as extreme heat and rainfall battered many American communities, Kalmus visited Maine searching for something different. He was looking not
for climate solutions, but for new ways to withstand climate change’s harms — and also to make peace with the grief that often feels inescapable for those who
reckon with global warming daily.

climate scientist exercise bicycle environmentalist (Evan Bush / NBC News)

That search brought him to an unusual community called the Possibility Alliance, a homestead where a small group of people is trying to figure out how to live
in a world headed for a breakdown (as they see it).

Its leaders eschew capitalism, grow much of their own food, and live without electricity. They’ve dedicated their lives to protest, living off the land as
simply as possible, and are preparing their community for ecological disaster and to host climate refugees.
</item 9>

<item 10>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
workforce, but they’re already radically redefining the meaning of work.

Gen Zers want a sense of purpose, so they prioritize environmental, social, and governance (ESG) in the workplace, which encompasses their companies taking
accountability for their sustainability and environmental impacts, providing education and awareness for social issues, and ensuring diverse and inclusive
boards and teams.

And it tracks: Roughly two-thirds of Gen Zers say they frequently speak about important societal issues while at work, according to the Edelman report. They’re
also influencing their older coworkers when it comes to areas like work-life balance, fair pay, and employers’ involvement on social issues.

Gen Z’s financial stress

But don’t forget, Gen Zers have also lived through a global pandemic that shuttered the world for nearly two years, two recessions, and a mounting student debt
crisis, leaving them with little savings but an abundance of financial despair.
</item 10>
=== END ITEM id=34c05cddc3e8d721 ===

=== ITEM id=cfe5f4c3deaea7ca ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Bankman-Fried used Caroline Ellison as a front.
  Fact 0.1: Caroline Ellison is Bankman-Fried's on-again off-again girlfriend.
  Fact 0.2: Caroline Ellison is the CEO of Alameda.
  Fact 0.3: The government claimed Bankman-Fried had secret access to customer cash.
  Fact 0.4: The government claimed Bankman-Fried had secret access to customer crypto.
Sentence 1: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 1.0: She said it was Bankman-Fried's decision.
  Fact 1.1: Bankman-Fried was the CEO of FTX.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.
</item 1>

<item 2>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 2>

<item 3>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 6>

<item 7>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 8>

<item 9>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 9>

<item 10>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 10>
=== END ITEM id=cfe5f4c3deaea7ca ===
