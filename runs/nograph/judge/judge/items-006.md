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

=== ITEM id=c08686b020247a59 ===
Gold sentences and facts:
Sentence 0: His career has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early age, Mitchell is clear-eyed.
  Fact 0.0: His career has only blossomed in the years since he took that trip.
  Fact 0.1: Some artists might feel trepidation about attaining so much success from such an early age.
  Fact 0.2: Mitchell is clear-eyed.
Sentence 1: Since then he’s been linked to club jobs in Europe but opted to stick around with the national team.
  Fact 1.0: Since then, he has been linked to club jobs in Europe.
  Fact 1.1: He opted to stick around with the national team.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell Albany, Georgia, 2021 (detail). Credit: Courtesy of the artist and Jack Shainman Gallery, New York As a 20-year-old film student at New York
University, Mitchell became enamoured with still photography after he signed up to do a documentary photography course in Havana, Cuba. Frustrated by the amount
of time and money required to make films, he was drawn to the accessibility of still photography.

Advertisement

“I was impatient when I got started and just wanted to make visual things now, and I didn’t want to wait for permission, or didn’t want to wait on equipment, or
didn’t want to lug large amounts of equipment up flights of stairs. I just wanted to make something that I felt compelled by, immediately,” he says. His career
has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early
age, Mitchell is clear-eyed. “The idea of the road ahead only excites me,” he says.
</item 2>

<item 3>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell: “I’ve never been to Australia so it’s very exciting to sort of have a global or new conversation on a new continent with my work.″⁣ Credit:
Courtesy of the artist and Jack Shainman Gallery, New York The theme of the Triennial is “Magic, Matter, Memory”, making Mitchell’s pieces a logical inclusion.
“A lot of my work meditates on being between a heightened reality, or potentially even a magical reality and an actual reality,” he says.

Advertisement

“I’m creating what I call lightly staged moments. And so that means I’m setting up scenarios where these families are enjoying real moments of leisure in
outdoor spaces, usually open pastures and fields, and those moments are meant to stand in for a meditation on what it means to enjoy leisurely outdoor nature as
a black person.” Tyler Mitchell, Vastness, 2021.
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
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
Arnold – who also represented the Socceroos 56 times as a player – has been involved in national team coaching ranks since 2000, first as an assistant to Farina
and then Guus Hiddink, before a brief period as caretaker coach in 2006.

“It’s certainly a proud moment and an achievement that I’ll reflect on when the time is right,” Arnold said.

“I look at some of the names on that list – Farina, Arok, Thompson, Rasic – guys I played under or assisted in the early part of my coaching career, and it
gives you some perspective on what a journey it’s been so far.

Advertisement

“I think back to that caretaker period and how much I learnt about coaching; it was a challenging experience and one that on reflection I probably wasn’t ready
for. I’m grateful for the opportunity I was given and what it was able to teach me ahead of my pathway into club coaching.
</item 5>

<item 6>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
“The idea of the road ahead only excites me,” he says. “The idea that there will be very quiet, sleepy periods, and frenetic periods, or hyperactive periods …
It’s always refreshing as a young person to see how the career of a much older, more senior artist has really gone. “That senior artist that we know and love
today had decades where they were out of style, out of fashion, people thought their work was done, their best work was behind them, and yet, often had
resurgences. Loading “I think when we’re thinking about crafting a career … we’re really talking about allowing space for bad and good, allowing space for
artists to make and work through complicated and sometimes not-so-favourable ideas. And I try to give myself that same space. It’s only about giving yourself
grace and space and thinking one day at a time.” Asked if there’s a dream project, or a dream collaborator for him, Mitchell says there are too many.
</item 6>

<item 7>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell, Vastness, 2021. Credit: Courtesy of the artist and Jack Shainman Gallery, New York In the listless anxious days of the pandemic, as the Black
Lives Matter movement rose in the summer of 2020, and the conversation of what black life looked like was being routinely debated in the press, black artists
and activists looked to reframe that conversation, and force new perspectives of peace, luxury, and safety into the picture. Instead of seeking to record things
as they are, Mitchell’s work asks us to imagine new realities. He says he wants to challenge “visual regimes”: assumptions about certain people or cultures that
are reinforced over and over again. “I think that [photojournalism] is a well from which we all culturally – consciously or unconsciously – construct ideas of
how people live around the world,” he says. “My work is really about responding to, and questioning, and challenging those ideas.” Tyler Mitchell Albany,
Georgia, 2021 (detail).
</item 7>

<item 8>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
With moments left in extra time against the South Americans, Arnold dragged Mat Ryan and replaced him with Andrew Redmayne – who saved the decisive spot kick
(and also threw his opponents water bottle with cheat sheet attached away from the goal).

Advertisement

Ryan might well have gotten the Socceroos to Qatar, but if Australia had not advanced there’s little chance Arnold would still be their coach today.

And there’s virtually no chance he’ll have to endure a similar level of stress through to the next finals – with FIFA expanding the tournament from 36 to 48
teams meaning eight direct Asian confederation qualifiers will proceed directly instead of four.

Arnold, in his second stint as national team manager, will prowl the technical area for a 59th time on Thursday, taking him clear of Frank Farina.
</item 8>

<item 9>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Loading In one of his favourite works, A Glint of Possibility, a teenage boy hangs over a river suspended on a tyre swing, studying his own reflection. The
moment is pensive and peaceful at once, and relatable to anyone who’s ever been a child on a tyre swing on a hot summer day. Something about it speaks to the
turmoil and tension inherent in the experience of black American teenage boys, about the ways their lives can be complicated by the world we all inhabit. How
even when doing unremarkable things – walking home, playing in the park, talking to their crush – their lives can be destroyed or irrevocably altered. Mitchell
understands the significance of these contrasts. His work exists in the tension between what is possible and how things are. He describes it as: “A
consideration of joy, perseverance and self-determination against the backdrop of history.”
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
=== END ITEM id=c08686b020247a59 ===

=== ITEM id=e92e33653e98e23c ===
Gold sentences and facts:
Sentence 0: The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.
  Fact 0.0: The present Board of Trustees of the Fund consists of the Secretary of Education as Chairman.
  Fact 0.1: The present Board of Trustees of the Fund includes the Director-General of the National Economic and Development Authority.
  Fact 0.2: The present Board of Trustees of the Fund includes three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP), and the Philippine Association of Colleges and Universities (PACU).
  Fact 0.3: The present Board of Trustees of the Fund shall also constitute the proposed Board of Trustees of the PhilEd.
Sentence 1: In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five directors.
  Fact 1.0: Andrew Wilkinson is 72 years old.
  Fact 1.1: Andrew Wilkinson has informed the HSF board of his intention to retire as a board director.
  Fact 1.2: Andrew Wilkinson's retirement will reduce the company's board to five directors.

Retrieved items:
<item 1>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
its share price).

For example: On September 14, HSF’s market cap stood at approximately GBP £1.125 billion, which represented a 40% discount on the firm’s operative NAV of USD
$2.316 billion as of the end of March this year.

(Said ‘operative NAV’ was calculated by an independent valuer – Citrin Cooperman – using a discount rate of 8.5%.)

This is where the importance of that potential $440 million transaction comes in.

As MBW reported earlier this month, Hipgnosis Songs Fund’s shareholders are currently considering a proposal to sell 29 of the company’s catalogs to another
Hipgnosis company – the Blackstone-backed Hipgnosis Songs Capital (HSC) – for USD $440 million.
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
</item 3>

<item 4>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
the multiples that Merck Mercuriadis paid to acquire it?

Such questions will only be answered when Hipgnosis Songs Fund’s board elects to release the company’s six-monthly financial results to end of September 30.

The most important KPI within those results, in MBW’s view: the company’s Pro-Forma Revenue, and how it compares, like-for-like, vs. the previous year’s
equivalent period.

In the meantime, back in the ‘real world’ of the music biz, HSF is actually enjoying quite a fanciable Q4.

The company is benefitting from its cuts on hit albums including Taylor Swift’s 1989 (Taylor’s Version), via its acquired Jack Antonoff catalog, as well as
Swift’s Lover (via its acquired Joel Little catalog).
</item 4>

<item 5>
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and
Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of
the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.

The management team of PEAC will be the incoming management team of PhilEd.

Interestingly, PhilHealth, in addition to premiums collected from members as well as subsidies from local governments who have enrolled indigents in the
program, receives a share of the “sin taxes” collected from Tobacco and Sugar Sweetened Beverage. The justification for this is that tobacco smokers and sugar
sweetened beverage drinkers impose an additional burden on the healthcare system of the Filipinos and so must contribute to their health services. In Fiscal
Year 2022, this amounted to P74 billion.
</item 5>

<item 6>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
HSF also confirmed a number of other new measures in a ‘circular’ to its shareholders today.

These measures included the fact that, if Hipgnosis Songs Fund shareholders vote to remain a publicly-traded fund (via that 2023 ‘continuation vote’) at the
firm’s next AGM, then another ‘continuation vote’ will be put in front of HSF shareholders at an Extraordinary General Meeting in January 2026.

(The ‘continuation vote’ that will take place in Q4 2023 has been scheduled ever since HSF went public in 2018 – i.e. five years after it floated. The 2026
‘Continuation Vote’, as newly-agreed, means another one is now being scheduled for around two years’ time.)

In addition, HSF today confirmed some new amendments to its ‘Investment Advisory Agreement’ (IAA) with HSM.
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
WHAT’S HAPPENED?

MBW Explains is a series of analytical features in which we explore the context behind major music industry talking points – and suggest what might happen next.
MBW Explains is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

The drama at Hipgnosis Songs Fund isn’t letting up for the Holidays, it seems.

This morning (December 19), the company’s board – now led by Chairman, Rob Naylor – issued a statement announcing that HSF would no longer, as previously
expected, be issuing its latest financial results today.

Instead, the results (covering the six months to end of September 2023) are now expected to be published sometime before the close of December 31.

The reason for the delay?
</item 8>

<item 9>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
If a ‘superior proposal’ (versus the $440 million offer from HSC) is received during the ‘go-shop’ period, said HSF today – whether or not HSC uses its
‘matching right’ to beat it – it may be necessary to adjourn the HSF AGM (and therefore the 2023 ‘continuation vote’) into November.

In its note to shareholders today, the HSF board reiterated its belief – and Hipgnosis Song Management’s belief – that HSF owns “a unique portfolio of iconic,
culturally significant songs that will deliver strong long-term value as they benefit from the structural tailwinds in the music industry”.

The note added: “Furthermore, the [HSF] Board believes that the Investment Adviser’s approach to Song Management should enable the Company to outperform the
wider music market.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=e92e33653e98e23c ===

=== ITEM id=be55f5076789f8bc ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: Google gets other benefits from Android.
  Fact 0.1: Google could solve this problem today if it wanted to.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will suggest that Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 1.1: Google will counter that suggestion.
  Fact 1.2: Google will say it has provided the court with thousands of chat logs.
  Fact 1.3: Google will say it has provided the court with millions of documents.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 2>

<item 3>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
It’s always been in Google’s power to solve this problem. They make several billion dollars a year in unfairly earned profits from imposing this tax, which is
nothing compared to the money they make from search. For all the other benefits they get from Android, Google could solve this problem today if they wanted to.

Do you today believe that Activision Blizzard was really intending to build its own app store?

I don’t know about Supercell, but we know from the documents in the case that Riot was planning to distribute League of Legends directly through their website,
on mobile, exactly as they do on PC. And that’s what they planned to do until Google paid them off to not do that. Google’s payoff at the minimum dissuaded Riot
from distributing off Google Play.

I think they’re the strongest example of the three, but I’m curious about Activision.

Activision was building a store! We know they had a large team of dozens of engineers building a mobile app store to launch on Android.
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
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
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 6>

<item 7>
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
</item 7>

<item 8>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
This program, still in pilot testing, is open to all developers who sell apps in the 35 markets where it’s now available, including the U.S., and reduces the
standard commission by 4% for companies who choose to use their own payment processing solution. Spotify and Bumble were the initial testers for the new system,
first introduced in November 2022, and Match, as part of its settlement agreement, will also now take advantage of this option.

Google will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other features and controls to keep Android
users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to build and grow their apps. Google also
helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging from tablets to TVs to autos and more. And
it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.
</item 8>

<item 9>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The future of Google’s app store is at stake in a lawsuit by Fortnite publisher Epic Games. Epic sued Google in 2020 after a fight over in-app purchase fees,
claiming the Android operating system’s Google Play Store constituted an unlawful monopoly — while Google says its demands would damage Android’s ability to
offer a secure user experience and compete with Apple’s iOS. Follow along with updates here.

Bornstein showed jurors charts of Google’s fat app profit margins (70 percent on $12 billion in revenue a year, says Epic) and pointed out several ugly-seeming
ways Google has allegedly attempted to keep anyone from taking that money away — like paying game developers not to build their own app stores or standalone app
launchers like Epic did with Fortnite.

“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying,
‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
As an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one developer that wanted to go the
route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020, worth approximately $360 million,
to keep the developer on Google Play, Epic will argue.

Google, however, will characterize this program as a way to incentivize developers to launch their apps on Android at the same time as they launch on iOS and
plans to dispute Epic’s claims that it prevented developers from opening their own app stores.

Android allows sideloading

Google will make note that, unlike Apple, it also allows apps to be sideloaded on Android devices — something Epic takes advantage of today. It will note this
process is streamlined and helps keep users safe.
</item 10>
=== END ITEM id=be55f5076789f8bc ===

=== ITEM id=a147c500fb76a4ee ===
Gold sentences and facts:
Sentence 0: One of the major sources of Gen Z’s distress, of course, is climate change.
  Fact 0.0: Climate change is one of the major sources of Gen Z's distress.
Sentence 1: In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers.
  Fact 1.0: We need to improve our understanding of the drivers of sea surface temperature patterns.
  Fact 1.1: Improving the understanding of drivers of sea surface temperature patterns is the first step.
  Fact 1.2: One driver of sea surface temperature patterns is decadal coupled variability.
  Fact 1.3: Another driver is the pace of ocean heat uptake.
  Fact 1.4: Another driver is the spatial structure of ocean heat uptake.
  Fact 1.5: Another driver is forcing by aerosols.
  Fact 1.6: Another driver is forcing by greenhouse gases.
  Fact 1.7: Another driver is forcing by volcanoes.
  Fact 1.8: We need to improve our understanding of the relative timing of these drivers.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
In addition to our growing knowledge of the pattern effect, we also have learned that radiative feedbacks depend on global mean temperature itself: Warming
Earth by 1 K from the LGM emphasizes different feedbacks (e.g., the sea ice albedo feedback) than warming by 1 K from a Miocene hothouse world or warming from 4
to 5 K in a high-emission scenario in a century or two from today (e.g., the water vapor feedback [Bloch-Johnson et al., 2021]). The pattern effect and the
feedback temperature dependence add uncertainties to estimates of climate sensitivity based on the paleorecord, but quantifying their effects would make these
records more relevant to constraining climate sensitivity and expected future warming.

Outstanding questions concern how representative pattern changes in the past century, the past millennium, or quasi-equilibrated times millions of years ago are
of expected future changes.
</item 2>

<item 3>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
However, they also depend on the spatial pattern of surface warming: Much like applying a force uniformly over someone’s entire body will elicit a very
different reaction than tickling the soles of that person’s feet, a degree of global warming spread out evenly will cause a different radiative response than if
that same warming were concentrated in a climate sweet spot (a location where surface warming produces efficient radiative damping).

A wide variety of processes affect the evolution of surface temperature, from greenhouse gas forcing and regional aerosol forcing to natural oscillations
involving the ocean and atmosphere to the continental boundary conditions and the extent of ice sheets and sea ice. The pattern of surface temperature change
over the past 40 or so years featured a pronounced spatial structure, with some locations even cooling in spite of the global mean warming on the order of 1 K
(Figure 2, bottom left).

Fig. 2. At left is the surface and vertical structure of warming observed over the past few decades in the tropical Pacific.
</item 3>

<item 4>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
6 insurrection—an event unseen for centuries in American politics. Then there’s the first major European ground war since World War II in Ukraine, not to
mention market crashes in 2008 and 2020 that recall the Great Depression itself. As the first digitally native generation, Gen Z is experiencing it all through
videos, images, and articles online, which is shaping their mental health, workplace attitudes, and financial habits in visible ways.

It’s no wonder, this Gen Z reporter notes, that 46% of young workers ages 18 to 26 say that they are regularly so distraught over what is happening in the news
that they are unable to function at work, according to a 2023 Edelman report. By comparison, 38% of millennials, 24% of Gen Xers, and 19% of baby boomers and
older generations say the same.

Everything about their behavior communicates that Gen Z is just not okay with it.
</item 4>

<item 5>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Feedbacks involving clouds and the atmospheric temperature structure are most sensitive to spatial differences in warming. Deep convection in the warmest
tropical regions readily communicates surface conditions upward throughout the troposphere (up to about 10–15 kilometers) and then horizontally across much of
the globe, making the western Pacific a climate sweet spot. This warmer air sitting atop the relatively cool waters in the eastern Pacific or Southern Ocean
acts to stabilize the lowermost troposphere, allowing more extensive low-lying stratus and stratocumulus clouds to develop. Because of their location and
structure, these low clouds efficiently cool the planet and offset some of the initial warming (Figure 2, top left).

Approaching the Problem from Different Angles

Three strands of research have converged over the past few years, highlighting that understanding the pattern effect benefits from contributions from virtually
all climate research communities.

Historically, three strands of research have highlighted the dependence of radiative feedbacks on the surface warming pattern.
</item 5>

<item 6>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Feedbacks calculated from observations or from atmosphere-only model simulations forced with the observed surface warming pattern over the past couple of
decades imply less warming than those from model simulations with a fully interactive ocean, which have the freedom to create their own surface warming patterns
[e.g., Gregory et al., 2020].

The third strand of research came from oceanography, showing that the atmospheric cooling effect of ocean heat uptake differs depending on where it occurs: One
unit of ocean heat uptake in high latitudes cools Earth more effectively than the same unit taken up by the low-latitude oceans. This difference is relevant
because the largest heat uptake by the ocean occurs at higher latitudes. The effect, termed ocean heat uptake efficacy, turns out to be another manifestation of
the dependence of radiative feedbacks on surface temperature patterns [Winton et al., 2010; Lin et al., 2021].
</item 6>

<item 7>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial
structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers. An important goal is
to explain historical patterns quantitatively and to predict their likely future evolution. Second, we need to quantify the dependence of local and remote
top-of-the-atmosphere radiative fluxes on the magnitude, spatial scale, and sign (positive or negative) of surface temperature changes. Third, we have to
explain how ocean heat uptake and radiative feedback are connected on various timescales, both globally and regionally.

Interpreting Paleorecords Requires Caution

In the same way that the historical evolution of surface warming and the planet’s radiation budget inform Earth’s future only to a limited extent, analyses of
paleorecords for recent and deep time intervals must account for the pattern effect to be applicable to climate projections of the future.
</item 7>

<item 8>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
But they’re not just reading about the detrimental effects of human-caused global warming, they’re living through the consequences themselves.

This summer reached record-breaking temperatures, with July being the hottest month the planet has seen in over 100,000 years. As a result, Arizona experienced
a monthlong heat wave with temperatures at or above 110 degrees every day. Deadly fires broke out across the Mediterranean; suffocating smoke from Canadian
wildfires blanketed New York City and the Northeast for days; and ice melt in the Arctic accelerated.

And extreme heat is likely here to stay—and get worse—unless countries can rapidly reduce their carbon emissions. That’s why Gen Z is more concerned with
sustainability than any generation before them.
</item 8>

<item 9>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Acknowledgments

The authors thank all participants of the US CLIVAR pattern effect workshop. M.Z.’s work was supported by the U.S. Department of Energy (DOE) Regional and
Global Model Analysis program area and was performed under the auspices of DOE by Lawrence Livermore National Laboratory under contract DE-AC52-07NA27344. P.C.
was supported by the UK Natural Environment Research Council, grants NE/T006250/1 and NE/V012045/1. T.A. was supported by the Met Office Hadley Centre Climate
Programme funded by the Department of Business, Energy and Industrial Strategy and received funding from the European Union’s Horizon 2020 research and
innovation program under grant agreement 820829. M.R. was supported by NASA under grant 80NSSC21K1042.

References

Andrews, T., et al. (2022), On the effect of historical SST patterns on radiative feedback, J. Geophys. Res.
</item 9>

<item 10>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Winton, M., K. Takahashi, and I. M. Held (2010), Importance of ocean heat uptake efficacy to transient climate change, J. Clim., 23, 2,333–2,344,
https://doi.org/10.1175/2009JCLI3139.1.

Wills, R. C. J., et al. (2022), Systematic climate model biases in the large-scale patterns of recent sea-surface temperature and sea-level pressure change,
Geophys. Res. Lett., 49, e2022GL100011, https://doi.org/10.1029/2022GL100011.

Zhou, C., et al. (2021), Greater committed warming after accounting for the pattern effect, Nat. Clim. Change, 11, 132–136,
https://doi.org/10.1038/s41558-020-00955-x.
</item 10>
=== END ITEM id=a147c500fb76a4ee ===

=== ITEM id=50b7462540e7c3a2 ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Bankman-Fried used Caroline Ellison as a front.
  Fact 0.1: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 0.2: Caroline Ellison was the CEO of Alameda.
  Fact 0.3: Bankman-Fried had 'secret access' to customer money, both cash and crypto.
  Fact 0.4: The government claimed that Bankman-Fried had this secret access to customer money.
Sentence 1: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 1.0: The decision was Bankman-Fried's, according to a female speaker.
  Fact 1.1: Bankman-Fried was the CEO of FTX.
  Fact 1.2: According to a female speaker, Bankman-Fried made the decision because he was the CEO of FTX.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.
</item 2>

<item 3>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
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
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 5>

<item 6>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 6>

<item 7>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 8>

<item 9>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 9>

<item 10>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 10>
=== END ITEM id=50b7462540e7c3a2 ===
