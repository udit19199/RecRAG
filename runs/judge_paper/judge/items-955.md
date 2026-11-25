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

=== ITEM id=4e0a3e5121a15f45 ===
Gold sentences and facts:
Sentence 0: Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in the semi-finals — and England have beaten all of them.
  Fact 0.0: Argentina has provided highlights.
  Fact 0.1: Japan has provided highlights.
  Fact 0.2: Chile has provided highlights.
  Fact 0.3: Samoa has provided highlights.
  Fact 0.4: Fiji has provided highlights.
  Fact 0.5: Los Pumas are also in the semi-finals.
  Fact 0.6: England have beaten Argentina.
  Fact 0.7: England have beaten Japan.
  Fact 0.8: England have beaten Chile.
  Fact 0.9: England have beaten Samoa.
  Fact 0.10: England have beaten Fiji.
Sentence 1: S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by Fiji and Portugal in Toulouse.
  Fact 1.0: This report is datelined Saint-Etienne.
  Fact 1.1: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com.
  Fact 1.2: The Wallabies' Rugby World Cup fate is decided by Fiji and Portugal.
  Fact 1.3: The Fiji and Portugal match is played in Toulouse.
  Fact 1.4: The Wallabies' Rugby World Cup fate is decided in Toulouse.

Retrieved items:
<item 1>
Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. (The New York Times, 2023-10-08T21:56:00+00:00)
Hard to rank it any higher until we know what we’re dealing with.

Largest crowd ever for a game at Globe Life Field. #StraightUpTX pic.twitter.com/4I7NIB5eF8 — Chris Halicke (@ChrisHalicke) May 1, 2022

The old Rangers ballpark could drum up a Texas-sized atmosphere worthy of Stevie Ray Vaughan and Pat Green, but the new place — at least in the regular season —
feels very much like a multi-events space that happens to be hosting a ballgame. Maybe that changes in the playoffs? The Rangers haven’t been home since Sept.
24 when they won their last five games in Arlington, the last three of which went a long way toward keeping the Mariners out of the postseason.

With a closed roof containing all the noise, Globe Life is sure to get awfully loud, and nothing turns a glorified convention center into a real ballpark quite
like a late-inning rally in October.

No.
</item 1>

<item 2>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 2>

<item 3>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The hardcore football fans are pretty much done with the NFL harping about Travis Kelce and Taylor Swift. Reality TV star Rachel Lindsay, a longtime NFL fan,
recently slammed the “obnoxious” coverage, highlighting the “exhausting” uptick in Swifties obsessing over football. But that’s just one example of how fans are
finding this fixation a little too disrespectful towards the sport.

America’s Favorite Video Today

From the league’s humble beginnings in 1920 to the $18 billion juggernaut that it is today, the ride has been rocky with more than just a few small bumps along
the road. But with the ambitious target that NFL Commissioner, Roger Goodell, has set for 2027, is there a new strategy in the 104-year-old league’s master plan
to dominate the world sporting scene?

Why the ‘Traylor’ saga will be the key to Roger Goodell’s $27 billion target?
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
But they also haven’t really beaten anyone. They were down 14 points to Carolina. Do you think they are coming back from 13 down against Philly? In my opinion,
no.”

Advertisement

• Dallas Cowboys: Is this team any better than previous Dallas teams that fell short?

What I think: It’s a tough time to evaluate the Cowboys fairly because they were so bad against San Francisco on a big stage in their most recent game. I just
don’t see where Dallas is better or going to get considerably better, especially on offense. And every time they enjoy success, they seem to enjoy it a little
too much. That is a reflection of their owner.

Exec comment: “Run defense and consistency on defense, and then offensively, what is their identity? They just are not very good. Their best games are when they
play great defense and get turnovers, but that is not going to happen enough, especially against the good teams.”

• Detroit Lions: Have they fixed their defense?
</item 4>

<item 5>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 5>

<item 6>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
However, the NFL does not just want to stop at $18 billion. It wants to become a $27 billion league in 4 years. But it must address a few skeletons in its
closet before reaching that goal.

If you were around in the 1970s (that would make you at least 50 years old today), you would know that the NBA did not have the mass appeal that it enjoys now.
With average game attendance hovering around 8,000 a game, it was one of the darkest periods of the basketball league. Drug abuse, brawls on the hardwood court
that sometimes spilled over onto coaches and players – it wasn’t a pretty sight. As reported in the Washington Post, even though there are no reliable figures
on cocaine use by players, “estimates by people in the game range from 40 percent to 75 percent.”
</item 6>

<item 7>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 7>

<item 8>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 8>

<item 9>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The October 1 matchup between the Chiefs and the Jets drew 27 million viewers. And became the most-watched Sunday TV show since the last Super Bowl! And the
last Super Bowl had an actual concert by Rihanna.

Revealed: The Dark Side Of Pat McAfee & Why Rob Gronkowski Is In The Firing Line

Kaepernick, who? Deflategate, what? In the media frenzy surrounding the two lovebirds, other issues get thrown out of the window. How can fans forget that the
NFL is also organizing games outside the American borders with a view to garnering international appeal? The NFL has scheduled 5 games abroad this season. It is
also addressing the concussion controversies. It recently added both new concussion rules and a new $100 million initiative called “Play Smart, Play Safe,” to
make the sport safer for players. And all the effort is showing.
</item 9>

<item 10>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
And all the effort is showing.

According to Joe Pompliano, “The NFL will bring in $20 billion in revenue this year, more than the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1
combined.” Coming to the strategy part, a curious mind would perhaps think can they take a leaf out of the NBA’s playbook and reinvigorate its appeal as it aims
to conquer international audiences?

Can the NFL take a leaf out of the NBA’s grisly past and glorious recovery?

The NFL is already the world’s richest league. Calling it an incredible feat would be a gross understatement. Consider this: the English Premier League, which
is broadcast in over 212 territories, has a revenue of just $6 billion, while football games are primarily viewed in America only. However, the NFL does not
just want to stop at $18 billion.
</item 10>
=== END ITEM id=4e0a3e5121a15f45 ===

=== ITEM id=75bf4df055414981 ===
Gold sentences and facts:
Sentence 0: Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto.
  Fact 0.0: Shares of Rogers extended losses on news of the lawsuit.
  Fact 0.1: Rogers shares closed down 2.2% to C$53.07 in Toronto.
Sentence 1: Amazon tumbled 4 per cent after the Federal Trade Commission and 17 state attorneys general filed an antitrust lawsuit against it.
  Fact 1.0: Amazon tumbled 4 per cent after the Federal Trade Commission filed an antitrust lawsuit against it.
  Fact 1.1: 17 state attorneys general filed an antitrust lawsuit against Amazon.
  Fact 1.2: The Federal Trade Commission and the 17 state attorneys general filed the antitrust lawsuit against Amazon together.

Retrieved items:

=== END ITEM id=75bf4df055414981 ===

=== ITEM id=d052573d863775ef ===
Gold sentences and facts:
Sentence 0: It would seem that, as the Springboks play for their country, the All Blacks are playing for themselves as much as anyone – and the generation of players and leaders that will call time on their international careers at week’s end.
  Fact 0.0: It would seem that the Springboks are playing for their country.
  Fact 0.1: It would seem that the All Blacks are playing for themselves as much as anyone.
  Fact 0.2: A generation of All Blacks players and leaders will call time on their international careers at week's end.
Sentence 1: The All Blacks had their chances, but their usual safe handling, a feature throughout the World Cup after their first-up loss to France, evaded them against the Springboks’ blitz defence in the wet conditions.
  Fact 1.0: The All Blacks had their chances.
  Fact 1.1: The All Blacks' usual safe handling was a feature throughout the World Cup.
  Fact 1.2: The All Blacks lost their first-up match at the World Cup to France.
  Fact 1.3: The All Blacks' safe handling evaded them against the Springboks' blitz defence.
  Fact 1.4: The conditions were wet.
Sentence 2: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.
  Fact 2.0: Argentina knocked over the All Blacks last year in Christchurch.
  Fact 2.1: This comes despite Argentina knocking over the All Blacks last year in Christchurch.
  Fact 2.2: The All Blacks suffered a maiden defeat in 2020 in Sydney.

Retrieved items:
<item 1>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 1>

<item 2>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
When he was evaluating the exchange, he thought the sheer volume of users was important, too — at the time, FTX was among the top three biggest exchanges. Plus,
major VC firms had invested, and “they don’t commit hundreds of millions without doing due diligence, checking the books, the accountancy of the firm, going
through several compliance process[es], so that was a vote of confidence for me,” Juilliard said. (Evidently he had not paid attention to the Elizabeth Holmes
trial.)

He also noted FTX’s glossy ads — featuring Gisele Bündchen, for instance — suggested a very high budget. It wouldn’t make sense to spend that much money unless
FTX had very strong financials, Juilliard figured. He opened an account, transferred in both regular money and cryptocurrency, and used the exchange to execute
his plan: buying Bitcoin to sell back in five to ten years at higher prices.
</item 2>

<item 3>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 4>

<item 5>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
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
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 7>

<item 8>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 9>

<item 10>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
It is a thankless task to cross-examine a customer whose money is gone

In November 2022, things went bad for Julliard. He followed Bankman-Fried on Twitter, and read aloud the “FTX is fine. Assets are fine” tweets, along with “FTX
has enough to cover all client holdings. We don’t invest client assets” and a few others, which gave Julliard the impression that his money was there — the
problem might have been technical (anti-spam measures) or regulatory. When he tried to get his money out on November 8th, it was too late. We saw screenshots of
his withdrawal attempts: $20,000 USD and about 4 Bitcoin, which were worth about $20,000 at the time: about $100,000 money, inaccessible.

It is a thankless task to cross-examine a customer whose money is gone, but Cohen tried anyway.
</item 10>
=== END ITEM id=d052573d863775ef ===

=== ITEM id=4ca9b8b696a20540 ===
Gold sentences and facts:
Sentence 0: Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.
  Fact 0.0: The Fortnite maker announced on September 28 that it is laying off 16% of its workforce.
  Fact 0.1: The layoff amounts to 870 people.
Sentence 1: Because while Google spent most of its first day attempting to explain complicated ins and outs of business, Epic was able to paint a black-and-white picture of good and evil with itself as the clear underdog.
  Fact 1.0: Google spent most of its first day attempting to explain complicated ins and outs of business.
  Fact 1.1: Epic was able to paint a black-and-white picture of good and evil.
  Fact 1.2: In that picture Epic itself was the clear underdog.
  Fact 1.3: Epic was able to do this because Google spent most of its first day on those explanations.

Retrieved items:
<item 1>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thank you for being here with us. It’s been a very engaging trial to watch. I’ve been there every day of the trial, and you’ve been there every day save one. So
my first question is why did you personally attend this trial every day save one — and what the heck happened on that one day?

Yeah, Epic is asking a lot of the court system and the jury here, spending four weeks on a major antitrust trial full of complicated facts and evidence. It
wouldn’t be right to start something like this and not show up. So I had to do that. And, you know, Phil Schiller sat throughout the entire Epic v. Apple trial,
as did I, so I think it’s just necessary to show respect for the legal process.

What happened on that one day you weren’t here?

Oh — sorry I can’t share, but there were some... non-Epic issues I had to deal with.

So this trial has been four years in the making.
</item 1>

<item 2>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What were your settlement talks with Google CEO Sundar Pichai like?

We can’t talk about content, but we met for an hour and had a professional discussion... in which we didn’t reach a settlement. We were rather far apart, let’s
say, because what Epic wants ultimately is free competition and fair competition for everybody, and the removal of the payments tie and removal of the
anticompetitive measures, which obviously leads to far better deals for consumers and developers.

What we don’t want is a special deal just for ourselves, and Google’s strategy so far as you’ve seen from all these deals has been to basically take out all
their potential competitors one at a time while they’re still weak and small, before they build up or unite into a formidable force. Epic’s not going to go
along with something like that and do a special deal just for ourselves.

You’ve said that a couple times, but the email that you accused Google of leaking suggested that Epic was interested in a special deal of some sort at that
time.
</item 2>

<item 3>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
February 2023

Twitter

Announced on February 26, the company laid off more than 200 employees, including Esther Crawford, Haraldur Thorleifsson and Leah Culver. Since Musk took over
Twitter in October last year, the company’s headcount has fallen by more than 70%.

Poshmark

Announced February 24, Poshmark confirmed with TechCrunch that less than 2% of its workforce was affected, primarily in the U.S. The company employs roughly
800+ employees.

Green Labs

We do not have an exact figure of how many Green Labs plans to lay off its staff. Green Labs confirmed to TechCrunch that it is conducting a round of layoffs
that could impact at least 50% of its workforce.

Chipper Cash

Announced on February 20, the African cross-border payments platform conducted a second round of layoffs just 10 weeks after it cut approximately 12.5% of its
workforce. Chipper Cash relieved almost one-third of its workforce, about 100 employees.
</item 3>

<item 4>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This wasn’t the only layoff round of the year. Unity lowered its headcount three times in 2023, affecting about 900 jobs. In its quarterly financial results in
November, Unity reported a yearly revenue increase of 69 percent and it told investors, “We continued to manage costs well.”

Sony cut 100 jobs at Bungie, a company it bought for $3.6 billion in 2022. According to developers that are still there, Sony executives are attempting to use
this upheaval to wrest more control of the studio from Bungie founders and leaders.

Epic Games fired roughly 830 people this year, or 16 percent of its staff. This included significant job cuts at Mediatonic, the studio behind Fall Guys that
Epic purchased in 2021.

Mediatonic

“For a while now, we've been spending way more money than we earn,” CEO Tim Sweeney wrote about the layoffs.
</item 4>

<item 5>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Alphabet

Google’s parent company announced laying off 6% of its global workforce on January 21, equating to 12,000 employees. These cuts impact divisions such as Area
120, the Google in-house incubator and Alphabet’s robotics division, Intrinsic.

Fandom

The entertainment company announced an unspecified number of employees impacted across multiple properties on January 20. According to a report by Variety, the
company employs around 500 people, and the layoffs have affected roughly 10% of its staff across different sites.

Swiggy

Announced plans to lay off 380 jobs on January 20 and shut down its meat marketplace.

Sophos

Announced on January 18, 10% of its global workforce, about 450 people were let go.

Microsoft

As announced on January 18, 10,000 employees will be impacted.

GoMechanic

Laid off 70% of its workforce on January 18.

Clearco

Announced on January 17, impacting 30% of staff across all teams.
</item 5>

<item 6>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
The channels publishers will pursue are varied, but Twitch streamers command a percentage of it. In recent years, we’ve seen a shift toward personality-based
promotion, with more and more publishers eschewing traditional press at launch in favor of coverage they can control. Blevins, by himself, apparently commanded
a $1 million payday for streaming Apex Legends, a game that’s popularity can be attributed to its enormous Twitch-focused launch.

Who can actually compete in the environment all of these conditions have created? Giant publishers who make live-service games, mostly. All 10 of the
most-watched games on Twitch in 2022 were live-service games with regular updates. And the cost goes beyond money. As Fortnite took off, developers at Epic
Games said that they experienced months of crunch that required 70-to-100-hour weeks to produce that industry-shaping content despite the publisher’s deep
pockets. That’s the model that a lot of this hinges on, an egregious work week that often still requires the cheap benefits of outsourcing and contractors who
are routinely laid off.
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Who will Epic and Google call as witnesses?

We are going straight to the top of the org chart, folks. Google CEO Sundar Pichai and Epic CEO Tim Sweeney are absolutely expected to attend. Android
co-founder Andy Rubin might be deposed, and Google says we’ll hear from current Android (and Chrome, and Chrome OS) boss Hiroshi Lockheimer. Google also tells
us representatives from Apple, Netflix, Motorola, and AT&T should testify as well.

How much hypocrisy can we expect in the courtroom?

I’d say quite a bit! For example:

Epic says it’s not in this for the money, it’s only suing to end Google’s monopoly — but Epic would surely profit if it doesn’t have to pay Google’s cut.

It looks like Epic’s going to argue that you’re locked into Android after buying an Android phone, but I can personally attest that switching back and forth
between the platforms is easier than ever.
</item 7>

<item 8>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
On January 19, the company announced it would end AmazonSmile.

Salesforce

Announced on January 4 that it’s cutting 10% of its workforce, impacting more than 7,000 employees. A month later, some Salesforce employees had just found out
they were also a part of the 10% layoff announcement.

Vimeo

Announced on January 4, cutting 11% of its workforce.

Editor’s note: Natasha Mascarenhas has contributed to this compilation.
</item 8>

<item 9>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
</item 9>

<item 10>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A comprehensive list of 2023 tech layoffs From major layoffs at Google, Amazon and Microsoft to small fintech startups and apps

Last year’s techwide reckoning continues. The tech industry has seen more than 240,000 jobs lost in 2023, a total that’s already 50% higher than last year and
growing. Earlier this year, mass workforce reductions were driven by the biggest names in tech like Google, Amazon, Microsoft, Yahoo, Meta and Zoom. Startups
across many sectors also announced cutbacks through the first half of the year. And while tech layoffs slowed down in the summer and fall, it appears that cuts
are ramping up yet again.

Many economists have cautioned against fears of a recession, which would seem like a reason for optimism. But the momentum for a tech sector rebound has been
slow to build, resulting in tech companies continuing to cut back on their workforces and pivot from a growth mindset to one based on efficiency in the face of
stubborn market conditions.
</item 10>

<item 11>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So there are absolutely no hard feelings, and we’re grateful that they joined the case because they did help in critical ways. Epic will continue to fight for
all developers, seeking remedies, and... I don’t know if Sundar is going to be calling me, but if he does, all of our discussions will be directed toward
solving the problem for everyone.

A lot has changed since you filed the original lawsuit over three years ago. For a bit there, it seemed like Epic had an interest in expanding outside of games
with acquisitions like Houseparty and Bandcamp. What changed? And did app store restrictions play into that at all?

Let’s see. No... well, not directly, anyway. We have a really broad strategy of building games and technology. We’ve had huge success in recent years with the
Unreal Engine gaining adoption among all kinds of industries even beyond games. We’ve also run into our own financial limitations. You know, we expanded the
company to nine times the size we were before Fortnite took off.
</item 11>

<item 12>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
Fortnite set a more cutthroat standard: free weekly content updates.

Epic could swing this ceaseless delivery because the bulk of its profits at the time came from its engine, which many games — including Fortnite’s competitors —
license. It was these deep pockets that made Fortnite possible in the first place: The bubbly shooter was initially a flop before Epic perfected the formula.
The cadence also let Epic quickly adapt Fortnite to whatever was trending or novel, whether that was Among Us or Splatoon. Much like Roblox, Fortnite became a
game that could give you anything — and if Epic didn’t emulate it, the players would.

“You can’t replicate Fortnite’s growth from, like, the [content creator] standpoint and also definitely from the developer standpoint,” says a Twitch streamer
turned game developer who requested anonymity because he isn’t authorized to discuss the popular AAA live-service shooter that he works on.

The conditions that made everything possible were unique, yet immaterial to publishers who could only imagine profits.
</item 12>

<item 13>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I understand the significance, but... you were there in person. You had a smile on your face, you shook the Google attorney’s hand, you clapped Bornstein on the
back. How did you feel in that moment?

Well, it was a great relief. The conventional wisdom that attorneys tell you is that when there’s a rapid jury verdict, it’s typically not good for the
plaintiffs making a complicated case, and so there was some trepidation going on — but it was awesome to see.

Something that we’d suspected all along was the jury was really following the case carefully. They weren’t snoozing off as you might expect with the complexity
of these documents and things — everybody was paying attention, they’re looking at the question asker and the witness and taking notes and reading documents. It
was just awesome to see that the justice system works, even with the most complex type of tech antitrust case that you can possibly throw at a jury.
</item 13>

<item 14>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
This comes just a few months after the company announced a significant wave of layoffs.

May 2023

Taxfix

Announced on May 30 that it has laid off 20% of staff—120 employees.

Meta

Announced May 24 that it is laying off about 6,000 people. In total, about 21,000 people have lost their jobs at Meta since November.

JioMart

Reliance Retail’s online shopping platform laid off over 1,000 employees on May 22, and plans to cut as many as 9,900 more roles over the coming weeks.

Krebs Stamos Group

Announced May 18 that the consulting firm laid off six people. In April, the firm had 18 employees including the founders. On May 18, the site shows only 14
team members.

TuSimple

Announced May 18 that it is laying off about 30% of employees.

Prior to the layoff, TuSimple had about 550 employees in the U.S.
</item 14>

<item 15>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
No, we’ve always turned down special deals just for Epic. We’ve always fought on the principal that all developers should be, you know, given the same
opportunities. One of the documents in evidence was a 2019 email I sent — right before Fortnite launched the Marvel season with all this amazing new content and
the Star Wars event featuring J.J. Abrams. Right before that, I sent an email to all the Google senior executives saying that we wanted to bring Fortnite to the
Google Play Store in time for that event, and we wanted them to allow us and all other developers to process certain payments and keep them all.

That was our proposal to Google in 2019. If Google had said yes to that, that would have been awesome for all developers — the Android ecosystem would have
become much, much stronger, and Google would be in a much better position in the smartphone industry than they are today. We would’ve never had a dispute
because the problem would have been solved.

It’s always been in Google’s power to solve this problem.
</item 15>

<item 16>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And so even as Microsoft was crushing corporate competitors, the developer experience was excellent. [Editor’s note: Netscape might feel differently.]

“Even as Microsoft was crushing corporate competitors, the developer experience was excellent.”

Google and Apple both treat developers as adversaries — they try to attack our revenue streams and prevent us from competing with their products. They’ve built
these massive self-preferencing schemes all around excluding developers and disadvantaging third-party developers. I think this is very shortsighted. I think
any tech company — Apple, Google included — would be much better off in the long term if they viewed developers as awesome partners and did everything they
could to support them and empower them and not get in their way financially.

And this has been our philosophy with Unreal Engine, for example, and the Epic Games Store. We just want to be a cool partner that helps other companies succeed
the way we do. And I think philosophy change... perhaps it will only come with a generational change in the company’s management.
</item 16>

<item 17>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So this trial has been four years in the making. I look back at a September 2019 email that was in discovery about a plan to draw Google into a legal battle
over antitrust. Can you describe what it felt like, after four years, to hear the jury find for Epic?

Well, it’s a great day for all developers to see that the Sherman Antitrust Act works in the new era of tech monopolies; we’ve not had a major antitrust verdict
against a tech company that meant change and benefits for everybody since the 1990s, with the US v. Microsoft. Back in the early days of the internet. So this
is an awesome thing and it’s much needed by the industry which is being strangled by a few gatekeepers imposing insane amounts of control and extracting huge
taxes, which not only raise prices for consumers but also make a lot of kinds of products just unviable.

I understand the significance, but... you were there in person.
</item 17>

<item 18>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Looking at this from a less academic perspective, it’s worth noting that Google charges up to ten times more per transaction than you pay with PayPal or a
credit card, which does seem like a lot! And though Epic can’t argue it in court, I’ll point out that Apple wasn’t able to explicitly justify a 30 percent fee
to that case’s judge.

On the other hand, it does kinda seem like Epic wants something for nothing! As far as I can tell, Epic hasn’t stated what kind of fee would actually be
reasonable for Google to charge in exchange for placement in Google’s store — instead, Epic CEO Tim Sweeney seemingly suggests there shouldn’t be a Google fee
at all if a developer wants to use its own payment system.

How did Epic’s argument fare against Apple?

Well… both sides lost! But Epic arguably lost more.
</item 18>

<item 19>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Informatica

Reported on January 11 that it will lay off 7% of it’s workforce, or 450 staffers globally.

Carta

Announced on January 11, the equity management platform cut 10% of its staff. Judging by LinkedIn data, the layoff could have impacted around 200 employees.

Citizen

Impacting 33 staff members on January 11.

Coinbase

To cut 950 jobs, or about 20% of its workforce, and shut down “several” projects, announced on January 10. This is the second round of major layoffs at the
crypto exchange, which eliminated 18% of its workforce, or nearly 1,100 jobs last June.

SuperRare

The NFT marketplace is cutting 30% of its staff, announced on January 6.

Amazon

Announced on January 5, eliminating more than 18,000 roles. This announcement extends a previously announced round of layoffs in November of 2022. On January
19, the company announced it would end AmazonSmile.
</item 19>

<item 20>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
Other companies with unions established in the past two years include Avalanche Studios, Anemone Hug, CD Projekt RED, Experis Game Solutions, Keywords Studios,
Sega of America, Tender Claws and Workinman Interactive.
</item 20>

<item 21>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The future of Google’s app store is at stake in a lawsuit by Fortnite publisher Epic Games. Epic sued Google in 2020 after a fight over in-app purchase fees,
claiming the Android operating system’s Google Play Store constituted an unlawful monopoly — while Google says its demands would damage Android’s ability to
offer a secure user experience and compete with Apple’s iOS. Follow along with updates here.

Bornstein showed jurors charts of Google’s fat app profit margins (70 percent on $12 billion in revenue a year, says Epic) and pointed out several ugly-seeming
ways Google has allegedly attempted to keep anyone from taking that money away — like paying game developers not to build their own app stores or standalone app
launchers like Epic did with Fortnite.

“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying,
‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”
</item 21>

<item 22>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Roofstock

Announced March 22 that it has laid off 27% of its staff, approximately 100 employees.

Twitch

Announced March 20 that it will lay off 400 employees.

Amazon

Announced March 20 another round of substantial layoffs, this time 9,000 people are set to lose their jobs. TechCrunch is hearing that around 10% of today’s
total came from AWS. As part of the new round of layoffs, Amazon is shutting down DPReview.

Livespace

Announced March 20 that it has laid off at least 100 employees, about 2% of the company’s workforce.

Course Hero

Announced March 16 that it has cut 15% of staff, or 42 people.

Klaviyo

Announced March 15 that it has laid off 140 of its staff across all teams.
</item 22>

<item 23>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The plant-based meat company says it will also review its entire global operations for cost-cutting opportunities.

Ava Labs

Is cutting 12% of its workforce, CEO Emin Gün Sirer confirmed November 6.

OpenSea

Laid off about 50% of its staff, the company confirmed November 3. CEO Devin Finzer posted a statement on X saying the NFT marketplace is now focused on
“OpenSea 2.0.”

Starz

Is laying off more than 10% of its staff ahead of the company spinning out from Lionsgate, CEO Jeffrey Hirsch announced November 3. The network and streaming
app will also exit Australia and the U.K.

Faire

Laid off about 250 people November 3 as part of restructuring. The wholesale marketplace raised a $416 million extension at a $12.6 billion valuation last year.

Niu

Confirmed to TechCrunch November 3 that it is issuing a round of layoffs affecting about 10% of their workforce, citing “fierce competition.”
</item 23>

<item 24>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Crunchbase

Announced on LinkedIn on July 9 that the company is laying off 52 employees.

Microsoft

Announced July 10 that the company is eliminating additional jobs a week after the start of its 2023 fiscal year. These layoffs are in addition to the 10,000
layoffs announced in January.

ClickUp

Announced on July 4 that it has laid off 10% of its workforce, about 90 people.

June 2023

Niantic

Announced June 29 that it laid of 230 employees — one year after it laid off around 90 employees.

Plex

Announced June 29 that it laid off approximately 20% of its staff, or 37 employees.

Grab

Announced the week of June 23 that it laid off more than a thousand people, or 11% of its staff.

Olx Group

Announced on June 20 that it has cut around 800 jobs globally. The news comes as the company started to close operations of its automotive business nit Olx
Autos in some markets.
</item 24>

<item 25>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.
</item 25>

<item 26>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
The gaming industry, which likes to boast about how much money it makes in comparison to other forms of entertainment, is laying off workers nearly every week
at top studios. Meanwhile, the overall gaming market is shrinking. The cracks in the system that produces and promotes big-budget games are getting hard to
ignore.

“I wouldn’t trade it for the world,” Tenney said in his goodbye video, which is overflowing with footage of esports casters and viewers in awe of his Fortnite
plays.

“I’ve been doing this since I was a kid, man,” he continued, shaking his head. “I feel like my childhood and even some of my adulthood kind of flushed away. I
feel like this is the end of the journey, man.”
</item 26>

<item 27>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
343 industries

These are just some of the biggest names in layoffs in 2023. Looking back on the carnage, it feels like a warning — as consolidation efforts increase, more game
studios will be controlled by just a handful of companies, and they’ll be vulnerable to moves like mass layoffs and closures. We’re laying the foundation for
the future of video games right now, and consolidation only makes the industry smaller and more generic, as accountants, investors and shareholders push for
low-risk concepts, rather than innovation and change.

What will rampant consolidation mean for all of these acquired studios in five years’ time? What will it mean when these teams aren’t shiny, new investments any
longer, and the people at the top are ready to get lean again? Remember that many of the shuttered studios listed above were purchased within the past three
years.

Being acquired is a cost-benefit analysis for smaller studios, where the benefits are immediate and the costs are potential. It’s easy to say that won’t happen
to us.
</item 27>

<item 28>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A GitHub spokesperson told TechCrunch that the layoff is part of the streamlining effort the firm had disclosed in February to cut roughly 10% of its workforce
by end of Q1 2023.

Disney

On March 27 in an internal memo to employees, Disney revealed there will be three rounds of layoffs, the first beginning this week. The job cuts will impact
approximately 7,000 employees, which was announced in February.

Salesforce

On March 24, Bloomberg reported that more layoffs could be on the way at Salesforce, quoting chief operating officer Brian Millham, who indicated that the
company could be adding to the ongoing job cutting at the CRM leader and in tech in general. If the layoffs happen, it would come on top of the 10% cut in
January.

Accenture

Announced March 23 that it plans to cut 19,000 jobs, or 2.5% of its workforce.

Indeed

Announced March 22 that it will lay off 2,200 employees, or 15% of its staff.
</item 28>

<item 29>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.
</item 29>

<item 30>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Original story continues:

Llamacorn what now? What is happening?

Before we get to the Llamacorn, a little background. Epic is of course the studio behind Fortnite, the extraordinarily popular free-to-play game. Fortnite makes
money by selling in-game items with its virtual currency V-Bucks. Players often buy V-Bucks the same place they play Fortnite. And until August 13th, 2020, if
the player used an Android or iOS device and installed the game through an official app store, that purchase triggered an in-app payment fee to Google or Apple.

Critics call such fees the “Google tax” or the “Apple tax,” and Epic definitely wasn’t a fan.

When Epic decided to take action against these respective “taxes,” it made August 13th, 2020 a very busy day for Apple, Google, Epic, and us here at The Verge.
First, Epic announced it was bypassing Apple and Google’s app store fees.
</item 30>

<item 31>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Insightec

Is reportedly laying off 100 employees, with 60 from the company’s Israel headquarters.

Udaan

Reportedly laid off up to 150 employees in a restructuring effort. This month, the business-to-business e-commerce startup scored $340 million in new funding.

Enphase

Is reducing their workforce by around 10%, impacting roughly 350 contractors and employees, president and CEO Badri Kothandaraman wrote in a blog post December
18.

Arm Holdings PLC

Reportedly laid off more than 70 software engineers in China and will relocate some roles.

Delivery Hero

Is shutting down operations in Taiwan and Turkey and plans to eliminate roles at its Berlin headquarters, the company said December 18.

eBay

Is reportedly eliminating nearly 10% of its workforce in Israel. It’s the company’s second round of layoffs this year.

Intel

Will cut 235 employees at its Folsom office campus at the end of the year, the company reported to state employment officials December 15.
</item 31>

<item 32>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Enough that the judge has decided to explicitly tell the jury we’re not seeing all the evidence, because Google employees all the way up to CEO Sundar Pichai
were caught setting those conversations to auto-delete. This issue has come up in the US v. Google antitrust trial in Washington, DC — still ongoing at the
start of Epic’s trial — as well.

And the dirt isn’t all on Google’s side. Epic was caught straight-up admitting it premeditated the whole plan to bypass Google’s payment systems. Since Google
is countersuing over that, it could be a big deal.

When you say “straight-up admitting...”

Here is an email from Epic CEO Tim Sweeney that, I think, speaks for itself:

“Here is our official plan for communicating with Google about bypassing the Google Play Store: SAY NOTHING TILL IT SHIPS” Image via Epic v. Apple

We’ll have a whole cast of intriguing people called as witnesses during the trial who might dish even more.

Who will Epic and Google call as witnesses?
</item 32>

<item 33>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
He continued, “I had long been optimistic that we could power through this transition without layoffs, but in retrospect I see that this was unrealistic.”

Electronic Arts was one of the first video game companies to institute significant layoffs this year, with a reduction of 6 percent of its workforce, or about
800 employees, in March. EA later cut jobs at Dirt and F1 studio Codemasters, which it purchased in 2021 for $1.2 billion. EA culled an estimated 1,130 jobs in
2023.

CD Projekt RED and Sega each laid off about 100 people in the past 12 months, while Ubisoft fired an estimated 255 employees. Microsoft cut 10,000 jobs across
its businesses early in the year, and that included about 100 people at Halo studio 343 industries.

343 industries

These are just some of the biggest names in layoffs in 2023.
</item 33>

<item 34>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Lyft

Announced on April 21 that employees will learn whether they have a job or not via an email that will be sent out April 27.

Lyft layoffs to affect 26% of workforce, or about 1,072 people as promised on April 27.

Meta

Announced April 18th that it is expected to lay off 10,000 jobs in the coming months. This is on top of the 11,000 jobs that were cut in November.

Redfin

Announced April 13th that it has laid off 201 employees, about 4% of its workforce. This is the third time the Seattle-based real estate company has reduced its
workforce since June.

Apple

Bloomberg reported on April 3 that Apple is laying off a small number of roles on its corporate retail teams.

March 2023

Netflix

Announced on March 31, Netflix confirms a “handful of layoffs,” which includes two longtime executives. The exact number of layoffs is unclear.
</item 34>

<item 35>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
The most extreme negative outcomes for an acquired indie studio are, of course, layoffs and closures. We saw a lot of these in 2023.

Layoffs

The post-acquisition power dynamic is playing out in public and in real-time. It’s estimated that more than 9,000 people in video games were laid off this year
and the firings affected teams of all sizes. This is a crisis amount of cuts. In 2022, just 1,000 video game jobs were lost, according to layoffstracker.com.

The Embracer Group provides the clearest example of rampant, surprise layoffs in 2023. Embracer has spent the past few years acquiring prominent midsize
studios, including Gearbox Software (Borderlands), Crystal Dynamics (Tomb Raider), Eidos-Montreal (Deus Ex) and Square Enix Montreal (Deus Ex Go).
</item 35>

<item 36>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I don’t know if that was on behalf of you and fellow developers. Can you tell us what that email was actually about?

Google didn’t leak the email; they leaked their ridiculously biased summary of it to Abner Li at 9to5Google. Don Harrison testified at trial that he didn’t
think Google leaked to the press. And then he was presented with a document including Google’s press team and Sameer Samat, head of all Android, basically
summarizing the articles that appeared as a result of Google’s leaking of our plans to the press.

So I was wondering if you could tell me what the email actually said.

Oh, it’s in evidence! This was the one I mentioned! Right before Christmas-ish 2019, we were planning to launch the new Marvel season with a bunch of Star Wars
content and its awesome lineup. I asked Google’s executives to let Fortnite come to the Google Play Store using our own payment method and to open up, you know,
using their own payment methods to all developers, not just Epic.
</item 36>

<item 37>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
VinFast

VinFast has not shared how many employees have been cut, but a LinkedIn post from a former employee said “nearly 35 roles” were affected. Announced on February
6.

Dell

Announced February 6, impacting 6,650 people, or 5% of worldwide workforce.

Getaround

Announced February 2, 10% of staff — about 42 employees.

Pinterest

Announced February 2, 150 employees impacted. This is the second job-cutting move within weeks of the first round in December 2022.

Rivian

Announced on February 1, cutting 6% of its workforce for the second time in less than a year.

January 2023

SoFi Technologies

Announced on January 31, cutting 65 jobs, or about 5% of its 1,300-person workforce. First reported by The Wall Street Journal.

NetApp

Announced on January 31, impacting 8% of its staff — about 960 people.
</item 37>

<item 38>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Sunfolding

Is reportedly winding down its operations. On its website, Y Combinator categorizes Sunfolding as an “inactive” company.

Zulily

Will lay off more than 500 employees after closing fulfillment centers in Ohio and Nevada in February 2024.

D2iQ

Is shutting down its operations and selling its assets to Nutanix.

Atmosphere

Has reportedly eliminated a large part of its workforce. The business-focused streaming TV service raised $100 million in 2022.

Tidal

Is laying off more than 10% of its staff, the company confirmed to TechCrunch December 7.

ZestMoney

Is shutting down operations. The Goldman Sachs-backed “buy now, pay later” startup was once valued at $450 million.

Navan

Is cutting about 145 roles as the travel startup previously known as TripActions plans to become public.

Incredibuild

Is laying off 20% of its workforce. The software development startup previously raised $35 million in a Series B funding in 2022.
</item 38>

<item 39>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
CoinDesk

Announced on August 14 that it is cutting 16% of staff.

NCC Group

Confirmed on August 10 that it is making more layoffs, its second round of layoffs in just six months. The company declined to say how many employees are
affected by this round of cuts. NCC Group cut 125 employees in February.

Rapid7

Announced on August 9 that it plans to lay off 18% of its workforce, affecting more than 400 employees globally.

Mobile Premier League

Announced on August 8 that it is laying off approximately 50% of its employees or 350 people.

Astra

Announced on August 4 that it has laid off 25% of its workforce since the beginning of the quarter.

Discord

Announced on August 3 that it has let go of nearly 40 employees or 4% of its workforce.

HackerOne

Announced on August 2 that it is laying off 12% of its workforce, approximately 50 employees.
</item 39>

<item 40>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
In October 2019, Epic internally said it might pursue an aggressive pursuit model with the Epic Games Store, where it would aggressively pay more for exclusive
games and really pull in more and more users. Are you in the aggressive pursuit model?

No, we’re in a different model, which is called the “Ridiculously aggressive pursuit model.” If it’s a 10-slide deck, that’s on slide 11.

Did you get a bingo on your Verge bingo card?

Yeah, I think it was almost a complete shutout. I think there was only one cell in the entire board that was missing. I was really impressed — the thing for
that bingo card that really impressed me was “lick the cookie.” Do you remember? You may not even be old enough, but “lick the cookie” was in the 1999 US v.
Microsoft antitrust trial.
</item 40>

<item 41>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Hippo

Plans to eliminate 120 roles, the company said in an SEC filing October 30.

Liberty Mutual

Plans to cut 850 members of its staff, the insurance company confirmed October 27.

Salsify

Cut 110 jobs and are moving some roles abroad, CEO and co-founder Jason Purcell announced October 25.

Slync

Is shutting down, months after its former CEO Chris Kirchner was arrested on fraud charges and sued the company to pay for his legal bills.

Shipt

Announced October 24 in a company statement that it laid off 3.5% of their workforce and will close many open roles.

Pebble

The X alternative previously known as T2 announced October 24 that it is shutting down its operations.

Nokia

Announced October 19 that it will cut up to 14,000 jobs, or 16% of its workforce, following a sizable dip in Q3.

Roblox China

Cut 15 roles on October 24, two years after the joint effort with Tencent abruptly paused its service.
</item 41>

<item 42>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Didn’t other parties sue Google too?

They did! And then, they all settled. All 50 state attorneys general and a number of consumers have tentatively settled with Google over app store antitrust
claims, though it’s not clear what the deal might include.

Match Group, the company behind Tinder, Match.com, OkCupid and other dating apps, just reached a surprise last-minute settlement too, one that makes it sound
like Match effectively gave up.

Epic stands alone.

Could we get a settlement between Google and Epic as well?

Epic seems to have no interest in settling as long as the “Google tax” exists, and the company’s repeatedly said it’s not interested in cash. So far, the
company’s been willing to push these cases all the way through multiple appeals, cost be damned.

It’s not clear what Google could proactively offer that would satisfy Epic, since CEO Tim Sweeney has also publicly suggested that Google would simply
circumvent any settlement it offers.
</item 42>

<item 43>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
September 2023

IronNet

In a regulatory filing published September 29, the cybersecurity startup has ceased all business activities as it prepares for Chapter 7 bankruptcy and has laid
off its remaining staff.

Naughty Dog

Is reportedly cutting at least 25 contract developers, two sources familiar with the situation told Kotaku.

Epic Games

Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.

Talkdesk

An impacted employee confirmed on September 27 that at least 140 people, if not more, were affected by Talkdesk’s third round of layoffs in less than 14 months.

Byju’s

Announced on September 26, the company plans to cut 5,000 jobs due to restructuring efforts. The startup has eliminated more than 10,000 positions in the past
two years.

Roblox

Confirmed to TechCrunch on September 21, Roblox let go of 30 employees in its talent acquisition organization. No other teams are impacted.
</item 43>

<item 44>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 44>

<item 45>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Groupon

Impacting another 500 employees announced on January 31. The company said this new set of layoffs will be spread across the first two quarters of 2023.

Impossible Foods

Reportedly affecting 20% of its staff, over 100 employees, Bloomberg reported first.

PayPal

Announced on January 30, about 2,000 full-time employees, or 7% of its workforce, were affected.

Arrival

Announced on January 30, with a newly appointed CEO, slashing 50% of its workforce — 800 employees globally.

Waymo

The self-driving technology unit under Alphabet quietly laid off workers on January 24, according to The Information and several posts on LinkedIn and Blind.
It’s not yet clear how many of Waymo’s staff will be affected.

Spotify

Announced on January 23, impacting around 6% of its global workforce — around 600 employees.
</item 45>

<item 46>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 46>

<item 47>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Mojocare

Announced on June 18 that the company will be laying off some 150 employees, a majority of the workforce.

Tada

Operated by Korean ride-sharing company Socar’s subsidiary VCNC announced on June 15 that it plans to let go of at least 50% of its staff or about 45 people at
Tada.

Better.com

Announced on June 7 that it laid off its real estate team, it is not clear how many people were impacted.

Ursa Major

At least 14 people have been laid off from the rocket engine startup, announced on June 7. Ursa Major did not respond to TechCrunch’s inquiry into the layoffs,
or the number of people affected. According to LinkedIn, 292 people are listed as currently working at Ursa.

Spotify

Announced June 5 that it will cut 200 jobs in its podcast unit, leading to a workforce reduction of 2%. This comes just a few months after the company announced
a significant wave of layoffs.
</item 47>

<item 48>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Or at least it was worse, before they were destroyed.”

The best Google could do in response was to plant its own feeble seed with the jury, too: “Is Epic using the chats to distract me from all the evidence I do
see?”

“It’s true that Google could have automatically saved all chats for all relevant employees, but just because Google didn’t save some chats didn’t mean it
violated antitrust laws,” Pomerantz argued.

Epic’s opening statements seemed to paint a clearer picture for the jury than those from Google. But things got complicated for both parties when the first two
witnesses — Epic Games Store head Steve Allison and Yoga Buddhi CEO Benjamin Simon, who also appeared in the earlier Epic v. Apple trial — took the stand.

Both Epic and Google spent a long, long time on subtle lines of questioning.
</item 48>

<item 49>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
[Editor’s note: Every journalist in the courtroom agreed the jury was attentive; most were also surprised by the rapid verdict.]

They got it, they got it quickly, and they were able to pull apart what was going on and contrast it with the fiction of the story that Google was trying to
tell.

What was something that surprised you in the courtroom, something you heard come to light that you hadn’t heard before?

I’d suspected a lot of the practices that Google had, you know, since 2018 or so when we first started this, to such an extent that some folks would
occasionally call me a conspiracy theorist. It was really, really interesting to see that my understandings of what Google was doing behind the scenes were
actually true — you’re leaking our conversations to reporters to get negative stories written about us; you’re paying other developers off to convince them not
to launch their own stores; they were going around and paying carriers and OEMs secretly not to carry competing stores.
</item 49>

<item 50>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 50>

<item 51>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 51>

<item 52>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
The deal guaranteed a pay raise, paid holidays and sick leave, and a copy of Starfield, the game they helped ship.

Bethesda Softworks

“We are now stronger at the bargaining table and are working to secure a fair contract for all workers — direct employees and contractors," ZeniMax union member
Chris Lusco said. "We are all a part of ZeniMax Studio’s success and we all deserve our fair share. We hope to set a new precedent for workers across Microsoft
and the entire gaming industry so that all workers, regardless of their employment status, are able to improve their working conditions through collective
bargaining."

Meanwhile, executives at Microsoft’s newest acquisition, Activision Blizzard, spent the past few years stalling internal unionization efforts. However, QA
employees at Raven Software, a subsidiary of Activision, successfully voted to unionize in May 2022. Microsoft has vowed to respect organization attempts now
that Activision-Blizzard-King is under its control.
</item 52>

<item 53>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This was a year of upheaval in video games. The industry has shapeshifted over the past 12 months, and it’s not all due to Microsoft’s lengthy acquisition of
Activision, Blizzard and King. While Xbox executives were defending the legality of a $69 billion deal that would create the third-largest video game studio in
the world, smaller companies were firing staff and shutting down entire teams, even amid fervent collective-bargaining efforts. It’s been a wild ride.

In 2023, the main factors molding the video game landscape were consolidation, layoffs and unionization, with each of these phenomena feeding into each other.
This past year, the video game industry shrank, even as it grew financially.

Consolidation

When Microsoft's purchase of Activision-Blizzard-King was finally approved on October 12, 2023 after nearly two years of regulatory hurdles, it became the
world’s third-largest video game studio by revenue.
</item 53>

<item 54>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
Even still, Tencent is bigger than both Sony and Microsoft. Tencent is not a console manufacturer, so it isn’t a household name among most players, but it’s one
of the largest companies in the world, and it wields a ridiculous amount of financial power in video games. Tencent owns a portion of Bloober Team, Bohemia
Interactive, Don’t Nod, Epic, Paradox Interactive, PlatinumGames, Remedy Entertainment, Roblox and Ubisoft, among others. It has a majority stake in Supercell,
Grinding Gear Games, Klei Entertainment, Tequila Works, Techland, Yager Development and others. It fully owns Riot Games, Funcom, Sharkmob, Turtle Rock Studios,
and, of course, others. It also runs multiple internal development companies, including the Level Infinite and Tencent Games publishing labels.

Xinhua News Agency via Getty Images

Sure, Sony has a stake in Epic, but Tencent’s is bigger.
</item 54>

<item 55>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Multiverse

Will lay off nearly a third of their U.S. staff. It’s the second round of layoffs at the U.K.-based unicorn company in just over a month.

ONE

Is cutting its workforce by around 25% November 27, citing “market conditions” affecting the EV battery startup.

ByteDance

Started mass layoffs November 27 in its gaming department, Nuverse, after two years of tepid performance. The number of affected employees is unknown.

Veev

Is closing up shop, according to multiple reports, after reaching unicorn status in 2022.

Tulip

Laid off 25 employees, CEO Ali Asaria announced November 24.

Anar

Is shutting down operations and returning capital to investors, CEO Nishank Jain announced November 23.

Alerzo

Cut 100 workers November 21, reportedly due to increases in automation. The recent layoffs come after Alzero cut 15% of its workforce earlier this year and 5%
in 2022.
</item 55>

<item 56>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 56>

<item 57>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
But tracking these layoffs helps us to understand the impact on innovation, which companies are facing tough pressures and who is available to hire for the
businesses lucky to be growing right now. Unfortunately, it also serves as a reminder of the deeply human impact of layoffs and how risk profiles could evolve
from here.

Below you’ll find a comprehensive list of all the known layoffs in tech that have occurred in 2023, to be updated monthly. If you have a tip on a layoff,
contact us here. If you prefer to remain anonymous, you can contact us here.

The running total of layoffs for 2023 based on full months to date is 224,503, according to Layoffs.fyi. Tech layoffs conducted to date this year currently
exceed the total number of tech layoffs in 2022, according to the data in the tracker.

Data visualization by Miranda Halpern, created with Flourish

December 2023

ShareChat

Is cutting 15% of its workforce in a “strategic restructuring” effort for 2024.
</item 57>

<item 58>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Last October, Davison told TechCrunch that Clubhouse had close to 100 employees.

Dropbox

Announced April 27 that it would be laying off 500 employees or 16% of staff.

Amazon

Announced on April 26 that it is shutting down its Halo Health division, effective July 31, among other divisions. The layoffs are part of the 9,000 employees
announced in March. Including the 18,000-person layoffs announced in January, this brings the total to 27,000 job cuts or 8% of Amazon’s corporate workforce
this year.

Rapid

Announced on April 25, Rapid, previously known as RapidAPI, lays off 50% of its staff. The layoffs are believed to have impacted 115 people.

Anthemis Group

Announced on April 25 that it will lay off 16 people from its staff, or 28% of employees.
</item 58>

<item 59>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Microsoft

As a part of its recent announcement to layoff 10,000 people, Microsoft laid off an entire team dedicated to guiding AI innovation that leads to ethical,
responsible and sustainable outcomes. On March 27, Microsoft laid off 559 workers from its Bellevue and Redmond operations.

Meta

CEO Mark Zuckerberg confirmed rumors March 14 that the company will be cutting 10,000 people from its workforce and around 5,000 open roles that it had yet to
fill.

Y Combinator

Announced March 13 that it will impact 20% of staff, or 17 team members.

Salesforce…continued

Salesforce first announced that it was laying off 10% of the workforce in January, but some employees didn’t know until February. The week of March 10, more
employees are just learning they have been laid off. Salesforce confirmed that these layoffs were part of the 10%.

Atlassian

Announced on March 6, Atlassian is laying off about 500 employees, or 5% of its total workforce.
</item 59>

<item 60>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We established a market worldwide, excluding China. So any remedies, we would presume, would be worldwide. We also have the European DMA; we have Epic v. Apple
and Epic v. Google cases coming up in Australia, and another one in the UK.

It’s not just Epic anymore, there’s a lot of legislators, there’s a lot of regulators, and there’s other litigation all pushing in the direction of openness.
And we’re going to do absolutely everything we can.

Do you have anything to say to your former partners in litigation, Match Group, that abandoned you at the last minute and may now be regretting it?

Oh, yeah, no. Match has been an awesome partner and a fellow member of the Coalition for App Fairness. I really hope that they got out of their settlement what
they needed to get for their business. Very few companies have the resources that Epic has to fight multinational litigation against the world’s two most
powerful companies.
</item 60>

<item 61>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
In the past decade, Embracer grew its portfolio to cover more than 100 game studios, including Volition (Saints Row), Coffee Stain (Goat Simulator), Free
Radical Design (TimeSplitters) and Zen Studios (Pinball FX). The holding company also secured the rights to The Lord of the Rings in 2022, promising to turn it
into “one of the biggest gaming franchises in the world.”

Volition

In June 2023, Embracer announced a six-year, $2 billion funding deal had fallen through, and it was going to restructure — meaning, layoffs and studio closures.
Since this announcement, Embracer has shut down Volition, Free Radical Design and Campfire Cabal, it divested Goose Byte and it’s fired developers at Saber
Interactive. More than 900 people lost their jobs during these moves. Meanwhile, Embracer’s share price rose by 11 points in November.

This wasn’t the only layoff round of the year.
</item 61>

<item 62>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
You know, we expanded the company to nine times the size we were before Fortnite took off. And so we’re just trying to forge the strategy where we’re living
within our means and doing everything we absolutely can.

But we’ve gotten so much traction with Fortnite in so many different ways, and with the Epic Games Store, right, we’re going to continue investing heavily here.
The Epic Games Store is the untold success story in the background. Since it launched in 2018, we now have 80 million monthly active users. Steam has 120
million, so we’re catching them fast! Fortnite has more concurrent users right now than all Steam games combined. You should really expect dramatic changes that
benefit all developers over the next few years.

[Editor’s note: Sweeney isn’t right about Fortnite having more concurrent users than Steam. Epic spokesperson Natalie Muñoz confirmed Sweeney misinterpreted a
Polygon story about how Fortnite hit 7.6 million players at a time when the entirety of Steam had 10.1 million.
</item 62>

<item 63>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 63>

<item 64>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 64>

<item 65>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
But his ascent was a complicated one that reflected many of gaming’s biggest growing pains during the influencer era. Tenney came up as an early Fortnite
esports champion; we now know that esports is a bubble that’s burned millions of dollars. Tenney rode a Fortnite wave that raised billions of dollars for Epic
Games; a few years later, the publisher laid off 830 of its employees. When Tenney blew up, everyone was salivating over Drake and what his livestream
appearance signaled for Twitch’s future; now both the rapper and many of the platform’s stars are better known for promoting gambling over hits. Tenney spurred
headlines after joining FaZe Clan, a gaming lifestyle brand so lucrative it was endorsed by celebrities like Snoop Dogg; FaZe Clan is now an infamous penny
stock that tried to rip him off, Tenney says.

Where all these disasters converge is Twitch, the livestreaming platform that’s home to over 30 million daily viewers.
</item 65>

<item 66>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Tekion

Announced on August 2 by Inc42, the startup laid off around 300 employees or 10% of its workforce. Of the 300 employees being let go, about 200 were from
Tekion’s India office.

Planet

Announced on August 1 that it is laying off 117 employees, or 10% of staff.

July 2023

Dunzo

Announced July 19 that it is postponing employee salaries for a month and plans a new wave of layoffs as soon as July 20. Dunzo executives informed employees
that they will likely be cutting over 200 jobs, in what will be its third layoff this year. The company has eliminated about 400 jobs this year so far.

Cameo

Announced on July 18 that it intends to lay off 80 workers.

Skill-Lync

Announced on July 13 that the Indian startup has laid off 20% of its workforce, around 225 employees.
</item 66>

<item 67>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
This comes after its first round of layoffs, which eliminated 150 roles about a year ago.

Shopify

Announced May 4 that it is laying off 20% of its workforce, impacting more than 2,000 people. It’s also selling its logistics business to Flexport for roughly
13% in stock.

Bishop Fox

Announced May 3 that it laid off around 50 employees — or 13% of its workforce — on May 2.

Neato Robotics

Announced May 1 that Vorwerk-owned Neato Robotics is shutting down, with nearly 100 employees impacted by the move.

April 2023

Clubhouse

Announced April 27 that it has laid off more than 50% of staff. A spokesperson for Clubhouse declined to comment on the number of people impacted by today’s
workforce reduction or the number of employees who remain at the company. Last October, Davison told TechCrunch that Clubhouse had close to 100 employees.
</item 67>

<item 68>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Chipper Cash relieved almost one-third of its workforce, about 100 employees.

Evernote

On February 17 the company confirmed laying off 129 people.

Jumia

Announced February 16 that it cut 20% of its staff, or more than 900 positions across its 11 markets, in Q4 of 2022.

Convoy

Announced on February 16 that it is shuttering its Atlanta office and laying off workers as part of restructuring. This is the third time in less than a year
that the company has laid off workers.

Sprinklr

Announced on February 15 that it will impact 4% of its global workforce — or more than 100 employees.

iRobot

Announced on February 13 that it will lay off 7% of its workforce, roughly 85 employees.

Twilio

Announced on February 13 that it will impact around 17% of its global workforce, about 1,400 people.
</item 68>

<item 69>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The software development startup previously raised $35 million in a Series B funding in 2022.

Bill

Will cut around 15% of its workforce, the company announced December 5, and close its Sydney office to increase profitability.

CourseHero

Announced December 5 it eliminated 23 employees across engineering, operations and development.

Pivo

Shut down operations December 5. The Nigerian fintech startup previously raised $2 million in a seed round.

Twilio

Announced hundreds of job cuts December 4, amounting to 5% of the company’s workforce.

Spotify

Is eliminating about 1,500 jobs, the company announced December 4. This marks Spotify’s third round of layoffs this year.

Bending Spoons

Has laid off the entire staff of Filmic, the team behind popular video and photo-editing apps that Bending Spoons acquired in July.

November 2023

Zepz

Cut 30 roles November 30 after the fintech unicorn eliminated 26% of its workforce earlier this year.
</item 69>

<item 70>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
That was our proposal.

Oh, I think I read that one in court. I didn’t realize that was the same thing. It’s so far off from what I heard [about a “special billing exception”] that I
didn’t recognize it being that. Does winning the verdict in this Google case help your appeal with the Apple case in any way?

There’s no linkage between the cases and law, so it would just come down to whether the court is in any way following current events on this topic. But there’s
no legal connection between the two. It really comes down to the decisions of the humans involved: the justices and the support teams involved when considering
the appeal.

Google will appeal this; they’ve told us they’re going to challenge the verdict. Will Epic rely on this ruling in the meantime?

We’re not going to wait. We’re going to do absolutely everything we can as quickly as we can to start changing the world. We not only have this verdict here in
the United States, it is a worldwide verdict, right?
</item 70>

<item 71>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Domo

Made reductions to 7% of staff, in addition to reductions in its contract workforce, CEO Joshua G. James announced in the company’s Q3 earnings call November
30.

Mojo

Is laying off 20% of its staff, the company announced to staff November 30. The A-Rod and Marc Lore-founded sports betting startup previously raised $100
million.

Loco

Is cutting its workforce by about 36%, co-founders Anirudh Pandita and Ashwin Suresh announced at a November 30 town hall.

Unity

Is laying off 265 workers November 29 after eliminating its Wētā Digital division, which it acquired in 2021.

Tier Mobility

Is cutting roughly 22% of its staff, CEO and co-founder Lawrence Leuschner announced November 28.

Dataminr

Is laying off about 20% of its staff starting November 28, citing “the recent rapid advancements of our AI platform,” according to a memo from founder and CEO
Ted Bailey.
</item 71>

<item 72>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Bandcamp

Eliminated 50% of its staff on October 16 after ownership changed from video game company Epic to music licensing platform Songtradr.

LinkedIn

Confirmed October 16 that it will cut 668 more jobs, bringing the total to nearly 1,400 this year, with the bulk of the most recent cuts impacting those with
R&D roles.

Flexport

On October 13 began laying off 20% of its workers, about 600 people, as it prepares for restructuring, and continues a cost-cutting campaign that began with
rescinding offers and working to lease office space.

Qualcomm

Is cutting 1,258 jobs in two of its California offices. According to a filing with the California Employment Development Department, the semiconductor company
plans to make the reductions in December.

Blue Origin

Laid off 40 people October 10, including software engineers and program managers, according to The Information.

Stitch Fix

Will lay off 558 employees after the personal styling subscription service closes a distribution center in Dallas.
</item 72>

<item 73>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Tropic

Eliminated 26 roles, CEO David Campbell confirmed in a LinkedIn post October 21.

Convoy

Is halting operations due to a “massive freight recession,” CEO Dan Lewis announced in a memo October 19, a week after laying off 20% of its workforce.

LegalZoom

Plans to lay off more than 100 workers in its sales division, the company disclosed in a Worker Adjustment and Retraining Notification Act letter.

Plume

Laid off more than two dozen workers, TechCrunch learned October 18, about a year after the trans healthcare startup raised a $24 million Series B.

Google

Eliminated at least 40 roles in its news division, Google confirmed October 18.

Expedia Group

Cut around 100 jobs, Skift reports, in its second round of layoffs in recent months. The cuts reportedly impacted employees across data and AI.

Stack Overflow

Laid off 28% of its staff, more than likely impacting 100 people, the Prosus-owned company announced October 16.
</item 73>

<item 74>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
The lawyers can see them, but I saw almost all the key elements in this case at the same time the jury saw them. Two things stand out big time. Google’s Project
Hug was an astonishingly corrupt effort at a massive scale — the same corrupt deal structure they tried to do with Epic when they wanted to pay us off to launch
Fortnite on Google Play and not launch our own store.

As soon as we thwarted their effort, they went around to 27 different developers and offered each one a payoff to undermine any effort we had to get their games
onto our store exclusively. Activision and Riot and Supercell had direct distribution plans that they were planning on; Google paid them not to pursue those
plans. Just direct blatant violations of anti-competition law, it’s crazy a company of Google’s scale would do that.

“If you’re a smaller developer than Spotify, you get screwed.”

The other astonishing one was the Spotify deal. Spotify is the one company that had comparable negotiating power to Fortnite.
</item 74>

<item 75>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
</item 75>

<item 76>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Spotify is the one company that had comparable negotiating power to Fortnite. Instead of using their power to fight for the good of all developers, they did a
special deal with Google. Google gave them a 0 percent fee. Google let Spotify process their own payments, and Spotify kept 100 percent. They do it for Spotify
and for nobody else. If you’re a smaller developer than Spotify, you get screwed.

When Spotify uses Google’s own payment service, instead of paying the 30 percent that Google forces other developers to pay, they pay 4 percent. That’s what the
rate should be! Four percent is a perfectly reasonable rate for an unbundled payment system.

If instead of offering you a $147 million deal, Google said, “You can pay 0 percent to use your own payments system or 4 percent for Google Play billing,” would
you be here today? Would you have fought this lawsuit to begin with if they’d simply offered something more fair to you?

No, we’ve always turned down special deals just for Epic.
</item 76>

<item 77>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
GitHub

Announced February 9, 10% of its staff will be impacted through the end of the company’s fiscal year. Before this announcement, which was first reported by
Fortune, GitHub had about 3,000 employees.

Yahoo

Announced on February 9, 20% of its staff, impacting 1,600 employees in its adtech business. Yahoo is the parent company to TechCrunch.

GitLab

Announced February 9 that it’s reducing its headcount by 7%. The round of redundancies will impact around 114 people, though that specific figure is dependent
on its actual headcount as of February 9.

Affirm

Announced on February 8 that it is reducing its staff by 19%, or about 500 employees, and shutting down its crypto unit.

Zoom

Announced the cut of 15% of its staff, or 1,300 people on February 7.
</item 77>

<item 78>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
C3.ai

Reportedly cut jobs across departments November 20, citing a need to reduce costs. People familiar with the matter told Bloomberg that several impacted
employees received just one month of severance.

Jodo

Reportedly cut around 100 roles across engineering, customer success, data and sales November 20.

Amazon

Is reportedly laying off “several hundred” employees in their Alexa division and its freshly launched Artificial General Intelligence team.

FreshBooks

Laid off 6% of their workforce November 17 as the company shuts down operations in Raleigh, North Carolina. President Mark Girvan and CEO Don Epperson are also
leaving the company.

Beamery

Is reportedly cutting 25% of its workforce amid a larger restructuring and cost-cutting effort.

Paystack

Laid off 33 employees in Europe and Dubai November 16 as the company cuts down on operations outside of Africa.

Sega

Is reportedly planning to “phase out” 80 temporary workers by 2024.
</item 78>

<item 79>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Also, that one’s not over till the Supreme Court weighs in or declines to take a look.

2) Google can’t tell jurors that Apple won its case, or that other plaintiffs settled. The judge in this case says so!

3) Oh, and by the way, this is a jury trial.

Wait, why does that matter?

Epic and Google have to convince a jury, not the judge, which is totally different from how the Apple case played out. (That one was a “bench trial.”)

Maybe all the evidence of tricky deals inside Google might sway a jury against the company? Maybe Google scaremongering that sideloaded apps equal gaping
security holes will sway a jury against Epic instead? Who knows!

(If you’re a Epic juror reading this — stop! Judge Donato explicitly said you’re in a “news-free bubble” through mid-December, folks.)

Didn’t other parties sue Google too?

They did! And then, they all settled.
</item 79>

<item 80>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 80>

<item 81>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
It’s easy to say that won’t happen to us. But it can happen, and it does, and as consolidation increases, bulk layoffs are only going to occur more often.

Unions

Unionization is one approach that can help protect the livelihoods of people in the video game industry, and there was progress on this front in 2023.
Developers at multiple studios now have union support, from small indies to AAA powerhouses.

Microsoft is currently the home of the industry’s largest union, with representation for more than 300 quality assurance workers at ZeniMax Media. ZeniMax is
the parent company of Bethesda, id Software and Arkane, and Microsoft purchased the whole caboodle for $7.5 billion in 2021. Microsoft formally recognized the
ZeniMax union this January and the parties started negotiating in April. In December, Microsoft announced it would hire 77 contract workers as full-time
employees under the ZeniMax Workers United-CWA union.
</item 81>

<item 82>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Tim Sweeney finally has a win.

On Monday, a federal jury surprised the world by siding with Fortnite maker Epic Games in its fight to break Google’s control over Android apps — even though
“walled garden” rival Apple almost entirely won a similar case two years ago. The nine-person jury decided that Google has an illegal monopoly over Android app
distribution and in-app payment systems, and that Google illegally tied its Google Play billing system to its app store.

Sweeney is Epic’s CEO, co-founder, and importantly its controlling shareholder. He’s the one behind these lawsuits, and it was his idea to challenge these
companies in court. It’s been his fight from the very beginning, and he watched almost the entire trial in person from the best seat in the house — with a clear
view of the jury, the judge, each witness, and the faces of Google’s lawyers.

Last night, I asked him why, what he learned, and what’s next.

This interview has been lightly edited for brevity and clarity.

Thank you for being here with us.
</item 82>

<item 83>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The exact number of layoffs is unclear. Netflix is scheduled to report Q1 2023 results on April 18.

Roku

Announced March 30 that it is letting go of about 200 employees, or 6% of its workforce. The company had laid off 200 U.S. employees back in November.

Unacademy

Announced March 30 that it has laid off more than 350 roles, or 12% of its workforce — just over four months after cutting about 350 roles in November.

Shift Technologies

Announced March 29 that it laid off 30% of its workforce in Q1 2023.

Lucid

Announced March 28 that it is laying off 1,300 employees, or 18% of its workforce, to be completed by the end of Q2 2023.

GitHub

Announced on March 28 that it has eliminated over 100 jobs in the South Asian market, laying off virtually its entire engineering team in India.
</item 83>

<item 84>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Synapse

Confirmed October 6 that it had laid off 86 people, about 40% of the a16z-backed fintech company. Earlier in the year, Synapse laid off 18% of its employees
after its growth slowed.

Bird

Issued a new round of layoffs after acquiring Spin, according to an email interim CEO Michael Washinushi sent to the company. The layoffs came one week after
the company was delisted from the New York Stock Exchange, and the exact number remains unknown.

Qualtrics

Announced on October 4 that 780 roles are being eliminated and that “several hundred” roles are changing or moving locations in 2024.

Hopper

Is reducing its full-time staff by 30%, amounting to 250 job cuts, the company announced October 3.

Chainalysis

Announced October 2 that it is laying off around 15% of its workforce, affecting approximately 150 employees.
</item 84>

<item 85>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Snap

Laid off fewer than 20 project managers November 8. CNBC reported that Snap’s vice president of engineering, ​​Nima Khajehnouri, will also be leaving.

Amazon

Started cutting workers in its music streaming division in Latin America, North America and Europe, the company confirmed November 8.

Google

Confirmed November 8 it eliminated “a small number of roles” in its consumer support staff, Google Users & Products.

Virgin Galactic

Said in a memo to staff November 8 it is cutting costs and reducing staff. The number of employees impacted is currently unknown.

Ubisoft Montreal

Confirmed November 7 it will cut 98 roles in business administration and IT. Ubisoft said 124 positions will be eliminated overall across Canada, including
reductions in its global IT team and its VFX studio, Hybride.

Nextdoor

Revealed in its Q3 earnings that it is reducing its workforce by 25%.

Beyond Meat

Announced cuts November 6 affecting around 19% of its non-production employees.
</item 85>

<item 86>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 86>

<item 87>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 87>

<item 88>
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
</item 88>

<item 89>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 89>

<item 90>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 90>

<item 91>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
100 Thieves

Is reducing its workforce by 20%, CEO John Robinson announced November 2. The esports brand is spinning out its game studio as well as its energy drink brand,
Juvee.

OpenSpace

Conducted a round of layoffs, CEO Jeevan Kalanithi announced November 2. The number of employees impacted by the layoffs has yet to be reported.

Informatica

Is laying off 10% of its global workforce, the cloud data management company announced in its Q3 results November 1.

Splunk

Is set to cut around 500 employees CEO Gary Steele said in a message to employees November 1 ahead of its planned acquisition by Cisco.

October 2023

Bungie

Is laying off about 100 employees, CEO Pete Parsons announced 10/30. The Sony-owned game studio is also delaying two of its upcoming titles as a result.

Karat

Reportedly cut an estimated 10% of its workforce October 27 after the financial startup for creators raised a $70 million Series B round earlier this year.
</item 91>

<item 92>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Well… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company
didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than,
say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.

But she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their
own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.

How can Epic possibly have a case against Google when Apple already won?

Epic declined to answer this on the record, among other questions we asked... but three things to consider:

1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.
</item 92>

<item 93>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The company is now facing an unfair labor practice complaint, with workers alleging the layoffs are retaliation for unionizing.

Chewy

Laid off more than 200 employees November 14. The cuts impact roles in HR, recruiting, data, business intelligence, plus directors and higher managers.

Amazon

Is cutting “just over 180” roles in its gaming division, VP Christoph Hartmann wrote in an email to employees November 13.

Carta

Is reportedly cutting more of its workforce. The number of positions affected are unknown, but it would be the unicorn startup’s third layoff round this year.

Pico

Is cutting a “few hundred” roles, a person familiar with the matter told TechCrunch November 9. ByteDance’s Oculus challenger held an internal meeting
announcing it is halting its expansion this week.

Cruise

Started laying off contingent workers who supported driverless operations November 9. Those layoffs came after the company recalled 950 robotaxis and lost key
commercial permits to operate in California.
</item 93>

<item 94>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 94>

<item 95>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
These days, Steam generally peaks at over 30 million, with valleys of 20 million. Fortnite’s valleys are closer to 3.8 million.]

Tim Sweeney says he plays Jellie in Fortnite. Image: Epic Games

Okay, let’s do a lightning round. What’s your favorite Fortnite skin?

I play as Jellie! Jellyfish. You know, the big tentacles? It’s just so cool!

Will Epic ever announce Unreal Tournament 3 X?

Forever in motion, the future is.

Why is Fortnite still not playable on Steam Deck?

If we only had a few more programmers. It’s the Linux problem. I love the Steam Deck hardware. Valve has done an amazing job there; I wish they would get to
tens of millions of users, at which point it would actually make sense to support it.
</item 95>

<item 96>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Clearco

Announced on January 17, impacting 30% of staff across all teams.

ShareChat

Announced on January 15, ShareChat laid off 20% of its workforce — or over 400 employees — just a month after eliminating more than 100 roles.

SmartNews

Announced on January 12 a 40% reduction of its U.S. and China workforce, or around 120 people.

Intrinsic

Alphabet’s robot software firm, Intrinsic, is laying off 40 employees TechCrunch confirmed on January 12. Amounting to around 20% of the headcount.

Greenlight

The fintech startup offering debit cards to kids laid off 104 employees on January 12, or over 21% of its total headcount of 485 employees.

Career Karma

Learning navigation platform Career Karma laid off another 22 people on January 12 across its global and domestic workforce.

DirectTV

Announced on January 12 plans to lay off about 10% of its management staff on January 20.
</item 96>

<item 97>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
showed that Epic knew it was breaking bad at the time it did the deed.

But Epic mentioned Project Liberty in its own opening statement — so, by that point, it had already been an hour since it admitted it intentionally broke
Google’s rules. “Epic decided to stand up because that’s what you do to a bully,” Bornstein told the jury.

“All we know is whatever is in the destroyed chats, as bad as the documents are, is worse.”

And it’s possible no examination will be able to take the stink off one of Google’s ugliest moves: the one where Google employees up to and including CEO Sundar
Pichai were caught setting sensitive chats to auto-delete to keep them out of a court’s hands. The court has already decided Google should be sanctioned in some
way for making potential evidence disappear, and Bornstein used it to plant persistent seeds of doubt in the minds of jury members. “All we know is whatever is
in the destroyed chats, as bad as the documents are, is worse.
</item 97>

<item 98>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Could Google actually lose?

When I walked into the courtroom on Monday morning, it seemed impossible. If Epic couldn’t prove Apple’s walled iOS garden is a monopoly, how could the
comparatively open Google do worse against the windmill-tilting Fortnite developer?

But now that both sides have made their opening arguments to a jury, I’m not quite as sure. Because while Google spent most of its first day attempting to
explain complicated ins and outs of business, Epic was able to paint a black-and-white picture of good and evil with itself as the clear underdog.

Epic lead attorney Gary Bornstein was tasked with making the case that Android functions as an unlawful monopoly. He did so by basically calling Google a bully
and a cheat that “bribes” or “blocks” any attempt to compete with Android’s Google Play store. The result? A status quo where the vast, vast majority of Android
app installs are from Google Play, with only a tiny sliver attributable to the Galaxy Store that comes preinstalled on every Samsung phone.
</item 98>

<item 99>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
No other teams are impacted.

Divvy Homes

Announced on September 11 that it is laying off 94 employees.

Sensor Tower

Announced on September 8 that it laid off approximately 40 people out of the 270+ at the company, including C-suite executives.

Roku

Announced September 6 that it plans to lay off approximately 10% of its workforce, representing over 300 employees.

August 2023

Malwarebytes

Announced on August 31 that it laid off 100 employees, mainly impacting corporate employees.

Twiga

Announced on August 21 that it is laying off 283 employees, or 33% of its workforce.

SecureWorks

Announced on August 14 that it will let go of 15% of its workforce, impacting about 300 employees. This is the company’s second round of layoffs this year.

CoinDesk

Announced on August 14 that it is cutting 16% of staff.
</item 99>

<item 100>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This investment alone means any time you buy a game built on Epic’s Unreal Engine, Tencent (and Sony) is getting a cut. Tencent is the biggest investor in
games, with thousands of tendrils across the industry — if you played something this year, Tencent was probably involved.

On a smaller scale, companies like Netflix and Devolver Digital have also dabbled in acquisitions recently. Devolver started buying studios in 2020, and it now
owns Croteam, Dodge Roll, Doinksoft, Firefly Studios, Nerial and System Era Softworks. Annapurna Interactive bought South African studio 24 Bit Games in
November. Netflix launched its Games division in 2021, and it’s already purchased four studios, including Oxenfree developer Night School and Alphabear company
Spry Fox.

Night School co-founder Sean Krankel told Engadget in June that the move to Netflix was a boon for the studio, providing financial security, a dedicated working
space and plenty of marketing support for its projects.
</item 100>
=== END ITEM id=4ca9b8b696a20540 ===

=== ITEM id=4b25fcbb0656a68c ===
Gold sentences and facts:
Sentence 0: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 0.0: The person quoted has the Eras Tour booked all through next year.
  Fact 0.1: The person quoted would rather die than cancel a show.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker said it was almost like a very strange, subtle clue to the media that they don't know everything that happened in that relationship.
  Fact 1.1: The speaker said she can have something really major and traumatic happen to her and they don't know about it.
  Fact 1.2: The speaker told NPR this.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Taylor Swift has shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:

=== END ITEM id=4b25fcbb0656a68c ===
