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

=== ITEM id=b0c1c070f59829e2 ===
Gold sentences and facts:
Sentence 0: In the above scenario, supply disruption will lead to higher inflation and central bankers will have to adjust their policies and hold rates higher for longer than anticipated.
  Fact 0.0: In the above scenario, supply disruption will lead to higher inflation.
  Fact 0.1: In the above scenario, central bankers will have to adjust their policies.
  Fact 0.2: In the above scenario, central bankers will have to hold rates higher for longer than anticipated.
Sentence 1: Back then, just like today, home prices had boomed for years before Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
  Fact 1.0: Back then, home prices had boomed for years before Fed officials were ultimately forced to hike interest rates aggressively.
  Fact 1.1: Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
  Fact 1.2: The home price boom back then was like the home price boom today.

Retrieved items:
<item 1>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
Trillions of dollars, euros, yen and pounds of asset purchases did little to raise prices in the face of global disinflationary forces until governments used
the money they raised to stuff cash into consumers’ pockets during Covid lockdowns.

But that’s also been blamed for distorting financial markets. Episodes such as the Silicon Valley Bank blow-up are seen by some as a direct result of central
bank reserves creation under QE, along with regulatory and supervision failures.

Only 40% of economists surveyed predict central banks will use QE the same way as they did before. A quarter expect them to deploy it more sparingly, about 30%
see its only role going forward as a tool to address financial-stability concerns and a small minority doesn’t see it being used again at all.

There are other problems with bond-buying that may affect how it’s used in the future. QE effectively swaps long-term borrowing costs for short-term ones.
</item 1>

<item 2>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
That’s reviving questions about whether central banks can deliver price stability all on their own.

“If we were designing optimal policy arrangements from scratch, monetary and fiscal policy would both have a role in managing the economic cycle and inflation,
and that there would be close coordination,” Philip Lowe said in his last speech as Australian central bank governor in September.

Economists surveyed by Bloomberg predict fiscal policy will somewhat counteract the Fed’s efforts to rein in inflation in the US.

“It’s true that there are circumstances where working hand in hand and supporting each other has proved helpful,” ECB President Christine Lagarde told a panel
discussion in June at the institution’s annual economic forum.

Fed Chair Jerome Powell, who sat to her right, signaled he wasn’t ready to rely on that kind of cooperation. “Our assignment is to deliver price stability kind
of regardless of the stance of fiscal policy.”

Central bankers warn that any failure to scale back fiscal spending risks coming at the cost of yet higher interest rates.
</item 2>

<item 3>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
The ECB faces an extra legal burden on bond holdings that comes with operating in a currency union of 20 countries. Concerns around illegally financing
governments and debt mutualization have already landed the central bank in court several times.

Mixing Policies

Low interest rates and large-scale QE programs allowed treasuries to borrow on the cheap to finance stimulus campaigns, protecting labor markets, businesses and
consumers from collapse. But the spending blowout throughout and since the pandemic — part critical emergency funding, part political need to show an
all-hands-on-deck approach in crisis — contributed to the latest outbreak in inflation.

While the same kind of pulling in the same direction is needed to restrain demand, many governments are concerned that if they tighten policy too hard, voters
will kick them out and replace them with populists or extremists. That’s reviving questions about whether central banks can deliver price stability all on their
own.
</item 3>

<item 4>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
She recommends they be ready to react preemptively, even when inflation hasn’t yet spun out of control.

They may be called into action soon on that front, should an escalation in the conflict in the Middle East hit oil deliveries.

When the next big global slowdown comes, though, flexibility may be needed the other way. Europe’s eight-year experiment with negative rates ended with mixed
reviews last summer as to whether it was all worth it.

The Bank for International Settlements argues that there’s room for greater tolerance for moderate shortfalls even if they’re persistent, because “low-inflation
regimes, in contrast to high-inflation ones, have self-stabilizing properties.”

Rethinking Quantitative Easing

With a more flexible approach to those 2% targets, monetary policy after the 2008 financial crisis would have looked very different in many parts of the world.
</item 4>

<item 5>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
After the most aggressive monetary-tightening campaign in four decades, academics and economic practitioners are running autopsies on what could have prevented
the cost-of-living crisis and how to ensure the same mistakes won’t be repeated.

Markets have scrambled to price in high-for-longer interest rates, with a new war in the Middle East adding yet more risk to an already uncertain outlook
confronting central bankers as they gather for their penultimate meetings of a tumultuous year.

The policy navel-gazing is centering around three debates. How much flexibility central banks can allow in reaching their inflation targets, the effectiveness
of asset purchases in the policy mix, and the merits of monetary and fiscal coordination.

Bloomberg surveyed economists from around the world to gather views on those three debates. Their verdict: Central banks won’t break their economies in a rush
to hit inflation targets, QE will be used more sparingly in the future, and fiscal policy risks countering the work of monetary authorities.
</item 5>

<item 6>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
It’s important to talk to the private sector, and to understand in every place and every industry what the main damage is and what they need in order to get
back to normal. It should be done professionally, preferably with no politicization of the recommendations."

What do you think the Bank of Israel should do in its next interest rate decision, on October 23?

"It’s still to early to tell how the war will affect the rate of inflation in the short and medium term. Just as there are new inflationary factors, such as the
depreciation of the shekel, there will also be factors working in the opposite direction, given the expected decline in private consumption, tourism, and other
items.

"As with fiscal policy, here too a country at war is allowed to deviate temporarily from the set inflation target. At the moment there is no surge in inflation
expectations, and the investment community is aware of the existence of index-linked assets and the exchange rate that can assist in hedging inflation risk.

"As for cutting the interest rate, it’s too early to consider that.
</item 6>

<item 7>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
What Bloomberg Economics Says… “A long period of galloping price gains, and fears that the last yards back to target could be most painful for workers, have
reignited the debate about whether central banks should aim for a higher rate of inflation. That’s a conversation worth having. But for monetary policymakers,
the imperative of retaining credibility means the right time for it is after inflation is back at target, not before.” — Tom Orlik, chief economist

Rethinking Targets

So long as people believe prices will get back toward 2%, central bankers have some leeway in deciding how aggressive they need to be in pursuing that goal.

Economists covering 16 of the world’s most important central banks say policymakers will allow more time to bring inflation back to target if it means less
damage to their economies. The Bloomberg special survey also shows that a sizable minority sees them going even further, accepting price pressures that are
either slightly too strong or too weak — as long as expectations remain anchored.
</item 7>

<item 8>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
They also want elected officials to put in place policies that help deliver sustainable growth.

“A change in mindset needs to happen,” said Agustin Carstens, the former governor of the Bank of Mexico who’s now the general manager of the BIS. “Growth needs
to depend less on fiscal and monetary policy, it should depend more on structural policies.”
</item 8>

<item 9>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
The rise in the consumer price index peaked at around 14% in 1980 before then–Fed Chair Paul Volcker’s hawkish policies sent mortgage rates to 18% in a year’s
time, cooling inflation, but sparking a recession. This caused a serious downturn in the housing market in which home sales and building levels cratered.
However, national home prices actually remained stable.

At the start of Volcker’s term as Fed chair in August 1979, the median U.S. home sales price was $64,700. And even after a near doubling of mortgage rates, that
figure rose to $69,400 by the second quarter of 1981.

Over the past 18 months, the Fed’s current chair, Jerome Powell, has been following a very similar game plan to Volcker’s, raising interest rates aggressively
to quash inflation.
</item 9>

<item 10>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
Olivier Blanchard, a former IMF chief economist, has long argued in favor of raising the inflation target, and former European Central Bank Vice President Vitor
Constancio has also embraced the idea. But it’s a controversial view and only possible from a position of credibility, which means central banks would likely
have to get inflation back to 2% first.

“It would be a mistake of the first order to think you can change a goal you have set if you can’t achieve it,” according to Bundesbank President Joachim Nagel.

Global trends suggest inflation will be stronger than in the past, with former Bank of England Governor Mark Carney among those saying rates won’t return to
pre-pandemic lows.

One lesson Gita Gopinath, the IMF’s No. 2 official, draws from the latest inflation episode is that policymakers mustn’t assume that looking through supply
shocks — as text books suggest — is the optimal response.
</item 10>
=== END ITEM id=b0c1c070f59829e2 ===

=== ITEM id=e0367b7f54bd89e4 ===
Gold sentences and facts:
Sentence 0: Beware of Terry McLaurin (vs. Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.
  Fact 0.0: Beware of Terry McLaurin this week, versus the Dolphins.
  Fact 0.1: Beware of Puka Nacua this week, versus the Browns.
  Fact 0.2: Beware of Tutu Atwell this week, versus the Browns.
  Fact 0.3: Beware of Drake London this week, playing at the Jets.
  Fact 0.4: Beware of Tyler Lockett this week, playing at the Cowboys.
  Fact 0.5: Beware of Jaxon Smith-Njigba this week, playing at the Cowboys.
Sentence 1: HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year.
  Fact 1.0: Harris had over 120 total yards in a Week 17 win in Baltimore last year.
  Fact 1.1: Harris had a touchdown in a Week 17 win in Baltimore last year.

Retrieved items:
<item 1>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
On the flip side, three of the four quarterbacks Chicago faced found three touchdowns on their limited throws.

Start Him (Lineup Decisions) Projections powered by Sportsline Terry McLaurin WR WAS Washington • #17

Age: 28 • Experience: 5 yrs. Weekly Breakdown MATCHUP vs CHI WAS -6 O/U 44.5 OPP VS WR 19th PROJ PTS 15.2 WR RNK 21st YTD Stats REC 21 TAR 26 REYDS 212 TD 1
FPTS/G 12.1 The Commanders may wish to win this one with a big dose of the run game (Brian Robinson is a must-start), but if the Bears put up any offense at
all, McLaurin will be in position to put up nice numbers. He's a low-end WR2 worth the nod over George Pickens, Zay Flowers, Drake London and Jahan Dotson.
</item 1>

<item 2>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Bowden Jr., alongside A.T. Perry and Keith Kirkwood, are the only healthy Saints WRs right now, and a favorable matchup against Detroit gives each some flex
value in deep leagues. Mingo also draws a plus matchup against a reeling Buccaneers secondary, as a potential negative game script should boost his Week 13
fantasy ceiling.

Tillman is another talented rookie who's getting more run as the season progresses, and although P.J. Walker's struggled under center, Tillman is set up for a
high-usage day against the Rams with Amari Cooper (ribs) banged up.

WEEK 13 DFS:

DK lineup | FD lineup | Best stacks | Best values

Who are the biggest potential WR busts in Week 13?

Beware of Terry McLaurin (vs. Dolphins), Puka Nacua and Tutu Atwell (vs.
</item 2>

<item 3>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Tack on the ascending secondary the Falcons seem to have and the Commanders' diffusion of targets and McLaurin is at best a low-end flex. I'd rather roll the
dice on Drake London, Tyler Lockett or Garrett Wilson.

LAST WEEK: Hopeful that his target edge over his teammates would keep growing, and even in a favorable game script where the Commanders were trailing by three
scores before halftime, McLaurin only saw five targets against the Bears' heavy zone coverage.

FILM: On the 10 third-down routes McLaurin ran last week, he drew extra coverage on four of them, got open on two of them (Howell threw elsewhere) and was a
decoy on Curtis Samuel's touchdown. What happened? Sam Howell was getting rid of the ball quickly and to shorter distances as the Bears were getting some
pressure on him. By the end of the game McLaurin's body language seemed dispirited.
</item 3>

<item 4>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He has more upside than Jerry Jeudy, Jahan Dotson and Tutu Atwell but I'd rather take a shot on Drake London, Terry McLaurin or Chris Godwin.

LAST WEEK: Over 19 snaps without Justin Jefferson last week, Osborn ran 18 routes and caught both of his two targets for 30 total yards. The receiving average
was delightful, but the small-sample target share (12.5%) was saddening. What's funny is that Osborn's target share was much higher when Jefferson was playing
(23.3%) but his receiving average tanked (6.3).

FILM: Osborn didn't beat coverage with his speed last week and his burst and agility were just OK. His 22-yard catch came on a Kirk Cousins scramble where he
waited in a spot and Cousins hit him, then Osborn ran inside instead of outside where he might have been able to score.
</item 4>

<item 5>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.

McLaurin's averaging just 42 receiving yards over his past three games, and a heavy dosage of Jalen Ramsey hinders his Week 13 fantasy value.

Nacua and Atwell see a Cleveland coverage unit who entered Week 12 allowing the fewest fantasy points per game to WRs. Heck, even Cooper Kupp, who hasn't looked
like himself of late, could underperform again.

We can't trust London and the rest of the Falcons passing offense against a well-rounded Jets defense. To make matters worse, he'll be matched up against Sauce
Gardner for most of the day, too.

Lockett and Smith-Njigba might be better suited for the bench in PPR leagues against a lockdown Dallas coverage unit.
</item 5>

<item 6>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PAST THREE WEEKS: McLaurin has led all Commanders wide receivers in targets and all Commanders players in receptions. He's the No. 1 guy, and his big Week 4
game (8-86-0 with an offensive fumble recovery for a touchdown) would have been even bigger if Sam Howell had better ball placement on a deep pass McLaurin was
open for.

BEARS: Are expected to play again without top outside cornerback Jaylon Johnson and safety Eddie Jackson. Not that those two absences led to a bonanza from
Broncos receivers; Russell Wilson spread the ball around. But if McLaurin remains the target leader then he'll have the best shot at a big game. Rookies Tyrique
Stevenson and Terell Smith figure to line up across from McLaurin the most.
</item 6>

<item 7>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Terry McLaurin WR WAS Washington • #17

Age: 28 • Experience: 5 yrs. Weekly Breakdown MATCHUP @ ATL ATL -2.5 O/U 42.5 OPP VS WR 6th PROJ PTS 12.1 WR RNK 37th YTD Stats REC 25 TAR 31 REYDS 261 TD 1
FPTS/G 11.4 The only two games McLaurin posted good numbers in were Washington's 35-33 win over Denver in Week 2 and Washington's 34-31 OT loss to Philadelphia
in Week 4. This matchup has an over/under of 42.5 points, tied for fourth-lowest of the week.
</item 7>

<item 8>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 8>

<item 9>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Bills

65 Tutu Atwell, Rams @ Ravens

66 Treylon Burks, Titans @ Dolphins

67 Josh Reynolds, Lions @ Bears

68 Nick Westbrook-Ikhine, Titans @ Dolphins

69 Cedric Tillman, Browns vs. Jaguars

70 Marquez Valdes-Scantling, Chiefs vs. Bills

71 Dontayvion Wicks, Packers @ Giants

72 Hunter Renfrow, Raiders vs. Vikings

73 Darnell Mooney, Bears vs. Lions

74 Skyy Moore, Chiefs vs. Bills

75 Kalif Raymond, Lions @ Bears

76 A.T. Perry, Saints vs. Panthers

77 Cedrick Wilson, Dolphins vs. Titans

78 Trenton Irwin, Bengals vs. Colts

79 Michael Gallup, Cowboys vs.
</item 9>

<item 10>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
By the end of the game McLaurin's body language seemed dispirited.

COMMANDERS: Throw to their wideouts on 58.7% of their targets, which is below league average (but not as bad as, say, the Falcons at 36.4%). What makes that
number worse is that McLaurin has just one more target (31) than Jahan Dotson (30), who has three more targets than Curtis Samuel (27). They're all nearly even
after five games and each of them has had a turn leading in targets at least once.

FALCONS: Are expected to play a heavy dose of zone coverage, if only because the Commanders have struggled to be explosive against zone coverage this year.
Atlanta also gave a full dose of snaps last week to outside cornerback Jeff Okudah, who wasn't flawless but does offer another sizable body to help against
stronger perimeter receivers. Between Okudah and A.J.
</item 10>
=== END ITEM id=e0367b7f54bd89e4 ===

=== ITEM id=4ddcb053c3b4157f ===
Gold sentences and facts:
Sentence 0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 0.0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley's sought-after voice on the promise and potential dangers of artificial intelligence.
  Fact 0.1: Altman had a sudden and mostly unexplained exit.
  Fact 0.2: Altman's exit brought uncertainty to the industry's future.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: This week's coverage includes Sam Altman backing a teen's AI startup.
  Fact 1.1: This week's coverage includes Google's hardware event.
  Fact 1.2: This week's coverage includes first impressions of the Pixel 8 Pro.
  Fact 1.3: This week's coverage includes Flexport drama.
  Fact 1.4: This week's coverage includes the ongoing FTX fallout.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory floating around right now is based on the board's language.
  Fact 2.1: The prevailing theory is based on the way these giant tech companies work.

Retrieved items:
<item 1>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
In a dramatic turn of events late Friday, ex-Y Combinator president Sam Altman was fired as CEO of AI startup OpenAI, the company behind viral AI hits like
ChatGPT, GPT-4 and DALL-E 3, by OpenAI’s board of directors. Then, the company’s longtime president and co-founder, Greg Brockman, resigned — as did three
senior OpenAI researchers. And the fallout continues.

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

It’s a fast-moving situation that we’re still trying to get to the bottom of. No doubt more will become clear as time goes on. To make it easier to follow all
that’s happened in the meantime, though, we’ve put together a timeline; we’ll do our best to keep it current.
</item 1>

<item 2>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 2>

<item 3>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 3>

<item 4>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He’s sat with multiple heads of state to discuss AI’s potential and perils. Just Thursday, he took part in a CEO summit at the Asia-Pacific Economic Cooperation
conference in San Francisco, where OpenAI is based.

He predicted AI will prove to be “the greatest leap forward of any of the big technological revolutions we’ve had so far.” He also acknowledged the need for
guardrails, calling attention to the existential dangers future AI could pose.

Some computer scientists have criticized that focus on far-off risks as distracting from the real-world limitations and harms of current AI products. The U.S.
Federal Trade Commission has launched an investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false
information through its chatbot.
</item 4>

<item 5>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman posted Friday on X: “i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved
working with such talented people. will have more to say about what’s next later.”

The Associated Press and OpenAI have a licensing and technology agreement allowing OpenAI access to part of the AP’s text archives.

Altman helped start OpenAI as a nonprofit research laboratory in 2015. But it was ChatGPT’s explosion into public consciousness that thrust Altman into the
spotlight as a face of generative AI — technology that can produce novel imagery, passages of text and other media. On a world tour this year, he was mobbed by
a crowd of adoring fans at an event in London.

He’s sat with multiple heads of state to discuss AI’s potential and perils.
</item 5>

<item 6>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Any significant snafu in that department would reflect on her, as well as Altman.

5. Perhaps a difference of AI ethics or philosophy?

Altman is a proud techno-optimist, and often speaks fondly of the possibilities of AGI, or artificial general intelligence, a theoretical software system that
achieves human-like intellect and versatility.

The board’s statement pointedly includes that “OpenAI was deliberately structured to advance our mission: to ensure that artificial general intelligence
benefits all humanity” and that new leadership was necessary. It’s possible that Sam’s zeal for AGI, even absent a secret project or agreement, led to a major
rift between him and the board.

It’s been obvious to all that Altman took the company in a much more corporate direction from its origins, changing its legal status and aggressively pursuing
enterprise and consumer applications. That doesn’t sound a lot like the “mission” the board wants to advance.
</item 6>

<item 7>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
Some CEOs get a pass on things like having kids with direct reports, but not all.

Altman also has three siblings, and his younger sister Annie has publicly accused him of abuse. We have no way of evaluating these allegations, which involve
private matters.

Our read on the board’s language in dismissing Altman, however, is that it was not a legal or personal problem that provoked the action, but a professional or
business one.

We probably won’t know the full truth on this for a long time, as the characters in the drama are likely to be NDA’ed up. Per various whispers and leaks, an
all-hands meeting about the situation this afternoon didn’t produce any revelations beyond banal reassurances that the company is fine and they’ll get a fresh
CEO soon. Expect to hear a lot of rumors before we hear the real thing.
</item 7>

<item 8>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.
</item 8>

<item 9>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
The Information, speaking to three sources formerly with the company, reports that they no longer expect the sale — led by Thrive Capital — to happen, or, if it
does, to come with a lesser valuation because of the recent turn of events.

Altman planning new venture

Altman has been telling investors that he’s planning to launch a new venture, according to The Information. Brockman is expected to join the effort — whatever
form it takes. (Possibly an AI chip startup.)

i love you all. today was a weird experience in many ways. but one unexpected one is that it has been sorta like reading your own eulogy while you’re still
alive. the outpouring of love is awesome. one takeaway: go tell your friends how great you think they are.
</item 9>

<item 10>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Altman has a long history steering OpenAI. After co-founding the company alongside Peter Thiel, LinkedIn cofounder Reid Hoffman and others, Altman, who formerly
headed startup accelerator Y Combinator and also holds a seat on the board at Worldcoin, the ambitious cryptocurrency project, initially served as a co-chair of
OpenAI alongside Elon Musk. Musk left in 2018 to avoid a conflict of interest with Tesla.

Altman over the past several months played an active role in attempting to shape regulators’ responses to AI, appearing at U.S. congressional hearings and
meeting personally with world leaders including President Joe Biden, German Chancellor Olaf Scholz and French President Emmanuel Macron, among others, on an
international policy tour.

More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
I've never met someone who has supported and lifted up more people around them than him. Couldn't be a prouder brother.
</item 10>
=== END ITEM id=4ddcb053c3b4157f ===

=== ITEM id=f420b08c308e2c3d ===
Gold sentences and facts:
Sentence 0: Unfortunately, she did say that she had conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine.
  Fact 0.0: The witness said she had conversations with investors as part of their due diligence.
  Fact 0.1: Alameda was taking on losses from FTX to keep FTX's balance sheet pristine.
Sentence 1: It was not, however, possible for those accounts to avoid liquidation, as Wang testified Alameda could do — or to have an overall negative balance.
  Fact 1.0: It was not possible for those accounts to avoid liquidation.
  Fact 1.1: Wang testified that Alameda could avoid liquidation for those accounts.
  Fact 1.2: It was not possible for those accounts to have an overall negative balance.
Sentence 2: It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy and sell cryptocurrencies.
  Fact 2.0: The argument described Alameda as a 'very innovative, successful company'.
  Fact 2.1: The argument's speaker claimed the business practices between Alameda and FTX were reasonable.
  Fact 2.2: The argument's speaker claimed Alameda acted legally as an FTX customer.
  Fact 2.3: The argument's speaker claimed Alameda acted legally as an FTX payment processor.
  Fact 2.4: The argument's speaker claimed Alameda acted legally as an FTX market maker.
  Fact 2.5: A market maker is a financial entity that acts as a trading partner for customers looking to buy and sell cryptocurrencies.

Retrieved items:
<item 1>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
FTX lied about how much money was in the backstop fund

In the process of liquidating, FTX would try to sell the collateral on the open market, but if it couldn’t finish, then backstop liquidity providers would step
in. These were market makers, including Alameda, which could be compensated for the losses they take from a “backstop fund,” Wang said.

But FTX lied about how much money was in the backstop fund, Wang said. In court we saw the code that generated the fake number published on the website: it took
the daily trading volume on FTX, multiplied it by a random number, divided it by a billion, and added it to the existing number displayed on the site. It had
nothing to do with the actual amount of money in the insurance fund.

And when there wasn’t enough money in the fund, money was moved there from Alameda’s accounts in order to pay the insurance out, Wang said.

In late 2019, Alameda had a negative balance on FTX.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Well, Bankman-Fried remembers it differently. See, Yedidia was just asking about Alameda’s risk profile, and Bankman-Fried wasn’t talking about insolvency at
all.

Also, when Singh and Bankman-Fried had the dramatic balcony conversation at their penthouse, it was just that Singh thought Alameda’s liabilities had gotten too
high, and FTX was spending too much money on marketing. But Bankman-Fried still thought that Alameda had more assets than liabilities, so it was all fine, and
besides, if Singh thought he was going to be better at marketing, he could take it over. It didn’t have anything to do with the money Alameda owed FTX at all.

Of course not! Bankman-Fried didn’t learn about the $8 billion liability associated with Alameda until October 2022, he said. And he learned it all by himself,
by looking at a computer database. When he found it, he was “very surprised!”
</item 4>

<item 5>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Then a bigger one. Then an effectively limitless one.

Ellison testified to something similar, choking back tears: “When I started working at Alameda, I don’t think I would have believed you if you told me I would
be sending false balance sheets to our lenders or taking customer money, but over time, it was something I became more comfortable with.”

Ellison’s tenure as CEO of Alameda Research was an exercise in optics, she testified. Alameda Research was, by then, FTX’s sin-eater. When FTX took a loss in
the hundreds of million dollars because a trader had exploited its system, Wang moved that loss onto Alameda’s balance sheet. FTX’s “insurance fund,” meant to
keep traders from getting stuck with other traders’ losses, was totally made up — Alameda simply picked up the tab.

Even that wasn’t good enough.
</item 5>

<item 6>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It was turned on in July. No other customer trading on FTX had “allow_negative” privileges — only Alameda. Wang’s testimony suggests that the fraud at FTX
started very early since the same day that allow_negative was switched on, Bankman-Fried tweeted that Alameda’s account “was like everyone else’s.”

The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating. For instance, Alameda Research first had a
negative balance in late 2019, Wang testified. Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money
from the account as long as the amount was less than FTX’s total trading revenue. A little later, Wang discovered that Alameda was negative by $50 million more
than FTX’s revenue. Bankman-Fried told him it was fine, and it just didn’t account for the FTT token that the two of them had launched. Wang trusted him.
</item 6>

<item 7>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Wang said he trusted Bankman-Fried’s judgment and didn’t pursue the issue further. Later, though, Alameda’s balance was more negative than FTT and the trading
revenue, and Wang had more conversations with Bankman-Fried about it. Wang said he knew at the time that Alameda was using FTX’s customers’ funds and that he
knew it was wrong.

We heard yesterday about the line of credit Alameda had on FTX, a staggering $65 billion. Today we saw it in the database. The line of credit didn’t start that
big — Wang testified that Bankman-Fried had asked him to increase it a few times because Alameda kept running into its limits. First it was “a few hundred
million,” then a billion, then even that wasn’t enough. It wasn’t clear how Wang arrived at $65 billion specifically, but he said he’d discussed the number with
Bankman-Fried. Other customers didn’t have the same privileges.
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 8>

<item 9>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 9>

<item 10>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Other customers didn’t have the same privileges.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses. In 2021, FTX was facing a “several hundred million dollar” loss from an exploit
in MobileCoin, Wang said. (This seems to have happened in April 2021, though Wang didn’t give a month.) Bankman-Fried directed him to have Alameda take it on
instead, because FTX’s balance sheets were visible to investors and Alameda’s weren’t.

Yesterday, Paradigm Capital’s Matt Huang showed us the balance sheets he received from Bankman-Fried. They didn’t include the loss, which should have been
included in the second quarter numbers he’d received — they showed $63 million in trading expenses and $63 million in “other expenses” in that quarter, and an
estimated profit of $732 million.

“FTX was not fine and assets were not fine.”

By 2022, Alameda’s negative balance on FTX was a cause for concern for Caroline Ellison, Singh, and Wang.
</item 10>
=== END ITEM id=f420b08c308e2c3d ===

=== ITEM id=373cdaaef8109ccc ===
Gold sentences and facts:
Sentence 0: “Users’ natural behavior for search is to follow the path of least resistance, i.e.
  Fact 0.0: Users' natural behavior for search is to follow the path of least resistance.
Sentence 1: “We make Google be the default search engine,” he said, “because we’ve always thought it was the best.
  Fact 1.0: A speaker said 'We make Google be the default search engine.'
  Fact 1.1: The speaker said the reason was 'because we've always thought it was the best.'

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 1>

<item 2>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
However, the startup wants subscriptions to be its primary revenue driver. Deft is currently testing two paid tiers: a $10 a month plan, which will offer better
search customization to filter out some brands, and a $20 per month plan, which will enable personal shopping service, that will continue to look for suitable
products even after you stop searching manually.

Up next, Deft aims to release a browser extension that helps users search for products by the end of this year or early next year. In the next three years, the
company wants to build and release search APIs that other e-commerce sites could use.

Challenges of building a search startup

Given that it’s building a search solution for e-commerce, Deft already faces challenges from Amazon and Google. In September, Amazon updated its visual search
feature that lets users snap a photo of a product and search for similar items on the platform. Google released its multi-search feature last year that lets
users fuse images and text.
</item 2>

<item 3>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
The startup said it is on track to hit 50,000 monthly active users by the end of this year. While co-founders didn’t reveal the exact search volume the product
is catering to, they said Deft is seeing “tens of thousands” of searches every week. They added that people using Deft for the second time do more searches than
their first visit.

The team also noted that it has a conversion rate that is five times higher — a measure of people buying the product versus seeing it — compared to the standard
e-commerce rate, which is around 2-3%, according to various estimates.

Deft doesn’t show ads on its platform. Rather, it relies on affiliate commissions to generate revenue. However, the startup wants subscriptions to be its
primary revenue driver.
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 5>

<item 6>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
I see Deft as creating a new kind of UX for purchasing,” he said.

Startups like Neeva — which was founded by former Google execs — tried to convince people to switch to paid and private search with additional features.
However, earlier this year, the startup shut down its consumer efforts and eventually got acquired by Snowflake.

Pienaar said that Deft has to give compelling reasons to users to switch to its service.

“Users’ natural behavior for search is to follow the path of least resistance, i.e. most people just pick up their phone and just use the default search engine.
For Deft to change that behavior, they will need to create a high-value proposition,” he said.

Amir Konigsberg, an investor and founder of e-commerce search startup Twiggle (which shut down), said his startup wanted to solve ChatGPT-type interactions in
the shopping world. He said that the main problem for startups in this space is the go-to-market strategy and fundraising.
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 7>

<item 8>
Deft is building e-commerce search to help you find the right product quickly (TechCrunch, 2023-11-03T13:00:14+00:00)
He said that the main problem for startups in this space is the go-to-market strategy and fundraising.

“The problem with competing with e-commerce players is not just about user experience, it’s also about logistics infrastructure and supply chain. So a startup
catering to consumer search has to invest a lot of money into making it a viable business,” Konigsberg told TechCrunch.

He also added that it will be difficult for a startup like Deft to build user loyalty while competing against Google and Amazon.

Both Pienaar and Konigsberg said that Deft will have to involve retailers in the mix either as partners or as customers of their search product. Either way,
it’s going to be interesting to see if Deft can build a user base on its own or sell its search feature to e-commerce retailers.
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“Before 2003,” Cue said, “the way that you searched the web was you had to go in and you had to type in google.com in the URL field, or you could type in
another URL. We came up with the idea that if you type anything in the URL field that’s not a URL, it just goes to search.”

Schmidtlein’s overall point was that Google helped Safari succeed not by forcing Apple’s hand, but by being a great product that integrated seamlessly with
Apple’s own stuff. He referenced Apple’s deals with Yahoo and Bing that make those services easy to find, and both men argued that switching search engines is
so easy as to be a non-issue. Bellshaw briefly stepped up to rebut that notion, and that was it for Cue’s testimony.

At least, that’s all the testimony we saw. Like so many things in this trial, the star witness was kept mostly under wraps thanks to complaints and worries
about revealing confidential numbers and corporate secrets.
</item 10>
=== END ITEM id=373cdaaef8109ccc ===
