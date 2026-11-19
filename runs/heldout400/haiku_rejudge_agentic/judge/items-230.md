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

=== ITEM id=502debb7d3acdf0c ===
Gold sentences and facts:
Sentence 0: In the third quarter, Uber generated $394 million in operating income and $219 million worth of net income.
  Fact 0.0: In the third quarter, Uber generated $394 million in operating income.
  Fact 0.1: In the third quarter, Uber generated $219 million worth of net income.
Sentence 1: The ride-hail company claims that the rate of sexual assault reported on the app decreased 38% between its first and second reports.
  Fact 1.0: The ride-hail company claims that the rate of sexual assault reported on the app decreased 38%.
  Fact 1.1: The 38% decrease was between the ride-hail company's first and second reports.

Retrieved items:
<item 1>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
After reporting, shares of Uber are up 1.6% in trading, after an up-and-down morning of shares exchange.

Against a backdrop of macroeconomic uncertainty, the former startup darling’s results can be viewed as an indication that its business model has matured and is
now in a stable, profitable and cash-generating position. On the other hand, under-performing demand results from a company as global as Uber could indicate
that consumer spend is coming in lighter than anticipated.

For startups in the transport and on-demand sectors, Uber’s earnings are a regular informational zeitgeist. So let’s work to understand where Uber’s revenue
came from in the third quarter, and how each of its top-line sources converted — or not — to bottom-line results.

Where did the money come from?

In the third quarter, Uber saw total bookings rise from $29.1 billion to $35.3 billion, a gain of around 21%.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, transport and on-demand sectors, macroeconomic uncertainty, consumer spend, Uber’s Q3 numbers include impressive profitability gains,
slower-than-expected growth, Uber
</item 1>

<item 2>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Of course, adjusted EBITDA is to profit as hidden heels in men’s shoes are to height, so we’ll want some harder figures as well. In the third quarter, Uber
generated $394 million in operating income and $219 million worth of net income. Mix in $966 million worth of positive operating cash flow, and Uber looks very
healthy, given that all of the figures in this paragraph were improvements on its year-ago results.

But while food delivery and scooting humans about town were profitable for Uber in the quarter, the final major portion of its business had a more lackluster
quarter.

Freight falls

One area that continues to drag on Uber is its freight business.

While Uber’s ride-hailing and delivery business saw an uptick in gross bookings in the third quarter, Uber Freight experienced a 27% drop year-over-year.
Revenue, as a result, had a similar fall.

The business unit reported revenue of $1.3 billion in the third quarter, a 27% drop from the same period last year.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber Freight, Uber
</item 2>

<item 3>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
In business segment terms, Uber generated $17.9 billion in ride-hailing bookings (+31% year over year), and $16.1 billion worth of delivery bookings (+18%).
Those key business groups at the company generated $5.1 billion and $2.9 billion in revenue, respectively, during the September quarter.

There’s nuance to the revenue figures that we need to consider as the rule-changes in question do affect startups that operate in related categories. On the
ride-hailing front, Uber told its investors that its Q3 2023 revenue was “negatively impacted by business model changes in some countries that classified
certain sales and marketing costs as contra revenue by $161 million.” That was not the only legal change that led to Uber’s results changing shape. Under its
delivery business results, the company added that its revenue result was “negatively impacted by business model changes that classified certain sales and
marketing costs as contra revenue by $360 million.”

The combined impact of those two items was 8 percentage points of growth.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, ride-hailing, delivery, business model changes,
contra revenue, Uber, startups
</item 3>

<item 4>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The combined impact of those two items was 8 percentage points of growth.

The profit perspective

From bookings to revenues to profits: How did the main portions of Uber’s business generate black ink?

Turning to what Uber calls “segment adjusted EBITDA,” it’s not hard to see how the company managed to rack up more income in its most recent quarter.
Ride-hailing adjusted profit rose to $1.29 billion, up 43% from $898 million in the year-ago quarter, while delivery saw its own profitability skyrocket from
$181 million in Q3 2022 to $413 million in its most recent fiscal period.

Uber did spend more during Q3 2023 when compared to its Q3 2022 result, but the 5% gain in “Corporate G&A and Platform R&D” costs to $595 million was far less
than the gains we saw above. So, Uber’s adjusted EBITDA rose from $516 million in aggregate to $1.09 billion in the third quarter of this year.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, ride-hailing, delivery, Corporate G&A and Platform
R&D, segment adjusted EBITDA, Uber
</item 4>

<item 5>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Uber hasn’t released safety data for 2021 and 2022, and won’t say when it plans to do so. Older data backs up some of Abrams’ claims.

Uber has published two safety reports — one containing data from 2017 to 2018 and another with data from 2019 to 2020. The ride-hail company claims that the
rate of sexual assault reported on the app decreased 38% between its first and second reports.

While the rate decrease is positive, not all the data delivers the same message. Over that same time period, the total number of sexual assault reports across
five categories fell from 5,981 to 3,824. That drop could be explained by the decrease of trips from 1.4 billion in 2019 to 650 million in 2020.

And while the total number of incidents as a percentage of total number of rides does decrease through the years, the number of incidents in certain categories
actually increases.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Abrams, Uber sexual assault survivors call for in-car cameras, tech upgrades, Uber safety reports, sexual assault
</item 5>

<item 6>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The question is whether Uber believes in the long-term income potential of freight.

So what?

When discussing Uber’s results internally, our vibe was that it is a very profitable and healthy company today, albeit one that is not growing as fast as the
market had hoped. The fact that Uber’s share price is up as we write this indicates that investors are content thus far to excuse the small Q3 misses and keep
their eyes more focused on its year-over-year improvements and forward guidance. More when we get Lyft’s numbers.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, freight, Uber, Lyft
</item 6>

<item 7>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, FactSet, Refinitiv, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber
</item 7>

<item 8>
Graph query row:
article: Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth
publisher: TechCrunch
passage: Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.
</item 8>

<item 9>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
On a quarter-over-quarter basis, Uber Freight had a 1% gain in revenue.

The results don’t get any better once we turn to net income. On an adjusted basis, Uber Freight lost $13 million in the third quarter compared to a $1 million
profit in the same quarter last year.

Uber said the dismal year-over-year revenue results for its freight business was driven by lower revenue per load and volume. Both are consequences of the
challenging freight market cycle.

Uber Freight isn’t alone. Other more recent entrants to the freight and logistics industry such as Flexport and Convoy have struggled this year. In Convoy’s
case, the business was forced to shut down with its assets gobbled up by Flexport.

Uber Freight continues to plug along, despite these economic headwinds. The question is whether Uber believes in the long-term income potential of freight.

So what?

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber Freight, challenging freight market cycle,
lower revenue per load and volume, Uber, Convoy, Flexport
</item 9>

<item 10>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
The number of incidents increased in the categories of nonconsensual kissing, nonconsensual touching and nonconsensual penetration, or rape, between 2017 and
2019.

The total number of rape incidents also rose from 2017 to 2019. And while the number of reported rapes dropped in 2020, the rate has remained the same since
2017. In other words, there has not been an improvement in the rate of rapes on the platform.

Sexual assaults are an ongoing problem for Uber

Uber has been sued numerous times over the past few years by passengers who claim they were sexually assaulted during a ride. Lyft has faced similar lawsuits
and accusations.

Under Uber’s terms of use, class action lawsuits can’t be filed against the company in cases of sexual assault, so each case has to be heard individually. This
has prevented survivors from advocating for themselves collectively.

Judge Charles Breyer in the Northern District of California will preside over pretrial hearings.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Uber sexual assault survivors call for in-car cameras, tech upgrades, Lyft, Charles Breyer, pretrial hearings, sexual assault, class
action lawsuits, Northern District of California, rape
</item 10>
=== END ITEM id=502debb7d3acdf0c ===

=== ITEM id=f5d93743213f0f7e ===
Gold sentences and facts:
Sentence 0: 10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late.
  Fact 0.0: At 10:59 p.m., the Cowboys are moving the ball.
  Fact 0.1: The Cowboys are moving the ball when it is a bit too late.
Sentence 1: Mixon has been forced to step up without Burrow under center, doubling as a runner and receiver for Cincinnati.
  Fact 1.0: Mixon has been forced to step up.
  Fact 1.1: Burrow is not under center.
  Fact 1.2: Mixon is forced to step up without Burrow under center.
  Fact 1.3: Mixon doubles as a runner for Cincinnati.
  Fact 1.4: Mixon doubles as a receiver for Cincinnati.

Retrieved items:
<item 1>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
D.J. Turner is expected to replace Awuzie.

GENO SMITH: Has a 47.8% completion rate and a lowly 4.1 yards per attempt average against man coverage so far this season. He's also barely thrown deep (4.3% of
his attempts went 20-plus Air Yards). A huge reason for the low numbers is because his pass protection has been dreadful. The anticipated return of left tackle
Charles Cross should help him have more time, which in turn helps his receivers.

OUTSIDE WRs: Have caught just 50% of their targets against the Bengals this season, third-lowest in football. However, when those receivers have caught the ball
they've averaged 17.9 yards per catch, which is the third-most in the league. The Bengals defense has really only let up big games to receivers who catch
touchdowns.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Bengals, CBSSports.com, D.J. Turner, Awuzie, NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game,
fantasy football, Geno Smith, Charles Cross
</item 1>

<item 2>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least 9 of those 15 throws went into the end zone.

PASS RUSH: Bengals opponents have started to sniff out Burrow's issues and have come after him. The Browns and Titans had a pass rush pressure rate between 38&
and 42% in their wins over Cincy with high blitz rates (at least 31% each). Compare that to the Rams and Ravens, who barely brought pressure or blitzes and
either lost or nearly lost to the Bengals.

CARDINALS: Are dead-last in blitz rate this season (14%) and third-to-worst in pass rush pressure rate (28%). They actually had nine sacks after Weeks 1 and 2
but have had three since against the Cowboys and 49ers. Arizona seems to prefer using just four rushers to affect the quarterback, something that should be
helpful to Burrow this week.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, pass rush, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Titans, Bengals, Blitz Rate, Pass
Rush Pressure Rate, Sacks, CBSSports.com, Cardinals, fantasy football, Ravens, Browns, Rams, Burrow
</item 2>

<item 3>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The Cowboys' offense looked very much like it did in last season's playoff exit -- uncomfortable and out of sorts. Dallas was sluggish through the first half of
the game, and the night unraveled quickly when Dak Prescott threw three interceptions in the second half. It got out of hand so quickly that Sam Darnold entered
the game for the 49ers with more than 12 minutes remaining.

For Prescott and the Cowboys, questions about whether they can get over the hump against the class of the NFC will persist. While their wins have been dominant,
Dallas already sits two games behind the Eagles in the NFC East.

The 49ers are rolling right along, with all of their major offensive weapons now healthy and the defense living up to its billing as well. San Francisco, now
5-0, doesn't face a team with a winning record until Week 10, though the Browns, Vikings and Bengals are far from automatic wins for Kyle Shanahan's team.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Vikings, NFC East, Bengals, NFC, Eagles, Sporting News, Sam Darnold, last season's playoff exit, Cowboys vs. 49ers game, Dak Prescott, Cowboys
vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game, Browns, Cowboys, Kyle Shanahan
</item 3>

<item 4>
Graph query row:
article: Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game
title: None
published_on: 2023-10-08T22:47:03+00:00
connected_entities: ['Brock Purdy', 'Fubo', 'DAZN', 'NFL', 'Vikings', 'NFC', 'Dallas Cowboys', 'NFC East', 'Bengals', 'Jerry Jones', 'Al Michaels', 'Mike
Tirico', 'Cris Collinsworth', 'Melissa Stark', 'Eagles', 'Deebo Samuel', 'Dre Greenlaw', 'Jim Harbaugh', 'Sunday Night Football', 'Sam Darnold', "last season's
playoff exit", 'Kevin Givens', 'Cooper Rush', 'Bryan Anger', 'Mitch Wishnowsky', "Cowboys vs. 49ers 'Sunday Night Football' game", 'Oren Burks', 'Prescott',
'Jordan Mason', 'Leighton Vander Esch', 'Jauan Jennings', 'Aiyuk', 'Brandon Aubrey', "Cowboys vs. 49ers NFL 'Sunday Night Football' game", 'Drake Jackson',
'Ferguson', 'McCaffrey', 'KaVontae Turpin', 'Cowboys vs. 49ers game', 'Purdy', 'Kyle Juszczyk', 'Jayron Kearse', 'Lamb', 'Dak Prescott', 'Nick Bosa', 'Johnathan
Hankins', 'Frank Gore', 'Vernon Davis', 'NFC championship games', "49ers' last golden era", 'Cowboys vs. 49ers', 'Tirico', 'Stark', 'Michele Tafoya', "Levi's
Stadium", 'Santa Clara', 'Calif.', 'Browns', 'NBC', 'NBCSports.com', 'Micah Parsons', 'Tony Pollard', 'San Francisco', 'playoffs', 'Pollard', 'CeeDee Lamb',
'Cowboys', 'Los Angeles', 'Kyle Shanahan', "NFL 'Sunday Night Football' game", 'Canada', 'Christian McCaffrey', 'Randy Gregory', '49ers', 'Fred Warner', 'San
Francisco 49ers', 'Brandon Aiyuk', 'Kittle', 'Tashaun Gipson', 'U.S.', 'Dallas', 'George Kittle', 'football', 'Peacock']
passages: ["Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\nJerry Jones said
earlier in the week that Sunday night's game would show him how the Cowboys stack up against the NFL's best.\n\nThe answer? Not very well.\n\nEverything that
could go wrong for Dallas on Sunday did go wrong, in a game that confirmed a 49ers team that ended the Cowboys' season in each of the last two years remains a
level above its NFC counterpart.\n\nThe 49ers carved up Dallas' defense on the opening drive, and that was a sign of what was to come. Brock Purdy threw four
touchdowns in the 42-10 rout, including three to George Kittle. Christian McCaffrey scored a touchdown for a 14th consecutive game as all of San Francisco's
offensive weapons got involved.\n\nMORE: Comparing stats for NFC stars Nick Bosa, Micah Parsons\n\nDefensively, the 49ers were relentless. Nobody had a larger
impact than Fred Warner, who finished the night with an interception, sack and forced fumble in a swarming performance.", "Cowboys vs. 49ers live score,
updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\nThe Cowboys' offense looked very much like it did in last
season's playoff exit -- uncomfortable and out of sorts. Dallas was sluggish through the first half of the game, and the night unraveled quickly when Dak
Prescott threw three interceptions in the second half. It got out of hand so quickly that Sam Darnold entered the game for the 49ers with more than 12 minutes
remaining.\n\nFor Prescott and the Cowboys, questions about whether they can get over the hump against the class of the NFC will persist. While their wins have
been dominant, Dallas already sits two games behind the Eagles in the NFC East.\n\nThe 49ers are rolling right along, with all of their major offensive weapons
now healthy and the defense living up to its billing as well. San Francisco, now 5-0, doesn't face a team with a winning record until Week 10, though the
Browns, Vikings and Bengals are far from automatic wins for Kyle Shanahan's team.", 'Cowboys vs. 49ers live score, updates, highlights from NFL \'Sunday Night
Football\' game (Sporting News, 2023-10-08T22:47:03+00:00)\nMORE: Why Brock Purdy remains one of the NFL\'s cheapest starting quarterbacks\n\nThe Sporting News
tracked live scoring updates and highlights from "Sunday Night Football" between the Cowboys and 49ers. Check out all of the key moments you might have
missed.\n\nCowboys vs. 49ers final score\n\n1 2 3 4 F Cowboys 0 7 3 0 10 49ers 7 14 7 14 42\n\nCowboys vs. 49ers results, highlights from \'Sunday Night
Football\'\n\n(All times Eastern)\n\nFinal: 49ers 42, Cowboys 10\n\n11:26 p.m. — That\'ll do it! The 49ers trounce the Cowboys and reassert themselves as one of
the NFL\'s most dominant teams. Through five games, San Francisco is unbeaten.\n\n11:24 p.m. — These teams are trading possessions as the clock winds down.',
"Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\n11:24 p.m. — These teams
are trading possessions as the clock winds down. There was a brief skirmish officials had to break up as frustrations mount for Dallas.\n\n11:14 p.m. — A sack
by Kevin Givens ends Rush's first drive of the night and sets up a Bryan Anger punt. Backups are in nearly all around for the 49ers.\n\n11:09 p.m. — Cooper Rush
is in the game for the Cowboys...\n\n11:07 p.m. — Mitch Wishnowsky with an excellent punt for the 49ers, who are just trying to get through the rest of the game
injury-free.\n\n11:04 p.m. — Sam Darnold has entered the game with 49ers. So, that's where we are with 12 minutes still on the clock.\n\n11:00 p.m. INTERCEPTION
— So much for that.", "Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\n11:00
p.m. INTERCEPTION — So much for that. Oren Burks gets the 49ers' third interception of the night as Prescott continues to struggle against this San Francisco
defense.\n\n10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late. Tony Pollard rips off a 26-yard run to finally break through.\n\n49ers
42, Cowboys 10\n\n10:53 p.m. TOUCHDOWN — And just like that, Jordan Mason runs in for another 49ers touchdown! This is ugly on all levels for the Cowboys, and
Leighton Vander Esch is now down as well. 42-10, 49ers.\n\n10:52 p.m. INTERCEPTION — Fred Warner now has a sack, forced fumble and an interception tonight. A
pass from Prescott is deflected into his hands, and the 49ers have another instant scoring opportunity.", "Cowboys vs. 49ers live score, updates, highlights
from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\n49ers 35, Cowboys 10\n\n10:47 p.m. TOUCHDOWN — Purdy hits Kyle Juszczyk for
the 49ers' fifth touchdown of the night. That's four touchdowns for Purdy alone, and San Francisco is pouring it all on the Cowboys.\n\nEnd of third quarter:
49ers 28, Cowboys 10\n\n10:43 p.m. — San Francisco is moving the ball inch-by-inch now, and the 'Niners are a yard away from making this a 25-point
game.\n\n10:38 p.m. — The 49ers are on the verge of entering the red zone after a 19-yard catch by Jauan Jennings. Purdy is up to 242 yards to go along with his
three touchdowns.\n\n10:32 p.m. INTERCEPTION — Prescott heaves it downfield under pressure and it's intercepted by Tashaun Gipson.", "Cowboys vs. 49ers live
score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\n— The Purdy-to-Aiyuk connection continues to get
stronger. Aiyuk goes for 23 yards to give the 49ers some breathing room after opening their drive with a false start.\n\n49ers 21, Cowboys 10\n\n10:17 p.m.
FIELD GOAL — The 49ers' defense tightens the screws and limits Dallas to a field goal. Brandon Aubrey drills it from 50 yards out, and it's down to a 21-10 lead
for San Francisco.\n\n10:13 p.m. — Prescott hits CeeDee Lamb for a big 29-yard gain. That quickly takes the Cowboys into 49ers territory to start the second
half.\n\nEnd of first half: 49ers 21, Cowboys 7\n\n9:58 p.m. — San Francisco's defense is all over the Cowboys yet again. It's Dre Greenlaw's turn to sack
Prescott, and Dallas is waving the white flag on the half.", "Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting
News, 2023-10-08T22:47:03+00:00)\nThey'll start the second half with the ball.\n\n9:56 p.m. — 49ers edge rusher Drake Jackson is walking off after requiring
some medical attention on the field.\n\n9:52 p.m. — The 49ers tried to put something together, but a third down pass to Aiyuk was too low. Dallas has a minute
to try and get something before halftime.\n\n9:45 p.m. — The Cowboys looked better on third down on their last drive, but Prescott's third down pass to Ferguson
is short of the line to gain and ends this drive quickly. The 49ers will have a couple minutes to blow this game open before halftime.\n\n49ers 21, Cowboys
7\n\n9:38 p.m. TOUCHDOWN — McCaffrey pushes through for the touchdown. That's his 14th consecutive game finding the end zone, and it puts the 'Niners back up
two scores.\n\n9:37 p.m.", "Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News,
2023-10-08T22:47:03+00:00)\n9:37 p.m. — Deebo Samuel takes some big hits but gets the 49ers down to the one-yard line. San Francisco is knocking on the door
again.\n\n9:35 p.m. — The Cowboys had done a pretty nice job of containing McCaffrey to this point, but the former Pro-Bowler takes a screen pass for 19 yards
to push the Niners to the edge of the red zone.\n\n9:31 p.m. — Aiyuk with a quick 15-yard catch to start the 49ers' drive. He's up to 35 yards on three
catches.\n\n49ers 14, Cowboys 7\n\n9:27 p.m. TOUCHDOWN — KaVontae Turpin flashes his speed, gets separation, and hauls in Prescott's pass in the end zone! A
perfectly drawn-up 26-yard play puts Dallas right back in this game.", "Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game
(Sporting News, 2023-10-08T22:47:03+00:00)\nEnd of first quarter: 49ers 7, Cowboys 0\n\n9:04 p.m. — Confirmed: 49ers ball. San Francisco has an instant
opportunity here.\n\n9:01 p.m. FUMBLE — Oh no! Pollard gets the ball knocked out deep in the Cowboys' own territory, and it looks like it was recovered by the
49ers. Officials are checking to make sure nobody was out of bounds, but that ball was very much out of Pollard's control. The 49ers could be set up in the red
zone.\n\n8:59 p.m. — The drive completely falls apart for the 49ers. A delay of game penalty sets San Francisco back, and Christian McCaffrey is dropped for a
five-yard loss on 3rd & 15. Not allowing even a field goal is a huge win for Dallas.\n\n8:56 p.m.", "Cowboys vs. 49ers live score, updates, highlights from NFL
'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\nNot allowing even a field goal is a huge win for Dallas.\n\n8:56 p.m. — The 49ers are
excelling on third down early in this game, and Purdy flips it to Kyle Juszczyk to keep the Niners' drive alive on 3rd & 1.\n\n8:54 p.m. — Now Kearse lines up
offsides... That extends the drive for the 49ers, who aren't moving as freely as they were on that opening drive.\n\n8:50 p.m. — Jayron Kearse comes through
with the Cowboys' first sack of the night, but Deebo Samuel makes up 11 of the 12 yards San Francisco needs on the next play.\n\n8:46 p.m. — There's the first
catch of the night for Lamb, but he doesn't get the first down and Dak Prescott is sacked by Nick Bosa to quickly end the drive. This 49ers defense has picked
up right where it left off against the Cowboys in January.", "Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting
News, 2023-10-08T22:47:03+00:00)\nThis 49ers defense has picked up right where it left off against the Cowboys in January.\n\n8:44 p.m. — Dallas' second drive
starts out with Tony Pollard getting dropped for a loss.\n\n8:40 p.m. — Big-time stop by the Cowboys' defense as Johnathan Hankins deflects Purdy's third down
pass with his helmet and forces a punt.\n\n8:36 p.m. — The 49ers' defense was all over Dallas on that drive. Three plays, two incompletions, and now the Cowboys
will punt it right back.\n\n49ers 7, Cowboys 0\n\n8:30 p.m. TOUCHDOWN — Purdy finds a wide open Kittle in the end zone for the score! That was a well-scripted
opening drive by Kyle Shanahan, and San Francisco has an early 7-0 lead.\n\n8:27 p.m.", "Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday
Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\n8:27 p.m. — A little bit of chaos as Deebo Samuel drops a ball that was pretty close to being a
fumble, but it's all a non-issue as Dallas is hit with an offsides penalty. The 49ers are right about in field goal range.\n\n8:25 p.m. — After a facemask
penalty and a nine-yard pass to Brandon Aiyuk, the 49ers are already over midfield.\n\n8:23 p.m. — Kickoff is a bit on the later end, but we are about to get
underway. The 49ers will start with the ball.\n\n8:10 p.m. — Jerry Jones said this week that this game will be a test of how the Cowboys match up with the NFL's
best, as if there wasn't already pressure on Dallas to show something after the 49ers knocked them out of the playoffs in back-to-back years.\n\n7:56 p.m. —
Frank Gore and Vernon Davis are in the house!", "Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News,
2023-10-08T22:47:03+00:00)\n7:56 p.m. — Frank Gore and Vernon Davis are in the house! A couple greats from the 49ers' last golden era under Jim
Harbaugh.\n\nFrank Gore and Vernon Davis in the building for SNF 🔥 pic.twitter.com/XMKgK4aeaF — OurSF49ers (@OurSf49ers) October 8, 2023\n\n7:36 p.m. — CeeDee
Lamb is coming off back-to-back games with only four catches. Is he due for a huge performance tonight?\n\n7:11 p.m. — New 49ers acquisition Randy Gregory is
officially inactive against his former team.\n\n7:01 p.m. — The Eagles are going to win in Los Angeles, so the 49ers need a win to keep pace atop the NFC. After
they had to play the last two NFC championship games on the road, the No. 1 seed is undoubtedly in the back of their minds.\n\n6:45 p.m.", 'Cowboys vs. 49ers
live score, updates, highlights from NFL \'Sunday Night Football\' game (Sporting News, 2023-10-08T22:47:03+00:00)\n1 seed is undoubtedly in the back of their
minds.\n\n6:45 p.m. — The run of great prime-time weather continues, as it\'s set to be in the high 70s and sunny in Santa Clara tonight. No excuses for these
powerful offenses.\n\nWhat channel is Cowboys vs. 49ers on today?\n\nGame: Cowboys vs. 49ers\n\nCowboys vs. 49ers Date: Sunday, Oct. 8\n\nSunday, Oct. 8 TV
channel: NBC\n\nNBC Live stream: Peacock | NBCSports.com | Fubo (U.S.) | DAZN (Canada)\n\nCowboys vs, 49ers can be watched on NBC, which has long held the
broadcast rights to "Sunday Night Football."\n\nMike Tirico (play-by-play) and Cris Collinsworth (color analyst) will be on the call from Levi\'s Stadium while
Melissa Stark will serve as the sideline reporter.', "Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News,
2023-10-08T22:47:03+00:00)\nThe crew is entering their second season together after Tirico replaced play-by-play legend Al Michaels while Stark took over for
Michele Tafoya.\n\nFans can also stream the game via Peacock or NBCSports.com with a subscription, along with Fubo, which offers a free trial.\n\nViewers in
Canada can stream every NFL game via DAZN.\n\nCowboys vs. 49ers start time\n\nDate: Sunday, Oct. 8\n\nSunday, Oct. 8 Start time: 8:20 p.m. ET (5:20 p.m.
CT)\n\nSunday night's game between the Cowboys and 49ers is set to kick off at 8:20 p.m. ET (5:20 p.m. local time) from Levi's Stadium in Santa Clara,
Calif.\n\nThe 49ers and Cowboys have a long history with one another, but San Francisco has gotten the better of Dallas in each of their last two meetings.",
"Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)\nThe 49ers knocked the
Cowboys out of the playoffs in both of the last two seasons, so a win for Dallas on Sunday could be helpful just from a psychological standpoint."]
</item 4>

<item 5>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But just three players have averaged over 4.0 yards per carry: Joe Mixon (4.5 yards per run, didn't have 100 total yards, didn't score), Zack Moss (4.1 yards
per run, ran for 122 yards on 30 carries and caught a touchdown) and Pierre Strong (9.8 yards per carry, most of it coming in garbage time last week).

HISTORY: Harris had over 120 total yards and a touchdown in a Week 17 win in Baltimore last year. He also scored on them with exactly 50 total yards against
them in at home in Week 14. He has posted at least 10 non-PPR points in 3 of 4 career games.

Graph facts:
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Harris
- Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant -MENTIONS-> Harris
- UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl -MENTIONS-> Harris
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> NFL Fantasy Football
- Harris -PARTICIPATED_IN-> Harris’s remarks

Entities: NFL Fantasy Football, Pierre Strong, Harris, Week 17 win, Week 14 game, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to
know for every game, CBSSports.com, Zack Moss, Joe Mixon, Baltimore
</item 5>

<item 6>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.

Graph facts:
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Joshua Dobbs
- Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Joshua Dobbs
- 2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR -MENTIONS-> Joshua Dobbs
- Joshua Dobbs -WORKS_FOR-> Vikings
- Joshua Dobbs -PARTICIPATED_IN-> Vikings vs. Broncos game

Entities: Joshua Dobbs, NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game, Niners, Week 2, CBSSports.com,
fantasy football, Zach Wilson, Purdy, Daniel Jones, Cowboys, Giants, Patriots, Brandon Aiyuk
</item 6>

<item 7>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
RB Joe Mixon, Bengals vs. Colts ($7,200). Mixon has been forced to step up without Burrow under center, doubling as a runner and receiver for Cincinnati. Jake
Browning has relied heavily upon the veteran RB, evidenced by Mixon's two touchdowns on Monday night against the Jaguars. Burrow's injury has made Mixon
arguably the Bengals' top fantasy option now. We'll gladly insert the RB into the lineup in a good matchup against a Indianapolis team that's had some issues
defending RBs, notably getting torched by Derrick Henry and Tyjae Spears one week ago.

WEEK 14 DFS:

DK lineup | Best values | Best stacks

RB Breece Hall, Jets vs. Texans ($6,400). It's Week 14 and there is a Jets player in this DFS lineup. How bad can it get? The good news is that it can always
get worse, and Hall's role in this offense makes him a worthy dart throw in this lineup.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Jaguars
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Jaguars
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jaguars
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jaguars
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Jaguars

Entities: Jaguars, Bengals, Cincinnati, Texans, fantasy football, NFL, Jets, Breece Hall, Sporting News, daily fantasy football, FanDuel Picks Week 14: Best NFL
DFS lineup advice for daily fantasy football tournaments, Burrow, Tyjae Spears, Joe Mixon, Derrick Henry
</item 7>

<item 8>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
For a Cowboys team that has embarrassed a few opponents this year, they might be on the bad end of an ugly game here.

49ers 28, Cowboys 10

10:27 p.m. TOUCHDOWN — Not one, not two, THREE touchdowns for George Kittle. The 49ers have everything working offensively tonight, and any Kittle fantasy
owners are thrilled.

10:25 p.m. — Aiyuk's loss is Deebo Samuel's gain. Samuel goes 43 yards on the very next play, which tells you how automatic the 49ers' offense is right now. San
Francisco is in the red zone looking to make this a three-score game.

10:23 p.m. — An offensive holding call wipes out a 40-yard gain from Aiyuk. Wow. That's a momentum swing. Spencer Burford is the culprit.

10:21 p.m. — The Purdy-to-Aiyuk connection continues to get stronger.
</item 8>

<item 9>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Jerry Jones said earlier in the week that Sunday night's game would show him how the Cowboys stack up against the NFL's best.

The answer? Not very well.

Everything that could go wrong for Dallas on Sunday did go wrong, in a game that confirmed a 49ers team that ended the Cowboys' season in each of the last two
years remains a level above its NFC counterpart.

The 49ers carved up Dallas' defense on the opening drive, and that was a sign of what was to come. Brock Purdy threw four touchdowns in the 42-10 rout,
including three to George Kittle. Christian McCaffrey scored a touchdown for a 14th consecutive game as all of San Francisco's offensive weapons got involved.

MORE: Comparing stats for NFC stars Nick Bosa, Micah Parsons

Defensively, the 49ers were relentless. Nobody had a larger impact than Fred Warner, who finished the night with an interception, sack and forced fumble in a
swarming performance.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Brock Purdy
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Brock Purdy
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Brock Purdy
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Brock Purdy

Entities: Brock Purdy, NFL, Jerry Jones, NFC, Sporting News, Cowboys vs. 49ers game, Nick Bosa, Cowboys vs. 49ers live score, updates, highlights from NFL
'Sunday Night Football' game, Micah Parsons, Cowboys, Christian McCaffrey, 49ers, Fred Warner, George Kittle
</item 9>

<item 10>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
: Mixon gets five yards on a run outside to the right, he's pushed out of bounds, but that will do it. The Bengals can kneel the game out and will win 24-18.

11:19 p.m.: Mixon rushes for four yards to bring up a third-and-4 at the Bills' 37, and bring up the two-minute warning. A first down will win the game for the
Bengals, since Buffalo cannot stop the clock. A stop on third will bring up an interesting decision for Zac Taylor and Cincinnati.

11:17 p.m.: Mixon rushes for two yards after the big gain, and the Bills use their final timeout with 2:37 left.

11:17 p.m.: Feels like a time to run the ball, right? Well, not to the Bengals, who trust Burrow to air out a deep pass to Boyd for 32 yards to the Bills 43.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Bills, Bengals, Sporting News, Burrow, Mixon, Zac Taylor, Boyd, Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night
Football' game, NFL 'Sunday Night Football' game
</item 10>
=== END ITEM id=f5d93743213f0f7e ===
