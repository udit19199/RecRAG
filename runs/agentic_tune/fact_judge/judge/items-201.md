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

This file has 6 items.

=== ITEM id=c3f1a6403c3ef1e1 ===
Gold sentences and facts:
Sentence 0: Instead, Marshall will start freshman Cole Pennington, who is the son of former Marshall star QB Chad Pennington.
  Fact 0.0: Marshall will start freshman Cole Pennington instead.
  Fact 0.1: Cole Pennington is a freshman.
  Fact 0.2: Cole Pennington is the son of former Marshall star QB Chad Pennington.
Sentence 1: Geno Smith's night appears to be done as Drew Lock is back in for the Seahawks.
  Fact 1.0: Geno Smith's night appears to be done.
  Fact 1.1: Drew Lock is back in for the Seahawks.

Retrieved items:
<item 1>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Broncos

12 Russell Wilson, Broncos @ Chargers

13 Justin Fields, Bears vs. Lions

14 Gardner Minshew, Colts @ Bengals

15 Matthew Stafford, Rams @ Ravens

16 Trevor Lawrence, Jaguars @ Browns

17 Geno Smith, Seahawks @ 49ers

18 Derek Carr, Saints vs. Panthers

19 Joe Flacco, Browns @ Jaguars

20 Baker Mayfield, Buccaneers @ Falcons

21 Josh Dobbs, Vikings @ Raiders

22 Jake Browning, Bengals vs. Colts

23 Bryce Young, Panthers @ Saints

24 Aidan O'Connell, Raiders vs. Vikings

25 Desmond Ridder, Falcons vs. Buccaneers

26 Will Levis, Titans @ Dolphins

27 Bailey Zappe, Patriots @ Steelers

28 Mitch Trubisky, Steelers @ Patriots

29 Tommy DeVito, Giants vs. Packers

30 Zach Wilson, Jets vs. Texans
</item 1>

<item 2>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
9:30 p.m.: At the two-minute warning, Drew Lock has gotten the Seahawks to their own 48-yard line. Lock will continue to lead the Seahawks' two-minute drill.

Geno Smith is out of the medical tent and headed to the locker room for further testing on his knee. His status for the remainder of the game is questionable.

The Seahawks will look to add some cushion to their lead as they will also receive the ball to open the second half.

9:26 p.m.: Geno Smith's displeasure with the tackle from earlier is beginning to make sense. Seattle's QB has been in the medical tent with an undetermined
injury.

Drew Lock will be taking snaps until there is more clarity on Smith's status moving forward.

9:23 p.m.: FIELD GOAL! Graham Gano connects on a 55-yarder to get the Giants on the board.
</item 2>

<item 3>
Monday Night Football DraftKings Picks: NFL DFS lineup advice for Week 15 Eagles-Seahawks Showdown tournaments (Sporting News, 2023-12-18T22:47:00+00:00)
The receiver has been prone to being undisciplined on the field, however, resulting in an ejection against the 49ers last week. That is always a concern when
trusting Metcalf, but we take the good with the bad since the ceiling is so high. No matter who ends up starting at QB, the WR is in line for another dominant
showing against one of the league's most generous secondaries.

FLEX QB Geno Smith, Seahawks ($9,400)

Metcalf won't be throwing the ball to himself and he's not one to get opportunities in the running game, meaning that someone will be setting him up for
success. And since Smith appears ready to play, he finds himself with a spot in our lineup. In the event that he doesn't play, however, we will turn to Drew
Lock ($9,200) in his place. Regardless, the Seattle quarterbacks are set up for success and should be able to light up the skies on Monday night.
</item 3>

<item 4>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
New York has been dominated at home twice this season now, as this performance follows a 40-0 blowout in Week 1 on "Sunday Night Football." Daniel Jones' career
mark in primetime falls to 1-13 and 0-3 this season.

11:09 p.m.: TURNOVER ON DOWNS! Daniel Jones is sacked for an 11th time as Seattle takes back over at the New York 35-yard line.

Geno Smith's night appears to be done as Drew Lock is back in for the Seahawks.

11:06 p.m: Daniel Jones is back under center for New York, which will look to drive and score to at least lessen the damage on the scoreboard.

11:04 p.m.: FIELD GOAL! The turnover leads to three points for the Seahawks, who get a 34-yard field goal from Jason Myers. It's a three-touchdown lead for
Seattle now with just over five minutes to play.
</item 4>

<item 5>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
As the Seahawks get the ball back, they run the clock out as boos begin to rain down on the Giants from their home crowd.

Making matters worse, Seattle is set to receive the second-half kickoff.

9:37 p.m: TOUCHDOWN! RB Kenneth Walker III punches it in from one yard out. It's his second touchdown of the 2023 season.

The play doesn't come without extracurriculars, however, as it is followed by a scrum between linemen. The result is offsetting penalties from Seahawks G
Anthony Bradford and Giants DE Leonard Williams.

Seahawks 14, Giants 3.

9:35 p.m.: Drew Lock is balling.

Lock connects with Noah Fant for a 51-yard completion that has the Seahawks on the 1-yard line. Credit to Fant, as he used his footwork to toe the sideline and
turn the completion to a big play.

9:30 p.m.
</item 5>

<item 6>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
Because we splurged for McCaffrey and Kamara, we had to sacrifice salary elsewhere, and London is a high-upside option even in an offense that doesn't throw the
ball with efficiency.

WR Jaxon Smith-Njigba, Seahawks @ 49ers ($4,100). Is a takeover underway in the Seahawks offense? Smith-Njigba is coming off a game against the Cowboys that saw
him collect seven-of-11 targets for 62 yards. It's a step up from a disappointing outing against San Francisco, but Tyler Lockett's involvement seems to be
lessening as of late with the rookie stepping up. It wasn't a great day for the Seattle offense on Thanksgiving against the Niners, but the Seahawks should put
up a better effort against a divisional opponent they know so well. If they're trailing, JSN should see action with a negative game script.
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Week 11 of the 2023 NFL season has provided plenty of drama, from the Bears hanging with the Lions to the Giants getting a rebound from emergency quarterback
Tommy DeVito. But the latest round of regular-season action has also brought another wave of unfortunate injuries. Here's a roundup of the latest big names to
go down around the league:

QB

C.J. Stroud (Texans) left for a single play against the Cardinals after absorbing a hard hit, with backup Davis Mills entering in relief. Stroud did not receive
an injury designation, however, and returned after a quick check-in from trainers.

Geno Smith (Seahawks) suffered an injury to his throwing arm after guiding a 16-7 lead over the Rams, showing frustration while trying to stretch the arm after
a hit from Aaron Donald. Seattle quickly ruled him questionable to return with an elbow issue, and backup Drew Lock took over under center with 18 seconds left
in the third quarter.
</item 7>

<item 8>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 8>

<item 9>
Graph query row:
article: None
article_name: UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl
published_on: 2023-12-18T22:29:26+00:00
publisher: Sporting News
mentioned_entity: Marshall Thundering Herd
passage: UTSA vs. Marshall odds, props, predictions: Roadrunners lay big spread in Frisco Bowl (Sporting News, 2023-12-18T22:29:26+00:00)
UTSA is 5-6-1 ATS and 4-7-1 to the OVER.

UTSA Roadrunners opting out of Scooter's Coffee Frisco Bowl

The Roadrunners will be without their best defensive player, Trey Moore, as he looks to jump to a Power 5 school. Moore led the conference in sacks, so he's a
star-level absence for the unit.

Here are USTA players who won't play in the Frisco Bowl.

Trey Moore, EDGE (transfer portal)

Syrus Dumas, DB (transfer portal)

Xavier Spencer, CB (transfer portal)

Kyle Eaves, WR (transfer portal)

Marshall Thundering Herd betting news

The toughest blow for Marshall in this game is being without second-year starting QB Cam Fancher (transfer portal). Instead, Marshall will start freshman Cole
Pennington, who is the son of former Marshall star QB Chad Pennington.
</item 9>

<item 10>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.
</item 10>
=== END ITEM id=c3f1a6403c3ef1e1 ===

=== ITEM id=bf953ef9c4e54db3 ===
Gold sentences and facts:
Sentence 0: In the third quarter, Uber generated $394 million in operating income and $219 million worth of net income.
  Fact 0.0: In the third quarter, Uber generated $394 million in operating income.
  Fact 0.1: In the third quarter, Uber generated $219 million worth of net income.
Sentence 1: The ride-hail company claims that the rate of sexual assault reported on the app decreased 38% between its first and second reports.
  Fact 1.0: The ride-hail company claims that the rate of sexual assault reported on its app decreased 38%.
  Fact 1.1: The 38% decrease is measured between the company's first and second reports.

Retrieved items:
<item 1>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
The combined impact of those two items was 8 percentage points of growth.

The profit perspective

From bookings to revenues to profits: How did the main portions of Uber’s business generate black ink?

Turning to what Uber calls “segment adjusted EBITDA,” it’s not hard to see how the company managed to rack up more income in its most recent quarter.
Ride-hailing adjusted profit rose to $1.29 billion, up 43% from $898 million in the year-ago quarter, while delivery saw its own profitability skyrocket from
$181 million in Q3 2022 to $413 million in its most recent fiscal period.

Uber did spend more during Q3 2023 when compared to its Q3 2022 result, but the 5% gain in “Corporate G&A and Platform R&D” costs to $595 million was far less
than the gains we saw above. So, Uber’s adjusted EBITDA rose from $516 million in aggregate to $1.09 billion in the third quarter of this year.
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
</item 2>

<item 3>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
After reporting, shares of Uber are up 1.6% in trading, after an up-and-down morning of shares exchange.

Against a backdrop of macroeconomic uncertainty, the former startup darling’s results can be viewed as an indication that its business model has matured and is
now in a stable, profitable and cash-generating position. On the other hand, under-performing demand results from a company as global as Uber could indicate
that consumer spend is coming in lighter than anticipated.

For startups in the transport and on-demand sectors, Uber’s earnings are a regular informational zeitgeist. So let’s work to understand where Uber’s revenue
came from in the third quarter, and how each of its top-line sources converted — or not — to bottom-line results.

Where did the money come from?

In the third quarter, Uber saw total bookings rise from $29.1 billion to $35.3 billion, a gain of around 21%.
</item 3>

<item 4>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.
</item 4>

<item 5>
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
</item 5>

<item 6>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
In business segment terms, Uber generated $17.9 billion in ride-hailing bookings (+31% year over year), and $16.1 billion worth of delivery bookings (+18%).
Those key business groups at the company generated $5.1 billion and $2.9 billion in revenue, respectively, during the September quarter.

There’s nuance to the revenue figures that we need to consider as the rule-changes in question do affect startups that operate in related categories. On the
ride-hailing front, Uber told its investors that its Q3 2023 revenue was “negatively impacted by business model changes in some countries that classified
certain sales and marketing costs as contra revenue by $161 million.” That was not the only legal change that led to Uber’s results changing shape. Under its
delivery business results, the company added that its revenue result was “negatively impacted by business model changes that classified certain sales and
marketing costs as contra revenue by $360 million.”

The combined impact of those two items was 8 percentage points of growth.
</item 6>

<item 7>
Graph query row:
article: Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth
passage_index: 4574
passage: Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
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
</item 7>

<item 8>
Graph query row:
article: Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth
passage_index: 4570
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
The question is whether Uber believes in the long-term income potential of freight.

So what?

When discussing Uber’s results internally, our vibe was that it is a very profitable and healthy company today, albeit one that is not growing as fast as the
market had hoped. The fact that Uber’s share price is up as we write this indicates that investors are content thus far to excuse the small Q3 misses and keep
their eyes more focused on its year-over-year improvements and forward guidance. More when we get Lyft’s numbers.
</item 9>

<item 10>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Uber hasn’t released safety data for 2021 and 2022, and won’t say when it plans to do so. Older data backs up some of Abrams’ claims.

Uber has published two safety reports — one containing data from 2017 to 2018 and another with data from 2019 to 2020. The ride-hail company claims that the
rate of sexual assault reported on the app decreased 38% between its first and second reports.

While the rate decrease is positive, not all the data delivers the same message. Over that same time period, the total number of sexual assault reports across
five categories fell from 5,981 to 3,824. That drop could be explained by the decrease of trips from 1.4 billion in 2019 to 650 million in 2020.

And while the total number of incidents as a percentage of total number of rides does decrease through the years, the number of incidents in certain categories
actually increases.
</item 10>
=== END ITEM id=bf953ef9c4e54db3 ===

=== ITEM id=1e04ec1c9e111a17 ===
Gold sentences and facts:
Sentence 0: “It was the millionaires of the Premier League against the youngsters of the Championship,” reflected the former Denmark forward.
  Fact 0.0: The former Denmark forward reflected on the match.
  Fact 0.1: The former Denmark forward said it was the millionaires of the Premier League against the youngsters of the Championship.
Sentence 1: The Commanders are coming off a dud last week against the Giants, while Dallas is flying high.
  Fact 1.0: The Commanders are coming off a dud last week against the Giants.
  Fact 1.1: Dallas is flying high.

Retrieved items:
<item 1>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
1 seed is undoubtedly in the back of their minds.

6:45 p.m. — The run of great prime-time weather continues, as it's set to be in the high 70s and sunny in Santa Clara tonight. No excuses for these powerful
offenses.

What channel is Cowboys vs. 49ers on today?

Game: Cowboys vs. 49ers

Cowboys vs. 49ers Date: Sunday, Oct. 8

Sunday, Oct. 8 TV channel: NBC

NBC Live stream: Peacock | NBCSports.com | Fubo (U.S.) | DAZN (Canada)

Cowboys vs, 49ers can be watched on NBC, which has long held the broadcast rights to "Sunday Night Football."

Mike Tirico (play-by-play) and Cris Collinsworth (color analyst) will be on the call from Levi's Stadium while Melissa Stark will serve as the sideline
reporter.
</item 1>

<item 2>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
MORE: Why Brock Purdy remains one of the NFL's cheapest starting quarterbacks

The Sporting News tracked live scoring updates and highlights from "Sunday Night Football" between the Cowboys and 49ers. Check out all of the key moments you
might have missed.

Cowboys vs. 49ers final score

1 2 3 4 F Cowboys 0 7 3 0 10 49ers 7 14 7 14 42

Cowboys vs. 49ers results, highlights from 'Sunday Night Football'

(All times Eastern)

Final: 49ers 42, Cowboys 10

11:26 p.m. — That'll do it! The 49ers trounce the Cowboys and reassert themselves as one of the NFL's most dominant teams. Through five games, San Francisco is
unbeaten.

11:24 p.m. — These teams are trading possessions as the clock winds down.
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
</item 3>

<item 4>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The 49ers knocked the Cowboys out of the playoffs in both of the last two seasons, so a win for Dallas on Sunday could be helpful just from a psychological
standpoint.
</item 4>

<item 5>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The crew is entering their second season together after Tirico replaced play-by-play legend Al Michaels while Stark took over for Michele Tafoya.

Fans can also stream the game via Peacock or NBCSports.com with a subscription, along with Fubo, which offers a free trial.

Viewers in Canada can stream every NFL game via DAZN.

Cowboys vs. 49ers start time

Date: Sunday, Oct. 8

Sunday, Oct. 8 Start time: 8:20 p.m. ET (5:20 p.m. CT)

Sunday night's game between the Cowboys and 49ers is set to kick off at 8:20 p.m. ET (5:20 p.m. local time) from Levi's Stadium in Santa Clara, Calif.

The 49ers and Cowboys have a long history with one another, but San Francisco has gotten the better of Dallas in each of their last two meetings.
</item 5>

<item 6>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
For a Cowboys team that has embarrassed a few opponents this year, they might be on the bad end of an ugly game here.

49ers 28, Cowboys 10

10:27 p.m. TOUCHDOWN — Not one, not two, THREE touchdowns for George Kittle. The 49ers have everything working offensively tonight, and any Kittle fantasy
owners are thrilled.

10:25 p.m. — Aiyuk's loss is Deebo Samuel's gain. Samuel goes 43 yards on the very next play, which tells you how automatic the 49ers' offense is right now. San
Francisco is in the red zone looking to make this a three-score game.

10:23 p.m. — An offensive holding call wipes out a 40-yard gain from Aiyuk. Wow. That's a momentum swing. Spencer Burford is the culprit.

10:21 p.m. — The Purdy-to-Aiyuk connection continues to get stronger.
</item 6>

<item 7>
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
</item 7>

<item 8>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
8:27 p.m. — A little bit of chaos as Deebo Samuel drops a ball that was pretty close to being a fumble, but it's all a non-issue as Dallas is hit with an
offsides penalty. The 49ers are right about in field goal range.

8:25 p.m. — After a facemask penalty and a nine-yard pass to Brandon Aiyuk, the 49ers are already over midfield.

8:23 p.m. — Kickoff is a bit on the later end, but we are about to get underway. The 49ers will start with the ball.

8:10 p.m. — Jerry Jones said this week that this game will be a test of how the Cowboys match up with the NFL's best, as if there wasn't already pressure on
Dallas to show something after the 49ers knocked them out of the playoffs in back-to-back years.

7:56 p.m. — Frank Gore and Vernon Davis are in the house!
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
</item 9>

<item 10>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
— The Purdy-to-Aiyuk connection continues to get stronger. Aiyuk goes for 23 yards to give the 49ers some breathing room after opening their drive with a false
start.

49ers 21, Cowboys 10

10:17 p.m. FIELD GOAL — The 49ers' defense tightens the screws and limits Dallas to a field goal. Brandon Aubrey drills it from 50 yards out, and it's down to a
21-10 lead for San Francisco.

10:13 p.m. — Prescott hits CeeDee Lamb for a big 29-yard gain. That quickly takes the Cowboys into 49ers territory to start the second half.

End of first half: 49ers 21, Cowboys 7

9:58 p.m. — San Francisco's defense is all over the Cowboys yet again. It's Dre Greenlaw's turn to sack Prescott, and Dallas is waving the white flag on the
half.
</item 10>
=== END ITEM id=1e04ec1c9e111a17 ===

=== ITEM id=fcc2014582c22bd0 ===
Gold sentences and facts:
Sentence 0: Love Story This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No.
  Fact 0.0: This finally brings us to the fated sighting of Swift at Arrowhead Stadium.
  Fact 0.1: The sighting was on Monday.
  Fact 0.2: Swift was cheering on 'No.'
Sentence 1: The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before.
  Fact 1.0: The game is taking place at Arrowhead Stadium.
  Fact 1.1: Swift has performed at Arrowhead Stadium herself.
  Fact 1.2: Swift has attended a game at Arrowhead Stadium before.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Oct. 15: Kelce, Swift make 'SNL' cameos

The two were spotted out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift
was getting during games, while the singer introduced the musical act of the night, Ice Spice.

Oct. 22: Swift attends fifth Chiefs game

Swift has now attended games broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and
will be shown on CBS. She is sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes' wife, Brittany.
</item 3>

<item 4>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 4>

<item 5>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But at the same time, they're overdoing it a little bit."

Jason also chimed in on the matter, saying, "I just think the NFL is not used to celebrities coming to the games. Like basketball has it all figured out.
They're all courtside, they're sitting there. They show 'em once or twice, but then they get back to the game."

Oct. 12: Swift attends Chiefs-Broncos

"Now it's too late for you and your white horse *Bronco, to come around" - White Horse

One day after her movie premiere, Swift is in the house for the Chiefs' "Thursday Night Football" divisional matchup against the Broncos. The game is taking
place at Arrowhead Stadium, where Swift has performed herself and attended a game before. According to Al Michaels, the game will have minimal Swift coverage,
though so far there has already been a mention of the superstar.

Oct.
</item 6>

<item 7>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Suddenly, football fans and Swifties were living through this surprising collision. Swift lyrics were infiltrating NFL game commentary, and her fans were
furiously becoming football experts. All that needed to happen was for Swift herself to infiltrate a game, determining once and for all whether there was
anything to these rumours or not. This Is Me Trying

On September 21, Kelce announced on The Pat McAfee Show he had invited Swift to watch one of his games. “I told her, ‘I’ve seen you rock the stage in Arrowhead,
you might have to come see me rock a stage at Arrowhead and we can see which one’s a little more lit,’” he said. “We’ll see what happens in the near future.”
Were they finally about to hard-launch their friendship/potential romance to the world, laying all speculation to rest? Love Story This finally brings us to the
fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No. 87 with gusto.
</item 7>

<item 8>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."
</item 9>

<item 10>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She explained that she doesn’t necessarily pay attention to the cameras and fans around her in those stadiums because she is watching her boyfriend play.

(Denver Post via Getty Images)

“I’m just there to support Travis,” she said. “I have no awareness of if I’m being shown too much and pissing off a few dads, Brads, and Chads.”

When discussing how many Swifties have now started watching NFL games since she started dating Kelce, she quipped about how much she likes the sport. “Football
is awesome, it turns out,” she joked. “I’ve been missing out my whole life.”

Swift’s effect on the NFL didn’t go unnoticed in October, as one game between the Chiefs and New York Jets - which she attended with some of her famous friends
- became the most-watched Sunday NFL show since the Super Bowl LVII in February 2023.
</item 10>
=== END ITEM id=fcc2014582c22bd0 ===

=== ITEM id=4e49cba52ac2201a ===
Gold sentences and facts:
Sentence 0: As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse gases.
  Fact 0.0: They are never going to replace the need for reducing emissions of CO2.
  Fact 0.1: They are never going to replace the need for reducing emissions of other greenhouse gases.
Sentence 1: He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about 90%, according to his math.
  Fact 1.0: The person stopped flying.
  Fact 1.1: The person became a vegetarian.
  Fact 1.2: The person ditched gasoline-powered cars.
  Fact 1.3: The person drives a Tesla.
  Fact 1.4: The person cut his personal emissions by about 90%, according to his math.

Retrieved items:
<item 1>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 1>

<item 2>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Kalmus, trailed by a reporter, buzzed around the homestead like a giddy kid, hiking back and forth to a home in an eco-village near the Possibility Alliance
where he was living and working on his research. He tramped down a forest trail in bare feet, paused work for a midday dip in a shady stream and never changed
his T-shirt.

He ground his coffee with a mortar and pestle, thumbed through a book of Enneagram personality types on the porch with his son Zane and listened to Ethan’s
stories about sailing across the Atlantic in a cargo ship to avoid the carbon emissions of flying.

“I feel met by Peter,” Sarah Wilcox-Hughes said of Kalmus. “I don’t feel like there are many people prioritizing the needs of the planet as he is and we are.”

Kalmus found life in Belfast, and at the Possibility Alliance, slower, richer and more connected to the natural world.
</item 2>

<item 3>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
BELFAST, Maine — How does it feel to be a climate scientist in 2023?

Let’s set the scene. Hundreds of picnickers are sprawled across a lawn speckled with colorful blankets. Yachts bob in the bay. Guitars blare, children wrestle
and dancers twirl across a concert lawn in the golden glow of a dying summer night.

It’s a picture-perfect August evening for the seaside town of Belfast’s “Summer Nights” concert series.

Off to the side, Peter Kalmus sits alone. His eyes are closed. His toes are tucked beneath his body. He’s catching an occasional side-eye from onlookers.

“When I’m meditating, I don’t feel anxious,” Kalmus later says, explaining that he strives for two hours of meditation a day to keep his climate anxiety at bay.
Otherwise, “it’s completely overwhelming.”

Many of us avoid thinking too deeply about the worst scenarios of climate change. He is fixated on them.
</item 3>

<item 4>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Neighbors drop by without notice.

“We’ve created what existed 100 years ago,” Ethan Hughes said.

On a steamy August morning, Kalmus found himself huddled around a faded picnic table at the center of the homestead in Belfast, Maine, sipping his coffee with a
rare breed — people as alarmed by climate change as him.

climate scientist environmental (Evan Bush / NBC News)

Ethan, the charismatic, folksy co-founder of the Possibility Alliance, started the day off with a poem about a deer in the forest and then began to outline
daily chores — chopping wood, building a new deck, caring for goats.

Soon, talk turned to climate change and Kalmus’ grief began to flow as he shared findings from unpublished research on his projections of heat in moderate
emissions scenarios through the year 2300.

“Even in the coolest time of day, people will be dying if they don’t have air conditioning,” Kalmus told the group.

“Everything is already spinning out of control,” said Sarah Wilcox-Hughes, dismayed.
</item 4>

<item 5>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He is fixated on them.

Kalmus, 49, thinks civilization is on the path to break down, the Biden administration is clueless on climate, and that he might get fired from his job at NASA
if he is arrested for a third time protesting what he views as downright madness: the continued use of fossil fuels.

Kalmus visited the seaside town of Belfast, Maine, in July and August to visit the Possibility Alliance homestead as he looked for inspiration on how to best
live with the effects climate change. (Evan Bush / NBC News)

“It’s kind of weird and lonely,” he said. “People don’t want to talk about this stuff at parties.”

Kalmus, whose strident tweets reach some 340,000 followers on X, is at the vanguard of a group of scientists who are increasingly disillusioned and growing
convinced that more radical action is needed to wake up political leaders.
</item 5>

<item 6>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Twice, he’s been arrested during climate protests.

His social media posts often evoke heartbreak.

“We’re in the process of losing basically everything,” he wrote recently on X.

Is anyone listening?

climate scientist natural farming (Evan Bush / NBC News)

Not his suburban neighbors in North Carolina, who continue to drive emissions-belching SUVs to big-box stores. Perhaps not even some in the science community,
like those who kicked him out of the American Geophysical Union’s annual fall meeting last year after he unfurled a banner and protested on the major science
conference’s biggest stage.

He has grown increasingly frustrated with President Joe Biden, who signed the Inflation Reduction Act as his signature climate bill. Kalmus thinks it does too
little to shut out the fossil fuel industry.

(Rhodium Group, a nonpartisan think tank, estimates greenhouse gas emissions will drop 32%-42% below 2005 levels by 2030, well short of Biden’s own benchmark
for progress. The inflation bill is responsible for about a quarter of that projected decrease.)
</item 6>

<item 7>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
While on his working vacation in Maine, Kalmus invited NBC News for a short visit to the community to see the crisis through the eyes of Americans taking it as
seriously as anyone.

The Possibility Alliance is an 11-acre homestead that operates without power, running water or a sewer system. The Hughes family, which operates the alliance,
pays only one monthly bill for landline phone service. (Evan Bush / NBC News)

Detoxifying

About a decade ago, Kalmus ditched a promising academic career in astrophysics to study the physics of climate change instead.

He made other changes too. He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about
90%, according to his math.

It wasn’t a political statement, but a detoxification.

“Burning fossil fuels started to feel more and more disgusting,” he said.

Ethan Hughes, who operates the Possibility Alliance, rejects globalization and thinks capitalism is a ruinous symptom of “greed and separation.”
</item 7>

<item 8>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He’s trying to build a gift economy in Belfast, in which neighbors share resources and services with no expectation of repayment. “I’m trying to role-model
post-petrol,” Hughes said. (Evan Bush / NBC News)

Kalmus, now a married father of two teenagers, is introspective, earnest and narrowly focused on climate change in both his personal and professional life. Tall
with an athletic build, a neat beard and a tangle of messy hair that has a skunklike streak of white, he can sometimes come off like an overgrown teenager: He’s
the smartest person in the room, brimming with idealism and still indignant — with a zeal many adults can no longer muster — about how life’s most annoying
systems work (like the U.S. medical system, a symptom of what he calls the “capitalist death cult”).

The changes he made weren’t enough. Kalmus wrote a book on climate change and chained himself to a Chase bank in protest of fossil fuel financing. Twice, he’s
been arrested during climate protests.
</item 8>

<item 9>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
These materials are harder to make than MXenes. So most research on them has been done with math or computer models. But MBenes could have unique mechanical and
electrical traits, Ozkan’s team finds. And those just might help them last through many, many cycles of CO 2 capture and release.

Each year, human activities spew tens of billions of tons of CO 2 into the air. That’s equivalent to the weight of tens of billions of elephants. Carbon
absorbers, including MXenes or MBenes, can’t catch it all. As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse
gases.

Still, given the urgency of climate change, even a quick or partial fix — such as carbon capture — can’t be ignored. “At this point in time,” says Persson, “we
need to look at all the possible solutions.” And MXene-like materials are emerging as potentially important candidates.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=4e49cba52ac2201a ===

=== ITEM id=841962a9b15a85a2 ===
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
</item 1>

<item 2>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The future of Google’s app store is at stake in a lawsuit by Fortnite publisher Epic Games. Epic sued Google in 2020 after a fight over in-app purchase fees,
claiming the Android operating system’s Google Play Store constituted an unlawful monopoly — while Google says its demands would damage Android’s ability to
offer a secure user experience and compete with Apple’s iOS. Follow along with updates here.

Bornstein showed jurors charts of Google’s fat app profit margins (70 percent on $12 billion in revenue a year, says Epic) and pointed out several ugly-seeming
ways Google has allegedly attempted to keep anyone from taking that money away — like paying game developers not to build their own app stores or standalone app
launchers like Epic did with Fortnite.

“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying,
‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”
</item 2>

<item 3>
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
</item 3>

<item 4>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 4>

<item 5>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
I did like Pomerantz’s commonsense argument that Google can’t possibly have a monopoly on Android app stores when “every single Samsung phone comes with two app
stores right on the homescreen,” which continued:

When they show these charts that show all these downloads from Play and not from the Galaxy Store, that’s what the Samsung phone users are choosing. They’re
touching Play. Nothing’s keeping them from touching the Galaxy Store; it’s just what works for them.

I called Google “comparatively open” earlier, and that openness will likely be heavily debated in the weeks to come. Epic promised to “show that Google has
closed off each and every other option” to the Play Store during this trial. But Google points to the simple fact that it allows alternate app channels at all —
something Android rival iOS doesn’t.

Pomerantz boasted that over a billion people have gone through the process Epic portrays as needlessly onerous to get apps outside the Play Store.
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 7>

<item 8>
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
</item 8>

<item 9>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
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
=== END ITEM id=841962a9b15a85a2 ===
