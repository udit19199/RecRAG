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

=== ITEM id=92849eadc0cc6259 ===
Gold sentences and facts:
Sentence 0: They’d all have significant value if president of baseball operations Jerry Dipoto decided to move one for a bat.
  Fact 0.0: Jerry Dipoto is president of baseball operations.
  Fact 0.1: Those referred to as 'they' would all have significant value if Jerry Dipoto decided to move one for a bat.
Sentence 1: Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.
  Fact 1.0: Aaron Judge came away with a new contract.
  Fact 1.1: Justin Verlander came away with a new contract.
  Fact 1.2: Trea Turner came away with a new contract.
  Fact 1.3: Xander Bogaerts came away with a new contract.

Retrieved items:
<item 1>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
20. Brandon Drury, 2B, Angels

The Halos signed Drury to a two-year free agent deal last winter. The righty-hitting infielder turned in a nice season, connecting on 26 home runs with a
.262/.306/.497 line across 523 plate appearances. Next year’s $8.5M salary looks like strong value, particularly in an offseason where the supply of
up-the-middle players is thin.

The Angels probably aren’t kicking off a rebuild. They could do a shorter-term retool if Ohtani walks, or they could simply look to move Drury for pitching
while turning second base to Luis Rengifo and/or an external acquisition.

21. Brendan Donovan, 2B, Cardinals

Donovan is a longer shot trade possibility if the Cards move a controllable hitter for starting pitching. He’s more valuable than Carlson and less likely to be
on the move.

At the same time, he’d net a clearer rotation upgrade if St.
</item 1>

<item 2>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
It was the first time he stayed healthy for a 162-game schedule. The result was a season that’ll get him down-ballot MVP support.

Robert is a prime-aged superstar and, unlike Soto or Burnes, he’s nowhere near free agency. He’ll make $12.5M next year, $15M in 2025 and is controllable via
$20M team options for 2026-27. Trading Robert would signify a complete teardown that the Sox probably don’t want. Yet the haul would be astronomical if they
decided to go in that direction.

25. Pete Alonso, 1B, Mets

Alonso was the subject of early offseason trade speculation. With the Mets projecting a less active offseason and placing a priority on 2025, it wasn’t out of
the question he could be available. Alonso is projected for a $22M arbitration salary and will be a free agent after next year.

New York has subsequently walked back some of their statements about a possible retool.
</item 2>

<item 3>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
He is sitting on 417 career saves, ranking eighth all-time. He has a chance to vault all the way up to fourth, with Francisco Rodriguez (fourth) at 437 and
Kenley Jansen (420 saves) the only other active reliever ahead of him.

Trent Grisham likely headed to New York in Soto deal

Date: Wednesday, Dec. 6

Source: Joel Sherman, New York Post

The Yankees won't just be getting one outfielder from the Padres; it appears they'll be getting two. The NY Post's Joel Sherman reported Trent Grisham would
also likely be on the move headed to New York. Grisham enjoyed a pair of breakout seasons to start his career in San Diego and has twice earned Gold Gloves
(2020 and 2022), but his offensive production has dwindled, with him batting under .200 in back-to-back seasons.

Juan Soto to Yankees 'close to being finalized'

Date: Wednesday, Dec.
</item 3>

<item 4>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
It appears the team could be close to landing the biggest trade acquisition of the offseason. YES' Jack Curry reported the Yankees have "intensified efforts" to
trade for Soto and that a deal "is likely." He expects New York to deal four or five players.

Additionally, WFAN's Brandon Tierney reported YES was told by the team to prepare for a 2 p.m. ET emergency "Hot Stove Show" and that there is a chance the
Yankees complete the deal by lunch.

Orioles 'close to deal' with reliever Craig Kimbrel

Date: Wednesday, Dec. 6

Source: Ken Rosenthal, The Athletic and Joel Sherman, New York Post

The Orioles are said to be "close" to a deal with free agent and former Phillies closer Kimbrel, Rosenthal reported Wednesday morning.

Baltimore will be without All-Star closer Felix Bautista for the entire 2024 season after receiving Tommy John surgery in October.
</item 4>

<item 5>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
And fans will have to wait a bit longer before hearing a final resolution on his decision. The Newark Star-Ledger's Bob Klapisch reported Yamamoto isn't
expected to make a decision until after Christmas, and that it could run until around Jan. 1-4.

Pirates re-sign franchise legend Andrew McCutchen

Date: Tuesday, Dec. 19

Source: Jason Mackey, Pittsburgh Post-Gazette

The Pirates have been making a few veteran free-agent signings to bolster the rotation. Now, they're bringing back a franchise icon. Pittsburgh has reportedly
inked Andrew McCutchen to a one-year, $5 million.

McCutchen enjoyed a solid return to Pittsburgh in 2023, slashing .256/.378/.397 with 12 home runs in 112 games. It was his first season in Pittsburgh since
2017.

Angels discussing Blake Snell

Date: Tuesday, Dec.
</item 5>

<item 6>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Shohei Ohtani’s predicted $535 million for 11 years (according to a Sports Illustrated segment) will bring the Dodgers down to pay him a good $48.63 million
annually.

The Dodgers already have six players whom they have to pay above $10 million a year: Freddie Freeman ($27 million), Mookie Betts ($25 million), Clayton Kershaw
($20 million), Chris Taylor ($15 million), Max Muncy ($13.5 million), and J.D. Martinez ($10 million). Among the top-paid players, Clayton Kershaw is exiting
the team this season. That, along with a few other player offloads, will free up enough space in the Dodgers payroll to support two $25 million or three $10
million p/a players on their team. Neither Trout nor Ohtani falls within that price range, so keeping both together at the same time would not be a feasible
option for the Dodgers right now.

Pitchers or Hitters?
</item 6>

<item 7>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Follow along below for the latest news.

MLB free agency tracker, live updates

Padres ink five-year deal with NPB star Matsui

Date: Saturday, Dec. 23

Source: ESPN's Jeff Passan

Another Japanese star is headed to the Sunshine State. Star reliever Yuki Matsui inked a five-year, $28 million deal to join the Friars on Saturday. Matsui was
one of NPB's best arms out the pen, starring for Tohoku Rakuten Golden Eagles for nearly a decade. He tallied up 236 saves over the course of his career,
posting a 2.40 ERA in 659.2 innings of work. The contract features opt-out clauses in Years 4 and 5, offering San Diego an out if Matsui deals with elbow
trouble.

Dodgers sign Yamamoto to $325 million contract

Date: Tuesday, Dec.
</item 7>

<item 8>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
But MLB Network's Jon Morosi said it's unlikely the Jays will land both star players, reporting they might land one of the two, but they will not add both this
offseason.

Ohtani has already received several $500+ million offers

Date: Sunday, Dec. 3

Source: Jon Heyman, NY Post

It's no surprise the bidding for Ohtani is heating up, with the star expected to sign soon. And New York Post's Jon Heyman reported Ohtani has already received
several deals that have topped $500 million, which could mean he winds up being the first $600 million player in MLB history. He added there are believed to be
five teams in the running for Ohtani.

Decision near for Ohtani

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

Nothing gets the hot-stove crowd excited like a big move during the winter meetings, and the biggest move of the offseason could come this week in Nashville or
soon after.
</item 8>

<item 9>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Bellinger would be returning to the Cubs, the team with which he rebuilt his value after a down year with the Dodgers. If he signed with the Blue Jays, he'd be
bringing the powerful left-handed bat the team sought when it pursued Ohtani.

Dodgers send stars to meet with Yamamoto

Date: Wednesday, Dec. 13

Source: Fabian Ardaya, The Athletic

The Dodgers added one Japanese star this offseason. They're now looking to add another. Los Angeles reportedly sent Ohtani, Mookie Betts, Freddie Freeman and
Will Smith to meet with Yamamoto, trying to convince the standout starting pitcher to join the National League superpower.

Signing Yamamoto would be a major boost for the Dodgers, who still have a need at starting pitcher.

Rays-Dodgers working on a Tyler Glasnow trade

Date: Wednesday, Dec.
</item 9>

<item 10>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
MLBTR projects Verdugo for a $9.2M salary in his final season of arbitration. He’s coming off another league-average offensive showing, when he hit
.264/.324/.421 with 13 home runs through 602 plate appearances.

Verdugo hasn’t become the All-Star player the Sox envisioned when building the Mookie Betts return around him. He’s a solid regular, a high-contact hitter who
rates as a decent defender in right field.

8. Trent Grisham, CF, Padres

9. Ha-Seong Kim, 2B, Padres

If the Padres hold Soto, dealing Grisham and/or Kim could be an alternative means of trimming payroll. While neither is going to make anywhere near Soto money
in 2024, they’re each on mid-level salaries of note.

Grisham is projected for a $4.9M salary in his second-to-last season of arbitration.
</item 10>

<item 11>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Coincidentally, Ohtani made his original decision to sign with the Angels on Dec. 8, 2017, six years to the day.

Dave Roberts says Dodgers met with Ohtani

Date: Tuesday, Dec. 5

Source: Alden Gonzalez, ESPN

Dodgers manager Dave Roberts told reporters Tuesday that the team met with Ohtani for 2-3 hours over the weekend. Roberts said Ohtani is "clearly" the Dodgers'
top priority, adding that the team is confident the two-time MVP will pitch again. Roberts wasn't able to offer a timeline for Ohtani's decision.

Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec. 5

Source: Bob Nightengale, USA Today

The Cubs have been viewed as one of Ohtani's likeliest destinations for a while. It appears that might no longer be the case.
</item 11>

<item 12>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Unfortunately, Trout has become a victim of an injury streak, starting in 2022. He got his hands, left groin, back, and foot injured five times in the entire
season; in 2023, he faced two hand injuries in two consecutive months.

Despite his injury-prone history, the more logical option for the Dodgers would be to choose Trout over Ohtani; the team will not be able to afford them both at
the same time. Trout comes with a cheaper price tag of $35.45 million compared to Ohtani’s projected $48.63 million. The Dodgers could use his All-Star form in
between injuries if he still keeps getting them in 2024. With Shohei Ohtani, the Dodgers would be wasting a year letting him play at-bat, with a high chance of
getting him further injured and missing more pitching time.
</item 12>

<item 13>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
Also, given he is currently in arbitration with the Yankees, it adds to his trade chances. The reason is that the team won’t want to lose the opportunity to
cash in some money from his trade. The decision appears promising as the team seeks a shift, but the data tells a different story. Gleyber Torres’s Exit Becomes
the Talk of the Town as the Yankees Look for a Rebuild The New York Yankees have been in the news since the league stage ended. First, because of the
possibility of a third-party audit, which was later shunted out. Later, the comments from Aaron Judge brought a lot of attention to the workings of the team,
conflicts, and divisions between the core four. Now it’s the assumptions of changes with the idea of cutting down the $300 million budget for the next season.
One name that is making the rounds for an exit is Gleyber Torres.
</item 13>

<item 14>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
10.

Yamamoto is the biggest fish on the free-agent market right now, and he's being pursued by a handful of big-market teams including the Yankees and Dodgers. The
Giants have missed out on some of the most prominent free agents over the last two offseasons, so there might be incentive to make Yamamoto a massive offer to
try and stay relevant in the NL West after the Dodgers' Ohtani signing.

MORE: What to know about Japanese ace Yoshinobu Yamamoto as he moves to MLB

Royals agree to 1-year deal with reliever Will Smith

Date: Sunday, Dec. 10

Source: Jeff Passan, ESPN

The Royals have agreed to a 1-year, $5 million contract with reliever Will Smith, who spent this past season with the Rangers.

Smith has won the World Series in each of the last three seasons, but Kansas City's expectations in 2024 don't quite extend to a championship.
</item 14>

<item 15>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Pavlovic added that it is very likely that Yamamoto ends up either in Los Angeles with the Dodgers, or New York with the Mets or Yankees.

Yamamoto met with Giants, won't decide on team until after Christmas

Date: Tuesday, Dec. 19

Source: Andrew Baggarly, The Athletic and Bob Klapisch, Newark Star-Ledger

San Francisco has been among the teams most heavily believed to be in on Yamamoto, and according to The Athletic's Andrew Baggarly, the team met with him in San
Francisco last week. Baggarly reported the Giants have been willing to "blow teams out of the water" to land Yamamoto, but noted he believes the crowded picture
for the Japanese ace could make it more challenging for the Giants to be the winner of the Yamamoto sweepstakes.

And fans will have to wait a bit longer before hearing a final resolution on his decision.
</item 15>

<item 16>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Even though stud reliever Yennier Cano emerged as the top candidate to take the job, Kimbrel could bolster the back end of the O's bullpen.

Victor Caratini lands with Astros

Date: Tuesday, Dec. 5

Source: Chandler Rome, The Athletic and Mark Feinsand, MLB.com

Veteran catcher Victor Caratini is headed to Houston on a two-year deal worth $12 million. The 30-year-old will serve as the backup to new starter Yainer Diaz
now that Martin Maldonado has moved on.

Caratini filled a similar role in Milwaukee the last two years, slashing .259/.327/.383 with seven home runs in 62 games in 2023.

Pirates acquire Marco Gonzales from Braves

Date: Tuesday, Dec.
</item 16>

<item 17>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
The right-hander pitched to a 5.87 ERA over 19 big league outings. His walks more than doubled while his strikeouts fell from 22.9% to a modest 19% clip.

He lost his spot in the Toronto rotation and was eventually demoted to the minors. Manoah never pitched in Triple-A and received an injection in his throwing
shoulder at the end of the season.

How will the Jays proceed after that disastrous year? General manager Ross Atkins said last week the team would give him “a strong leg up” on the No. 5 spot in
the rotation. The Jays aren’t broadcasting a desire to trade him, although they wouldn’t be doing themselves any favors if they declared they were ready to move
on.

Dealing Manoah to bring in a lower upside but higher floor pitcher wouldn’t be a surprise with the Jays firmly in win-now mode. The former first-round pick is
controllable for four more seasons and won’t reach arbitration until next winter.

20.
</item 17>

<item 18>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
20

Source: Ken Rosenthal, The Athletic

Buried at the bottom of Rosenthal’s piece on what an Ohtani contract might look like — incentives, opt-outs and all sorts of stuff — is this little bit of
reporting:

An agreement at the winter meetings in Nashville the first week of December would be the ideal outcome for MLB, but not necessarily for Ohtani. He perhaps will
be better off waiting for the signing of Japanese righty Yoshinobu Yamamoto, who is expected to be posted for free agency as soon as Monday, triggering a 45-day
window for teams to negotiate with him. Losing out on the 25-year-old Yamamoto, who could command more than $200 million, might put certain teams on tilt,
improving the landscape for Ohtani.

Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov.
</item 18>

<item 19>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
That’s gotta be super disappointing for Mariners fans starved for a title, right?

On the other hand, maybe the M’s know all about Ohtani’s desire to keep the rumor mill quiet, and leaking “news” like this would please Ohtani and his camp.

‘Timing could be right for a Cubs-Ohtani union’

Date: Nov. 16

Source: Jesse Rogers, ESPN

The Cubs were one of the seven finalists for Ohtani the first time around, but lack of the DH in the NL at the time made it hard for Ohtani to choose Chicago.
His primary goal was showing he could both hit and pitch at a high level for an MLB team, and trying to play the outfield instead of just DHing would have been
too much, even for Ohtani.

Now, though? The NL has the DH, and the Cubs are itching to make a sport-shaking signing. Well, another sport-shaking signing.
</item 19>

<item 20>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Passan wrote a piece about 10 teams to watch the rest of the offseason, and he threw this little note in the section about the Red Sox, and how they needed to
sign Yoshinobu Yamamoto.

Boston's bats are good, or good enough at least, to avoid having to spend like mad in a down year for free agent hitters -- unless, of course, it's for Ohtani,
who has an affinity for Boston. If the Red Sox don't hit that jackpot, they're backfilled nicely with a system that includes outfielder Roman Anthony, shortstop
Marcelo Mayer and catcher Kyle Teel, among the best trios of offensive prospects in baseball.

It's cool when the best player in the world is a free agent and has an "affinity" for your town.

'Money's great, but winning's better'

Date: Nov.
</item 20>

<item 21>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
That informs the offseason trade landscape, which is headlined by a few stars deep into their arbitration windows. Without many clear rebuilders, it’s possible
we see more swaps of big leaguers at positions of need between teams that expect to compete.

Before we get to the list, a note on methodology. This isn’t a strict ranking of players’ trade value, nor is it solely about likelihood of being moved. We’re
trying to balance both of those things, an inherently subjective exercise.

All projected salaries for arbitration-eligible players are courtesy of MLBTR contributor Matt Swartz.

1. Juan Soto, LF, Padres

Other than Shohei Ohtani’s free agency, there may be no bigger story of the offseason than Soto’s future in San Diego. He is entering his final year of club
control, projected for a $33M salary that would be the highest ever for an arbitration-eligible player.

The Padres are scaling back payroll, perhaps as much as $50M.
</item 21>

<item 22>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
He still has a lot left to give to the diamond for approximately 6 more years. If the Dodgers have the chance to draft them, should they go with Ohtani, Trout,
both, or neither? Let us know your opinions in the comment box.

Watch This Story: Shohei Ohtani’s Two-Way Dream: Before Injury, He Always Wanted Both!
</item 22>

<item 23>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
All eyes since the start of the offseason have been on Shohei Ohtani. And for the most part, there hasn't been much to see.

The two-way phenom is the most sought-after free agent in MLB history, with his combination of talent and marketability making him a franchise-changing player.

And the bidding for Ohtani could be winding down soon. Multiple reports have indicated Ohtani could sign at some point during the winter meetings, which are
scheduled to take place from Dec. 3 through Dec. 6.

Many of the top teams are expected to be in the mix for Ohtani. The Dodgers and Cubs are among the top teams expected to be aiming to sign the two-time AL MVP,
and plenty of others should continue to compete through the end to land him.

Here's the latest on Ohtani's free agency pursuits.
</item 23>

<item 24>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
New York has subsequently walked back some of their statements about a possible retool. They’re still aiming to compete in ’24 even if they’re signaling they
won’t blow away the top of the free agent market.

First-year president of baseball operations David Stearns has twice publicly said he anticipates Alonso being the Opening Day first baseman in Queens. That
won’t stop teams from trying, but an offseason deal looks decidedly unlikely.

Others of note

A’s: Seth Brown

Braves: Vaughn Grissom

Brewers: Adrian Houser, Devin Williams, Brandon Woodruff

Cardinals: Alec Burleson, Tommy Edman, Nolan Gorman, Tyler O’Neill

Giants: J.D.
</item 24>

<item 25>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Kelenic, 24, is the headliner of the deal as a former No. 6 overall pick in the 2018 MLB Draft and top five-ranked prospect in 2021. The 24-year-old has not
found any consistency in his MLB career thus far, hitting .206 with a .656 OPS through three tumultuous seasons. He has 32 home runs, 109 RBI and 24 stolen
bases in 974 plate appearances.

Gonzales, 31, is also a decent rotation arm with 65 wins, 665 strikeouts and a 4.14 ERA over 893.0 innings pitched in his nine-year career.

Giants emerging as possible frontrunners for Yoshinobu Yamamoto

Date: Sunday, Dec.
</item 25>

<item 26>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
NY Post's Jon Heyman reported Miami is at least believed to be listening to trade packages for its young starters, including Jesus Luzardo, Braxton Garrett and
Edward Cabrera, among others.

Alex Verdugo 'expected' to stay with Yankees despite potential Juan Soto trade

Date: Wednesday, Dec. 6

Source: Jon Morosi, MLB Network

The Yankees acquired Alex Verdugo yesterday from the Red Sox. With the team's reported interest in Juan Soto, there was some speculation that Verdugo could be
flipped to the Padres, however, MLB Network's Jon Morosi reported that New York is "expected" to keep the outfielder rather than include him in a deal with San
Diego.

Juan Soto trade to Yankees 'is likely'

Date: Wednesday, Dec. 6

Source: Jack Curry, YES and Brandon Tierney, WFAN

New York landed Alex Verdugo last night in a deal with the Red Sox.
</item 26>

<item 27>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
While he won't pitch again until 2025, Ohtani's power and on-base ability gives Los Angeles the most dangerous trio of hitters in the sport.

MORE: How Dodgers' 2024 payroll looks after handing out monster deal to Shohei Ohtani

Red Sox swing deal for Cardinals' Tyler O'Neill

Date: Friday, Dec. 8

Source: Jon Heyman, New York Post

Days after trading Alex Verdugo to the Yankees, the Red Sox have acquired OF Tyler O'Neill from the Cardinals in a deal that sends pitching prospects Nick
Robertson and Victor Santos to St. Louis.

O'Neill enjoyed a breakthrough 2021 season with the Cardinals, batting .286 with 34 home runs and a .912 OPS, but he's struggled to recapture that level of
production over the last two seasons and is set to be a free agent after 2024.
</item 27>

<item 28>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
17. Dylan Carlson, CF, Cardinals

It was a bit of a surprise that Carlson didn’t change hands over the summer. Various reports suggested the Cards were making him available as they looked for
ways to add a controllable starter. It didn’t happen, but the same logic applies this winter.

Carlson has been surpassed by Lars Nootbaar as the starting center fielder. St. Louis needs multiple starting pitchers and will likely bring in at least one via
trade. Carlson probably has less value than he did before the deadline. Not only did he finish with a mediocre .219/.318/.333 batting line, he underwent
season-ending surgery on his left ankle in September.

There’d nevertheless be teams willing to take a shot on a former top prospect who looked like an average or better center fielder 12 months ago. Carlson is
projected for a modest $1.8M salary and has three years of arbitration control.

18.
</item 28>

<item 29>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
It also makes him less likely to be traded this offseason.

First-year GM Chris Getz has suggested no one on the roster is truly untouchable, an understandable approach for a team that just lost 101 games. Neither owner
Jerry Reinsdorf nor Getz has called this a rebuild, though.

Any slim hopes the White Sox may have of competing in 2024 would be dashed by trading Cease, the only above-average starting pitcher on the roster.

7. Alex Verdugo, RF, Red Sox

Verdugo’s name has been floated in trade rumors for a few seasons. The Red Sox have held onto him thus far. They have a new baseball operations leader in Craig
Breslow, who has acknowledged the team has gotten early calls on the left-handed hitting outfielder.

Jarren Duran and Masataka Yoshida also hit from the left side. Dealing Verdugo could free some at-bats for a right-handed addition while perhaps bringing in
immediate help for the pitching staff or at second base.
</item 29>

<item 30>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
The Giants are considered to be an unknown.

In addition to the teams, Passan also reported Ohtani is expecting a contract somewhere between $550 and $600 million, which would be the largest guaranteed
contract in North American sports.

Blue Jays could alter roster with Ohtani signing

Date: Nov. 30

Source: Jon Heyman, New York Post

The Blue Jays are one of the teams believed to still be among the strongest in the mix for Ohtani. The New York Post's Jon Heyman reported there could be some
changes if Ohtani signs with the Blue Jays, however, reporting that some rivals believe Bo Bichette and Vladimir Guerrero Jr. could hit the trade block if
Toronto lands the star.

Trading the two infielders might seem counterproductive to signing Ohtani in the first place. Those two would be the biggest supporting bats in the lineup
around Ohtani.
</item 30>

<item 31>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
If the Sox are prepared to move on, another team could look to buy low.

He’ll make a $13M salary next year, the final guaranteed season of his contract. He’s guaranteed a $3M buyout on a $16.5M club option for 2025 and the deal
contains an $18.5M team option for the ’26 campaign.

13. Max Kepler, RF, Twins

14. Jorge Polanco , 2B, Twins

Minnesota president of baseball operations Derek Falvey has confirmed the team is paring back payroll. The Athletic’s Dan Hayes reported the organization could
end in the $125M-140M range. At the moment, they have around $120M in projected spending.

The Twins have a few ways to clear room. Trading either Kepler or Polanco would accomplish that. The former is making $10M, while the latter is playing on a
$10.5M deal.
</item 31>

<item 32>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
It will include the shift of Oswaldo Peraza from third baseman to second. However, Peraza could not do well this season, even after all the positives. His
involvement in unique positions could be a risk for the team.

ADVERTISEMENT Article continues below this ad

Moreover, free agents Whit Merrifield and Adam Frazier have not done well. Given that the Yankees finished second last in total hits and had over 1400
strikeouts, losing a player with the best record in both parameters will not be ideal. Notably, Torres has a 14.6% strikeout rate, with only a handful of
players recording a lower rate. He also covered a lot of bases this season to finish second to Judge. All these factors make him a player that the Yankees
should keep as they look for a change in 2024.

ADVERTISEMENT Article continues below this ad

not in the best shape, losing Torres could haunt the Yankees further.
</item 32>

<item 33>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Twins could trade Max Kepler, Jorge Polance

Date: Sunday, Dec. 17

Source: Jon Morosi, MLB Network

The Twins are coming off an AL Central title, but could be looking to deal away two of their key players form the team. Max Kepler and Jorge Polanco could both
reportedly be on the trade market, per MLB Network's Jon Morosi, though their market could hinge on the status of other infield free agents.

Kepler appeared in 130 games for the Twins, and posted an impressive .260/.332/.484 slash line with 24 homers for Minnesota. Polanco was moved to an infield
utility role with Edouard Julien and Royce Lewis emerging into regulars, but was still solid in 80 games, logging a .255/.335/.454 slash with 14 homers.

Yamamoto requests meeting with Yankees

Date: Sunday, Dec.
</item 33>

<item 34>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
Kim is slated for an $8M salary and will be due a $2M buyout on a 2025 mutual option at season’s end. He’ll be a free agent next winter.

Grisham has the extra year of control but is the worse player. He draws plenty of walks but has hit below the Mendoza line in two straight seasons. He’s coming
off a .198/.315/.352 line over 555 plate appearances. The appeal is on the other side of the ball.

Grisham is an above-average defensive center fielder, keeping him as a low-end regular even if he’s best suited for the bottom of a lineup. San Diego could kick
Fernando Tatis Jr. over to center field if they dealt Grisham, freeing right field for a bigger offensive threat than Grisham provides.

It’s a similar story with Kim. Dealing him would free Jake Cronenworth to move back to second base, opening first base for a better hitter. It’d be tough to
replace Kim’s overall production, though.
</item 34>

<item 35>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Dodgers sign Yamamoto to $325 million contract

Date: Tuesday, Dec. 19

Source: ESPN's Jeff Passan, YES Network's Jack Curry

The Yamamoto sweepstakes are over, as the Dodgers have reportedly signed the Japanese ace to a 12-year, $325 million. It's the largest MLB deal for a pitcher,
surpassing Gerrit Cole's deal with the Yankees by $1 million. Los Angeles' incredible offseason continues, as Yamamoto joins Shohei Ohtani on the defending NL
West champs.

Giants out of Yamamoto sweepstakes

Date: Thursday, Dec. 21

Source: Alex Pavlovic, NBC Sports Bay Area

The Giants made a hard push to land Yamamoto, but according to NBC Sports Bay Area's Alex Pavlovic, San Francisco has been notified that the team is out of the
bidding to land the Japanese ace.
</item 35>

<item 36>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov. 17

Source: Daniel Kramer, MLB.com

Here’s the key piece of the article: “Industry sources told MLB.com this week that landing Ohtani doesn’t appear to be within the Mariners’ realistic agenda
this offseason.”

In a way, that’s surprising. The Mariners seem to check a lot of boxes for Ohtani. They’re set up to win for a long time with a bunch of young starting pitchers
and a co-superstar in Julio Rodriguez. Plus, Ohtani would get to be part of the quest for the franchise’s first World Series title, he’d get to wear the same
uniform as his baseball hero, Ichiro, and he’s lived there in the offseason. So it must be about the money, and Seattle’s reluctance to even get in the
conversation. That’s gotta be super disappointing for Mariners fans starved for a title, right?
</item 36>

<item 37>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
He joins a catching room that also features breakout 2023 rookie Patrick Bailey and Joey Bart. Bart is out of minor-league options, and could be on the
outside-looking-in to a 2024 Opening Day roster spot if the Giants decide to have only two catchers on the active roster.

Pirates add to rotation, sign Martin Perez

Date: Monday, Dec. 18

Source: Jason Mackey, Pittsburgh Post-Gazette

Pittsburgh continues to add to its rotation this offseason. The Pirates have reportedly agreed to a one-year, $8 million deal with former Ranger starter Martin
Perez. He was used as a swing man in 2023, making 20 starts and 35 total appearances, pitching to a 4.45 ERA with 93 strikeouts and 49 walks in 141.2 innings en
route to winning a WorldSeries ring.
</item 37>

<item 38>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
His 3.39 mark across 193 2/3 innings this past season wasn’t quite so dominant but is still high-end production.

Burnes struck out just under 26% of opposing hitters. While his lowest mark since his rookie season, that’s still an above-average figure.

Adames is coming off a less impressive year. The 28-year-old hit .217/.310/.407 through 638 trips to the plate. That’s slightly below-average offense, although
he still connected on 24 home runs.

Adames is an excellent defensive shortstop with plus power in an offseason where there are essentially no free agent shortstops of note. Even with a mediocre
on-base percentage, he’d generate plenty of interest.

4. Shane Bieber, SP, Guardians

Develop high-end starting pitching, trade a top starter as they become more expensive, replace him with talented younger arms. The Guardians have trodden this
path with Corey Kluber, Mike Clevinger and Carlos Carrasco.
</item 38>

<item 39>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
While the offseason has gotten rolling, trade season hasn’t quite arrived. The Tigers acquired Mark Canha from Milwaukee in the biggest swap of the first couple
weeks. Otherwise, the GM Meetings simply served to set the stage for future transactions.

That’ll surely involve some notable trades. Unlike the summer deadline, projecting which players will change hands is difficult at this stage of the offseason.

Aside from the A’s, there aren’t any teams clearly in rebuild mode. Clubs like the Rockies, Royals, White Sox, Nationals and Angels have uphill paths to
contention, yet none has indicated they’re prepared to punt on the 2024 season entirely.

The Padres, Twins, Rays, Brewers and Guardians are all aiming to contend but have questions about their payroll. That’s standard operating procedure for Tampa
Bay, Milwaukee and Cleveland. San Diego and, to a lesser extent, Minnesota have shown a willingness to spend in recent years but are scaling back — in both
cases, likely tied to their uncertain local television rights contracts.
</item 39>

<item 40>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
It’d be tough to replace Kim’s overall production, though. He’s a plus defender at multiple infield spots.

He hit .260/.351/.398 with 17 home runs and stole 38 bases in 152 games. Trading Kim isn’t as damaging to the lineup as a Soto deal would be, but it’d be
tougher than parting with Grisham.

10. Gleyber Torres, 2B, Yankees

Another player one year from free agency, Torres is projected for a $15.3M salary in his last arbitration season. That’s below his market value but a notable
figure for a bat-first second baseman — a general profile that the league has devalued in recent years.

Torres was New York’s second-best offensive player in 2023, hitting .273/.347/.453 with 25 homers through 672 trips to the plate.

Even if they’re not interested in an extension, New York could hold Torres for his final season of club control. They’re already light on consistent hitting
beyond Aaron Judge.
</item 40>

<item 41>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
4

Source: Jon Morosi, MLB Network

It's not surprising that any and all teams should be interested in Ohtani's services, but imagine the Braves landing the superstar? According to MLB Network's
Jon Morosi, the Braves are "engaged" in talks with Ohtani's camp. Imagine a lineup consisting of Ohtani, Ronald Acuna Jr., Ozzie Albies, Austin Riley and Matt
Olson?

Sources: Braves are among the teams still engaged with Shohei Ohtani’s camp, as I reported in this @MLBNetwork segment. @MLB pic.twitter.com/lP4lISzttN — Jon
Morosi (@jonmorosi) December 4, 2023

Korean star Jung Hoo Lee to be posted Monday

Date: Monday, Dec. 4

Source: Jeeho Yoo, Yonhap News

Star Korean outfielder Jung Hoo Lee is set for a move to MLB.
</item 41>

<item 42>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Yes, Ohtani does have magnificent hitting form as well. However, the team already has hitters like Freeman and Betts, who have scored 211 and 179 hits this
regular season, respectively. Speaking of hitters, why will they still want Trout to join the team despite having better-than-average hitters on the team?

Why will the Dodgers sign an injury-prone Trout?

Mike Trout might have been absent for most of the 2023 season, but his career records boast the story of a legendary hitter who has given more than his salt’s
worth to his current team. He has bagged 1624 hits and 940 RBIs with an 85.2 WAR and .301 BA across his 12-year-long baseball career. Unfortunately, Trout has
become a victim of an injury streak, starting in 2022.
</item 42>

<item 43>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Follow for any signings and trades from Nashville.

MLB Winter Meetings rumors, news, signings

Diamondbacks reportedly land Eduardo Rodriguez

Date: Wednesday, Dec. 6

Source: Jeff Passan, ESPN

The reigning NL champions have landed one of the free agent market's top pitchers. Left-hander Eduardo Rodriguez has reportedly agreed to a four-year contract
with Arizona worth around $20 million per year, according to ESPN's Jeff Passan. The deal is still pending a physical, per Passan's report.

A season ago, Rodriguez posted a career-low 3.30 ERA across 152.2 innings with the Tigers, and pitched to a 3.66 FIP. He decided to opt out of his deal with
Detroit to test the free-agent market.

Juan Soto reportedly traded to Yankees

Date: Wednesday, Dec. 6

Source: Joel Sherman, New York Post

The biggest star on the trade market has reportedly been traded.
</item 43>

<item 44>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
While his power was inconsistent between seasons, Lee was a sought-after bat in free agency and drew heavy interest from the Padres as well.

Dodgers meeting with Yamamoto on Tuesday

Date: Tuesday, Dec. 12

Source: Russell Dorsey, Bally Sports

Bally Sports' Russell Dorsey reports the Dodgers met with Yoshinobu Yamamoto on Tuesday, extending Yamamoto's free agent four.

Yamamoto has already met with four teams and still plans to meet with the Red Sox and Blue Jays in the coming days.

Royals finalizing deal with SP Seth Lugo

Date: Tuesday, Dec. 12

Source: Anne Rogers, MLB.com

The Royals and starter Seth Lugo are finalizing a 3-year, $45 million contract, according to MLB.com's Anne Rogers.

The Padres stretched Lugo back into a full-time starter in 2023, and the 34-year-old posted a 3.57 ERA across 146.1 innings.
</item 44>

<item 45>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
'Money's great, but winning's better'

Date: Nov. 21

Source: Max Scherzer, Rangers pitcher on Foul Territory

OK, so this isn't necessarily a rumor, but an example of how much players would love for Ohtani to join their team. Scherzer was a guest on Foul Territory on
Monday.

"Money's great, but winning's better."



Max Scherzer shares his pitch for Shohei Ohtani to join the @Rangers 🏆



▶️ https://t.co/6KoSRqSgk2 pic.twitter.com/PHW2jHWj6C — Foul Territory (@FoulTerritoryTV) November 20, 2023

Maybe Ohtani won’t sign quite so soon after all

Date: Nov.
</item 45>

<item 46>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
The price would only be higher now that he’s a year and a half closer to free agency.

In the absence of an extension or a definitive declaration that Soto is off the trade market, speculation will linger. Alden González of ESPN wrote last week
that many throughout the industry believe the three-time All-Star will be available.

2. Corbin Burnes, SP, Brewers

3. Willy Adames, SS, Brewers

Burnes and Adames are tied together with good reason. They’re each one year from the open market and projected for a noteworthy arbitration salary — $15.1M for
Burnes, $12.4M for Adames. Milwaukee is reportedly open to offers on the majority of the roster.

A former Cy Young winner, Burnes may be the best pitcher available in trade. The right-hander posted a sub-3.00 ERA each season from 2020-22.
</item 46>

<item 47>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec. 5

Source: Bob Nightengale, USA Today

The Cubs have been viewed as one of Ohtani's likeliest destinations for a while. It appears that might no longer be the case. USA Today's Bob Nightengale
reported one high-ranking executive believes the Cubs' optimism of signing Ohtani "has now significantly wanted." The report stated that leaves the Dodgers,
Blue Jays, Angels and Giants as the remaining finalists.

Ohtani has 'positive feeling' about Blue Jays

Date: Tuesday, Dec. 5

Source: Jon Heyman, The New York Post

The Blue Jays reported met with Ohtani on Monday, but even before he went to talk with the team, Ohtani reportedly had a "positive feeling about Toronto," along
with other cities and teams, including the Dodgers. He has been linked to Toronto and Los Angeles throughout the process.
</item 47>

<item 48>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
Davis, Mike Yastrzemski

Guardians: Josh Naylor, Cal Quantrill

Mariners: Ty France

Marlins: Josh Bell

Mets: Jeff McNeil

Nationals: Hunter Harvey, Lane Thomas

Orioles: Anthony Santander

Padres: Scott Barlow

Phillies: Nick Castellanos

Rays: Manuel Margot, Harold Ramírez

Red Sox: Kenley Jansen, Chris Martin, Nick Pivetta

Rockies: Brendan Rodgers

Royals: Salvador Perez

Tigers: Spencer Turnbull

Twins: Kyle Farmer
</item 48>

<item 49>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Morosi reports Shohei Ohtani is "likely to decide on a team within the next week," according to a source.

Status quo on Yankees' Juan Soto trade talks

Date: Sunday, Dec. 3

Source: Jon Heyman, New York Post

The Yankees reportedly are pushing hard to acquire Juan Soto from the Padres, but Heyman says the teams haven't spoken since San Diego requested a six-player
package in return for Soto and Trent Grisham. That ask included pitchers Michael King and Drew Thorpe, who the Yankees aren't inclined to move. Stay tuned...

Shohei Ohtani field growing smaller

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

It has been expected since before he hit free agency that Ohtani would set the record for the largest fully guaranteed contract in North American sports
history, with Passan reporting the deal could reach as much as $600 million.
</item 49>

<item 50>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Roberts said Ohtani is "clearly" the Dodgers' top priority, adding that the team is confident the two-time MVP will pitch again. Roberts wasn't able to offer a
timeline for Ohtani's decision.

Pair of pitchers could be signed after Ohtani deal

Date: Tuesday, Dec. 5

Source: Jon Morosi, MLB Network

MLB is largely in a holding pattern as it awaits news on Ohtani. MLB Network's Jon Morosi reported the Rays could wind up trading Tyler Glasnow shortly after
Ohtani signs, listing the Cubs and Reds as top contenders for the ace. He also reported Yoshinobu Yamamoto could sign following Ohtani's deal, and that the Red
Sox have him among their top priorities.

Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec.
</item 50>

<item 51>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
6

Source: Jon Heyman, New York Post

Corbin Burnes has been named as one of the top trade candidates in the offseason, and it's looking increasingly plausible he might be dealt. NY Post's Jon
Heyman reported one rival believes there's "a decent chance" Burnes is traded this winter. The 2021 NL Cy Young winner is a free agent after the 2024 season,
and could be one of the most expensive pitchers to sign.

Marlins 'listening' to offers on young pitchers

Date: Wednesday, Dec. 6

Source: Jon Heyman, New York Post

Miami made a run to the postseason last year on the strength of its young pitching. And there's a chance the Marlins might now look to part with some of that
depth to improve other areas of need.
</item 51>

<item 52>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Shohei Ohtani and Juan Soto have already found new homes, but much of the MLB offseason has yet to play out.

Japanese pitcher Yoshinobu Yamamoto is the focus of many teams as he prepares to sign a massive deal, while Blake Snell, Cody Bellinger and Jordan Montgomery
remain on the market. The trade market is also expected to be active once more dominoes fall, with Dylan Cease, Corbin Burnes and Tyler Glasnow among starters
who could be moved this winter.

The Winter Meetings are over, but because of the timeline of major decisions by Ohtani and Yamamoto, there could be a flurry of activity leading up to the end
of the year.

MORE: Why Shohei Ohtani signed with Dodgers over Blue Jays, other suitors

Sporting News is tracking the rumors, signings and trades as MLB free agency continues. Follow along below for the latest news.
</item 52>

<item 53>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
He has been linked to Toronto and Los Angeles throughout the process.

Mariners considering Cody Bellinger

Date: Tuesday, Dec. 5

Source: Jon Morosi, MLB Network

Following the trade the Mariners made with the Braves, MLB Network's Jon Morosi believes Seattle has cleared payroll space and now has room to make a splash for
top free agent slugger Cody Bellinger. Seattle has Julio Rodriguez in center, but with Teoscar Hernandez still on the market and Jarred Kelenic now dealt, there
is room for him in the outfield.

Blue Jays 'believed' to have met with Shohei Ohtani

Date: Monday, Dec. 4

Source: Ken Rosenthal, The Athletic

While one report earlier Monday indicated Ohtani isn't expected to sign before the end of the Winter Meetings, the process seems to be moving along.
</item 53>

<item 54>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
We have been promised very little of that, though. ESPN’s Jeff Passan reported this: “If visits between Ohtani and a team are reported publicly, it will be held
against the team.”

So, yeah. Probably won’t hear a ton, but that won’t stop the speculation. And it won’t stop us from passing along what precious few nuggets that are scattered
about.

MORE: Way-too-early MLB Power Rankings for 2024

Shohei Ohtani free agent rumors and updates

Ohtani has an 'affinity' for Boston, it seems

Date: Nov. 27

Source: Jeff Passan, ESPN

Hope everyone enjoyed the Thanksgiving holiday. Us baseball writers are pleased that Ohtani didn't decide to sign while the turkey was still warm. This isn't
exactly "breaking news" because rumors have been swirling, but it's an aside from an in-the-know reporter that just might give one fan base a little too much
hope and excitement.
</item 54>

<item 55>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Here's the latest on Ohtani's free agency pursuits.

MORE: Projecting Shohei Ohtani's contract in free agency

Shohei Ohtani news: Latest updates, rumors for MLB free agency

Ohtani agrees to 10-year, $700 million deal with Dodgers

Date: Saturday, Dec. 9

Shohei Ohtani announced Saturday that he has agreed to a contract with the Dodgers. It's a colossal 10-year, $700 million deal with "significant deferrals," so
while Los Angeles won't pay him $70 million per year, the franchise will be paying him beyond those 10 seasons.

While rumors connected Ohtani to the Blue Jays on Friday, it's the Dodgers who hand him a record-shattering contract and pair the two-time MVP with two former
MVPs in Mookie Betts and Freddie Freeman.

Ohtani not headed to Toronto, at his home in Southern California

Date: Friday, Dec.
</item 55>

<item 56>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Kansas City can use plenty of rotation help after a dismal 2023 season, so Lugo figures to be one of the Royals' top arms in 2024 and perhaps beyond.

Yamamoto to meet with Red Sox, Blue Jays

Date: Tuesday, Dec. 12

Source: Mark Feinsand, MLB.com

MLB.com's Mark Feinsand reports Japanese ace Yoshinobu Yamamoto will meet with both the Red Sox and Blue Jays in the coming days.

Yamamoto has already met with the Mets, Giants and Yankees, with the Dodgers also believed to be involved in the bidding. With both meetings still upcoming, the
report is an indication the Yamamoto sweepstakes aren't on the verge of a conclusion.

MORE: Ranking the best fits for Japanese star Yoshinobu Yamamoto

Dodgers formally announce Ohtani signing

Date: Monday, Dec.
</item 56>

<item 57>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
The winter meetings end on Thursday.

Yoshinobu Yamamoto meeting with Yankees, already met with Mets

Date: Wednesday, Dec. 6

Source: Jon Morosi, MLB Network and Will Sammon, The Athletic

The Yankees could be on the verge of adding Soto, but they might not be done. The Yankees are reportedly going to meet with Yoshinobu Yamamoto in New York, the
top pitcher on the market, on Monday. MLB Network's Jon Morosi reported New York is considered to be one of the top teams to sign him, along with the Red Sox,
Giants, Dodgers and Cubs.

That will be the second time Yamamoto has met with a New York team. He reportedly already met with Mets owner Steve Cohen in Japan last week.

'Decent chance' Corbin Burnes is traded this winter

Date: Wednesday, Dec.
</item 57>

<item 58>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
The Padres have reportedly sent Juan Soto to the Yankees in exchange for four pitchers and catcher Kyle Higashioka, according to the New York Post's Joel
Sherman. Soto immediately adds to a potent left-handed bat to a lineup sorely in need of some thump from the left side, and provides some much-needed support to
2022 AL MVP Aaron Judge.

MORE: Full Juan Soto trade details

Craig Kimbrel signs with Orioles

Date: Wednesday, Dec. 6

Source: Ken Rosenthal, The Athletic and Joel Sherman, New York Post

The Orioles are bolstering their bullpen, signing Craig Kimbrel to a one-year, $13 million deal, according to The Athletic's Ken Rosenthal. The NY Post's Joel
Sherman reported he has a $1 million buyout for 2025 and a $13 million club option. The 35-year-old right-hander amassed 23 saves in 69 innings in 2023 with a
3.26 ERA.
</item 58>

<item 59>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
The Padres are scaling back payroll, perhaps as much as $50M. They need multiple starting pitchers and could stand to upgrade at first base and/or designated
hitter.

Trading Soto would allow the Friars to recoup MLB-ready talent while clearing significant spending room for multiple smaller additions. It would also be a
devastating blow to the lineup. Soto overcame a relatively slow start (by his standards) to hit .275/.410/.519 with 35 home runs this year.

The Padres are still motivated to win in the short term. Barring what would be a stunning run at Ohtani, they’re not going to acquire anyone better than Soto.

President of baseball operations A.J. Preller continues to maintain they’ll explore a long-term deal. There’s nothing to suggest an extension is particularly
likely. Soto rejected a $440M offer from the Nationals before he was traded at the 2022 deadline. The price would only be higher now that he’s a year and a half
closer to free agency.
</item 59>

<item 60>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Those two would be the biggest supporting bats in the lineup around Ohtani. However, both are free agents after the 2025 season, and given the added payroll of
signing Ohtani, it's possible Toronto would view keeping the other two phenoms as too pricey.
</item 60>

<item 61>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Verdugo had been the Red Sox's starting outfielder for the last four seasons, coming over as the centerpiece in the trade that sent Mookie Betts to the Dodgers.
Verdugo hit .281 with 43 home runs, 206 RBI and 16 stolen bases over 2,071 plate appearances with Boston. He also led the American League in outfield assists in
2020 and 2023.

In return, Boston will get Weissert and two pitching prospects headlined by Fitts — the Yankees' No. 4-ranked pitching prospect and 12-ranked prospect overall.

Relievers Yates, Devenski get one-year deals for 2024

Date: Tuesday, Dec. 5

Source: Ken Rosenthal, The Athletic and Jeff Passan, ESPN

Veteran relievers Kirby Yates and Chris Devenski know where they'll be pitching in 2024, according to reports Tuesday.
</item 61>

<item 62>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
15. Randy Arozarena, LF, Rays

Arozarena hasn’t been the subject of the same level of trade speculation as Glasnow has. There’s an argument the Rays should be more willing to move the
All-Star outfielder if they’re looking to create payroll room.

Tampa Bay has greater depth in the outfield than they do on their injury-riddled pitching staff. Josh Lowe, Jose Siri, Harold Ramírez, Luke Raley and Manuel
Margot are all on hand. (Margot has come up in trade rumors as well.)

None of those players is as good as Arozarena. Trading star players for younger talent is familiar territory for the Tampa Bay front office though. Arozarena is
projected for a $9M arbitration salary and controllable for three seasons. He’d have significant trade value if the Rays were to consider moving him.

16. Mariners SP

We initially had Bryan Woo in this spot.
</item 62>

<item 63>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Passan reported a trade of Soto appears to be "edging closer to becoming a reality," and while several teams are out on him because of his large salary and
pending free-agent status, there still could be a large market for the star outfielder.

Much of the rest of the names on the trade market are arms. Dylan Cease and Tyler Glasnow are the pitchers most likely to be dealt this offseason, and Cleveland
could also hold a pair of intriguing trade chips in closer Emmanuel Clase and former Cy Young winner Shane Bieber. The Brewers have also begun trade talks for
former Cy Young winner Corbin Burnes.
</item 63>

<item 64>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
4

Source: Ken Rosenthal, The Athletic

While one report earlier Monday indicated Ohtani isn't expected to sign before the end of the Winter Meetings, the process seems to be moving along.

The Athletic's Ken Rosenthal reports Ohtani is "believed" to have met with the Blue Jays at the team's complex in Florida on Monday, far away from the meetings
in Nashville. The Blue Jays have been pursuing Ohtani and Juan Soto in hopes of striking a deal for one of the two stars.

Braves are reportedly 'engaged' with Ohtani's camp

Date: Dec. 4

Source: Jon Morosi, MLB Network

It's not surprising that any and all teams should be interested in Ohtani's services, but imagine the Braves landing the superstar? According to MLB Network's
Jon Morosi, the Braves are "engaged" in talks with Ohtani's camp.
</item 64>

<item 65>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Imagine a lineup consisting of Ohtani, Ronald Acuna Jr., Ozzie Albies, Austin Riley and Matt Olson?

Sources: Braves are among the teams still engaged with Shohei Ohtani’s camp, as I reported in this @MLBNetwork segment. @MLB pic.twitter.com/lP4lISzttN — Jon
Morosi (@jonmorosi) December 4, 2023

Ohtani has already received several $500+ million offers

Date: Dec. 3

Source: Jon Heyman, NY Post

It's no surprise the bidding for Ohtani is heating up, with the star expected to sign soon. And New York Post's Jon Heyman reported Ohtani has already received
several deals that have topped $500 million, which could mean he winds up being the first $600 million player in MLB history. He added there are believed to be
five teams in the running for Ohtani.
</item 65>

<item 66>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
It appears that might no longer be the case. USA Today's Bob Nightengale reported one high-ranking executive believes the Cubs' optimism of signing Ohtani "has
now significantly wanted." The report stated that leaves the Dodgers, Blue Jays, Angels and Giants as the remaining finalists.

Ohtani has 'positive feeling' about Blue Jays

Date: Tuesday, Dec. 5

Source: Jon Heyman, The New York Post

The Blue Jays reported met with Ohtani on Monday, but even before he went to talk with the team, Ohtani reportedly had a "positive feeling about Toronto," along
with other cities and teams, including the Dodgers. He has been linked to Toronto and Los Angeles throughout the process.

Blue Jays 'believed' to have met with Shohei Ohtani

Date: Monday, Dec.
</item 66>

<item 67>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
He added there are believed to be five teams in the running for Ohtani.

Ohtani decision coming this week

Date: Dec. 3

Source: Jon Morosi, MLB Network

With the winter meetings set to begin in Nashville, it seems a resolution is near for Ohtani. MLB Network's Jon Morosi reports the superstar is "likely to
decide on a team within the next week," according to a source.

Ohtani race starting to clear up

Date: Dec. 1

Source: Jeff Passan, ESPN

Two days before the start of the winter meetings, the picture on Ohtani is beginning to clear a bit more. ESPN's Jeff Passan reported the Rangers, Red Sox and
Mets appear to be out of the running for Ohtani, while the Dodgers, Cubs, Blue Jays and Angels remain in the mix. The Giants are considered to be an unknown.
</item 67>

<item 68>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Angels discussing Blake Snell

Date: Tuesday, Dec. 19

Source: Jon Morosi, MLB Network

The Angels came up empty on bringing back Ohtani. Now, they're looking to land another splashy name on the free-agent market. MLB Network's Jon Morosi reported
Los Angeles has had discussions with two-time Cy Young winner Blake Snell.

Per Morosi, the Angels are not believed to be a strong contender for Yamamoto, thus allowing them to put more of an effort on landing Snell rather than waiting
for Yamamoto to make a decision.

Cody Bellinger looking for package starting at $200 million

Date: Tuesday, Dec. 18

Source: Jesse Rogers, ESPN

With Ohtani signed to Los Angeles, Bellinger remains the top free-agent bat on the market. And he is clearly looking for a top-of-the-market contract.
</item 68>

<item 69>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
The Pirates also added another left-hander who pitched in 2023 for an AL West team in Marco Gonzales, though he had an offseason stop in Atlanta in the middle.

Bidding for Yamamoto begins Monday

Date: Monday, Dec. 18

Source: Jeff Passan, ESPN

There has been plenty of speculation about offers sent over to Yamamoto to this point, but Passan reports he has yet to receiver any formal offers. However,
that will start to change on Monday, with the bidding slated to begin.

Braves cut Matt Carpenter after trying to trade him

Date: Monday, Dec. 18

Source: Robert Murray, FanSided

Atlanta reportedly was hoping to trade Matt Carpenter after acquiring him in a deal earlier this offseason from the Padres. The team acquired him and took on $4
million along with trading for Ray Kerr, but was looking to deal away his contract to someone else, according to The Athletic's Ken Rosenthal.
</item 69>

<item 70>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
MLB's winter meetings are always a prime time for moves to happen as many of the top executives, players and agents all come together at the same spot.

Last year's winter meetings wound up being explosive. Aaron Judge, Justin Verlander, Trea Turner and Xander Bogaerts all came away with new contracts.

This year's winter meetings could wind up being similarly busy. There is a growing sense Shohei Ohtani could sign during the winter meetings, and a move of that
magnitude could kick off the next wave of moves. There's also a number of top players on the trade market, including star Padres outfielder Juan Soto.

Starting Sunday, Dec. 3, and running through Wednesday, Dec. 6, expect MLB's offseason to really kick into high gear.

WINTER MEETINGS: Shohei Ohtani rumors, Juan Soto trade destination come to forefront

Sporting News will be tracking all the latest news and rumors from the 2023 Winter Meetings.
</item 70>

<item 71>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Who will hand him that contract has been much more of a mystery. Little has been leaked about the top suitors for Ohtani beyond the usual suspects. Passan
reported that at least three teams initially believed to be in the running — the Mets, Rangers and Red Sox — are believed to be out now. The teams most likely
to still be in the mix are the Dodgers, Cubs, Blue Jays and Angels, with the Giants' status in the race for Ohtani unknown.

Star players could be traded

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

The free agency pool outside of Ohtani, Cody Bellinger, Blake Snell and Yoshinobu Yamamoto is a bit lacking in star power, but the trade market could ensure
several big-name players are on the move this offseason.
</item 71>

<item 72>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
The Hot Stove dominoes have started to fall, but everyone who loves baseball — fans, players, coaches, front-office types and everyone else — is waiting for
free agent Shohei Ohtani’s decision. Where will the global superstar sign his next contract?

He’s played his entire six-year MLB career with the Angels, but his teams haven’t even sniffed the postseason and Ohtani’s made his desire to win crystal clear,
so he won’t be back. Will he chase every dollar possible, going solely by the highest offer? That seems unlikely, too. Remember, Ohtani wants to win.

Lots of teams can meet Ohtani in that middle ground, offering both lots of money — he will absolutely sign the largest contract in baseball history — and lots
of chances to compete for a World Series title (or two or three). It’s a long list.

MORE: Ranking 15 possible destinations for Shoehei Ohtani

So, the baseball world waits for news.
</item 72>

<item 73>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
If another team feels he can hold his own at second base or in center field, he becomes far more desirable.

The Cubs have no urgency to trade him. Morel is controllable through 2028 and won’t reach arbitration for at least another season. Yet president of baseball
operations Jed Hoyer has alluded to the possibility of a deal, saying last week that “another team might be able to put him [at second base],” where the Cubs
have Nico Hoerner.

23. Christian Vázquez, C, Twins

Trading Vázquez is another route Minnesota could consider to clear money. The Twins signed the veteran catcher to a three-year, $30M free agent deal last
offseason. He had a tough first season in Minneapolis, limping to a .223/.280/.318 line over 355 plate appearances — albeit with his typical brand of excellent
defense.

Ryan Jeffers easily surpassed him as the No. 1 option on the depth chart. A $10M per year backup catcher is a luxury for a team trying to trim salary.
</item 73>

<item 74>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
18. Paul Blackburn, SP, A’s

Blackburn is arguably the most desirable realistic trade candidate on the Oakland roster. The A’s have torn things to the studs. They’re unlikely to compete
within the two remaining years of Blackburn’s arbitration window. The right-hander (30 next month) worked to a 4.43 ERA with a decent 22.4% strikeout rate over
103 2/3 innings this past season.

He’s projected for a $3.2M salary. A Blackburn trade isn’t going to be an offseason defining move, but he has a good chance to be flipped to a contender seeking
affordable back-end starting pitching.

19. Alek Manoah, SP, Blue Jays

Perhaps no player had a more disappointing 2023 performance than Manoah. Within one season, he went from Cy Young finalist to essentially unplayable. The
right-hander pitched to a 5.87 ERA over 19 big league outings.
</item 74>

<item 75>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Isn’t it better to have a consistently wounded soldier than risk losing an overperforming one in a game of chances?

ADVERTISEMENT Article continues below this ad

Why would Mike Trout switch to another Los Angeles team?

Trout has been pretty persistent about sticking with the Los Angeles Angels from the very start. The team has been transparent about not stopping him should he
choose to play for another team, but he does not want to do that. However, it seems that the Angels will have a hard time affording a player like Trout in the
foreseeable future, as they have to win a good number of games to stay in the playoffs and keep a steady income of revenue throughout the seasons. Considering
their 73-game win record this season, that seems like a pipe dream right now.

ADVERTISEMENT Article continues below this ad

Trout is 32 years old, which means that he has already hit the MLB average retirement age of 30.
</item 75>

<item 76>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
The Los Angeles Angels have gone through a tough year. After a heartbreaking 2023 season, they now have to cope with the impending exit of their star two-way
star, Shohei Ohtani. The team is also open to trading their nine-time Silver Slugger hitter Mike Trout, should he request it. In a huge miss, a team that
couldn’t do well even with two of the league’s biggest talents on their roster may lose said stars at the same time. Will they choose to stay in LA?

America’s Favorite Video Today

The Halos’ cross-town rivals, the Los Angeles Dodgers, have been eyeing the dynamic Troutani duo for a long time. However, despite fans’ expectations, the
Dodgers’ Ohtani-Trout conquest might not take place—and for good reasons.

Do the Dodgers have the budget to afford Ohtani and Trout on the same roster?
</item 76>

<item 77>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
The Athletic's Ken Rosenthal reported Gurriel has re-signed a contract with the Diamondbacks.

According to ESPN's Jeff Passan, the deal is for $42 million over three years with a club option for a fourth year and an opt out after his second season with
the club.

Yamamoto dines with Mets owner

Date: Sunday, Dec. 17

Source: Joel Sherman, New York Post

The Mets continued their strong push for the top free agent available, Yoshinobu Yamamoto, by hosting the Japanese pitcher for dinner Saturday night at team
owner Steve Cohen's Connecticut home. Cohen and other top Mets officials previously flew to Japan to meet with Yamamoto on his home turf, so this is at least
the second time the key players have met face-to-face.

Sherman reports the bidding process for the 25-year-old "is supposed to intensify this week," with the Yankees, Dodgers, Giants, Red Sox, Phillies and Blue Jays
also in the mix.
</item 77>

<item 78>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
And he is clearly looking for a top-of-the-market contract. Rogers reported sources indicate Bellinger's agent, Scott Boras, is seeking a deal worth well over
$200 million for Bellinger, with one source saying he is "not budging."

Rogers went on to note that some believe the Cubs won't be a team willing to pay a player that type of money until he's 40 years old, while a team like the Blue
Jays might given their desperation to round out the roster.

Padres closing in on contract for Yuki Matsui

Date: Tuesday, Dec. 18

Source: Yuki Yamada, Sankei Sports

The Padres are nearing a deal for one of the top Japanese pitchers on the market. Per Sankei Sports' Yuki Yamada, the Padres are "close to signing" Rakuten
Golden Eagles' reliever Yuki Matsui to a deal.
</item 78>

<item 79>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
8

Source: J.P. Hoornstra, Dodgers Nation

The Blue Jays have reportedly won the sweepstakes for Ohtani. According to Dodgers Nation's J.P. Hoornstra, Ohtani has selected the Blue Jays. The official
details have not been reported, though the report indicated it will shatter Mike Trout's record 12-year, $426.5 million deal.

However, Sportsnet's Ben Nicholson-Smith reported Ohtani has not yet made his decision, and there is not a timeline, though he noted Toronto is a finalist.

Decision from Ohtani 'imminent'

Date: Friday, Dec. 8

Source: Jon Morosi, MLB Network

The wait could soon be over. MLB Network's Jon Morosi reported Ohtani is expected to make his decision soon, reporting it could come as early as Friday.
Coincidentally, Ohtani made his original decision to sign with the Angels on Dec.
</item 79>

<item 80>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Yates, who will be 37 when the season begins, is headed to the Rangers on a one-year deal, per Rosenthal. The right-hander, an All-Star closer for the Padres in
2019, posted a 3.28 ERA in 61 appearances for the Braves in 2023.

Devenski, 33, has agreed to a new one-year deal with the Rays that includes a club option for 2025, according to Passan. Devenski began 2023 with the Angels but
was released in late August and signed with Tampa Bay. He posted a 5.08 ERA in 29 games for the Angels and dropped three full runs off that number in nine games
with the Rays.

Dave Roberts says Dodgers met with Ohtani

Date: Tuesday, Dec. 5

Source: Alden Gonzalez, ESPN

Dodgers manager Dave Roberts told reporters Tuesday that the team met with Ohtani for 2-3 hours over the weekend.
</item 80>

<item 81>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
16. Mariners SP

We initially had Bryan Woo in this spot. It could just as easily have been Bryce Miller or Emerson Hancock, so we’ll cheat and just mention the Seattle rotation
in general.

They’re almost certainly not going to trade George Kirby. It’d take a lot to pry away Logan Gilbert. Moving any of Woo, Miller or Hancock could be more
appealing for a front office that is looking for ways to add long-term offensive upside.

None of that trio has even reached one year of major league service. Hancock only has three big league starts under his belt. Miller and Woo spent more time in
the rotation this year, starting 25 and 18 games respectively. They both turned in mid-rotation results.

Other clubs probably have differing opinions on which right-hander they prefer. (The Mariners likely have their own slight preferences.) They’d all have
significant value if president of baseball operations Jerry Dipoto decided to move one for a bat.

17.
</item 81>

<item 82>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
They’re already light on consistent hitting beyond Aaron Judge. On the surface, trading their second-best hitter seems counterintuitive.

The big question is whether the front office believes Oswald Peraza is capable of stepping into an everyday second base role in 2024. If they feel the
23-year-old is ready, trading Torres to clear spending room while bringing back outfield or pitching help makes sense.

11. Jonathan India, 2B, Reds

The Reds’ infield surplus has fueled trade speculation on India back to the deadline. Cincinnati didn’t seem inclined to make that move midseason, pointing to
India’s role as a clubhouse leader. While that’s certainly still a factor, he didn’t hit well down the stretch and finished the year with a middling
.244/.338/.407 batting line. He’s a well below-average keystone defender.

There’s an argument the Reds are better off relying on an up-the-middle tandem of Matt McLain and either Noelvi Marte or Elly De La Cruz.
</item 82>

<item 83>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
Perhaps the offseason is a better time to consider moving India for a back-end starter.

The former Rookie of the Year is projected for a $3.7M salary and has three seasons of remaining arbitration control.

12. Eloy Jiménez, DH, White Sox

Jiménez no longer looks like a core piece for the White Sox. Various injuries have kept him off the field throughout his big league tenure. He only has two
seasons of 100+ games and hasn’t gotten to 500 plate appearances since his 2019 rookie year.

Jiménez was still an excellent hitter when healthy as recently as 2022, when he ran a .295/.358/.500 line over 84 games. His production fell this year, as he
posted a .272/.317/.441 slash with 18 homers over 489 trips to the dish.

That’s not especially imposing for a player who’s best suited at DH. Yet Jiménez has shown greater offensive upside that isn’t matched by many in this winter’s
free agent class.
</item 83>

<item 84>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Rays, Mariners talking trades in winter meetings

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

The Mariners and Rays have been frequent trade partners in recent years, and there's a chance they could be talking shop again at the winter meetings. MLB
Network's Jon Morosi reported the teams are discussing Rays utility infielder Isaac Paredes, and potentially star left fielder Randy Arozarena, in talks to
start the winter meetings.

Jays unlikely to land both Ohtani, Soto

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

The Blue Jays have been among the most active names in both the Ohtani and Soto markets. And of course, those talks have had fans dreaming of a lineup that
features, Ohtani, Soto, Vladimir Guerrero Jr., Bo Bichette and George Springer.
</item 84>

<item 85>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Yoshinobu Yamamoto meets with Phillies

Date: Thursday, Dec. 14

Source: Jon Heyman, New York Post

The Phillies have not been one of the more frequent names connected to Yamamoto, however the Japanese free agent reportedly met with the team. The starting
pitcher has met with a number of teams, including the Dodgers, Mets, Yankees and Red Sox.

Philadelphia re-signed ace Aaron Nola this offseason, bringing back the powerful duo of Nola and Zack Wheeler. Adding Yamamoto would immediately give the
Phillies' one of the best rotations in baseball.

Cubs, Blue Jays the favorites for Cody Bellinger

Date: Wednesday, Dec. 13

Source: Jon Heyman, New York Post

The Blue Jays and Cubs were believed to be among the favorites to land Ohtani. But with the star two-way talent off to Los Angeles, the two are now believed to
be the favorites to land Cody Bellinger, according to Heyman.
</item 85>

<item 86>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
The right-handed hitter owns a .241/.311/.471 line in just over 850 MLB plate appearances over the past two seasons. When he’s hot, he can carry a lineup.

He also strikes out more than 30% of the time and doesn’t have a defensive fit. Morel has played all three outfield spots and each of second base, third base
and shortstop in the majors. He hasn’t rated well anywhere.

The Cubs used him mostly at designated hitter this year and have suggested they’ll get him first base run in 2024. That’s not a great fit for a plus runner with
top-of-the-scale arm strength, yet Morel hasn’t shown himself capable of handling more important positions on a regular basis.

How teams project Morel defensively could impact both whether he’s traded and what kind of return the Cubs could receive. As a first baseman or DH, he’s a
fringe regular with a statistical profile that’d paint him as a lumbering slugger as opposed to a dynamic athlete.
</item 86>

<item 87>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
The Athletic's Ken Rosenthal reports Ohtani is "believed" to have met with the Blue Jays at the team's complex in Florida on Monday, far away from the meetings
in Nashville. The Blue Jays have been pursuing Ohtani and Juan Soto in hopes of striking a deal for one of the two stars.

Shohei Ohtani, Yoshinobu Yamamoto decision not expected until after winter meetings

Date: Monday, Dec. 4

Source: Jon Heyman, The New York Post

The baseball markets are waiting for the first major domino to fall this offseason in free agency, but it does not appear as though that will happen before the
winter meetings conclude. Jon Heyman reports that neither Shohei Ohtani not Yoshinobu Yamamoto are expected to get a deal done before the final meetings on
Wednesday.

Wade Miley back with Brewers on 1-year deal

Date: Monday, Dec.
</item 87>

<item 88>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
He allowed a 4.58 ERA over 177 innings this past season. That’s nearly two and a half runs higher than his sparkling 2.20 mark of the prior year. As is often
the case, his true talent probably lies somewhere in the middle.

Cease still missed plenty of bats — 27.3% strikeout rate, 13.6% swinging strike percentage — while issuing a few more walks than ideal. His fastball averaged
95.6 mph, down a tick from the preceding season.

The right-hander has shown the ability to pitch at an ace-caliber level. He is two seasons from free agency and projected for an $8.8M arbitration salary,
making him more affordable than the other high-end starting pitchers near the top of this list.

The extra year of control probably gives him more trade value than any of Burnes, Bieber or Glasnow. It also makes him less likely to be traded this offseason.
</item 88>

<item 89>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
At the same time, he’d net a clearer rotation upgrade if St. Louis were to give him up — likely turning the middle infield over to a combination of Tommy Edman,
Nolan Gorman and Masyn Winn in that scenario.

The 26-year-old hit .284/.365/.422 with 11 homers through 371 plate appearances this past season. A flexor tendon injury pushed him to designated hitter and
eventually required surgery that ended his year a couple months early. He’s expected back for spring training.

If healthy, Donovan brings excellent contact skills and a solid plate approach. He’s a multi-positional defender who can play any of the corners in addition to
second base. Donovan has four years of remaining control and won’t get to arbitration until next offseason.

22. Christopher Morel, DH, Cubs

Morel is likely to be a divisive player around the league. He has huge raw power and connected on 26 home runs while slugging .508 in 107 games for the Cubs.
</item 89>

<item 90>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Pirates acquire Marco Gonzales from Braves

Date: Tuesday, Dec. 5

Source: Jeff Passan, ESPN

Gonzales, who was acquired by the Braves in the trade that helped them land Jarred Kelenic on Sunday, is being re-routed to Pittsburgh for cash and a player to
be named later.

Gonzales, 31, is a decent rotation arm with 65 wins, 665 strikeouts and a 4.14 ERA over 893.0 innings pitched in his nine-year career.

Yankees land Alex Verdugo from Red Sox

Date: Tuesday, Dec. 5

Source: Jeff Passan, ESPN

For just the seventh time in the last 50 years (!), the Red Sox and Yankees have agreed to a deal.

Boston will send starting OF Alex Verdugo to New York for RHP Richard Fitts, RHP Greg Weissert and RHP Nicholas Judice.
</item 90>

<item 91>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Yamamoto requests meeting with Yankees

Date: Sunday, Dec. 17

Source: Jon Heyman, New York Post

Yamamoto has been making the rounds with MLB teams, and per Heyman, he reportedly asked to meet with the Yankees for a second time. The meeting with New York
comes a day after he met with the Mets at team owner Steve Cohen's Connecticut home.

Lourdes Gurriel Jr. re-signs with Diamondbacks

Date: Sunday, Dec. 17

Source: Ken Rosenthal, The Athletic and ESPN's Jeff Passan

The Diamondbacks acquired Lourdes Gurriel Jr. last offseason in a trade involving Daulton Varsho and Gabriel Moreno. And after helping Arizona get to the World
Series, Gurriel is ready to come back to Arizona. The Athletic's Ken Rosenthal reported Gurriel has re-signed a contract with the Diamondbacks.
</item 91>

<item 92>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
3

Source: Susan Slusser, San Francisco Chronicle

The Giants have missed out on big names in the past like Aaron Judge and Carlos Correa, but they are looking to turn that around in 2023 with heavy involvement
and Ohtani, among others. Per San Francisco Chronicle's Susan Slusser, teams believe the Giants "might have an edge" in the pursuit of Japanese ace Yoshinobu
Yamamoto.

Dylan Cease trade market heating up

Date: Sunday, Dec. 3

Source: Jon Morosi, MLB Network

One of the biggest names on the market early is White Sox ace Dylan Cease in trade talks. MLB Network's Jon Morosi reported three teams, the Dodgers, Braves and
Orioles, are among several involved discussing a deal for Cease as the winter meetings begin.

Rays, Mariners talking trades in winter meetings

Date: Sunday, Dec.
</item 92>

<item 93>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
11

Source: Fabian Ardaya, The Athletic

According to The Athletic, Ohtani will defer a whopping $680 million of his $700 million contract from the Dodgers, receiving only $2 million per year.

While it's important to note the deal will still count as $46 million annually toward the luxury tax thresholds for Los Angeles, Ohtani is still giving the
organization much more flexibility than expected over the course of the deal.

The two-time MVP will receive $68 million per year, without interest, from 2034-43, making the present-day value of the contract much less than it originally
appeared.

MORE: Complete breakdown of Shohei Ohtani's Dodgers contract, by the minute

Giants reportedly meet with Yamamoto

Date: Monday, Dec. 11

Source: Buster Olney, ESPN

ESPN's Buster Olney reports the Giants met with Japanese ace Yoshinobu Yamamoto on Sunday, Dec. 10.
</item 93>

<item 94>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
Kepler will be a free agent after next season; Polanco’s contract has a $12.5M team option for 2025 that comes with a $750K buyout.

They’re each good players. Kepler is a plus defender in right field who hit .260/.332/.484 with 24 homers this year. Polanco was limited to 80 games but ran a
.255/.335/.454 showing. He’s one of the sport’s better offensive second basemen and saw some third base action for Minnesota late in the year.

The Twins have Matt Wallner and Trevor Larnach as left-handed hitting corner outfield options. They’re deeper on the infield, where Willi Castro and Kyle Farmer
(another trade candidate) are possibilities at second/third base behind starters Edouard Julien and Royce Lewis.

Top prospect Brooks Lee, the No. 8 overall pick in 2022, isn’t far from MLB readiness. The Twins could look to move Kepler or Polanco for rotation depth or to
bring in a righty bat to balance the lineup.
</item 94>

<item 95>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Juan Soto to Yankees 'close to being finalized'

Date: Wednesday, Dec. 6

Source: Jon Morosi, MLB Network

A trade between the Yankees and Padres for Soto is reportedly "close to being finalized," with right handers Michael King and Drew Thorpe expected to highlight
the return to San Diego. MLB Network's Jon Morosi reported the package will include at least two more players, neither of which will be Alex Verdugo.

Ohtani decision expected to be made before end of weekend

Date: Wednesday, Dec. 6

Source: Jon Morosi, MLB Network

MLB free agency remains largely on hold as teams wait for Ohtani to sign, and though it had at at point been reported he could sign during the winter meetings,
there appears to be a new timeline. MLB Network's Jon Morosi reported Ohtani is expected to make his pick before the end of the upcoming weekend. The winter
meetings end on Thursday.
</item 95>

<item 96>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
Pitchers or Hitters? Who Will the Dodgers Prioritize This Free Agency?

The Dodgers have been in dire need of pitchers for a few seasons. Their current highest strikeout rate by a pitcher in a single postseason is 137 SOs by Clayton
Kershaw, followed by 119 SOs by Bobby Miller and 117 SOs by Julio Urias. With Shohei Ohtani’s high strikeout rate of 167 this season, he’ll be an invaluable
addition to the team. However, there’s a little hiccup that could keep that from happening.

Ohtani will not be available to pitch throughout the 2024 season due to an elbow ligament injury he’s currently recovering from. Given the pitcher’s unavoidable
absence, it will not be advisable for the Dodgers to hire him until 2025. Yes, Ohtani does have magnificent hitting form as well.
</item 96>

<item 97>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Well, another sport-shaking signing. They already poached Craig Counsell from the rival Brewers to be their new manager, and it would be shocking if they didn’t
make a corresponding player-acquisition move. Ohtani’s at the top of the list.

Ohtani could potentially sign before Winter Meetings

Date: Nov. 11

Source: Alden Gonzalez, ESPN

Pretty much everyone in baseball would LOVE for this to happen. The team that winds up signing him, of course, will kick the Ohtani hype machine into gear ASAP.
Not only is there merchandise to sell, but the front office will use that signing to help lure other potential free agents, maybe even for a “play with Ohtani”
discount. But even for the other teams, knowing that Ohtani is off the board relatively soon into the offseason would allow them to shift to other priorities,
like trading for Juan Soto or deciding which free-agent pitchers to pursue with the money that might have been earmarked for Ohtani.
</item 97>

<item 98>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
Bieber, one year from free agency with a projected $12.2M salary, could be next.

Cleveland has the likes of Tanner Bibee, Gavin Williams, Logan Allen, Triston McKenzie and Cal Quantrill (himself a potential trade candidate) as rotation
options. While Bieber is no longer a Cy Young caliber pitcher, he turned in a 3.80 ERA with a below-average but serviceable 20.1% strikeout rate over 21 starts
this past season.

Bieber lost a good chunk of the second half to elbow inflammation but returned to make two appearances in late September.

5. Tyler Glasnow, SP, Rays

The Rays signed Glasnow to an extension in August 2022 when he was completing his rehab from Tommy John surgery. The big right-hander is set to make $25M next
season, the largest single-year salary in franchise history.

That naturally leads to questions about whether the Rays are prepared to meet that figure, particularly in an offseason where they have a loaded arbitration
class.
</item 98>

<item 99>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
It will be interesting to see how the most successful team approaches this. Will they be able to find genuine replacements on the market? Only time will tell.
His trading is not worth it for the Yankees, which will further reduce the options in the outfield. With rookies and the farm not compensating for the demand,
it would be a big ask for them to fulfill. With Jasson Dominguez Anthony Rizzo , and DJ LeMahieu

Watch the Story: Ranking the New York Yankees Biggest World Series Failures
</item 99>

<item 100>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
A $10M per year backup catcher is a luxury for a team trying to trim salary.

Minnesota likely wouldn’t be able to offload the entire deal, but they could try to shed around half the money or take on a smaller contract in return.

Vázquez’s year wasn’t much worse than a season that got Omar Narváez two years and $15M last winter, while Tucker Barnhart secured a two-year, $6.5M guarantee
despite a much lesser offensive track record.

24. Luis Robert Jr., CF, White Sox

Robert has the most trade value of any player on this list. He may also be the least likely to move. Last week, Getz called him a franchise building block and
noted he had no plans to proactively shop Robert even as he left open the possibility of listening to offers on anyone.

It’s not hard to understand why. Robert’s an elite defensive center fielder who connected on 38 home runs while hitting .264/.315/.542.
</item 100>
=== END ITEM id=92849eadc0cc6259 ===
