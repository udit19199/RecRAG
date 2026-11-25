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

This file has 3 items.

=== ITEM id=f4e9feba6bd13a9a ===
Gold sentences and facts:
Sentence 0: Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a return to fitness, and has been included on the squad.
  Fact 0.0: Johnny Cardoso had to pull out of the September camp due to an ankle injury.
  Fact 0.1: Johnny Cardoso has returned to action with Brazilian club Internacional.
  Fact 0.2: Johnny Cardoso's return to action indicates a return to fitness.
  Fact 0.3: Johnny Cardoso has been included on the squad.
Sentence 1: He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.
  Fact 1.0: Johnny Cardoso and his fresh legs are back just in time.
  Fact 1.1: Rookie sensation De'Von Achane hit Injured Reserve this week.
  Fact 1.2: De'Von Achane hit Injured Reserve with a knee injury.

Retrieved items:
<item 1>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
It's a cold, wet, windy night here.

Canada vs Jamaica: Pre-match commentary, analysis, stats, and more

10 min until kickoff: We are edging closer to the start of this match. Can Jamaica overturn a one-goal aggregate deficit on the road?

30 min until kickoff: Jamaica have selected their starting lineup, with Demarai Gray starting as striker in place of the injured Antonio. Leon Bailey and Bobby
De Cordova-Reid start on the wings.

60 min until kickoff: The first XI has been released for Canada. Alphonso Davies and Jonathan David start for the home side.

90 min until kickoff: Jamaica will be without West Ham striker Michail Antonio, who suffered an ACL injury in the first leg.

2 hours until kickoff: Welcome to tonight's coverage of the CONCACAF Nations League quarterfinal between Canada and Jamaica in Toronto.
</item 1>

<item 2>
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
</item 2>

<item 3>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
33 mins: Jamaica 0-0 Canada

Alphonso Davies has a chance to send in a freekick from a useful position not far from the Jamaica penalty area in the right channel.

His low delivery bobbles through almost everyone before the hosts clear.

30 mins: Jamaica 0-0 Canada

The action resumes with a goalkick to Canada. The game has been played at a ponderous pace at times – you sense these teams need to up the ante if a
breakthrough is to come before the interval, although perhaps both are happy to remain cagey during the early stages of the tie.

27 mins: Jamaica 0-0 Canada

Drinks break time. Jamaica coach Heimir Hallgrimsson is wearing a fetching back-to-front white baseball cap as he offers instructions to Trivante Stewart, among
others.

A man in a marvellous yellow wig and black-and-yellow tracksuit can be seen in the stands, which are somewhat sparsely populated.
</item 3>

<item 4>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Canada took a major step towards Copa America 2024 qualification as Stephen Eustaquio's 85th-minute strike gave them victory in their CONCACAF Nations League
quarterfinal first leg and a first win in Jamaica for 35 years.

Porto midfielder Eustaquio slotted in from Richie Laryea's surging run and low delivery, restoring the lead Canada had initially established through Jonathan
David during first-half added time.

Shamar Nicholson fired in Jamaica's equaliser 13 minutes after the break but their opponents will seal their place at Copa America if they avoid defeat in the
second leg at Ontario's BMO Field on Tuesday.
</item 4>

<item 5>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
However, even a moderate MCL sprain could mean a multi-week absence for the injury-prone veteran back, especially in a lost season for the 4-6 Pack. Rookie
Emanuel Wilson also hurt his shoulder in Week 11, so AJ Dillon could be getting the keys to Green Bay's backfield castle (although it's more of an oversized
shack, at this point).

Dolphins rookie De'Von Achane, who looked like the runaway Offensive Rookie of the Year earlier this season, looked to have aggravated his MCL injury upon his
Week 11 return against the Raiders. Achane has a chance to return for Miami's Week 12 tilt with the Jets on Friday, but we wouldn't be surprised if the 7-3
'Fins play it safe and keep him out until he's back to 100 percent. Meanwhile, Miami placed backup Salvon Ahmed on IR with a foot injury, so Jeff Wilson Jr.
could be back in play as a handcuff stash.
</item 5>

<item 6>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
It's possible for them to cancel the first leg and simply play the second leg in Canada, but the teams would have to agree to that. It would put Jamaica at a
serious disadvantage in a matchup with significant ramifications — Copa America qualification — on the line.

1 hour to potential kickoff: There is still no update on whether this game will take place tonight, or if so, what time. At least the players are keeping
themselves loose as they wait for the weather to lighten up!

Keeping warm in the hallway 👊 pic.twitter.com/COjBxNgQ8z — Canada Soccer (@CanadaSoccerEN) November 18, 2023

1 hour 15 mins to potential kick: Uh oh. More rain has struck in Kingston, leaving this match in even further limbo.

1 hour 30 mins to potential kick: No updates have been released since the previous delay. Lineups have yet to be announced with the viability of the match still
in flux.
</item 6>

<item 7>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
player is, as Christian Pulisic once again backed up his nickname of 'Captain America'. The first half stunner was emblematic of his performance, but beyond
that he constantly proved a threat to German's out-of-position right-back Jonathan Tah.

Pulisic completed 27 of 29 passes, a tidy haul for a player of his attacking intent, and created two chances in addition to getting the goal. His corners remain
awful, and it's a wonder why he is still on duty for that role, but otherwise Pulisic was a menace.

Christian Pulisic with an absolute banger 🚀



Watch USA vs.
</item 7>

<item 8>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Lineups have yet to be announced with the viability of the match still in flux. Here's a reminder of what this match means from a tangible competitive
standpoint, whether it should be played tonight or on a future date.

A reminder of what's on the line for #CANMNT tonight! ⬇️pic.twitter.com/sVfKRZHXMl — Canada Soccer (@CanadaSoccerEN) November 17, 2023

50 mins 1 hour 50 mins to potential kick: **Kickoff has been pushed back another hour**

This was the update we expected, as Canada Soccer have announced that the match will not kick off before at least 9 p.m. ET. However, it does appear more and
more likely that a game will indeed take place tonight. Will it clear up enough to create a playable surface, or has there been too much rain to proceed safely?
We'll find out soon.

Update from Concacaf: KO time has been pushed back another hour.
</item 8>

<item 9>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Cardinals at Steelers (-5.5)

Despite missing the entire week of practice, the Cardinals are holding out hope for Brown to play by listing him as questionable. McBride is also questionable
after being limited on Thursday and Friday, but told reporters that he's "good to go."

Pittsburgh has listed Adams as its lone player with a game designation to wrap up the week. Safety Minkah Fitzpatrick (hamstring) practiced fully throughout the
week, carries no designation, and told reporters Friday that he will play this weekend.

Colts (-1) at Titans

Colts : CB JuJu Brents (quad), RB Jonathan Taylor (thumb) OUT

: CB JuJu Brents (quad), RB Jonathan Taylor (thumb) OUT Titans: WR Treylon Burks (concussion) QUESTIONABLE

Indianapolis ruled out Brents and Taylor after they both missed the entire week of practice.
</item 9>

<item 10>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
24 mins: Jamaica 0-0 Canada

Substitute Trivante Stewart turns and tries his luck from distance for Jamaica, to no avail.

These sides are yet to find their range yet.

21 mins: Jamaica 0-0 Canada

Ismael Kone sends a fierce shot from outside the penalty area wide for Canada.

Trivante Stewart has replaced Jamaica's Michail Antonio, who limped off a few minutes ago.

18 mins: Jamaica 0-0 Canada

Bad news for West Ham United fans: Michail Antonio is down for an extended period of time and asks to be substituted.

He eventually rises to his feet and hobbles down the tunnel, accompanied by medical staff. Speculatively, that could be a hamstring issue.

15 mins: Jamaica 0-0 Canada

Leon Bailey should have scored for Jamaica again! An aerial ball down the flank bounces kindly for the Aston Villa attacker inside the penalty area on the left,
deceiving Steven Vitoria.
</item 10>

<item 11>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
That would allow them to rotate Kamal Miller out, who picked up a yellow card in the first leg, keeping him safe for the semifinals.

In would step Alistair Johnston and Derek Cornelius, with Steven Vitoria retained alongside them.

The only injury absence for Canada at the moment is striker Lucas Cavallini, who withdrew from the squad prior to the two matches, replaced by Jacen
Russell-Rowe.

Canada projected starting lineup (3-5-2): Borjan (GK) — Cornelius, Vitoria, Johnston — Laryea, Eustaquio, Kone, Kaye, Davies — David, Larin.

Jamaica saw star striker Michail Antonio go down with a gutting ACL injury in the first leg, and he now begins a lengthy rehab process. West Ham are reportedly
furious about how it was handled, angered by both the conditions of the pitch in Jamaica and the fact he played another 10 minutes after sustaining the injury.
</item 11>

<item 12>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
I'd rather start both Colts RBs, Javonte Williams and Chuba Hubbard if Miles Sanders is out.

LAST WEEK: After James Conner left with an injury, Demercado played 100% of the Cardinals snaps. They could have used veteran RB Corey Clement to help him out
but kept him on special-teams duty. And Demercado actually played fairly well, flashing solid speed and physicality on his runs with some patience blended in.
He's not as big as Conner but he played kind of like him.

THIS WEEK: Arizona released a depth chart with second-year back Keaontay Ingram as the first stringer. It is expected that Ingram will work with Demercado in
the Cardinals backfield this week.

INGRAM: Through 15 regular-season games over two years, Ingram has averaged 1.9 rush yards per carry. His career-long run is 13 yards.
</item 12>

<item 13>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
local (7:30 p.m ET / 4:30 p.m. PT)

Location: BMO Field (Toronto, Canada)

Referee: Cesar Ramos (MEX)

Starting lineups:

Canada (4-4-2): Borjan - Laryea, Miller, Cornelius, Johnston (Millar 84') - Davies, Eustaquio (Russell-Rowe 89'), Kone (Kaye 74'), Buchanan (Hoilett 74') -
Larin (Osorio 74'), David

Jamaica (4-2-3-1): Blake - Leigh (Gray 94'), Pinnock, Lowe, Lembikisa - Johnson, Latibeaudiere (Hector 46') - Bailey (Bernard 90'), Nicholson, De Cordova-Reid -
Gray

It looked like it was going to be a comfortable night for Canada when Davies opened the scoring on 25 minutes after a lung-bursting run.
</item 13>

<item 14>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
In addition, K.J. Osborn (concussion) is also listed as "questionable," though his status steadily improved throughout the week from DNP on Wednesday to full
practice on Friday. If Jefferson is back, he's a must-start. If Jefferson is out but Osborn returns, both Osborn and Jordan Addison are worth using.

WEEK 10 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is Ja'Marr Chase playing Week 10?

UPDATE: ESPN's Jeremy Fowler reported late Saturday that the Bengals are "hopeful" that Chase will play against Houston, but they won't know for sure until
after a pre-game workout.

UPDATE 2: Chase is officially ACTIVE.

Chase (back) appears ready to take his status down to kickoff. According to The Athletic's Dianna Russini, the Bengals' star receiver will work out on Sunday
morning and his status will be determined afterward.
</item 14>

<item 15>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
8

Source: Bob Nightengale, USA Today; Ben Nicholson-Smith, Sportsnet

Ohtani had a change of plans, apparently. Hours after MLB Network's Jon Morosi reported that the two-way star was headed to Toronto for a meeting on Friday, USA
Today's Nightengale and Sportsnet's Nicholson-Smith reported that Ohtani is at his home in Southern California.

The Blue Jays are still finalists to land Ohtani but his landing spot is not official yet.

Ohtani headed to Toronto on Friday

Date: Friday, Dec. 8

Source: Jon Morosi, MLB Network

Ohtani might not have officially picked a team, but he is headed to Toronto, per MLB Network's Jon Morosi. It had been previously reported he had picked the
Blue Jays, though several reports have since denied that he has made a decision.

Report: Ohtani picks Blue Jays over Dodgers

Date: Friday, Dec. 8

Source: J.P.
</item 15>

<item 16>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Sporting News reviews how each USMNT player performed across the 90 minutes.

MORE: Recap the 3-1 defeat for the USMNT against Germany in Hartford

USA player ratings vs Germany

USA starting lineup (4-3-3, right to left): 1. Turner (GK) — 19. Scally, 4. Richards (Carter-Vickers, 66'), 13. Ream, 2. Dest — 7. Reyna (de la Torre, 46'), 6.
Musah, 8. McKennie (Cardoso, 75') — 21. Weah (Aaronson, 66'), 20. Balogun (Pepi, 66'), 10. Pulisic (Paredes, 75').
</item 16>

<item 17>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Nagelsmann's dismissal at Bayern Munich came as a shock to much of the European football landscape, as the club had performed exceptionally in the Champions
League and was still very much in the title race, sitting just a point behind Dortmund. Now, with a chance for redemption at the helm of the national team,
Nagelsmann will see his time begin with a friendly against the United States.

MORE: A look at the 23-man squad for Gregg Berhalter as USMNT play Germany & Ghana

Germany squad, players for USA friendly

Julian Nagelsmann will be very familiar with much of the 26-man German national team squad, as five players hail from his former club Bayern Munich, including
critical midfield duo Josh Kimmich and Leon Goretzka, who were instrumental in his setup at Allianz Arena.
</item 17>

<item 18>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 18>

<item 19>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: Breaking down Germany squad, tactics as Julian Nagelsmann debuts as manager

USA roster for October friendlies vs Germany, Ghana

The USMNT squad for the October friendlies was announced on Thursday, October 5. Gregg Berhalter has called in 23 players, which is one fewer than were brought
in for the friendlies against Uzbekistan and Oman last month.

Below is the official 23-man roster for the upcoming October friendlies against Germany and Ghana.

PSV midfielder Malik Tillman was forced to withdraw from the squad due to injury, with Alex Zendejas of Club America named to the roster in his place.

MORE: Tracking every goal scored by Christian Pulisic at new club AC Milan

Pos Name Club Caps Goals GK Ethan Horvath Nott. Forest (ENG) 9 0 GK Matt Turner Nott. Forest (ENG) 33 0 DEF Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino
Dest PSV Eindhoven
</item 19>

<item 20>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 20>

<item 21>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Hoilett blasts a shot well over the bar from outside the box.

90 mins: Canada 2-3 Jamaica, 4-4 aggregate

There will be nine minutes of added-on time. As it stands Jamaica are through on away goals. What a result this could be.

88 mins: Canada 2-3 Jamaica, 4-4 aggregate

Canada are dominating with their man advantage. David gets a free header in the six-yard box but it balloons over the bar. That was a decent chance.

84 mins: RED CARD (Demarai Gray) Canada 2-3 Jamaica, 4-4 aggregate

What a brain-fade by Jamaica's Gray. He gets two yellow cards in a minute and is dismissed. He booted the ball away for no reason. Clear second yellow card. Can
they hold on now?

83 mins: Canada 2-3 Jamaica, 4-4 aggregate

What an incredible result this would be for Jamaica.
</item 21>

<item 22>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
At the same time, he’d net a clearer rotation upgrade if St. Louis were to give him up — likely turning the middle infield over to a combination of Tommy Edman,
Nolan Gorman and Masyn Winn in that scenario.

The 26-year-old hit .284/.365/.422 with 11 homers through 371 plate appearances this past season. A flexor tendon injury pushed him to designated hitter and
eventually required surgery that ended his year a couple months early. He’s expected back for spring training.

If healthy, Donovan brings excellent contact skills and a solid plate approach. He’s a multi-positional defender who can play any of the corners in addition to
second base. Donovan has four years of remaining control and won’t get to arbitration until next offseason.

22. Christopher Morel, DH, Cubs

Morel is likely to be a divisive player around the league. He has huge raw power and connected on 26 home runs while slugging .508 in 107 games for the Cubs.
</item 22>

<item 23>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Projected stats for all starting quarterbacks are available at Sportsline.
</item 23>

<item 24>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Jamaica lineup (4-4-2, right to left): Blake (GK) – Leigh, Pinnock, Bernard, Lembikisa – Decordova-Reid, Latibeaudiere, Johnson, Bailey – Antonio, Nicholson

Attacking midfielder Liam Millar was out with a calf injury for Canada, whose interim head coach Biello says his players are on a "mission" to progress.

"We know that these matches ​will be​ against a solid opponent in difficult conditions," Biello added. "That is why ​ we decided on an experienced and seasoned
group of players who are used to high-pressure games.”
</item 24>

<item 25>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Even if one can argue he has the skill set to succeed at the No. 6, he did not appear comfortable transitioning on short notice.

MORE: USA vs. Germany player ratings

“Tyler’s a great player. It’s not nice to talk about guys that aren't here, though,” Berhalter told reporters afterward. “I’d rather talk about the guys that
are on the field.”

He didn’t, though, relative to how this group filled Adams’ huge vacancy. He was pleased to talk about the pressure the USMNT applied in the first half,
including a ball played over the top of Germany’s defense to Pulisic early in the game that was stopped by an offside call both player and coach disputed, and
Pulisic’s brilliant goal in the 27th minute that involved him driving to his right and blasting a curving shot while four defenders converged and ace goalkeeper
Marc-Andre Ter Stegen watched it sail into the upper righthand corner.
</item 25>

<item 26>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
ET. We continue to stand by for more official updates from Canada Soccer.

If played, the rain-soaked conditions could rival those of the match Canada played in Honduras in the summer of 2022, which was wet and wild.

50 mins 1 hour 50 mins to potential kick: ** Tonight's match will not kick off on time**

Our first true update from Canada Soccer concerning tonight's match states that kickoff has been pushed back at least an hour. The statement claims that the
match will not kickoff before 8 p.m. ET, language which seems to leave the possibility open for further postponements or even calling off the game. We will keep
you updated as we get more information.

UPDATE: Tonight's #CanMNT vs. Jamaica @CNationsLeague Match will not kickoff before 8PM ET due to "on-field issues"
</item 26>

<item 27>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 27>

<item 28>
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
</item 28>

<item 29>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
“I will commit to being the catalyst that will set the tone for our group for every game, to help restore our club to where it deserves to be, I am confident of
that. We will be back and better.”#NERevs pic.twitter.com/A0mVyGdaAZ — Seth (@SethMan31) November 10, 2023

Jamaica vs Canada lineups & team news

Jamaica manager Hallgrimsson also described his side as being on a mission, but they faced it without New York Red Bulls forward Cory Burke because of a groin
problem.

Karoy Anderson, who plays for English club Charlton Athletic, was called up for the first time. "I was just shocked," the midfielder told the Addicks' website.
"I was looking at it for a long time just taking it in and I told my mum and she started crying."
</item 29>

<item 30>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 30>

<item 31>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Cheering scenes.

The official attendance is a little more than 3,500.

79 mins: Jamaica 1-1 Canada

Alphonso Davies shows some invention for Canada on the left, but the ball ends up in Andre Blake's arms.

76 mins: Jamaica 1-1 Canada

Leon Bailey is booked for an altercation with his marker. It's been a while since we've had a clear chance in this game – settle for a draw all-round?

73 mins: Jamaica 1-1 Canada

It's drinks break time. Canada emerged the stronger following the first-half drinks break...

70 mins: Jamaica 1-1 Canada

Canada have never lost at home to Jamaica, so you suspect they'll be satisfied with this result. Their opponents have shown enough to suggest they could spring
a surprise in the return game, though.

67 mins: Jamaica 1-1 Canada

A Canada substitution we could have mentioned...

SUBSTITUTION 🔁
</item 31>

<item 32>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Stroud in the league's concussion protocol and Nico Collins (calf) joining Tank Dell (fibula) on the Texans' injury list, Houston suddenly finds itself without
many impact skill-position players. Singletary, however, has been pretty damn good and showed us last week that he's the main man over Dameon Pierce in DeMeco
Ryans' backfield. He had 13 carries for 65 yards and a TD against the Jets, while Pierce totaled just five touches for seven yards. Don't be scared of
Tennessee's run D because it ain't what it used to be under Mike Vrabel. Raheem Mostert and De'Von Achane just posted 35.1 PPR points against the Titans, Chuba
Hubbard had 20.2 a few weeks ago, and Rachaad White had 17.8 against them in Week 10.
</item 32>

<item 33>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He was constantly unsure of where he should be as Germany built forward with incisive pace, and he was one of many U.S. players caught unaware on the first-half
Ilkay Gundogan goal.

McKennie didn't put a first-half pass wrong, going 18/18 before the break and 28/30 overall, but his positioning was all wrong out of possession. McKennie was a
sieve as the U.S. midfield was invisible defensively, leaving their defenders completely unshielded, particularly at the top of the penalty area.

It was only a matter of time before the floodgates opened, and that they did after halftime. McKennie was eventually withdrawn in what felt like one of Gregg
Berhalter's only tactical substitutions of the match.
</item 33>

<item 34>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: All players in U.S. national team pool currently playing for a European club

Gio Reyna to make USMNT return under Gregg Berhalter

The ability of Berhalter and Reyna to work together moving forward has been a major talking point since the fallout of their saga after the 2022 World Cup.

Reyna's lack of playing time at the World Cup was exacerbated by the actions of his parents in the weeks that followed, which sparked a domestic violence
investigation of Berhalter before his eventual re-hire as national team head coach.

Now, having recovered from a leg fracture suffered in summer Nations League duty under an interim USMNT head coach, the 20-year-old Borussia Dortmund playmaker
returns to the international fold under Berhalter for the first time.

The USMNT head coach has been forthcoming about the situation, admitting upon his re-hire that the two have a "lot of work to do" but insisting time was needed
to broach the situation.
</item 34>

<item 35>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
RENTSCHLER FIELD, HARTFORD — The United States appeared well positioned to fight with a European power through 45 minutes, but the house of cards came crashing
down after the break as the U.S. men's national team succumbed to a 3-1 defeat to Germany in Hartford.

Presented with a chance to test themselves against a world-class opponent on home soil, the U.S. entered the break level as a Christian Pulisic stunner was
cancelled out by Ilkay Gundogan's poaching.

Yet after halftime, as Gio Reyna was withdrawn from his first start back under Gregg Berhalter to build his fitness, the U.S. midfield was overrun and the
defense committed a multitude of errors to allow the Germans to gain full control of proceedings.

"It’s frustrating because its just little moments," said U.S. goalkeeper Matt Turner after the match. "Little moments made a big difference at the World Cup,
and it’s kinda the same story here.”
</item 35>

<item 36>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Along the back, Nagelsmann has oddly selected five center-backs compared to just two natural full-backs in David Raum and Robin Gosens, both left-sided players,
possibly indicating a desire to play a three-CB system, although it's hard to know for sure as Nagelsmann preferred a 4-2-3-1 for much of his club managerial
career.

One of the center-backs present in the squad is Mats Hummels, who has not been called in since 2021 and dropped for the entirety of Hansi Flick's time in
charge. This came at the expense of Hummels' Borussia Dortmund teammate Nico Schlotterbeck.

First national team call-ups were handed to Kevin Behrens of Union Berlin, Chris Fuhrich of Stuttgart, and Robert Andrich of Bayer Leverkusen.
</item 36>

<item 37>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
.@pulisic with a SCREAMER to open the scoring 😱 pic.twitter.com/KlG0oMeNPm — U.S. Men's National Soccer Team (@USMNT) October 14, 2023

“I think we created some problems,” Berhalter said. “And then just the threat we had a lot in the first half was really good. It put Germany on their heels at
times. But to me, it’s also about controlling the tempo of the game, and that’s what we really need to learn.”

It was a victory of sorts for Reyna to play 45 minutes after he’d gone only 27 in his one club appearance this season for Borussia Dortmund. He’d been injured
in June while playing for the U.S. and missed all of preseason training and nearly two months of the Bundesliga season.

“The goal is to send him back to Dortmund in a really good spot,” Berhalter said.
</item 37>

<item 38>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
It appeared as if Canada had weathered the storm until first-leg hero Stephen Eustaquio's handball in the box allowed De Cordova-Reid to score the winning
penalty with 12 minutes of regulation to play.

Jamaica managed to hold on through nine minutes of stoppage time, despite Gray being given a second yellow card for dissent on 84 minutes.

Canada vs Jamaica live updates, highlights, commentary

Fulltime: Canada 2-3 Jamaica, 4-4 aggregate

Jamaica's first-ever win on Canadian soil sees them advance by away goals to the CONCACAF Nations League semi-final and Copa America 2024. Extraordinary.

95 mins: Canada 2-3 Jamaica, 4-4 aggregate

Canada doing all the attacking but the Jamaican defence is well-organised. Hoilett blasts a shot well over the bar from outside the box.
</item 38>

<item 39>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.

DOLPHINS: Aren't quite the bastion of rushing that some headlines have made them out to be, but they're just over league average in rush rate at 43.7% and above
league average in RB runs per game at 22.6. All of these numbers are weighted heavily by their 50-point win over Denver in Week 3 when they ran the ball 42
times with their running backs. Their 7.6 yard per carry average is due to Achane's breakout start more than anything else.
</item 39>

<item 40>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 40>

<item 41>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
FC Cincinnati are otherwise light on availability concerns, with only a leg injury for Arquimides Ordonez and a long-term injury to young midfielder Stiven
Jimenez to worry about.

Talisman forward Luciano Acosta was partially rested midweek, coming off the bench to score the team's only goal against the Red Bulls in 32 minutes of play, so
he is ready to go as he chases the league's Golden Boot award. Also rested midweek was Alvaro Barreal who leads the team in minutes and also logged just 32 of
them on Wednesday.

FC Cincinnati starting lineup (5-3-2): Celentano (GK) — Arias, Hagglund, Miazga, Murphy, Barreal — Nwobodo, Acosta, Moreno — Vazquez, Santos.

FC Cincinnati subs (9): Kann (GK), Kubo, Mosquera, Angulo, Badji, Pinto, Halsey, Valenzuela, Boupendza.
</item 41>

<item 42>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.
</item 42>

<item 43>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 43>

<item 44>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Christian Pulisic with an absolute banger 🚀



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/G7t6KxLjFO — B/R Football (@brfootball) October 14, 2023

Player ratings for USMNT substitutes vs Germany

45th min: Luca de la Torre — 5

Brought on to replace Gio Reyna at halftime, Luca de la Torre was unable to stem the German tide, despite being a more defensive-minded option than Reyna in
what should have been a swap to shore up the midfield.

65th min: Cameron Carter-Vickers — 7

With Tim Ream and even Chris Richards struggling at various stages in this match, Cameron Carter-Vickers remains a top-tier center-back in this U.S. player pool
and a clear starter when healthy. He came off the bench due to his recent hamstring troubles, but was stout in his 25 minutes of action.
</item 44>

<item 45>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
The Canada goalkeeper was clutching his left calf.

Shamar Nicholson, who appeared to be the possible culprit, exchanges a hug with Borjan after he rises to his feet but is still booked for his trouble.

39 mins: Jamaica 0-0 Canada

Leon Bailey hits the crossbar! Bobby De Cordova-Reid sends a corner towards the far post, where Bailey beats his man with his run and sends his low first-time
strike against the near post and away.

Jamaica could not have come any closer!

36 mins: Jamaica 0-0 Canada

Canada have had the better of the action since the drinks break. Alphonso Davies fails to beat the first defender with a corner from the right, but Canada find
him in a good position on the same flank again, from where he whips a cross towards Jonathan David at the far post.

The forward loops a header back in the opposite direction, and the backtracking Andre Blake stretches to claim with reasonable ease.
</item 45>

<item 46>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 46>

<item 47>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
6, especially as Ilkay Gundogan enters the twilight of his career. The new boss will also want to raise Kimmich's level of play back to old heights, as the
ever-present Bayern star has seen his level drop in recent years for both club and country but remains a special talent.

Germany criticized for travel schedule vs USA, Mexico

The DFB has taken significant flak from the German press for their travel to North America to play the United States and Mexico.

There's likely some form of superiority complex involved, with those following the German national team surely miffed that a European giant is traveling across
the globe to meet CONCACAF sides with far thinner World Cup resumes. However, a closer look at their travel schedule does invite some questions as to how
they've made use of the international break.

Most notably, the Germany squad will return to Europe just a single day prior to Borussia Dortmund's first match out of the break, which sees BVB play Werder
Bremen on Friday, October 20.
</item 47>

<item 48>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Watch:

Jonathan David lights up the scoreboard for Canada! 🇨🇦 pic.twitter.com/bCVlAEQpge — Concacaf Nations League (@CNationsLeague) November 18, 2023

45 mins: Jamaica 0-0 Canada

End-to-end fare. Ethan Pinnock stops Jonathan David from getting a shot away inside the penalty area, and Andre Blake then punches Alphonso Davies' cross clear
from a crowded box.

At the other end, Sam Adekugbe shows strength and speed to stop Trivante Stewart from racing clear onto a long ball.

44 mins: Jamaica 0-0 Canada

Bobby De Cordova-Reid almost set up an opener for Leon Bailey five minutes ago. This time he takes aim himself from a long way out, zipping a shot into Milan
Borjan's arms.

41 mins: Jamaica 0-0 Canada

Milan Borjan is down for an extended period of treatment.
</item 48>

<item 49>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 49>

<item 50>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
All patrons who bought tickets will be allowed to enter the Grandstand only. Those with scanned tickets will be given appropriate credit to enter.⚽️🇯🇲
pic.twitter.com/Duc0R5tnn4 — Official J.F.F (@jff_football) November 18, 2023

Two hours to kickoff: Hello and welcome to our coverage of this rescheduled game between Jamaica and Canada! Stick with us - heavy rain permitting - for live
updates as the match finally takes place.

UPDATE: Concacaf has rescheduled #CANMNT’s @CNationsLeague match due to adverse weather conditions and it will be played at 10:30AM ET tomorrow morning.
pic.twitter.com/JsTuer642p — Canada Soccer (@CanadaSoccerEN) November 18, 2023

UPDATE (8:35 p.m. ET): Canada Soccer has made the postponement official, as the match will kick at 10:30 a.m.
</item 50>

<item 51>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
We need a full stadium there to give us that support." #CANMNT pic.twitter.com/lwuD0qLvp2 — Canada Soccer (@CanadaSoccerEN) November 18, 2023

The Sporting News followed the rescheduled Jamaica vs Canada match live, providing updates below

Jamaica vs Canada updates from rescheduled CONCACAF Nations League match

Fulltime: Jamaica 1-2 Canada

Canada will reach Copa America 2024 if they preserve their unbeaten record at home to Jamaica on Wednesday!

The hosts are a little unfortunate, having scored an equaliser and held out until five minutes from time, but they now face a huge task to reach the semifinals.

90 mins: Jamaica 1-2 Canada

Good news for Jamaica: there will be six minutes of added time.

87 mins: Jamaica 1-2 Canada

It's been 35 years since Canada have won in Jamaica.
</item 51>

<item 52>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
While Christian Pulisic starred on the left, Tim Weah was equally devastating down the right, dazzling with electric pace and good link-up.

Weah's biggest moment came when he scorched Robin Gosens on the wing in the first half, knocking it past the German defender and then beating him with
blistering speed to get back on the ball. Weah's statistical haul don't quite do him justice, but he was a threat throughout his 66 minutes of action.

ST: Folarin Balogun — 7

U.S. fans are salivating for Folarin Balogun to get more touches up front, and as Gregg Berhalter foreshadowed pre-match, they made a clear attempt to do that.
Gio Reyna and Tim Weah did the most of the work in that regard,

LW: Christian Pulisic — 9

There's no doubting who remains the most electric and incisive U.S.
</item 52>

<item 53>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Dest was lively up front as he combined with Christian Pulisic and worked very well on the ball. He got Mats Hummels booked on a counter-attack and delivered a
few very quality crosses.

Dest was also guilty of some of the game's worst defensive errors. He was completely non-existent as Leroy Sane blazed by him on the first Germany goal, and he
nearly got his team in trouble again right before halftime by holding onto the ball too long after making an interception rather than just clearing it out of
danger. Filling in for Antonee Robinson at left-back rather than his natural right, Germany targeted him on multiple occasions, to great profit.

He had a few good moments guarding Leroy Sane in the second half, but he was a top culprit for Germany's second goal by playing Niklas Fullkrug onside. If the
U.S.
</item 53>

<item 54>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The third German goal was emblematic of this lack of chemistry between the two U.S. center-backs: Richards stepped up to guard a free Niklas Fullkrug who got by
Sergino Dest, and Ream was unaware as that left Leroy Sane free on goal himself. These two just didn't have it, and Ream was powerless to pick up the slack.

Jamal Musiala dances through the USA defense and Germany lead 3-1 😮‍💨



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/PdjsjkObZR — B/R Football (@brfootball) October 14, 2023

LB: Sergino Dest — 4

If it were possible to give Sergino Dest two different scores, one for the attacking half and one for the defensive half, we'd do it, but that's not how this
works.
</item 54>

<item 55>
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
</item 55>

<item 56>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
ET (7:30 a.m. PT) tomorrow morning .

Nothing else was announced as far as what should happen if the pitch remains water logged. That means one of two things: either the federations haven't sorted
that out yet, or they are looking to keep that in-house as not to risk publicly announce plans for something that never ends up happening.

Here's to hoping that won't be necessary anyways, and tomorrow morning's kickoff can proceed as planned!

UPDATE (8:25 p.m. ET): **GAME POSTPONED UNTIL TOMORROW**

The match has been called off for the evening, and they will try again tomorrow. Journalists have announced that the match has been pushed back until 10:30 a.m.
ET / 7:30 a.m. PT. No official word from the federation yet, but this seems as concrete as ever.

They tried to wait it out tonight, but it just wasn't to be.
</item 56>

<item 57>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 57>

<item 58>
49ers cap space 2024: Why Chase Young trade may not be a pure rental for San Francisco (Sporting News, 2023-10-31T22:22:43+00:00)
The 49ers made the biggest move ahead of the 2022 NFL trade deadline when they acquired running back Christian McCaffrey from the Panthers. They once again made
a splash ahead of the 2023 trade deadline when they traded for Commanders pass rusher Chase Young.

Young was among the most notable players dealt ahead of the 2023 trade deadline. The No. 2 pick in the 2020 NFL Draft is in the midst of a breakout season,
totaling five sacks in eight games and generating a solid 18 pressures while getting into the backfield often for the Commanders.

But as Washington jettisoned its top two defensive ends, Young and Montez Sweat, the 49ers found themselves positioned to pounce. They traded a 2024 third-round
pick to get the talented defender and reunite him with his college teammate and fellow No. 2 pick, Nick Bosa.
</item 58>

<item 59>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Achane is clearly far ahead of Mostert.

LAST WEEK: Before he fumbled the ball away twice, Mostert was splitting reps with De'Von Achane in a way that was unfavorable for him. Before the Dolphins got
inside the 10-yard line Mostert had a 16-to-9 edge, then Achane played every single snap inside the 10. Then after Mostert's second fumble of the half, Mostert
played just nine of the remaining 35 snaps including nothing near the end zone. Coach Mike McDaniel preached the importance of his offense not having
"self-inflicted wounds" immediately after the game.

LAST TWO WEEKS: Achane played all but two snaps inside the 10-yard line for Miami, so that seems like a big part of his role that will stick. As for third
downs, Mostert has a one-snap edge over Achane at 6 to 5, which means they're splitting those valuable touches.
</item 59>

<item 60>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
The Bayern star saw his shot saved onto the crossbar before the ball spun in on the goal line.

This gave Canada a 3-1 aggregate lead and if it wasn't for Jamaican stopper Andre Blake, there could have easily been a couple more goals, making two great
saves from Cyle Larin prior to Davies' opener.

However, the home side couldn't get the second goal and Jamaica grew in confidence, particularly after half-time.

In three second-half minutes, the tie was blown apart by two goals from Shamar Nicholson. A Canadian mistake handed the first to him on a plate before Demarai
Gray's individual brilliance set up the second for a tap-in.

With the scoreline 2-1 on the night, the match was destined for extra time and penalties with a 3-3 aggregate.

But Jamaica could barely contemplate possible glory before Ismael Kone's header made it 2-2 on the night and gave Canada a single-goal aggregate advantage.
</item 60>

<item 61>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Matt Turner says no ❌



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/x2HslCk3s2 — B/R Football (@brfootball) October 14, 2023

RB: Joe Scally — 6

Joe Scally was tidy in possession, and while he offered very little in the attack, that was largely by design as they didn't need him on that right side with
Tim Weah cooking.

Scally defended ok, and wasn't guilty of any big mistakes, which is more than most of the other U.S. defenders can say. It wasn't a standout performance by any
means, and the whole back line has some soul searching to do, but others were far more culpable than the Borussia Monchengladbach man.
</item 61>

<item 62>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 62>

<item 63>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He featured his usual pressing, and came close to scoring once on a messy chance that Germany just barely managed to clear, but otherwise he didn't quite cause
the opposing back line problems to open spaces up for teammates.

75th min: Johnny Cardoso — 6

Johnny Cardoso only had 15 minutes at the end to showcase his abilities, but it begged the question why he or Lennard Maloney weren't the ones introduced at
halftime instead of Luca de la Torre. Cardoso lost possession just once in his 10 touches and completed a tackle, although he was also dribbled past a few times
as well.

75th min: Kevin Paredes — 5

With 10 touches in his 15 minutes of action towards the end of a mucked up game of numerous substitutions, Kevin Paredes didn't stand out, losing possession
four times and completing four passes. Tough to evaluate his time on the field, but he didn't stand out.
</item 63>

<item 64>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Gio Reyna looked sharp in the starting lineup for the United States men’s national team, in all the ways a young man can: his unyielding energy, his surprising
eagerness to cover the whole of the midfield, the periodic flashes of skill when the opportunity presented itself, even the bleach-blond hair that allowed
viewers to spot him in an instant.

There was one substantial element missing from his performance, though, in an immense friendly Saturday against Germany: being able to endure from starting to
finishing.

It was kind of like that for the USMNT as a whole, as they blew an early lead and fell, 3-1, in East Hartford, Conn.

MORE: USMNT undone by errors in loss to Germany

It is quite possible this was the most challenging friendly the U.S.
</item 64>

<item 65>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
CM: Gio Reyna — 7

In his first 45 minutes of international football under Gregg Berhalter since the World Cup fallout, Gio Reyna was his usual self, drawing the attention of
German defenders and proving visionary with his passing. He combined with Folarin Balogun on a number of occasions on the counter, which is a link-up that U.S.
fans have been begging for more of.

Reyna was only fit for 45 minutes of play here as he returns from a leg injury, but it's no coincidence that Germany truly secured midfield dominance when he
came off the field. A promising performance for the young playmaker who should head back to Borussia Dortmund after this international break with plenty of
confidence moving forward.

RW: Tim Weah — 8

Though Germany carved the U.S. defense to pieces, their attacking threat remained strong, and almost all the danger came from the wings.
</item 65>

<item 66>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
German national team squad for USA, Mexico friendlies

Pos Player Club Caps Goals GK Oliver Baumann Hoffenheim 0 0 GK Bernd Leno Fulham (ENG) 9 0 GK Marc-Andre ter Stegen Barcelona (SPA) 36 0 GK Kevin Trapp
Eintracht Frankfurt 7 0 DEF Robin Gosens Union Berlin 18 2 DEF Mats Hummels Borussia Dortmund 76 5 DEF David Raum RB Leipzig 18 0 DEF Antonio Rudiger Real
Madrid 62 2 DEF Niklas Sule Borussia Dortmund 47 1 DEF Jonathan Tah Bayer Leverkusen 17 0 DEF Malick Thiaw AC Milan (ITA) 2 0 MID Robert Andrich Bayer
Leverkusen 0 0 MID Julian Brandt Borussia Dortmund 44 3 MID Chris Fuhrich VfB Stuttgart 0 0 MID Leon Goretzka Bayern Munich 53 14 MID Pascal Gross Brighton
(ENG) 2 0 MID Ilkay Gundogan Barcelona
</item 66>

<item 67>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP vs IND ATL -1 O/U 44.5 OPP VS WR 9th PROJ PTS 10.7 WR RNK 19th YTD Stats REC 58 TAR 86 REYDS 769 TD 2 FPTS/G 11.5 Amari Cooper WR CLE
Cleveland • #2

Age: 29 • Experience: 9 yrs. Weekly Breakdown MATCHUP @ HOU HOU -2.5 O/U 42.5 OPP VS WR 21st PROJ PTS 13.6 WR RNK 20th YTD Stats REC 61 TAR 113 REYDS 985 TD 3
FPTS/G 12.5 Romeo Doubs WR GB Green Bay • #87

Age: 23 • Experience: 2 yrs.
</item 67>

<item 68>
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
</item 68>

<item 69>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 69>

<item 70>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 70>

<item 71>
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
</item 71>

<item 72>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
If the U.S. had profited more from his attacking presence the narrative around his performance might be different, but with the story of the match focused on
the defensive errors, Dest is a guilty party.

CM: Yunus Musah — 6

Another player who did well in possession but struggled defensively, Yunus Musah dazzled at times on the ball but was beaten on the first Germany goal by
whiffing with his slide.

Musah's nutmeg of Florian Wirtz was a highlight moment of the match, and he was active on both ends of the pitch. Yet a few bad giveaways or defensive moments
betrayed his usual stoutness in the middle of the pitch. Overall, the mistakes didn't take away from his performance quite as much as Dest's, but they were
still concerning.

CM: Weston McKennie — 4

Other than a pair of critical blocks right before halftime, Weston McKennie was a mess through the first 45 minutes.
</item 72>

<item 73>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
🔚 pic.twitter.com/jSis5C72tj — Concacaf Nations League (@CNationsLeague) November 18, 2023

How can Canada qualify for the Copa America 2024?

The Rouges' task is now ostensibly straightforward: avoid defeat against Jamaica and they will be competing alongside the likes of reigning champions Argentina
at the Copa America in the US when it starts in June.

Jamaica have never won in Canada, although they showed enough here — in sunshine, mercifully, following the postponement of the game in torrential conditions
the previous night — to suggest they could spring an upset.

Their task will be made harder by the probable absence of Michail Antonio, while Bailey could not match speed and timing with accurate finishes when presented
with good early chances.

West Ham United forward Antonio immediately signalled for treatment before limping down the tunnel with a possible hamstring injury during the first half.

Reaction from Steven Vitoria:
</item 73>

<item 74>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The Texans offense looks like one to avoid as long as C.J. Stroud is out. Ja'Marr Chase WR CIN Cincinnati • #1

Age: 23 • Experience: 3 yrs. Tee Higgins is a No. 2 receiver and Tyler Boyd is a fine flex.

WR Preview Numbers to Know

69.9% -- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets.

-- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets. 11.4 -- Deebo Samuel is
on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons. The other three are
Tyreek Hill, Tyler Lockett, and DeSean Jackson.
</item 74>

<item 75>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 75>

<item 76>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
1 hour to kickoff: With lineups still yet to be released, let's take a look at one player who is, in fact, an MVP candidate for this year.

FC Cincinnati forward Luciano Acosta is one back of Denis Bouanga for the MLS goal scoring lead with two matches left, sitting on 16 for the year. With 13
assists alongside that, his 29 total goal contributions are best in the league, above the 27 of Thiago Almada and 25 of Hany Mukhtar. Acosta ranks as the
second-highest rated player by WhoScored this season across all of Major League Soccer.

You wonder if he's going to be rested tonight with the Supporters' Shield and No. 1 seed locked up. It would be devastating for him to be injured in a
meaningless game, regardless of the opponent.

"Luciano Acosta's the MVP... He's been an unbelievable player for us this year."
</item 76>

<item 77>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
CB: Chris Richards — 4

Statistically, Chris Richards had a pretty good match, missing with just one pass of his 42 attempted and not being beaten at any point by any German attackers.
Yet his lack of volume speaks... well, volumes.

Richards was off the pace for much of his 65 minutes, and failed to get involved as his teammates were beaten repeatedly on the dribble. It was Richards'
positioning that let him down, and he should have been more involved with the U.S. under siege in such an open match.

CB: Tim Ream — 5

Chris Richards often left Tim Ream on an island, and the 36-year-old just could not cope with the waves of German attackers.

The third German goal was emblematic of this lack of chemistry between the two U.S.
</item 77>

<item 78>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Especially when Canada were cruising with a 3-1 aggregate lead until the 62nd minute.

78 mins: GOAL - Canada 2-3 Jamaica (Bobby De Cordova-Reid), 4-4 aggregate

Jamaica are going through as it stands on away goals. The Fulham attacker converts from the spot after a handball from Eustaquio and they are back in front on
the night.

74 mins: Canada 2-2 Jamaica, 4-3 aggregate

Triple change for Canada. Junior Hoilett, Jonathan Osorio and Mark Kaye come on for Cyle Larin, Ismael Larin and Tajon Buchanan.

69 mins: GOAL - Canada 2-2 (Ismael Kone), 4-3 aggregate

Canada strike back. What an absolute rollercoaster. Davies floats in a deft cross and Kone's redirects it over Blake to send the crowd wild.
</item 78>

<item 79>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
To sum it up: oof pic.twitter.com/FRlizvoajM — Alexandre Gangué-Ruzic (@AlexGangueRuzic) November 17, 2023

1 hour 15 mins to kick: The weather in Kingston is torrential, and it's been raining hard for quite some time now. It's at the point where you wonder if this
match will even take place, and there seems to be some initial chatter of it possibly being called off. Postponing international matches is extremely difficult,
but player safety has to be considered.

They will have to make a call soon, as starting lineups are due in 15 minutes.

Okay, remember when we said to bring 10ft x 10ft tarps?
</item 79>

<item 80>
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
</item 80>

<item 81>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Which looks like this 😅👇 pic.twitter.com/iPhqIdvgCo — OneSoccer (@onesoccer) November 17, 2023

1 hour to kickoff: Lineups have yet to be released, and the likelihood of this match being played seems to be thinning by the moment. The Canada Soccer social
media admin just went on Instagram Live to show the conditions at Independence Park, and they are apocalyptic. The individual said on the live social media
broadcast that "it's not looking likely" the game will be played.

The @CanadaSoccerEN admin just went live on Instagram to showcase the state of the pitch ahead of tonight’s #CanMNT match😂
</item 81>

<item 82>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Reaction from Steven Vitoria:



"It feels great to get our first win here in 35 years. It wasn't easy, far from it. Still a lot to be done but we're one step closer to where we want to
go."#CANMNT pic.twitter.com/3YYwSNCgFm — Canada Soccer (@CanadaSoccerEN) November 18, 2023

Lille striker David had two promising early opportunities, poking a shot into Milan Borjan's arms under pressure from two markers and applying a looping header
to an Alphonso Davies cross, forcing the goalkeeper to backpedal and claim.

Even if they are knocked out on Wednesday, there will be another chance for the losers of the tie to reach the Copa America.
</item 82>

<item 83>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Antonio was replaced by Salernitana striker Trivante Stewart, but he struggled and was himself withdrawn in the 81st minute. Demarai Gray could step in, but he
himself is struggling for fitness, and while he remains with the team he was not part of the first leg matchday squad. Another option is Colorado Switchbacks
forward Romario Williams, who scored 15 goals in 29 USL Championship matches this season.

Other injuries for Jamaica to contend with include Cory Burke and Amari'i Bell, who both miss out on this international window.

Jamaica projected starting lineup (4-3-3): Blake (GK) — Leigh, Bernard, Pinnock, Lembikisa — Latibeaudiere, D. Johnson, Bailey — Reid, Nicholson, R. Williams.
</item 83>

<item 84>
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
</item 84>

<item 85>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Under the lights 🏟️ pic.twitter.com/PQjPAvTCMD — Canada Soccer (@CanadaSoccerEN) November 21, 2023

What time does Canada vs Jamaica kick-off?

This CONCACAF Nations League quarterfinal second leg at BMO Stadium kicks off on Tuesday, November 21 at 7:30 p.m. local time in Toronto.

Here's how that time translates across the North American regions:

Kickoff time Eastern Time 7:30 p.m. Central Time 6:30 p.m. Mountain Time 5:30 p.m. Pacific Time 4:30 p.m.

Canada vs Jamaica projected lineups & team news

Canada interim head coach Mauro Biello deployed a 4-4-2 against Jamaica in the first leg on the road, but expect him to bring back their more-utilized
three-defender system.
</item 85>

<item 86>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
He will probably remain sidelined for weeks.

Other than Casemiro, Tyrell Malacia and Amad Diallo are suffering from knee issues, Lisandro Martinez is nursing a foot injury, Jadon Sancho is out due to
disciplinary issues, and Luke Shaw is yet to recover from a muscle problem. Adding to that, Victor Lindelof is suffering from a bug that could see him miss this
weekend’s match.

On the other hand, Raphael Varane who missed last day’s match due to an illness of his own, could be back in this match. Aaron Wan-Bissaka was also able to
participate in the last match coming on as a substitute. Hence, he should be able to make a comeback. Marcus Rashford, Rasmus Hojlund, and Bruno Fernandes all
could see themselves in the starting lineup.

Fulham Team News

Fulham’s injury situation has not improved so far.
</item 86>

<item 87>
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
</item 87>

<item 88>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Jamaica's first-ever victory on Canadian soil has seen the Reggae Boyz fight back to reach the CONCACAF Nations League semifinal and qualify for Copa America
2024.

A 3-2 win in the second leg at BMO Field in Toronto allowed Jamaica to progress on away goals after a 4-4 aggregate tie across the two legs.

After Canada opened the scoring through Alphonso Davies they held a comfortable 3-1 aggregate lead, but that advantage eroded in 15 second-half minutes as
Jamaican striker Shamar Nicholson scored a brace, before Fulham attacker Bobby De Cordova-Reid converted the penalty that sealed progression.

Canada will now have to defeat Trinidad and Tobago in a one-off playoff in March to reach the Copa America 2024 tournament to be hosted in the U.S.

MORE: Highlights, recap of Canada's narrow win in the first leg

The Sporting News followed the Canada vs. Jamaica match live, providing score updates, commentary and highlights as they happened.
</item 88>

<item 89>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
The decision about how to replace Adams in the formation was complex — too much so to be resolved in one afternoon.

Reyna was exceptional throughout the first half: less influential than he might have been if given greater access to the ball but evincing a welcome taste for
covering ground to become helpful on defense or to retrieve the ball and launch an attack. Weston McKennie was his typical self, particularly when he could
connect with star winger Christian Pulisic along the left side.

Young Yunus Musah was deployed in the defensive midfield role, and it did not work. Whenever Germany were able to quell an American surge in the first half and
regain possession, they plowed with ease through through the center of the park. He is elite at maintaining possession when operating as a box-to-box
midfielder, and lately his club play, with Pulisic at AC Milan, has seen Musah in more of an attacking role.
</item 89>

<item 90>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
(@ Browns) even in a tough matchup. Trevor Lawrence and his high-flying offense should dominate the time of possession against the Joe Flacco-led Browns,
leaving ETN plenty of opportunities to add to his per-game PPR average of 17.4 points (third best in the NFL entering Week 13).

Guess who's back? Back again? Achane's back — tell a friend! Stud rookie De'Von Achane (vs. Titans) went right back to roughing up the opposition in his Week 13
return from a knee injury that sidelined him one week prior. The potential league-winner casually put up 103 yards and two TDs on the Commanders, and we can't
help but place him and running mate Raheem Mostert in the top five this week despite a somewhat-tough matchup with Tennessee.
</item 90>

<item 91>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
“I think he played a good game, showed his quality. I think everyone in the first half played well. So it was good to see him, good to have him on the field.”

It was not good to not have him on the field, because the collapse from a 1-1 halftime tie not only followed Reyna’s departure, but probably was connected. The
U.S. was able to threaten the opposing goal enough with Reyna on the field to cause the opposition to concern itself with defending more often than it
preferred. Replacement Luca de la Torre dulled the attack and did nothing to dissuade Germany’s rampage toward the American net.

There was plenty of blame to be shared on the goals by Niclas Fullkrug in the 58th minute and Jamal Musiala three minutes later.
</item 91>

<item 92>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Jamaica match live, providing score updates, commentary and highlights as they happened.

Canada vs. Jamaica result (full-time)

Score Aggregate Goal scorers Canada 2 4 Alphonso Davies (25)

Ismael Kone (69) Jamaica 3 4 Shamar Nicholson (63), (66)

Bobby De Cordova-Reid (78)

Kickoff: 7:30 p.m. local (7:30 p.m ET / 4:30 p.m.
</item 92>

<item 93>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
🎥: @onesoccer pic.twitter.com/XzBldBSXfX — Canada Soccer (@CanadaSoccerEN) November 17, 2023

2 hours to kickoff: While the Canadian federation is reeling from the departure of John Herdman, one player is happy to see change happen at the top. Central
midfielder Mark-Anthony Kaye had been left out of multiple squads towards the end of Herdman's tenure, and he believes his relationship with the former Canada
head coach was the reason.

Me and John had some really tough conversations where he blatantly told me I wasn’t good enough to be part of the team. That was hard for me to accept,” Kaye
said to Sportsnet, after also posting to social media an honest self-reflection following his New England Revolution's elimination from the MLS playoffs a week
ago.

Heartfelt post by Mark-Anthony Kaye.
</item 93>

<item 94>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Celtic defender had 22 touches and gave away possession just two times, completing two dribbles and making two clearances. Carter-Vickers was not a target
for Germany to attack nearly as much as the other three defenders on the pitch, a strong nod to his abilities.

65th min: Brenden Aaronson — 7

Probably the the U.S. substitute to make the biggest impact, Brenden Aaronson was bright in the attack, dazzling with his dribbling skills and helping usher the
U.S. forward amidst a second-half onslaught from Germany. A good showing from the Union Berlin winger who needs confidence at the moment.

65th min: Ricardo Pepi — 5

It wasn't an ideal situation for Ricardo Pepi to enter in, coming on with the U.S. shell-shocked from the quickfire Germany double. Still, he provided very
little up front, with just eight touches in 24 minutes of action.
</item 94>

<item 95>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.
</item 95>

<item 96>
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
</item 96>

<item 97>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week
9 as a big part of the offense with Josh Dobbs under center.

As long as Hockenson is playing, he's not being benched in any fantasy format. The tight end projects as a top option at the position and remains the Vikings'
leader in targets with 80 on the season. Dobbs enjoyed targeting the tight ends in Arizona and continued that trend in his Minnesota debut. The floor for
Hockenson is equivalent to that of a rooftop bar.

WEEK 10 DFS: DK lineup | FD lineup | Best values | Best stacks

Is Curtis Samuel playing Week 10?

UPDATE: Samuel is officially ACTIVE.

Samuel (toe) is expected to play on Sunday when the Commanders take on the Seahawks at 4:25 p.m. ET, according to ESPN's Jeremy Fowler.
</item 97>

<item 98>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
So many national team fans sided with Reyna during the controversy that developed as the team competed at last winter's World Cup. They got what they argued for
in this game, but not what they wished.
</item 98>

<item 99>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
They tried to wait it out tonight, but it just wasn't to be.

When Canada Soccer make the decision official, we will pass that information along as well as any contingency plan made public as well for if the pitch remains
unplayable at the new time.

Jamaica vs. Canada has been rescheduled to 10.30 AM ET tomorrow.#CanMNT — Oliver Platt (@plattoli) November 18, 2023

50 mins to potential kickoff: No word on tonight's match inside an hour until the possible kick time is not good news, as further delay could mean the
federation's officials are hashing out alternate plans.

It sounds like a decision will be made soon on whether tonight's match can go ahead. If it is necessary to push the game back, they would likely try to play
tomorrow, but if the pitch is still unplayable at that point, then further action would need to be taken.
</item 99>

<item 100>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Sergino Dest allowed the ball to deflect off his foot and into dangerous territory at the top of the box, de la Torre became a spectator with a too-great view
and central defender Chris Richards missed two opportunities to shut down the threat. McKennie, central defender Tim Ream and right back Joe Scally all bore
some responsibility for the second.

“When you watch the goals back — I had the opportunity to do that in the coaches’ office — it’s really these fine moments, these split seconds where you need to
be well-positioned, not open up any gaps in your back line,” Berhalter said. “And that’s where it went wrong.”

With the attempt to transform Musah into some semblance of Adams, though, it almost was certain to go wrong. The USMNT conjured a more compelling attack because
of Reyna’s presence, but the ability to control the game went missing.
</item 100>
=== END ITEM id=f4e9feba6bd13a9a ===

=== ITEM id=86bb7f34d9dff74f ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: The parties in question receive other benefits from Android
  Fact 0.1: Google could solve this problem today if it wanted to
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: It will attempt to argue that Google restrains competition within two separate markets
  Fact 1.1: One of the two markets is the distribution of apps to Android users
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle
  Fact 2.1: The case argues that Google siphons off news publishers' content
  Fact 2.2: The case argues that Google siphons off news publishers' readers
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue
  Fact 2.4: The case argues that Google does this through anticompetitive means

Retrieved items:

=== END ITEM id=86bb7f34d9dff74f ===

=== ITEM id=6c461e3e2a7d3d7e ===
Gold sentences and facts:
Sentence 0: Because GPT-4 is more likely to follow the instructions of “jailbreaking” prompts, the co-authors claim that GPT-4 can be more easily prompted than other LLMs to spout toxic, biased text.
  Fact 0.0: GPT-4 is more likely to follow the instructions of 'jailbreaking' prompts.
  Fact 0.1: The co-authors claim that GPT-4 can be more easily prompted than other LLMs to spout toxic, biased text.
Sentence 1: Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.
  Fact 1.0: The entity's earlier actions were focused on concerns about the spread of illegal content.
  Fact 1.1: The entity's earlier actions were focused on concerns about the spread of disinformation related to the Israel-Hamas war.

Retrieved items:
<item 1>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 1>

<item 2>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.
</item 2>

<item 3>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
If they're trailing, JSN should see action with a negative game script.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

TE Brevin Jordan, Texans @ Jets ($3,100). Jordan was a great streaming option and fill-in for the crowd that lost Dalton Schultz (hamstring) on Sunday, hauling
in three-of-four targets for 64 yards against the Broncos. If Schultz is out again, Jordan is a serviceable option that has touchdown upside against a reeling
Jets team that is busy booking vacations or plotting how to find themselves on a different roster in 2024. If Schultz does play, we'll pivot to the Jets' Tyler
Conklin, who is $100 less and has plenty of opportunity in an even better matchup.

FLEX WR Jameson Williams, Lions @ Bears ($3,500).
</item 3>

<item 4>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 4>

<item 5>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week
9 as a big part of the offense with Josh Dobbs under center.

As long as Hockenson is playing, he's not being benched in any fantasy format. The tight end projects as a top option at the position and remains the Vikings'
leader in targets with 80 on the season. Dobbs enjoyed targeting the tight ends in Arizona and continued that trend in his Minnesota debut. The floor for
Hockenson is equivalent to that of a rooftop bar.

WEEK 10 DFS: DK lineup | FD lineup | Best values | Best stacks

Is Curtis Samuel playing Week 10?

UPDATE: Samuel is officially ACTIVE.

Samuel (toe) is expected to play on Sunday when the Commanders take on the Seahawks at 4:25 p.m. ET, according to ESPN's Jeremy Fowler.
</item 5>

<item 6>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 6>

<item 7>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 7>

<item 8>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 8>

<item 9>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
In addition, K.J. Osborn (concussion) is also listed as "questionable," though his status steadily improved throughout the week from DNP on Wednesday to full
practice on Friday. If Jefferson is back, he's a must-start. If Jefferson is out but Osborn returns, both Osborn and Jordan Addison are worth using.

WEEK 10 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is Ja'Marr Chase playing Week 10?

UPDATE: ESPN's Jeremy Fowler reported late Saturday that the Bengals are "hopeful" that Chase will play against Houston, but they won't know for sure until
after a pre-game workout.

UPDATE 2: Chase is officially ACTIVE.

Chase (back) appears ready to take his status down to kickoff. According to The Athletic's Dianna Russini, the Bengals' star receiver will work out on Sunday
morning and his status will be determined afterward.
</item 9>

<item 10>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
It can be tough to trust Taysom Hill (vs. Panthers) in full-point PPR leagues, but a matchup against the league's second-worst run defense gives him a high
ceiling regardless of format. He's been more involved as a receiver this year, so he'll get a few catches to go along with the rushing stats, too. With Derek
Carr (concussion) banged up, Hill might throw more this week, too.

Gerald Everett (vs. Broncos) and Isaiah Likely (vs. Rams) have matchups against bottom-six TE defenses, and Cole Kmet (vs. Lions), Jake Ferguson (vs. Eagles),
and Kyle Pitts (vs. Buccaneers) also have above-average matchups that give them higher floors and higher ceilings than usual.

For those in deeper leagues, Tyler Conklin (vs.
</item 10>
=== END ITEM id=6c461e3e2a7d3d7e ===
