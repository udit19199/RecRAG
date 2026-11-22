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

=== ITEM id=7a73145ec0aa3b68 ===
Gold sentences and facts:
Sentence 0: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 0.0: Miami has five remaining opponents.
  Fact 0.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.
Sentence 1: 152.67 -- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.
  Fact 1.0: The sentence begins with the number 152.67.
  Fact 1.1: Tyreek Hill missed last week's game.
  Fact 1.2: Tyreek Hill now needs to average almost 153 yards per game in his final three games.
  Fact 1.3: Tyreek Hill's stated goal is 2,000 receiving yards this season.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 1>

<item 2>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 2>

<item 3>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Which WRs are on bye in Week 14?

The final set of bye weeks sees only the Commanders and Cardinals idle, resulting in the losses of Terry McLaurin, Marquise Brown, Jahan Dotson, Curtis Samuel,
Rondale Moore, and Greg Dortch.

WEEK 14 PPR RANKINGS:

QBs | RBs | TEs | D/STs | Kickers

Even worse, Washington and Arizona sit as two of the weakest pass defenses in the league, so fantasy owners won't be able to count on big stats from their
receivers going against the Commanders and Cardinals.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy WRs in Week 14?

The league's leading receiver, Tyreek Hill (vs.
</item 3>

<item 4>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 4>

<item 5>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 5>

<item 6>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 6>

<item 7>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
1 seed in the conference.

Compared to the 49ers, Cowboys and Eagles, Goff and the Lions may have the easiest schedule to finish the season.

But bettors should also pay attention to C.J. Stroud. He leads the league in passing yards, is one of four quarterbacks eho have started the whole season with a
passer rating over 100, and his touchdown to interception ratio is the best in the NFL (4 to 1)

As for the value of his play, Stroud has taken one of the worst teams in the NFL (11 total wins the last three seasons) and turned it into a playoff contender
and potential division champion.

Offensive Player of the Year

Current favorite: Tyreek Hill

Longshot: CeeDee Lamb

There's a good reason why Tyreek Hill is such a heavy favorite to win OPOY. The Dolphins wide receiver is on track to become the first player with 2,000-plus
receiving yards in a single season.
</item 7>

<item 8>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 8>

<item 9>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 9>

<item 10>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 10>
=== END ITEM id=7a73145ec0aa3b68 ===

=== ITEM id=df5f276af6ae6742 ===
Gold sentences and facts:
Sentence 0: 13 Dec 2023 00.58 GMT 12th over: West Indies 102-4 (Hope 26, Hetmyer 1) Just a single and the wicket off the over, Shai Hope can’t break the legspin shackles.
  Fact 0.0: The 12th over is timestamped 13 Dec 2023 00.58 GMT.
  Fact 0.1: At the end of the 12th over, West Indies were 102-4.
  Fact 0.2: At the end of the 12th over, Hope had 26 runs.
  Fact 0.3: At the end of the 12th over, Hetmyer had 1 run.
  Fact 0.4: The 12th over had just a single and a wicket.
  Fact 0.5: Shai Hope could not break the legspin shackles in the 12th over.
Sentence 1: 6d ago 22.54 GMT 16th over: West Indies 112-4 (Hope 29, Powell 8) Hope, who is playing an exemplary innings, opens the face to steer Rehan for his first boundary.
  Fact 1.0: The 16th over is timestamped 6d ago, 22.54 GMT.
  Fact 1.1: At the end of the 16th over, West Indies were 112-4.
  Fact 1.2: At the end of the 16th over, Hope had 29 runs.
  Fact 1.3: At the end of the 16th over, Powell had 8 runs.
  Fact 1.4: Hope is playing an exemplary innings.
  Fact 1.5: Hope opens the face to steer Rehan for Hope's first boundary.

Retrieved items:
<item 1>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.45 GMT WICKET! Shai Hope c Brook b Rehan Ahmed 36 (West Indies 123-5) Ahmed is bunted for SIX by Shai Hope but strikes with the next ball as Hope
holes out!



13 Dec 2023 01.44 GMT 14th over: West Indies 112-4 (Hope 27, Powell 10) Liam Livingstone instead of Rashid and he does well to keep it to singles, England
trying to take this as deep as possible and hope West Indies fluff their lines at the death. West Indies need 55 from 36 balls.
</item 1>

<item 2>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Updated at 23.20 GMT

6d ago 23.16 GMT I suppose it’s appropriate that it ended with a six, the 120th of the series. Hope played a gem of an innings there, calmly anchoring West
Indies in an increasingly nervous chase with a run-a-ball 43. England fought admirably to defend a below-par total; but for Hope, they would probably have done
it.

Updated at 23.17 GMT

6d ago 23.14 GMT WEST INDIES WIN THE SERIES 3-2! 19.2 overs: West Indies 133-6 (Hope 43, Holder 4) That’s it! Hope pings Woakes over extra cover for six to seal
another series victory for West Indies! West Indies’ Shai Hope (right) celebrates with Jason Holder after hitting a six to win the match by 4 wickets.
</item 2>

<item 3>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
after newsletter promotion

Quick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhone or the Google Play store on Android
by searching for 'The Guardian'.

If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.

Halfway through their innings they were 62 for three, and if England had 15 more runs at the same point they had also made a complete hash of the rest of it.
That was a trap which, thanks to Hope’s 43-ball 43 as he again steered his side to their target, West Indies avoided. This batting display may have lacked their
usual flamboyance but until the final ball it was entirely appropriate for the occasion.

Amid the wider theme of transformation Adil Rashid’s excellence remains constant.
</item 3>

<item 4>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
That made it 70 for four and the game was one ball into its ninth over, at which point Liam Livingstone and Moeen Ali had to focus for a while on disaster
prevention rather than entertainment.

Gudakesh Motie is embraced his his West Indies teammates after dismissing England’s in-form Phil Salt. Photograph: Randy Brooks/AFP/Getty Images

Suddenly bowlers were energised by the prospect of posting, for a change, less-than-humiliating figures. Akeal Hosein’s were outstanding, his four overs going
for just 20 and bringing the wickets of Will Jacks and Moeen, undone by a relay catch on the boundary after combining with Livingstone to add 40 for the fifth
wicket. Livingstone followed precisely two overs later, prodding the ball limply back into the hands of Motie, at which point things really fell apart, the last
five wickets surrendered in 20 balls for just 11 runs.
</item 4>

<item 5>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Brutal elegance. Rehan comes back well with two sharply turning leggies. Bosh! Shai Hope gets in on the act and smears a length ball over mid on for the third
six of the over. It’s a learning curve for the tyro spinner but that doesn’t make it any less bruising.



13 Dec 2023 00.23 GMT 6th over: West Indies 59-1 (Mayers 22, Hope 13) Curran is called back, Mills granted just the one over for the time being. Two slower
balls bring two soothing dots for England. A single each to Mayers and Hope. Curran does very well to get out of the over with just three runs conceded.
Powerplay complete. How long till we see some spin from Rehan? Not long at all, he’s on for the next over.
</item 5>

<item 6>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.00 GMT 17th over: West Indies 114-5 (Hope 30, Russell 1) The wicket England really need is that of Shai Hope, who is quietly guiding West Indies
towards the finish line.
</item 6>

<item 7>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.20 GMT 5th over: West Indies 56-1 (Mayers 20, Hope 12) Woakes continues, after five balls he’s gone for just three runs. Can he get out of the
over cheaply? No – A length ball is clubbed down the ground for SIX with a minimum of fuss by Shai Hope. West Indies have come out of the traps as well as
England did, one over of the Powerplay left.
</item 7>

<item 8>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Often there is a sense as a tour enters its final days of thoughts turning to home, and perhaps the focus was not as laser-sharp as it might have been for this
game, determination less than absolute to grasp this opportunity to end England’s run of reverses in the Caribbean. Whatever the reason, the pyrotechnics that
rocketed England to victory in the two previous matches never sparked in this one. It took a fabulous delivery from Gudakesh Motie to end Phil Salt’s run of
absurd form with his score a modest 38, but some of the other wickets that fell during the period that defined England’s innings and the game seemed a little
careless.

In the fourth over Jos Buttler casually turned a Jason Holder delivery to fine leg, where Oshane Thomas was positioned, hands cupped. Harry Brook tried to
paddle-sweep Motie but sent the ball only a few feet into the air, giving Nicholas Pooran an easy catch.
</item 8>

<item 9>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.45 GMT 14th over: West Indies 94-3 (Hope 19, Rutherford 30) West Indies are rushing towards a series victory. Rutherford hooks the new bowler Sam
Curran for six more, and there are 11 from the over in total. Rutherford has 30 from 23 balls, Hope 19 from 28. They’re playing their roles to perfection.



6d ago 22.40 GMT 13th over: West Indies 83-3 (Hope 17, Rutherford 21) Rehan Ahmed returns, perhaps in the hope of tempting Rutherford into one big shot too
many. Instead it’s Shai Hope, who is happily playing the anchor role, who faces five of the six deliveries. England are running out of time.
</item 9>

<item 10>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Amid the wider theme of transformation Adil Rashid’s excellence remains constant. The 35-year-old, who rose to the top of the T20 bowling rankings for the first
time on Wednesday, took two wickets and conceded just 21 while Topley was even better, also claiming a pair of victims but conceding only 17. England take
plenty of positives from this series, but the result was not one.
</item 10>
=== END ITEM id=df5f276af6ae6742 ===

=== ITEM id=ecb565de846b2118 ===
Gold sentences and facts:
Sentence 0: He is helping to usher in a youth movement up front for Argentina, with Julian Alvarez and Alejandro Garnacho considered the future of the forward line.
  Fact 0.0: He is helping to usher in a youth movement up front for Argentina.
  Fact 0.1: Julian Alvarez is considered the future of the forward line.
  Fact 0.2: Alejandro Garnacho is considered the future of the forward line.
Sentence 1: 2 hours to kickoff: A report from well-connected Argentine journalist Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for this match in some capacity.
  Fact 1.0: The kickoff is 2 hours away.
  Fact 1.1: Argentine journalist Gaston Edul is well-connected.
  Fact 1.2: A report from Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for this match in some capacity.

Retrieved items:
<item 1>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
MORE: Inter Miami season tickets for 2024 season see unprecedented jump in price

(Also, you'll notice a photo of Lionel Messi's jersey hanging in his locker in that tweet below, a pretty good indication that he'll return tonight)

Scene is set for Fan Appreciation Night 🏟️#MIAvCIN | 7:30 PM ET pic.twitter.com/9I07uq9hGW — Inter Miami CF (@InterMiamiCF) October 7, 2023

2 hours to kickoff: A report from well-connected Argentine journalist Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for
this match in some capacity. Edul reported that "the idea is for Leo Messi to get minutes when Inter Miami plays FC Cincinnati".

That makes it sound like Messi will only be available off the bench, but it's better than nothing to have a game-changer like him available at all.
</item 1>

<item 2>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
@fccincinnati goalkeeper @roman_413 had nothing but praise for @LuchoAcosta94 and Pat Noonan 🟠🔵 pic.twitter.com/Xyp2ozr7n5 — CBS Sports Golazo ⚽️
(@CBSSportsGolazo) October 3, 2023

1 hour 15 mins to kick: Lionel Messi was nominated for not only MLS Newcomer of the Year for the 2023 season, but also 2023 MLS MVP.

Laughable doesn't even begin to describe that blatant PR move, considering that most of the damage Messi has done for Inter Miami has come in Cup competitions.
He has one goal and two assists in 247 minutes of play in actual MLS competition. Lionel Messi has been fun, but let's hope the league doesn't make a mockery of
itself by actually giving Messi either honor.
</item 2>

<item 3>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
With the 2023 MLS season closing earlier than expected for Inter Miami, eliminated from playoff contention, Lionel Messi will have significant time off before
the new campaign begins in February 2024.

With every new season come preseason friendlies, as the players hope to ramp up their match fitness prior to the start of the coming campaign. Players need game
time to build their bodies to the point where they can play full 90-minute matches, especially when they are contesting multiple games a week.

Inter Miami's preseason will be watched with extra observation as Messi draws considerable attention wherever he goes, home or abroad. The Argentina star had
looked like being part of a tour of China, although that's no longer the case, while the prospect of facing a familiar side from home is a tantalising one.

The Sporting News has compiled all the known preseason matches either confirmed or reported prior to the start of Inter Miami's 2024 season.
</item 3>

<item 4>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
🤔 pic.twitter.com/Svq79fvbBG — Major League Soccer (@MLS) October 4, 2023

1 hour to kickoff: With lineups not yet released, there is finally a report regarding Lionel Messi's availability. According to Michelle Kaufman of the Miami
Herald, Messi "did not arrive with the team" to Soldier Field. While that's not outright confirmation that he will not take part in the match, it's a very
strong indication that he will not be involved tonight.

1 hour 15 mins to kick: With lineups due imminently, and nothing to indicate if Lionel Messi is involved tonight or not, the Chicago Fire media team has jokes.
</item 4>

<item 5>
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
</item 5>

<item 6>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Had Inter Miami been still in the playoff hunt, with a potential postseason run on the horizon, Argentina might have played it a bit safer with Messi in this
international window. Yet now that his long offseason break is confirmed, Messi may be more aggressive with his minutes here knowing that, after Inter Miami's
final two meaningless regular-season matches, he will only have international breaks to keep him sharp until February.

According to a report earlier today, Messi will not start tonight, but is likely to play much or all of the second half, while Lautaro Martinez and Julian
Alvarez both start a national team match for the first time. World Cup hero Mario Kempes is a big fan of that.

"No deberíamos entrar en la polémica, como pasó con Batistuta y Crespo. Lautaro tuvo un bajón en el Mundial, entró Julián y cambió muchos las cosas, pero ahora
volvió a hacer goles. Y Julián es muy molesto.
</item 6>

<item 7>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
Inter Miami no va a viajar a China. La gira que incluía dos amistosos se cae por temas burocráticos/comerciales. pic.twitter.com/8I0jBxg4Zz — Gastón Edul
(@gastonedul) October 30, 2023

Instead, Inter Miami met NYCFC in a hastily arranged friendly at DRV PNK Stadium on November 10 to honor Lionel Messi's most recent Ballon d'Or. The friendly is
being dubbed the "Noche d'Or" or "Night of Gold" as a nod to his latest achievement.
</item 7>

<item 8>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Meanwhile, Fabrizio Romano also shot it down from a journalistic standpoint.

There isn't enough of a break for him to reasonably integrate with Barcelona, and at 36 years old he's sure to welcome the extended time off.

Tata Martino on rumors of Messi playing on loan with Barcelona this winter:



"Is he going for fun? Maybe he's going to visit Barcelona. Other than that, I don't know anything else about it." pic.twitter.com/qJwoJOwkC6 — Felipe Cárdenas
(@FelipeCar) October 8, 2023

2 hours to kickoff: Lionel Messi's club Inter Miami were eliminated from MLS playoff contention after their latest defeat, a 1-0 loss to FC Cincinnati last
weekend. Messi had missed the previous four matches but returned to the field for the final 30 minutes. He was powerless to turn the tide of the game, after
Inter Miami had missed so many gilt-edged chances in the first half.
</item 8>

<item 9>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
The team from Argentina hold a special place in the heart of Messi and Martino. Not only are both men from the city of Rosario, where Newell's are based, but
Messi developed at the club's youth academy and Martino played more than 400 senior games for the club before becoming head coach in 2012.

"I am delighted to welcome my beloved Newell's to our home here in Miami. It will be a special match due to everything Newell's Old Boys means to me," said the
Inter Miami coach. "It will also be a good opportunity to prepare for what will surely be an exciting season."

Messi signed for Barcelona after spending six years as a youth player at Newell's, during which time he reportedly scored 234 goals.

Will Inter Miami face River Plate in Dallas?

In late November, it was reported that Inter Miami have arranged to contest a friendly against Argentine giants River Plate in Dallas, Texas in late January.
</item 9>

<item 10>
Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match (Sporting News, 2023-10-04T23:33:00+00:00)
@MiamiHerald @HeraldSports @InterMiamiCF #InterMiamiCF #Messi𓃵 #cf97 #MLS pic.twitter.com/hp8KHzkwNB — Michelle Kaufman (@kaufsports) October 4, 2023

2 hours to kickoff: Tata Martino has played his cards extremely close to his chest when it comes to the fitness of Lionel Messi. Each week we see how
tight-lipped the Inter Miami boss has been regarding Messi's injury.

Prior to tonight's match against the Chicago Fire, Martino was asked about the return of Jordi Alba, and he stated that there is a timeline in place for Alba's
return. When asked if there's a timeline for Messi's return, Martino refused to acknowledge if there is.

Tata Martino when I asked about Lionel Messi's return timeline:



"I understand your desire to know the exact timelines, but if I had the exact timelines I would share them.



"I just don't have them."
</item 10>
=== END ITEM id=ecb565de846b2118 ===

=== ITEM id=d96d5decdfec730f ===
Gold sentences and facts:
Sentence 0: Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.
  Fact 0.0: Jim Harbaugh was suspended.
  Fact 0.1: Jim Harbaugh held his usual weekly news conference on Monday despite the suspension.
  Fact 0.2: Jim Harbaugh mostly avoided addressing the only topic on anyone's mind at the news conference.
Sentence 1: It would be Harbaugh's second suspension of the season, as the Wolverines self-imposed a three-game ban on Harbaugh to start the season amid an NCAA investigation that found Harbaugh lied to investigators.
  Fact 1.0: The suspension would be Harbaugh's second suspension of the season.
  Fact 1.1: The Wolverines self-imposed a three-game ban on Harbaugh to start the season.
  Fact 1.2: The self-imposed ban occurred amid an NCAA investigation.
  Fact 1.3: The NCAA investigation found that Harbaugh lied to investigators.
Sentence 2: Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns.
  Fact 2.0: Harbaugh reportedly watched the game from the team hotel.
  Fact 2.1: The leading piece of drama now returns.

Retrieved items:
<item 1>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Nicole Auerbach (@NicoleAuerbach) November 11, 2023

Harbaugh hearing coming down to wire before Penn State game

ESPN's Pete Thamel reported it's unclear Harbaugh will be granted a hearing in time to be able to coach against Penn State. With kickoff scheduled for noon ET,
Thamel said "the drama's going to go right up 'till noon."

Latest from ⁦@CollegeGameDay⁩ on Jim Harbaugh’s TRO legal filing. Will he coach Michigan at Penn State today? pic.twitter.com/bz3R1eYxI9 — Pete Thamel
(@PeteThamel) November 11, 2023

The case is reportedly in the hands of Judge Carol Kunhke, a Michigan alumnus.

Harbaugh was not on the Michigan team bus with other coaches and players when it arrived at Beaver Stadium Saturday morning.

Michigan players and coaches arrive. No Jim Harbaugh on the bus.
</item 1>

<item 2>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Although the Big Ten has closed its investigation, we are continuing to cooperate with the NCAA as it moves forward with its ongoing investigation. Consistent
with our commitment to integrity, we will continue to take the appropriate actions, including disciplinary measures, based on information we obtain. Earlier
today, Michigan Athletics relieved Chris Partridge of his duties as a member of the Michigan Football staff. Due to employee privacy laws, we are unable to
comment further."

Because Michigan and Harbaugh on Thursday dropped their litigation against the Big Ten, he will not be on Michigan's sideline this week against Maryland and
next week against Ohio State to end the regular season.

Former Wolverines assistant and Harbaugh confidant Biff Poggi on Friday offered his former boss an invitation. Poggi, in his first year as Charlotte's head
coach, asked Harbaugh to join the 49ers for their games vs. Rice on Saturday and at South Florida on Nov. 25.
</item 2>

<item 3>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Rice on Saturday and at South Florida on Nov. 25.

I am extending an open invitation to Jim Harbaugh to join me on the sidelines for the next two Charlotte football games. It would be my honor to stand next to
you again as we did for three years — Biff poggi (@BiffPoggi) November 17, 2023

Thursday, Nov. 16

Michigan accepts Harbaugh's three-game suspension

Michigan has dropped its litigation against the Big Ten over Harbaugh's three-game suspension, and it will accept the punishment levied against the head coach.

From Michigan:

"This morning, the University, Coach Harbaugh, and the Big Ten resolved their pending litigation. The Conference agreed to close its investigation, and the
University and Coach Harbaugh agreed to accept the three-game suspension. Coach Harbaugh, with the University's support, decided to accept this sanction to
return the focus to our student-athletes and their performance on the field.
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
The Conference has confirmed that it is not aware of any information suggesting Coach Harbaugh's involvement in the allegations. The University continues to
cooperate fully with the NCAA's investigation.

From the Big Ten:

"The Big Ten Conference's commitment to student-athletes, sportsmanship and the commissioner's duty to protect the integrity of competition will never waver.
Today's decision by the University of Michigan to withdraw its legal challenge against the conference's November 10th notice of disciplinary action is
indicative of the high standards and values that the conference and the university seek to uphold, the University of Michigan is a valued member of the Big Ten
Conference and the conference will continue to work cooperatively with the university and the NCAA during this process."

Monday, Nov. 13

Harbaugh says he'll state his case at Friday hearing

Despite the suspension, Jim Harbaugh held his usual weekly news conference Monday, though he mostly avoided addressing the only topic on anyone's mind.
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
On the field, Michigan is in the midst of what could be a historic season. Off the field, the the focus has rarely been on the team's success.

An investigation into Michigan's alleged sign-stealing scheme, centered on now-fired staffer Connor Stalions, continues to evolve by the hour, with one update
after another adding another wrinkle to the saga that began in October. The Big Ten suspended coach Jim Harbaugh for three games on Friday with the team on a
plane en route to Penn State, and although Michigan had hoped to block the suspension, the Wolverines will be without their coach on the sidelines in Happy
Valley.

The Sporting News is tracking the latest updates from the Michigan sign-stealing scandal. Follow along with the newest developments below.

MORE ON MICHIGAN SIGN-STEALING:

Michigan sign-stealing scandal updates

Friday, Nov.
</item 5>

<item 6>
Michigan didn't need Jim Harbaugh against Penn State, but Ohio State is a different game (Sporting News, 2023-11-11T21:32:44+00:00)
Was this old-school Big Ten football or Big Ten West football in disguise?

Michigan's defense clamped on Penn State quarterback Drew Allar (10 of 22, 70 yards, TD), and the Nittany Lions had one play of more than 20 yards. Corum closed
the drama with a 30-yard TD.

The same-old questions about Nittany Lions coach James Franklin in big games resurfaced, but those are back-page headlines in the Big Ten East race. The quirky
tiebreaker involving Big Ten West opponents? That's out the door. The winner-take-all showdown between Ohio State and Michigan? There are a full two weeks of
anticipation for that now.

Harbaugh reportedly watched the game from the team hotel, and now the leading piece of drama returns. Moore's postgame emotion was a headline, but there is a
reality that set in through a brutal watch of a second half.

McCarthy is a leader.
</item 6>

<item 7>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Michigan players and coaches arrive. No Jim Harbaugh on the bus. pic.twitter.com/wnGzqAxwcp — Adam Rittenberg (@ESPNRittenberg) November 11, 2023

Michigan offensive coordinator Sherrone Moore will serve as acting head coach if Harbaugh is not available.

Friday, Nov. 10

Michigan files temporary restraining order against the Big Ten

In an effort to delay Harbaugh's suspension, the university filed a temporary restraining order in Washtenaw County Trial Court on Friday night. If granted, the
suspension would be delayed, but the clock is ticking before the Wolverines' game vs. Penn State on Saturday.

Michigan's temporary restraining order request has been filed.



Plaintiffs: Jim Harbaugh, U-M regents

Defendants: Big Ten, Tony Petitti



Judicial officer: Timothy P. Connors
</item 7>

<item 8>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Here is Michigan’s 10-page letter to the Big Ten responding to the sportsmanship investigation, reporting with @RossDellenger 1/3 pic.twitter.com/P0oYgIbS67 —
Dan Wetzel (@DanWetzel) November 9, 2023

No timetable for Big Ten decision

The Big Ten is reviewing the "lengthy" official response from Michigan and Jim Harbaugh and it's unclear when a potential ruling might come down, says ESPN's
Pete Thamel. His colleague Adam Rittenberg reported Wednesday that a decision was expected by Thursday.

Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh. There’s no indication of any timetable on a decision on
potential Big Ten discipline in the case of Michigan’s alleged illegal signal stealing.
</item 8>

<item 9>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
OFFICIAL: Jim Harbaugh suspended for the remaining three regular season games.



"Michigan has been found in violation of the Big Ten Sportsmanship Policy for conducting an impermissible, in-person scouting operation over multiple years..."
https://t.co/htnUkshmnF — Chris Vannini (@ChrisVannini) November 10, 2023

Jim Harbaugh joins Michigan for trip to State College

According to the sleuths on the ground in Ann Arbor, Harbaugh is on the team bus and heading to the airport with his team as Michigan awaits word on a potential
suspension.

Jim Harbaugh boarding the bus pic.twitter.com/CWBJbwEEhi — Austin Meek (@byAustinMeek) November 10, 2023

Michigan still awaiting word from Big Ten

According to ESPN's Adam Rittenberg, Michigan still hasn't heard about an impending punishment from the Big Ten as its 1 p.m. ET flight looms.
</item 9>

<item 10>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
pic.twitter.com/Qm9iehYVTR — Pete Thamel (@PeteThamel) November 11, 2023

Harbaugh will not be on the sidelines vs. Penn State

Michigan filed a temporary restraining order against the Big Ten on Friday night, hoping to block Harbaugh's suspension. But there will not be a ruling before
the Wolverines' game vs. Penn State, with the hearing scheduled for Friday, Nov. 17.

No ruling on Michigan’s TRO filing today, so Jim Harbaugh will not coach vs. Penn State.



Statement from the school: “We look forward to presenting our case next week where we intend to demonstrate that the Big Ten has not acted legally or fairly.”
</item 10>
=== END ITEM id=d96d5decdfec730f ===

=== ITEM id=bfd5d2dd239da074 ===
Gold sentences and facts:
Sentence 0: 13 Dec 2023 00.58 GMT 12th over: West Indies 102-4 (Hope 26, Hetmyer 1) Just a single and the wicket off the over, Shai Hope can’t break the legspin shackles.
  Fact 0.0: The timestamp is 13 Dec 2023, 00:58 GMT.
  Fact 0.1: It is the 12th over.
  Fact 0.2: West Indies were 102 for 4 wickets.
  Fact 0.3: Hope had 26 runs and Hetmyer had 1 run.
  Fact 0.4: In that over there was just a single and a wicket.
  Fact 0.5: Shai Hope could not break the legspin shackles.
Sentence 1: 6d ago 22.54 GMT 16th over: West Indies 112-4 (Hope 29, Powell 8) Hope, who is playing an exemplary innings, opens the face to steer Rehan for his first boundary.
  Fact 1.0: The timestamp is 6 days ago, 22:54 GMT.
  Fact 1.1: It is the 16th over.
  Fact 1.2: West Indies were 112 for 4 wickets.
  Fact 1.3: Hope had 29 runs and Powell had 8 runs.
  Fact 1.4: Hope is playing an exemplary innings.
  Fact 1.5: Hope opens the face to steer Rehan for his first boundary.

Retrieved items:
<item 1>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Just as surely as when they came out on Tuesday facing an almost unachievable task, the outcome of the West Indies’ run chase seemed inevitable from the start.
With such a meagre target they needed little more than to keep their heads, and though both Topley and Sam Curran conceded only two with their final overs to
push the contest into its final over, that never looked in doubt.

skip past newsletter promotion Sign up to The Spin Free weekly newsletter Subscribe to our cricket newsletter for our writers' thoughts on the biggest stories
and a review of the week’s action Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more
information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy
. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Quick Guide How do I sign up for sport breaking news alerts?
</item 1>

<item 2>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Often there is a sense as a tour enters its final days of thoughts turning to home, and perhaps the focus was not as laser-sharp as it might have been for this
game, determination less than absolute to grasp this opportunity to end England’s run of reverses in the Caribbean. Whatever the reason, the pyrotechnics that
rocketed England to victory in the two previous matches never sparked in this one. It took a fabulous delivery from Gudakesh Motie to end Phil Salt’s run of
absurd form with his score a modest 38, but some of the other wickets that fell during the period that defined England’s innings and the game seemed a little
careless.

In the fourth over Jos Buttler casually turned a Jason Holder delivery to fine leg, where Oshane Thomas was positioned, hands cupped. Harry Brook tried to
paddle-sweep Motie but sent the ball only a few feet into the air, giving Nicholas Pooran an easy catch.
</item 2>

<item 3>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Amid the wider theme of transformation Adil Rashid’s excellence remains constant. The 35-year-old, who rose to the top of the T20 bowling rankings for the first
time on Wednesday, took two wickets and conceded just 21 while Topley was even better, also claiming a pair of victims but conceding only 17. England take
plenty of positives from this series, but the result was not one.
</item 3>

<item 4>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
What a difference two days make. Somehow Tuesday’s ludicrous, record-smashing, high-scoring entertainment was followed by a sequel on the same surface that
eventually delivered thrilling low-scoring entertainment, and a different result claimed in very different style. It was eventually won for West Indies with
four balls to spare and in the most unfitting of ways – with a massive six – by Shai Hope, again proving himself the right man for a run-chase.

So a series marked by stunning power hitting, roaring run rates and an unexpected midway transformation in England’s performances and prospects concluded with a
3-2 triumph for West Indies, who extended their supremacy over England at home to four series in three formats across nearly five years.

England assembled exactly the same players in exactly the same place and exactly the same situation – losing the toss, put in to bat – but this was an entirely
different performance.
</item 4>

<item 5>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
A wicket once so true turned untrustworthy and they struggled to a score of 132, less than half the 267 to which they had joyously motored just 48 hours
previously. And though the home side’s response was anything but overwhelming, neither was the result ever in significant doubt.

“We’ve had some high-scoring games, but the last game was almost the complete opposite of what happened here,” Reece Topley said. “We were probably just under
par with the first-innings score but with the ball we were excellent, pushed them right to the end. I think it’s been an amazing series. Both teams have played
some unbelievable cricket. I was so excited turning up today, because it was basically a final and those are the games you want to play in and be on the right
side of. The bottom line is you want to win this series, I want to win every series I can for England.”
</item 5>

<item 6>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
after newsletter promotion

Quick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhone or the Google Play store on Android
by searching for 'The Guardian'.

If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.

Halfway through their innings they were 62 for three, and if England had 15 more runs at the same point they had also made a complete hash of the rest of it.
That was a trap which, thanks to Hope’s 43-ball 43 as he again steered his side to their target, West Indies avoided. This batting display may have lacked their
usual flamboyance but until the final ball it was entirely appropriate for the occasion.

Amid the wider theme of transformation Adil Rashid’s excellence remains constant.
</item 6>

<item 7>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
That made it 70 for four and the game was one ball into its ninth over, at which point Liam Livingstone and Moeen Ali had to focus for a while on disaster
prevention rather than entertainment.

Gudakesh Motie is embraced his his West Indies teammates after dismissing England’s in-form Phil Salt. Photograph: Randy Brooks/AFP/Getty Images

Suddenly bowlers were energised by the prospect of posting, for a change, less-than-humiliating figures. Akeal Hosein’s were outstanding, his four overs going
for just 20 and bringing the wickets of Will Jacks and Moeen, undone by a relay catch on the boundary after combining with Livingstone to add 40 for the fifth
wicket. Livingstone followed precisely two overs later, prodding the ball limply back into the hands of Motie, at which point things really fell apart, the last
five wickets surrendered in 20 balls for just 11 runs.
</item 7>

<item 8>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Updated at 23.20 GMT

6d ago 23.16 GMT I suppose it’s appropriate that it ended with a six, the 120th of the series. Hope played a gem of an innings there, calmly anchoring West
Indies in an increasingly nervous chase with a run-a-ball 43. England fought admirably to defend a below-par total; but for Hope, they would probably have done
it.

Updated at 23.17 GMT

6d ago 23.14 GMT WEST INDIES WIN THE SERIES 3-2! 19.2 overs: West Indies 133-6 (Hope 43, Holder 4) That’s it! Hope pings Woakes over extra cover for six to seal
another series victory for West Indies! West Indies’ Shai Hope (right) celebrates with Jason Holder after hitting a six to win the match by 4 wickets.
</item 8>

<item 9>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12.4 overs: West Indies 108-4 (Hope 27, Powell 6)

Updated at 01.09 GMT

13 Dec 2023 00.58 GMT 12th over: West Indies 102-4 (Hope 26, Hetmyer 1) Just a single and the wicket off the over, Shai Hope can’t break the legspin shackles.
Windies skipper Rovman Powell is the new man and Buttler cranks the pressure by inserting a slip.

Updated at 01.04 GMT

13 Dec 2023 00.52 GMT WICKET! Hetmyer c Duckett b Rashid 1 (West Indies 101-4) Adil Rashid takes his 100th T20I wicket! A crucial one in the context of the game
too – the dangerous Hetmyer is sent on his way for just a single as England look to ramp up the pressure heading into the nub end of this match.
</item 9>

<item 10>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.57 GMT WICKET! West Indies 113-5 (Powell c Rashid b Topley 8) England haven’t given this up. Powell edges Topley to short third man, where Rashid
takes a smart low catch to his left. West Indies need 20 from 21 balls. Reece Topley (right) celebrates the dismissal of Rovman Powell. Photograph: Randy
Brooks/AFP/Getty Images

Updated at 23.07 GMT

6d ago 22.54 GMT 16th over: West Indies 112-4 (Hope 29, Powell 8) Hope, who is playing an exemplary innings, opens the face to steer Rehan for his first
boundary. Rehan has struggled as the series has progressed, an d later in the over Powell muscles him over long-on for six. West Indies are 21 runs away from a
hat-trick of T20 series wins this year: South Africa away, India and England at home.
</item 10>
=== END ITEM id=bfd5d2dd239da074 ===

=== ITEM id=8d5d36a720f2c6bb ===
Gold sentences and facts:
Sentence 0: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 0.0: Miami has five remaining opponents.
  Fact 0.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.
Sentence 1: 152.67 -- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.
  Fact 1.0: Tyreek Hill missed last week's game.
  Fact 1.1: Tyreek Hill now needs to average almost 153 yards per game in his final three games.
  Fact 1.2: The figure cited is 152.67 yards per game.
  Fact 1.3: Tyreek Hill's stated goal is 2,000 receiving yards this season.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Six of the 17 won anyway. The Dolphins were the only one of those 223 teams to win and cover the spread, which was 13.5 for this game.

• Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games. That ranks second in NFL
history to Hall of Famer Don Hutson, who had 819 yards through six with Green Bay in 1942.
</item 1>

<item 2>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
But there is a solid chance he does not hit that mark. Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight. If he
slows down, someone like CeeDee Lamb could catch up.

Based on his current season average, Lamb is on track for close to 1,600 yards, but based on his production over the last seven games, he could finish with
closer to 1,800 yards.

He will probably need Hill to fade down the stretch to have a shot. But if Hill does and he continues to produce, Lamb could steal OPOY from Hill.

Defensive Player of the Year

Current favorite(s): Micah Parsons , Myles Garrett , and T.J. Watt

, Myles Garrett , and T.J. Watt Longshot: Danielle Hunter

Hunter leads the league in tackles for a loss and is third in sacks with 13.5 (behind Khalil Mack and Watt).
</item 2>

<item 3>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
MIAMI DOLPHINS | WR Tyreek Hill: Arguably the best wide receiver in the NFL, Hill has somehow reached a new level of dominance in his age-29 season. The
seven-time Pro Bowler leads the NFL in receiving yards (1,076) and TD catches (eight). As long as the Dolphins maintain their success on offense down the
stretch, Hill should earn Offensive Player of the Year honors. He's on pace to break Calvin Johnson's season record for receiving yards (1,964).



NEW YORK JETS | LB Quincy Williams: While his brother, Quinnen, may be the better-known Jets defender in the family, Williams — the AFC Defensive Player of the
Month for October — is also making a name for himself. Aside from being the fifth-highest-rated linebacker in the NFL by Pro Football Focus, Williams is tied
for 17th in tackles (74) and tied for ninth in TFLs (nine) among all defenders.
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Receiving Yards Through Six NFL Games Rk Player Season Yds 1 Don Hutson 1942 819 2 Tyreek Hill 2023 812 3 Wes Welker 2011 785 4 Elroy Hirsch 1951 766 5 Santana
Moss 2005 743 6 DeAndre Hopkins 2015 726 7 Raymond Berry 1960 725 8 Jerry Rice 1986 723 9 Bob Boyd 1954 718 10 Bobby Mitchell 1962 714

• Jared Goff is enjoying the second-best start to a season for his career from a statistical standpoint, behind only his 2018 Super Bowl season with the Rams.
With the Lions at 5-1 and Goff heading toward a likely contract extension, could another team come after his injured backup, former University of Tennessee
quarterback Hendon Hooker?
</item 4>

<item 5>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 5>

<item 6>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 6>

<item 7>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Can Raiders QB Aidan O’Connell earn next season’s starting job?: After leading the offense on seven touchdown scoring drives in last week's 63-21 win over the
Chargers, the rookie fourth-rounder aims to further establish himself as a serious contender for Las Vegas' starting job next season. Despite the encouraging
performance, O'Connell has been inconsistent across six games as the full-time starter, throwing for 1,300 yards, seven touchdowns, and five interceptions.

However, a stellar showing against the NFL's fifth-ranked pass defense (202.5 yards per game allowed) would benefit his bid to be the starter in 2024.

Chiefs TE Travis Kelce: Yes, his girlfriend will be in town for the holiday. Kelce is averaging 89.6 yards per game with global popstar Taylor Swift in
attendance this season but is coming off his worst outing in her presence, catching five passes for 28 yards.
</item 7>

<item 8>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
Weekly Breakdown MATCHUP @ MIA MIA -1.5 O/U 51 OPP VS WR 15th PROJ PTS 19.4 WR RNK 2nd YTD Stats REC 103 TAR 141 REYDS 1306 TD 10 FPTS/G 21.6 The choice between
Lamb and Tyreek Hill will not be an easy one. The tiebreaker for me is that Lamb is $900 cheaper and Hill may still not be 100% by the time the Cowboys and
Dolphins kick off. Last week was the first time since Week 9 that Lamb failed to reach the end zone and the second time since Week 5 that he didn't score at
least 17 PPR Fantasy points.

Contrarian DFS Play (WR Preview) Projections powered by Sportsline Jaylen Waddle WR MIA Miami • #17

Age: 25 • Experience: 3 yrs.
</item 8>

<item 9>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
1 seed in the conference.

Compared to the 49ers, Cowboys and Eagles, Goff and the Lions may have the easiest schedule to finish the season.

But bettors should also pay attention to C.J. Stroud. He leads the league in passing yards, is one of four quarterbacks eho have started the whole season with a
passer rating over 100, and his touchdown to interception ratio is the best in the NFL (4 to 1)

As for the value of his play, Stroud has taken one of the worst teams in the NFL (11 total wins the last three seasons) and turned it into a playoff contender
and potential division champion.

Offensive Player of the Year

Current favorite: Tyreek Hill

Longshot: CeeDee Lamb

There's a good reason why Tyreek Hill is such a heavy favorite to win OPOY. The Dolphins wide receiver is on track to become the first player with 2,000-plus
receiving yards in a single season.
</item 9>

<item 10>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson.

-- Deebo Samuel is on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons.
The other three are Tyreek Hill, Tyler Lockett, and DeSean Jackson. 18 -- Puka Nacua needs 18 more catches to set the rookie record for most catches in a
season.

-- Puka Nacua needs 18 more catches to set the rookie record for most catches in a season. 152.67 -- After missing last week's game, Tyreek Hill now needs to
average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving yards this season.

-- After missing last week's game, Tyreek Hill now needs to average almost 153 yards per game in his final three to reach his stated goal of 2,000 receiving
yards this season.
</item 10>
=== END ITEM id=8d5d36a720f2c6bb ===
