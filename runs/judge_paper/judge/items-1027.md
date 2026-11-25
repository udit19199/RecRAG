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

=== ITEM id=c2bef936c71d456e ===
Gold sentences and facts:
Sentence 0: UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.
  Fact 0.0: NFL Network's Ian Rapoport reported on a Sunday morning.
  Fact 0.1: Ian Rapoport reported that Hockenson is expected to play against New Orleans.
Sentence 1: Head coach Dennis Allen told reporters Friday that despite Olave being a full participant in Friday's practice he is still in concussion protocol though has a chance to play.
  Fact 1.0: Head coach Dennis Allen told reporters on Friday.
  Fact 1.1: Dennis Allen said Olave was a full participant in Friday's practice.
  Fact 1.2: Dennis Allen said Olave is still in concussion protocol.
  Fact 1.3: Dennis Allen said Olave has a chance to play.

Retrieved items:
<item 1>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
They haven't stopped playing a lot of man coverage lately, which shouldn't bother Justin Herbert because he's efficient against all coverages (and a little bit
better against man-to-man). L.A. hasn't put up more than 20 points in two straight, but 20 points should be more than enough to beat a Patriots offense that's
scored 17 or fewer points in 9 of its past 10 games.

STARTS: Justin Herbert, Keenan Allen, Austin Ekeler, Rhamondre Stevenson

Justin Herbert, Keenan Allen, Austin Ekeler, Rhamondre Stevenson SITS: Hunter Henry, Quentin Johnston, all Patriots WRs; Demario Douglas is OUT

Hunter Henry, Quentin Johnston, all Patriots WRs; Demario Douglas is OUT SLEEPERS: Gerald Everett, Jalen Guyton

Start 'em 10.0-7.00 6.99 - 5.01 5.
</item 1>

<item 2>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Cornerback Cam Taylor-Britt hurt his foot in practice and is out.

For the Jaguars, Travis Etienne (ribs) is also expected to play, Doug Pederson told reporters, despite the running back being limited in practice all week.
Strange was a late addition to the injury report, not practicing Saturday.
</item 2>

<item 3>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Jourdan Lewis. JSN can mirror some of the production Curtis Samuel gave the Commanders last week in Dallas, but forget 100 yards. Around 50 yards seems like the
ceiling with a big boost from garbage time.

Sit: TEs Noah Fant and Will Dissly

Noah Gray is a better fantasy player than Noah Fant this season. Will Dissly is barely a better fantasy player than Will Mallory this season.

WEEK 13 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Sit: Seahawks D/ST

The Seahawks are getting better defensively against the pass, but that's not enough to trust them here, as they're getting gashed on the ground, too.

Sit: K Jason Myers

This is actually not a good spot for one of the most reliable fantasy kickers, given the Cowboys don't give up many field-goal attempts.
</item 3>

<item 4>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Lions need an answer for their ugly pass defense -- since their Week 9 bye they've been mostly great against the run (3.1 yards per carry allowed, one RB to
get 15-plus PPR points) but they rank bottom-six in completion rate allowed (68.4%), yards per attempt allowed (8.0), passing ADOT against (8.84 yards) and
defensive Expected Points Added (EPA) per dropback (-0.40, which is league-worst). And of their second-most eight passing touchdowns given up in three games,
five have come on passes of 15-plus Air Yards, which is tied for most in the league. They should catch a break with field stretcher Rashid Shaheed (thigh)
missing the matchup for New Orleans; Chris Olave looks like he will play.
</item 4>

<item 5>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 5>

<item 6>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Week 13 is in full swing and teams are starting to truly jockey for playoff position as we rocket down the stretch run of the regular season. Of course, with it
being so late in the year, injuries continue to be a predominant storyline to monitor heading into each slate and this weekend is no different. Specifically,
the quarterback position in Cleveland has taken an interesting turn with veteran Joe Flacco named the starter against the Rams with Dorian Thompson-Robinson
ruled out with a concussion. And that's just one situation that we'll be monitoring below.

Here's a look at all of the final injury reports for Week 13.

All NFL odds via SportsLine Consensus odds.

Chargers (-5.5) at Patriots

Allen made his return to practice on a limited basis on Friday and is officially questionable for Sunday's matchup in New England due to a quad injury.
Meanwhile, Pipkins was also limited on Friday.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
He is regulated to pass from the pocket, but you are taking away the biggest weapon the offense had when you do that. Odell (Beckham) is not that guy anymore.
Who are you truly afraid of on that offense? Nobody but Lamar, and if you are not using Lamar in a certain capacity, he becomes just like the rest of these
quarterbacks around the league — hit or miss.”

• Jacksonville Jaguars: Are they ready for prime time?

What I think: Trevor Lawrence’s knee injury was the big concern coming out of Sunday. If he’s out, expectations shift dramatically. More than the other AFC
contenders, the Jaguars seem early in their life cycle, still finding their way. That makes them a little harder to trust in big spots but also leaves greater
space for growth. Let’s hope those test results on Lawrence’s knee come back clear.

Exec comment: “I like Jacksonville. Doug (Pederson) has a championship pedigree. This team is relatively young. They are scrappy.
</item 7>

<item 8>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Easy peasy!

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Both teams had a week of rest, so don't treat this like a typical Thursday game. But do expect the Cowboys defense to keep the pressure on Geno Smith and a
Seattle offense that's struggled to gain consistency over the last month. Smith actually did OK completing passes last week when pressured (60% completion rate)
but not at an efficient level (5.4 yards per attempt). Just two QBs have exceeded 250 yards and four QBs have thrown multiple touchdowns against Dallas this
year; I don't like Smith's chances to do either, which hurts any upside case for any Seahawks receiver. By the way, Curtis Samuel was the first receiver to get
100 yards on the Cowboys this year, and it took him 12 targets and 9 catches to get there.
</item 8>

<item 9>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
For those in deeper leagues, Tyler Conklin (vs. Texans), and Cade Otton (@ Falcons) both have matchups against defenses that entered last week in the top five
for FPPG allowed to TEs. Brevin Jordan (@ Jets) and Tanner Hudson (vs. Colts) also have highly favorable matchups and can work as deep-league PPR plays, though
Jordan's outlook is entirely dependent on whether Dalton Schultz (hamstring) is healthy.

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential fantasy TE busts in Week 14?

Dallas Goedert (@ Cowboys) isn't a "must-start" if he returns from his forearm injury this week. That said, he's also not a "must-sit." He had three catches and
50 yards against Dallas in Week 9 before exiting early with the forearm injury, so he can clearly produce against this defense.
</item 9>

<item 10>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 10>

<item 11>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I'd rather start Breece Hall, Rhamondre Stevenson, Dameon Pierce and Jaleel McLaughlin (if Javonte Williams sits out).

LAST WEEK: Harris had his first game with over 100 total yards against the Texans and averaged 5.1 yards per rush, just shy of the season-high he technically
had on six carries against the 49ers in Week 1. A 32-yard reception helped him cross the triple-digit barrier.

HARRIS: This might shock you, but Harris is averaging 4.3 yards per run this year with a 12.2% explosive run rate. Would you believe the explosive run rate,
which reflects the frequency of his 12-plus-yard runs, is fifth-best among 44 qualifying running backs?! He's also 13th in avoided tackle rate and top-10 in
yards after contact per carry at 3.53.

TOUCHDOWNS: Harris has zero.
</item 11>

<item 12>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Who are the TEs on bye in Week 14?

Few would've expected Week 14's two-team bye to be as problematic as it is, but losing Logan Thomas and Trey McBride this late in the season could have some
fantasy owners scrambling.

WEEK 14 PPR RANKINGS:

QBs | RBs | WRs | D/STs | Kickers

Fortunately, both the Commanders and Cardinals have above-average TE defenses, so taking those off the board only helps the position this week.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy TEs in Week 14?

As usual, Travis Kelce (vs. Bills in Week 14) tops our rankings, but T.J. Hockenson (@ Raiders), Sam LaPorta (@ Bears), David Njoku (vs. Jaguars), and Dalton
Kincaid (@ Chiefs) aren't far behind.
</item 12>

<item 13>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
If they're ailing, the Rams would be smart to try and use the run game early and often. Expect a low-scoring game.

STARTS: Kyren Williams, Amari Cooper, Puka Nacua, David Njoku, Jerome Ford,

Kyren Williams, Amari Cooper, Puka Nacua, David Njoku, Jerome Ford, SITS: Cooper Kupp (flex at best), Kareem Hunt, Tutu Atwell, Tyler Higbee, Royce Freeman,
both QBs, both DSTs (low-end options)

Cooper Kupp (flex at best), Kareem Hunt, Tutu Atwell, Tyler Higbee, Royce Freeman, both QBs, both DSTs (low-end options) SLEEPERS: Elijah Moore (PPR flex)

Start Him (Lineup Decisions) Projections powered by Sportsline Amari Cooper WR CLE Cleveland • #2

Age: 29 • Experience: 9 yrs.
</item 13>

<item 14>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'm pretty certain they're not bad enough to make Tim Boyle look like Tim Tebow or even Tim Hasselbeck, but I do wonder if they're juuust bad enough to give
better-than-normal numbers to Garrett Wilson and Breece Hall. Reinforcements along the O-line should help the Jets' cause against a Falcons defense that's
blitzed at a league-average rate, is eighth-lowest in pass rush pressure rate and is fifth-lowest in sacks (28).
</item 14>

<item 15>
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
</item 15>

<item 16>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He has more upside than Jerry Jeudy, Jahan Dotson and Tutu Atwell but I'd rather take a shot on Drake London, Terry McLaurin or Chris Godwin.

LAST WEEK: Over 19 snaps without Justin Jefferson last week, Osborn ran 18 routes and caught both of his two targets for 30 total yards. The receiving average
was delightful, but the small-sample target share (12.5%) was saddening. What's funny is that Osborn's target share was much higher when Jefferson was playing
(23.3%) but his receiving average tanked (6.3).

FILM: Osborn didn't beat coverage with his speed last week and his burst and agility were just OK. His 22-yard catch came on a Kirk Cousins scramble where he
waited in a spot and Cousins hit him, then Osborn ran inside instead of outside where he might have been able to score.
</item 16>

<item 17>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
For Taylor, he underwent surgery on his thumb this week and is expected to miss at least a couple of weeks, which will thrust Zack Moss into a more prominent
role in the Colts backfield.

Burks is the only Titan to have an injury designation entering the weekend, but the wideout was a full participant in Friday's practice. However, he will need
to clear the final stages of concussion protocol in order to play.

Dolphins (-9.5) at Commanders

Every player who the Dolphins listed as questionable for Sunday's game against Washington was able to practice on a limited basis on Friday. That includes
Cotton and Lamm, who both were DNPs earlier in the week. Rookie running back De'Von Achane was a full participant in Friday's practice and head coach Mike
McDaniel said he will play this week after missing time with a knee injury.

Washington could be without its starting center this weekend as it plays host to Miami.
</item 17>

<item 18>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Brown, Jahmyr Gibbs, Alvin Kamara, Chris Olave (if he passes concussion protocol), David Montgomery, Sam LaPorta, Taysom Hill SITS: Derek Carr, both DSTs

Derek Carr, both DSTs SLEEPERS: Juwan Johnson

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Falcons defense finally had a strong showing against the Saints last week, but they got a little lucky because so many players from New Orleans left with
injuries. In the three games prior their defense had a knack for making mediocre offenses look amazing (Will Levis' breakout game, Josh Dobbs' first game with
the Vikings, Kyler Murray's first game with the Cardinals).
</item 18>

<item 19>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 19>

<item 20>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs CAR MIA -13.5 O/U 48.5 OPP VS RB 31st PROJ PTS 7 RB RNK NR YTD Stats RUYDS 0 REC 0 REYDS 0 TD 0 FPTS/G 0 Once it's clear that Wilson
will play, the matchup combined with the Dolphins' approach to utilize multiple backs should put Wilson in position to have some good opportunities. Remember,
Wilson's injury was to his abdomen, not his legs, so there might not be much of a limit on how much work he'll do (though it's unlikely he'll have a ton of
touches). I'd take my chances with him over Samaje Perine, Jerick McKinnon and Antonio Gibson.

WILSON: Started practicing this week after being sidelined with finger and abdomen injuries since late August.
</item 20>

<item 21>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
Bills

33 Tee Higgins, Bengals vs. Colts

34 Jerry Jeudy, Broncos @ Chargers

35 Brandin Cooks, Cowboys vs. Eagles

36 Diontae Johnson, Steelers vs. Patriots

37 Elijah Moore, Browns vs. Jaguars

38 Adam Thielen, Panthers @ Saints

39 Tyler Lockett, Seahawks @ 49ers

40 Josh Downs, Colts @ Bengals

41 George Pickens, Steelers vs. Patriots

42 Jaxon Smith-Njigba, Seahawks @ 49ers

43 Drake London, Falcons vs. Buccaneers

44 Zay Jones, Jaguars @ Browns

45 Jordan Addison, Vikings @ Raiders

46 Jayden Reed, Packers @ Giants

47 Robert Woods, Texans @ Jets

48 Gabe Davis, Bills @ Chiefs

49 Tyler Boyd, Bengals vs. Colts

50 Odell Beckham Jr., Ravens vs.
</item 21>

<item 22>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 22>

<item 23>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
What are the biggest WR injuries in Week 14?

In Week 13, Demario Douglas (concussion) and Rashid Shaheed (thigh) missed game action, and Tank Dell (leg), Amari Cooper (concussion), Christian Watson
(hamstring), and Marquise Brown (heel) exited early from their respective games.

UPDATE: Christian Kirk suffered a groin injury on Monday night, and he figures to be more doubtful than questionable for this week's game against Cleveland. If
he's out, Zay Jones and Tim Jones will see more targets, but with Trevor Lawrence (ankle) also hurt, it's tough to count on either.

UPDATE: Douglas (concussion) has been ruled OUT for Week 14.

As usual, we'll update these WR PPR rankings throughout the week, so check back for the latest player movement and analysis.
</item 23>

<item 24>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
This week might not matter for many fantasy football owners. Some have secured their spot in the postseason; others are well out of contention. However, for
those in must-win situations, this week really matters. It's not the time to have a hole in your lineup at tight end, but if you do, we're here to help with our
Week 14 fantasy TE PPR rankings.

Fortunately, there are several solid sleepers and streaming options despite two tough byes and some big injuries to notable TEs. It's always tough to know who
to trust at this volatile position, so we break down which tight ends have higher floors for those favored to win comfortably and which have higher ceilings for
those who need to really shoot for big points.

Let's get to it and check out the TE landscape for Week 14.

Who are the TEs on bye in Week 14?
</item 24>

<item 25>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
STARTS: Bijan Robinson, Garrett Wilson, Breece Hall, Drake London, Jets DST, Falcons DST

Bijan Robinson, Garrett Wilson, Breece Hall, Drake London, Jets DST, Falcons DST SITS: Kyle Pitts, Tyler Conklin, Tyler Allgeier, both QBs

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Pittsburgh's biggest change in offensive scheme last week involved Kenny Pickett throwing over the middle much more than before (Pat Freiermuth benefitted). But
they also ran the ball effectively at Cincinnati -- Najee Harris had one of his best games this year. Expect more of the same against the Cardinals this week.
Arizona has allowed five touchdowns to tight ends over its past five games but actually has been relatively efficient defending them otherwise.
</item 25>

<item 26>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
They have done a nice job staying close in coverage on those throws.

ZONE COVERAGE: Jacksonville ranks fifth in zone coverage rate (78.5%) and figures to play a lot of it to keep eyes on Josh Allen and deter him from throwing
deep. They did the same to Patrick Mahomes in Week 2.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Dameon Pierce RB HOU Houston • #31

Age: 23 • Experience: 2 yrs.
</item 26>

<item 27>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
The Commanders clearly value the position in their offense, Thomas sees the most throws from Howell, and if Atlanta plays heavy zone coverage then it should
mean a bunch of underneath throws for Thomas to take in. You're hoping for at least seven targets here, which is possible. He's worth the nod over Jake
Ferguson, Dalton Schultz and Tyler Conklin, and it's close in full PPR over Kyle Pitts.

LAST WEEK: Thomas was a big-time benefactor of the Commanders chasing points against Chicago's heavy zone coverage, seeing 10 of his 11 targets once the Bears
built a 17-point lead in the first half. A 24-yard catch on the final play of the first half against prevent defense didn't hurt either.

FALCONS: See the third-most targets per game to tight ends (9.0) which in turn has made them bottom-five in raw stats like total catches and yards allowed.
</item 27>

<item 28>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ DAL DAL -9 O/U 47.5 OPP VS WR 10th PROJ PTS 10.6 WR RNK 31st YTD Stats REC 54 TAR 79 REYDS 575 TD 4 FPTS/G 12.5 Lockett has
seven-plus targets in four of his past five, but he still doesn't have as many as Metcalf's gotten in those games. He also doesn't see nearly as many deep
throws as Metcalf -- literally two targets in his past two games traveled 15 or more Air Yards. Obviously he has potential to always pop off for a big game like
he did against a tough Browns defense five weeks ago, but recent matchups against the Niners, Rams and Ravens weren't kind to Lockett. They were even less kind
to Geno Smith, and this week shouldn't be much different.
</item 28>

<item 29>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The fact that he has 13 total targets in his past three games combined makes it tough to buy into him suddenly getting 10-plus targets this week. The matchup is
also a problem -- the Browns have the defensive backs to hang tight with a slower-than-normal Kupp in man-to-man coverage, which they play in a lot. The Browns
also have the pass rush to force Matthew Stafford to get the ball out quickly and without much distance. I'm not that nervous to bench Kupp for Rashee Rice,
Courtland Sutton and both of the top Seattle WRs.
</item 29>

<item 30>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP vs CAR TB -5.5 O/U 37 OPP VS WR 15th PROJ PTS 12.5 WR RNK 32nd YTD Stats REC 53 TAR 82 REYDS 606 TD 1 FPTS/G 11.2 Remember when Chris
Godwin was known for being a super-reliable SLOT receiver? Those were the days. This season he's lined up wide on 68% of his snaps. Can't help but think that's
caused the massive change in his production this year -- that and his quarterback isn't a serial slot-targeter (or a Hall of Famer). Godwin's been under 60
yards in five straight, hasn't had a game with even 10 PPR points since Week 8 and has one touchdown on the year despite being tied for 12th among all NFL
players in end-zone targets with nine.
</item 30>

<item 31>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Washington could be without its starting center this weekend as it plays host to Miami. Larsen, who missed Wednesday's practice due to a knee injury, was
limited on Friday and is officially listed as questionable. Meanwhile, the Commanders will play Week 13 without Forbes Jr., who has been sidelined since
suffering an elbow injury against the Giants in Week 11. Head coach Ron Rivera called the injury "significant" earlier this week.

Broncos at Texans (-3)

Denver's lone player with an injury designation for Week 13 is Mathis, who did not practice Friday after being limited on Thursday. Outside of him, the Broncos
are healthy heading into this matchup with Houston. Wideouts Jerry Jeudy (groin) and Marvin Mims Jr. (hamstring) were both full participants on Friday after
being limited on Thursday.

Dell logged back-to-back days as a limited participant in practice to end the week and is officially questionable.
</item 31>

<item 32>
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
</item 32>

<item 33>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
THREE WEEKS IN A ROW: With seven or fewer targets, Goedert has failed to get even 50 yards receiving. His target volume dipped from seven in Weeks 2 and 3 to
four in Week 4. Goedert had the same amount of targets as D'Andre Swift.

NICK SIRIANNI: The Eagles coach again said "[w]e need to get him the football," citing that plays designed for him wound up going to other Eagles because of how
defenses played Goedert.

FILM: If there's something teams are doing to take away Goedert, I didn't see it on any of his nine third-down routes last week. Goedert still looked sluggish
in his movements and didn't run a ton of deeper routes, though there were a couple where he did get open but Jalen Hurts had better opportunities elsewhere.
</item 33>

<item 34>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Terrell on Collins and make things a little harder on the Texans. Terrell shadowed a receiver for the first time this season in Week 4, contributing to Calvin
Ridley's modest day. It ultimately would mean multiple bodies on Dell.

Sit Him (Lineup Decisions) Projections powered by Sportsline Kyle Pitts TE ATL Atlanta • #8

Age: 23 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs HOU ATL -2 O/U 41.5 OPP VS TE 18th PROJ PTS 7.6 TE RNK 13th YTD Stats REC 11 TAR 21 REYDS 121 TD 0
FPTS/G 5.8 If Coach Smith is right, then holding on to Pitts in Fantasy makes sense. But it's hard to see the good in his game so far this season beyond him
finding holes in zone coverage.
</item 34>

<item 35>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
BEFORE WEEK 4: The Seahawks, Bengals and 49ers tight ends, including George Kittle, combined for eight non-PPR/19 PPR Fantasy points over three games against
the Rams. L.A.'s pass stats against tight ends were much better before the Colts laid some numbers on them.
</item 35>

<item 36>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
By the end of the game McLaurin's body language seemed dispirited.

COMMANDERS: Throw to their wideouts on 58.7% of their targets, which is below league average (but not as bad as, say, the Falcons at 36.4%). What makes that
number worse is that McLaurin has just one more target (31) than Jahan Dotson (30), who has three more targets than Curtis Samuel (27). They're all nearly even
after five games and each of them has had a turn leading in targets at least once.

FALCONS: Are expected to play a heavy dose of zone coverage, if only because the Commanders have struggled to be explosive against zone coverage this year.
Atlanta also gave a full dose of snaps last week to outside cornerback Jeff Okudah, who wasn't flawless but does offer another sizable body to help against
stronger perimeter receivers. Between Okudah and A.J.
</item 36>

<item 37>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs NO NO -1.5 O/U 42.5 OPP VS WR 14th PROJ PTS 12.8 WR RNK 23rd YTD Stats REC 25 TAR 36 REYDS 467 TD 3 FPTS/G 17.9 I'm not sure the
Saints will back off of their aggressive man-to-man coverage even if Stroud's stats suggest they should, and that could open the door for Collins to pick up
some targets. Unfortunately, neither of his matchups are particularly favorable. I could see Collins getting a lot of short targets and for that reason alone
I'd be OK starting him as a flex option in PPR ahead of Tyler Lockett and Chris Godwin, but behind Jakobi Meyers, Jaylen Waddle and Gabe Davis.

LAST WEEK: Even in a game where the Texans lost Tank Dell to a concussion before the half, Collins couldn't get much going.
</item 37>

<item 38>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
They were even less kind to Geno Smith, and this week shouldn't be much different. Tack on the tough assignment in Stephon Gilmore in coverage (unless the
Cowboys use shadow coverage for the first time this year), and Lockett could be in for another lean night. I don't mind him as a WR3 over Drake London, DeAndre
Hopkins and Diontae Johnson, but I'd start Rashee Rice and Adam Thielen over Lockett.

Sit Him (Lineup Decisions) Projections powered by Sportsline Jake Ferguson TE DAL Dallas • #87

Age: 24 • Experience: 2 yrs.
</item 38>

<item 39>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Steve Spagnuolo's defense has been blitzing its tails off over the past two weeks (nearly 50% of their snaps against the Eagles and Raiders), so the hunch is
they'll apply maximum pressure to that O-line in an effort to make Love more off-target than he's been and, hopefully, get some turnovers. Only one receiver has
had more than 15 PPR points against the Chiefs in their past four games (Jakobi Meyers last week), making it tough to expect grand games from Love's favorite
targets. I'd expect shutdown cornerback L'Jarius Sneed to line up across from Christian Watson the most -- and Jayden Reed the least.
</item 39>

<item 40>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Tua Tagovailoa, Terry McLaurin (low-end WR2), De'Von Achane (low-end RB2), Dolphins DST

Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2), De'Von Achane (low-end RB2), Dolphins DST SITS: Sam Howell (low-end
starter at best), Brian Robinson (desperation RB starter), Logan Thomas (PPR bye-week replacement), Antonio Gibson (PPR bye-week replacement), Commanders DST

Sam Howell (low-end starter at best), Brian Robinson (desperation RB starter), Logan Thomas (PPR bye-week replacement), Antonio Gibson (PPR bye-week
replacement), Commanders DST SLEEPERS: Curtis Samuel (especially PPR), Jahan Dotson (non-PPR only)

Start 'em 10.0-7.00 6.99 - 5.01 5.
</item 40>

<item 41>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
And, tight ends have scored three times in the Vikings past three games.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Flex Starter (Lineup Decisions) Projections powered by Sportsline Tyler Lockett WR SEA Seattle • #16

Age: 31 • Experience: 9 yrs. Weekly Breakdown MATCHUP @ CIN CIN -2.5 O/U 45 OPP VS WR 11th PROJ PTS 11.9 WR RNK 36th YTD Stats REC 17 TAR 27 REYDS 157 TD 2
FPTS/G 11.7 If you start Lockett, you're doing it for his upside and on the hope that Smith can have a minor resurgence against man coverage. It may also come
to pass that the Seahawks are trailing in the game and thus have to throw.
</item 41>

<item 42>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
STARTS: Michael Pittman, Derrick Henry, Zack Moss

Michael Pittman, Derrick Henry, Zack Moss SITS: DeAndre Hopkins (PPR flex at best), Josh Downs (PPR flex at best), Will Levis, Titans DST

DeAndre Hopkins (PPR flex at best), Josh Downs (PPR flex at best), Will Levis, Titans DST SLEEPERS: Colts DST, Gardner Minshew

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

With edge rusher Jaelan Phillips, the Dolphins ranked third in sacks (38) and sixth in pass rush pressure rate (39.9%). Both those numbers were boosted by the
defense's effort against a sagging Jets offense last Friday. Adding veteran Jason Pierre-Paul isn't a cure-all, but he will help the Dolphins from being forced
to blitz a ton week in and week out.
</item 42>

<item 43>
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
</item 43>

<item 44>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
But they float around league average in catch rate allowed (77.8%, below league average) and yards per catch (8.9, better than league average). However, no team
has allowed more first-down receptions to tight ends (17). Keep in mind, this defense hasn't seen tight ends better than Evan Engram, Sam LaPorta or Dalton
Schultz.

COMMANDERS: Throw to their tight ends on 26.3% of their throws, fourth-most in football. But that hasn't always meant a huge target share for Thomas. In Weeks 1
and 5 he saw at least 22.9% of Sam Howell's targets. He left Week 2 after suffering a concussion and was absent in Week 3; then was not part of the game plan
much in Week 4 (7.7% target share against Philadelphia). Perhaps it's all health related and Thomas' volume from last week will stick.
</item 44>

<item 45>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
's pass stats against tight ends were much better before the Colts laid some numbers on them.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline Jordan Addison WR MIN Minnesota • #3

Age: 21 • Experience: Rookie Weekly Breakdown MATCHUP vs KC KC -4 O/U 52.5 OPP VS WR 10th PROJ PTS 9.8 WR RNK 34th YTD Stats REC 13 TAR 20 REYDS 185 TD 2 FPTS/G
10.9 While the Vikings are almost guaranteed to throw a ton against the Chiefs, rolling with Addison remains a risk because he's not the definitive No. 2
target, nor is the matchup any good for him. He's a flex at best who I'd sit for Romeo Doubs, Gabe Davis and Christian Watson.
</item 45>

<item 46>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
COLLINS: It's minor but Collins does see a higher target per route run rate against man coverage (25%) than zone (22.2%). He also has a higher catch rate
(87.5%) and higher receiving average (23.3) against man than zone (65.4% and 17.8). However, he's run nearly four times as many routes against zone coverage
compared to man.

MATCHUPS: It's also worth pointing out that Collins' two really big games came in Week 2 (vs. Indianapolis) and Week 4 (vs. Pittsburgh), both teams that rank in
the bottom-10 in Fantasy points allowed to wide receivers. The Saints are in the top half.

STROUD: Has been outstanding against man coverage all year, completing a higher rate of passes (67.6%) for 3.5 more yards per attempt (11.1) than versus zone
(7.6).
</item 46>

<item 47>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Fantasy Football is all about the matchups. Even though you drafted your team with certain hopes and intentions, your weekly lineup decisions shouldn't be
determined by the order you picked your players in. You need to check who your players play and make sure you've got the right guys in -- and the wrong guys
out.

It's too early to be absolutely sure on which matchups will be easy and which ones will be tough, but we can take some educated guesses based on healthy
personnel, defensive schemes, track records and key details of offenses. The things we know can help us minimize the impact of the things we don't know. This
should lead to better decisions being made.

We'll go through every game and highlight the players who aren't obvious starts and sits (because you don't need to be told to start Justin Jefferson). You
should feel more comfortable starting or sitting players based on the information given, and feeling comfortable with your Fantasy lineup before the games start
is the best feeling in the world.
</item 47>

<item 48>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 48>

<item 49>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Perhaps it's all health related and Thomas' volume from last week will stick.

ZONE COVERAGE: Atlanta usually plays a lot of zone coverage early on and if it works they stick with it for the whole game (Week 5 was an example of that). In
Weeks 3 and 4 against the Lions and Jaguars they played much more man coverage, especially when they needed a fourth-quarter turnover. And because Howell is
much less explosive against zone looks, I expect the Falcons to stick with zone much more this week.

Start Him (Lineup Decisions) Projections powered by Sportsline Kyle Pitts TE ATL Atlanta • #8

Age: 23 • Experience: 3 yrs.
</item 49>

<item 50>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: On the plus side, two of his targets were in the red zone and a third was a deep contested catch for 17 yards. On the downside, he dropped that
fourth target on a short throw.

GOEDERT: Has three red-zone targets and zero end-zone targets this season.

RAMS: Got stung for two touchdowns to tight ends at Indy last week, one on a 35-yard play where Mo Alie-Cox got matched up with DE-turned-OLB Mike Hoecht. The
other was a five-yard toss to Andrew Ogletree, who ran past Hoecht off the snap for an easy touchdown grab. These guys aren't considered primary tight ends like
Goedert, though, so it's unlikely the Rams would cover Goedert the same way as they covered them (which is to say, they barely did).
</item 50>

<item 51>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP vs WAS ATL -2.5 O/U 42.5 OPP VS TE 10th PROJ PTS 10.3 TE RNK 9th YTD Stats REC 18 TAR 32 REYDS 208 TD 0 FPTS/G 7.8 If last week had
been Week 1 then I would have been encouraged by Pitts' performance and would probably keep expectations high. But we've seen four really measly weeks from
Pitts before Week 5, so forgive me for not rushing back to him as a must-start tight end. The truth is that the matchup is a little tougher for him and for
Ridder, plus teams have beaten the Commanders on deep shots to wide receivers, not lumbering tight ends, so I'm not convinced Pitts will maintain his target
share or develop more explosive plays.
</item 51>

<item 52>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least we can say over the past two weeks that the Dolphins have used both backs rather evenly in touches, even though WHERE they've gotten them haven't been
as equal.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Start Him (Lineup Decisions) Projections powered by Sportsline DeAndre Hopkins WR TEN Tennessee • #10

Age: 31 • Experience: 10 yrs. Weekly Breakdown MATCHUP @ IND TEN -2 O/U 43 OPP VS WR 25th PROJ PTS 11.6 WR RNK 30th YTD Stats REC 18 TAR 31 REYDS 216 TD 0
FPTS/G 9.9 It's been cool to bench Hopkins so far this season but we're staring at a terrific matchup for him to take advantage of, just as other receivers have
as recently as last week (Puka Nacua).
</item 52>

<item 53>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 53>

<item 54>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 54>

<item 55>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Brown is also questionable but did not practice Friday due to a knee injury.

Panthers at Buccaneers (-5.5)

Carolina may be down two tight ends this weekend as the club has ruled out Hurst and is listing Tremble as doubtful. This is the third straight game Hurst will
miss. The team will also be without Bell and will likely look to rookie Alex Cook to start. Meanwhile, corner CJ Henderson (concussion) was a full participant
on Friday and carries no designation heading into the weekend.

Tampa Bay is pretty banged up coming into Week 13, particularly on the defense side of the ball. The Bucs will be without key pieces David, Dean, and White when
they play host to Carolina. Meanwhile, Godwin did not practice Friday after being added to the injury report on Thursday with a neck injury.

49ers (-3) at Eagles

Each of the 49ers' questionable players were limited Friday and did not practice the previous two days.
</item 55>

<item 56>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The uncertainty of how the Panthers will operate moving forward with Brown basically calling the offense without any oversight keeps these players are risky
Fantasy propositions, though it helps a lot that the Buccaneers are dealing with injuries that hurt their run defense (Lavonte David and Devin White will miss
the game) and keep their pass defense from improving (Jamel Dean will miss the game).
</item 56>

<item 57>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
It's been a journey back. He's going to get there."

FILM ON SMITH: From what I see, Smith is running shorter routes and is getting open faster than Pitts. That's earned him some targets from Ridder, who seemed to
have better timing with Smith than Pitts. Smith is not drawing any kind of tight coverage like Pitts either, mostly because he's staying closer to the line of
scrimmage. But after the catch Smith has been a handful, breaking three tackles just last week and picking up extra yards with it. He's not fast, but he's
faster than Pitts, and for at least last week and Week 2 it changed the calculus for how the Falcons divvied up targets.

TEXANS: Have had a bundle of troubles with tight ends, giving up an 84% catch rate and 6.76 YAC/reception to the position, both bottom-six in the league.
</item 57>

<item 58>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.
</item 58>

<item 59>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
D.J. Turner is expected to replace Awuzie.

GENO SMITH: Has a 47.8% completion rate and a lowly 4.1 yards per attempt average against man coverage so far this season. He's also barely thrown deep (4.3% of
his attempts went 20-plus Air Yards). A huge reason for the low numbers is because his pass protection has been dreadful. The anticipated return of left tackle
Charles Cross should help him have more time, which in turn helps his receivers.

OUTSIDE WRs: Have caught just 50% of their targets against the Bengals this season, third-lowest in football. However, when those receivers have caught the ball
they've averaged 17.9 yards per catch, which is the third-most in the league. The Bengals defense has really only let up big games to receivers who catch
touchdowns.
</item 59>

<item 60>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Mark Andrews had a great game against Seattle, but even he didn't score. I suspect the Seahawks will have a plan to try and cover Ferguson near the end zone
without having to sacrifice too much elsewhere. He's a low-upside play I'd sit for Juwan Johnson, Gerald Everett and Logan Thomas.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Seven straight quarterbacks to face the Broncos have failed to get to 20 Fantasy points, including Josh Allen once and Patrick Mahomes twice! Denver's forced at
least one turnover from a quarterback in eight straight games and kept quarterbacks to under 7.0 yards per attempt in five straight. Denver also seems to be
playing a lot of heavy zone coverage to deter quarterbacks from throwing deep, mostly Cover-3, and they're also starting to blitz more over the last month than
they did before.
</item 60>

<item 61>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
It can be tough to trust Taysom Hill (vs. Panthers) in full-point PPR leagues, but a matchup against the league's second-worst run defense gives him a high
ceiling regardless of format. He's been more involved as a receiver this year, so he'll get a few catches to go along with the rushing stats, too. With Derek
Carr (concussion) banged up, Hill might throw more this week, too.

Gerald Everett (vs. Broncos) and Isaiah Likely (vs. Rams) have matchups against bottom-six TE defenses, and Cole Kmet (vs. Lions), Jake Ferguson (vs. Eagles),
and Kyle Pitts (vs. Buccaneers) also have above-average matchups that give them higher floors and higher ceilings than usual.

For those in deeper leagues, Tyler Conklin (vs.
</item 61>

<item 62>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
He got open on a handful of third-down plays when he didn't get a target last week.

TANNEHILL: Is the top problem Hopkins has right now. Hopkins' 38-yard deep-ball catch last week could have been as many as 50 yards if Tannehill hit him in
stride, and the gimmick-play touchdown would have obviously been a massive point-getter had Tannehill been slightly more on-target there. There was even a
target into the end zone that was a miscommunication between Hopkins and Tannehill. On the season Tannehill has a 13% off-target rate but over the past two
games that number has ballooned to 18%, which is ... yikes.

TITANS: The second problem Hopkins has right now is a Titans offense that's stayed true to their run-first preference -- at least when they win.
</item 62>

<item 63>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.
</item 63>

<item 64>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
5. As for Cox, he is questionable despite not practicing Friday due to a groin injury.

Browns at Rams (-3.5)

Browns : QB Dorian Thompson-Robinson (concussion), WR Marquise Goodwin (concussion), CB Denzel Ward (shoulder) OUT; C Nick Harris (knee) QUESTIONABLE

: QB Dorian Thompson-Robinson (concussion), WR Marquise Goodwin (concussion), CB Denzel Ward (shoulder) OUT; C Nick Harris (knee) QUESTIONABLE Rams: S Quentin
Lake (hamstring) OUT; OLB Michael Hoecht (knee) QUESTIONABLE

The Browns have officially ruled out Thompson-Robinson due to a concussion he suffered last week against Denver, which will thrust veteran Joe Flacco in as the
starter for Cleveland in Los Angeles.

For the Rams, Hoecht didn't practice the last two days after being a limited participant on Wednesday.
</item 64>

<item 65>
Seahawks-Cowboys Fantasy Start 'Em Sit 'Em Week 13: Should you start Zach Charbonnet, Jake Ferguson on TNF? (Sporting News, 2023-11-30T21:31:00+00:00)
Again, you can't expect more than WR3 production, which is where his overall scoring has been in 2023.

Start: WR Tyler Lockett

While Metcalf is WR34 in terms of total points this season in half-point PPR, Lockett is only two notches higher at WR32. They finished as WR13 and WR18,
respectively, last season, so it's a big dropoff. Lockett also has a decent matchup vs. Stephon Gilmore. He, like Metcalf, remains a volume-based WR3.

WEEK 13 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Sit: WR Jaxon Smith-Njigba

Smith-Njigba can be a bit interesting if you are desperate in a deeper league because he has the matchup on par with the other two wideouts vs. Jourdan Lewis.
</item 65>

<item 66>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
After the injuries to Shaheed and Olave last Sunday it was Alvin Kamara who predictably led the way in targets per routes run (30.8%). Kamara gets a boost both
in the pass game and run game with Lions linebacker Alex Anzalone missing action with a hand injury. For the Lions offense, expect the run game to get a bit
more focus with Saints linebacker Pete Werner sidelined and All-Pro defensive end Cameron Jordan questionable to play.

STARTS: Jared Goff, Amon-Ra St. Brown, Jahmyr Gibbs, Alvin Kamara, Chris Olave (if he passes concussion protocol), David Montgomery, Sam LaPorta, Taysom Hill

Jared Goff, Amon-Ra St.
</item 66>

<item 67>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He's still worth trotting out there over Logan Thomas, Dallas Goedert and Cole Kmet, but keep expectations around 70 yards on six catches.

LAST WEEK: Came alive against the Texans' zone-heavy defense with season-highs in targets (11), catches (seven) and yards (87). This was his second game of his
last three where his target share was over 23%. It was nice to see him as a key part of the offense, but I can't help but notice he played a season-low 53% of
the snaps.

FILM: Two things helped Pitts out: One, he looked a little quicker off the snap and a little faster when running straight. He's still lumbering when he changes
directions in his routes but perhaps he's slowly getting healthier. Two, Desmond Ridder seemed very comfortable throwing into tighter windows to Pitts in zone
coverage as the two had some good timing down, mainly on dig and out-breaking routes.
</item 67>

<item 68>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 68>

<item 69>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
It gets worse: The Panthers offense can't score and their run defense is so bad, leaving pass-negative game scripts for their opponents. Just two teams (the
Saints and Dolphins) have rostered multiple receivers with 10-plus PPR points against the Panthers in 2023. I'd sit Godwin for Josh Downs, Brandin Cooks,
Christian Watson and Curtis Samuel (especially if PPR).

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

I wouldn't expect the Rams offensive line to handle the Browns' defensive front like they did the Cardinals' last week, but it must be mentioned that the Browns
run defense isn't anywhere near as dominant as it was earlier this year.
</item 69>

<item 70>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Armstead has stated that he will suit up on Sunday.

Philadelphia saw the return of tight end Goedert at practice on Friday. He was limited for the session and officially listed as doubtful after practicing for
the first time since suffering a fractured forearm against the Cowboys on Nov. 5. As for Cox, he is questionable despite not practicing Friday due to a groin
injury.
</item 70>

<item 71>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 71>

<item 72>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Sit Him (Lineup Decision) Projections powered by Sportsline Tank Dell WR HOU Houston • #13

Age: 24 • Experience: Rookie Weekly Breakdown MATCHUP @ ATL ATL -2 O/U 41.5 OPP VS WR 8th PROJ PTS 11 WR RNK 27th YTD Stats REC 16 TAR 24 REYDS 267 TD 2 FPTS/G
14 Dell's upside for one huge catch keeps him in play as a middle-of-the-pack flex, but you should be willing to consider other receivers with safer floors and
nearly as much upside. And if the game is low-scoring or a one-sided win for Houston (don't laugh, they just did it last week), it further lowers Dell's
ceiling. I'd rather start Romeo Doubs, Christian Watson and Adam Thielen.

LAST WEEK: The Texans didn't need him!
</item 72>

<item 73>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 73>

<item 74>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Tack on the ascending secondary the Falcons seem to have and the Commanders' diffusion of targets and McLaurin is at best a low-end flex. I'd rather roll the
dice on Drake London, Tyler Lockett or Garrett Wilson.

LAST WEEK: Hopeful that his target edge over his teammates would keep growing, and even in a favorable game script where the Commanders were trailing by three
scores before halftime, McLaurin only saw five targets against the Bears' heavy zone coverage.

FILM: On the 10 third-down routes McLaurin ran last week, he drew extra coverage on four of them, got open on two of them (Howell threw elsewhere) and was a
decoy on Curtis Samuel's touchdown. What happened? Sam Howell was getting rid of the ball quickly and to shorter distances as the Bears were getting some
pressure on him. By the end of the game McLaurin's body language seemed dispirited.
</item 74>

<item 75>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Weekly Breakdown MATCHUP @ LAR LAR -3.5 O/U 40.5 OPP VS WR 3rd PROJ PTS 11.4 WR RNK 23rd YTD Stats REC 47 TAR 86 REYDS 765 TD 2 FPTS/G 12.3 It appears Cooper
will play through a rib injury he suffered last week, and it also appears he'll catch passes from new Browns quarterback Joe Flacco. This is actually a great
thing for Cooper; over his past two weeks his ADOT sat at 7.57 yards, which was roughly half of what it was in his first nine games at 15.42 yards. Clearly, the
Browns wanted Cooper to run shorter routes and make things easier on rookie QB Dorian Thompson-Robinson.
</item 75>

<item 76>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Jonnu even has more red-zone targets than Pitts (by one, three to two). The matchup could create some receiving opportunities, so maybe he catches five passes
for 50 yards, but that's not anything to be excited about. I'd rather start Jake Ferguson, Zach Ertz and Dallas Goedert.

Sneaky Sleeper (Lineup Decisions) Projections powered by Sportsline Jonnu Smith TE ATL Atlanta • #81

Age: 28 • Experience: 7 yrs. Weekly Breakdown MATCHUP vs HOU ATL -2 O/U 41.5 OPP VS TE 18th PROJ PTS 5.2 TE RNK 24th YTD Stats REC 15 TAR 20 REYDS 179 TD 0
FPTS/G 8.2 I feel like Smith is doing what Pitts is supposed to do: get open and make plays after the catch.
</item 76>

<item 77>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Meanwhile, Pipkins was also limited on Friday.

The Patriots will be down a couple of wide receivers as Douglas and Boutte have been ruled out after not practicing on Friday. For Douglas, he missed the entire
week due to a concussion he suffered against the Giants, while Boutte was downgraded on the injury report on Thursday and missed back-to-back days due to an
illness on top of his shoulder injury.

Lions (-4) at Saints

The Lions just designated Hooker for return, who has been rehabbing a torn ACL he suffered at Tennessee last year, this week so the rookie quarterback was not
expected to be activated this week. Meanwhile, Anzalone did not practice at all this week, so his prospects look bleak for Week 13. As for Jackson, he was a
full participant on both Thursday and Friday.

Head coach Dennis Allen told reporters Friday that despite Olave being a full participant in Friday's practice he is still in concussion protocol though has a
chance to play.
</item 77>

<item 78>
Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver (Sporting News, 2023-12-07T22:11:00+00:00)
The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five
catches, 157 yards, and two TDs. CeeDee Lamb (vs. Eagles) and the Cowboys look to avenge their road loss to the Eagles in Week 9, as Dallas' WR1 routinely has
double-digit reception potential.

Amon-Ra St. Brown (@ Bears) was relatively quiet this past week, but he's likely to rebound against a Bears defense that he totaled nine catches for 95 yards
against in Week 11. A.J. Brown (@ Cowboys) reached the century mark for the first time in four games in Week 13, and another high-usage day looks to be in store
for a potential shootout.

Keenan Allen (vs. Broncos) caught just five-of-nine targets last week, but as L.A.
</item 78>

<item 79>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Chiefs (-6) at Packers

Kansas City is relatively healthy coming into Week 13. McKinnon was a limited participant on both Thursday and Friday, so his prospects of playing on Sunday
appear to be trending in a positive direction. As for Bolton, he is still on IR and working his way back from a wrist injury.

Green Bay will be without Jones as the running back continues to deal with a knee injury. Reed returned to practice Friday after missing both Wednesday and
Thursday due to a chest injury, while Alexander -- who has missed six of the Packers 11 games this year -- was limited throughout the week.
</item 79>

<item 80>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Dolphins), Puka Nacua and Tutu Atwell (vs. Browns), Drake London (@ Jets), and Tyler Lockett and Jaxon Smith-Njigba (@ Cowboys) this week.

McLaurin's averaging just 42 receiving yards over his past three games, and a heavy dosage of Jalen Ramsey hinders his Week 13 fantasy value.

Nacua and Atwell see a Cleveland coverage unit who entered Week 12 allowing the fewest fantasy points per game to WRs. Heck, even Cooper Kupp, who hasn't looked
like himself of late, could underperform again.

We can't trust London and the rest of the Falcons passing offense against a well-rounded Jets defense. To make matters worse, he'll be matched up against Sauce
Gardner for most of the day, too.

Lockett and Smith-Njigba might be better suited for the bench in PPR leagues against a lockdown Dallas coverage unit.
</item 80>

<item 81>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
3 wide receiver in the offense behind Justin Jefferson and K.J. Osborn, but even with the donut Addison still has a five-catch lead on Osborn on the season with
68 more yards gained.

THIS WEEK: You can bank on the Vikings reversing course against the Chiefs. Two of the past three offenses against Kansas City wound up throwing at least 71% of
their snaps. The team that didn't? Chicago, who was so badly blown out by halftime that they ran the ball 55% of their second-half snaps just to get out of
Arrowhead.

CHIEFS: Have been stout against receivers all season long, ranking top-five in catch rate allowed (53.4%) and YAC/reception allowed (2.96), and sixth-best in
yards per catch given up (11.1).
</item 81>

<item 82>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Sure, even including the Vikings. Doesn't mean it's not a good look.

NOT SURPRISINGLY: Sanders has two runs of 12-plus yards through four games -- and neither went more than 15 yards. It's not all on him -- the offensive line for
the Panthers barely gave him any daylight last week (and for much of the year) and defenses aren't yet scared of getting beat by Bryce Young's arm.

LIONS: Are a tough matchup, giving up 2.9 yards per rush to running backs on the season. That ranks top-three, as does their yards before contact allowed (0.49)
and runs of 10-plus yards allowed (three). Forgive them for the 2.38 yards after contact per rush allowed -- that's only fifth-best in the league. Only Kenneth
Walker, on the strength of two touchdowns, has had more than five non-PPR and nine PPR points against these Lions.
</item 82>

<item 83>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 83>

<item 84>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
DEEP PASSES: The Commanders have allowed the seventh-most receptions of throws that traveled 15-plus Air Yards this season with 19. Of those 19, only two were
caught by tight ends (Cole Kmet, Dallas Goedert) and neither went for more than 23 yards.

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

Sit Him (Lineup Decisions) Projections powered by Sportsline K.J. Osborn WR MIN Minnesota • #17

Age: 26 • Experience: 4 yrs.
</item 84>

<item 85>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
FILM ON PITTS: There's no change in his explosiveness whatsoever. He's running at a slower-than-expected pace, but the Jaguars still paid him plenty of
attention defensively, creating a double-whammy that left him without many targets. His longest opportunity was a deep ball from Desmond Ridder that was
well-placed and perfectly leapt for by Pitts, but he couldn't secure it to the ground, nor was he separated from cornerback Tyson Campbell at all. Pitts was
best on shorter routes against zone coverage.

ARTHUR SMITH: Said this week that Pitts isn't at 100%, adding " there are certain things, you see him, and he's doing really well right now and there are
certain things that he can't [do well]. It's been a journey back. He's going to get there."
</item 85>

<item 86>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Consider Lockett more of a top-30 receiver who you should sit for Zay Flowers and Gabe Davis but start over Chris Godwin, Nico Collins, Garrett Wilson and Drake
London.

BENGALS: Consistently tailor their defensive coverage to the weaknesses of the offenses they face. So when a quarterback struggles against zone coverage,
they'll play more zone. This worked in their favor in all but one game this year (Ravens, Week 2). Cincinnati also has turned on the juice for their pass rush,
getting a pressure rate of over 40% in each of their past three games, all versus teams with suspect offensive lines.

BENGALS: Are expected to be without top cornerback Chidobe Awuzie, and they also are rotating one of their safety spots between Nick Scott and Jordan Battle.
D.J. Turner is expected to replace Awuzie.
</item 86>

<item 87>
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
</item 87>

<item 88>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Packers) always has a high ceiling as the Giants' de facto No. 1 receiver.

Tyler Higbee (neck) was banged up last week, and he's no sure thing to suit up this week against Baltimore. Even if he does, he's not a recommended play against
the Ravens' top-10 TE defense.

We'll be updating these TE PPR rankings throughout the week, so check back for the latest player movement and analysis.

Fantasy TE PPR Rankings Week 14: Who to start, sit at tight end

Rankings based on full-point PPR formats

Rank Player

1 Travis Kelce, Chiefs vs. Bills

2 T.J. Hockenson, Vikings @ Raiders

3 Sam LaPorta, Lions @ Bears

4 David Njoku, Browns vs. Jaguars

5 Dalton Kincaid, Bills @ Chiefs

6 George Kittle, 49ers vs. Seahawks

7 Isaiah Likely, Ravens vs.
</item 88>

<item 89>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
He's the lone analyst that picks the Bills to beat the Bengals.

Almost a Bengals sweep... 👀 pic.twitter.com/KDh3awFV2y — Sunday Night Football on NBC (@SNFonNBC) November 6, 2023

7:56 p.m.: Before the game, former punter Kevin Huber and former long-snapper Clark Harris are recognized on the jumbotron. The two are Cincinnati legends, with
Huber having played in the most games as a Bengal (216 games) and Harris ranking fourth (202 games). The two are the rulers of the jungle for the game.

7:45 p.m.: Rasul Douglas will be playing, but The Athletic's Joe Buscaglia is reporting Dane Jackson is getting more of the first-team snaps during warmups.

6:50 p.m.: As expected, Hamlin is officially listed as inactive for the game.

6:47 p.m.: Hamlin running out onto the field before the game.

6:29 p.m.
</item 89>

<item 90>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
The hope is we keep seeing that for Smith, especially if the game is competitive like most Falcons games tend to be. He certainly hasn't done anything to lose
his role. I don't mind chancing it with Smith as a streaming tight end, though I just don't have the conviction to start him over Pitts.

LAST WEEK: Was the second time in three weeks where Smith outproduced Pitts, but the first time it was so lopsided. The difference in snaps played and routes
run is marginal -- Pitts did more of both -- but Smith is taking advantage of shorter routes with a quarterback who doesn't like to challenge downfield all that
much. What's more, Smith averaged more yards per catch not just in Week 4 but also in Week 2, and he was behind Pitts by less than a yard in Week 3 when both
had at least eight targets.

FILM ON PITTS: There's no change in his explosiveness whatsoever.
</item 90>

<item 91>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Tyreek Hill (@ Commanders in Week 13) did his thing despite a tough matchup in Week 12, resulting in Miami's WR1 returning to the top spot against a putrid
Washington stop unit. CeeDee Lamb (vs. Seahawks) isn't far behind, as Dallas' WR1 aims to maintain his efficient play against Seattle.

Amon-Ra St. Brown (@ Saints) has been a target hog the past several weeks, seeing at least nine looks in each of his past six games. Don't expect his workload
to decrease in a bounce-back game for the Lions. Keenan Allen (@ Patriots) looks to continue his dominant ways, as Justin Herbert's go-to target routinely has
double-digit target potential.

Mike Evans (vs. Panthers) sits in our top five for Week 13, totaling four TDs over the past three weeks. A.J. Brown (vs.
</item 91>

<item 92>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: The Texans didn't need him! Houston built a nice lead thanks to its defense and held on without having to throw the ball a ton.

FILM: The Steelers really emphasized not getting beat deep by Dell, often using a safety over the top to his side of the field to deter C.J. Stroud from even
thinking about throwing at him. It worked out for the Steelers because Dell had zero deep targets, and it worked for the Texans because Nico Collins beat soft
coverage a bunch including on a 24-yard deep ball (Dell drew triple coverage!) and on his long touchdown score (Dell got a safety to cheat over to his side).

FALCONS: Should be able to deploy the same type of defensive coverage with a safety cheating over to Dell's side of the field. They could also stick outside
cornerback A.J. Terrell on Collins and make things a little harder on the Texans.
</item 92>

<item 93>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
For 2023, the NFL league average of QB throws to running backs is 18.2%, so he's just below the current league average.

Sit Him (Lineup Decisions) Projections powered by Sportsline Amari Cooper WR CLE Cleveland • #2

Age: 29 • Experience: 9 yrs. If you start Cooper, you should not only have lower expectations but basically pray to the Fantasy Gods for some garbage production
or ludicrous target volume. It's hard to trust Cooper to come through for a reliable stat line, and there's a lot of downside for a single-digit-point week.
He's at best a low-end No. 3 receiver; I'd rather start Tyler Lockett, Chris Godwin or Drake London.

LAST GAME: With rookie Dorian Thompson-Robinson at quarterback, Cooper saw six targets and caught one for 16 yards. Half of the six targets were uncatchable.
</item 93>

<item 94>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
I think he'll continue to see high-value targets near the end zone and, hopefully, a small uptick in targets. I'd start him as a No. 2 receiver over Terry
McLaurin, any Broncos wideouts and any Packers wideouts.

LAST WEEK: Was so close to having a monster game. He was the designed target on a trick play and saw the ball fall just past his outstretched hands in the end
zone. He also had two targets inside the 10-yard line (his first two of the season) and two end-zone targets including the aforementioned trick play (his first
two of the season). The heavy dose of targets inside the 10 and into the end zone are positive signs that the Titans are trying to get him going as a playmaker.

FILM: Hopkins might be 31 years old but he moved really, really well, creating separation with outstanding footwork and having just enough speed to stay a step
ahead of defenders.
</item 94>

<item 95>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
That included in the second half when he saw one target and couldn't catch it. He definitely wasn't shadowed by anyone and was almost never double-teamed,
though there were a couple of routes where a safety was in proximity. It could have had more to do with the Texans attacking the middle of the field against the
Falcons' zone coverage.

SAINTS: Have played the fifth-fewest snaps of zone coverage this season and have ignored tendencies some quarterbacks have had through the early portion of the
season. It's just what they like to do. Stud cornerback Marshon Lattimore has not followed receivers much this season, playing 87% of his snaps at right
cornerback (to the left of the quarterback). Their other primary outside corner, Paulson Adebo, has allowed a 50% catch rate on the season with minimal YAC.
</item 95>

<item 96>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Weekly Breakdown MATCHUP @ CHI MIN -2.5 O/U 44.5 OPP VS WR 20th PROJ PTS 10.7 WR RNK 33rd YTD Stats REC 13 TAR 26 REYDS 166 TD 2 FPTS/G 8.3 No doubt there's
some potential here, but some hunches keep me from falling for Osborn this week. One: I think the Vikings will try to get their run game going since they don't
have Justin Jefferson. Two: Jordan Addison is far more explosive and T.J. Hockenson has been a much more relied-upon weapon. They should have more targets than
Osborn. And the expected zone coverage Chicago will play further will limit Osborn unless he sees more targets than expected.
</item 96>

<item 97>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
The Commanders offensive line has had some ugly showings lately against the Cowboys (four sacks, 49% pass rush pressure rate allowed), the Giants (four sacks,
32.7% pressure rate allowed) and the Seahawks (three sacks, 46.8% pressure rate allowed). Somehow these results are better than what they were allowing through
much of the first two months of the year. Sam Howell getting the ball out faster has been a positive in this regard but a negative for his numbers in three of
his past four games (the one game he did really well in was sort of fluky). The matchup is a good test for the Dolphins pass rush post-Phillips, one that it
should pass without giving up much in the way of deep completions.

STARTS: Tyreek Hill, Jaylen Waddle, Raheem Mostert, Tua Tagovailoa, Terry McLaurin (low-end WR2),
</item 97>

<item 98>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 98>

<item 99>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Still, it's smart to check your other options before trusting Goedert.

The same goes for Evan Engram (@ Browns). He's generally more reliable in PPR leagues, but he's facing a Browns' defense that allows the fewest FPPG to TEs.
Both Engram and Pat Freiermuth (vs. Patriots) are in tough spots against top-five TE defenses.

What are the biggest TE injuries in Week 14?

We already mentioned Goedert (forearm) and Schultz (hamstring), but another TE who could be back this week is Darren Waller (hamstring). The Giants really have
no reason to bring back their star TE if they don't feel like he's 100 percent, but it's something to monitor throughout the week, as Waller (vs. Packers)
always has a high ceiling as the Giants' de facto No. 1 receiver.
</item 99>

<item 100>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Bengals at Jaguars (-8.5), Monday

Bengals : CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT

: CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT Jaguars: CB Tyson Campbell (hamstring), RB Travis Etienne (ribs), TE Brenton
Strange (foot), TE Luke Farrell (toe) QUESTIONABLE

The Bengals got good news on the injury front ahead of Monday's game against the Jaguars. Wide receiver Tee Higgins (hamstring/ankle) and linebacker Logan
Wilson (ankle/foot) are set to play, according to head coach Zac Taylor. Higgins and Wilson were full participants in practice. Cornerback Cam Taylor-Britt hurt
his foot in practice and is out.
</item 100>
=== END ITEM id=c2bef936c71d456e ===

=== ITEM id=6f717c941f585c54 ===
Gold sentences and facts:
Sentence 0: But by the age of 16, she is public property – incapable of going outside without being mobbed, and her love of singing and dancing now a lucrative resource.
  Fact 0.0: By the age of 16, the person described is public property.
  Fact 0.1: By the age of 16, the person described is incapable of going outside without being mobbed.
  Fact 0.2: By the age of 16, the person described's love of singing and dancing is now a lucrative resource.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker described it as almost a very strange, subtle clue to the media.
  Fact 1.1: The clue signals that the media does not know everything that happened in that relationship.
  Fact 1.2: The speaker says something really major and traumatic could have happened to her that the media does not know about.
  Fact 1.3: The speaker told NPR this.

Retrieved items:

=== END ITEM id=6f717c941f585c54 ===
