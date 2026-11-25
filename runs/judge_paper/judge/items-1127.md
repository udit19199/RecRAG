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

=== ITEM id=de8d7d84567a08b2 ===
Gold sentences and facts:
Sentence 0: 9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.
  Fact 0.0: At 9:47 p.m., the Bears answer right back and sack Dobbs.
  Fact 0.1: Sweat and Brisker are in there to take Dobbs down.
Sentence 1: The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in points (16.7) and 10th in yardage (310 per game).
  Fact 1.0: The Ravens defense has allowed the fewest points per game this season, 16.1.
  Fact 1.1: The Ravens defense has allowed the second-fewest total yards this season, 4,030.
  Fact 1.2: San Francisco's defense ranks second in points, at 16.7.
  Fact 1.3: San Francisco's defense ranks 10th in yardage, at 310 yards per game.

Retrieved items:
<item 1>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
In their past three games, they gave up 4.1 yards per rush to the Ravens (just 6.3% of their runs went 10-plus yards), 7.8 yards per rush to the Steelers thanks
to a long TD run by Jaylen Warren (23.8% of their runs went 10-plus yards) and 4.6 yards per rush to the Broncos (18.5% of their runs went 10-plus yards).
Playing more snaps because the Browns offense has been inept isn't the reason for it either because they struggled with the run in the first half against all
three teams (over 5.3 yards per carry allowed against each). It's silly to say the Rams passing game is struggling when Matthew Stafford just threw four
touchdowns, but Cooper Kupp looked like he played hurt last week and Puka Nacua didn't look that much better.
</item 1>

<item 2>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
He goes 18 yards here to put Minnesota near midfield.

10:29 p.m. FUMBLE — That's exactly what CAN'T happen if you're the Bears. Fields loses the football, Sheldon Day recovers, and the Bears waste an opportunity to
make this a two-score game.

Josh Metellus forces a fumble and the Vikings take over 😤



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/fKnE0eTxsP — NFL (@NFL) November 28, 2023

10:27 p.m. — Fields finds Mooney for a big third down conversion as the Bears get firmly into field goal range.

10:25 p.m. INTERCEPTION — Josh Dobbs throws his FOURTH interception of the night. Justin Jones nearly had it, but it was Kyler Gordon who came away with it.
</item 2>

<item 3>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
— And now this one is officially over. The Bears pick up their third win of the season, while Carolina drops to 1-8 and faces more tough questions about Bryce
Young and the offense.

11:08 p.m. — Bagent hits Mooney for the first down on 3rd & 7, and the Panthers now won't be able to stop the clock.

11:06 p.m. — Pineiro's attempt is SHORT from 59 yards out. The gamble fails for Frank Reich. The Panthers will hope to get the ball back.

11:04 p.m. — Young is very nearly intercepted on 3rd & 10, and the Panthers say they've seen enough of the offense. Eddy Pineiro is attempting a 59-yard field
goal to tie the game.

11:00 p.m. — Miles Sanders gets himself wide open on fourth down to extend the drive.
</item 3>

<item 4>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They also seemed invigorated for Jonathan Gannon on defense and special teams. Look for the Cardinals to finish strong and play a whole lot of spoiler.

30. New England Patriots 2-8 (28)

The Patriots tried to win a game by hiding Mac Jones and their quarterbacks in general. They almost pulled it off with the running game dominating and
short-handed defense making a lot of plays, but they didn't see the results in Germany ahead of a tough bye for Bill Belichick.

31. New York Giants 2-8 (30)

The Giants can just flush away this season without Daniel Jones and Tyrod Taylor. The injuries go to their offense early and remained relentless, and now their
defense is falling apart, too. The question is, will Brian Daboll get a massive mulligan after his rookie playoff leadership?

32.
</item 4>

<item 5>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Brissett didn't throw downfield as much as Walker, contributing to a higher ADOT for Walker, but not by much (8.8 to 8.3). Both quarterbacks were right around
6.5 yards per pass attempt.

49ERS: Oh yeah, the matchup isn't great. No team allows fewer yards per catch to wideouts than the Niners (9.7), and they're thrown at a ton by offenses that
have to play from behind. Only three receivers -- Puka Nacua (20 targets), Marquise Brown (10 targets) and Michael Wilson (two touchdowns) have had more than 15
PPR points against them. And only one other receiver -- Tutu Atwell (nine target) -- had more than 12 PPR points.
</item 5>

<item 6>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They can win with offense or defense. Dan Campbell is coach of the year.

4. San Francisco 49ers 6-3 (8)

The 49ers had a surprise offensive get-well game in Jacksonville with Brock Purdy leading the way and all his key supporting weapons doing damage, even without
Christian McCaffrey scoring. The biggest development was the defense going back to dominating a good offense.

5. Baltimore Ravens 7-3 (2)

The Ravens' offense is just fine with the running and passing of Lamar Jackson, but he still can't afford to make big mistakes, which he did vs. the Browns. The
defense needs to pick up the pieces, too, stat, ahead of facing the Bengals on a short week.

6. Miami Dolphins 6-3 (6)

The Dolphins got a much-needed bye to figure out how to beat the better teams in the league.
</item 6>

<item 7>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 7>

<item 8>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Now let's see whether the Panthers can get the passing game going at all.

Second straight third down that ends in a scramble for a first down by Bryce Young#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/mC5txmYqzP — NFL (@NFL) November 10, 2023

8:23 p.m. — Nice awareness by Bryce Young to run for the first down on 3rd & 7, and the Panthers add five extra yards on an illegal contact penalty against
Chicago.

8:21 p.m. — Donte Jackson comes through with a nice pass breakup on third down, and the Bears will punt shortly after the quick first down.

8:19 p.m. — D.J. Moore instantly makes an impact against his former team with a 16-yard grab.

8:17 p.m. — Bagent and the Bears offense will start with the ball, and we are underway!

8:09 p.m.
</item 8>

<item 9>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
It's 4th & 10, and Chicago is lining up to got for it.

8:36 p.m. — Sacks are killing the Bears in this game. Multiple Vikings defenders wrap up Fields for a 12-yard loss and quickly set Chicago back to 2nd & 22.

8:35 p.m. — The Bears' offense is picking up where it left off. After Fields takes off for 14 yards, he hits Roschon Johnson for 22 yards to take Chicago into
Vikings territory.

8:31 p.m. — That's what the Bears got Montez Sweat to do. Sweat sacks Dobbs on third down to force a quick three-and-out and get the ball right back in Fields'
hands.

8:27 p.m. — A great start for the Bears ends with zero points as Cairo Santos misses a 48-yard field goal. That's a deflating sequence for Chicago.

8:25 p.m. — The rookie Ivan Pace Jr.
</item 9>

<item 10>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their defense should be getting a little better to support their often explosive offense. They should feel good about a Week 11 rebound vs. the Raiders.

7. Pittsburgh Steelers 6-3 (9)

The Steelers are playing the way Mike Tomlin wants, with the running game rolling to big totals with Jaylen Warren and Najee Harris and the defense doing what
it needs, more so vs. pass than run, in key moments. You can't argue with the overachieving results.

8. Cleveland Browns 6-3 (10)

The Browns proved they can win games with defense in another way, with big plays. They also have more offensive pop when Deshaun Watson plays well and the
running game doesn't slump. Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9.
</item 10>

<item 11>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
3.

We have two sets of teammates in our top 10 this week: Detroit's David Montgomery and Jahmyr Gibbs (@ Bears) and Miami's Raheem Mostert and De'Von Achane (vs.
Titans). Both duos have such solid workloads in high-flying, run-first offenses — and such juicy matchups — that we can't imagine any of these studs finishing
outside the elite eight. DMont and Gibbs combined for 193 yards and two TDs when they faced Chicago in Week 11. Tennessee just surrendered 31 points to Indy, so
we know Miami will be dominating the time of possession and running early and often against Will Levis and company.

Jets fans will be hoping for some Breece Lightning from stud second-year back Breece Hall (vs. Texans), and we think they will be rewarded for their patience.
</item 11>

<item 12>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
The Pats have surrendered 116 scrimmage yards per game and nine total TDs to RBs this season, so we like Barkley to pick up where he left off this Sunday (drive
up that free-agency value, 'Quon!).

You don't see running mates cracking the top five very often, but we had to make the exception for David Montgomery and Jahmyr Gibbs (vs. Packers). Entering
Week 11, Green Bay had surrendered nearly 140 all-purpose yards per game to RBs. Between Detroit's o-line, the Lions' high-flying offensive attack, and the
ruthless 1-2 punch DMont and Gibbs have been this season, the feast could be on very early for Dan Campbell's guys.

Take a collective sigh of relief, Tony Pollard (vs. Commanders) owners! TP looked solid against Carolina last week, averaging 5.1 yards per carry and finally
getting into the end zone for the first time since Week 1.
</item 12>

<item 13>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The Bears defense comes up big with a 4th down stop



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/yndUqwgIbf — NFL (@NFL) November 28, 2023

9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.

9:46 p.m. — Alexander Mattison has drawn the ire of Vikings fans all season, but he just got Minnesota's second half of to a booming start with a 21-yard run.

End of first half: Bears 3, Vikings 3

9:32 p.m. FIELD GOAL — Greg Joseph drills it from 34 yards out, and this mess of a game is all tied at three entering the half.

9:30 p.m.
</item 13>

<item 14>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
The fans don't like it.

9:26 p.m. — There's a welcomed sight for Lions fans: Goff hits Jameson Williams for a 22-yard gain. Williams was held without a catch despite six targets in
Baltimore last week.

9:25 p.m. — Crosby is doing so much more than just getting to the quarterback tonight. He wraps up Gibbs, the third time tonight he's had an explosive play on a
ball-carrier.

9:21 p.m. TOUCHDOWN — That was a statement drive by the Raiders. Jacobs looked like the dominant runner he was last season and takes it in himself for the
game's first touchdown.

9:20 p.m. — Davante Adams doesn't have a single catch tonight, but Jacobs has 12 touches. Right now, it's starting to work. The Raiders are set up with 1st &
Goal.

9:19 p.m. — The Raiders are quickly into the red zone after another nice gain for Josh Jacobs.
</item 14>

<item 15>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Not far behind sits Tua Tagovailoa (vs. Titans), who shredded the Commanders' defense to the tune of 280 yards and two TDs in Sunday's blowout. Another juicy
matchup awaits against a lackluster Tennessee coverage unit. Dak Prescott (vs. Eagles) remains a top-three play in Sunday's rematch versus the Eagles, a team he
torched for 374 yards and three TDs in a narrow 28-23 loss in Week 9.

Jalen Hurts (@ Cowboys) and the Eagles look to get back on track in a pivotal NFC East clash in Dallas, facing a Cowboys defense that let Geno Smith post
top-end QB1 numbers against them last week. Patrick Mahomes (vs. Bills) is in line to put up strong numbers in a potential shootout against Buffalo, and
although his receiving corps has its issues, the reigning MVP is still a top-five play this week.

Lamar Jackson (vs.
</item 15>

<item 16>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
The Panthers have allowed the most rushing TDs (14), the third-most points per game (28.4), and the fourth-most rushing yards per game (139.4) during their 1-7
start. The Colts will cruise in this one.

FINAL SCORE PREDICTION: Colts 27, Panthers 17

Seahawks (+6) at Ravens

Baltimore's defense has been incredible of late, and the Ravens seem to be flying under the radar (so to speak) at 6-2. However, the sportsbooks have clearly
taken note of their awesome run, giving the NFC West-leading Seahawks a full touchdown on the road. This seems like a great opportunity for us to rock with the
'dog, considering Seattle has won two straight contests and gone 2-1 on the road (including a big win over the high-flying Lions in Detroit).
</item 16>

<item 17>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.
</item 17>

<item 18>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
He's already missed from 48 tonight.

End of third quarter: Bears 6, Vikings 3

10:14 p.m. — The Bears are starting to look respectable on third down again, this time completing to D.J. Moore to move deeper into Vikings territory. Chicago
still might need another first down to give Cairo Santos a fair shot at a field goal.

10:08 p.m. INTERCEPTION — Dobbs throws his third interception of the night! Jaylon Johnson deflected it, Edwards caught it, and interestingly, the Bears have
less favorable field position than they would've if it had fallen incomplete.

And Edwards follows it up with the Bears 3rd INT of the game ‼️
</item 18>

<item 19>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
32. Carolina Panthers 1-8 (31)

The Panthers found better defense and special teams to keep them in the game in Chicago, but Bryce Young still can't unlock a limited passing game and the
offensive line and running game hasn't help nearly enough.
</item 19>

<item 20>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 20>

<item 21>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Vikings 10, Broncos 3

9:08 p.m.: FIELD GOAL — Greg Joseph drills a 34-yarder to give the Vikings a 10-3 lead with 8:09 to play in the half. The Vikings' offense stalled out in the
red zone as Joshua Dobbs was sacked for an 11-yard loss on 3rd & Goal.

9:06 p.m.: Joshua Dobbs has some magic in him. In a cloud of smoke in the pocket, Dobbs found TE TJ Hockenson on a 29-yard gain to pick up a first down on 3rd &
2. The big play brings the Vikings into the red zone.

9:00 p.m.: Another three-and-out for Russell Wilson and the Broncos. On 3rd & 3, Wilson was under heavy pressure and threw the ball away as he was nearly
brought down for a sack.
</item 21>

<item 22>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:04 p.m. — Dobbs is picked off again! That's two interceptions and just nine passing yards for the Vikings quarterback, but this one doesn't look like his
fault - Jordan Addison saw it bounce off his hands. It's Jaquan Brisker with the pick.

9:01 p.m. — Another penalty gifts the Vikings a first down... Both teams are piling on the mistakes.

8:58 p.m. — A roughing-the-passer call goes against the Bears and gives the Vikings some breathing room early in the drive.

8:54 p.m. — Fields overthrows Darnell Mooney on third down, and Mooney gets destroyed in a collision. The Vikings will get the ball back, still somehow only
trailing by three. Fields will want that one back.

8:50 p.m. INTERCEPTION — Jaylon Johnson intercepts a throw from Dobbs! That just hung up there.
</item 22>

<item 23>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He also ducked under an end-zone target instead of leaping for it. It's worth noting that a 36-yard crosser he ran against the Chargers flies in the face of
this as he separated from man coverage with speed and technique, then turned upfield and ran outside before racing past a defender and diving for the pylon.

OSBORN: Has a 50% catch rate on the year and that includes whether he faces zone coverage or man coverage. He sees more targets per route run against man
coverage and averages more yards per catch and yards after catch against man coverage. Over his career, targets do seem to come more frequently against man
coverage and he has been more explosive with them.

BEARS: Play zone coverage at the fourth-highest rate of any defense this season (79.2%). And because Cousins is much more explosive of a passer against man
coverage, expect Chicago to keep up their zone defense.
</item 23>

<item 24>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:30 p.m. — The Bears' defense has suddenly morphed back into an elite unit on these last two plays, forcing intentional grounding by Dobbs and pressuring him
heavily on the following play. It's 3rd & 24 for the Vikings, who might just settle for a gain of a few yards to set up the field goal.

9:27 p.m. — Another first down by K.J. Osborn drives the Vikings into Bears territory, and the Vikings suddenly find themselves 13 yards from the end zone after
Kyler Gordon is called for interfering with T.J. Hockenson. The Vikings have a chance to take the lead with 38 seconds on the clock.

9:25 p.m. — What an effort by Brandon Powell to turn around and make a 28-yard grab while falling to the ground. That's the biggest offensive play of the night
for the Vikings.

A perfect catch from Brandon Powell on a perfect pass
</item 24>

<item 25>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Viewers in Canada can stream every NFL game via DAZN.

MORE: How Joshua Dobbs got his 'Passtronaut' moniker

Bears vs. Vikings start time

Date: Monday, November 27

Monday, November 27 Start time: 8:15 p.m. ET

Monday's game between the Bears and Vikings will kick off at 8:15 p.m. ET (7:15 p.m. local time) from U.S. Bank Stadium in Minneapolis, Minn.

The Bears haven't beaten the Vikings since 2020, with Minnesota 6-1 against Chicago over the last four seasons. The Vikings are 8-3 at home against the Vikings
since 2012.
</item 25>

<item 26>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Week 10 in the NFL had some of the league's best teams not playing. But that didn't stop the excitement, with five teams winning on walk-off field goals and
more games coming right down to the wire.

With that, there were some surprising results and expected domination. Some favorites dug deep to win. Some underdogs turned in shocking upsets.

Putting the latest wins and losses into a blender, here's Sporting News' updated look at how all teams stack up against each other, 1-32 going into Week 11:

NFL WEEK 11 PICKS: Straight up | Against the spread

NFL power rankings Week 11

1. Philadelphia Eagles 8-1 (previous week: 1)

The Eagles enjoyed a bye trying to get a little healthier for the more difficult stretch run in a top-heavy NFC as they have tough games at the Chiefs (the
Super Bowl 57 rematch), vs. the Bills, vs.
</item 26>

<item 27>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
STEELERS: Run defense has improved considerably over their past two games, giving up just 3.5 yards per tote and 2.22 yards after contact per run against the
Raiders and Texans. That could open up more targets for Flowers.

STEELERS: Haven't seen many WR screens but the ones they've defended have yielded a pathetic 13.75 YAC/reception. This is an exploitable play for Flowers.

DEEP BALL: The Steelers haven't been so hot here either, allowing 16 of 33 targets of 15-plus Air Yards to be completed for 22.5 yards per catch and for three
touchdowns. This is an exploitable play for Flowers, who has caught 4 of 5 deep passes for a sizzling 136 yards.

HISTORY: In five career games against Pittsburgh, Lamar Jackson has thrown four career touchdowns. At least all of them went to his wideouts.
</item 27>

<item 28>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
8:44 p.m.: The Vikings' defense responds with a stop of their own. A quick three-and-out for Russell Wilson and the Broncos. Minnesota will take over from its
own 35 after a 12-yard punt return from Brandon Powell. Broncos lead 3-0 with 5:01 to play in the first quarter.

8:38 p.m.: Joshua Dobbs had the Vikings' offense moving but the Broncos' defense came up with a big stop on 3rd & 8. Minnesota had picked up 28 yards in six
plays but Dobbs was nearly picked off on third down, forcing a punt. Denver will take over from its own 25-yard line after a touchback.

8:32 p.m.: Joshua Dobbs was evaluated in the medical tent after taking a big hit on the last drive, but he did not miss a snap at QB for the Vikings.

Broncos 3, Vikings 0

8:30 p.m.
</item 28>

<item 29>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
THE JETS: Have given up over 16 Fantasy points to a quarterback ONCE all season -- Dak Prescott in Week 2. The other three they faced, including Josh Allen in
Week 1 and Patrick Mahomes last Sunday, were under 16 Fantasy points.

PASS RUSH: The Jets tend to get to the quarterback without blitzing (19.5% blitz rate is bottom-five), getting some pressure on 41.6% of their snaps so far this
year, which is third-best in the league. And it's been consistent -- their only "low" game was at Dallas and they still bugged Prescott 32.5% of the time.
That's a lot.

O-LINE: Denver's front five allows the sixth-highest pass rush pressure rate this year at 39.5%. The Bears had pressure on 32.3% of their dropbacks against
Denver last week, their second-highest of the season (they did blitz a bunch).
</item 29>

<item 30>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
Can QB Jalen Hurts and the Eagles offense finally bounce back?: Philadelphia's defense isn't the only unit that needs to rebound, as the offense has struggled
mightily over the last three games, with Hurts' ineffectiveness playing a significant role. After averaging the third-most points in the NFL over the first 11
games (28.2), the Eagles have scored four offensive touchdowns during their current skid, three of which came via the "tush push."

The Eagles may have difficulty throwing against a solid Giants pass defense, though they could alleviate some of their offensive woes by running the ball
efficiently. New York gives up 131.6 rushing yards per game, the fourth-most in the league.

Does Giants QB Tommy DeVito have any magic left?: The Giants may have a terrible offense, but this unit has operated at its best under DeVito, an undrafted
rookie and native of Cedar Grove, New Jersey.
</item 30>

<item 31>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Stroud leading their offense and now they've found some running game, too. The defense is just making enough plays to put them in position to win high-scoring
games. Houston is a legitimate playoff contender under rookie coach DeMeco Ryans.

14. Cincinnati Bengals 5-4 (7)

The Bengals are battling some key injuries, leaving them much less room for error. Unfortunately, Joe Burrow made key mistakes and the defense wilted against
run and pass at home in the loss to the Texans.

15. Buffalo Bills 5-5 (12)

The Bills fell right into the trap game against the Broncos at home, losing because of more mistakes from Josh Allen. The defense can't tolerate those mistakes
with all of its injuries.

16.
</item 31>

<item 32>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Carolina has serious questions to answer.

10:05 p.m. — Young gets a few yards back but is sacked by Yannick Ngakoue on third down. The Panthers will punt again.

Yannick Ngakoue gets his team off the field on 3rd down 😤#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/JPt1xc3b93 — NFL (@NFL) November 10, 2023

10:04 p.m. — Bryce Young is now lining up in his own end zone after Miles Sanders was dropped for a six-yard loss.

10:01 p.m. — Punt. Again.

9:57 p.m. — The Panthers get a grand total of one yard (-4 including a delay of game penalty) and punt it away less than a minute into the first half.

End of first half: Panthers 10, Bears 9

9:40 p.m.
</item 32>

<item 33>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Good luck, check back often, and enjoy the remainder of the first week of December!

Which RBs are on bye in Week 14?

After a six-team byepocalypse in Week 13, only the Cardinals and Commanders will be off this coming weekend. That takes lead backs Brian Robinson Jr. and James
Conner off the table, plus Antonio Gibson for those in deep leagues.

WEEK 14 STANDARD RANKINGS:

QBs | WRs | TEs | D/STs | Kickers

One thing that stinks for fantasy is that both Washington and Arizona have been defensive turnstiles. Many of the stud running backs have tough matchups in Week
14, but it's only right that many of the elite backs must beat the best defenses to lift fantasy squads to the big dance.

WEEK 14 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy RBs in Week 14?
</item 33>

<item 34>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
That just hung up there. The Bears have been all over Dobbs and the Vikings' offense early on. A taunting penalty after the play hurts Chicago's field position.

Bears 3, Vikings 0

8:44 p.m. FIELD GOAL — The drive fizzles out, and the Bears opt to kick the field goal on 4th & 3. It's good. The Bears are out to an early 3-0 lead, but
they've left plenty of points on the board.

End of first quarter: Bears 0, Vikings 0

8:39 p.m. — What a play by Fields to get the ball to a wide open Kmet! The Vikings brought intense pressure, allowing Kmet to get wide open and go for 24 yards.
The Bears are in business.

8:38 p.m. — 2nd & 22 becomes 2nd & 27, but the Bears make up 17 yards on two plays.
</item 34>

<item 35>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Bears 6, Vikings 3

9:58 p.m. FIELD GOAL — A questionable play call on third down as Fields throws behind the line of scrimmage to Moore, who gets nothing. Cairo Santos is able to
extend the Bears' lead with a field goal, though.

9:56 p.m. — Fields runs right up the middle for a first down and a bit more as the Bears look to get into field goal range (or better).

9:54 p.m. — Roschon Johnson picks up a first down to keep the Bears' drive alive. He's looked impressive in spurts tonight.

9:50 p.m. — Woah! Kevin O'Connell goes for it on 4th & 7, but Hockenson only gets six yards. The play looked good, the Bears' defense looked better. It's
Chicago's ball with favorable field position.

The Bears defense comes up big with a 4th down stop
</item 35>

<item 36>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 36>

<item 37>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
could be back in play as a handcuff stash.

Bears veteran D'Onta Foreman injured his ankle in Week 11. He could miss some time now that Khalil Herbert and Roschon Johnson are both back in action for
Chicago. The best runner on the Bears, however, is QB Justin Fields when he's active.

Texans back Dameon Pierce returned to practice late last week, but he remained out with his ankle injury against the Cardinals. We could see him return to the
field this weekend, but for now, we're keeping him off our rankings until we hear more positive news. Either way, it seems Devin Singletary is the new RB1 in
this breakout offense.

We'll be updating these standard RB rankings all week, so check back for the latest player movement.
</item 37>

<item 38>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Bills, vs. 49ers and at the Cowboys in the next four. They still look like the conference team to beat as they have proved they still win in many ways
around Jalen Hurts.

2. Kansas City Chiefs 7-2 (3)

The Chiefs had a pretty good week off given they jumped back into the AFC's top seed and home-field advantage spot with the Ravens losing. They already are
rolling defensively and expect them to figure out a lot more around Patrick Mahomes after the key Andy Reid bye.

3. Detroit Lions 7-2 (5)

The Lions are becoming like the Eagles, and that's what makes them the second-best team in the NFC. They can win shootouts with Jared Goff, Amon-Ra St. Brown
and their passing game. They also can dominate with the running game. They can win with offense or defense. Dan Campbell is coach of the year.

4.
</item 38>

<item 39>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Buccaneers (DK: $3,000 | FD: $4,700 | SD: 1.8x)

NFL DFS Picks Week 15: D/ST sleepers, values for DraftKings, FanDuel

Los Angeles Rams vs. Commanders (DK: $3,100 | FD: $4,300)

The Rams got destroyed by MVP candidate Lamar Jackson and the high-flying Ravens in Week 14, but push that out of your memory banks. Before that game, Sean
McVay's squad had surrendered 20 or fewer points in four straight contests. Aaron Donald and company also have 17 sacks and six takeaways over their past six
games, so they'll be licking their chops looking across the line at Sam Howell this weekend. No QB has more interceptions than Howell, who has also taken 10
more sacks than any other signal-caller this season.
</item 39>

<item 40>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
PURDY: Has hit at least 20 Fantasy points in 8 of his 9 regular-season starts, and it really should be 9 for 9 because of that first touchdown that was ruled a
lateral back in Week 2. If you include the two playoff games he completed, Purdy has delivered at least 20 in 9 of 11 games. And in 5 of those 9 he hit at least
23 Fantasy points including each of his past two games.

COWBOYS: Dallas' defense lost top cornerback Trevon Diggs and they're still red hot. They lead the league in pass rush pressure rate this season (55.6%) and
they're in the top-10 in blitz rate (31.9%). It's that pressure that's helped the Cowboys rank in the top-five in completion rate allowed (57.4%), passing
touchdowns allowed (two) and missed tackles on pass plays (just five, that's best in football).
</item 40>

<item 41>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
The Ravens would win that spread bet because the margin of victory (0) was narrower than 1.5.

Related Pages:
</item 41>

<item 42>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 42>

<item 43>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears cut the deficit to 7-3.

8:44 p.m. — A false start by Cody Whitehair erases what would've been a good 49-yard field goal...the Bears will keep Cairo Santos out there.

8:43 p.m. — Chicago will try a field goal from 49 yards out after the drive fizzles out.

8:41 p.m. — Foreman has been the Bears' best weapon tonight. He rips off an 11-yard run and now has 43 scrimmage yards in this first quarter.

8:40 p.m. — The Bears rattle off two first downs on their first three plays, with Moore and Foreman grabbing passes from Bagent to take Chicago over midfield.

Panthers 7, Bears 0

8:36 p.m. TOUCHDOWN — Ihmir Smith-Marsette returns the punt 79 yards for a touchdown!
</item 43>

<item 44>
Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-19T23:00:04+00:00)
Denver punted to Minnesota's 25, where Joshua Dobbs and the offense will take over.

Vikings 7, Broncos 3

8:56 p.m.: TOUCHDOWN! Joshua Dobbs, are you kidding me?! Dobbs scrambled out right and looked like he had nowhere to go. The QB shed a tackle on the sidelines
and regained his composure, finding TE Josh Oliver wide open in the end zone for a touchdown after the play broke down.

End of first quarter: Broncos 3, Vikings 0

8:53 p.m.: The Vikings' run game is marching the offense down the field. Spell RB Ty Chandler has collected 29 total yards on this drive while starting RB
Alexander Mattison has picked up 27 yards. Minnesota is down to Denver's 4-yard line as the first quarter comes to a close.

8:44 p.m.: The Vikings' defense responds with a stop of their own.
</item 44>

<item 45>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
He's the lone analyst that picks the Bills to beat the Bengals.

Almost a Bengals sweep... 👀 pic.twitter.com/KDh3awFV2y — Sunday Night Football on NBC (@SNFonNBC) November 6, 2023

7:56 p.m.: Before the game, former punter Kevin Huber and former long-snapper Clark Harris are recognized on the jumbotron. The two are Cincinnati legends, with
Huber having played in the most games as a Bengal (216 games) and Harris ranking fourth (202 games). The two are the rulers of the jungle for the game.

7:45 p.m.: Rasul Douglas will be playing, but The Athletic's Joe Buscaglia is reporting Dane Jackson is getting more of the first-team snaps during warmups.

6:50 p.m.: As expected, Hamlin is officially listed as inactive for the game.

6:47 p.m.: Hamlin running out onto the field before the game.

6:29 p.m.
</item 45>

<item 46>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
It looked like that score might be the difference, particularly after Fields fumbled away the ball again on the next possession, but the Bears' defense put the
ball back in his hands in the final minutes. Fields delivered.

Facing 3rd & 10 outside of field goal range, Fields hit an open D.J. Moore for a 36-yard gain. At that point, all the Bears needed to do was kneel and put Cairo
Santos in position to win the game. Santos drilled the 30-yard attempt with 10 seconds remaining, and that would seal it for Chicago.

MORE: Vikings playoff chances after loss to Bears

As he embarks on a crucial stretch that could determine his future with the Bears, Fields flashed the good and the bad. He was excellent on Chicago's opening
drive, albeit one that ended without points, but missed throws and costly turnovers allowed the Vikings to hang around.
</item 46>

<item 47>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
— Clark Dalton

AFC North

BALTIMORE RAVENS | QB Lamar Jackson: Jackson has inserted himself into the league MVP conversation during his team’s four-game win streak. The 2019 NFL MVP has
thrown for 1,954 yards and nine touchdowns and posted 440 yards and five touchdowns rushing. He is the reason, once again, Baltimore is a Super Bowl contender.

CLEVELAND BROWNS | Defensive end Myles Garrett: Garrett could be leading the race for the Defensive Player of the Year. The No. 1 overall pick in 2017 is tied
for the second-most sacks (9.5) and most forced fumbles (four) in the league. He also blocked a field-goal attempt in a Week 7 win over the Colts. He's the
ringleader of a defense that allows the fewest yards per game (234.8).

CINCINNATI BENGALS | QB Joe Burrow: Battling a calf injury, Burrow had a rough start to the season.
</item 47>

<item 48>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
MATCHUPS: Each top-target earning wide receiver that plays mostly outside (like McLaurin) has found at least 10 PPR points in every game against the Bears. That
includes Rashee Rice in Week 3 (7 targets, 5-59-0), Courtland Sutton in Week 4 (5 targets, 3-27-1), Romeo Doubs in Week 1 (5 targets, 4-26-2) and Mike Evans in
Week 2 (8 targets, 6-171-1). Note that these receivers combined to catch 18 of 25 targets (72%) against Chicago.
</item 48>

<item 49>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
10:37 p.m. — A neutral zone infraction pulls the Panthers a bit closer, and they will line up to go for it on 4th & 6...

End of third quarter: Bears 16, Panthers 10

10:32 p.m. — The Panthers run it with Chuba Hubbard on 1st & 30. They're not showing much interest in letting Bryce Young air it out, even while trailing in a
lost season.

10:31 p.m. — It's 1st & 30 for the Panthers after back-to-back penalties. That's one way to erase great field position.

10:29 p.m. — A less than impressive punt by Trenton Gill sets up Carolina well inside Bears territory. If there was ever a time for the Panthers' offense to get
something going...

10:28 p.m.
</item 49>

<item 50>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 50>

<item 51>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
As we do every week, we're deep-diving the running back position and analyzing all the stats, matchups, and injuries before breaking down our rankings. So,
let's get to it and survey the RB landscape for the final week of the fantasy regular season.

Which RBs are on bye in Week 14?

After a whopping six teams were on bye last week, only the Commanders and Cardinals will be off in Week 14. That takes Brian Robinson Jr. and James Conner off
the table, but it doesn't hurt us much beyond that.

WEEK 14 PPR RANKINGS:

QBs | WRs | TEs | D/STs | Kickers

It does hurt a bit to have two below-average run defenses off the table, as Washington and Arizona have both ranked among the 10 most generous teams to RBs this
season.
</item 51>

<item 52>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
But the reeling defense and pass protection continue to be un-winnable assets.

24. Atlanta Falcons 4-5 (20)

The Falcons were forced into another QB change because of injury. They found a few more answers offensively but that position still made it an overall mess. The
defense also doesn't seem to be well prepared to different styles of offenses.

25. Green Bay Packers 3-6 (25)

The Packers saw some improved work from Jordan Love and his young weapons as they spread the ball well in the passing game, but the run defense woes and key
holes in coverage have all but doomed their playoff hopes.

26. Los Angeles Rams 3-6 (26)

The Rams are hoping to find a few more answers during a bye, as Sean McVay has seen his good coaching job earlier in the season go by the wayside with key
offensive injuries.
</item 52>

<item 53>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Purdy was off-target on 0.0% of his throws.

FANTASY: Has hit at least 20 Fantasy points in 9 of his 10 regular-season starts, and it really should be 10 for 10 because of a touchdown that was ruled a
lateral back in Week 2. If you include the two playoff games he completed, Purdy has delivered at least 20 in 10 of 12 games. And in six of those 10 he hit at
least 23 Fantasy points including each of his past three games.

BROWNS: Have played the least amount of zone coverage snaps of any team in football this season, and they've done it consistently. There isn't a game this
season where they've played even 65% of zone coverage snaps, and there have been only three quarters when they've called more than 70% of their snaps in zone
coverage.
</item 53>

<item 54>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Rams) and think he has the potential to smash Los Angeles at home this week. The speedster has quietly racked up over 300 yards and two TDs over the Ravens'
past four games, and with Lamar Jackson facing Aaron Donald and a still-talented pass-rush, John Harbaugh's squad might turn to the rookie to move the chains
between the 20s. All we need is one or two big chunk gains to make him a worthwhile start — he can run friggin' 22 miles per hour!

WEEK 14 DFS:

DK lineup | FD lineup | Best values | Best stacks

Who are the biggest potential fantasy RB busts in Week 14?

While he just barely cracks our top 25 thanks to his passing-down usage, James Cook (@ Chiefs) doesn't thrill us at Arrowhead against the pissed-off defending
champs.
</item 54>

<item 55>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
With the Ravens, Bills, Vikings, Raiders, Giants, and Bears idle in Week 13, that means Davante Adams, Stefon Diggs, Justin Jefferson, DJ Moore, Zay Flowers,
Jakobi Meyers, and Jordan Addison are among the most notable wideouts omitted from our rankings. Deep-league owners will also be without Odell Beckham Jr., K.J.
Osborn, Gabe Davis, Jalin Hyatt, and Khalil Shakir.

WEEK 13 PPR RANKINGS:

QBs | RBs | TEs | D/STs | Kickers

Unfortunately, we also won't get to pick on the poor WR defenses of the Giants and Vikings, but it's a relief to see the Ravens' shut-down secondary on the
sidelines.

WEEK 13 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy WRs in Week 13?
</item 55>

<item 56>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
MORE: Inside the Cowboys historic offensive explosion vs. Giants

11. Seattle Seahawks 6-3 (13)

The Seahawks needed every ounce of their offense around Geno Smith to rally to beat the Commanders, but the get-well performance worked well for him, Kenneth
Walker and the wide receivers. The defense could be more consistent.

12. Minnesota Vikings 6-4 (14)

The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running
game for Kevin O'Connell and the defense overall under Brian Flores. Their 0-3 and 1-4 starts are distant memories as the Vikings look for a sixth straight win
in Week 11 vs. Denver.

13. Houston Texans 5-4 (16)

The Texans are looking great with C.J. Stroud leading their offense and now they've found some running game, too.
</item 56>

<item 57>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
When it mattered most, though, Fields was poised and sealed the Bears' first win over an NFC North opponent since 2021.

While Chicago's hopes are still extraordinarily distant at 4-8, the Vikings saw their playoff chances take a hit on Monday night. Minnesota (6-6) dropped to the
No. 7 seed in the NFC with the loss, with the Packers, Rams and Saints all just a half-game behind in the wild-card race. While a win would have put the Vikings
in position to compete for the division crown down the stretch, just holding onto a playoff spot will now be a battle for Kevin O'Connell's squad.

MORE: Latest updates on Justin Jefferson's return date

The Sporting News tracked live updates from the "Monday Night Football" game between the Bears and Vikings. Check out all of the key moments you might have
missed.

Bears vs.
</item 57>

<item 58>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They need Matthew Stafford to help save face the rest of the way or the season will look a lot like 2022.

27. Tennessee Titans 3-6 (27)

Will Levis is going through rookie lumps after his dazzling debut as the brutal offensive line woes exacerbated by injuries are getting to him and the entire
Titans offense. The Titans still need to keep starting him, because they can't overcome those systemic problems or those in overall pass defense.

28. Chicago Bears 3-7 (29)

The Bears don't care how they win, but they should be pleased their running game and defense bailed out their limited passing game with Tyson Bagent to secure
another win in Week 10. They will hope to finish strong with Justin Fields soon.

29. Arizona Cardinals 2-8 (32)

The Cardinals got an immediate spark for their offense with Kyler Murray and James Conner back.
</item 58>

<item 59>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 59>

<item 60>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:25 p.m. — The rookie Ivan Pace Jr. finally gets to Fields and drops him for a five-yard loss. It'll be 3rd & long for the Bears.

8:24 p.m. — Fields throws on the run to find Roschon Johnson for a first down, and he quickly follows up with a 13-yard pass to Khalil Herbert. The Bears are
driving, and Fields doesn't have an incompletion yet.

8:21 p.m. — The Bears have taken it into Vikings territory with a string of short gains. For Chicago, that's progress.

8:19 p.m. — Fields finds Moore for a short gain and the first 1st down of the night.

8:16 p.m. — The Bears will start with the ball on the road, looking to build early momentum.

8:00 p.m. — Will the Josh Dobbs show continue tonight?
</item 60>

<item 61>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Lions need an answer for their ugly pass defense -- since their Week 9 bye they've been mostly great against the run (3.1 yards per carry allowed, one RB to
get 15-plus PPR points) but they rank bottom-six in completion rate allowed (68.4%), yards per attempt allowed (8.0), passing ADOT against (8.84 yards) and
defensive Expected Points Added (EPA) per dropback (-0.40, which is league-worst). And of their second-most eight passing touchdowns given up in three games,
five have come on passes of 15-plus Air Yards, which is tied for most in the league. They should catch a break with field stretcher Rashid Shaheed (thigh)
missing the matchup for New Orleans; Chris Olave looks like he will play.
</item 61>

<item 62>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The 23-year-old ranks first in the NFL in QBR (76) and touchdown passes (29) while ranking third in passing yards (3,795), all of which are especially
impressive considering he's coming off offseason elbow surgery.

Jackson might not be posting the ridiculous numbers he did during his 2019 MVP campaign, but the adjustment in his play style in his first season under OC Todd
Monken has led to sustainable, winning football. Nevertheless, Baltimore needs an upset win for Jackson to capture another MVP Award, as he has thrown for 3,105
yards and 17 touchdowns while adding 741 yards on the ground with five scores this season.

49ers RB Christian McCaffrey: If Purdy wasn't playing at such a high level — and the MVP wasn't a quarterback-driven award — it'd be difficult to argue against
McCaffrey taking home the honors.
</item 62>

<item 63>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Justin Jones nearly had it, but it was Kyler Gordon who came away with it. Everything the Vikings have done on the offensive side of the ball tonight looks like
a chore.

10:24 p.m. — A third down catch by Jordan Addison extends the drive for Minnesota. Addison only has 25 yards tonight.

10:23 p.m. — Ty Chandler gets the ball on second down and gets nothing. Chandler has not looked very effective tonight with 11 yards on five touches.

Bears 9, Vikings 3

10:18 p.m. FIELD GOAL — After Santos missed from 48 earlier, he drills it from 55. It's 9-3, Cubs Bears.

10:16 p.m. — Byron Murphy comes in to break up a third down pass by Fields, and Santos will come out for a long 55-yard field. He's already missed from 48
tonight.
</item 63>

<item 64>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Cover 7 | Monday A daily NFL destination that provides in-depth analysis of football’s biggest stories. Each Monday, Mike Sando breaks down the six most
impactful takeaways from the week.

The San Francisco 49ers entered Week 6 riding a 15-game winning streak and looking unstoppable with a young quarterback, Brock Purdy, who was making the game
look easy. They exited with a 19-17 defeat at Cleveland, injuries to key players and, for the first time this season, a question mark or two.

Advertisement

With the previously unbeaten Philadelphia Eagles also losing ugly Sunday, this edition of the Pick Six column addresses key questions facing Super Bowl
favorites with one-third of the 18-week regular season nearly complete.

So far this season, we’ve seen the Miami Dolphins set NFL records for offense, only to lose by four touchdowns in their lone game against a true contender.
We’ve seen the Kansas City Chiefs win with defense, sometimes in spite of their offense. We’ve seen the Buffalo Bills incur devastating injuries.
</item 64>

<item 65>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
8:09 p.m. — Perhaps the most interesting storyline tonight will be whether Al Michaels can keep a brave face throughout what might not be the most appealing
game.

8:00 p.m. — We're going to keep spirits high tonight despite a matchup of teams with a combined three wins. All football is good football! Let's see how long
this mentality lasts...

7:46 p.m. — It's 52 degrees and clear in Chicago right now, so certainly not as cold as it could be in the Windy City this time of year. Still, it's a rare
colder-weather game for Bryce Young.

7:30 p.m. — The Bears will be donning orange tonight, including an alternate orange helmet.

7:10 p.m. — D.J. Moore is back facing his former team tonight, eight months after he was traded to the Bears in the deal that sent the No. 1 pick to Carolina.

6:50 p.m.
</item 65>

<item 66>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least 9 of those 15 throws went into the end zone.

PASS RUSH: Bengals opponents have started to sniff out Burrow's issues and have come after him. The Browns and Titans had a pass rush pressure rate between 38&
and 42% in their wins over Cincy with high blitz rates (at least 31% each). Compare that to the Rams and Ravens, who barely brought pressure or blitzes and
either lost or nearly lost to the Bengals.

CARDINALS: Are dead-last in blitz rate this season (14%) and third-to-worst in pass rush pressure rate (28%). They actually had nine sacks after Weeks 1 and 2
but have had three since against the Cowboys and 49ers. Arizona seems to prefer using just four rushers to affect the quarterback, something that should be
helpful to Burrow this week.
</item 66>

<item 67>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 67>

<item 68>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
We’ve seen the Buffalo Bills incur devastating injuries. Here we’ll address 10 teams with the shortest Super Bowl odds, running through what I think and what
NFL insiders have to say.

The full Pick Six menu this week:

• Unanswered questions for contenders

• When “playing to win” goes too far

• Championship roster, rookie kicker

• Do they make wristbands for coaches?

• Giants, Vikings and winning too soon

• Two-minute drill: Wackiest win in a while

GO DEEPER The Eagles' 'tush push' play is taking over the NFL. But is it working?

1. Where do the top 10 current Super Bowl contenders stand one-third of the way through the season? Let’s hit key questions facing the Chiefs, Eagles, 49ers,
Bills, Bengals, Cowboys, Ravens, Lions, Dolphins and Jaguars.

You can scroll through the latest Super Bowl future odds from BetMGM below.

• San Francisco 49ers: What do they have in Purdy?
</item 68>

<item 69>
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
</item 69>

<item 70>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
1 pick to Carolina.

6:50 p.m. — Standout edge rusher Brian Burns is out with a concussion, dealing a major blow to the Panthers' defense.

Brian Burns, CJ Henderson and DJ Chark among inactives.https://t.co/mL8EYWQKTs — Carolina Panthers (@Panthers) November 9, 2023

6:35 p.m. — Bryce Young threw an interception in only one of his last five games before crashing back to Earth with three picks against the Colts last week.
More than ever, it feels like this game will be won by the quarterback who can avoid turnovers. Bagent has thrown five picks in his last two starts.

6:15 p.m. — Tyson Bagent will get the start for the Bears tonight, so it'll be an undrafted rookie facing a rookie No. 1 pick. Despite that, Chicago is favored.

Bears vs. Panthers start time

Date: Thursday, Nov.
</item 70>

<item 71>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
DeVito has thrown for 858 yards, seven touchdowns, and one interception in five games since being named New York's starting quarterback, posting the
13th-highest passer rating in the NFL during that stretch (97.3).

Baltimore Ravens (11-3) at San Francisco 49ers (11-3) | 8:15 p.m. ET

Potential Super Bowl sneak peek and MVP implications: Easily the most hyped game of the week, with the top seed in each respective conference squaring off in
what could be a potential Super Bowl LVIII preview as well as the decisive clash for the MVP Award.

According to oddschecker, 49ers QB Brock Purdy is listed as the heavy favorite to be named league MVP (-230), while Baltimore signal-caller Lamar Jackson holds
the second-shortest odds (+525).

Purdy, the final pick of the 2022 NFL Draft, has seen his career unfold as if it were a fairy tale that just keeps getting more remarkable.
</item 71>

<item 72>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
It was the first time he had a double-digit ADOT and more than six targets. He lined up wide on the majority of snaps (66.7%) and hitches were still his route
of choice but there was more diversity there than before. It's also notable Flowers did this while sharing the field with Odell Beckham and Rashod Bateman back
on the field.

TITANS: Run defense started the season strong but got smashed last week by Zack Moss and the Colts to the tune of 6.3 yards per carry and four explosive runs
(they had allowed two total headed into the game). Moss beat a stacked box on his 56-yard TD scamper. And even when Richardson wasn't on the field the Colts
averaged 4.7 yards per run against Tennessee.
</item 72>

<item 73>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Brown, Jahmyr Gibbs, Alvin Kamara, Chris Olave (if he passes concussion protocol), David Montgomery, Sam LaPorta, Taysom Hill SITS: Derek Carr, both DSTs

Derek Carr, both DSTs SLEEPERS: Juwan Johnson

Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

The Falcons defense finally had a strong showing against the Saints last week, but they got a little lucky because so many players from New Orleans left with
injuries. In the three games prior their defense had a knack for making mediocre offenses look amazing (Will Levis' breakout game, Josh Dobbs' first game with
the Vikings, Kyler Murray's first game with the Cardinals).
</item 73>

<item 74>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 74>

<item 75>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
Could this be the weekend that Keaton Mitchell (vs. Rams) gets back into RB2 range for the Ravens? Los Angeles might find itself spying Lamar Jackson and Gus
Edwards so often that the speedy rookie busts a couple of big runs. Mitchell has racked up 330 yards and two TDs since he broke out four games ago, but the
majority of that damage (134 yards and a TD) came in the initial breakout against Seattle. He has still yet to see 12 touches in a game, something we can't
imagine will be the case by the end of the regular season.

We also think Roschon Johnson (vs. Lions) could have a chance to take over lead-back duties in Chicago, as both Khalil Herbert and D'Onta Foreman have battled
injuries and inconsistent touches this season.
</item 75>

<item 76>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Now they'll take on a Bengals offense that was borderline lucky to even score 10 points last week against the Steelers. Two of Ja'Marr Chase's four receptions
were tipped by Steelers first, then the Bengals' longest pass play was on a designed dump-off pass to Joe Mixon for 39 yards. Jake Browning had his moments
evading the pass rush but his arm strength is just not very good. Expect the Jaguars to bring plenty of heat to Browning, then lean into Travis Etienne on the
ground to bring home a win.
</item 76>

<item 77>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Start 'em 10.0-7.00 6.99 - 5.01 5.00 - 0 Sit 'em

In three games since getting Chase Young, the Niners have allowed 30 points total, yielded just 3.3 yards per rush to RBs, 5.9 yards per attempt to QBs,
pressured quarterbacks at a 43.2% rate (without blitzing much!), accumulated 15 sacks and picked off four interceptions. As many problems as the Eagles present
with their offense, the Niners figure to be just as menacing with their collective defense. That will make things tough for the Eagles, whose defense played
over 90 snaps last Sunday and could be without a couple of starters in the front seven, and whose offense could be forced into an uncomfortable game script if
the Niners build a lead.
</item 77>

<item 78>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
The Ravens will win it with a big game from Lamar Jackson.

Pick: Ravens 30, Chargers 23

Monday, 8:15 p.m. ET (ABC/ESPN, fubo)

These two are coming off tough losses, with the Bears blowing a huge lead late. It was more damning for the Vikings, a team that had been surging. Minnesota's
defense will be the difference here as they get the best of Justin Fields. Vikings take it.

Pick: Vikings 24, Bears 14
</item 78>

<item 79>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
Can't say the same against the run as the Redbirds rank in the bottom five in pretty much every single run defense metric except yards after contact per rush
(where they're league average). A running back has scored 17 or more PPR points in four of their past five against the Cardinals, and eight in 12 games have hit
at least 15 PPR points. With Pickett nursing an ankle injury we could see the Steelers run the ball aplenty.
</item 79>

<item 80>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
TOUCHDOWNS: Harris has zero. Somehow it gets worse: He has one carry inside the 15-yard line. The Steelers have run the second-fewest red-zone plays (18, one
more than New England) and second-fewest plays inside the 5-yard line (two, one more than New England). The Steelers have played four games! This is nuts.

RAVENS: Have yet to allow a rushing score to a running back this year. They've been otherwise solid against RBs, holding them to 4.0 yards per carry on the
season and roughly league-average in other metrics including missed tackles (15).
</item 80>

<item 81>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Plus the Browns just saw the Cowboys play 90% of their snaps in zone coverage and Purdy cooked them like BBQ chicken.

PURDY: Has been sensational against both man and zone coverages this season. His completion rate and yards per attempt are noticeably higher against zone, but
five of his seven passing touchdowns have come versus man coverage. However, Purdy's been at his worst when pressured (like most quarterbacks), completing 50%
of his throws for 6.7 yards per attempt with a gaudy 15.9% off-target rate. These stats are actually better than they were last week.

BROWNS: Rank third in pass rush pressure rate at 41.5% and blitz at an above-average 29.3% rate. Granted, the Cowboys ranked higher than that last week and got
beat anyway.
</item 81>

<item 82>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
His off-target rate in the fourth was 9.1%, which was closer to what he had averaged in his first three games (13.6%).

BRONCOS: Have allowed season-best Fantasy numbers to each of the past three quarterbacks they've played this season. They're a terrific matchup for anybody
right now. Remember this when we talk about the Jets.

COMMANDERS: Sport a slightly below-average pass rush pressure rate thanks to only one game all year with a rate over 30% (52.7% at Denver, and Wilson still had
a great game even if you take away a Hail Mary touchdown catch). They rank seventh in sacks (13) and 13h in pressures (53), which again floats around league
average.

BEARS: Rank league-worst in pass rush pressure rate allowed at 47.4% of their snaps -- nearly one out of every two drop backs is pressured.
</item 82>

<item 83>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
It was also his third game out of four with four or fewer targets.

BOOM/BUST INDEED: Davis has a massive ADOT of 16.8 yards (top-10 among 96 receivers with at least 10 targets this season), but he also has a minuscule target
share of 13.6% and a target per route run rate of 13.4% (86th out of those 96 receivers). He's clearly a role player with a big-time skill-set that isn't leaned
on too much.

JAGUARS: Statistically are top-three in catch rate allowed to wideouts on throws of 15-plus Air Yards (29.4%) with seven passes defensed, but dead-last in yards
per catch (43.8!) and YAC/reception (15.0!) on those passes that are completed. They have done a nice job staying close in coverage on those throws.
</item 83>

<item 84>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Sure, even including the Vikings. Doesn't mean it's not a good look.

NOT SURPRISINGLY: Sanders has two runs of 12-plus yards through four games -- and neither went more than 15 yards. It's not all on him -- the offensive line for
the Panthers barely gave him any daylight last week (and for much of the year) and defenses aren't yet scared of getting beat by Bryce Young's arm.

LIONS: Are a tough matchup, giving up 2.9 yards per rush to running backs on the season. That ranks top-three, as does their yards before contact allowed (0.49)
and runs of 10-plus yards allowed (three). Forgive them for the 2.38 yards after contact per rush allowed -- that's only fifth-best in the league. Only Kenneth
Walker, on the strength of two touchdowns, has had more than five non-PPR and nine PPR points against these Lions.
</item 84>

<item 85>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
the Patriots. At least Jonathan Taylor is doing his normal things and the defense is playing well again while Gardner Minshew looks meh.

18. Los Angeles Chargers 4-5 (17)

The Chargers got well offensively with Justin Herbert, Keenan Allen and Austin Ekeler against the Lions. But the defense went back to crumbling vs. big pass
plays and added wilting against the run. The pressure is right back on Brandon Staley with a trip to Green Bay next.

20. Tampa Bay Buccaneers 4-5 (23)

Baker Mayfield and the offense is back on track to move the ball well after a brief midseason slump. They're still not lighting it up, but the Bucs are back to
playing smart complementary football with their defense rebounding and still might have found a long-term QB answer.

21.
</item 85>

<item 86>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
The defense can't tolerate those mistakes with all of its injuries.

16. Las Vegas Raiders 5-5 (22)

The Raiders are hot, having won two consecutive games for interim coach Antonio Pierce, bringing up flashbacks to the playoff push they made with Rich Bisaccia
in 2021. They are right back in the AFC wild-card race with the Dolphins, Chiefs, Vikings ahead.

17. New Orleans Saints 5-5 (15)

The Saints will go into a bye not knowing what do at quarterback. with Derek Carr further hurting his right shoulder and needed to be checked for a concussion.
Jameis Winston did provide a spark, too little too late at Minnesota. But the division lead is precarious with poor QB play overall.

19. Indianapolis Colts 5-5 (19)

The Colts stopped scoring at least 20 points in every game during their ugly, run-heavy victory in Germany vs. the Patriots.
</item 86>

<item 87>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
He is regulated to pass from the pocket, but you are taking away the biggest weapon the offense had when you do that. Odell (Beckham) is not that guy anymore.
Who are you truly afraid of on that offense? Nobody but Lamar, and if you are not using Lamar in a certain capacity, he becomes just like the rest of these
quarterbacks around the league — hit or miss.”

• Jacksonville Jaguars: Are they ready for prime time?

What I think: Trevor Lawrence’s knee injury was the big concern coming out of Sunday. If he’s out, expectations shift dramatically. More than the other AFC
contenders, the Jaguars seem early in their life cycle, still finding their way. That makes them a little harder to trust in big spots but also leaves greater
space for growth. Let’s hope those test results on Lawrence’s knee come back clear.

Exec comment: “I like Jacksonville. Doug (Pederson) has a championship pedigree. This team is relatively young. They are scrappy.
</item 87>

<item 88>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Or Denver. Or Chicago. You get the gist. The Cardinals pass rush shouldn't be too dangerous and Burrow, now over a month into his season, figures to eventually
show some signs of improvement with his health. In no way is he a must start, and there's plenty of good reason to go with Trevor Lawrence, Russell Wilson, C.J.
Stroud and even Joshua Dobbs over him, but there's some mild appeal that should at least keep him in lineups over Dak Prescott and Daniel Jones.

LAST WEEK: Looked like a quarterback playing on one leg. Burrow didn't get much into his throws, didn't challenge downfield much, got the ball out as fast as he
could and was sensitive to pass rush pressure. Even when the game was in garbage time Burrow couldn't get much going.
</item 88>

<item 89>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Arizona has been just as inefficient, entering Week 13 ranking 30th in dropback EPA.

WEEK 14 STANDARD RANKINGS:

RBs | WRs | TEs | D/STs | Kickers

Who are the best fantasy QBs in Week 14?

Despite Buffalo being on the outside looking in for a playoff berth, Josh Allen (@ Chiefs in Week 14) has come through in fantasy, sitting as our QB1 for Week
14. Yes, the Chiefs' defense has done its part in keeping QBs in check this season, entering Week 13 allowing the fifth-fewest fantasy points per game (FPPG) to
the position, but Allen has performed well against Steve Spagnuolo's defense in the past. He's averaging 276.4 passing yards per game with a strong 14:2
TD-to-INT ratio over five career games against KC.

Not far behind sits Tua Tagovailoa (vs.
</item 89>

<item 90>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Bears 16, Panthers 10

10:13 p.m. TOUCHDOWN — Foreman runs it in for the score! That's the first offensive touchdown of the night, and it gives the Bears their first lead of the
night.

10:12 p.m. — A well-designed play is broken up by Donte Jackson in the end zone after the Bears pick up the first down.

10:11 p.m. — The Bears are inching their way toward the goal line...it's 3rd & 1 from the five.

10:10 p.m. — Foreman runs 11 yards for a first down, and the Bears are already in field goal range thanks to strong starting field position.

10:06 p.m. — Outside of a 45-yard completion to Mike Strachan, Young is 7-of-16 for 24 yards. It's tough to recall any positive moments for the Panthers'
offense tonight aside from that one play.
</item 90>

<item 91>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
The Ravens defense has allowed the fewest points per game this season (16.1) and second-fewest total yards (4,030), while San Francisco's unit ranks second in
points (16.7) and 10th in yardage (310 per game).

The 49ers defense might be slightly inferior statistically, but it could have more of an impact in this matchup. San Francisco has racked up an NFL-leading 25
turnovers and has given up the second-fewest rushing yards (1,252), which should help slow down the team with the league's most effective running game.
</item 91>

<item 92>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears and Panthers entered Thursday night looking for progress, not playoff contention. If any team achieved that goal, it was Chicago.

It wasn't an aesthetically pleasing game, but that was never the expectation. The Bears grinded out a 16-13 victory behind a terrific defensive performance, a
solid running game, and a few timely throws by Tyson Bagent.

The Panthers jumped out to an early lead on a punt returned for a touchdown by Ihmir Smith-Marsette, but they never found the end zone again. The only offensive
touchdown of the night came in the third quarter, when D'Onta Foreman rumbled his way into the end zone to give Chicago a 16-10 lead.

NFL WEEK 10 PICKS: Straight up | Against the spread

As it turned out, 16 points was all the Bears needed.
</item 92>

<item 93>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
1 pick in March.

MORE: Why Frank Reich's job might already be at risk

The Bears (3-7) are hoping to get Justin Fields back in time for their Week 11 game in Detroit, but they've gotten a shot in the arm from Bagent, who improved
to 2-2 as a starter with the win. The victory on Thursday was a win-win for Chicago, as the franchise owns the first-round pick of the Panthers (1-8).

The Sporting News tracked live scoring updates and highlights from Bears vs. Panthers on "Thursday Night Football." Check out all of the key moments you might
have missed.

Bears vs. Panthers final score

1 2 3 4 F Panthers 7 3 0 3 13 Bears 3 6 7 0 16

Bears vs. Panthers results, highlights from Thursday Night Football

(All times Eastern)

Final: Bears 16, Panthers 13

11:11 p.m.
</item 93>

<item 94>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
BEARS: Are one of the easiest matchups for a quarterback. They rank in the bottom-seven in completion rate allowed (70.1%), yards per catch allowed (12.2),
Yards After Catch per reception (YAC/reception) allowed (6.26, second-most) and missed tackles (52). One more factor: Chicago ranks poorly in pass rush pressure
rate, blitz rate and has just two sacks all season.

HOWELL: When not pressured has completed 73.8% of his throws for 7.2 yards per attempt with a 9.7% off-target rate. None of these are particularly impressive --
they're all about league-average -- but that's still a plus for Howell considering this week's matchup.

PASS ATTEMPTS: Through four weeks the Bears are seeing an average of 31.8 pass attempts against them per game. Even in a game the Bears led for much of last
week, the Broncos tried just 28 throws.
</item 94>

<item 95>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
21. New York Jets 4-5 (18)

Aaron Rodgers can't come back soon enough, and going for a December miracle might be too late after two straight key losses to the Chargers and Raiders. Zach
Wilson will always cap the potential of a talented offense that keeps making mistakes. The Bills and Dolphins in back-to-back division games next might be their
season.

22. Denver Broncos 4-5 (24)

The Broncos got another big win against a AFC power, sandwiching their Week 9 bye with upsets over the Chiefs and Bills. The offense has hit a balanced grooved
with Russell Wilson and the defense is making a lot more plays, too.

23. Washington Commanders 4-5 (21)

The Commanders should be pleased about Sam Howell and the offense showing more fight with additional weapons and the running backs look better all-around again.
But the reeling defense and pass protection continue to be un-winnable assets.

24.
</item 95>

<item 96>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
Their playoff hopes look much better after their stunning comeback vs. the Ravens.

9. Jacksonville Jaguars 6-3 (4)

The Jaguars got an NFC wake-up call against the 49ers, who were much more prepared after a Week 9 bye than they were. Jacksonville needs to forget that debacle
and get back on track, stat. The team gets a division double vs. Tennessee and Houston before facing Cincinnati.

10. Dallas Cowboys 6-3 (11)

The Cowboys took advantage of their "no contest" vs. the Giants at home with Dak Prescott continuing to ball as a passer and runner. They need to keep the
defense and running game success, too, in bigger games going forward.

MORE: Inside the Cowboys historic offensive explosion vs. Giants

11.
</item 96>

<item 97>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
But there's something else -- Love was pass rush ignorant, meaning he didn't let the Lions' pressure scare him, even if it did wind up overwhelming him for some
early-game sacks.

RAIDERS: Deserve credit for trying -- in the past two weeks they lead the NFL in pass rush pressure rate and are 10th in blitz rate. Playing weaker offensive
lines in Pittsburgh and L.A. have helped boost those numbers, but quarterbacks are still finding success to the tune of 13.9 yards per catch and 7.07
YAC/reception over those two games, both bottom-three in the league. And, both quarterbacks the Raiders played in Weeks 3 and 4 each posted at least 22 Fantasy
points (Kenny Pickett, Justin Herbert without his second-best receiver and without his top running back).

RAIDERS: The hunch is the Raiders will keep bringing the blitz and play a lot of gutsy man coverage behind it.
</item 97>

<item 98>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Only Joshua Dobbs, who ran for 55 yards, has scored double-digit Fantasy points against the Niners (18 points). Not that the Cowboys have been tested much --
Daniel Jones, Zach Wilson and the Patriots QBs are the other guys they've man-handled.

O-LINE: San Francisco has allowed a pass rush pressure on 36.4% of their snaps, which sounds like a lot but it's just a shade worse than league average. That
number is weighted by the Giants' decision to blitz on 84.6% of Purdy's dropbacks in Week 2, creating an inflated pass rush pressure rate. Despite the Giants'
efforts, Purdy still threw for 310 yards and two touchdowns in a game he didn't have Brandon Aiyuk in.

COWBOYS: Have become one of the most aggressive defenses in the league, ranking third from the bottom in zone coverage snaps.
</item 98>

<item 99>
NFL Best Bets Week 9: Colts clobber Panthers, Dolphins drop Chiefs in Germany (Sporting News, 2023-11-03T20:56:00+00:00)
Expect more of the same from QB Justin Herbert and his stud back against the Jets, who have allowed 46 receiving yards per game to RBs (sixth-most in the NFL).
They'll try to shut him down this Sunday, but it would feel so empty without AE!

Jahan Dotson, WR, Commanders @ Patriots — OVER 45.5 receiving yards

We've come full circle with Dotson, who made our preseason sleepers column then faded into early-season obscurity only to emerge again over the past two weeks.
Curtis Samuel was clearly not right last week, and the second-year wideout more than made up for it. Dotson caught 8-of-10 targets for 108 yards and a TD, a
stellar encore to his Week 7 line of five catches for 43 yards. The banged-up Patriots defense will be primarily focused on stopping stud Commanders receiver
Terry McLaurin, which should open up plenty of opportunities for Dotson.
</item 99>

<item 100>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
At least all of them went to his wideouts. All were in the red zone, none were on deep throws and none were on WR screens.

Sit Him (Lineup Decisions) Projections powered by Sportsline Najee Harris RB PIT Pittsburgh • #22

Age: 25 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs BAL BAL -4 O/U 38 OPP VS RB 12th PROJ PTS 9.3 RB RNK 30th YTD Stats RUYDS 210 REC 4 REYDS 34 TD 0
FPTS/G 7.1 The Steelers' lack of scoring opportunities really crushes the upside for Harris, who admittedly has improved this season and could be on his way to
being a reliable RB2. The Ravens have been pretty good against the run but they can give up some numbers. Harris is a volume-based back who is at best a low-end
starting option given everything that's going on.
</item 100>
=== END ITEM id=de8d7d84567a08b2 ===
