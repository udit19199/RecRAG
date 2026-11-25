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

=== ITEM id=7785e971dc4d4bcc ===
Gold sentences and facts:
Sentence 0: For the masses of NFL fans across the U.S., Thursday's clash between the Lions and Packers will be aired on Amazon Prime Video.
  Fact 0.0: Thursday's game between the Lions and Packers will be aired on Amazon Prime Video.
  Fact 0.1: The broadcast is for NFL fans across the U.S.
Sentence 1: For the masses of NFL fans across the U.S., Thursday's clash between the Bears and Panthers will be aired on Amazon Prime Video.
  Fact 1.0: Thursday's game between the Bears and Panthers will be aired on Amazon Prime Video.
  Fact 1.1: The broadcast is for NFL fans across the U.S.

Retrieved items:
<item 1>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Justin Jones nearly had it, but it was Kyler Gordon who came away with it. Everything the Vikings have done on the offensive side of the ball tonight looks like
a chore.

10:24 p.m. — A third down catch by Jordan Addison extends the drive for Minnesota. Addison only has 25 yards tonight.

10:23 p.m. — Ty Chandler gets the ball on second down and gets nothing. Chandler has not looked very effective tonight with 11 yards on five touches.

Bears 9, Vikings 3

10:18 p.m. FIELD GOAL — After Santos missed from 48 earlier, he drills it from 55. It's 9-3, Cubs Bears.

10:16 p.m. — Byron Murphy comes in to break up a third down pass by Fields, and Santos will come out for a long 55-yard field. He's already missed from 48
tonight.
</item 1>

<item 2>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
Aaron Rodgers is no longer in Green Bay, and a new love interest is now under center, Jordan Love. It was these Lions that ended the Rodgers era and swept Green
Bay last season. Now they meet again in prime time on one of the league's most hallowed grounds in a battle for NFC North supremacy. The tundra won't be frozen
at Lambeau Field at this point in the season, but the game sure is important, and for a game like this, there's no way we could pass up the chance to win some
cash with our FanDuel single-game DFS lineup.

MORE THURSDAY NIGHT FOOTBALL:

DK lineup | Best props | Betting preview | Start-Sit

Before we break down our picks and strategy, here are the most notable scoring rules for FanDuel contests: four-point passing TDs and half-point PPR.
</item 2>

<item 3>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
— After an offensive pass interference call backs up the Bears, Chicago shows no interest in taking a shot and hands it off on 3rd & 19. The Panthers will have
solid field position.

10:26 p.m. — The Bears come out with three Foreman runs in four plays. After turnovers doomed them last week, they're playing it conservatively.

10:21 p.m. — Young throws behind the line of scrimmage on 3rd & 15, which doesn't tend to convert. It'll be a Panthers punt, giving the Bears poor field
position in the process.

10:19 p.m. — Now Young hits Adam Thielen for another first down. How about that? Carolina is just about to midfield.

10:18 p.m. — A rare first down for the Panthers as Young throws to Marshall for an 11-yard gain.

Bears 16, Panthers 10

10:13 p.m.
</item 3>

<item 4>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
1 pick to Carolina.

6:50 p.m. — Standout edge rusher Brian Burns is out with a concussion, dealing a major blow to the Panthers' defense.

Brian Burns, CJ Henderson and DJ Chark among inactives.https://t.co/mL8EYWQKTs — Carolina Panthers (@Panthers) November 9, 2023

6:35 p.m. — Bryce Young threw an interception in only one of his last five games before crashing back to Earth with three picks against the Colts last week.
More than ever, it feels like this game will be won by the quarterback who can avoid turnovers. Bagent has thrown five picks in his last two starts.

6:15 p.m. — Tyson Bagent will get the start for the Bears tonight, so it'll be an undrafted rookie facing a rookie No. 1 pick. Despite that, Chicago is favored.

Bears vs. Panthers start time

Date: Thursday, Nov.
</item 4>

<item 5>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears are in field goal range.

The always reliable Cole Kmet gets the Bears into field goal range before the end of the half 👀#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/ArYhsjmh7M — NFL (@NFL) November 10, 2023

9:37 p.m. — The Panthers could use this D.J. Moore guy. Moore with a 14-yard grab to take the Bears into Carolina territory and give Chicago a real chance at
points here. 46 seconds left.

9:34 p.m. — The Bears still have plenty of ground to make up to get into field goal range, but a 15-yard unnecessary roughness penalty on Deion Jones helps inch
them closer.

9:29 p.m. — Tyrique Stevenson comes VERY close to a interception on an errant pass by Young, and the Panthers will punt with just over a minute left in the
half.

9:28 p.m.
</item 5>

<item 6>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Lions 34, Packers 17

10:50 p.m.: Quay Walker's leaps over a blocker on Riley Patterson's field goal attempt, prompting a unsportsmanlike penalty. That will put Detroit on the cusp
of another score, pending a Packers defensive rally.

10:43 p.m.: Lions beginning to salt the game away with strong running through the tackles. Montgomery and Gibbs creeping through the crevices to push Detroit
down the field. And an impressive toss from Goff into the outstretched hands of Reynolds, as well. The Lions are on the doorstep of the red zone.

10:36 p.m.: TOUCHDOWN! Love scampers into the end zone from short distance to cut that Detroit lead down even more. He got held short of the end zone on the
ensuing two-point effort. But Green Bay is firmly back in it.
</item 6>

<item 7>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
7:56 p.m. — Frank Gore and Vernon Davis are in the house! A couple greats from the 49ers' last golden era under Jim Harbaugh.

Frank Gore and Vernon Davis in the building for SNF 🔥 pic.twitter.com/XMKgK4aeaF — OurSF49ers (@OurSf49ers) October 8, 2023

7:36 p.m. — CeeDee Lamb is coming off back-to-back games with only four catches. Is he due for a huge performance tonight?

7:11 p.m. — New 49ers acquisition Randy Gregory is officially inactive against his former team.

7:01 p.m. — The Eagles are going to win in Los Angeles, so the 49ers need a win to keep pace atop the NFC. After they had to play the last two NFC championship
games on the road, the No. 1 seed is undoubtedly in the back of their minds.

6:45 p.m.
</item 7>

<item 8>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
The Sporting News tracked live updates and highlights throughout the "Monday Night Football" matchup between the Raiders and Lions. Check out all of the key
moments you might have missed.

Raiders vs. Lions final score

1 2 3 4 Final Raiders 0 7 7 0 14 Lions 3 13 7 3 26

Raiders vs. Lions results, highlights from 'Monday Night Football'

(All times Eastern)

Final: Lions 26, Raiders 14

11:15 p.m. — That'll do it! The Lions bounce back and move to 6-2 with the win over Las Vegas. Detroit left points on the board, but the Raiders were thoroughly
outplayed in this one.

11:09 p.m. — The Lions get the ball back and will try to ice the game at this point.

11:08 p.m. — Adding insult to injury, Garoppolo is sacked for the sixth time tonight.
</item 8>

<item 9>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears cut the deficit to 7-3.

8:44 p.m. — A false start by Cody Whitehair erases what would've been a good 49-yard field goal...the Bears will keep Cairo Santos out there.

8:43 p.m. — Chicago will try a field goal from 49 yards out after the drive fizzles out.

8:41 p.m. — Foreman has been the Bears' best weapon tonight. He rips off an 11-yard run and now has 43 scrimmage yards in this first quarter.

8:40 p.m. — The Bears rattle off two first downs on their first three plays, with Moore and Foreman grabbing passes from Bagent to take Chicago over midfield.

Panthers 7, Bears 0

8:36 p.m. TOUCHDOWN — Ihmir Smith-Marsette returns the punt 79 yards for a touchdown!
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
the Bears. — Adam Schefter (@AdamSchefter) November 27, 2023

What channel is Bears vs. Vikings on today?

Game: Chicago Bears vs. Minnesota Vikings

Chicago Bears vs. Minnesota Vikings Date: Monday, Nov. 27

Monday, Nov. 27 TV channel: ESPN

ESPN Live stream: ESPN+, Fubo (U.S.) | DAZN (Canada)

Bears vs. Vikings will be aired on ESPN, which has long held the broadcast rights to "Monday Night Football."

Joe Buck will lead the broadcast, serving as the game's play-by-play announcer, with Troy Aikman serving as color analyst. Lisa Salters will provide updates
from the sidelines.

Fans can also stream the game via ESPN+ or Fubo, which offers a free trial. Viewers in Canada can stream every NFL game via DAZN.
</item 10>

<item 11>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:04 p.m. — Dobbs is picked off again! That's two interceptions and just nine passing yards for the Vikings quarterback, but this one doesn't look like his
fault - Jordan Addison saw it bounce off his hands. It's Jaquan Brisker with the pick.

9:01 p.m. — Another penalty gifts the Vikings a first down... Both teams are piling on the mistakes.

8:58 p.m. — A roughing-the-passer call goes against the Bears and gives the Vikings some breathing room early in the drive.

8:54 p.m. — Fields overthrows Darnell Mooney on third down, and Mooney gets destroyed in a collision. The Vikings will get the ball back, still somehow only
trailing by three. Fields will want that one back.

8:50 p.m. INTERCEPTION — Jaylon Johnson intercepts a throw from Dobbs! That just hung up there.
</item 11>

<item 12>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
The rookie picks up his fourth touchdown of the season and extends the Lions' lead. Detroit found success running the ball early, but that drive featured nice
work from Goff. 16-7. Lions.

9:35 p.m. — LaPorta picks up a first down for the Lions, and now Dan Campbell is going to start using his timeouts with 31 seconds left in the half. Detroit is
at the Raiders' 18-yard line.

9:33 p.m. — St. Brown picks up another first down for the Lions as they enter field goal range again. St. Brown is up to 94 yards in the first half after
entering the day listed as questionable.

9:30 p.m. — What would've been a horsecollar penalty on a nice Gibbs run is offset by a holding call. The fans don't like it.

9:26 p.m.
</item 12>

<item 13>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
End of first half: Panthers 10, Bears 9

9:40 p.m. FIELD GOAL — In a game that might be lucky to see an offensive touchdown, the Bears deserve some credit for that drive to work their way into field
goal range. They trail 10-9 at the half.

9:39 p.m. — Bagent narrowly avoids a sack and actually runs for positive yardage to move Santos' attempt closer. Chicago is down to three seconds, so here comes
Santos.

9:38 p.m. — A perfectly executed play to get the ball to Cole Kmet, who picks up the first down and then some but still gets out of bounds to stop the clock.
The Bears are in field goal range.
</item 13>

<item 14>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
FLEX TE, Luke Musgrave ($8,000). Much like with the quarterbacks, we also like the other tight end in this game. Musgrave is coming off his best game of the
season, hauling in six catches for 49 yards, and he gets a matchup against the second-worst defense against tight ends. According to FantasyPros, the Lions are
giving up 14.8 fantasy points per game to the position, leaving Musgrave in a great spot to deliver. If Watson does return on Thursday, it could help the
rookie, as Detroit's defense will have to focus their attention elsewhere.

Best SuperDraft NFL DFS Picks for Lions-Packers

This year, we're also listing our favorite weekly DFS plays on SuperDraft.

For those unfamiliar, SuperDraft's DFS scoring involves point multipliers as opposed to traditional salary-capped or tiered contests.
</item 14>

<item 15>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Lions are closed to adding a second score.

8:40 p.m.: Since the interception, it's felt as if Goff and the Lions can get anything they want. Gibbs is lightning in a bottle, while Montgomery brings the
thunder. This drive just continues to get lengthier and lengthier.

8:34 p.m.: Lions are looking menacing as can be in the trenches. Isaiah Buggs sends Love spiraling with a crunching hit behind the line. The Packers are giving
the ball back to the Lions on a three-and-out.

8:29 p.m.: TOUCHDOWN! Amon-Ra St. Brown leaps, stretches and reels in Goff's feather towards the corner of the end zone! Detroit showing just what its offense
can do when its clicking.

Lions 7, Packers 3

8:27 p.m.: Sam LaPorta gets beyond his marker with a strong move and plods 35 yards down the field.
</item 15>

<item 16>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
It's 4th & 10, and Chicago is lining up to got for it.

8:36 p.m. — Sacks are killing the Bears in this game. Multiple Vikings defenders wrap up Fields for a 12-yard loss and quickly set Chicago back to 2nd & 22.

8:35 p.m. — The Bears' offense is picking up where it left off. After Fields takes off for 14 yards, he hits Roschon Johnson for 22 yards to take Chicago into
Vikings territory.

8:31 p.m. — That's what the Bears got Montez Sweat to do. Sweat sacks Dobbs on third down to force a quick three-and-out and get the ball right back in Fields'
hands.

8:27 p.m. — A great start for the Bears ends with zero points as Cairo Santos misses a 48-yard field goal. That's a deflating sequence for Chicago.

8:25 p.m. — The rookie Ivan Pace Jr.
</item 16>

<item 17>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
The Packers' first three games have come against the No. 15, 29, and 31 ranked teams in passing yards per game, or the No. 21, 29, and 32 passing offenses as
graded by PFF. The Lions love to throw and have a quarterback that rarely turns it over. Goff should be in position for a solid night with his arm, lighting it
up through the air against a defense that might be without their star corner, Jaire Alexander (back), again in Week 4.

FLEX QB Jordan Love, Packers ($16,000). From one quarterback to another, we like this game to feature a lot of points. It's something that doesn't happen often
on "Thursday Night Football," but that should change this week. The Lions seem incapable of playing in low-scoring, grind-it-out games, so there will be chances
for Love to get the Packers on the scoreboard often.
</item 17>

<item 18>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
Cris Collinsworth (color analyst) returns to the fore, joining lead broadcaster Mike Tirico and sideline reporter Melissa Stark.

Cord-cutters can find the action on Peacock or Fubo, which offers a free trial.

Viewers in Canada can watch on DAZN, which carries every NFL game, or on TSN 1/4.

Chiefs vs. Packers start time

Date : Sunday, Dec. 3

: Sunday, Dec. 3 Start time: 8:20 p.m. ET

The Chiefs and Packers will clash at Lambeau Field in Green Bay, with kickoff slated for 8:20 p.m. ET.
</item 18>

<item 19>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears and Panthers entered Thursday night looking for progress, not playoff contention. If any team achieved that goal, it was Chicago.

It wasn't an aesthetically pleasing game, but that was never the expectation. The Bears grinded out a 16-13 victory behind a terrific defensive performance, a
solid running game, and a few timely throws by Tyson Bagent.

The Panthers jumped out to an early lead on a punt returned for a touchdown by Ihmir Smith-Marsette, but they never found the end zone again. The only offensive
touchdown of the night came in the third quarter, when D'Onta Foreman rumbled his way into the end zone to give Chicago a 16-10 lead.

NFL WEEK 10 PICKS: Straight up | Against the spread

As it turned out, 16 points was all the Bears needed.
</item 19>

<item 20>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
ET / 5:15 p.m. PT

Seahawks vs. Giants from MetLife Stadium in East Rutherford, N.J. will kick off at 8:15 p.m. ET, the customary kickoff time for "MNF" matchups this year.

NFL live stream for Seahawks vs. Giants

Live stream (U.S.): ESPN+, NFL+, Fubo

ESPN+, NFL+, Fubo Live stream (Canada): DAZN

American viewers have a few different options to live stream Monday's game. ESPN will broadcast the game over its proprietary streaming service, ESPN+
(subscription required). NFL+ will also carry the game, but a subscription is required, as well.

Cord-cutters in the U.S. can turn to Fubo for their streaming needs. Fubo carries the ESPN family of networks, ABC, NBC, CBS and NFL Network, so fans can catch
a bevy of football action throughout the 2023 season.
</item 20>

<item 21>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:25 p.m. — The rookie Ivan Pace Jr. finally gets to Fields and drops him for a five-yard loss. It'll be 3rd & long for the Bears.

8:24 p.m. — Fields throws on the run to find Roschon Johnson for a first down, and he quickly follows up with a 13-yard pass to Khalil Herbert. The Bears are
driving, and Fields doesn't have an incompletion yet.

8:21 p.m. — The Bears have taken it into Vikings territory with a string of short gains. For Chicago, that's progress.

8:19 p.m. — Fields finds Moore for a short gain and the first 1st down of the night.

8:16 p.m. — The Bears will start with the ball on the road, looking to build early momentum.

8:00 p.m. — Will the Josh Dobbs show continue tonight?
</item 21>

<item 22>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Johnny Hekker punts it to the 1 and @PatMcAfeeShow nods in approval somewhere#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/WKffggayuG — NFL (@NFL) November 10, 2023

8:28 p.m. — Young is forced to throw it away on third down, so he Panthers will punt from just past midfield.

8:26 p.m. — Young misses a wide open Terrace Marshall Jr. on second down, sailing the pass too high.

8:25 p.m. — Two third downs, two scrambles for a first down by Young. He's not exactly known for his mobility, but he's far from immobile. Now let's see whether
the Panthers can get the passing game going at all.
</item 22>

<item 23>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
That would've opened up the field a bit for the Panthers.

10:54 p.m. — Young drops back on 4th & Inches from deep in his own territory, but fortunately for Carolina he's able to run to the right and pick up the first
down. The Panthers still have plenty of ground to cover to get into field goal range.

Bryce Young does just enough to extend the drive on 4th and inches#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/YWvltQGepB — NFL (@NFL) November 10, 2023

10:52 p.m. — The tush-push doesn't work for everyone. The Panthers try to push Chuba Hubbard forward on 3rd & 1, and he goes nowhere. It looks like they'll go
for it again.

10:49 p.m. — Deion Jones makes a great tackle to drop Roschon Johnson for a three-yard loss.
</item 23>

<item 24>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
8:13 p.m.: Lions starting the game with the ball. Packers defense looking stout as ever through the opening two plays.

8:12 p.m.: Weezy F Baby leading the Packers out onto the field! Wasn't expecting that!

7:53 p.m.: Marshawn Lynch kicking it with Barry Sanders in a pre-game segment. This is what you love to see. It looks like two legends can co-exist.

7:40 p.m.: Aaron Jones' return should offer Love a much-needed reprieve. Green Bay's run game hasn't been all that effective thus far — the Pack are averaging
just 3.4 yards per carry so far this year. But Jones is a dynamic force coming out of the backfield, someone capable of carrying an offense on his back for
large portions of a game. That's a young QB's dream.

7:30 p.m.: Jordan Love and the Packers heading onto the field ahed of tonight's game!
</item 24>

<item 25>
NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season (Sporting News, 2023-12-03T21:49:04+00:00)
7. Green Bay Packers (6-6), third place, NFC North. The Packers moved into the final playoff spot thanks to having the second-best conference record after the
Vikings, to whom they already lost but face again in Minnesota in Week 17. The Packers also hold key wins over the Rams and Saints in the wild-card race.

Who's in the hunt?

8. Los Angeles Rams (6-6)

9. Seattle Seahawks (6-6)

10. Tampa Bay Buccaneers (5-7)

11. New Orleans Saints (5-7)

12. New York Giants (4-8)

13. Chicago Bears (4-8)

14. Washington Commanders (4-9)

The Rams and Seahawks are the best wild-card bets here now, while the Saints and Buccaneers will still have their sights on making the playoffs via still
winning the weak South over the Falcons.
</item 25>

<item 26>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
Inactive for the Seahawks: CB Coby Bryant, CB Tre Brown, CB Artie Burns, LB Nick Bellore, G McClendon Curtis, T Charles Cross and G Ben Brown.

6:25 p.m.: With less than two hours remaining until kickoff, there's still no concrete update on the availability of Giants RB Saquon Barkley, who is currently
listed as doubtful with a high ankle sprain.

Barring a last-minute change, it doesn't appear as though No. 26 will be suiting up for New York.

6 p.m.: Jamal Adams is back!

For the first time since Week 1 of the 2022 NFL season, the three-time All-Pro safety will take the field for the Seahawks. Almost poetically, Adams makes his
return to the field in the place he called home for the first three years of his career as a member of the Jets.

What channel is Seahawks vs. Giants on today?
</item 26>

<item 27>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Bears vs. Panthers start time

Date: Thursday, Nov. 9

Thursday, Nov. 9 Start time: 8:15 p.m. ET

Thursday night's game between the Bears and Panthers is set to kick off at 8:15 p.m. ET (7:15 p.m. local time) from Soldier Field in Chicago.

What channel is Bears vs. Panthers on today?

Game: Chicago Bears vs. Carolina Panthers

Chicago Bears vs. Carolina Panthers Date: Thursday, Nov. 9

Thursday, Nov. 9 TV channel (Chicago) : WFLD (Channel 32)

: WFLD (Channel 32) TV channel (Charlotte) : WSOC (Channel 9)

: WSOC (Channel 9) Live stream: Amazon Prime Video | DAZN (in Canada)

For the masses of NFL fans across the U.S., Thursday's clash between the Bears and Panthers will be aired on Amazon Prime Video.
</item 27>

<item 28>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 28>

<item 29>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The broadcast will be spearheaded by play-by-play announcer Al Michaels and color commentator Kirk Herbstreit. Sideline reporter Kaylee Hartung will offer
sideline updates throughout the game.

Fans in the Chicago and Charlotte markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 29>

<item 30>
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
</item 30>

<item 31>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
8:13 p.m.: Think the Bengals' fans are ready for this game? They're striping the stadium.

A look at the Paycor striping pic.twitter.com/9VsW19qGZ1 — Jay Morrison (@ByJayMorrison) November 6, 2023

8:12 p.m.: We're just minutes away from the start of this highly anticipated "Sunday Night Football" matchup.

Sunday Night Football's kicking into high gear!@carrieunderwood pic.twitter.com/7FlQhb198m — Sunday Night Football on NBC (@SNFonNBC) November 6, 2023

8:09 p.m.: Jason Garrett keeps it from being a clean sweep on the FNIA panel. He's the lone analyst that picks the Bills to beat the Bengals.
</item 31>

<item 32>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
But Green Bay is firmly back in it.

Lions 27, Packers 17

End of third quarter: Lions 27, Packers 11

10:31 p.m.: Impressive pitch-and-catch by Love and Jayden Reed. Packers hurler drops it right into Reed's bread basket for the 44 yard gain as the quarter comes
to an end.

10:24 p.m.: Things slow down for Green Bay on Drive No. 2 of the second half. Punt incoming. Packers need less of that as they attempt their second-straight
rally in as many weeks.

10:15 p.m.: Green Bay defense firms up, forcing yet another punt. Can Love and the Packers bring things back?

10:10 p.m.: TOUCHDOWN! Love finds Watson for the TD, giving Packers fans a much-needed boost of adrenaline. Good drive by Green Bay, which finally shows signs
of life after a putrid first half.
</item 32>

<item 33>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Thursday's game appeared to be destined to be a showcase of quarterbacking excellence; Jared Goff and Jordan Love have proved themselves to be formidable
options under center since taking the reins of their respective offenses.

But it was the supporting cast that proved so pivotal to Thursday's result for Dan Campbell's side. David Montgomery was a nuisance, slipping through crevices
to puncture the heart of the Packers' defense. When it wasn't him, it was that receiving corp, made up of unheralded youngsters who have all come up trumps in
recent weeks.

As for Love and Co., Thursday's affair was another indication of just how much work there is to be done before they compete for the divisional crown. There were
certainly some flits of delight, throws that highlighted just how much talent Love possesses in his golden arm.

But there were low points, too, misshapen efforts that fell harmlessly to the turf, or harmfully into the opposition's hands.
</item 33>

<item 34>
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
</item 34>

<item 35>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
It looked like that score might be the difference, particularly after Fields fumbled away the ball again on the next possession, but the Bears' defense put the
ball back in his hands in the final minutes. Fields delivered.

Facing 3rd & 10 outside of field goal range, Fields hit an open D.J. Moore for a 36-yard gain. At that point, all the Bears needed to do was kneel and put Cairo
Santos in position to win the game. Santos drilled the 30-yard attempt with 10 seconds remaining, and that would seal it for Chicago.

MORE: Vikings playoff chances after loss to Bears

As he embarks on a crucial stretch that could determine his future with the Bears, Fields flashed the good and the bad. He was excellent on Chicago's opening
drive, albeit one that ended without points, but missed throws and costly turnovers allowed the Vikings to hang around.
</item 35>

<item 36>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
11:03 p.m. — The Vikings are using their two timeouts, but there isn't much they'll be able to do once this hits fourth down. Once the Bears kneel it on third
down, Santos will be running out for what should be an automatic field goal.

11:01 p.m. — Fields hits Moore for 36 yards! That puts the Bears well into field goal range, and it's exactly what Chicago got Moore to do.

11:00 p.m. — A pair of incompletions set up a crucial 3rd & 10 for the Bears. They're not in field goal range at the Vikings' 49.

10:58 p.m. — Fields picks up a first down with his legs to take the Bears just over midfield. Chicago doesn't have timeouts, but there are still 85 seconds on
the clock.

10:55 p.m.
</item 36>

<item 37>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
— Adding insult to injury, Garoppolo is sacked for the sixth time tonight. It's 4th & 17.

11:07 p.m. — Davante Adams has had his way with Cameron Sutton at times tonight, and Garoppolo has just not made the throw each time. It happens again here on
what would've been a game-changing play. Garoppolo has had a brutal night.

11:04 p.m. — Riley Patterson missed a 26-yard field goal attempt after making his first four attempts tonight. Strange. The Lions can't come away with points on
that drive.

11:01 p.m. — What would've been a Reynolds touchdown is erased by a holding penalty. That does give the Lions a chance to erase some more time on the clock,
though.

11:00 p.m. — Craig Reynolds, up to 66 yards, has the Lions on the verge of the end zone.

10:58 p.m.
</item 37>

<item 38>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
📺: @NBC & @Peacock pic.twitter.com/phvMXDOH8s — Sunday Night Football on NBC (@SNFonNBC) December 4, 2023

Taylor Swift and Brittany Mahomes are cheering on the Chiefs in Green Bay! #ChiefsKingdom



📺: @NBC & @Peacock pic.twitter.com/phvMXDOH8s — Sunday Night Football on NBC (@SNFonNBC) December 4, 2023

What channel is Chiefs vs. Packers on today?

TV channel : NBC (U.S.) | TSN 1/4 (Canada)

: NBC (U.S.) | TSN 1/4 (Canada) Live stream: Peacock, Fubo (U.S.) | DAZN (Canada)

Per usual, this week's edition of 'Sunday Night Football' will be broadcast on NBC.
</item 38>

<item 39>
Bills vs. Bengals live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-05T23:47:00+00:00)
5

Sunday, Nov. 5 TV channel: NBC

NBC Live stream: Peacock, NBCSports.com, Fubo (U.S.) | DAZN (Canada)

Sunday's game will be carried by NBC, which has carried "Sunday Night Football" since 2006, and will continue to do so through at least 2033.

The game will feature the usual broadcast group of Mike Tirico (play by play) and Cris Collinsworth (color commentary) in the booth, and Melissa Stark reporting
from the sidelines.

There will be several options to live stream the game. NBCSports.com will have a broadcast of the game for fans with a cable login. Peacock will have NBC's
broadcast, as will Fubo, which offers a free trial. Fans in Canada can stream the game live with DAZN.

Bills vs. Bengals start time

Date: Sunday, Nov. 5

Sunday, Nov. 5 Start time: 8:20 p.m.
</item 39>

<item 40>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
That play call has not worked for the Bears in this half.

Vikings 10, Bears 9

10:43 p.m. TOUCHDOWN — The Vikings turn the Fields fumble into the first touchdown of the night! Hockenson goes up and gets Dobbs' pass for the score. The
Vikings lead, 10-9.

The first TD of the game is a Dobbs dot to Hockenson!



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/b1IiFoqcqo — NFL (@NFL) November 28, 2023

10:41 p.m. — Dobbs hits Hockenson and pushes the Vikings into the red zone. Minnesota is driving. No one has scored a touchdown yet tonight.

10:40 p.m.
</item 40>

<item 41>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
10:37 p.m. — A neutral zone infraction pulls the Panthers a bit closer, and they will line up to go for it on 4th & 6...

End of third quarter: Bears 16, Panthers 10

10:32 p.m. — The Panthers run it with Chuba Hubbard on 1st & 30. They're not showing much interest in letting Bryce Young air it out, even while trailing in a
lost season.

10:31 p.m. — It's 1st & 30 for the Panthers after back-to-back penalties. That's one way to erase great field position.

10:29 p.m. — A less than impressive punt by Trenton Gill sets up Carolina well inside Bears territory. If there was ever a time for the Panthers' offense to get
something going...

10:28 p.m.
</item 41>

<item 42>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Amon-Ra St. Brown was in the midst of a tempest, shrouded in a sea of green with few options around him. The Lions WR had done the hard part, stretching beyond
an exasperated Packers defense to trot into the end zone.

But he had a choice to make: rollick in the joy of his sapphire-clad teammates. Or take a leap of faith into the Lambeau Field abyss.

He opted for the latter. St. Brown hurled himself into the green lagoon. Then, he smiled.

St. Brown's impersonation of the Lambeau Leap was one of many high points in a memorable night for Detroit. Fueled by a stout defensive showing, a gritty run
game and a bit of fairy dust from the likes of St. Brown, Sam LaPorta and Josh Reynolds, the Lions cruised to a 34-20 win over their hated NFC North rivals,
Green Bay.
</item 42>

<item 43>
Seahawks vs. Giants live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-02T22:00:04+00:00)
Fubo also offers a free trial.

Viewers in Canada can sign up for DAZN, which carries NFL Game Pass.

Monday Night Football schedule 2023
</item 43>

<item 44>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
And Edwards follows it up with the Bears 3rd INT of the game ‼️



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/2d4C1s2O09 — NFL (@NFL) November 28, 2023

10:06 p.m. — After a hold by Jaylon Johnson gifts the Vikings a first down, they still can't capitalize. Ty Chandler can't make T.J. Edwards miss on third down,
so Minnesota will go for it on fourth down.

10:02 p.m. — The running game is working, the passing game is not for the Vikings. Mattison has 19 yards three plays into this drive for Minnesota.

Bears 6, Vikings 3

9:58 p.m.
</item 44>

<item 45>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Game: Dolphins vs. Eagles

Dolphins vs. Eagles Date: Sunday, Oct. 22

Sunday, Oct. 22 TV channel: NBC

NBC Live stream: Peacock | NBCSports.com | Fubo (U.S.) | DAZN (Canada)

Dolphins vs. Eagles can be watched on NBC, which has long held the broadcast rights to "Sunday Night Football."

Mike Tirico (play-by-play) and Cris Collinsworth (color analyst) will be on the call from Levi's Stadium while Melissa Stark will serve as the sideline
reporter. The crew is in its second season together after Tirico replaced play-by-play legend Al Michaels in 2022 while Stark took over for Michele Tafoya.

Fans can also stream the game via Peacock or NBCSports.com with a subscription, along with Fubo, which offers a free trial.

Viewers in Canada can stream every NFL game via DAZN.

Dolphins vs.
</item 45>

<item 46>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
After Eddy Pineiro cut Carolina's deficit to three early in the fourth quarter, Frank Reich put his trust in Pineiro again with the game on the line, sending
him out for a 59-yard attempt on fourth-and-10 in the final minutes rather than trusting the offense to pick up the 10 yards. Pineiro's kick was short.

While the decision to kick from 59 yards out might be second-guessed, Carolina's offense didn't give Reich much reason to believe. Bryce Young struggled to
develop any kind of rhythm through the first three quarters, and even the Panthers' most successful drives were laborious, consistently requiring conversions on
third and fourth down.

While Young got little help from his receivers, who failed to get open all night, his struggles — including multiple passes that were nearly intercepted — are a
reminder that the Panthers' offense is still a long way from resembling anything like the unit owner David Tepper envisioned when the franchise traded for the
No. 1 pick in March.
</item 46>

<item 47>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
Brown is officially active after being listed as questionable with an illness.

6:25 p.m. — This could be an important one: Raiders K Daniel Carlson will play tonight after being listed as questionable with a groin injury. Even more
important will be whether Carlson's injury changes Josh McDaniels' decision-making.

Raiders’ kicker Daniel Carlson, listed as questionable for tonight due to a groin injury, will be Las Vegas’ kicker tonight, per league source. — Adam Schefter
(@AdamSchefter) October 30, 2023

What time is Raiders vs. Lions tonight?

Date: Monday, Oct. 30

Monday, Oct. 30 Time: 8:15 p.m. ET

Raiders vs. Lions is set to kick off at 8:15 p.m. ET at Ford Field in Detroit, Mich. for "Monday Night Football."
</item 47>

<item 48>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
That just hung up there. The Bears have been all over Dobbs and the Vikings' offense early on. A taunting penalty after the play hurts Chicago's field position.

Bears 3, Vikings 0

8:44 p.m. FIELD GOAL — The drive fizzles out, and the Bears opt to kick the field goal on 4th & 3. It's good. The Bears are out to an early 3-0 lead, but
they've left plenty of points on the board.

End of first quarter: Bears 0, Vikings 0

8:39 p.m. — What a play by Fields to get the ball to a wide open Kmet! The Vikings brought intense pressure, allowing Kmet to get wide open and go for 24 yards.
The Bears are in business.

8:38 p.m. — 2nd & 22 becomes 2nd & 27, but the Bears make up 17 yards on two plays.
</item 48>

<item 49>
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
</item 49>

<item 50>
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
</item 50>

<item 51>
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
</item 51>

<item 52>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
— And now this one is officially over. The Bears pick up their third win of the season, while Carolina drops to 1-8 and faces more tough questions about Bryce
Young and the offense.

11:08 p.m. — Bagent hits Mooney for the first down on 3rd & 7, and the Panthers now won't be able to stop the clock.

11:06 p.m. — Pineiro's attempt is SHORT from 59 yards out. The gamble fails for Frank Reich. The Panthers will hope to get the ball back.

11:04 p.m. — Young is very nearly intercepted on 3rd & 10, and the Panthers say they've seen enough of the offense. Eddy Pineiro is attempting a 59-yard field
goal to tie the game.

11:00 p.m. — Miles Sanders gets himself wide open on fourth down to extend the drive.
</item 52>

<item 53>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
He goes 18 yards here to put Minnesota near midfield.

10:29 p.m. FUMBLE — That's exactly what CAN'T happen if you're the Bears. Fields loses the football, Sheldon Day recovers, and the Bears waste an opportunity to
make this a two-score game.

Josh Metellus forces a fumble and the Vikings take over 😤



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/fKnE0eTxsP — NFL (@NFL) November 28, 2023

10:27 p.m. — Fields finds Mooney for a big third down conversion as the Bears get firmly into field goal range.

10:25 p.m. INTERCEPTION — Josh Dobbs throws his FOURTH interception of the night. Justin Jones nearly had it, but it was Kyler Gordon who came away with it.
</item 53>

<item 54>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Viewers in Canada can stream every NFL game via DAZN.

MORE: How Joshua Dobbs got his 'Passtronaut' moniker

Bears vs. Vikings start time

Date: Monday, November 27

Monday, November 27 Start time: 8:15 p.m. ET

Monday's game between the Bears and Vikings will kick off at 8:15 p.m. ET (7:15 p.m. local time) from U.S. Bank Stadium in Minneapolis, Minn.

The Bears haven't beaten the Vikings since 2020, with Minnesota 6-1 against Chicago over the last four seasons. The Vikings are 8-3 at home against the Vikings
since 2012.
</item 54>

<item 55>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
TNF INJURY UPDATES: Montgomery | Jones | Watson

Lions-Packers FanDuel Picks Week 4: NFL DFS lineup for GPP tournaments

Lions-Packers TNF, $60,000 budget

MVP (1.5x points) QB Jared Goff, Lions ($15,500). The Lions might not be the Super Bowl contenders we thought they could be, but that doesn't mean we aren't in
for a treat on Thursday night at Lambeau Field. Detroit is finding its footing in 2023 and should be able to have some success against the Packers. On the
surface, Green Bay's pass defense is playing around league average, allowing 207.3 passing yards per game. That slots them in at No. 18, while Pro Football
Focus grades them the No. 8 team in coverage with an 81.5. Let's look beneath the surface a little more. The Packers' first three games have come against the
No.
</item 55>

<item 56>
Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-10-19T22:00:04+00:00)
19 Time: 8:15 p.m. ET (7:15 p.m. CT)

The Week 7 edition of "Thursday Night Football," featuring the Saints and Jaguars, is set to begin at 8:15 p.m. ET. That will be the normal start time for all
of Amazon's "TNF" games in 2023.

The game is taking place in New Orleans, which is located in the Central time zone. So, those in that market will be able to catch the came at 7:15 p.m. local
time.

What channel is Thursday Night Football on tonight?

TV channel (Jacksonville) : WJAX (Channel 47)

: WJAX (Channel 47) TV channel (New Orleans) : WDSU (Channel 6)

: WDSU (Channel 6) Live stream: Amazon Prime Video | DAZN (in Canada)

Amazon Prime serves as the home of "Thursday Night Football" and will be for the next decade.
</item 56>

<item 57>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
— Miles Sanders gets himself wide open on fourth down to extend the drive. The Panthers are 3-for-3 on fourth down tonight despite their severe struggles for
much of the game.

Bryce Young to Miles Sanders for the 2nd fourth down conversion of the drive#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/Moaaj7cpqw — NFL (@NFL) November 10, 2023

10:58 p.m. — It'll be fourth and about two yards when the Panthers return from the two-minute warning... Carolina is trying to claw its way into field goal
range.

10:57 p.m. — Thielen comes through with a nice catch on the run to give Carolina another first down.

10:56 p.m. — Hayden Hurst with a bad drop on 2nd & 3. That would've opened up the field a bit for the Panthers.

10:54 p.m.
</item 57>

<item 58>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
ET at Ford Field in Detroit, Mich. for "Monday Night Football."

The Lions have won four of their last five matchups with the Raiders, though Las Vegas got the better of them when they last met in 2019.

MORE: Dan Campbell explains why Lions 'probably needed' blowout loss to Ravens

What channel is Raiders vs. Lions on tonight?

Joe Buck and Troy Aikman will be on the call for Raiders vs. Lions, with Lisa Salters providing sideline reports for the matchup.

For cord-cutters, Fubo carries the ESPN family of networks, as well as ABC, CBS, Fox, NBC and NFL Network, meaning you will be able to catch nationally
televised games throughout the 2023 season. Fubo also offers a free trial.

Viewers in Canada can catch the matchup on DAZN, which carries NFL Game Pass.
</item 58>

<item 59>
Chiefs vs. Jets live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-01T23:47:00+00:00)
Mike Tirico (play-by-play) and Cris Collinsworth (color analyst) will be on the call from MetLife Stadium while Melissa Stark will serve as the sideline
reporter. The crew is entering their second season together after Tirico replaced play-by-play legend Al Michaels while Stark took over for Michele Tafoya.

Fans can also stream the game via Peacock or NBCSports.com with a subscription, along with Fubo, which offers a free trial.

Viewers in Canada can stream every NFL game via DAZN.

Chiefs vs. Jets start time

Date: Sunday, Oct. 1

Sunday, Oct. 1 Start time: 8:20 p.m. ET (7:20 p.m. CT)

The Chiefs vs. Jets game will kick off at 8:20 p.m. ET, which is the same time that all "Sunday Night Football" games for the 2023 NFL season are set to begin.
</item 59>

<item 60>
Prisco's Week 12 NFL picks: Jaguars beat Texans in AFC South battle, Saints get key divisional win vs. Falcons (CBSSports.com, 2023-11-23T21:30:38+00:00)
Thursday, 12:30 p.m. ET (Fox, fubo)

This is a big game in the division for both teams, a survival test for Green Bay and a big one for the Lions if they want to try and get home-field advantage in
the conference. The Lions have had defensive issues as of late and Jordan Love has played well the past two weeks. That will keep the Packers in this game, but
the Lions win it.

Pick: Lions 31, Packers 27 (Result: Packers 29, Lions 22)

Thursday, 4:30 p.m. ET (CBS, Paramount+)

The Commanders are coming off a dud last week against the Giants, while Dallas is flying high. The Cowboys are playing as well as any team in the league.
Washington's offensive line is an issue, which will lead to the Cowboys having a field day with their defensive front. Dak Prescott plays well again. Cowboys
roll.
</item 60>

<item 61>
Will the Panthers fire Frank Reich? Impatient David Tepper could turn up Carolina's hot seat with 'Thursday Night Football' loss (Sporting News,
2023-11-06T21:17:51+00:00)
Instead, he decided to pull the trigger and part with him.

While Reich isn't guaranteed to meet the same fate, it's becoming clearer that he will need to deliver some wins to help earn a second season in charge of the
Panthers.

And as Schefter notes, the team's "Thursday Night Football" game against the Bears may be among the most important games remaining on Carolina's schedule, as
the team grapples with justifying the trade-up to select Young.

Let's keep in mind, [the Panthers] play the Bears Thursday night. The Bears have their No. 1 draft pick. So if the Bears win on Thursday, that's a huge victory
and in a bigger way than just the game itself.

Again, that's not to say that Reich will be fired after Thursday's game. After all, he might not be fired at all if he can help Carolina to earn some wins
during the second half of the season.
</item 61>

<item 62>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
He's already missed from 48 tonight.

End of third quarter: Bears 6, Vikings 3

10:14 p.m. — The Bears are starting to look respectable on third down again, this time completing to D.J. Moore to move deeper into Vikings territory. Chicago
still might need another first down to give Cairo Santos a fair shot at a field goal.

10:08 p.m. INTERCEPTION — Dobbs throws his third interception of the night! Jaylon Johnson deflected it, Edwards caught it, and interestingly, the Bears have
less favorable field position than they would've if it had fallen incomplete.

And Edwards follows it up with the Bears 3rd INT of the game ‼️
</item 62>

<item 63>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Justin Jones brings the heat to force a punt 😤#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/wyEgT3GZ7q — NFL (@NFL) November 10, 2023

9:05 p.m. — A holding call sets the Bears back to 2nd & 20...

9:03 p.m. — After Foreman pushes forward for a key first down, rookie Tyler Scott takes off for 16 yards to take the Bears over midfield. Chicago is looking to
tie the game on this drive.

9:00 p.m. — Sexiest Man Alive Jason Kelce is in the booth with Al Michaels and Kirk Herbstreit to start the second quarter.

Panthers 10, Bears 3

8:55 p.m. FIELD GOAL — Young comes under pressure on 3rd & 11 and just throws it into the ground.
</item 63>

<item 64>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
He has seven passing touchdowns through the first three weeks and faces a bottom-10 passing defense, per PFF. Love will enjoy a date with Detroit's banged-up
secondary that is just beginning life without C.J. Gardner-Johnson (pec).

FLEX WR Romeo Doubs ($10,500). Christian Watson appears primed for a return on Thursday night, but does it really matter? With him being held out for multiple
weeks with a hamstring injury, we question the type of workload that the former North Dakota State standout is in line for. As a result, we look to Doubs. The
second-year receiver is tied for the team lead in targets and has the most touchdowns with three. Along with Jayden Reed, Doubs has been a favorite of Love's in
the red zone, as the duo has combined for 12-of-16 targets to receivers.
</item 64>

<item 65>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
The rookie tight end has been producing at a historic clip so far. And it looks like he's on pace for another standout showing this week.

8:26 p.m.: Jahmyr Gibbs and David Montgomery look a formidable duo in the backfield. Just collaborated to push the Lions forward as they move the chains for the
first time today.

8:21 p.m.: FIELD GOAL! Packers open the scoring courtesy of an Anders Carlson field goal. Lions defense held tough, all things considered. Aidan Hutchinson
burst through the trenches to send Love tumbling to precipitate the stop.

Packers 3, Lions 0

8:14 p.m.: INTERCEPTION! Rudy Ford swipes Jared Goff's errant pass from the sky and takes it back 16 yards to put the Packers on the cusp of the end zone!
That's Goff's third interception in as many games!

8:13 p.m.: Lions starting the game with the ball.
</item 65>

<item 66>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
9:28 p.m. — An illegal contact penalty, the second of the night on the Bears, gives Carolina a fresh set of downs.

9:25 p.m. — The Panthers will have 3rd & 3 out of the two-minute warning, just hoping to get in position for some points before halftime. They'll start the
second half with the ball as well.

Panthers 10, Bears 6

9:19 p.m. FIELD GOAL — Bagent is fortunate just to hold onto a bad snap, but his pass is incomplete and brings out Cairo Santos. It's 10-6, Panthers.

9:17 p.m. — A second big play by rookie Tyler Scott, who catches a 15-yard pass from Bagent to give the Bears a fresh set of downs. Chicago is into field goal
range.
</item 66>

<item 67>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The crew is entering their second season together after Tirico replaced play-by-play legend Al Michaels while Stark took over for Michele Tafoya.

Fans can also stream the game via Peacock or NBCSports.com with a subscription, along with Fubo, which offers a free trial.

Viewers in Canada can stream every NFL game via DAZN.

Cowboys vs. 49ers start time

Date: Sunday, Oct. 8

Sunday, Oct. 8 Start time: 8:20 p.m. ET (5:20 p.m. CT)

Sunday night's game between the Cowboys and 49ers is set to kick off at 8:20 p.m. ET (5:20 p.m. local time) from Levi's Stadium in Santa Clara, Calif.

The 49ers and Cowboys have a long history with one another, but San Francisco has gotten the better of Dallas in each of their last two meetings.
</item 67>

<item 68>
Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-10-19T22:00:04+00:00)
The NFL agreed to a deal with the streaming giant to make it the home of the package for 11 years, so those looking to watch "TNF" nationally will need a
subscription to Amazon Prime to watch the games.

That said, those in the local markets of the teams playing on "Thursday Night Football" can watch the contests without subscriptions to Amazon. CBS affiliate
WJAX (Channel 47) will career the game to those in the Jacksonville area while viewers in the New Orleans market can catch the program on WDSU (Channel 6), an
NBC affiliate.

Amazon's broadcast crew for "Thursday Night Football" will consist of Al Michaels (play-by-play), Kirk Herbstreit (color commentary) and Kaylee Hartung
(sideline reporter). The trio is in their second season working together.

Viewers in Canada will not have to have Amazon Prime to access "Thursday Night Football" in 2023. They will be able to watch the games on DAZN.
</item 68>

<item 69>
Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-11-12T23:30:03+00:00)
12

Sunday, Nov. 12 TV channel: NBC

NBC Live stream: Peacock, NBCSports.com, Fubo (U.S.) | DAZN (Canada)

New York and Las Vegas will face off on "Sunday Night Football," which is carried by NBC. Mike Tirico will handle play-by-play with Cris Collinsworth providing
commentary. Melissa Stark will report from the sidelines.

Those hoping to stream the game can find it on Peacock or NBCSports.com with a cable subscription. Fubo, which offers a free trial, will carry NBC's broadcast.
DAZN carries all NFL games in Canada.

Jets vs. Raiders start time

Date: Sunday, Nov. 12

Sunday, Nov. 12 Start time: 8:20 p.m. ET

The Jets and Raiders will start their "Sunday Night Football" game at 8:20 p.m. ET, from Allegiant Stadium in Las Vegas.
</item 69>

<item 70>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
ET.

What channel is Lions vs. Packers on today?

Game: Detroit Lions at Green Bay Packers

Detroit Lions at Green Bay Packers Date: Thursday, September 28

Thursday, September 28 TV channel (Detroit) : WJBK (Channel 2)

: WJBK (Channel 2) TV channel (Green Bay) : WITI (Channel 6) or WGBA (Channel 26)

: WITI (Channel 6) or WGBA (Channel 26) Live stream: Amazon Prime Video | DAZN (in Canada)

For the masses of NFL fans across the U.S., Thursday's clash between the Lions and Packers will be aired on Amazon Prime Video. The broadcast will be
spearheaded by legendary play-by-play announcer Al Michaels. He'll be joined by longtime ESPN college football commentator Kirk Herbstreit, who will offer
occasional musings as an analyst. Sideline reporter Kaylee Hartung will offer updates throughout the affair.
</item 70>

<item 71>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The Bears defense comes up big with a 4th down stop



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/yndUqwgIbf — NFL (@NFL) November 28, 2023

9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.

9:46 p.m. — Alexander Mattison has drawn the ire of Vikings fans all season, but he just got Minnesota's second half of to a booming start with a 21-yard run.

End of first half: Bears 3, Vikings 3

9:32 p.m. FIELD GOAL — Greg Joseph drills it from 34 yards out, and this mess of a game is all tied at three entering the half.

9:30 p.m.
</item 71>

<item 72>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
A winning lineup will likely feature a strong mix of stud fantasy contributors with 1x-1.6x boosts, as well as matchup-based sleepers with larger multipliers.
Each player is assigned a scoring multiplier based on their value as determined by SuperDraft. Users can roster any player in their lineup with no restrictions.

MORE: Win BIG with SuperDraft! Get a free $10 deposit when you use promo code "TSN"!

Here are the most notable scoring rules for SuperDraft NFL contests: Half-point PPR, four-point passing TDs, two-point bonuses for 100 rushing yards, 100
receiving yards, and 300 passing yards.

MORE SUPERDRAFT DFS: How to play SuperDraft

Champion (+50% points): WR Amon-Ra St.
</item 72>

<item 73>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
NFL WEEK 4 PICKS: Straight up | Against the spread | Projection model

The Sporting News tracked live updates from Thursday's clash between the Lions and the Packers. Scroll below to check out highlights and live results as two of
the NFC's oldest sides faced off for NFC North glory.

Lions vs. Packers score

1 2 3 4 F Lions 14 13 0 7 34 Packers 3 0 8 9 20

Lions vs. Packers live updates, highlights from Thursday Night Football

(All times Eastern)

End of fourth quarter: Lions 34, Packers 20

11:19 p.m.: Lions complete yet another fourth down, officially putting the game on ice.

11:12 p.m.: Onside kick spills out of bounds, and it's looking like that might be that. Packers have one timeout remaining but Montgomery's fiery run through
the tackles should make that obsolete.

11:11 p.m.: FIELD GOAL!
</item 73>

<item 74>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
10:58 p.m. — Garoppolo is sacked AGAIN. It's Anzalone for the second time on the drive, and that sends the ball right back to the Lions.

10:56 p.m. — Now the Lions sack Garoppolo on back-to-back plays as Alex Anzalone and Alim McNeill come in to send the Raiders to 3rd & 28.

10:53 p.m. — Garoppolo hits Jakobi Meyers for a big 19 yards to give the Raiders something to work with.

10:51 p.m. — The Garoppolo-to-Adams connection is just not working tonight. Adams was all by himself downfield, and Garoppolo throws it well out of bounds.

Lions 26, Raiders 14

10:46 p.m. FIELD GOAL — 52 yards? No problem for Patterson, who extends the Lions' lead to 12.

10:45 p.m.
</item 74>

<item 75>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The honeymoon period is over for Joshua Dobbs in Minnesota.

The folk hero quarterback threw four interceptions in the Vikings' 12-10 "Monday Night Football" loss to the Bears, leading a sluggish offense that didn't find
its footing until the fourth quarter in a game that Minnesota wouldn't have needed many points to win.

The Bears' defense swarmed Dobbs throughout the night, sacking him only twice but bringing a great deal of pressure as the defensive backs made plays. Despite
the barrage of turnovers, Chicago largely failed to take advantage. The Bears scored only three points off of the four Dobbs interceptions.

MORE: How Joshua Dobbs got his 'Passtronaut' nickname

Meanwhile, Chicago's first turnover of the night — a deflating fumble by Fields when the Bears had a chance to extend their lead to two possessions — was
quickly turned into a touchdown by the Vikings.
</item 75>

<item 76>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
When it mattered most, though, Fields was poised and sealed the Bears' first win over an NFC North opponent since 2021.

While Chicago's hopes are still extraordinarily distant at 4-8, the Vikings saw their playoff chances take a hit on Monday night. Minnesota (6-6) dropped to the
No. 7 seed in the NFC with the loss, with the Packers, Rams and Saints all just a half-game behind in the wild-card race. While a win would have put the Vikings
in position to compete for the division crown down the stretch, just holding onto a playoff spot will now be a battle for Kevin O'Connell's squad.

MORE: Latest updates on Justin Jefferson's return date

The Sporting News tracked live updates from the "Monday Night Football" game between the Bears and Vikings. Check out all of the key moments you might have
missed.

Bears vs.
</item 76>

<item 77>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
11:11 p.m.: FIELD GOAL! Carlson's try from distance makes the deficit two possessions.

11:08 p.m.: Not so fast. ... A quick three-and-out gives Green Bay a glimmer. That glimmer grows even brighter after an impressive return by Keisean Nixon.
Let's see what they can do with 3:30 left.

11:04 p.m.: INTERCEPTION! That's wraps. Love misses his target, instead finding the mitts of Jacobs for his second pick of the day. Strong showing by this
Detroit defense. And that should put things to bed.

10:54 p.m.: TOUCHDOWN! Walker's penalty proves vital as Montgomery finds pay dirt from a couple yards out on fourth down. Gutsy call by Campbell. And one that
pays off in spades for Detroit, which extends its lead to 17.

Lions 34, Packers 17

10:50 p.m.
</item 77>

<item 78>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
9:06 p.m. — Big play alert! Goff hits an open Amon-Ra St. Brown, who goes 44 yards to breathe life into the Lions' offense.

9:03 p.m. — The Raiders' offense continues to have nothing in the tank. Las Vegas gains four yards and is forced to punt after less than a minute with the ball.

Lions 6, Raiders 0

8:58 p.m. FIELD GOAL — Detroit was simply trying to draw the Raiders offsides. It didn't work. Riley Patterson's field goal is good, and the Lions extend their
lead to six.

8:56 p.m. — Raymond makes up some yardage on fourth down but comes up about eight yards short of the goal line. The Lions may be considering going for it...

8:55 p.m. — The Lions are moving backwards after a holding call on Taylor Decker.
</item 78>

<item 79>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:30 p.m. — The Bears' defense has suddenly morphed back into an elite unit on these last two plays, forcing intentional grounding by Dobbs and pressuring him
heavily on the following play. It's 3rd & 24 for the Vikings, who might just settle for a gain of a few yards to set up the field goal.

9:27 p.m. — Another first down by K.J. Osborn drives the Vikings into Bears territory, and the Vikings suddenly find themselves 13 yards from the end zone after
Kyler Gordon is called for interfering with T.J. Hockenson. The Vikings have a chance to take the lead with 38 seconds on the clock.

9:25 p.m. — What an effort by Brandon Powell to turn around and make a 28-yard grab while falling to the ground. That's the biggest offensive play of the night
for the Vikings.

A perfect catch from Brandon Powell on a perfect pass
</item 79>

<item 80>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
@jackharlow will be the performer for our Thanksgiving Halftime Show powered by @Verizon!



📰 https://t.co/oUNhlBkWe3 pic.twitter.com/vMCAMpCFfk — Detroit Lions (@Lions) October 30, 2023

7:20 p.m. — Jack Harlow is in the house supporting the Lions.

.@JackHarlow repping the Honolulu blue 🦁



Catch his performance at the @Lions Thanksgiving Halftime Show powered by @Verizon pic.twitter.com/haMpQ95y5U — NFL (@NFL) October 30, 2023

6:49 p.m. — Brian Hoyer is inactive tonight, so Aidan O'Connell is the Raiders' backup quarterback behind Jimmy Garoppolo

6:45 p.m. — Lions WR Amon-Ra St. Brown is officially active after being listed as questionable with an illness.

6:25 p.m.
</item 80>

<item 81>
Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-10-30T22:20:03+00:00)
— The Lions are moving backwards after a holding call on Taylor Decker.

End of first quarter: Lions 3, Raiders 0

8:51 p.m. — Goff comes under heavy pressure on 3rd & 11 but finds a wide open Kalif Raymond for the first down. The Lions are into the red zone.

8:49 p.m. — There's Amon-Ra St. Brown. Goff hits him on back-to-back plays for a total of 20 yards.

8:47 p.m. — Garoppolo's fourth down pass goes to no man's land after Hunter Renfrow falls down. The Lions will be set up not too far from field goal range.

8:46 p.m. — The Raiders will go for it on 4th & 2 from about midfield...

8:43 p.m. — The Raiders' defense rises to the challenge and forces a quick three-and-out. Detroit is punting for the first time tonight.
</item 81>

<item 82>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Former Bears kicker Eddy Pineiro drills the field goal, so Carolina still adds to its lead.

End of first quarter: Panthers 7, Bears 3

8:50 p.m. — Young throws downfield and hits Mike Strachan for a 44-yard gain. Mike Strachan? Mike Strachan. The Panthers are suddenly in business in Bears
territory.

Mike Strachan's first catch as a Panther goes for 45 yards!#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/15stE4IGAz — NFL (@NFL) November 10, 2023

Panthers 7, Bears 3

8:45 p.m. — Santos drills it anyway! A 54-yard kick. The Bears cut the deficit to 7-3.

8:44 p.m.
</item 82>

<item 83>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 83>

<item 84>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Sideline reporter Kaylee Hartung will offer updates throughout the affair.

Fans in the Detroit and Green Bay markets can watch the Amazon broadcast over the air on local channels.

Those in Canada can stream it on DAZN, which carries every NFL game all season.
</item 84>

<item 85>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
On a night that could be defined by bad offense, the Panthers get a huge early boost from their special teams unit. Carolina leads, 7-0.

Ihmir Smith-Marsette takes one back against his former team 🏠📞#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/4MTENIL0r6 — NFL (@NFL) November 10, 2023

8:35 p.m. — Bagent throws incomplete on third down, and the Bears will punt for the second time tonight.

8:33 p.m. — Darnell Mooney goes for a first down after a nice run by D'Onta Foreman set up a 3rd & 2. The Bears have some breathing room.

8:29 p.m. — Johnny Hekker pins the Bears at the one, so it'll be an uphill battle for Bagent and the offense on this upcoming drive.
</item 85>

<item 86>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
6 and No. 7 seed for the Vikings. The ramifications could be much bigger than that, though. A loss would mean the Packers, Rams and Saints only trail Minnesota
by a half-game in the wild card race, while a win would keep the Vikings well within striking distance in the NFC North.

6:55 p.m. — D'Onta Foreman remains out for the Bears, along with CB Tyrique Stevenson.

6:40 p.m. — There's no understating the importance of these final six games for Justin Fields, who is fighting to prove he's the Bears' franchise quarterback.
Chicago has a strong chance of landing the No. 1 overall pick in April's draft, thanks to the Panthers' ugly season.

6:20 p.m. — The Vikings are officially without Justin Jefferson tonight, as they'll hold out their star receiver until they return from their bye in Week 14.

Vikings officially downgraded WR Justin Jefferson to out for tonight’s game vs. the Bears.
</item 86>

<item 87>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Check out all of the key moments you might have missed.

Bears vs. Vikings final score

1 2 3 4 F Bears 0 3 3 6 12 Vikings 0 3 0 7 10

Bears vs. Vikings results, highlights from 'Monday Night Football'

(All times Eastern)

Final: Bears 12, Vikings 10

11:07 p.m. — This game is OVER. The Bears don't score a touchdown and barely take advantage of the Vikings' four turnovers, but they walk out of Minnesota a
winner as the Vikings fall back to .500.

Bears 12, Vikings 10

11:04 p.m. FIELD GOAL — The Bears are back in front! Santos drills it, and the Vikings have 10 seconds to work a miracle.

11:03 p.m.
</item 87>

<item 88>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
A perfect catch from Brandon Powell on a perfect pass



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/f6FTBDH6rC — NFL (@NFL) November 28, 2023

9:22 p.m. — Fields had all day to find open receivers on third down and couldn't locate any. Chicago is 0-for-6 on its last six third down opportunities. The
Bears are preparing to punt after the two-minute warning.

9:19 p.m. — Fields hits Moore for a 22-yard gain, but his next pass is nearly intercepted by D.J. Wonnum. Neither offense has been able to string together
positive plays since that opening drive by Chicago.

9:16 p.m. — Dobbs is nearly intercepted by Jaylon Johnson (again) on third down.
</item 88>

<item 89>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
9:15 p.m.: FIELD GOAL! The good times keep on rolling for Detroit, which adds to its lead after another standout drive spearheaded by Goff and Montgomery. The
Lions are rolling early.

Lions 17, Packers 3

End of first quarter: Lions 14, Pacers 3

8:51 p.m.: Impressive play-calling for the Lions, who continue to carve their way up the field.

8:50 p.m.: Love has been unable to find a rhythm in his first few possessions. Another gives Detroit another chance to make the Lions' life a living hell.

8:43 p.m.: TOUCHDOWN! Montgomery goes into his right tackle before finding the end zone. Detroit's office os cooking with gan.

Lions 14, Packers 3

8:42 p.m.: Decent scamper by Goff brings Detroit right to the cusp of pay first. Lions are closed to adding a second score.
</item 89>

<item 90>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
Bears 6, Vikings 3

9:58 p.m. FIELD GOAL — A questionable play call on third down as Fields throws behind the line of scrimmage to Moore, who gets nothing. Cairo Santos is able to
extend the Bears' lead with a field goal, though.

9:56 p.m. — Fields runs right up the middle for a first down and a bit more as the Bears look to get into field goal range (or better).

9:54 p.m. — Roschon Johnson picks up a first down to keep the Bears' drive alive. He's looked impressive in spurts tonight.

9:50 p.m. — Woah! Kevin O'Connell goes for it on 4th & 7, but Hockenson only gets six yards. The play looked good, the Bears' defense looked better. It's
Chicago's ball with favorable field position.

The Bears defense comes up big with a 4th down stop
</item 90>

<item 91>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Chicago is into field goal range.

Rookie-to-rookie for a key first down 👀#CARvsCHI on Prime Video

Also available on #NFLPlus pic.twitter.com/nIw42qweEE — NFL (@NFL) November 10, 2023

9:14 p.m. — Bagent hits Roschon Johnson for a first down to give the Bears some life. Chicago started with very strong field position, so this could quickly
become a rare scoring opportunity.

9:11 p.m. — A sack by Justin Jones quickly ends the Panthers' drive. Three-and-out for Carolina.

9:07 p.m. — A promising drive ends in a punt for the Bears. Carolina has a chance to extend its lead to two possessions if Bryce Young can build on that 44-yard
completion he had last time he was on the field.
</item 91>

<item 92>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
Brown, Lions (Multiplier 1.5x)

Super FLEX: QB Jared Goff, Lions (Multiplier 1.1x)

Super FLEX: RB Jahmyr Gibbs, Lions (Multiplier 1.2x)

Super FLEX: QB Jordan Love, Packers (Multiplier 1.15x)

Super FLEX: RB Aaron Jones, Packers (Multiplier 1.1x)

Last week was all about defense. This week is all about offense. The Lions feature a potent offense that's three-headed monster of Goff, St. Brown, and Gibbs
should be able to produce. On the Green Bay side of things, Love will hopefully have his top option back in the backfield and that duo cooked the Bears defense
in the opener. If Jones is healthy and suits up, then he's a must-have in your roster after the clunkers that AJ Dillon has posted for the last two weeks.
Otherwise, pivot to a pass-catcher.
</item 92>

<item 93>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 93>

<item 94>
Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-12-03T23:00:04+00:00)
For 59 minutes, the story of Week 13's 'Sunday Night Football' clash was Jordan Love. The first-year starter continued to showcase his evolution as a big-game
player, dicing up Kansas City's All-World defense with incredible efficiency.

Then came the blown calls.

The Packers edged past the Chiefs on Sunday, beneficiaries of another Love masterpiece — an increasingly common sight in the 2023 season. But it was the game's
final moments that drew the most intrigue, mostly because of their incredulous nature.

Jonathan Owens was whistled for an unnecessary roughness that never was on the game's final drive, penalized for crashing into a scrambling Patrick Mahomes on
the field of play. Soon after, Marquez Valdes-Scantling was seemingly impeded by Green Bay cornerback Carrington Valentine on a streak play, an unabashed pass
interference call in most circumstances. The umpires fell silent.
</item 94>

<item 95>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
Good drive by Green Bay, which finally shows signs of life after a putrid first half.

Lions 27, Packers 11

10:05 p.m.: Great throw by Love, who threads the needle to find Romeo Doubs to keep the chains moving on fourth down. Packers on the brink of a much-needed
touchdown.

9:58 p.m.: This Green Bay offense has to come to life sooner rather than later. If not, this scoreboard could get even more crooked.

Good start to the third frame by the Pack, who find themselves marauding down the field with relative ease in the quarter's opening possession. Every journey
began with a single step...

9:20 p.m.: INTERCEPTION! Love's errant pass gets snagged by Jerry Jacobs, who plods 20 yards back to put Detroit on the brink of the promised land. It's been a
putrid start for Love and the Green Bay offense so far.

9:15 p.m.: FIELD GOAL!
</item 95>

<item 96>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
10:55 p.m. — Fields hits Moore for a 16-yard gain on the first play of the drive, gathering some momentum for the Bears ahead of the two-minute warning. Chicago
has plenty of time to sneak into field goal range.

10:52 p.m. — Now it's the Vikings' turn to throw behind the line of scrimmage, which means it's the Vikings' turn to lose yardage. Minnesota passes on a long
field goal attempt and instead will punt it away with 2:36 remaining and the lead just a point.

10:49 p.m. FUMBLE — Fields loses the ball AGAIN. Josh Metellus knocks it out for the second time, and this time Anthony Barr recovers. This has been an ugly,
ugly sequence for the Bears.

10:47 p.m. — Fields throws behind the line of scrimmage to Mooney, and Mooney is predictably dropped for a four-yard loss. That play call has not worked for the
Bears in this half.
</item 96>

<item 97>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
Bears 16, Panthers 10

10:13 p.m. TOUCHDOWN — Foreman runs it in for the score! That's the first offensive touchdown of the night, and it gives the Bears their first lead of the
night.

10:12 p.m. — A well-designed play is broken up by Donte Jackson in the end zone after the Bears pick up the first down.

10:11 p.m. — The Bears are inching their way toward the goal line...it's 3rd & 1 from the five.

10:10 p.m. — Foreman runs 11 yards for a first down, and the Bears are already in field goal range thanks to strong starting field position.

10:06 p.m. — Outside of a 45-yard completion to Mike Strachan, Young is 7-of-16 for 24 yards. It's tough to recall any positive moments for the Panthers'
offense tonight aside from that one play.
</item 97>

<item 98>
Bears vs. Panthers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-11-09T23:04:33+00:00)
The Bears will punt, giving the Panthers a chance to tie the game or more.

10:46 p.m. — Bagent finds Kmet for a first down and more. Another nice play by Kmet, who has become a trusted target for Bagent in these four starts for the
rookie.

Bears 16, Panthers 13

10:42 p.m. FIELD GOAL — Pineiro's kick is good, and the Panthers now trail by a field goal.

10:41 p.m. — Young's pass to Marshall is incomplete on third down. Marshall was well-defended, but he probably should've had that. It'll be a field goal try for
Carolina.

10:38 p.m. — You have to give some credit to Carolina's offense for converting, as Young finds Tremble for the first down on 4th & 6. The Panthers turn a 1st &
30 into a fresh set of downs.

10:37 p.m.
</item 98>

<item 99>
Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments (Sporting News, 2023-09-26T21:53:00+00:00)
For Detroit, it's all about courage. For a team that was once down and out, much like their city, they are now once again on the rise. The Lions are supposed to
be the kings of the jungle and hope to become the NFL's version in 2023. In terms of the food chain, a basic concept taught in science classes across the
country, lions are the hunter rather than the hunted, but it's been the other way around for far too long, as 31 other teams have feasted on a once proud
franchise.

Their motors have been restored and the team has been rebuilt with an attitude and moxie that would make even the most "football guy" of "football guys" blush.
It's a combination of quiet confidence and a downright arrogant confidence that draws people in to root for -- or against -- an underdog.

While the Packers have packed up the Lions for years, that all began to change in 2022.
</item 99>

<item 100>
Lions vs. Packers live score, updates, highlights from NFL 'Thursday Night Football' game (Sporting News, 2023-09-28T23:54:07+00:00)
: Jordan Love and the Packers heading onto the field ahed of tonight's game! We're about 45 minutes from kickoff at the Pantheon of football excellence, Lambeau
Field.

Lions vs. Packers start time

Date: Thursday, Sept. 28

Thursday, Sept. 28 Start time: 8:15 p.m. ET

Thursday night's game between the Lions and Packers will feature in the primetime slot of 8:15 p.m. ET.

What channel is Lions vs. Packers on today?
</item 100>
=== END ITEM id=7785e971dc4d4bcc ===
