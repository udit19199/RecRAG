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

=== ITEM id=31d33f465472725f ===
Gold sentences and facts:
Sentence 0: The winger came firing out of the gate, but the outlook has changed for the Devils in recent days after losing the superstar center Hughes to injury.
  Fact 0.0: The winger came firing out of the gate.
  Fact 0.1: The outlook has changed for the Devils in recent days.
  Fact 0.2: The Devils lost the superstar center Hughes to injury.
  Fact 0.3: The outlook for the Devils changed after they lost Hughes to injury.
Sentence 1: Having to depend much on his weaker arm, Hamlin stayed in contention for the championship four race until the final race in Martinsville.
  Fact 1.0: Hamlin had to depend much on his weaker arm.
  Fact 1.1: Hamlin stayed in contention for the championship four race.
  Fact 1.2: Hamlin stayed in contention until the final race in Martinsville.

Retrieved items:
<item 1>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 1>

<item 2>
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
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 4>

<item 5>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
Stroud in the league's concussion protocol and Nico Collins (calf) joining Tank Dell (fibula) on the Texans' injury list, Houston suddenly finds itself without
many impact skill-position players. Singletary, however, has been pretty damn good and showed us last week that he's the main man over Dameon Pierce in DeMeco
Ryans' backfield. He had 13 carries for 65 yards and a TD against the Jets, while Pierce totaled just five touches for seven yards. Don't be scared of
Tennessee's run D because it ain't what it used to be under Mike Vrabel. Raheem Mostert and De'Von Achane just posted 35.1 PPR points against the Titans, Chuba
Hubbard had 20.2 a few weeks ago, and Rachaad White had 17.8 against them in Week 10.
</item 5>

<item 6>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Oct 8, 2023; Concord, North Carolina, USA; Cars crowd into turn four after a restart during the Bank of America Roval 400 at Charlotte Motor Speedway Road
Course. Mandatory Credit: Jim Dedmon-USA TODAY Sports

There is no denying how the new media rights deal has stunned the NASCAR community. With many still trying to digest it, the deal has undoubtedly given rise to
divided opinions. While the figure of $7.7 billion sounds lucrative, there seems to be more to it than what meets the eye.

America’s Favorite Video Today

It is public knowledge that NASCAR officials invested plenty of time in negotiating the media rights deal. To secure an average annual revenue of $1.1 billion
is something, especially when the sport is supposedly on the wane. However, it looks like the sanctioning body has made it count when it mattered the most.

Is the new media rights deal a game-changer for NASCAR?
</item 6>

<item 7>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
While serving as a guest of the team, Joe Gibbs was seen introducing Aric Almirola to Commanders owner Josh Harris, with lip readers online ascertaining that
Gibbs appeared to tell Harris that Almirola is now driving for them.

According to Bob Pockrass, the expectation is that Almirola will drive part-time for Joe Gibbs Racing -- where he began his NASCAR career in the 2000s -- in the
Xfinity Series, joining a driver lineup that will also include Chandler Smith and Sheldon Creed. Such would align with the expectations for Almirola's racing
future, as he had entertained continuing to race part-time when it was announced he would not return to Stewart-Haas Racing in 2024.

The matter of Almirola's replacement in the SHR No.
</item 7>

<item 8>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
After running the car full-time in 2023, it was announced Thursday that A.J. Allmendinger would go back to full-time Xfinity Series racing for Kaulig in 2024,
though he will continue to run a partial Cup schedule.

The assumption of many had been that the No. 16 would be taken over by Ty Dillon, but that assumption may have been premature. Weaver shared in a recent thread
on X that information suggesting Dillon had been hired to drive the No. 16 was "incorrect," casting ambiguity on both who will join Daniel Hemric at Kaulig next
season as well as the exact plans of Dillon, who has bounced from team to team since Germain Racing shut down following the 2020 season.

Nuts and Bolts
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 9>

<item 10>
FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-13T20:28:00+00:00)
The best part of fantasy football is watching the team you draft go out there and have a chance to win a championship, forever cementing your bragging rights
over your friends for years to come. The worst part of fantasy is watching the team you drafted go out there and get injured, forget how to play the sport, and
then plummet to the bottom of the standings. So if you were one of the lucky ones to reach postseason football in the fantasy world -- congratulations. If you
weren't, good news -- just because the regular season has come to a close doesn't mean we can't still keep searching for ways to win. That's why we're heading
over to FanDuel to craft the perfect Week 15 DFS lineup to give us something to root for all day long while our friends stress about their team faltering
despite dominating for 14 weeks.

Before we break down our picks and strategy, here are the most notable scoring rules for FanDuel contests: four-point passing TDs and half-point PPR.
</item 10>

<item 11>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”
</item 11>

<item 12>
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
</item 12>

<item 13>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
These game plans were conservative enough to signal a potential in-season quarterback change. Denver faces Green Bay and Kansas City over the next two weeks
before heading into its Week 9 bye.

5. You play to win the game, but can you win too quickly? The Giants and Vikings are test cases.

The New York Giants and Minnesota Vikings exceeded expectations under first-year head coaches in 2022. Both reached the playoffs. Both are struggling in their
second seasons, facing difficult questions about the future. Would they have been better off enjoying only moderate success early? That could depend on how well
the football leadership has communicated with ownership.

Every head coach and general manager wants to win as much as possible, but the goal is to build for the long term, which can require time to do things with
sustainability in mind. Bad teams that win right away can reset expectations for fans and ownership, making it less practical to take one step backward in the
short term for a better shot at taking two steps forward down the line.
</item 13>

<item 14>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 14>

<item 15>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
The team had two drivers in the Cup Series playoffs for the first time since its inception. One of the major reasons behind its success in 2023 is #45 Toyota
Camry driver Tyler Reddick.

Alongside team owner Denny Hamlin, Tyler Reddick was the pick of the drivers in the first round of the NASCAR playoffs. The 27-year-old California native
showcased his road course prowess with a win at COTA earlier this year. Carrying the momentum to the playoffs, Reddick won in Kansas in the Round of 16 to gain
more traction for his championship push. He also had one of the best average starting positions among the drivers, with an average of 10.167, despite having
only two poles to his name.

Watch This Story: NASCAR legend Kevin Harvick’s unexpected EV confession

Tyler Reddick finished sixth in the final standings and ended the season with two wins, ten top-5s, and 16 top-10s.
</item 15>

<item 16>
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
</item 16>

<item 17>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Mandatory Credit: Gary A. Vasquez-USA TODAY Sports

While his on-track expertise was one of the notable points in 2023, his off-track antics didn’t go unnoticed either. Ross Chastain was clouded in controversy
after a fistfight with Noah Gragson and on-track altercations with Denny Hamlin and Kyle Larson. While his cutthroat racing style appeased the fans, his driving
skills didn’t waver either.

ADVERTISEMENT Article continues below this ad

Chastain scored two wins, ten 1op-5s, and 14 top-tens, while leading 640 laps in 2023. He also led the NASCAR Cup Series standings for seven weeks and had a
career-best 28 lead lap finishes. Overall, Ross Chastain has been one of the most resilient NASCAR drivers of 2023 and deserves a mention among NASCAR’s very
best.
</item 17>

<item 18>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Since Brad Keselowski arrived in 2022, the team has continued evolving on and off the track. While Keselowski has certainly instilled his experience and winning
mentality within the team, he failed to produce a race win in 2023. Instead, it was Chris Buescher who outshone the Cup Series veteran with his best-ever season
in NASCAR’s premier competition.

Chris Buescher ended his season with three wins, nine top-5s, and 17 top-tens. Two of his three wins in 2023 came in succession at Richmond and Michigan.
Buescher continued his dream run in the Cup Series playoffs and eventually finished 7th in the final standings. The #17 Ford Mustang driver could not, however,
secure a win in the Round of 8 and did not have enough points to make a case for the championship four race at Phoenix.
</item 18>

<item 19>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
It wouldn't make much sense for a four-win team to rush back its franchise player.

Patriots dual-purpose back Rhamondre Stevenson (ankle) suffered an injury on a nasty hip-drop tackle in New England's Week 13 loss to the Chargers. Bill
Belichick and company already noted that 'Mondre would miss multiple weeks, so we're bumping Ezekiel Elliott way up against the Steelers.

D'Andre Swift took a hard hit and exited Philadelphia's game late against the 49ers, but it's tough to know if he's actually hurt or was just kept out because
of the lopsided score. If Swift can't go in Week 14 against Dallas, Kenneth Gainwell and Boston Scott will split touches, with Gainwell likely garnering flex
value.

Jonathan Taylor underwent thumb surgery last week and is expected to miss a few weeks. Tough for Taylor owners to ride the home stretch of the regular season
without JT, but at least they could get him back for the end of the fantasy playoffs.
</item 19>

<item 20>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Discovery, those races are going to be simulcast not just on MAX streaming but also on TNT and TNT is cable, not a free-to-air channel.”

Stern continued explaining how it can come back to haunt. He said, “You have that on the cable subscription. So, the number of races that are going down that
have been on the network, that’s what teams are concerned about because the ratings will probably be lower.”

ADVERTISEMENT Article continues below this ad

With not every fan having access to all platforms, it is a tricky situation both for the fans and NASCAR. However, the sanctioning body has gone ahead and made
its move with the new media rights deal. Now, it remains to be seen how the execution part works out. For now, it looks like NASCAR has secured itself well by
bringing more money to the table.

Kyle Larson Believes NASCAR’s Multi-Billion Dollar Media Rights Deal Is a Step in the Right Direction Despite Fans’ Backlash
</item 20>

<item 21>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
The season of giving is here as we wind down the 2023 NFL season. We've already given you winners and losers, highs and lows, sleepers and busts, and a few
moments to forget, but we hope to keep giving you more of the positives to look back on as the season enters its final weeks. If you need to scratch that
fantasy itch with your season-long league coming to an end, look no further than our FanDuel DFS lineup for Week 14, which will ideally provide us with enough
cash to boost the economy this holiday season.

Before we break down our picks and strategy, here are the most notable scoring rules for FanDuel contests: four-point passing TDs and half-point PPR.
</item 21>

<item 22>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Autosport has revealed its annual list of the 2023’s Top 50 drivers. The renowned motorsports publishing brand has continued the tradition since 2002, and every
year, the announcement creates some suspense within the racing community. Highlighting the 50 standout performers in motorsports for the year, Autosport has
included several NASCAR drivers in the prestigious list.

America’s Favorite Video Today

William Byron leads the list of NASCAR drivers with a 12th-place ranking as Kyle Larson and 2023 Cup Series champion Ryan Blaney join him. The scarcity of
NASCAR drivers in the rankings has raised some eyebrows. With the year producing numerous cases of NASCAR drivers standing out with their consistency and skills
behind the steering wheel, let us look at some of the major exclusions from Autosport’s Top 50 Drivers of 2023.

1. Chris Buescher

ADVERTISEMENT Article continues below this ad

RFK Racing found a fresh breath of motivation this year.
</item 22>

<item 23>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 23>

<item 24>
“We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change (Essentially Sports, 2023-11-06T23:01:17+00:00)
The wisdom of experienced leaders like Roger Penske plays a crucial role in the success of their teams.

“We’re Champions”- Roger Penske Flexes on Rick Hendrick As He Publicly Boasts Team Penske’s Strength With Back-to-Back Titles
</item 24>

<item 25>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
"There have been some challenges this year, but I wouldn't trade them for anything in the world," Gragson said. "I don't know that I would change anything that
has happened because I have learned from it and it gives me a chance to be a better person.

"In life, you either win or you learn, on and off the racetrack. The wins have been great, but you learn through the hardships and defeats. I am grateful to
have good people around me, and I'm working really hard every day to hopefully get a second chance and to make the most of it."

While the Stewart-Haas No. 10 seat is seemingly filled, some intrigue still exists as to exactly who will drive the No. 16 for Kaulig Racing next season. After
running the car full-time in 2023, it was announced Thursday that A.J.
</item 25>

<item 26>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
As for this season, not so much.

Sean Payton should be at the forefront of the conversation. It's one thing to turn around a team that was dreadful last season. But he has taken one that got
off to a horrible start and turned things around in-season.

Oh — and he has resurrected Russell Wilson's career.

That alone should put him at the top of the list, but he may need to get Denver into the playoffs to convince voters.

Comeback Player of the Year

Current favorite: Demar Hamlin

Longshot: Russell Wilson

Coming back from a life-threatening injury like a cardiac arrest makes Hamlin a great candidate to win CPOY. However, it would be understandable if voters were
to look elsewhere since Hamlin has seldom been active on game day and rarely seen the field.

It will depend on how vital voters deem playing to be.

Wilson was supposed to be the next great franchise quarterback for the Broncos when they acquired him prior to last season.
</item 26>

<item 27>
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
</item 27>

<item 28>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Nobody has really emerged in the AFC to unseat these teams at the top.”

• Miami Dolphins: Has Tua Tagovailoa proven he’s built to last?

What I think: It was tough watching Tagovailoa last season without worrying for his physical well-being. I’ve never felt that way watching him so far this
season, and will feel even better if he plays a full game against Philadelphia in Week 7. New England, Kansas City, the Jets (twice), Washington, Tennessee,
Dallas, Baltimore and Buffalo will all get their shots at Tagovailoa. I’m cautiously optimistic, but think there will be games when defense muddy the picture
sufficiently to create turnovers.

Exec comment: “Interesting team. They put up 70 on Denver and should have put up 70 on the Giants. No one can keep up with that kind of firepower. But they also
haven’t really beaten anyone. They were down 14 points to Carolina.
</item 28>

<item 29>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
They also seemed invigorated for Jonathan Gannon on defense and special teams. Look for the Cardinals to finish strong and play a whole lot of spoiler.

30. New England Patriots 2-8 (28)

The Patriots tried to win a game by hiding Mac Jones and their quarterbacks in general. They almost pulled it off with the running game dominating and
short-handed defense making a lot of plays, but they didn't see the results in Germany ahead of a tough bye for Bill Belichick.

31. New York Giants 2-8 (30)

The Giants can just flush away this season without Daniel Jones and Tyrod Taylor. The injuries go to their offense early and remained relentless, and now their
defense is falling apart, too. The question is, will Brian Daboll get a massive mulligan after his rookie playoff leadership?

32.
</item 29>

<item 30>
NFL awards odds: Six longshots bettors should consider for MVP, OPOY, DPOY and more (Sporting News, 2023-12-05T20:45:25+00:00)
NFL award longshots can be easy to overlook since each has a favorite(s). The longshot may be playing well, but other players have commanded the spotlight,
making it easier for them to go unnoticed.

Whatever the reason, they have not gotten the credit they deserve. However, with five weeks left in the regular season, there is plenty of time for a longshot
to become a contender.

But if you wait too long before placing a bet, your payout will be a lot less than if you were to roll the dice on a longshot right now.

Longshots to consider for NFL MVP, OPOY, DPOY, Coach of the Year and Comeback POY

There is always a reason why a player or several players have relatively short odds — they've been the best at their position throughout the season.
Consequently, there is just as good a reason why longshots have odds as long as they do.
</item 30>

<item 31>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
With an average starting position of 9.6, the 29-year-old was the sole driver to have an average starting position below 10. A win in the Round of 8 race at
Homestead-Miami propelled him to a consecutive Championship 4 appearance. Christopher Bell also dominated the qualifying races throughout the season, especially
in the playoffs. Bell recorded poles in the first three races of the playoffs to begin his playoff campaign on a superb note.

The Joe Gibbs Racing driver finished the season with two wins, 19 top-tens, and ten top-5s in 2023. He eventually finished fourth in the final NASCAR Cup Series
standings and was another one of the prominent performers of the calendar year.

4. Tyler Reddick

2023 was a year of milestones for 23XI Racing. The team had two drivers in the Cup Series playoffs for the first time since its inception.
</item 31>

<item 32>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 32>

<item 33>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
The baseball season for the New York teams didn’t turn out to be their most successful one. Both the New York Mets and the New York Yankees failed to make it to
the playoffs and came under scrutiny as the highest spenders in the league. But the level of accounting has come higher for the 27-time winners because of the
highest expectations. The management has comprehended the situation and is actively seeking to make changes for the 2024 season. Be it trimming the squad and
releasing a few players, including superstar Gleyber Torres, they look to bring about changes. But given that they are already short on options, it brings up
the question: Is the move justified?

America’s Favorite Video Today

Although it was a decent season for the second baseman, the lack of consistency brought Torres under scrutiny for changes for the upcoming season. Also, given
he is currently in arbitration with the Yankees, it adds to his trade chances.
</item 33>

<item 34>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
"I think the cars can be quieter just to help the fan experience."

Notable crew changes

Not one, but two of the biggest stars in NASCAR will have new eyes in the sky in 2024, as the spotters for both Chase Elliott and Ryan Blaney announced they
will be stepping down. Eddie D'Hondt announced Saturday he would not return as Elliott's spotter next season, followed Monday by Josh Williams announcing he
would step down as Blaney's spotter.

According to Bob Pockrass of Fox Sports, former NASCAR Xfinity Series driver Tim Fedewa, the longtime spotter for Kevin Harvick, will take over as the spotter
for Blaney next season. A possible landing spot for Williams exists at Spire Motorsports, as Williams has previously spotted for Zane Smith in the Craftsman
Truck Series.

Spire Motorsports announced a number of personnel moves this week, hiring Stephen Doran as the crew chief for Smith's No.
</item 34>

<item 35>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
DEF New Orleans Saints vs. Panthers ($4,300). The Panthers are and have been a complete disaster this year, only shielded by the fact that they play in
Charlotte rather than the Boston and New York areas. They have a QB who can't compete with the extreme lack of talent up and down the roster, a lame-duck
coaching staff, and countless other factors. They're 1-11 and are playing a divisional rival that, despite being 5-7, is somehow not out of the playoff race.
New Orleans' offense is a mess, but the defense should be primed to bounce back after a rough Week 13.

Best SuperDraft NFL DFS Picks for Week 14

This year, we're also listing our favorite DFS plays on SuperDraft.

For those unfamiliar, SuperDraft's DFS scoring involves point multipliers as opposed to traditional salary-capped or tiered contests.
</item 35>

<item 36>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
Most season-long fantasy leagues are entering their final week of the regular season in Week 14, and if your team isn't going to qualify for the postseason,
then welcome to the DFS side of things, where every week is a new season and every DraftKings lineup can be the one that changes your life.

In this world, we don't have to sweat the injury reports, make difficult trades, or even concern ourselves with bad matchups. Instead, we get to enjoy each week
for what it is -- a new opportunity. Whether it's your first time or your 200th time, there's something to keep us all coming back. Without further ado, welcome
to our DraftKings lineup for Week 14.

Before we break down all of our picks, here are the most notable scoring rules for DraftKings contests: Full-point PPR, four-point passing TDs, three-point
bonuses for 100 rushing yards, 100 receiving yards, and 300 passing yards.
</item 36>

<item 37>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
He said, “NASCAR is in a tough situation because they had to deliver more money as well and clearly if they went this route, that’s a good indication that’s
because they had to go this route to deliver more money for the industry.”

While NASCAR secured an impressive deal, there could be downsides to it and Stern explained it beautifully.

Adam Stern points out a key potential drawback with NASCAR’s new media rights deal

Safe to say, NASCAR has been struggling to keep their teams happy in terms of profitability. Since the introduction of the Next Gen cars, it has only become
more difficult for the Cup Series teams and teams in other competitions. While they have taken a commercial hit, they have piled up pressure on NASCAR to help
out in this case.

ADVERTISEMENT Article continues below this ad

In addition to that, Sports Business Journalist Adam Stern detailed one important issue with the new media rights deal. Speaking on the same podcast, he said,
“When you look at the deal with Warner Bros.
</item 37>

<item 38>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
All you have to do is choose your favorites, sit back, and relax.

WEEK 15 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

If daily fantasy was as easy as picking all top-end skill-position players from one week to the next, it would be a lot easier to win the milli-maker. Let's
face it, in order to win big, we need to hit on some diamonds in the rough, and in order to afford a CeeDee Lamb ($9,200 on DK), we have to invest in the
occasional Demarcus Robinson ($3,600).

We have enjoyed consistent success with our DFS sleepers and values all season. Each Tuesday, we share a big list of our hits from the previous week. — with
each hit typically netting at least 16-22 fantasy points.
</item 38>

<item 39>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
“If before the game someone offered Philly to put its defense against Zach Wilson on a need-field-goal drive starting inside his own 15 with no timeouts and a
minute left, you think they would have taken it?”

In the situation Sunday, the Jets’ offense had netted 19 yards on its previous three possessions, producing one field goal on a drive that gained 16 of those
yards. The Eagles had already suffered three turnovers. They knew Wilson would be taking over deep in his territory with very little time remaining.

Advertisement

“To me, these are exactly the type of factors that a coach needs to consider when deciding,” the exec said. “I think they were ignored here.”

The situation Philly faced against the Jets was the seventh time since 2000 that a team leading by 1-2 points in the final two minutes faced third down with
7-10 yards to go from the minus-40 yard line to midfield, with the opponent holding no timeouts.
</item 39>

<item 40>
FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-13T20:28:00+00:00)
Jordan Love appears to have turned a corner, meaning we should expect him to put on a show back at home. Reed will be the primary beneficiary on Sunday.

DEF New York Giants @ Saints ($3,500). There is a lot of bad offense across the NFL this season, and despite signing Derek Carr to make sure they wouldn't be
listed among them, the Saints are there anyway. The QB refuses to sit out and deal with the laundry list of injuries ailing him at this point despite them
clearly being an issue. Teammates appear fed up with Carr and would seemingly prefer the team return him to the dealership. The Saints are upside-down on this
deal right now, with no end in sight. The Giants aren't good, but they have enough defensive pop to contain an offense that keeps spinning its tires most weeks.

Best SuperDraft NFL DFS Picks for Week 15

This year, we're also listing our favorite DFS plays on SuperDraft.
</item 40>

<item 41>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The winger came firing out of the gate, but the outlook has changed for the Devils in recent days after losing the superstar center Hughes to injury. Hughes
went down after slamming into the boards against the Blues, forcing New Jersey to change things on the fly. The changes have directly impacted Bratt, who was
featured on the top line alongside Hughes. With fellow center Hischier also out with an injury, leaving the Devils fairly weak down the middle and trusting
Michael McLeod to hold down the fort in between Bratt and Tyler Toffoli. As a result, Bratt has been held without an assist in each of the last two games for a
team that's missing a key contributor on offense. And it'll need him to look to score rather than play the role of playmaker. Until the Devils figure things
out, we can't trust this team to do anything positive in the offensive zone.

Win BIG with SuperDraft!
</item 41>

<item 42>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
He also led 470 laps throughout the season and recorded his best-ever finish in the Cup Series. Reddick was among the dark horses for the 2023 NASCAR Cup Series
title and certainly held the potential to win the competition.

5. Ross Chastain

Ross Chastain‘s performance in 2023 was one of the many positives for Trackhouse Racing this season. The 30-year-old Floridian created history in the season
finale by becoming the first non-championship driver to win the Championship 4 race since the introduction of the new playoff system in 2014. Driving the #1
Chevrolet Camaro, the “Melon Man” equaled his career-best tally of two wins in a season after victories in Nashville and Phoenix.

USA Today via Reuters Nov 5, 2023; Avondale, Arizona, USA; NASCAR Cup Series driver Ross Chastain (1) celebrates his race victory following the Cup Series
Championship race at Phoenix Raceway.
</item 42>

<item 43>
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
</item 43>

<item 44>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Happy December, fantasy friends! We have somehow reached the final week of the fantasy football regular season, so the running back decisions we make in our PPR
leagues this week could very well impact our playoff fate. Our Week 14 fantasy RB PPR rankings will help you optimize your lineup at RB1, RB2, and flex heading
into this crucial weekend.

It hasn't been all holly, jolly this season — in fact, it has often been a bumpy sleigh ride. We've suffered a plethora of injuries to stud RBs the past few
weeks, but we keep rolling with the punches and always do our best to stay two steps ahead of our opponents. One major positive is that only two teams will be
on bye this week, a welcome relief after the six-team byepocalypse that was Week 13. Still, the injury-ravaged teams will have trouble replacing their studs and
the deep squads will have difficult start 'em, sit 'em decisions to make.
</item 44>

<item 45>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”
</item 45>

<item 46>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
The matter of Almirola's replacement in the SHR No. 10 has been a rather poorly-kept secret: It is widely expected the team will hire Noah Gragson as its new
driver as Gragson looks to reclaim his standing as a bright young star following a disastrous rookie year in Cup with Legacy Motor Club. Gragson was recently
spotted at Stewart-Haas Racing's shop, and he also drove a Ford in this past weekend's Snowball Derby with Rette Jones Racing.

Gragson did not let on to any 2024 plans in a recent conversation with Matt Weaver of Sportsnaut, instead choosing to discuss his regrets and personal growth
from the end of his rookie season. Gragson resigned as the driver of the No. 42 in August while serving a month-long suspension after he was spotted having
liked a racially insensitive meme on his Instagram account.
</item 46>

<item 47>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
The wideout still has another hurdle to pass to be cleared to play on Sunday so his status is still very much up in the air.

Falcons (-2) at Jets

Atlanta is relatively healthy coming into this matchup. Hollins, who injured his ankle in Week 9, participated on a limited basis every day this week, which is
encouraging for his prospects to play on Sunday. As for Hughes, he practiced fully all week after suffering a hand injury last weekend against the Saints.
Linebacker Nate Landman (thigh) missed practice on Wednesday but will play vs. New York.

Rodgers was once again limited in practice but, as expected, will not be activated this week and will be sidelined as he continues his recovery from a torn
Achilles just a few months ago. Meanwhile, Cook popped up on the injury report due to a shoulder injury, but head coach Robert Saleh told reporters Friday that
the back should be fine.
</item 47>

<item 48>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
3. Christopher Bell

As unfortunate as Denny Hamlin was, his JGR teammate Christopher Bell must have felt hard done by after an impressive season. The #20 Toyota Camry driver
progressed to the championship race for a second successive season. However, a cruel twist of fate waited for him in the season finale at Phoenix Raceway, as he
was forced to retire early from the biggest race of his career with a broken brake rotor.

via Imago NASCAR, Motorsport, USA Food City Dirt Race Apr 9, 2023 Bristol, Tennessee, USA NASCAR Cup Series driver Christopher Bell 20 wins the Food City Dirt
Race at the Bristol Motor Speedway Dirt Course. Bristol Bristol Motor Speedway Dirt Course Tennessee USA, EDITORIAL USE ONLY PUBLICATIONxINxGERxSUIxAUTxONLY
Copyright: xRandyxSartinx 20230409_lbm_bs1_024

Bell’s 2023 season has gone under the radar.
</item 48>

<item 49>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Tampa Bay has a pretty good run defense, so Allgeier can't be trusted outside of a desperate flex play with his nine average touches over the past two weeks.

What are the biggest RB injuries in Week 14?

Derrick Henry (head/neck), Rhamondre Stevenson (ankle), Kenneth Walker III (oblique), Aaron Jones (knee), D'Onta Foreman (ankle), and D'Andre Swift
(undisclosed) are all "questionable," at best, to suit up this week. We know for sure Stevenson will be out, but we'll have to monitor the injury reports
throughout the week for the rest.

Jonathan Taylor (thumb) underwent surgery last week and will miss the next few weeks, opening the door for Zack Moss to resume his role as Indy's super-sub lead
back.

We'll be updating these RB PPR rankings all week, so check back for the latest player movement.
</item 49>

<item 50>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
71 Chevrolet and Luke Lambert as the crew chief of the No. 77 for Carson Hocevar. Doran joins Spire from Stewart-Haas Racing, where he was an engineer on the
No. 4 team, while Lambert moves to Spire after being crew chief of the No. 42 at Legacy Motor Club in 2023.

In addition, Spire has also hired Doug Duchardt, whose career in racing has included stints at General Motors and Chip Ganassi Racing, as its new team
president.

Drivers on the move



Unbeknownst to most at last Sunday's Washington Commanders game, many in NASCAR became aware of a part of Joe Gibbs Racing's 2024 plans in the Xfinity Series
and the apparent hiring of a Cup veteran.
</item 50>

<item 51>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.
</item 51>

<item 52>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
"It was big. The first laps I had I was like, 'Man, this thing drives way different,'" Blaney said. "... It's a massive aero loss when you do that. And then the
simple diffuser wasn't as big of a change as the splitter, but it was still something to feel."

The topic of mufflers -- while anathema to the purest of gearheads -- has been one that has occasionally popped up in the Next Gen era, particularly with NASCAR
now using mufflers in urban settings at stadiums (Los Angeles Coliseum) and on street courses (Chicago) where noise ordinances are more of a consideration. Kyle
Larson told reporters he did not notice a difference in cockpit cooling with the mufflers tested on Wednesday, but spoke positively of the idea overall.

"I definitely think our racecars are way too loud, and probably are still too loud with the mufflers," Larson said. "I think the cars can be quieter just to
help the fan experience."
</item 52>

<item 53>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
Every year, the wintertime and the NASCAR offseason is a welcome -- but very short -- respite from what is otherwise a demanding, year-long grind of racing.
Come 2024, that grind is soon to begin again, and exactly what it looks like will be based in large part on these months where driver lineups, crew lineups and
rules packages are set.

With much activity going on within the confines of race shops across NASCAR, here are some major storylines for the week of Dec. 4 as the offseason continues.

Offseason testing in Phoenix

One month after Ryan Blaney was crowned the Cup Series champion and the 2023 season was completed out west, Blaney and a select group of other drivers and teams
returned to Phoenix Raceway for a two-day test as NASCAR looked at a number of aerodynamic and technical changes for the Next Gen car.
</item 53>

<item 54>
“We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change (Essentially Sports, 2023-11-06T23:01:17+00:00)
It’s a win-win situation for both the brand and racing enthusiasts. Although, all this might not have come to pass were it not for a timely intervention by “The
Captain.”

Roger Penske played a big part in calming Ryan Blaney down in the clutch moment of the race

In the heat of the championship race at Phoenix, Ryan Blaney’s competitive spirit almost got the best of him. He was racing for the championship and was eager
to secure the lead from Ross Chastain in the closing few laps. However, he was also aware that he needed to maintain his position to win the title.

Watch This Story | Star Wars Saga: Bubba Wallace’s Galactic Surprise for Ryan Blaney

ADVERTISEMENT Article continues below this ad

In the post-race conference, the Team Penske driver expressed his frustration, stating, “There is no secret that I can snap on the radio. That’s been my whole
life. That’s been my whole career.
</item 54>

<item 55>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Tuesday night's NHL national games feature an Original Six matchup at Madison Square Garden in addition to a battle between the team that left and the team that
took its place. The Rangers and Red Wings will face off at the world's most famous arena as both squads have jumped out to hot starts in the new season. Dylan
Larkin is a game-time decision for the Motor City while injuries to the Blueshirts' Adam Fox, Filip Chytil and Igor Shesterkin are making sure some of this
matchup's biggest stars will not be watching in uniform. Regardless, there is still plenty of star power on Broadway in what should feature plenty of offense.

In the late game, the Devils return to Colorado -- the place they left in order to move to New Jersey -- for a showdown with the Avalanche. Again, injuries are
a big part of the storyline here, thanks to Jack Hughes and Nico Hischier out until further notice.
</item 55>

<item 56>
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
</item 56>

<item 57>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
USA Today via Reuters Jul 30, 2023; Richmond, Virginia, USA; NASCAR Cup Series driver Chris Buescher (17) celebrates in Victory Lane after winning the Cook Out
400 at Richmond Raceway. Mandatory Credit: Amber Searls-USA TODAY Sports

Overall, he has been one of the season’s standout performers and displayed incredible grit and consistency throughout the year. His performances in 2023 make a
strong case for a place in the best 50 drivers of the year.

2. Denny Hamlin

Finding a fan of Denny Hamlin in the NASCAR fandom is a tough task. Continuously surrounded by controversy, Joe Gibbs Racing‘s #11 Toyota Camry driver has been
subject to a lot of criticism. However, his ability as a race car driver has never been in doubt. As the season entered its decisive phase of the playoffs, it
seemed like 2023 would be the year Hamlin’s championship drought finally ended.
</item 57>

<item 58>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Is the new media rights deal a game-changer for NASCAR?

ADVERTISEMENT Article continues below this ad

Just a couple of days back, NASCAR finalized a $7.7 billion deal with Fox Sports, NBC, Amazon, and Warner Bros. Discovery. With such a lucrative deal securing
seven years of revenue, surely it seems like a lot of money. But is that really the case?

NASCAR secured almost a 40% increase over their current media rights deal yet there’s something that doesn’t fit perfectly. Sports Business Journal journalist
Adam Stern recently cleared a few things out when he attended the Dirty Mo Live podcast. With the Cup Series already facing a 5% drop in viewership this year,
Stern discussed how the new deal could impact the ratings.

At first, the renowned journalist explained how the sanctioning body would have loved to stay with Fox Sports and NBC. However, given that it would have most
likely resulted in less money, NASCAR opted to explore other options.
</item 58>

<item 59>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
It's make-or-break time in fantasy land. Only one week stands between now and the fantasy football playoffs, so every decision we make could be the difference
between an optimal finish and a womp-womp ending. Our Week 14 fantasy RB rankings for standard leagues will help steer your fantasy ship to the smoothest
possible sailing, guiding you to the running back studs, sleepers, busts, and injuries of note going into the most important week of the regular season.

The rollercoaster that the 2023 season has been rolled on through Week 13, with yet another fresh batch of injuries. Luckily for us, we have strong stomachs
(and we're not above taking offers from potential sponsors, TUMS and Kaopectate!). So, like always, we will navigate through the injuries to provide you with
the most up-to-date information and analysis.

Let's get to our Week 14 RB rankings and hopefully help you close out the regular season in style.
</item 59>

<item 60>
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
</item 60>

<item 61>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
It will be interesting to see how the most successful team approaches this. Will they be able to find genuine replacements on the market? Only time will tell.
His trading is not worth it for the Yankees, which will further reduce the options in the outfield. With rookies and the farm not compensating for the demand,
it would be a big ask for them to fulfill. With Jasson Dominguez Anthony Rizzo , and DJ LeMahieu

Watch the Story: Ranking the New York Yankees Biggest World Series Failures
</item 61>

<item 62>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
There isn’t a Josh Allen on the Giants or Vikings.

Minnesota in particular finds itself in an interesting spot, with Kirk Cousins approaching free agency and playing well enough, most likely, to keep the Vikings
from climbing too high in the draft order.

Advertisement

“They already gave a contract to (tight end) T.J. Hockenson, and they are going to give one to (receiver) Justin Jefferson,” an exec said. “Now, what? Are you
going into the rookie pool and taking the fourth-best quarterback in the draft? You can look like Atlanta with (Desmond) Ridder pretty quick.”

6. Two-minute drill: There were already many ways to win a game before the Dolphins invented another one.

The Dolphins were one of 223 teams since 2000 to be favored by at least 13.5 points in a regular-season or playoff game. Seventeen of those 223 teams fell
behind by 14 or more. Six of the 17 won anyway.
</item 62>

<item 63>
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
</item 63>

<item 64>
Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil (Essentially Sports,
2023-09-28T21:31:48+00:00)
Events like these also help to keep these smaller race tracks across the country in good shape. If a track isn’t used or taken care of for an extended period of
time, it will start to wear out and eventually become completely unsuitable for any kind of racing unless it is taken apart and rebuilt. But one of Kyle
Larson’s reasons for doing what he does with the High Limits Racing Sprint Car Series is a lot simpler, and he revealed that goal to Dale Earnhardt Jr. at the
Lernerville Speedway.

“Can’t Have Everything You Want”- Dale Earnhardt Jr. Ruthlessly Trashes Home Track as He Endorses Wild NASCAR Rumors

“I was hoping this would add pressure to the weekend shows to raise their purses as well so I think we’ve seen a big step in that this year and I think it’s
gonna continue forward so, I would say so far it’s been successful,” he added.
</item 64>

<item 65>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 65>

<item 66>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 66>

<item 67>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
However, even a moderate MCL sprain could mean a multi-week absence for the injury-prone veteran back, especially in a lost season for the 4-6 Pack. Rookie
Emanuel Wilson also hurt his shoulder in Week 11, so AJ Dillon could be getting the keys to Green Bay's backfield castle (although it's more of an oversized
shack, at this point).

Dolphins rookie De'Von Achane, who looked like the runaway Offensive Rookie of the Year earlier this season, looked to have aggravated his MCL injury upon his
Week 11 return against the Raiders. Achane has a chance to return for Miami's Week 12 tilt with the Jets on Friday, but we wouldn't be surprised if the 7-3
'Fins play it safe and keep him out until he's back to 100 percent. Meanwhile, Miami placed backup Salvon Ahmed on IR with a foot injury, so Jeff Wilson Jr.
could be back in play as a handcuff stash.
</item 67>

<item 68>
Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player (The New York Times, 2023-12-15T21:32:52+00:00)
It’s hard. This s— is hard. And it’s not just (the losses). I’m 32, not this young spring chicken anymore. You know, s— hurts all the time. But my perspective
is I get to play a game that I love, and I’m healthy enough to play. Every time you come out of a game, that’s a win in its own way.”

GO DEEPER Bills training camp preview: Is 2023 final run for Micah Hyde-Jordan Poyer tandem?

Poyer’s commitment is such that last season, when a rib injury precluded him from airplane travel, he took a 32-hour round-trip drive to play in a game in
Kansas City. He wants to lift a Lombardi Trophy badly, as much as ever. But if it doesn’t happen, he believes he can handle the disappointment better.

On Fridays this season, Poyer, fellow safety Taylor Rapp and assistant strength coach Will Greenberg play crystal bowls at Bills headquarters.
</item 68>

<item 69>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.
</item 69>

<item 70>
Kyle Larson Breaks Down His Sprint Car Series Journey and How He Partnered Up With Dale Earnhardt Jr.’s Former Pupil (Essentially Sports,
2023-09-28T21:31:48+00:00)
Kyle Larson had taken part in the Lernerville race on Tuesday, only a couple of days after a hectic race in Texas, and it seems like he had brought some of that
form into the dirt track race because he made a slightly similar mistake.

Cup Series hopeful has Texas deja vu at Lernerville

With just nine laps to the checkered flag, Kyle Larson was leading the race and seemed to take the victory and earn himself $50,000 in his own competition.
However, much like what happened in Texas, Larson lost control of his car as he was trying to defend the lead of the race from co-owner Brad Sweet.

Unlike Texas, he did not DNF and managed to get his car back on track, but his chance to win had gone. The Cup Series HMS driver could only finish in eighth
place. A racer as competitive as Larson would have liked a win, especially after what had happened the Sunday before, but it was a good day for him as the owner
as well.
</item 70>

<item 71>
NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups (Sporting News, 2023-12-13T20:32:00+00:00)
It's somehow already mid-December, meaning millions of fantasy football owners just reached their respective "offseason." Luckily for those who missed the
playoffs, there's Week 15 DFS. We hold the key to consistent daily fantasy success: pinpointing and playing the cheapest sleepers and highest-ceiling value
plays each week.

Every Tuesday, we scour DraftKings, FanDuel, and SuperDraft for affordable sleepers and under-the-radar value plays that will set you apart from the DFS pack.
By finding stud production at modest prices, we allow ourselves plenty of remaining budget to afford elite weekly performers at other roster spots. When our
studs and our sleepers hit, we usually win.

We do all the leg work for you -- poring over the player pools, like my daughter searching the Target Christmas catalog — then we handpick the most promising
sleepers and tell you why they're worth drafting to your next DFS squad.
</item 71>

<item 72>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The Rangers head into this matchup with a makeshift lineup thanks to those injuries that is forcing them to adjust. The absences of Fox and Chytil are a big
deal, however, as Panarin loses Fox, who quarterbacks the power play, and Chytil, who centered his line at even strength. Panarin is still all over the ice for
the Blueshirts though and is even seeing 18.8 percent of his shifts start in the offensive zone, a welcome sight after the last two seasons with Gerard Gallant
behind the bench. The Red Wings have allowed the 11th most SOG this season (371) and the fourth most power play opportunities (46). If the Rangers can play with
the man advantage, Panarin should hit this number with ease, but he's enough of a factor at even strength that we like the OVER here anyway.

MORE SUPERDRAFT DFS & FANTASY PROPS: How to play SuperDraft

2.
</item 72>

<item 73>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
Also, given he is currently in arbitration with the Yankees, it adds to his trade chances. The reason is that the team won’t want to lose the opportunity to
cash in some money from his trade. The decision appears promising as the team seeks a shift, but the data tells a different story. Gleyber Torres’s Exit Becomes
the Talk of the Town as the Yankees Look for a Rebuild The New York Yankees have been in the news since the league stage ended. First, because of the
possibility of a third-party audit, which was later shunted out. Later, the comments from Aaron Judge brought a lot of attention to the workings of the team,
conflicts, and divisions between the core four. Now it’s the assumptions of changes with the idea of cutting down the $300 million budget for the next season.
One name that is making the rounds for an exit is Gleyber Torres.
</item 73>

<item 74>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.
</item 74>

<item 75>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 75>

<item 76>
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
</item 76>

<item 77>
Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-11-30T22:10:58+00:00)
I'm pretty certain they're not bad enough to make Tim Boyle look like Tim Tebow or even Tim Hasselbeck, but I do wonder if they're juuust bad enough to give
better-than-normal numbers to Garrett Wilson and Breece Hall. Reinforcements along the O-line should help the Jets' cause against a Falcons defense that's
blitzed at a league-average rate, is eighth-lowest in pass rush pressure rate and is fifth-lowest in sacks (28).
</item 77>

<item 78>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 78>

<item 79>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
With Zach Wilson reportedly reluctant to play and further destroy a career hanging on by a thread, that leaves us with Tim Boyle, Trevor Siemian, or the lucky
fan from Section 139 to lead the Jets to yet another embarrassing defeat. As the Texans try to make a playoff push, this one could get ugly.

Best SuperDraft NFL DFS Picks for Week 14

This season, we're also listing our favorite Week 14 DFS lineup on SuperDraft, with some of our picks and analysis overlapping.

For those unfamiliar, SuperDraft's DFS scoring involves point multipliers as opposed to traditional salary-capped or tiered contests. A winning lineup will
likely feature a strong mix of stud fantasy contributors with 1x-1.6x boosts, as well as a few matchup-based sleepers with larger multipliers. Each player is
assigned a scoring multiplier based on their value as determined by SuperDraft. Users can roster any player in their lineup with no restrictions.

Win BIG with SuperDraft!
</item 79>

<item 80>
FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-13T20:28:00+00:00)
With the season essentially over for Gang Green barring a miracle, the staff took the handcuffs off Zach Wilson and unleashed an offense that's been
run-run-pass-punting their way to irrelevance all season long. In doing so last week, Hall was a weapon as a receiver, posting over 100 total yards for the
first time since Week 5. If the Jets continue their willingness to take chances downfield and let Wilson stop being a game manager, then Hall is in line for
another big day.

WR Deebo Samuel, 49ers @ Cardinals ($8,400). We'll just keep riding with Samuel down the stretch, as he isn't giving us any reason to reconsider. He has
back-to-back games with 30 fantasy points and has been a man on a mission since Thanksgiving. He shouldn't have any problems continuing that run against the
Cardinals on Sunday.
</item 80>

<item 81>
“We Sell More F-150s to NASCAR Fans” – Ford Insider Reveals Insane Mustang Stat Amid Big Change (Essentially Sports, 2023-11-06T23:01:17+00:00)
That’s been my whole life. That’s been my whole career. It’s just kind of something I do.” He described the intense battle with Ross Chastain, who blocked him
multiple times, making it a tough fight.

The final push came when Blaney made contact with Chastain. A move made out of frustration, as he admitted. But at this crucial moment, team owner Roger Penske,
often referred to as “The Captain,” stepped in. He reminded Blaney that he was not racing against Chastain but against the HMS duo. Penske’s calming influence
helped Blaney regain composure, ensuring that he didn’t take any unnecessary risks. The result? A championship for Ryan Blaney and another testament to the
guidance of “The Captain” in high-pressure situations.

ADVERTISEMENT Article continues below this ad

In NASCAR, it’s not just about speed and strategy; it’s also about maintaining composure under intense pressure.
</item 81>

<item 82>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
GO DEEPER Behind the AI magic that lets Amazon's Prime Vision show the NFL like never before

• Buffalo Bills: Are the defensive injuries too much to overcome?

What I think: The rest of the conference is the key variable. The Bills are diminished. They have imperfections. But Buffalo already defeated the Dolphins. The
other AFC teams aren’t exactly peaking. Kansas City and Cincinnati appear easier to defend this season than in the recent past. We’ll see how well the Dolphins
hold up. This can still be Buffalo’s season.

Exec comment: “Matt Milano and Tre’Davious White, those are huge injuries for that football team. The Bills are not as dominant as they felt going into it with
so many injuries on defense. But the division they are in, they will still be able to slug it out and get to where they need to go. Nobody has really emerged in
the AFC to unseat these teams at the top.”
</item 82>

<item 83>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
32. Carolina Panthers 1-8 (31)

The Panthers found better defense and special teams to keep them in the game in Chicago, but Bryce Young still can't unlock a limited passing game and the
offensive line and running game hasn't help nearly enough.
</item 83>

<item 84>
Yankees Gear Up for 2024 Season After Horrendous Year, But Could Letting This Star Go Spell Another Big Mistake? (Essentially Sports, 2023-10-10T22:30:19+00:00)
The performance from Gleyber has been crucial in a season filled with injuries on both fronts. But the Yankees are not favoring his services for the upcoming
season. There are two reasons behind the interest of the Miami Marlins and the push for lower spending. As mlbtraderumors.com suggests, Torres is expected to
earn $15.3 million next year; his trade could save the Yankees some money. The Marlins have been interested in him for a long time, and the Pinstripes can have
a good deal for the outfielder. Considering the advantages, this appears promising. However, is it a worthwhile endeavor? Should the New York Yankees Trade
Gleyber Torres Ahead of the New Season? Nothing has worked well for the New York Yankees this season. They had their worst finish in seven years, even after
having all the resources. The Yankees are seeking a shift in their strategy, which has exceeded their initial expectations.
</item 84>

<item 85>
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
</item 85>

<item 86>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 86>

<item 87>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• San Francisco 49ers: What do they have in Purdy?

What I think: Let’s start with what the 49ers do not have in Purdy. They do not have an AFC North-certified, weatherproof Ben Roethlisberger clone able to
overpower the elements, break away from defensive linemen and thrive in the conditions Purdy faced against Cleveland. What they do have is a good, young
quarterback well-suited to run Kyle Shanahan’s offense at a high level by making quick decisions when given time to find open receivers.

Exec comment: “The 49ers knew they needed a good quarterback to win the Super Bowl, and they knew they could help Purdy be that. Give him a great receiving
tight end, two good receivers and a great halfback who fits the system, and they’ll get someone open on every pass play. But then Deebo Samuel gets hurt,
Christian McCaffrey gets hurt, and now we don’t have as many guys open.
</item 87>

<item 88>
FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:17:00+00:00)
Everett hasn't posted 50 yards or more in a game this season, but this should be his best chance to do that plus have a good shot at a TD. We'll take the
discount for someone who should outperform his DFS salary.

FLEX RB Dameon Pierce, Texans @ Jets ($6,100). MetLife Stadium promises to be the site of yet another bad weather game and all that rain means we'll be seeing
plenty of the RBs on Sunday. Pierce appears to have regained the RB1 spot from Devin Singletary, evidenced by his 15 carries last week to just eight for his
teammate. Efficiency has been a problem for the sophomore back, but the Jets present another decent matchup for a RB to find their footing. With an ugly game on
tap, Pierce should have more than enough opportunities to deliver from a fantasy standpoint.

DEF New Orleans Saints vs. Panthers ($4,300).
</item 88>

<item 89>
NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights Deal (Essentially Sports, 2023-12-01T20:55:57+00:00)
Stern then pointed out how the media rights revenue plays a key role in sports as he used the MLS as an example.

He said, “You look at MLS. MLS takes a big deal with Apple, almost all of their games behind the Apple TV’s paywall. That was the money they could get and they
had to deliver for their industry. Their industry is expected to be a growing business and one of the most sporting ways to be a growing business in sports is
to have growing media rights revenue.”

While he explained that it was a similar case with NASCAR, he added one more angle because of which this deal possibly materialized. It is no secret that the
cable universe has been declining. As Stern discussed, every quarter, there is a drop in a couple of percentages of subscribers, which is very concerning.

This is one of the reasons why NASCAR did not want to gamble solely on cable users.
</item 89>

<item 90>
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
</item 90>

<item 91>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
He is regulated to pass from the pocket, but you are taking away the biggest weapon the offense had when you do that. Odell (Beckham) is not that guy anymore.
Who are you truly afraid of on that offense? Nobody but Lamar, and if you are not using Lamar in a certain capacity, he becomes just like the rest of these
quarterbacks around the league — hit or miss.”

• Jacksonville Jaguars: Are they ready for prime time?

What I think: Trevor Lawrence’s knee injury was the big concern coming out of Sunday. If he’s out, expectations shift dramatically. More than the other AFC
contenders, the Jaguars seem early in their life cycle, still finding their way. That makes them a little harder to trust in big spots but also leaves greater
space for growth. Let’s hope those test results on Lawrence’s knee come back clear.

Exec comment: “I like Jacksonville. Doug (Pederson) has a championship pedigree. This team is relatively young. They are scrappy.
</item 91>

<item 92>
NASCAR offseason news roundup: Short track testing in Phoenix, new spotters for stars, more moves for 2024 (CBSSports.com, 2023-12-07T20:15:30+00:00)
Blaney was one of six drivers to participate in the test, joining Chris Buescher, Erik Jones, Christopher Bell, Kyle Larson and Corey LaJoie.

The first day of testing centered around different splitters and diffusers as well as tires engineered to degrade more during green flag runs as NASCAR tries to
build upon marginal improvements made to the Next Gen car's short track package in 2023. According to Cole Cusumano, teams ran a simplified diffuser --
featuring just two fins in the middle at a 90-degree angle -- as well as an alternate splitter design.

After ascertaining a direction on aerodynamics and tires, NASCAR spent the second day of the test experimenting with existing mufflers to slightly soften the
otherwise deafening noise made by Cup Series motors. Gearbox and shifting changes were not included as part of this test.

Speaking to reporters, Blaney noted the biggest difference came in the alternate splitter, which significantly reduced downforce.

"It was big.
</item 92>

<item 93>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 93>

<item 94>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 94>

<item 95>
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
</item 95>

<item 96>
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
</item 96>

<item 97>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
Two more drivers to have featured in the 2023 NASCAR season and Autosport’s Top 50 Drivers are Supercars racers Shane van Gisbergen and Brodie Kostecki. While
the pair battled for the championship in the Australian racing circuit, they also raced at the Indianapolis road course race earlier this season. SVG had
earlier created history by winning in his NASCAR debut at the inaugural Chicago Street Race.

ADVERTISEMENT Article continues below this ad

These are some major exclusions that could have been featured in Autosport’s annual list. At the same time, motorsports is a vast field with elite competitions
taking place the world over. Some fan favorites are bound to be left out.
</item 97>

<item 98>
Major NASCAR Snubs as Autosport Releases the Top 50 Motorsport Drivers in 2023 (Essentially Sports, 2023-12-21T23:41:30+00:00)
The 43-year-old dominated the first round of the playoffs and was the driver to watch out for. He eventually won his first playoff race in Bristol and cruised
to the next round. However, Hamlin revealed recently that a shoulder injury prevented him from being at the best of his abilities as the playoffs progressed.
Having to depend much on his weaker arm, Hamlin stayed in contention for the championship four race until the final race in Martinsville.

He secured three wins, 14 top-fives, and 19 top-10 finishes in 2023. He also led a whopping 998 laps, which makes his omission from Autosport’s list seem
unjustified. He finished 5th in the final standings and was one of the most consistent and best performers of the 2023 NASCAR season. From the looks of it, a
championship trophy would have surely put him in the upper echelons of Autosport’s Top 50.

3.
</item 98>

<item 99>
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
</item 99>

<item 100>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 100>
=== END ITEM id=31d33f465472725f ===
