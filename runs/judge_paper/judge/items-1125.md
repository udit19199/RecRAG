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

=== ITEM id=2b21733b8b4f5daa ===
Gold sentences and facts:
Sentence 0: Lee was hampered by injury in 2023 but slashed .349/.421/.575 in 2022 with 36 doubles, 10 triples and a career-best 23 home runs.
  Fact 0.0: Lee was hampered by injury in 2023.
  Fact 0.1: Lee slashed .349/.421/.575 in 2022.
  Fact 0.2: Lee hit 36 doubles in 2022.
  Fact 0.3: Lee hit 10 triples in 2022.
  Fact 0.4: Lee hit a career-best 23 home runs in 2022.
Sentence 1: He picked up 16 wickets, conceding just 398 at an economy rate of 4.25 in 10 matches.
  Fact 1.0: Lee picked up 16 wickets in 10 matches.
  Fact 1.1: Lee conceded just 398 in those 10 matches.
  Fact 1.2: Lee's economy rate was 4.25 in those 10 matches.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Those two would be the biggest supporting bats in the lineup around Ohtani. However, both are free agents after the 2025 season, and given the added payroll of
signing Ohtani, it's possible Toronto would view keeping the other two phenoms as too pricey.
</item 2>

<item 3>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
We’ll find out in no time.

11.56 AM: Dubai’s Coca-Cola Arena

11.54 AM: Message Coming in for Sourav Ganguly

Will miss being at the auctions today but have no doubt that under the leadership of Kiran and @SGanguly99 we will build a very strong squad for @IPL – all the
best @DelhiCapitals – I know a lot of hard work has gone into today — Parth Jindal (@ParthJindal11) December 19, 2023

11.47 AM: If Aakash Chorpa is to be believed, Coetzee could get well over INR 10 cr.

11.45 AM: With only 77 slots to fill, who do you think will attract the biggest bid?

11.06 AM: Here are the released players from CSK, would they go for Rachin?
</item 3>

<item 4>
IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series (Zee Business,
2023-11-18T13:02:06+00:00)
For India, Ravindra Jadeja picked up three wickets, along with Kuldeep Yadav and Jasprit Bumrah picking up two each.

India was in a terrible position at the start of the second innings as both of the openers got out for a duck, and India lost three wickets for just two runs.

However, two brilliant and sensible innings by KL Rahul and Virat Kohli, where they scored 97 runs off 115 balls and 85 runs off 116 balls, took India to shore
as India won the match by six wickets.

Also Read: 2023 ICC World Cup: Virat Kohli vs Australia; know the ODI batting records and stats

Catch the latest stock market updates here. For all other news related to business, politics, tech, sports, and auto, visit Zeebiz.com.
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
</item 6>

<item 7>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
They’re already light on consistent hitting beyond Aaron Judge. On the surface, trading their second-best hitter seems counterintuitive.

The big question is whether the front office believes Oswald Peraza is capable of stepping into an everyday second base role in 2024. If they feel the
23-year-old is ready, trading Torres to clear spending room while bringing back outfield or pitching help makes sense.

11. Jonathan India, 2B, Reds

The Reds’ infield surplus has fueled trade speculation on India back to the deadline. Cincinnati didn’t seem inclined to make that move midseason, pointing to
India’s role as a clubhouse leader. While that’s certainly still a factor, he didn’t hit well down the stretch and finished the year with a middling
.244/.338/.407 batting line. He’s a well below-average keystone defender.

There’s an argument the Reds are better off relying on an up-the-middle tandem of Matt McLain and either Noelvi Marte or Elly De La Cruz.
</item 7>

<item 8>
Rollback Policy, a Boon or a Curse? The Controversial System That Might Redefine the Golfing World by the End of 2030 (Essentially Sports,
2023-12-03T23:45:00+00:00)
Secondly, a league that thrives on thrills and excitement will lose its USP if the ball suddenly doesn’t travel 300 yards on most occasions. As of this year, 28
of 48 LIV golfers average over 300 yards driving distance. Dustin Johnson and DeChambeau are both in the top five.

ADVERTISEMENT Article continues below this ad

But one crucial thing to remember here is that the golf ball rollback policy will render most balls totally ‘unplayable’. Moreover, the PGA Tour has
contradicted itself several times in the past few years. It won’t be surprising if Monahan & Co. eventually decides to agree with the policy change sometime
later. So, the question remains: how viable would it be for manufacturers to produce two types of balls, one for Majors and the other for LIV Golf? This just
throws the world of golf into uncharted territory.
</item 8>

<item 9>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Top 10 Bowlers of ODI World Cup 2023:

Rank Player Mat Wkts 1 Mohammed Shami (IND) 7 23 2 A Zampa (AUS) 11 23 3 D Madushanka (SL) 9 21 4 JJ Bumrah (IND) 11 20 5 G Coetzee (SA) 8 20 6 Shaheen Shah
Afridi (PAK) 9 18 7 M Jansen (SA) 9 17 8 RA Jadeja (IND) 11 16 9 Josh Hazlewood 11 16 10 MJ Santner (NZ) 10 16 11 BFW de Leede (NED) 9 16

*Still Bowling

10.
</item 9>

<item 10>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
The only difference was that Janes did not name the player.

The reporting of Arcia's comments proved to be divisive across the baseball community. Among the most prominent to call out Mintz was Rizzo. The "High Heat"
co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us" when "some jackoff comes in at the end
of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the players who were upset by the reporting of
Arcia's comments, before she called Mintz out by name.

MORE: Every time Nick Castellanos has interrupted a somber moment with a home run

"First of all, this guy, Jake Mintz, that's not even a reporter," Rizzo said. "That's taking away from true reporters and true journalists.
</item 10>

<item 11>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 11>

<item 12>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
CSK: Akash Singh, Ambati Rayudu, Ben Stokes, Bhagath Varma, Dwaine Pretorius, Kyle Jamieson, Sisanda Magala, Subhranshu Senapati.

10.40 AM: There will be a lot of teams that will be eyeing the English players, but then they won’t be available for the playoffs. This could go against them in
bagging a big contract.

10.10 AM: Kaushal Tambe (Maharashtra) and Sahil Dhiwan (Rajasthan) have been added to the list of players to go under the hammer. South African Nandre Burger
moved to capped players list.

9.43 AM: Here are some things to know. Out of the 77 players that can be bought, only 30 are reserved for overseas players. Also, the opening of the trade
window could see some surprising moves.
</item 12>

<item 13>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
4. Jasprit Bumrah, India

Bumrah has taken 20 wickets under his belt, conceding 373 runs. He also has the best economy rate in the whole tournament, with 4.06.

3. Dilshan Madushanka, Sri Lanka

Despite such a dreadful performance by the Sri Lankan side in this World Cup, Madhushanka managed to be in the top 2 on the highest wicket-takers list. He
picked up 21 wickets, conceding 525 runs.

2. Adam Zampa, Australia

Adam Zampa is having a World Cup to remember, as he is the highest wicket-taker for the Aussies. He picked up 23 wickets, in which there were three four-wicket
hauls, conceding 515 runs.

1.
</item 13>

<item 14>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
(Photo: Michael Owens / Getty Images)

Ohtani is committed to being a starting pitcher. If he cannot stay healthy enough for that role, he could aid his new team as a reliever. He closed the final
game of the World Baseball Classic, securing the crown for Japan by striking out Trout. His four-seam fastball averaged nearly 97 mph in 2023; the velocity of
the pitch figures to improve in short bursts.

Advertisement

Even if Ohtani never pitches again, his value as a hitter is immense. In 1920, his first season exclusively as a hitter, Ruth led the American League in homers,
on-base percentage and slugging percentage. In 2023, while still making 23 starts with a 3.14 ERA, Ohtani led the American League in the same three categories
as Ruth. He swatted 44 homers with a career-best 1.066 OPS. He did this while making starts and dealing with a torn ligament in his elbow.
</item 14>

<item 15>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
9:05 PM IST: It’s a wrap. A total of Rs 230.45 Cr have been spent for 72 players combined.

9:00 PM IST: Sourabh Chauhan to RCB for Rs 20 L

9:00 PM IST: South African Nandre Burger goes to RR for Rs 50 L

8:59 PM IST: Avinash Rao Aravely to CSK for 20 L

8:54 PM IST: Abid Mushtaq to RR for 20 L.

8:54 PM IST: Swastik Chikara goes to DC and with that Rishabh Pant & Co complete the squad.

8:51 PM IST: English bowler Gus Atkinson goes to KKR for Rs 1 Cr.

8:50 PM IST: Mohammad Nabi has a new home at Mumbai Indians for Rs 1.5 Cr.
</item 15>

<item 16>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.
</item 16>

<item 17>
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
</item 17>

<item 18>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Revisiting the NBA caricature back in time, the New York Knicks became the first all-Black team in 1979. But their fans made their disapproval very apparent.
“It wasn’t so much New York attitudes, but fans around the league let you know how they felt about you as an opponent and as a Black man as well,” Knicks star
Earl Monroe told Newsday.

USA Today via Reuters May 15, 2021; San Antonio, Texas, USA; Phoenix Suns forward Torrey Craig (12) and San Antonio Spurs forward Keldon Johnson (3) fight for
loose ball as Suns guards Devin Booker (1) and Chris Paul (3) look on in the third quarterat AT&T Center. Mandatory Credit: Scott Wachter-USA TODAY Sports

We could list out all the scandals, fights, and controversies here, but you get the idea.
</item 18>

<item 19>
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
</item 19>

<item 20>
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
</item 20>

<item 21>
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
</item 21>

<item 22>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
Since he was a teenager, Ohtani has ignored suggestions that he focus on only one pursuit.

Ohtani demonstrated his potential as the American League Rookie of the Year in 2018, but his two-way hopes were delayed after his first elbow surgery. It was
not until 2021 that the full flower of his ability bloomed. He has won the American League MVP in two of the past three seasons; in the intervening season, he
led all American League pitchers in strikeout rate while hitting 34 home runs with an .875 OPS. To create a comparison for him involves inventions that sound
freakish. “It’s like if Judge went out and was a 20-game winner as well,” former teammate Kole Calhoun said.

Ohtani will not pitch in 2024 after undergoing Tommy John surgery, but his agent stressed he’s committed to returning to the mound. (Photo: Michael Owens /
Getty Images)

Ohtani is committed to being a starting pitcher.
</item 22>

<item 23>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 23>

<item 24>
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
</item 24>

<item 25>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
It had been reported Monday night when the Braves beat the Phillies in Game 2 of the NLDS, ending on a critical base-running play involving Bryce Harper, that
Atlanta shortstop Orlando Arcia taunted about the Phillies' star in the clubhouse after the game.

That report wound up generating plenty of controversy.

Fox Sports' Jake Mintz, one of the two brothers that runs the popular Cespedes Family BBQ account, reported that Arcia had been celebrating after Atlanta's 5-4
win by saying "Atta boy Harper" in the Braves' clubhouse, jeering at the fact that Harper was doubled off on the basepaths after an outstanding catch by Michael
Harris in center field and an impressive relay to get Harper at first to end the game.

The comments wound up coming back to bite the Braves two days later, with Harper homering twice — staring down Arcia after each — as he powered the Phillies to
the 10-2 Game 3 win.
</item 25>

<item 26>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
The deal includes an additional $800,000 in incentives.

Tellez hit 35 home runs as a centerpiece of the Brewers' lineup in 2022, but he took a significant step back in 2023 after starting the season on a roll.
Between first base and the DH role, however, the Pirates figure to give Tellez near-everyday playing time in hopes his power bat finds its form.

Giants, Jung-Hoo Lee agree to six-year deal

Date: Tuesday, Dec. 12

Source: Jon Heyman, New York Post

The Giants have struggled to land big-name free agents over the last two offseasons, but they reeled in a pretty significant fish on Tuesday by inking Korean
outfielder Jung-Hoo Lee to a six-year, $113M contract. Lee can opt out after 2027.

Lee is just 25 and was an on-base machine in Korea, batting .340 in his KBO career.
</item 26>

<item 27>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Here is the list of the top 10 batsmen and bowlers in this edition of the ODI World Cups—

Top 10 Batsmen of ODI World Cup 2023:

Rank Player Mat Runs 1 V Kohli (IND) 11 765 2 RG Sharma (IND) 11 597 3 Q de Kock (SA) 10 594 4 R Ravindra (NZ) 10 578 5 DJ Mitchell (NZ) 10 552 6 DA Warner
(AUS) 11 535 7 SS Iyer (IND) 11 530 8 KL Rahul (IND) 11 452 9 HE van der Dussen (SA) 10 448 10 MR Marsh (AUS) 10 441

*Still Batting

10.
</item 27>

<item 28>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.
</item 28>

<item 29>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
Kim is slated for an $8M salary and will be due a $2M buyout on a 2025 mutual option at season’s end. He’ll be a free agent next winter.

Grisham has the extra year of control but is the worse player. He draws plenty of walks but has hit below the Mendoza line in two straight seasons. He’s coming
off a .198/.315/.352 line over 555 plate appearances. The appeal is on the other side of the ball.

Grisham is an above-average defensive center fielder, keeping him as a low-end regular even if he’s best suited for the bottom of a lineup. San Diego could kick
Fernando Tatis Jr. over to center field if they dealt Grisham, freeing right field for a bigger offensive threat than Grisham provides.

It’s a similar story with Kim. Dealing him would free Jake Cronenworth to move back to second base, opening first base for a better hitter. It’d be tough to
replace Kim’s overall production, though.
</item 29>

<item 30>
Rollback Policy, a Boon or a Curse? The Controversial System That Might Redefine the Golfing World by the End of 2030 (Essentially Sports,
2023-12-03T23:45:00+00:00)
This just throws the world of golf into uncharted territory.

Watch This Story: Is Tiger Woods Making Use Of Any Medical Aid After His Subtalar Fusion?
</item 30>

<item 31>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
Will the teams target him?

12.10 PM: As per a BCCI directive, all the IPL venues can’t hold any domestic game in February, and need to hand over the stadium to teams in first week of
March.

12.07 PM: Meanwhile, the IPL 2024 window starts on March 22. An early start to the tournament might force Ranji Trophy venues to shift.

12.04 PM: Harshal was released by RCB, but in the VHT, he got 18 wickets in 10 matches and scored 183 runs too. So, he would be considered for an all-rounder
role.

12.01 PM: Will the Vijay Hazare Trophy performances have an impact on the bidding? Will Harshal Patel get a bid? Will he be trusted with bowling in the death
overs again? We’ll find out in no time.
</item 31>

<item 32>
MVPs for all 16 AFC teams at the midpoint (Yardbarker, 2023-11-09T21:51:02+00:00)
Stroud: Stroud is having one of the best statistical rookie seasons of all time. He ranks in the top 10 in passing yards (2,270), touchdown passes (14) and
passer rating (102.9), and has only thrown one interception. The Texans (4-4) aren’t close to a .500 team without him.

INDIANAPOLIS COLTS | RB Zack Moss: With Jonathan Taylor missing the first four weeks of the season, Moss stepped up in a big way. After not playing in Week 1,
he rushed for 445 yards and scored four touchdowns over his next four games to help Indy win three of four and stay in the thick of the AFC South. Moss is
second in the NFL in rushing, 37 yards behind San Francisco's Christian McCaffrey.
</item 32>

<item 33>
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
</item 33>

<item 34>
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
</item 34>

<item 35>
Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek (Yardbarker, 2023-12-24T23:34:39+00:00)
McCaffrey, who seems poised to win the Offensive Player of the Year Award, leads the NFL in rushing (1,292 yards) and total yards per game (128.6) this season
and is tied for first in touchdowns (20).

Ravens TE Isaiah Likely thriving in the starting role: The Ravens offense hasn't lost a step since losing one of the team's top weapons in tight end Mark
Andrews (ankle) in Week 11, largely due to Likely's emergence.

The second-year tight end has averaged 64.3 yards over his last three games, which is particularly noteworthy since Andrews averaged 54.4 yards per game before
sustaining his injury.

Should we expect a defensive battle?: While both teams boast offenses that rank in the top five in yardage and scoring, this may not turn into a shootout.
</item 35>

<item 36>
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
</item 36>

<item 37>
Rollback Policy, a Boon or a Curse? The Controversial System That Might Redefine the Golfing World by the End of 2030 (Essentially Sports,
2023-12-03T23:45:00+00:00)
Surprisingly, the fifteen-time major winner feels a course correction is necessary here. “I think this should have happened a long time ago.”

In fact, he said something that might not sit well with most, “I’ve been of the position if you play in a pro event or you have a P next to your name, you
should be playing a pro ball. If you have an A next to your name and you’re playing an amateur event, you should use an amateur ball.”

ADVERTISEMENT Article continues below this ad

How will the golf ball rollback policy affect LIV golfers?

The R&A and USGA have not made it binding on any governing bodies to implement the rules. For the breakaway league, this is the easiest path. One reason is that
power hitters such as Bryson DeChambeau will be affected in more ways than one otherwise.
</item 37>

<item 38>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 38>

<item 39>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
After the game, Mintz wrote a well-reported story on how the Braves pulled off the improbable comeback, coming back from down four runs at one point, to win the
game, and deep in the article, wrote the following:

Kevin Pillar's young son, sporting a custom "Money Mike" headband, waddled over to his favorite center fielder for a fist bump. Third-string catcher Chadwick
Tromp strutted by Harris' locker offering a congratulatory "Mike is him!" All while Braves shortstop Orlando Arcia cackled emphatically about Harper's
misfortune, bellowing "ha-ha, atta-boy, Harper!" repeatedly as reporters circled the room.

As Mintz pointed out, he was not the only one to allude to the celebration from Arcia. The Washington Post's Chelsea Janes also wrote about it, saying that one
of Harris' teammates was yelling, "Atta boy Harper."
</item 39>

<item 40>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
When the Phillies beat the Braves in Game 4 to eliminate Atlanta and advance to the NLCS, players in the Philadelphia clubhouse celebrated by repeating, "Atta
boy Harper."

While the comments became a spark for the Phillies, it also became a subject of controversy. Players, including the Blue Jays' Kevin Gausman, decried the
situation as a mistake by Mintz to report the comments, which were said out loud in the locker room rather than said in an interview setting. Others defended
the situation, saying that anything in the clubhouse is open to be reported on.

MORE: Why Arcia's comments motivated Harper after NLDS Game 2

But perhaps the biggest backlash came from MLB Network's Alanna Rizzo, who blasted Mintz on "High Heat" for reporting on the incident. Following a BBWAA
statement, Rizzo both publicly and privately apologized to Mintz for her reaction.

Here's what you need to know about the controversy.
</item 40>

<item 41>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
1. Mohammad Shami, India

Shami announced his return to the World Cup with a bang, picking a fifer in the very first match he played for India. He has picked up 24 wickets so far in just
six matches, conceding just 257 runs. He has picked up three five-wicket hauls including a seven-wicket haul against New Zealand in the semi-final on Wednesday,
November 15 at the Wankhede Stadium in Mumbai.

Catch the latest stock market updates here. For all other news related to business, politics, tech, sports, and auto, visit Zeebiz.com.
</item 41>

<item 42>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Any other metric could be wrong and misleading.

For example, price by revenue or price by EBITDA is incorrect. Why? How do you differentiate between two companies A and B — both having same market cap of ₹100
and similar in all means except for debt levels — same revenue (₹100) and same gross margin and same EBITDA margins of 20 per cent and thus same EBITDA of ₹20.
But A has zero debt, while B has debt of ₹10. Since revenue or sales and EBITDA are same, if you do a market cap/revenue or Price /Sales, or marketcap/EBITDA,
both companies will have the same valuation.

However, let’s assume depreciation is ₹10, interest cost is 10 per cent and taxes are zero. This means the net profit of A is ₹10, while that of B is ₹9.
</item 42>

<item 43>
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
</item 43>

<item 44>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The October 1 matchup between the Chiefs and the Jets drew 27 million viewers. And became the most-watched Sunday TV show since the last Super Bowl! And the
last Super Bowl had an actual concert by Rihanna.

Revealed: The Dark Side Of Pat McAfee & Why Rob Gronkowski Is In The Firing Line

Kaepernick, who? Deflategate, what? In the media frenzy surrounding the two lovebirds, other issues get thrown out of the window. How can fans forget that the
NFL is also organizing games outside the American borders with a view to garnering international appeal? The NFL has scheduled 5 games abroad this season. It is
also addressing the concussion controversies. It recently added both new concussion rules and a new $100 million initiative called “Play Smart, Play Safe,” to
make the sport safer for players. And all the effort is showing.
</item 44>

<item 45>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
He is regulated to pass from the pocket, but you are taking away the biggest weapon the offense had when you do that. Odell (Beckham) is not that guy anymore.
Who are you truly afraid of on that offense? Nobody but Lamar, and if you are not using Lamar in a certain capacity, he becomes just like the rest of these
quarterbacks around the league — hit or miss.”

• Jacksonville Jaguars: Are they ready for prime time?

What I think: Trevor Lawrence’s knee injury was the big concern coming out of Sunday. If he’s out, expectations shift dramatically. More than the other AFC
contenders, the Jaguars seem early in their life cycle, still finding their way. That makes them a little harder to trust in big spots but also leaves greater
space for growth. Let’s hope those test results on Lawrence’s knee come back clear.

Exec comment: “I like Jacksonville. Doug (Pederson) has a championship pedigree. This team is relatively young. They are scrappy.
</item 45>

<item 46>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“The Taylor Swift and Travis Kelce news has been a pop cultural moment we’ve leaned into in real time, as it’s an intersection of sport and entertainment, and
we’ve seen an incredible amount of positivity around the sport.”

“The vast majority of our content has remained focused on the game, our players and variety of other initiatives, including our Toy Story Funday Football
alt-cast, the international games and more,” they continued.
</item 46>

<item 47>
India to be one of world's fastest growing economies over next decade (Zee Business, 2023-10-14T12:17:33+00:00)
Iron & steel have also witnessed accelerated long-term loan disbursements in the past six months. Infrastructure is witnessing an increase in credit flow with
67 per cent of the respondents indicating an increase in long-term loans as against 57 per cent in the previous round.

The survey suggests that the outlook on expectation on growth of non-food industry credit over next six months is optimistic with 42 per cent of the
participating banks expecting non-food industry credit growth to be above 12 per cent (as compared to 36 per cent in the previous round).

Rajan Jain, Head, Credit Research, SBI Caps, said in a note that whether the yield curve dynamics mean the US is moving to a structurally higher-rate economy
vs. post GFC normal remains a moot question, with important implications for investment demand and system risk.

These movements have had a cascading effect on global economies, which have seen routs in their respective bond markets, besides a fresh round of currency
depreciation.
</item 47>

<item 48>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
And he remains unsold.

3:55 PM IST: Spinners set is up next.

Akeal Hosein: Unsold

Mujeeb Rahman: Unsold

Adil Rashid: Unsold

Mohammad Waqar Salamkheil: Unsold

Tabraiz Shamsi: Unsold

Ish Sodhi: Unsold

3:52 PM IST: Dilshan Madushanka is up next and it’s a bidding war between LSG and MI.

3:48 PM IST: Jaydev Unadkat is up next and DC & SRH are in a bidding war. SRH win the war at 1.6 Cr.
</item 48>

<item 49>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
She said that reporters abstaining from writing on his comments would be akin to "protecting players from themselves."

She also went on to say that what Arcia said wasn't a slur or hateful, and added the only reason anyone was concerned about it was because "they know it would
motivate Harper to beat [the Braves]. Which implies the Braves do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they
lost the other day."

"I didn't say something sooner because I thought this was overblown," Janes continued. "And frankly, I thought it wouldn't help. But I am genuinely disgusted by
the way some media members are acting like a reporter was in the wrong for this. He wasn't.

"And if the Braves let this be the story of their series, that's on them. Win and it goes away. Period."
</item 49>

<item 50>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 50>

<item 51>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Shohei Ohtani and Juan Soto have already found new homes, but much of the MLB offseason has yet to play out.

Japanese pitcher Yoshinobu Yamamoto is the focus of many teams as he prepares to sign a massive deal, while Blake Snell, Cody Bellinger and Jordan Montgomery
remain on the market. The trade market is also expected to be active once more dominoes fall, with Dylan Cease, Corbin Burnes and Tyler Glasnow among starters
who could be moved this winter.

The Winter Meetings are over, but because of the timeline of major decisions by Ohtani and Yamamoto, there could be a flurry of activity leading up to the end
of the year.

MORE: Why Shohei Ohtani signed with Dodgers over Blue Jays, other suitors

Sporting News is tracking the rumors, signings and trades as MLB free agency continues. Follow along below for the latest news.
</item 51>

<item 52>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks want even betting so much that they’ll move a line or change the odds to draw more bets to the other side to get it that way.

That’s when you’ll see a difference between the lines and odds. Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like
an injury or roster move. It’ll be difficult for you to find value in that case.

But when a line moves or odds change because of betting patterns, only the sportsbook that saw the betting makes the change. That’s your opportunity right
there.

Line shopping is a tool to help you spot changes due to betting patterns. Then, you can simply take advantage of movement in a line or changes in the odds.

Choose a side, shop around at online sportsbooks, compare the lines and odds, and then pounce when you find one that has something different and more favorable
to your viewpoint.

See more: What is a parlay bet?
</item 52>

<item 53>
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
</item 53>

<item 54>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
The hardcore football fans are pretty much done with the NFL harping about Travis Kelce and Taylor Swift. Reality TV star Rachel Lindsay, a longtime NFL fan,
recently slammed the “obnoxious” coverage, highlighting the “exhausting” uptick in Swifties obsessing over football. But that’s just one example of how fans are
finding this fixation a little too disrespectful towards the sport.

America’s Favorite Video Today

From the league’s humble beginnings in 1920 to the $18 billion juggernaut that it is today, the ride has been rocky with more than just a few small bumps along
the road. But with the ambitious target that NFL Commissioner, Roger Goodell, has set for 2027, is there a new strategy in the 104-year-old league’s master plan
to dominate the world sporting scene?

Why the ‘Traylor’ saga will be the key to Roger Goodell’s $27 billion target?
</item 54>

<item 55>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
That should be their thought process. They got to be smart”

12:50 PM IST: RCB will focus on bowlers: AB de Villiers

“RCB will focus on the bowlers because it is difficult to defend at the Chinnaswamy Stadium. I am very excited with the Cameron Green but I am sure the major
focus will be on the bowlers.”

12:45 PM IST: LSG ‘think tank’ ready for IPL 2024 Auction!
</item 55>

<item 56>
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
</item 56>

<item 57>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
The right-handed hitter owns a .241/.311/.471 line in just over 850 MLB plate appearances over the past two seasons. When he’s hot, he can carry a lineup.

He also strikes out more than 30% of the time and doesn’t have a defensive fit. Morel has played all three outfield spots and each of second base, third base
and shortstop in the majors. He hasn’t rated well anywhere.

The Cubs used him mostly at designated hitter this year and have suggested they’ll get him first base run in 2024. That’s not a great fit for a plus runner with
top-of-the-scale arm strength, yet Morel hasn’t shown himself capable of handling more important positions on a regular basis.

How teams project Morel defensively could impact both whether he’s traded and what kind of return the Cubs could receive. As a first baseman or DH, he’s a
fringe regular with a statistical profile that’d paint him as a lumbering slugger as opposed to a dynamic athlete.
</item 57>

<item 58>
World Series 2023 live score: Rangers vs. Diamondbacks updates, highlights, results from Game 4 (Sporting News, 2023-10-31T22:00:04+00:00)
— Jung gets moved over to third on a chopper by Heim, who is out at first. Leody Taveras has a two-out RBI opportunity.

8:31 p.m. — Miguel Castro is in to face Heim with one out and Jung at second.

8:29 p.m. — Mantiply quickly strikes out Lowe for the first out of the inning, and that will be the end of his day.

8:27 p.m. — Keeping Mantiply in to face Josh Jung was not successful. Jung belts a double into the gap and gives Texas a leadoff runner in scoring position.
Mantiply will stay in to face Nathaniel Lowe.

8:25 p.m. — The Diamondbacks are sticking with Mantiply to start the second inning.

8:23 p.m. — Heim nabs Marte on the stolen base attempt! That'll end the inning.
</item 58>

<item 59>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
However, you’d have a winner at the fourth. If you didn’t shop around, you’d never have found that half-point victory and be stuck with a push instead of a
profit.

What is the Vegas line?

Las Vegas sportsbooks set the standard for lines and odds across the country. If a sportsbook is copying a line, it’s usually copying the Vegas line. Most
online betting sites don’t stray too far from the Vegas line, using it as a foundation for setting all lines in every major sport.

If those lines do shift, it’s usually because of increased betting on one side of a bet. Shop lines locally, and you can take advantage when sportsbooks stray
from the Vegas line.

Why do the lines move?

It’s worth reiterating that sportsbooks want an equal amount of money bet on each side of every bet. When that happens, they earn the small profit built into
the odds without concern about who wins or loses.
</item 59>

<item 60>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
All eyes since the start of the offseason have been on Shohei Ohtani. And for the most part, there hasn't been much to see.

The two-way phenom is the most sought-after free agent in MLB history, with his combination of talent and marketability making him a franchise-changing player.

And the bidding for Ohtani could be winding down soon. Multiple reports have indicated Ohtani could sign at some point during the winter meetings, which are
scheduled to take place from Dec. 3 through Dec. 6.

Many of the top teams are expected to be in the mix for Ohtani. The Dodgers and Cubs are among the top teams expected to be aiming to sign the two-time AL MVP,
and plenty of others should continue to compete through the end to land him.

Here's the latest on Ohtani's free agency pursuits.
</item 60>

<item 61>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
And it’s totally fine to be an investor relying on relative valuation to make your savings work for you, provided you approach it the right way and ensure
adequate risk management. In relative valuation, the broader you assess the valuation, ie assess valuation multiple against more parameters, the better the
perspective you get.

In simple words, one way to understand the difference between absolute valuation (explained in last week’s edition) and relative valuation is this – absolute
valuation is made up of an interesting story that is backed by credible and validated numbers, while relative valuation is made up of an interesting number
backed by a credible and validated story.

What are those numbers in relative valuation? Every number in relative valuation is the result of a number A divided by a number B. For example, PE is price of
the stock divided by its earnings. These are all numbers that are used to compare against similar metrics of other stocks and assess the merits of the stock.

Before we proceed on using relative valuation to pick stocks, it is important to understand that you can do relative valuation using equity multiples or
firm/enterprise multiples.
</item 61>

<item 62>
Rollback Policy, a Boon or a Curse? The Controversial System That Might Redefine the Golfing World by the End of 2030 (Essentially Sports,
2023-12-03T23:45:00+00:00)
“They definitely, in my mind, have done a lot of things that aren’t for the betterment of the game, although they claim it.”

The 2022 PGA Championship winner drew an interesting comparison. In the NBA, players now jump much higher than they did in the 1990s. But the hoop height has
stayed the same at 10 feet above the playing surface. The two-time Major winner credits the player’s grind for this.

What Is the Model Local Rule? As the Golf Ball Rollback Debate Takes Off Once Again, All You Need To Know About the USGA?s and R&A?s Brainchild

Echoing similar sentiments, Bryson DeChambeau, who joined LIV Golf in 2022, said, “It’s a great handicap for us guys that have worked really hard to learn how
to hit it farther.” The 2020 U.S.
</item 62>

<item 63>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
While Indian interest rates were always higher, investors in equities benefited from the global investors’ hunger for yields. That advantage is diminishing
today. This is not a recommendation on the stocks, but pointers on factors to consider to make your relative valuation process more foolproof.

With discipline and experience, the process will keep getting better.
</item 63>

<item 64>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Mitch Santner, New Zealand

New Zealand's bowling has mostly depended on Santner, as he has been effective for the Kiwis on the turning tracks of India, as he picked up 16 wickets and
conceded 449 runs in 10 matches.

9. Josh Hazlewood, Australia

The Aussie pacer has been phenomenal with the new ball. He has created movements with the ball which the batters struggle to play. He picked up 16 wickets in 11
matches.

8. Ravindra Jadeja, India

Jadeja has been bowling some beautiful spells in this tournament for India, especially in the middle overs. He picked up 16 wickets, conceding just 398 at an
economy rate of 4.25 in 10 matches.

7.
</item 64>

<item 65>
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
</item 65>

<item 66>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
7:16 PM IST: Finn Allen is up next in the accelarated round. But he goes unsold.

7:12 PM IST: Remaining purse:

CSK: Rs 3.2 Cr

DC: Rs 16.85 Cr

GT: Rs 21.45 Cr

KKR: Rs 6.55 Cr

LSG: Rs 4.15 Cr

MI: Rs 7.95 Cr

PBKS: Rs 13.15 Cr

RR: Rs 90 L

RCB: Rs 6.75 Cr

SRH: Rs 3.4 Cr

7:12 PM IST: Mitchell Starc on playing under Shreyas Iyer.

“I am really excited to play under Shreyas Iyer captaincy. He understands his thought process. I’m really happy to finally back at the best league in the
world.”

7:10 PM IST: The last accelarated round is set to begin soon.
</item 66>

<item 67>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Portfolio podcast: The art of valuing stocks Portfolio podcast: The art of valuing stocks

Let there be no doubt, valuation experts like Aswath Damodaran and many successful hedge fund managers believe that absolute valuation (which we discussed in
the Big Story in bl.portfolio edition dated October 8) is the better way to arrive at the value of a stock.

However, due to a combination of complexities involved in building appropriate models, insufficient information, time constraints and also the fact that there
are hundreds of stocks you want to consider, it is just not possible to do a DCF ( discounted cash flow model) for all your stock investments, and hence
relative valuation as an approach to picking stocks too is well accepted and widely prevalent.

When you decide to buy or sell a stock based on its PE or P/B or EV/EBITDA or EV/Revenue or even EV/subscriber ratios, you are engaging in the art of relative
valuation.
</item 67>

<item 68>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
The Hot Stove dominoes have started to fall, but everyone who loves baseball — fans, players, coaches, front-office types and everyone else — is waiting for
free agent Shohei Ohtani’s decision. Where will the global superstar sign his next contract?

He’s played his entire six-year MLB career with the Angels, but his teams haven’t even sniffed the postseason and Ohtani’s made his desire to win crystal clear,
so he won’t be back. Will he chase every dollar possible, going solely by the highest offer? That seems unlikely, too. Remember, Ohtani wants to win.

Lots of teams can meet Ohtani in that middle ground, offering both lots of money — he will absolutely sign the largest contract in baseball history — and lots
of chances to compete for a World Series title (or two or three). It’s a long list.

MORE: Ranking 15 possible destinations for Shoehei Ohtani

So, the baseball world waits for news.
</item 68>

<item 69>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Mitch Marsh, Australia

In the absence of Travis Head, Marsh has played the role of opener with much responsibility for the Aussies, providing good starts as he accumulated 441 runs in
10 matches.

9. Rassie van der Dussen, South Africa

Rassie has been a support system for the Proteas in the middle order. Though sometimes inconsistent, he has accumulated 448 runs in 10 innings. He has two
centuries and two fifties under his belt in his campaign, with a high score of 133 runs.

8. KL Rahul, India

The Indian number five has been on great form scoring 452 runs giving great support to the Indian middle order. He has one century and two fifties to his name
in the World Cup.

7.
</item 69>

<item 70>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
He still has a lot left to give to the diamond for approximately 6 more years. If the Dodgers have the chance to draft them, should they go with Ohtani, Trout,
both, or neither? Let us know your opinions in the comment box.

Watch This Story: Shohei Ohtani’s Two-Way Dream: Before Injury, He Always Wanted Both!
</item 70>

<item 71>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
However, the NFL does not just want to stop at $18 billion. It wants to become a $27 billion league in 4 years. But it must address a few skeletons in its
closet before reaching that goal.

If you were around in the 1970s (that would make you at least 50 years old today), you would know that the NBA did not have the mass appeal that it enjoys now.
With average game attendance hovering around 8,000 a game, it was one of the darkest periods of the basketball league. Drug abuse, brawls on the hardwood court
that sometimes spilled over onto coaches and players – it wasn’t a pretty sight. As reported in the Washington Post, even though there are no reliable figures
on cocaine use by players, “estimates by people in the game range from 40 percent to 75 percent.”
</item 71>

<item 72>
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
</item 72>

<item 73>
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
</item 73>

<item 74>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
It’s because of the fans that IPL is now the second most popular league in the world.

1:00 PM IST: We are just moments away from the start of the 17th IPL auction.

Decked up and HOW 😍



Setups and Arena looking stellar 🤌



Slowly building up to the #IPLAuction here in Dubai ⏳ pic.twitter.com/J0rppK0Mjq — IndianPremierLeague (@IPL) December 19, 2023

12:53 PM IST: KL is comfortable with the squad he has: LSG coach Justin Langer

“Everyone around the cricket world is watching today. Everyone is excited to see who the big earners are. KL Rahul is a calm captain and he is comfortable with
the squad he has.”

12:51 PM IST: LSG got to be smart: Zaheer Khan

“LSG will be looking at the smart buys. That should be their thought process.
</item 74>

<item 75>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was impossible to contain during this year’s National Championships booting 14 majors in four matches for Victoria Metro to top the goal kicking. He also
averaged 12.2 disposals and was rewarded with All Australian selection. He further expanded his game by spending more in the midfield and across half back for
the Eastern Ranges and averaged 20 disposals while kicking 11 goals in nine matches in the Coates Talent League.

Weapon: Speed, agility and leap

AFL player comparison: Shai Bolton (Richmond)

Advertisement

This is just what Nick Watson does ????
</item 75>

<item 76>
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
</item 76>

<item 77>
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
</item 77>

<item 78>
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
</item 78>

<item 79>
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
</item 79>

<item 80>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 80>

<item 81>
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
</item 81>

<item 82>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
7. Marco Jansen, South Africa

Standing at a staggering 6 feet 7 inches tall, Jansen has made full use of his height, and it was evident in his bowling too. The 23-year-old picked up 17
wickets in nine matches for the Proteas, conceding 450 runs.

6. Shaheen Shah Afridi, Pakistan

Afridi is the best bowler for Pakistan in this tournament, with 18 wickets and conceding 481 runs in nine matches.

5. Gerald Coetzee, South Africa

The 23-year-old Coetzee has picked up 20 wickets for the Proteas in eight matches, conceding 396 runs.

4.
</item 82>

<item 83>
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
</item 83>

<item 84>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"Thanks to all the reporters, journos, baseball twitter weirdos, acquaintances and friends who had my back over the last 48 hours," Mintz wrote. "And also, of
course, the BBWAA. Community feels good, humbled to be a part of this one. Now, let's enjoy some ball."
</item 84>

<item 85>
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
</item 85>

<item 86>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Randy Mueller raised the thought last week, recalling examples from his career as a GM when offering a pick one round higher than a prospect’s actual draft
round proved alluring. Detroit used a third-round choice for Hooker, who is recovering from ACL surgery. Hooker would have been valued higher if healthy. Could
that be true again at some point in the future?

• News that Indianapolis Colts rookie quarterback Anthony Richardson might undergo season-ending shoulder surgery could carry a silver lining. Richardson got to
play enough for the Colts to evaluate where he stands, but if he shuts it down for the season, neither he nor the team would have to suffer through a full
season of growing pains. Gardner Minshew’s three-pick game against Jacksonville on Sunday works against the idea Indy is better positioned to compete with the
journeyman in the lineup, but most coaches and execs I’ve spoken with believe that to be the case in the short term.
</item 86>

<item 87>
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
</item 87>

<item 88>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
The Los Angeles Angels have gone through a tough year. After a heartbreaking 2023 season, they now have to cope with the impending exit of their star two-way
star, Shohei Ohtani. The team is also open to trading their nine-time Silver Slugger hitter Mike Trout, should he request it. In a huge miss, a team that
couldn’t do well even with two of the league’s biggest talents on their roster may lose said stars at the same time. Will they choose to stay in LA?

America’s Favorite Video Today

The Halos’ cross-town rivals, the Los Angeles Dodgers, have been eyeing the dynamic Troutani duo for a long time. However, despite fans’ expectations, the
Dodgers’ Ohtani-Trout conquest might not take place—and for good reasons.

Do the Dodgers have the budget to afford Ohtani and Trout on the same roster?
</item 88>

<item 89>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"That's taking away from true reporters and true journalists. Where were you on the random Tuesday in April in Cincinnati when this team was playing in
Cincinnati? Where were you on a rain delay, and you have to sit in the press box for all of this time? It's ridiculous."

"And then some jackoff comes in at the end of the season that gets a credential, God only knows why. And the clubhouse is a sacred space." Alanna Rizzo went off
on Jake Mintz's clubhouse reporting on MLBN's "High Heat" Thursday. pic.twitter.com/6dooqeLPnh — Awful Announcing (@awfulannouncing) October 12, 2023

Rizzo's comments immediately drew heat from the baseball journalism world, including from Janes, who posted a thread defending Mintz. Janes reported that
cameras and recorders were recording, and that Arcia yelled the comments while they were recording and with more than a dozen reporters around.
</item 89>

<item 90>
Why is Brock Purdy called Mr. Irrelevant? Story behind 49ers QB's nickname explained (Sportskeeda, 2023-10-10T22:30:20+00:00)
Deebo Samuel or Odell Beckham Jr.? Check out our experts' Start/Sit projections for Week 16

NFL Predictions 2023-24 × Game Rules Prizes How to Play Join our free-to-play NFL prediction game & stand a chance to win incredible prizes Login is mandatory
to be eligible for prizes Terms & Conditions For more information, check out our For Top 10 on the Leaderboard How to claim prize + - Only Top 10 ranked winners
in the Leaderboard are eligible for rewards 1. Winners will have to update their phone number and email ID on their profile page before 1st April 2024. Not
doing so, will make them ineligible for prizes. 2. Sportskeeda's team will contact the winners (from email-id: [email protected] ) on their updated email ID
after 1st April. Please note that next set of instructions will be sent to the updated Email IDs only. Steps to Update Profile Details 1.
</item 90>

<item 91>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
His KBO team, Kiwoom Heroes, announced the 25-year-old will be posted Monday, so his 30-day negotiating window with MLB teams begins Tuesday at 8 a.m. ET.

Lee was hampered by injury in 2023 but slashed .349/.421/.575 in 2022 with 36 doubles, 10 triples and a career-best 23 home runs. He has a .340 career average
in seven KBO seasons and has mostly played center field in recent years.

Braves land Jarred Kelenic, Marco Gonzales from Mariners

Date: Sunday, Dec. 3

Source: Ryan Divish, Seattle Times

The Braves have made a splash.

Atlanta has traded its No. 7-ranked prospect, RHP Cole Phillips, and RHP Jackson Kowar to the Mariners in exchange for OF Jarred Kelenic, LHP Marco Gonzales, 1B
Evan White and cash.
</item 91>

<item 92>
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
</item 92>

<item 93>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
When other assets get more attractive one must assess scope for valuation change in equities as well. For example, interest rates and bond yields in many
developed markets, including the US, are at their highest levels since 2007. This means investors may have to assess current valuations not just against last 10
years’ valuations, but of prior periods as well.

Why? Near zero interest rates and low bond yields made stocks with earnings yield of mere 1-2 per cent attractive for an FPI — for example stocks such as Asian
Paints, Nestle India and ABB India which today trade at PE of 65, 83 and 85 times respectively. This implies earnings yield of 1.5 per cent for Asian Paints and
1.2 per cent for Nestle and ABB. Such earnings yields of these stocks were fine for FPIs to hold when compared to zero interest income in their domestic market.
But now, with US bond yields offering 5 per cent, these may not be so attractive even when considering the growth prospects.
</item 93>

<item 94>
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
</item 94>

<item 95>
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
</item 95>

<item 96>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
The Kiwi all-rounder has breached 1 Cr mark.

1:55 PM IST: Rachin Ravindra is up next and CSK are the opening bidder.

1:52 PM IST: Sunrisers Hyderabad have bid at the base price and with no interest in the player, Wanindu Hasaranga to play for a new franchise.

1:50 PM IST: Wanindu Hasaranga is the first up in Set 2. He has a base price of Rs 1.5 Cr.

1:50 PM IST: First set of all-rounders is up next

1:42 PM IST: Anil Kumble:

RR don’t need any bowlers and they needed an overseas batter who can finish games to complement Hetmyer. They wanted some solidity and Powell gives RR the
batting hold in the side.

1:41 PM IST: Only three batters are sold from the first round and they are: Travis Head, Rovman Powell and Harry Brook.
</item 96>

<item 97>
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
</item 97>

<item 98>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
His high score was 130 runs in this campaign, and he provided good support from time to time whenever the team needed it.

4. Rachin Ravindra, New Zealand

The biggest find of this year's World Cup has to be this 23-year-old all-rounder, Rachin Ravindra. Ravindra has smashed three centuries and two fifties in this
campaign and accumulated a total of 578 runs in 10 matches.

3. Quinton de Kock, South Africa

A pivotal part of the South African top-order, de Kock has been a phenomenal batter, scoring 594 runs in 10 matches. de Kock has the most number of centuries in
this tournament, with four so far.

2.
</item 98>

<item 99>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
Win and it goes away. Period."

The incident elicited a response from the Baseball Writers Association of American (BBWAA), which defended Mintz, a BBWAA member.

"For the record, Jake is a BBWAA member in good standing and has earned the right to be credentialed through his hard work and quality baseball coverage," the
BBWAA wrote in the statement. "To assert otherwise, in vulgar terms, is both unprofessional and unacceptable. The BBWAA is deeply troubled that the league's own
network would permit the disparaging of one of our members in this fashion. Scrutinizing our work is part of the territory but comments such as these should
have no place on MLBN."

Following the backlash to her comments, Rizzo apologized on "High Heat" to Mintz and the BBWAA. She said her comments were "completely inappropriate, it was
inaccurate and it was beyond unprofessional.
</item 99>

<item 100>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
12:35 PM IST: A recap of the purse remaining:

IPL Auction Remaining Purse

Royal Challengers Bangalore (RCB): ₹23.25 crore

Chennai Super Kings (CSK): ₹31.4 crore.

Mumbai Indians (MI): ₹17.25 crore.

Gujarat Titans (GT): ₹38.15 crore

Lucknow Super Giants (LSG): ₹13.15 crore

Kolkata Knight Riders (KKR): ₹32.7 crore

Rajasthan Royals (RR): ₹14.5 crore

Delhi Capitals (DC): ₹28.9 crore

Punjab Kings (PBKS): ₹29.1 crore

Sunrisers Hyderabad (SRH): ₹34 crore

12:25 PM IST:

12.17 PM: One of the most talked-about players is Josh Hazlewood, who might not be available till the last leg of the tournament. Will the teams target him?
</item 100>
=== END ITEM id=2b21733b8b4f5daa ===

=== ITEM id=8afc091da1e53e45 ===
Gold sentences and facts:
Sentence 0: In the wake of the Wallabies’ World Cup flop, the Queensland Rugby Union have reaffirmed their position that they are committed to moving to a centralised high-performance model but won’t follow the NSW Waratahs by handing over the keys to Rugby Australia.
  Fact 0.0: The Wallabies had a World Cup flop.
  Fact 0.1: The Queensland Rugby Union reaffirmed its position.
  Fact 0.2: The Queensland Rugby Union is committed to moving to a centralised high-performance model.
  Fact 0.3: The Queensland Rugby Union will not follow the NSW Waratahs by handing over the keys to Rugby Australia.
Sentence 1: He will also be tasked with “integrating” the programs, with RA trying to centralise all of the Super Rugby’s high-performance programs.
  Fact 1.0: A person will be tasked with integrating the programs.
  Fact 1.1: RA is trying to centralise all of the Super Rugby's high-performance programs.

Retrieved items:

=== END ITEM id=8afc091da1e53e45 ===

=== ITEM id=bd11343b62772460 ===
Gold sentences and facts:
Sentence 0: Also included is defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury.
  Fact 0.0: Cameron Carter-Vickers is also included.
  Fact 0.1: Cameron Carter-Vickers is a defender.
  Fact 0.2: Cameron Carter-Vickers only just made his debut for Celtic this season.
  Fact 0.3: Cameron Carter-Vickers's Celtic debut came thanks to a hamstring injury.
Sentence 1: 30 mins to kickoff: Paraguay have a new manager, as Daniel Garnero makes his debut as the new permanent boss.
  Fact 1.0: The match kickoff is 30 minutes away.
  Fact 1.1: Paraguay have a new manager.
  Fact 1.2: Daniel Garnero makes his debut as Paraguay's new permanent boss.

Retrieved items:

=== END ITEM id=bd11343b62772460 ===

=== ITEM id=0ad820e61dfb9652 ===
Gold sentences and facts:
Sentence 0: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 0.0: Wang testified that he heard Bankman-Fried tell another trader something.
  Fact 0.1: Bankman-Fried told the other trader it was okay to continue withdrawing money from the account.
  Fact 0.2: The withdrawals were allowed as long as the amount was less than FTX's total trading revenue.
Sentence 1: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 1.0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 1.1: Huang said Bankman-Fried told Paradigm he didn't want investors as directors.
  Fact 1.2: Huang said Bankman-Fried did plan on having a board with experts.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:

=== END ITEM id=0ad820e61dfb9652 ===

=== ITEM id=0e8f5a648e19aad3 ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google "siphons off" news publishers' content.
  Fact 0.2: The case argues that Google siphons off news publishers' readers.
  Fact 0.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 0.4: The case argues that Google does this through anticompetitive means.
Sentence 1: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 1.0: The argument in the case will probably come up later in the trial.
  Fact 1.1: Epic chose to focus more on simply painting Google as the bad guy.
  Fact 1.2: Epic made this choice on day one of the trial.

Retrieved items:

=== END ITEM id=0e8f5a648e19aad3 ===
