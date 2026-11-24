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

=== ITEM id=7901bcf18fe79e2f ===
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
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
However, the NFL does not just want to stop at $18 billion. It wants to become a $27 billion league in 4 years. But it must address a few skeletons in its
closet before reaching that goal.

If you were around in the 1970s (that would make you at least 50 years old today), you would know that the NBA did not have the mass appeal that it enjoys now.
With average game attendance hovering around 8,000 a game, it was one of the darkest periods of the basketball league. Drug abuse, brawls on the hardwood court
that sometimes spilled over onto coaches and players – it wasn’t a pretty sight. As reported in the Washington Post, even though there are no reliable figures
on cocaine use by players, “estimates by people in the game range from 40 percent to 75 percent.”
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
1. Mohammad Shami, India

Shami announced his return to the World Cup with a bang, picking a fifer in the very first match he played for India. He has picked up 24 wickets so far in just
six matches, conceding just 257 runs. He has picked up three five-wicket hauls including a seven-wicket haul against New Zealand in the semi-final on Wednesday,
November 15 at the Wankhede Stadium in Mumbai.

Catch the latest stock market updates here. For all other news related to business, politics, tech, sports, and auto, visit Zeebiz.com.
</item 4>

<item 5>
ICC World Cup 2023 Most Runs and Wickets: Full List of top 10 batsmen and bowlers in 2023 edition of cricket World Cup, Know where Virat Kohli, Rohit Sharma,
Shami, Bumrah, rank (Zee Business, 2023-11-18T15:51:59+00:00)
Top 10 Bowlers of ODI World Cup 2023:

Rank Player Mat Wkts 1 Mohammed Shami (IND) 7 23 2 A Zampa (AUS) 11 23 3 D Madushanka (SL) 9 21 4 JJ Bumrah (IND) 11 20 5 G Coetzee (SA) 8 20 6 Shaheen Shah
Afridi (PAK) 9 18 7 M Jansen (SA) 9 17 8 RA Jadeja (IND) 11 16 9 Josh Hazlewood 11 16 10 MJ Santner (NZ) 10 16 11 BFW de Leede (NED) 9 16

*Still Bowling

10.
</item 5>

<item 6>
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
</item 6>

<item 7>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
Win and it goes away. Period."

The incident elicited a response from the Baseball Writers Association of American (BBWAA), which defended Mintz, a BBWAA member.

"For the record, Jake is a BBWAA member in good standing and has earned the right to be credentialed through his hard work and quality baseball coverage," the
BBWAA wrote in the statement. "To assert otherwise, in vulgar terms, is both unprofessional and unacceptable. The BBWAA is deeply troubled that the league's own
network would permit the disparaging of one of our members in this fashion. Scrutinizing our work is part of the territory but comments such as these should
have no place on MLBN."

Following the backlash to her comments, Rizzo apologized on "High Heat" to Mintz and the BBWAA. She said her comments were "completely inappropriate, it was
inaccurate and it was beyond unprofessional.
</item 7>

<item 8>
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
</item 8>

<item 9>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
"That's taking away from true reporters and true journalists. Where were you on the random Tuesday in April in Cincinnati when this team was playing in
Cincinnati? Where were you on a rain delay, and you have to sit in the press box for all of this time? It's ridiculous."

"And then some jackoff comes in at the end of the season that gets a credential, God only knows why. And the clubhouse is a sacred space." Alanna Rizzo went off
on Jake Mintz's clubhouse reporting on MLBN's "High Heat" Thursday. pic.twitter.com/6dooqeLPnh — Awful Announcing (@awfulannouncing) October 12, 2023

Rizzo's comments immediately drew heat from the baseball journalism world, including from Janes, who posted a thread defending Mintz. Janes reported that
cameras and recorders were recording, and that Arcia yelled the comments while they were recording and with more than a dozen reporters around.
</item 9>

<item 10>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
From Colin Kaepernick taking a knee during the national anthem to protest against racism, to domestic violence by players, to the Deflategate and Bountygate
scandals. But there have been bigger crises as well. The Union of Concerned Scientists accused the NFL of trying to intimidate scientists studying the link
between CTE and pro football. The NFL is already way past other leagues in terms of revenue, and now with the pivotal shift in its strategy, “America’s favorite
pastime” is on its way to world domination.

ADVERTISEMENT Article continues below this ad

Editorial Disclaimer: The views expressed are those of the author and do not necessarily reflect the views of EssentiallySports.

Watch This Story: Will Patrick Mahomes Eclipse Tom Brady as the NFL Goat? Travis Kelce Responds
</item 10>

<item 11>
IPL 2024 Auction Live: Starc, Cummins in spotlight as Dubai set for ‘richie-rich’ carnival (Insidesport, 2023-12-18T23:30:00+00:00)
Will the teams target him?

12.10 PM: As per a BCCI directive, all the IPL venues can’t hold any domestic game in February, and need to hand over the stadium to teams in first week of
March.

12.07 PM: Meanwhile, the IPL 2024 window starts on March 22. An early start to the tournament might force Ranji Trophy venues to shift.

12.04 PM: Harshal was released by RCB, but in the VHT, he got 18 wickets in 10 matches and scored 183 runs too. So, he would be considered for an all-rounder
role.

12.01 PM: Will the Vijay Hazare Trophy performances have an impact on the bidding? Will Harshal Patel get a bid? Will he be trusted with bowling in the death
overs again? We’ll find out in no time.
</item 11>

<item 12>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
Billionaire entrepreneur Mark Cuban, who paid $1.8 million (approx.) in penalties on behalf of his team, the Dallas Mavericks, fondly remembered the former NBA
commissioner David Stern, on his death in January 2020. Stern came and instituted an anti-drug agreement and even banned All-Star player Michal Richardson for
life for drug abuse. The revitalized image of the NBA helped rake in the moolah from sponsors and made it more appealing to fans. He even went against the tide
to open up the Olympic basketball competition to NBA players resulting in the famous “Dream Team” of 1992. America has never failed to win a medal in the 19
Olympic basketball games that it has played since then.

ADVERTISEMENT Article continues below this ad

The NFL has had its fair share of, um, let’s say, “bumps” down the road as well.
</item 12>

<item 13>
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
</item 13>

<item 14>
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
</item 14>

<item 15>
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
</item 15>

<item 16>
Rollback Policy, a Boon or a Curse? The Controversial System That Might Redefine the Golfing World by the End of 2030 (Essentially Sports,
2023-12-03T23:45:00+00:00)
Secondly, a league that thrives on thrills and excitement will lose its USP if the ball suddenly doesn’t travel 300 yards on most occasions. As of this year, 28
of 48 LIV golfers average over 300 yards driving distance. Dustin Johnson and DeChambeau are both in the top five.

ADVERTISEMENT Article continues below this ad

But one crucial thing to remember here is that the golf ball rollback policy will render most balls totally ‘unplayable’. Moreover, the PGA Tour has
contradicted itself several times in the past few years. It won’t be surprising if Monahan & Co. eventually decides to agree with the policy change sometime
later. So, the question remains: how viable would it be for manufacturers to produce two types of balls, one for Majors and the other for LIV Golf? This just
throws the world of golf into uncharted territory.
</item 16>

<item 17>
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
</item 17>

<item 18>
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
</item 18>

<item 19>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
The only difference was that Janes did not name the player.

The reporting of Arcia's comments proved to be divisive across the baseball community. Among the most prominent to call out Mintz was Rizzo. The "High Heat"
co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us" when "some jackoff comes in at the end
of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the players who were upset by the reporting of
Arcia's comments, before she called Mintz out by name.

MORE: Every time Nick Castellanos has interrupted a somber moment with a home run

"First of all, this guy, Jake Mintz, that's not even a reporter," Rizzo said. "That's taking away from true reporters and true journalists.
</item 19>

<item 20>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
She said that reporters abstaining from writing on his comments would be akin to "protecting players from themselves."

She also went on to say that what Arcia said wasn't a slur or hateful, and added the only reason anyone was concerned about it was because "they know it would
motivate Harper to beat [the Braves]. Which implies the Braves do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they
lost the other day."

"I didn't say something sooner because I thought this was overblown," Janes continued. "And frankly, I thought it wouldn't help. But I am genuinely disgusted by
the way some media members are acting like a reporter was in the wrong for this. He wasn't.

"And if the Braves let this be the story of their series, that's on them. Win and it goes away. Period."
</item 20>
=== END ITEM id=7901bcf18fe79e2f ===

=== ITEM id=87c1e48d5106d644 ===
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
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Updated at 23.20 GMT

6d ago 23.16 GMT I suppose it’s appropriate that it ended with a six, the 120th of the series. Hope played a gem of an innings there, calmly anchoring West
Indies in an increasingly nervous chase with a run-a-ball 43. England fought admirably to defend a below-par total; but for Hope, they would probably have done
it.

Updated at 23.17 GMT

6d ago 23.14 GMT WEST INDIES WIN THE SERIES 3-2! 19.2 overs: West Indies 133-6 (Hope 43, Holder 4) That’s it! Hope pings Woakes over extra cover for six to seal
another series victory for West Indies! West Indies’ Shai Hope (right) celebrates with Jason Holder after hitting a six to win the match by 4 wickets.
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
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
That is all.”



6d ago 22.13 GMT 6th over: West Indies 39-2 (Charles 16, Hope 5) Charles mistimes a short ball from Woakes just short of Moeen in the covers. In the end Moeen
does well to save the boundary. This has been a good fightback by England, with only nine runs scored in the last three overs.



6d ago 22.09 GMT 5th over: West Indies 37-2 (Charles 15, Hope 4) Topley continues, which makes sense because the spinners will be doing most of the work after
the Powerplay, and concedes only three runs. He has quietly had a good series and is comfortably the most economical seamer on either side.
</item 3>

<item 4>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.45 GMT 14th over: West Indies 94-3 (Hope 19, Rutherford 30) West Indies are rushing towards a series victory. Rutherford hooks the new bowler Sam
Curran for six more, and there are 11 from the over in total. Rutherford has 30 from 23 balls, Hope 19 from 28. They’re playing their roles to perfection.



6d ago 22.40 GMT 13th over: West Indies 83-3 (Hope 17, Rutherford 21) Rehan Ahmed returns, perhaps in the hope of tempting Rutherford into one big shot too
many. Instead it’s Shai Hope, who is happily playing the anchor role, who faces five of the six deliveries. England are running out of time.
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
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
12.4 overs: West Indies 108-4 (Hope 27, Powell 6)

Updated at 01.09 GMT

13 Dec 2023 00.58 GMT 12th over: West Indies 102-4 (Hope 26, Hetmyer 1) Just a single and the wicket off the over, Shai Hope can’t break the legspin shackles.
Windies skipper Rovman Powell is the new man and Buttler cranks the pressure by inserting a slip.

Updated at 01.04 GMT

13 Dec 2023 00.52 GMT WICKET! Hetmyer c Duckett b Rashid 1 (West Indies 101-4) Adil Rashid takes his 100th T20I wicket! A crucial one in the context of the game
too – the dangerous Hetmyer is sent on his way for just a single as England look to ramp up the pressure heading into the nub end of this match.
</item 6>

<item 7>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.57 GMT WICKET! West Indies 113-5 (Powell c Rashid b Topley 8) England haven’t given this up. Powell edges Topley to short third man, where Rashid
takes a smart low catch to his left. West Indies need 20 from 21 balls. Reece Topley (right) celebrates the dismissal of Rovman Powell. Photograph: Randy
Brooks/AFP/Getty Images

Updated at 23.07 GMT

6d ago 22.54 GMT 16th over: West Indies 112-4 (Hope 29, Powell 8) Hope, who is playing an exemplary innings, opens the face to steer Rehan for his first
boundary. Rehan has struggled as the series has progressed, an d later in the over Powell muscles him over long-on for six. West Indies are 21 runs away from a
hat-trick of T20 series wins this year: South Africa away, India and England at home.
</item 7>

<item 8>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
What a difference two days make. Somehow Tuesday’s ludicrous, record-smashing, high-scoring entertainment was followed by a sequel on the same surface that
eventually delivered thrilling low-scoring entertainment, and a different result claimed in very different style. It was eventually won for West Indies with
four balls to spare and in the most unfitting of ways – with a massive six – by Shai Hope, again proving himself the right man for a run-chase.

So a series marked by stunning power hitting, roaring run rates and an unexpected midway transformation in England’s performances and prospects concluded with a
3-2 triumph for West Indies, who extended their supremacy over England at home to four series in three formats across nearly five years.

England assembled exactly the same players in exactly the same place and exactly the same situation – losing the toss, put in to bat – but this was an entirely
different performance.
</item 8>

<item 9>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Just as surely as when they came out on Tuesday facing an almost unachievable task, the outcome of the West Indies’ run chase seemed inevitable from the start.
With such a meagre target they needed little more than to keep their heads, and though both Topley and Sam Curran conceded only two with their final overs to
push the contest into its final over, that never looked in doubt.

skip past newsletter promotion Sign up to The Spin Free weekly newsletter Subscribe to our cricket newsletter for our writers' thoughts on the biggest stories
and a review of the week’s action Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more
information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy
. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Quick Guide How do I sign up for sport breaking news alerts?
</item 9>

<item 10>
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
</item 10>

<item 11>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
Amid the wider theme of transformation Adil Rashid’s excellence remains constant. The 35-year-old, who rose to the top of the T20 bowling rankings for the first
time on Wednesday, took two wickets and conceded just 21 while Topley was even better, also claiming a pair of victims but conceding only 17. England take
plenty of positives from this series, but the result was not one.
</item 11>

<item 12>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.10 GMT 3rd over: West Indies 36-1 (Mayers 10, Hope 4) Shai Hope is the new man and digs out a Woakes yorker and somehow manages to crunch it away
down the ground for four. The replay of the Duckett catch is extremely enjoyable, Woakes owes him one for that as it was a poor ball outside off stump.



13 Dec 2023 00.07 GMT WICKET! Brandon King c Duckett b Woakes 22 (West Indies 32-1) Chris Woakes into the attack to replace the beleaguered Sam Curran. OH. MY.
LORDY. Ben Duckett flies to his left at backward point to pull off a spectacular diving catch off a full blooded cut shot from King. Stunner!
</item 12>

<item 13>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
Adil Rashid (right) celebrates the dismissal of Johnson Charles (left). Photograph: Randy Brooks/AFP/Getty Images

Updated at 22.43 GMT

6d ago 22.17 GMT 7th over: West Indies 51-2 (Charles 26, Hope 7) Rehan Ahmed’s first over disappears for 12. It started quite gently, with four singles and a
two off the first five balls, but then Charles clattered a no-nonsense straight six. “We’re not going to win this, Rob,” writes Guy Hornsby. “But I’m just here
to say I’m still watching for Woakes’ seam position and Rehan’s googly. That is all.”
</item 13>

<item 14>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 22.05 GMT 4th over: West Indies 34-2 (Charles 15, Hope 1) England needed early wickets to have a realistic chance of victory. One more in the Powerplay,
before they bring on Adil Rashid, would put them right back in the game.



6d ago 22.03 GMT WICKET! West Indies 33-2 (Pooran b Woakes 10) UltraEdge shows Pooran did edge that ball from Topley – but it doesn’t matter because Woakes has
got him! Pooran tried to ride the bounce of an awkward delivery that cramped him for room, took the inside of an angled bat and ricocheted onto the stumps.
Nicholas Pooran looks glum as his bails go flying.
</item 14>

<item 15>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
Brutal elegance. Rehan comes back well with two sharply turning leggies. Bosh! Shai Hope gets in on the act and smears a length ball over mid on for the third
six of the over. It’s a learning curve for the tyro spinner but that doesn’t make it any less bruising.



13 Dec 2023 00.23 GMT 6th over: West Indies 59-1 (Mayers 22, Hope 13) Curran is called back, Mills granted just the one over for the time being. Two slower
balls bring two soothing dots for England. A single each to Mayers and Hope. Curran does very well to get out of the over with just three runs conceded.
Powerplay complete. How long till we see some spin from Rehan? Not long at all, he’s on for the next over.
</item 15>

<item 16>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 01.45 GMT WICKET! Shai Hope c Brook b Rehan Ahmed 36 (West Indies 123-5) Ahmed is bunted for SIX by Shai Hope but strikes with the next ball as Hope
holes out!



13 Dec 2023 01.44 GMT 14th over: West Indies 112-4 (Hope 27, Powell 10) Liam Livingstone instead of Rashid and he does well to keep it to singles, England
trying to take this as deep as possible and hope West Indies fluff their lines at the death. West Indies need 55 from 36 balls.
</item 16>

<item 17>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
The wicket has given Ahmed a confidence boost and he gets out of the over with three dots. 11th over: West Indies 101-3 (Hope 25, Hetmyer 1)



13 Dec 2023 00.43 GMT 10th over: West Indies 99-2 (Hope 25, Pooran 13) Rashid is given another after his excellent first over. The googly is spotted by Pooran
and driven down the ground handsomely for SIX! A glide behind point bring a single, Hope returns the favour with a flick into the leg side. A short ball is
panned into the deep by Pooran for a couple to end the over. Bon Jovi O’clock – West Indies need 73 from 60. Time for a quick slurp for the players and a
knuckle stretch for me. Is there anybody alive out there?!
</item 17>

<item 18>
West Indies v England: fifth and deciding men’s T20 cricket international – live (The Guardian, 2023-12-21T19:39:01+00:00)
6d ago 23.00 GMT 17th over: West Indies 114-5 (Hope 30, Russell 1) The wicket England really need is that of Shai Hope, who is quietly guiding West Indies
towards the finish line.
</item 18>

<item 19>
West Indies clinch T20 series after Shai Hope edges hosts past England total (The Guardian, 2023-12-21T23:45:58+00:00)
That made it 70 for four and the game was one ball into its ninth over, at which point Liam Livingstone and Moeen Ali had to focus for a while on disaster
prevention rather than entertainment.

Gudakesh Motie is embraced his his West Indies teammates after dismissing England’s in-form Phil Salt. Photograph: Randy Brooks/AFP/Getty Images

Suddenly bowlers were energised by the prospect of posting, for a change, less-than-humiliating figures. Akeal Hosein’s were outstanding, his four overs going
for just 20 and bringing the wickets of Will Jacks and Moeen, undone by a relay catch on the boundary after combining with Livingstone to add 40 for the fifth
wicket. Livingstone followed precisely two overs later, prodding the ball limply back into the hands of Motie, at which point things really fell apart, the last
five wickets surrendered in 20 balls for just 11 runs.
</item 19>

<item 20>
West Indies v England: first T20 international – live (The Guardian, 2023-12-12T21:49:56+00:00)
13 Dec 2023 00.20 GMT 5th over: West Indies 56-1 (Mayers 20, Hope 12) Woakes continues, after five balls he’s gone for just three runs. Can he get out of the
over cheaply? No – A length ball is clubbed down the ground for SIX with a minimum of fuss by Shai Hope. West Indies have come out of the traps as well as
England did, one over of the Powerplay left.
</item 20>
=== END ITEM id=87c1e48d5106d644 ===

=== ITEM id=c9155f4b73e27e08 ===
Gold sentences and facts:
Sentence 0: McHenry himself has brushed off attempts to take the job more permanently after he was appointed to the role after the unprecedented ouster of McCarthy more than two weeks ago.
  Fact 0.0: McHenry has brushed off attempts to take the job more permanently.
  Fact 0.1: McHenry was appointed to the role after the ouster of McCarthy.
  Fact 0.2: The ouster of McCarthy was unprecedented.
  Fact 0.3: The ouster of McCarthy happened more than two weeks ago.
Sentence 1: Wall Street is also absorbing the ouster of Kevin McCarthy as the speaker of the House of Representatives.
  Fact 1.0: Wall Street is also absorbing the ouster of Kevin McCarthy.
  Fact 1.1: Kevin McCarthy was the speaker of the House of Representatives.
  Fact 1.2: Kevin McCarthy was ousted as speaker of the House of Representatives.

Retrieved items:
<item 1>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
A once in a lifetime opportunity. On the flip side, if they lose Ohtani, the roster would need a solid revamp. Not to mention the Angles are infamous for their
below-par farm system and playoff drought spanning 9 seasons.

With that out in the open, what are the possible options for the Halos? Ray Montgomery could be an option, considering that he began his tenure with the Angels.
Walt Weiss could potentially be a decent fit, too. Weiss has been a part of a successful team in the Atlanta Braves recently, having been their bench coach
since before the 2018 season. Additionally, among other options are Ron Washington and Clayton McCullough. Though these options are open, no solid information
has been revealed yet as to who would fill up the position.

Moving On To The Houston Astros

After his fourth season with the Astros and his 26th big league managerial season, Dusty Baker announced his retirement last week.
</item 1>

<item 2>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
This newest president of baseball operations for the Mets is a former Brewer himself.

The Fifth Team In The List Is San Diego Padres

Bob Melvin was said to be returning to the Padres as manager, despite the team’s 82-80 record and 2023 postseason absence. But in a turn of events, the San
Francisco Giants hired Melvin when San Diego gave permission to interview him for SanFran’s managerial vacancy.

ADVERTISEMENT Article continues below this ad

Now, the Padres’ vacant position is linking to former Cardinals manager Mike Shildt multiple times. Bob Nightengale of USA Today Sports reports that Shildt is
the front-runner to take over. “The Padres will be hiring their sixth manager in just 10 years under (general manager) A. J. Preller. Former St. Louis Cardinals
manager Mike Shildt is the favorite while associate manager Ryan Flaherty is getting strong consideration,” Nightengale said.
</item 2>

<item 3>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
“He doesn’t have the votes to be speaker,” Rep. Carlos Gimenez, R-Fla., said after a late Thursday meeting when Jordan sought to hear detractors out and shore
up support.

The holdouts want “nothing” from Jordan, Gimenez said, adding that some of the lawmakers in the meeting simply called on Jordan to drop out of the race.

One extraordinary idea, to give the interim speaker pro tempore, Rep. Patrick McHenry, more powers for the next several months to at least bring the House back
into session and conduct crucial business, was swiftly rejected by Jordan’s own ultra-conservative allies.

A “betrayal,” said Rep. Jim Banks, R-Ind.

Next steps were highly uncertain as angry, frustrated Republicans predict the House could essentially stay closed for the foreseeable future — perhaps until the
mid-November deadline for Congress to approve funding or risk a federal government shutdown.

“We’re trying to figure out if there’s a way we can get back with a Republican-only solution,” said veteran legislator Rep.
</item 3>

<item 4>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
Don Bacon, R-Neb.

For more than two weeks the stalemate has shut down the U.S. House, leaving a seat of American democracy severely hobbled at a time of challenges at home and
abroad. The House Republican majority appears to have no idea how to end the political turmoil and get back to work.

With Republicans in majority control of the House, 221-212, any candidate can lose only a few detractors. It appears there is no Republican at present who can
win a clear majority, 217 votes, to become speaker.

In fact, the hard-charging Judiciary chairman lost rather than gained votes despite hours spent trying to win over holdouts, no improvement from the 20 and then
22 Republicans he lost in early rounds this week.

Friday’s vote was 194 for Jordan, his lowest tally yet, and 210 for Jeffries, with two absences on each side.

“He doesn’t have the votes to be speaker,” Rep.
</item 4>

<item 5>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
11, 2001, terror attacks to ensure continuity of government.

The novel concept of boosting the interim speaker’s role was gaining favor with a pair of high-profile Republicans: Former GOP speakers Newt Gingrich and John
Boehner.

Gingrich said while he likes Jordan, he has “no faith” the nominee can get much beyond the 200 votes he won in the first vote.

“We can’t sit around and suck our thumbs and hope the world will wait until the House Republicans get their act together,” Gingrich told Fox News’ Sean Hannity
on his show.

Boehner reposted Gingrich’s views saying, “I agree,” on social media.

The two men have deep experience with the subject. Both were chased to early retirement.

“The Republicans are unable to function right now,” Jeffries said late Tuesday. “All options are on the table to end the Republican civil war,” he added
Wednesday.

In nominating Jordan, veteran Republican Rep. Tom Cole of Oklahoma said it was time to end the upheaval that he had warned against with McCarthy’s sudden
ouster.
</item 5>

<item 6>
Will the Panthers fire Frank Reich? Impatient David Tepper could turn up Carolina's hot seat with 'Thursday Night Football' loss (Sporting News,
2023-11-06T21:17:51+00:00)
And I don't think that people are thinking that there's trouble right now with those people. But would anybody be surprised with anything that happens in
Carolina this year? No."

"One thing about David Tepper is he's not very patient and he does whatever he wants..



I don't think anybody would be surprised if anything happens in Carolina with the season that they're having" ~ @AdamSchefter #PMSLive
pic.twitter.com/PV7bfVUtHp — Pat McAfee (@PatMcAfeeShow) November 6, 2023

Schefter pointed to Carolina's firing of Rhule as an example of Tepper's impatience. There had been rumors that the owner was going to wait things out with
Rhule in the weeks leading up to his in-season departure. Instead, he decided to pull the trigger and part with him.
</item 6>

<item 7>
Hottest seat on each NFC team: Buns burning for these four head coaches (Yardbarker, 2023-11-30T22:29:33+00:00)
— Mike Santa Barbara

NFC South

ATLANTA FALCONS | Quarterback Desmond Ridder: The Falcons have a golden path toward a division crown, but Ridder’s two-interception performance against the
Saints in Week 12 was another reminder Atlanta isn’t a legitimate contender with him. He’ll finish the season leading the Falcons, who will probably get a home
playoff game, but the clock is ticking on Ridder’s time as an NFL starter.

CAROLINA PANTHERS | General manager Scott Fitterer: Owner David Tepper should have the hottest seat, but he isn’t going to fire himself. Instead, when he runs
out of people on the sidelines to fire, he’ll take out his frustration on a GM (partly) responsible for constructing the least impressive roster in football.

NEW ORLEANS SAINTS | Head coach Dennis Allen: New Orleans' 5-6 record against the league’s easiest schedule is bad.
</item 7>

<item 8>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
“And, to be honest, maybe 18, after I didn’t get the last list spot at Collingwood, but I’m glad the stars aligned the way they did.” In a twist of fate,
defender Charlie Dean re-fractured his right foot at training, opening another list spot for Collingwood to snatch Markov back from Carlton – and the rest is
history. “Having trained at Collingwood for two months, I felt like I was almost doing the dirty on ‘Fly’ [Craig McRae], going behind his back – I was almost
like a cheater,” he said. “He was very respectful and understanding, and it probably allowed me to open myself up and be vulnerable to that football club. I’m
super grateful for them showing interest in me, in the first place.

“I know I’ve signed with [Collingwood] and on the dotted lines, I’ve agreed to hate Carlton, but there’s a bit of me that is really grateful and appreciative to
them.”
</item 8>

<item 9>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
2023 has been a topsy-turvy year for many teams. While some have outshone others, these 5 teams are at a crossroads, with a common stifling problem. Amid the
heavy restructuring that many of the MLB teams have seen of late, there are pentagonal vacancies for managers. The New York Mets, the Los Angeles Angels, the
Houston Astros, the Cleveland Guardians, and the San Diego Padres are in dire need of filling up the managerial position as the year ends.

America’s Favorite Video Today

While some team representatives skim through names within the MLB world, some hope to look for an outside hire. It could be a risky state for all of them.
Nevertheless, there are many possibilities and roadblocks one might encounter. Be it budgetary issues, past performance, or availability of candidates, these
teams have not had success in putting a pin on one candidate. That said, the skipper market is abuzz!
</item 9>

<item 10>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
Congress faces daunting challenges, risking a federal shutdown at home if it fails to fund the government and fielding Biden’s requests for aid to help Ukraine
and Israel in the wars abroad.

First elected in 2006, Jordan has few bills to his name from his time in office. He also faces questions about his past. Some years ago, Jordan denied
allegations from former wrestlers during his time as an assistant wrestling coach at Ohio State University who accused him of knowing about claims they were
inappropriately groped by an Ohio doctor. Jordan has said he was never aware of any abuse.

___

Associated Press writers Kevin Freking and Mary Clare Jalonick contributed to this report.
</item 10>

<item 11>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Wall Street is also absorbing the ouster of Kevin McCarthy as the speaker of the House of Representatives. The unprecedented move to remove a speaker from the
position likely doesn’t change much in the short term, with funding for the US government set until November 17.

“That said, a leadership vacuum in the House raises the odds of a government shutdown when the current funding extension expires,” according to economists at
Goldman Sachs. A shutdown would drag on the US economy, raising the risk of a recession, though financial markets have held up relatively well through past
shutdowns. On Wall Street, big tech stocks helped to support the market after leading it lower a day earlier. They tend to move more sharply with expectations
for rates because high-growth stocks are seen as some of the biggest victims of high yields. In markets abroad, stock indexes were mixed across much of Europe.
</item 11>

<item 12>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
Jordan has been a top Trump ally, particularly during the Jan. 6 Capitol attack by the former president’s backers who were trying to overturn the 2020 election
he lost to Biden. Days later, Trump awarded Jordan a Medal of Freedom.

First elected in 2006, Jordan has few bills to his name from his time in office. He also faces questions about his past.

Some years ago, Jordan denied allegations from former wrestlers during his time as an assistant wrestling coach at Ohio State University who accused him of
knowing about claims they were inappropriately groped by an Ohio State doctor. Jordan has said he was never aware of any abuse.
</item 12>

<item 13>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
Tom Cole, R-Okla.

“That’s what normal majorities do. What this majority has done is prove it’s not a normal majority.”

What was clear was that Jordan was refusing to step aside, appearing determined to wait out his foes even as his path to become House speaker was all but
collapsing.

Many view Jordan, a founding member of the hard-right House Freedom Caucus, as too extreme for a central seat of U.S. power, second in line to the presidency.

“One thing I cannot stomach or support is a bully,” said a statement from Rep. Mariannette Miller-Meeks, R-Iowa, who voted against Jordan on the second ballot
and said she received “credible death threats.”

Democratic Leader Jeffries reiterated that his party was “ready, willing and able” to partner with more traditional Republicans on a path to re-open the House
—- particularly as Congress is being asked to consider President Joe Biden’s aid package for Israel, Ukraine and other needs.
</item 13>

<item 14>
Republicans still don’t know how to talk to young voters online (TechCrunch, 2023-10-02T21:42:14+00:00)
Influencer marketing agencies on both ends of the political spectrum are ramping up their recruitment faster than the Federal Elections Commission can regulate
the industry. This year, the conservative agency Influenceable has been recruiting Gen Z creators to rally behind far-right politicians and parrot GOP talking
points, without disclosing their pay. The tactic irked some Republicans, the Texas Tribune reported, including a Texas state representative who called for an
investigation into the company.

Given the resistance to Republican politicians in online spaces that attract young people, it’s unsurprising that candidates may rely on shadowy agencies like
Influenceable to do the work for them. Republican politicians have a reputation for botching even the most straightforward digital campaigns. In May, Florida
Gov. Ron DeSantis launched his presidential campaign on Twitter Spaces in an audio conversation with Elon Musk. The conversation started late and was so riddled
with technical issues that it ended after just 21 minutes. Critics on both the right and the left described the launch as a disaster.
</item 14>

<item 15>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
Republicans dropped Rep. Jim Jordan on Friday as their nominee for House speaker, making the decision during a closed-door session after the hard-edged ally of
Donald Trump failed badly on a third ballot for the gavel.

Afterward, Jordan said simply of his colleagues, “We put the question to them, they made a different decision.”

The hard-charging Judiciary Committee chairman said House Republicans now need to come together and “figure out who our speaker is going to be.”

The House impasse deepening into a full-blown crisis, Republicans have no realistic or workable plan to unite the fractured GOP majority, elect a new speaker
and return to the work of Congress that has been languishing since hard-liners ousted Kevin McCarthy at the start of the month.

Majority Leader Steve Scalise said they’re going “come back and start over” on Monday.

Angry, frustrated Republicans who have been watching their majority control descend into chaos left the private session blaming one another for the divisions
they have created.
</item 15>

<item 16>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
Republican Rep. Jim Jordan failed again Wednesday on a crucial second ballot to become House speaker, the hard-fighting ally of Donald Trump losing even more
GOP colleagues who are refusing to give him the gavel.

Next steps were highly uncertain as angry, frustrated Republicans looked at other options. A bipartisan group of lawmakers floated an extraordinary plan — to
give the interim speaker-pro-tempore, Rep. Patrick McHenry, R-N.C., more power to reopen the immobilized House and temporarily conduct routine business.

What was clear was that Jordan’s path to become House speaker was almost certainly lost. He was opposed by 22 Republicans, two more than he lost in first round
voting the day before.

“We’ll keep talking to members, keep working on it,” Jordan, a founding member of the hard-right Freedom Caucus, said after the vote, vowing to stay in the
race.

The House came to another abrupt standstill, stuck now 15 days without a speaker — a position of power second in line to the presidency — since the sudden
ouster of Kevin McCarthy.
</item 16>

<item 17>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
With that, one has to wait and watch who serves as the best fit.

ADVERTISEMENT Article continues below this ad

With the facts broken down, it is a matter of time that the teams narrow down their best options. MLB world will welcome 2024 with high expectations. With new
management, the fans expect teams to come back with great vigor and ferocity.

Watch This Story: Every MLB Team That Has Never Won a World Series
</item 17>

<item 18>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
“We have a chance today to end that chaos, end that uncertainty,” Cole said.

He said that Jordan was not a “shrinking violet” but someone who could lead the House.

Democratic Rep. Pete Aguilar of California nominated Jeffries, noting the Democratic leader continues to win more votes and is the best choice to move the
country forward.

“The country cannot afford more delays and more chaos,” Aguilar said.

Jordan had relied on backing from Trump, the party’s frontrunner in the 2024 election to challenge President Joe Biden, and groups pressuring rank-and-file
lawmakers for the vote, but it was not enough.

Flexing their independence, the holdouts are a mix of pragmatists — ranging from seasoned legislators and committee chairs worried about governing, to newer
lawmakers from districts where voters back home prefer President Joe Biden to Trump.

Some Republicans resent being pressured by Jordan’s allies and say they are being threatened with primary opponents if they don’t support him as speaker. Others
are simply upset at the way the whole process has dragged out.
</item 18>

<item 19>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
A closed-door meeting Thursday to regroup grew heated at times with Republican factions blaming one another for sending their majority into chaos, lawmakers
said.

Elevating McHenry to an expanded speaker’s role could be a possible off-ramp for the crisis, but it would not be as politically simple as it might seem.

Republicans are loath to partner with the Democrats in a bipartisan way on the arrangement, and it’s highly unlikely Republicans could agree to give McHenry
more powers on their own, since their hard-liners don’t like it.

McHenry himself has brushed off attempts to take the job more permanently after he was appointed to the role after the unprecedented ouster of McCarthy more
than two weeks ago.

To win over GOP colleagues, Jordan had relied on backing from Trump, the party’s front-runner in the 2024 election, and groups pressuring rank-and-file
lawmakers for the vote. But they were not enough and in fact backfired on some.

Jordan has been a top Trump ally, particularly during the Jan.
</item 19>

<item 20>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
The only difference was that Janes did not name the player.

The reporting of Arcia's comments proved to be divisive across the baseball community. Among the most prominent to call out Mintz was Rizzo. The "High Heat"
co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us" when "some jackoff comes in at the end
of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the players who were upset by the reporting of
Arcia's comments, before she called Mintz out by name.

MORE: Every time Nick Castellanos has interrupted a somber moment with a home run

"First of all, this guy, Jake Mintz, that's not even a reporter," Rizzo said. "That's taking away from true reporters and true journalists.
</item 20>

<item 21>
Will the Panthers fire Frank Reich? Impatient David Tepper could turn up Carolina's hot seat with 'Thursday Night Football' loss (Sporting News,
2023-11-06T21:17:51+00:00)
The Panthers aren't guaranteed to fire Reich even amid their poor season. But as ESPN's Adam Schefter implied on "The Pat McAfee Show," the retread coach's seat
is growing hotter with each passing week.

"I don't think anybody's got a long rope there just because of the type of season that they're having," Schefter said when asked about Reich's status in
Carolina.

Schefter stopped short of saying that Reich was on the hot seat. However, he detailed that owner David Tepper wouldn't be afraid to move on from Reich if he
deemed it necessary for the success of his team.

"One thing about David Tepper: he's not very patient," Schefter said. "He's as successful as he is because of the way he's operated his business. And it's
worked for him, basically at every level. He does what he wants. And I don't think that people are thinking that there's trouble right now with those people.
</item 21>

<item 22>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
Once a formality in Congress, the vote for House speaker has devolved into another bitter GOP showdown for the gavel.

As Republicans upset and exhausted by the infighting retreated for private conversations, hundreds of protesters, if not more, amassed outside the Capitol over
the Israel-Hamas war, a stark reminder of the dangers of having the House adrift as political challenges intensify at home and abroad.

Ahead of the morning vote, Jordan, the combative Judiciary Committee chairman, made an unusual plea for party unity — almost daring his colleagues to put
forward the alternative proposal for a temporary speaker.

“We’ve been at this two weeks,” Jordan said at the Capitol. “American people deserve to have their government functioning.”

But as the roll call got under way, he lost more than he gained, picking up three backers but adding more detractors. No further votes were scheduled.
</item 22>

<item 23>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
Others are simply upset at the way the whole process has dragged out.

They cast their ballots for McCarthy, Majority Leader Steve Scalise — who had been the party’s first nominee to replace McCarthy — and others, one vote even
going to the retired Boehner.

“Jim Jordan will be a great speaker,” Trump had said Tuesday said outside a courthouse in Manhattan, where he is facing business fraud charges. “I think he’s
going to have the votes soon, if not today, over the next day or two.”

Jordan has been a top Trump ally, particularly during the Jan. 6 Capitol attack by the former president’s backers who were trying to overturn the 2020 election
he lost to Biden. Days later, Trump awarded Jordan a Medal of Freedom.

The political climb has been steep for Jordan who is known more as a chaos agent than a skilled legislator, raising questions about how he would lead.
</item 23>

<item 24>
EU commissioner sidesteps MEPs’ questions about CSAM microtargeting (TechCrunch, 2023-10-25T21:02:44+00:00)
In a blog post ahead of today’s hearing — which deploys a crisis-PR-esque headline claim of “setting the record straight” — she criticized the article as “a
series of insinuations looking for a home”; claiming it paired an outline of “a selection of meetings I had, of events I attended, or conferences I addressed”
with “a conspiratorial tone” in an attempt “to create the impression of financial influence where there is none”.

Fotiadis was asked by the LIBE committee about the accusation that the journalists had, essentially, been spreading disinformation — and specifically whether he
believed Johansson and the Commission’s response to it amounted to a restriction on media freedom. He responded by saying he did not think that was the case.
But went on to express surprise at how the Commission had reacted to the scrutiny — to its instinct to deploy “spin-doctor” tactics to try to discredit the
article, rather than engaging with the substance of the concerns being raised.
</item 24>

<item 25>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
No further votes were scheduled.

The holdouts added to a surprisingly large and politically diverse group of 20 Republicans who had rejected Jordan’s nomination the day before, many resenting
the hardball tactics seeking to enforce support, and viewing the Ohio congressman as too extreme for a central seat of U.S. power.

With Republicans in majority control of the House, 221-212, Jordan must pick up most of his GOP foes to win. Wednesday’s tally, with 199 Republicans voting for
Jordan and 212 for Democratic leader Hakeem Jeffries of New York, left no candidate with a clear majority, as the 22 Republicans voted for someone else.

One new Jordan opponent, Rep. Vern Buchanan of Florida, explained his vote, “I think it’s time to move on.”

Bipartisan groups of lawmakers have been floating ways to operate the House by giving greater power to McHenry or another temporary speaker. The House had never
ousted its speaker before McCarthy, and McHenry could tap the temporary powers that were created after the Sept.
</item 25>

<item 26>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
Markov is “a big advocate” for all players being recognised.

“The best thing we can do is wrap our arms around them, and just let him feel what we sort of feel,” he said. “I’ve been on the other side of the fence, and it
is hard because you’re seeing all your friends succeed and sort of living the dream that you aspire to have, but I know it makes people hungrier. I know the
boys who missed out will be the ones who are training the house down, and pushing the boys who have seen the mountain top. They’re going to motivate the boys to
try and see it again.” ‘Incredibly sore’: Howe played out decider with rib injury Peter Ryan

This wasn’t the type of “car crash” injury Jeremy Howe has suffered before, but it was brutal nonetheless. The high-flying Magpie has revealed he played out the
grand final in pain after he was floored by a bone-crunching knock to the ribs from Lions forward Charlie Cameron during the last quarter.
</item 26>

<item 27>
Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? (Essentially Sports, 2023-10-30T22:30:39+00:00)
That said, the skipper market is abuzz!

Starting With The Los Angeles Angels

ADVERTISEMENT Article continues below this ad

After Joe Maddon was relieved of his managerial duties, Phil Nevin signed up as the interim manager for the remaining 106 games of 2022. Post that, he went on
to sign a one-year contract to manage the Angels in 2023, and the interim tag was lifted. Nevin’s 2024 club option was rejected by the Halos, who underwent
partial two seasons of 119-149 (.444) play under his supervision. Since this was a non-negotiable, the seat stands vacant.

The biggest challenge in this scenario is that the Angels star, Shohei Ohtani is yet to reveal which team he would be joining. In case the Angels re-sign the
two-way phenom, the manager would get to head the roster with the 2021 MVP and Mike Trout. A once in a lifetime opportunity.
</item 27>

<item 28>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
Next steps are highly uncertain as lawmakers start offering new ideas for a possible speaker. But it appears no one at present can win a GOP majority.

“We’re in a very bad place right now,” McCarthy said earlier.

In a floor vote Friday morning, Jordan’s third reach for the gavel, he lost 25 Republican colleagues, worse than he had fared earlier in the week, and far from
the majority needed, as they reject his hardline approach.

But after two failed votes, Jordan’s third attempt at the gavel essentially collapsed — in large part because more centrist Republicans are revolting over the
nominee and the hardball tactics being used to win their votes. They have been bombarded with harassing phone calls and even reported death threats.

Ahead of the vote, Jordan showed no signs of stepping aside, insisting at a Capitol press conference: “The American people are hungry for change.”

But afterward Jordan told colleagues behind closed doors, “I tried my best,” according to one of his opponents, Rep. Don Bacon, R-Neb.
</item 28>

<item 29>
The inside story of Dave Clark's tumultuous last days at Flexport (Cnbc | World Business News Leader, 2023-10-02T17:46:00+00:00)
Joe Manchin of West Virginia. He doesn't talk much about politics on social media or in interviews. Clark has donated to candidates on both sides of the aisle.
Upon his departure, The Wall Street Journal reported that he was considering running for governor of Texas, but two people familiar with his thinking say it's
not happening anytime soon. Flexport told CNBC that an employee's politics are not relevant in personnel decisions. "Ryan Petersen does not care at all about
anyone's political or personal affiliations. That is their business," the spokesperson said. "It is inappropriate for any employee to spend an excessive amount
of time during work hours on activities unrelated to their role." A person familiar with the female executive said her noncorporate endeavors were largely
related to charitable organizations. Clark has largely remained silent since he was forced to resign on Sept. 5, though in private he's expressed frustration at
how his former team was being treated by Flexport, according to people close to him.
</item 29>

<item 30>
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
</item 30>
=== END ITEM id=c9155f4b73e27e08 ===

=== ITEM id=4f54a27506bcbede ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Bankman-Fried used Caroline Ellison as a front.
  Fact 0.1: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 0.2: Caroline Ellison was the CEO of Alameda.
  Fact 0.3: Bankman-Fried had 'secret access' to customer money, both cash and crypto.
  Fact 0.4: The government claimed that Bankman-Fried had this secret access to customer money.
Sentence 1: He was the richest person in the world under 30 [and] he was going to spend it to save humanity from extinction,” he said.
  Fact 1.0: A quoted speaker said Bankman-Fried was the richest person in the world under 30.
  Fact 1.1: A quoted speaker said Bankman-Fried was going to spend his wealth to save humanity from extinction.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution portrayed Bankman-Fried as someone who committed fraud to achieve great wealth.
  Fact 2.2: The prosecution portrayed Bankman-Fried as someone who committed fraud to achieve great power.
  Fact 2.3: The prosecution portrayed Bankman-Fried as someone who committed fraud to achieve great influence.
  Fact 2.4: The defense countered that Bankman-Fried, the FTX founder, acted in good faith.
  Fact 2.5: The defense countered that Bankman-Fried, the FTX founder, never meant to commit fraud or steal.
  Fact 2.6: The defense countered that Bankman-Fried, the FTX founder, basically got in over his head.

Retrieved items:
<item 1>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 1>

<item 2>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
The one Wang said Alameda took to keep off FTX’s balance sheet? Yeah, so it was a totally innocent thing where what actually happened was that Bankman-Fried
thought it was appropriate that Alameda take the position as a backstop liquidity provider, that’s all.

In June 2022, Bankman-Fried heard about the account called “fiat@ftx” tracking how much money Alameda owed to FTX, he testified. He did not know what it was and
did not bother to find out. He was busy! That was when Bankman-Fried directed Ellison to repay Alameda’s lenders, because he thought Alameda was good for it. He
also gave BlockFi and Voyager, two crypto lenders, some capital infusions for good measure.

He was “very surprised!”

Remember that testimony Adam Yedidia gave about a conversation with Bankman-Fried in August 2022 about the enormous amount of money Alameda owed FTX? Well,
Bankman-Fried remembers it differently.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 3>

<item 4>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 4>

<item 5>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 5>

<item 6>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 7>

<item 8>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 8>

<item 9>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
How much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.

Shortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried
were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in
finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.

When Yedidia first took the stand, Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his
testimony. Why did he make such a deal with the government, she asked.

“I was concerned that I had unwittingly written code that contributed to a crime,” he said.
</item 10>

<item 11>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
When he found it, he was “very surprised!”

Besides the two obvious lies Bankman-Fried told on the stand — about Alameda Research’s name and about “allow_negative” — I have been struck by how little he
seems to know about his own companies. Apparently, Singh, Wang, and Ellison were out there just doing whatever their little hearts desired. Because
Bankman-Fried was a CEO, but definitely not the kind that pays any attention to money at his crypto trading firm and futures exchange.
</item 11>

<item 12>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 12>

<item 13>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 13>

<item 14>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.
</item 14>

<item 15>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 15>

<item 16>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 16>

<item 17>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 17>

<item 18>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 18>

<item 19>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
Prosecutor Danielle Sassoon had run an effective direct examination, creating an easy-to-follow narrative. By contrast, Cohen appeared to be bumbling around,
taking up one topic only to abruptly pivot. Sure, we’re still in the prosecution’s case, but Cohen had all night to prepare his lines of questioning.

Apparently, Alameda had a problem with retaining accountants

We established that Bankman-Fried had a much larger appetite for risk than Ellison. I thought perhaps it might be building to something, but this line of
questioning was quickly dropped. We established that Bankman-Fried and Ellison reacted differently to stress, and that they also had different approaches to
media: namely, that Ellison avoided it while Bankman-Fried sought it out. Okay?

We discovered that there was one accountant at Alameda in 2021, and two more junior accountants were hired in 2022. Apparently, Alameda had a problem with
retaining accountants, which didn’t surprise me much; CEOs generally don’t do the balance sheets for their companies.
</item 19>

<item 20>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
“He hadn’t known how to tell me,” she said.

Ellison was also, of course, in a more personal relationship with Bankman-Fried. (A juror who’d been asleep for a discussion of the FTT token woke up when she
started discussing it.) The two started sleeping together in the fall of 2018, on and off. At the time, she was a trader and Bankman-Fried was the CEO. They
didn’t date until later — twice. Their first relationship stretched from the summer of 2020 through the summer of 2021; they agreed to keep it secret. (Some
people found out, as they usually do.) The second time, from the fall of 2021 until the spring of 2022, they lived together.

That gave Ellison an unusual view of his character. “He was very ambitious,” she said.
</item 20>

<item 21>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.
</item 21>

<item 22>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That was the basis of the worst of her testimony — and the spreadsheet from hell.

Ellison said she’d prepared the spreadsheet at Bankman-Fried’s request in the fall of 2021 and shared it with him. The point was risk analysis around paying
back Alameda’s loans if they were abruptly recalled by Genesis, their lender. Bankman-Fried wanted to use $3 billion for venture investments, so Ellison was
ballparking what that would do to Alameda’s risk. In the as-is scenario, if things went south, she figured there was a 30 percent chance they wouldn’t be able
to meet the loan recalls. If Bankman-Fried used $3 billion to make investments, there was a 100 percent chance they couldn’t meet the recalls, even with FTX
customer funds.

Echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length

The problem here wasn’t really the math, which seemed pretty arbitrary.
</item 22>

<item 23>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
I didn’t want to call it Sam’s Crypto Trading Firm or anything like that. We — there are a lot of competitors and people who we didn’t particularly want to know
what we were building out because they would race to do it. “Research” was a sort of generic word, which filled out the company name. And that was — it was far
better than the internal name that we had at that point, which was Wireless Mouse.

I would find this much more believable if I hadn’t already watched a video of Bankman-Fried explaining on a podcast that the name made it easier to get a bank
account. That happened during the first day of Gary Wang’s testimony. Bankman-Fried was there, too. You know who else was there? The jury.

Anyway, Bankman-Fried went on a hiring spree for Alameda. He rounded up his merry gang of alleged co-conspirators. First, Wang, to program the computers. Then,
Nishad Singh, about a month after founding Alameda Research.
</item 23>

<item 24>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 24>

<item 25>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 25>

<item 26>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 26>

<item 27>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
I’ve got some shitty ex-boyfriends, but none of them made me the CEO of their sin-eater hedge fund while refusing to give me equity and bragging about how there
was a 5 percent chance they’d become the president of the United States, you know? Absolutely counting my blessings after Caroline Ellison’s first day on the
stand. I wonder how many of the nine women on the jury are doing the same.

Ellison was the head of Alameda Research, the aforementioned hedge fund, during the implosion of it and FTX. She’s already pleaded guilty to criminal charges
stemming from one of the worst romantic relationships I’ve ever heard of, and her testimony was widely anticipated before the trial. Today, that took the form
of discussing a damning spreadsheet — one she prepared for her ex and boss Sam Bankman-Fried, now the defendant in a criminal fraud trial.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy.
</item 27>

<item 28>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 28>

<item 29>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 29>

<item 30>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”
</item 30>
=== END ITEM id=4f54a27506bcbede ===

=== ITEM id=e5ffbf32685be232 ===
Gold sentences and facts:
Sentence 0: Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense, especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.
  Fact 0.0: Todd Boehly is American.
  Fact 0.1: Todd Boehly is a co-owner of Chelsea.
  Fact 0.2: Expanding Chelsea's profile and reach in the U.S. would make sense under Todd Boehly's co-ownership.
  Fact 0.3: Catarina Macario is a USWNT international.
  Fact 0.4: Mia Fishel is a USWNT international.
  Fact 0.5: Catarina Macario plays her club football in the U.S.
  Fact 0.6: Mia Fishel plays her club football in the U.S.
  Fact 0.7: CBS Sports holds WSL rights.
Sentence 1: It was 24 September, Chelsea’s second home Premier League defeat on the spin after the one against Nottingham Forest and they were booed off again.
  Fact 1.0: The date was 24 September.
  Fact 1.1: Chelsea suffered a home Premier League defeat on 24 September.
  Fact 1.2: The 24 September defeat was Chelsea's second home Premier League defeat in a row.
  Fact 1.3: Chelsea's home Premier League defeat before the 24 September defeat was against Nottingham Forest.
  Fact 1.4: Chelsea were booed off again after the 24 September defeat.
Sentence 2: Late checks will be made on the fitness of Reece James, Axel Disasi and Nicolas Jackson, while Benoit Badiashile could return from a long-term injury.
  Fact 2.0: Late fitness checks will be made on Reece James.
  Fact 2.1: Late fitness checks will be made on Axel Disasi.
  Fact 2.2: Late fitness checks will be made on Nicolas Jackson.
  Fact 2.3: Benoit Badiashile could return from a long-term injury.

Retrieved items:
<item 1>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

On another night, perhaps Chelsea score from one of these counterattacks. Perhaps in another game United come up against a better finisher than Nicolas Jackson,
who for all his talent looks like a man playing football in a hall of mirrors. This is the high-wire act that Ten Hag is enacting: a daring gamble that United
will gain more going forward than they lose going back.

That once Lisandro Martínez and Raphaël Varane come back, once he brings the defensive line up 10 yards, once Rasmus Højlund hits some form, once Mason Mount
gets up to speed, lawless abandon will begin to resemble a sustainable attacking blueprint.
</item 1>

<item 2>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
This began the race to confirm Hayes had been selected as the USWNT head coach.

Talks between U.S. Soccer and Hayes’ representatives continued even after Chelsea’s press release. The federation’s board convened late on Saturday, Nov. 4 to
approve the selection, even without the final details of the contract settled or signed.

At the end of it all, the sides have agreed to a deal that will make Hayes the highest-paid women’s football coach in the world — though her salary is not tied
to equal compensation with Berhalter. While her salary is in the same range as the USMNT head coach, it’s thought to be a reflection of the market value for
Hayes. With reports that Chelsea was prepared to quadruple her salary to keep her, Hayes herself danced around the details in her first media availability with
the club.

“I believe in private conversations,” she said. “Of course, I’m disappointed to hear things being said in the press.
</item 2>

<item 3>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.
</item 3>

<item 4>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea’s general manager Paul Green will stay at the club. It is unclear whether any other members of Chelsea’s technical staff are expected to depart.

GO DEEPER What next for Chelsea after Emma Hayes?

The relentless nature of managing a club takes its toll and after what will be 12 years at Chelsea come the end of the season, Hayes, a single parent whose
father died last month, decided that it was the right time for a change, professionally and personally.

Advertisement

“The biggest factors are my son, leaving at the top and giving the club enough time to be able to transition without there being too much disruption,” she said
in her press conference.

The club has received several applications regarding Hayes’ replacement but has not yet started an official recruitment process. She will meet with Chelsea’s
technical directors once a week to create a succession plan and will have a say in who takes the job after her.
</item 4>

<item 5>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s own messaging. Crocker, in a September meeting with U.S. reporters along with Cone and U.S. Soccer CEO J.T. Batson, said he hoped to have a new head
coach in place by December.

But the initial contact with Hayes was made a couple of months ago, early into the search, with all three top-level executives from U.S. Soccer involved in
those talks. The trio also described the interview process to journalists in that September meeting – a U.S. soccer statement describes it as involving
“psychometrics and abstract reasoning tests, in-depth discussions of strategy, coaching philosophy and the current player pool, as well as evaluation on the
reactions to pressure, culture-building and interactions with players and staff.” USMNT coach Gregg Berhalter went through a similar process, including an
abstract reasoning test, when he was re-hired by Crocker earlier in 2023.
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.
</item 7>

<item 8>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Sporting director Matt Crocker made the final decision to hire Hayes after a search process that began in August, following the team’s surprising exit in the
round of 16 at the World Cup and the subsequent departure of head coach Vlatko Andonovski.

“She has tremendous energy and an insatiable will to win,” U.S. Soccer president Cindy Parlow Cone said in a statement. “Her experience in the USA, her
understanding of our soccer landscape and her appreciation of what it means to coach this team makes her a natural fit for this role and we could not be more
pleased to have her leading our women’s national team forward.”

Though Hayes is seen as one of the world’s top coaches in women’s soccer, the appointment still comes as something of a surprise. Here’s how the deal got done.

Details of the deal

At least part of the surprise surrounding Hayes’ hire – and the six-month runway before she officially takes charge – is down to U.S. Soccer’s own messaging.
</item 8>

<item 9>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 9>

<item 10>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 10>

<item 11>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 11>

<item 12>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 12>

<item 13>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
This was a win that felt a lot tighter than it really was: United inexplicably hanging on at the end despite enjoying 28 shots and an xG tally of 4.07. For long
periods United played some of their most authentically thrilling football of the season, a doctrine of relentless attack that had Diogo Dalot flying forward
from right-back even as the minutes ticked down.

It was also a game that felt more chaotic than it really was. In fact the slippery, slingy nature of the play was largely by design: United hollowing out their
midfield in order to throw bodies forward, Chelsea hollowing out theirs in order to leave men up for the breakaway. It was a game that hung on United’s
determination to keep allowing Chelsea counterattacks, and Chelsea’s determination to keep messing them up. It was a game, above all, that hung on the curious
main character energy of Scott McTominay.
</item 13>

<item 14>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
She will not work with the U.S. in international windows.

“I’m here until the end,” she said in her press conference on Friday. “I haven’t died, I haven’t gone anywhere. I’m here, doing this job. My full focus and
attention is on what I do for Chelsea.”

Advertisement

Hayes could be tied up with Chelsea as late as May 25 if the London club makes the Champions League final; notably, an international window that would
theoretically be Hayes’ first in charge begins just two days later, on May 27.

However, there are ways in which the arrangement will benefit U.S. Soccer. The federation won’t owe any compensation to Chelsea, and Hayes will be fully
committed to the program, with a move to Chicago in the works for next year following the completion of her time in London. Eventually, she’s expected to
relocate to Atlanta thanks to U.S. Soccer’s planned combined headquarters and national training facility in Georgia.
</item 14>

<item 15>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
But he also pops up in the channels, on the wing, either supporting the man in possession or offering himself for the quick switch of play.

Great times for a great guy. But, of course, if you commit a midfielder to this role, then you leave gaps elsewhere. One of the reasons Chelsea found it so easy
to play through United was that, with the forwards pushed high and McTominay as the secret sauce in the final third, there were open spaces every time they
played through the press. Chelsea’s goal was a case in point: two simple vertical passes straight through the centre via Mykhailo Mudryk to Cole Palmer, passes
that McTominay could see coming but could do precious little to stop.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 15>

<item 16>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
The specifics of how that will work without Hayes’ involvement remain a mystery. It would be understandable for players to feel like they are still auditioning
for an absent director until May rolls around, while still knowing they must perform at the USWNT standard.

Along these same lines, there is at least the suggestion that the federation could be willing to sacrifice coherent preparation for this upcoming Olympic
tournament to focus more on the longer-term project of the 2027 World Cup. That itself represents a marked change from the expectations and pressure of constant
performance and winning that the team is known for.

Is that a good or a bad thing? Perhaps a little bit of both. The USWNT shouldn’t be ruled out of contention for the Olympics by any stretch, but this past
summer’s World Cup did reveal that the problems facing the team are far more foundational than just poor coaching decisions or the strange midfield chemistry.
Balancing realism with the pressure to win feels like a much more sustainable path forward for the USWNT.
</item 16>

<item 17>
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
</item 17>

<item 18>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 18>

<item 19>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 19>

<item 20>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
That is an understatement. And if there seemed a risk involved in swapping the stability of City for the seeming madhouse of Stamford Bridge, there was evidence
of more cool-headed thinking from a 21-year-old. Chelsea have been manic buyers but he said: “I did look at how many players were there but I don’t think people
realise how many players they actually got rid off, as well.”

The departures of, among others, Mason Mount, Kai Havertz, Christian Pulisic and Joao Felix, meant there could be opportunities in the attacking midfield
positions. “I looked at the squad and thought: ‘If I go there and know what I can do then I’ll have the chance of playing’,” he said. “And thankfully it’s
happened.” If there seemed plenty of traffic in the queue for places in Southgate’s attack, Palmer seems to have accelerated past several rivals. Now an England
debut could follow in the next week.
</item 20>
=== END ITEM id=e5ffbf32685be232 ===
