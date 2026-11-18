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

=== ITEM id=3d3bfd4a924c9667 ===
Gold sentences and facts:
Sentence 0: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 0.0: Taylor Swift has the Eras Tour booked all through next year.
  Fact 0.1: Taylor Swift would rather die than cancel a show.
Sentence 1: Love Story This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No.
  Fact 1.0: The text has a section titled Love Story.
  Fact 1.1: The text says this finally brings the story to the fated sighting of Taylor Swift at Arrowhead Stadium on Monday.
  Fact 1.2: Taylor Swift was cheering on No. at Arrowhead Stadium on Monday.
Sentence 2: The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before.
  Fact 2.0: The game is taking place at Arrowhead Stadium.
  Fact 2.1: Taylor Swift has performed at Arrowhead Stadium herself.
  Fact 2.2: Taylor Swift has attended a game at Arrowhead Stadium before.
Sentence 3: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 3.0: Taylor Swift has shared that she has nothing to hide.
  Fact 3.1: Taylor Swift has nothing to hide in her new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Head coach Andy Reid was also asked about his star tight end's supposed relationship, to which he quipped, "I set them up."

Sept. 24: Kelce and Swift leave stadium together

"You know when it's time to go." - It's Time To Go

This was the first time the two were photographed together. They walked side by side, Kelce fittingly wearing an outfit with "1989" in the title and Swift
wearing a Chiefs zip-up.

Sept. 24: The two leave in a convertible

"Nothing good starts in the getaway car." - Getaway Car

Swift and Swift were spotted in a convertible, with the tight end driving, after the Chiefs' win. Kelce reportedly rented out a restaurant for a private party
with Swift and his teammates.

"Taylor Swift likes that Travis Kelce 'pursued her' and the two are having a 'great time getting to know each other,'" a source told Entertainment Tonight.
</item 2>

<item 3>
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
Oct. 15: Kelce, Swift make 'SNL' cameos

The two were spotted out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift
was getting during games, while the singer introduced the musical act of the night, Ice Spice.

Oct. 22: Swift attends fifth Chiefs game

Swift has now attended games broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and
will be shown on CBS. She is sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes' wife, Brittany.
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
How do you not get that? That's called images right there."

Oct. 1: Swift attends Chiefs-Jets

"Welcome to New York, it's been waiting for you," - Welcome to New York

OK, so the Jets technically play in New Jersey, but we're just basing this off their name. Swift went to her second consecutive Chiefs game, showing up with
some famous friends, Blake Lively, Ryan Reynolds and Hugh Jackman. In the week leading up to the game, there were reports that the singer would be at the game.
The Chiefs defeated the Jets 23-20.

Oct. 5: NFL defends itself

Swift coverage is everywhere and the league even changed their social media bio to one of the singer's lyrics. The league's X (formerly Twitter) bio to read,
"NFL (Taylor's Version)," while their Instagram account read, "Chiefs are 2-0 as Swifties."
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Well I would say that Travis Kelce's had a lot of big catches in his career. This would be the biggest."

Sept. 26: Chiefs games get even more popular

"Welcome to New York (*New Jersey), it's been waiting for you." - Welcome to New York

The defending Super Bowl champions are already one of the more popular teams in the league, but it is becoming the hottest ticket in town, no matter what town
that is.

"Since Swift's appearance at the Chiefs game on Sunday, the Jets have sold more tickets for their game against the Chiefs in a single day than since the season
started -- by more than double," a representatives at StubHub told to Town & Country magazine.

Sept. 27: Mahomes confirms he met Swift

"It was enchanting to meet you, all I can say is I was enchanted to meet you."
</item 7>

<item 8>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“Well I know for DAMN SURE that s*** about ‘having kids within the next year’ is NOT coming from Taylor’s side,” one claimed. “She’s got the Eras Tour booked
all through next year, and she’d rather die than cancel a show. But we do know who’s been public about looking for a ‘BREEDER.’”

The comments about Travis and Swift’s romance came days after they seemingly confirmed their relationship. On 22 October, Swift made her return to Arrowhead
Stadium to watch as the Kansas City Chiefs won their game against the Los Angeles Chargers. The occasion marked Swift’s fourth time attending one of Kelce’s
games, amid ongoing reports that they are dating.

Following the event, Chariah Gordon, the partner of Kelce’s teammate Mecole Hardman Jr, took to Instagram to share a snap of the singer and NFL player’s sweet
PDA moment. In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=3d3bfd4a924c9667 ===

=== ITEM id=aa7c96ddfb78ecd5 ===
Gold sentences and facts:
Sentence 0: These delays may lead to bets being refunded depending on the circumstances and the specific rules of your sportsbook.
  Fact 0.0: Delays may lead to bets being refunded.
  Fact 0.1: Whether bets are refunded depends on the circumstances.
  Fact 0.2: Whether bets are refunded depends on the specific rules of the sportsbook.
Sentence 1: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move.
  Fact 1.0: Every sportsbook adjusts when a betting line moves for a legitimate reason.
  Fact 1.1: Every sportsbook adjusts when odds change for a legitimate reason.
  Fact 1.2: An injury is an example of a legitimate reason for a line or odds change.
  Fact 1.3: A roster move is an example of a legitimate reason for a line or odds change.
Sentence 2: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 2.0: Sportsbooks are also gathering this information.
  Fact 2.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.

Retrieved items:
<item 1>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
Wagers include extra innings unless specified otherwise.

The game must start on the scheduled day (local stadium time) for bets to have action.

The game must go 9 innings or 8½ if the home team is winning for the run line to have action.

On ‘team to score first’ wagers, once a run is scored, the prop will be graded, regardless of how long the game lasts.

Wagers on postseason baseball series are based on the team to win the series, regardless of the number of games involved.

For player props, the player must be in the starting lineup for wagers to have action.

It’s important to read through the rules for baseball betting before placing your MLB bets, and you can find them in the terms and conditions section of your
sportsbook’s website.

Related Pages:
</item 1>

<item 2>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Do point spread odds change?

Yes, point spread odds can change, and these shifts are commonly referred to as "line movement."

Lines move when the point spread or the odds associated with a particular bet change after sportsbooks have initially set them. There are several reasons for
these adjustments.

One of the most common causes is an imbalance in the amount of money being wagered on one side of the bet. If a significant majority of bettors are placing
their money on one team to cover the spread, the top sports betting apps will adjust the point spread to make the other side more appealing, thereby balancing
the betting action.

Another factor that can affect the spread odds is the occurrence of unexpected events, such as injuries to key players, changes in team strategies, or even
significant weather events that might impact a game's outcome.

For instance, if a star quarterback is announced as injured just days before a big game, the point spread might shift in favor of the opposing team.

Moreover, information and public perception play a role.
</item 2>

<item 3>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Moreover, information and public perception play a role. If a piece of news or a prevailing sentiment starts to gain traction about a particular team's
increased chances of winning, this can influence how people bet, leading sportsbooks to adjust their lines accordingly.

Point spread odds can and do change based on a combination of actual betting patterns, unexpected events, and shifts in public perception. Savvy bettors keep a
close eye on these line moves, as they can offer insights into where the value lies in a particular bet.
</item 3>

<item 4>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
By doing this research, you’ll find the best odds for the bets you were already planning on making, which means you can maximize your potential return on your
wager.

This can be a time-consuming process, but it also helps you boost your bankroll and give yourself the largest possible profit.

MLB betting rules at online sportsbooks

Every sportsbook has terms and conditions for all bets, MLB included. While those rules may be specific to each sportsbook, there are some general rules that
should apply everywhere.

For example, MLB First 5 Innings bets are usually considered actionable after five innings of play, or 4.5 innings in the case of the home team having the lead.

Another rule many sportsbooks follow for baseball bets is weather-related, such as if a game is suspended due to rain or other weather conditions. These delays
may lead to bets being refunded depending on the circumstances and the specific rules of your sportsbook.

Here are some other rules found at online sportsbooks for MLB betting.

Wagers include extra innings unless specified otherwise.
</item 4>

<item 5>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 5>

<item 6>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on one side might indicate that the
initial odds were potentially "off," prompting an adjustment. Event Dynamics: For live betting, where bets are placed while the event is in progress, the
evolving dynamics of the match can lead to frequent changes in the odds.

Sportsbooks adjust moneyline odds to ensure they remain profitable, balance their book, and reflect the most current information about the event.

Related Pages:
</item 6>

<item 7>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
What makes NBA betting special or unique?

Basketball odds are known to fluctuate more than those of other sports. The factors that play into those lines shifting can range from having only a few impact
players per team to the sheer frequency of games.

By following a single team, you can track its momentum swings, which players tend to get on hot or cold streaks, or how well they play at home vs. on the road.
You can use that information to your advantage for both pre-game and live wagers.

Similar categories of moneylines, spreads, totals, and NBA prop bets should appear across both formats.

The NBA is also special because of the number of bets available for every game, ranging from pre-game to live betting and parlays.

When you add up the standard bets, game props, player props, in-game betting, and more, there’s a staggering volume of bets for every NBA game at online
sportsbooks.
</item 7>

<item 8>
Totals Betting - What is an Over/Under Bet? (Sporting News, 2023-11-01T22:08:31+00:00)
Bookmakers adjust the line to balance the action and protect themselves from potential losses.

One of the most common reasons for line movement is a significant amount of money being wagered on one side (either over or under). Bookmakers adjust the line
to balance the action and protect themselves from potential losses. Injury Updates: If a key player is injured or unable to play, this can influence the
expected total score, leading bookmakers to adjust the line accordingly.

If a key player is injured or unable to play, this can influence the expected total score, leading bookmakers to adjust the line accordingly. Weather
Conditions: In outdoor sports, weather can play a significant role. For example, heavy rain or snow in a football game might lead to expectations of a
lower-scoring game, causing the line to move down.

In outdoor sports, weather can play a significant role. For example, heavy rain or snow in a football game might lead to expectations of a lower-scoring game,
causing the line to move down.
</item 8>

<item 9>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.
</item 9>

<item 10>
MLB Betting Sites & Apps - The Best Baseball Sportsbooks 2023 (Sporting News, 2023-10-24T22:32:27+00:00)
Betting apps offer real-time updates on all odds, which allows you to react to game flow as it’s happening.

Your MLB betting app can help you track shifting odds and facilitate live betting. Betting apps offer real-time updates on all odds, which allows you to react
to game flow as it’s happening. Bet from within legal states: As long as you are in a state that has legal online sports betting, you can use your app to make
bets, track your account, add or withdraw funds, and more. Remember that you can only place bets when you are in the state lines, but you can access your
account and monitor wagers from anywhere you have internet access.

MLB live betting

The best sports betting apps offer live, in-game betting for Major League Baseball.

Live betting means you can bet on a game in progress, right down to what the next batter will do with the next pitch or what team will win the inning.
</item 10>
=== END ITEM id=aa7c96ddfb78ecd5 ===

=== ITEM id=c3230ec1e823e874 ===
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
</item 1>

<item 2>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 2>

<item 3>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 3>

<item 4>
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
</item 4>

<item 5>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 5>

<item 6>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 6>

<item 7>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 7>

<item 8>
Gregg Berhalter named a 23-player USMNT squad for October friendlies: Germany in Hartford on Oct. 14, Julian Nagelsmann’s first match as coach, and Ghana in
Nashville on Oct. 17. Gio Reyna returned after injury and his 2022 World Cup fallout with Berhalter; Lennard Maloney earned his first senior call-up. Tyler
Adams and Antonee Robinson were unavailable through injury, while Malik Tillman withdrew and Alex Zende
</item 8>

<item 9>
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
</item 9>

<item 10>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.
</item 10>
=== END ITEM id=c3230ec1e823e874 ===

=== ITEM id=05e55db3fa0e2f5b ===
Gold sentences and facts:
Sentence 0: Because FTX had grown so much, he could no longer run both companies, he said.
  Fact 0.0: FTX had grown so much.
  Fact 0.1: Bankman-Fried said he could no longer run both companies because FTX had grown so much.
Sentence 1: It was Bankman-Fried’s decision, she said, as he was the CEO of FTX.
  Fact 1.0: A woman said the decision was Bankman-Fried's.
  Fact 1.1: A woman said the decision was Bankman-Fried's because he was the CEO of FTX.
  Fact 1.2: Bankman-Fried was the CEO of FTX.
Sentence 2: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 2.0: Bankman-Fried, known as SBF, has pleaded not guilty.
  Fact 2.1: SBF has pleaded not guilty to some seven charges.
  Fact 2.2: The seven charges against SBF are charges of fraud and conspiracy.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 3.1: The defense argued that the FTX founder, Bankman-Fried, acted in good faith.
  Fact 3.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 3.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
The Verge’s courtroom coverage portrays Sam Bankman-Fried’s defense as increasingly damaging. On the stand, he struggled to answer questions, admitted failing
to create a risk team or safeguards against Alameda spending FTX customer funds, and said he did not investigate an $8 billion shortfall despite warnings.
Prosecutors highlighted testimony from Caroline Ellison and others tying him to customer-fund misuse and deception; defense cross-examination was described as
ineffective. Bankman-Fried was ultimately convicted on seven counts, including
</item 1>

<item 2>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 3>

<item 4>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 4>

<item 5>
The Verge’s trial coverage portrays FTX as an alleged fraud from its founding: prosecutors said Bankman-Fried secretly gave Alameda unlimited access to customer
funds while publicly denying preferential treatment. Cooperating insiders Wang, Singh, Ellison, and Yedidia described code privileges, doctored accounts, and
misuse of deposits; expert testimony traced billions to loans, investments, real estate, insiders, and political causes. The collapse devastated customers and
Bankman-Fried’s friends and family, many now
</item 5>

<item 6>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 6>

<item 7>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 7>

<item 8>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 8>

<item 9>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=05e55db3fa0e2f5b ===

=== ITEM id=a6a3837bd8daff73 ===
Gold sentences and facts:
Sentence 0: While Israel has granted permits to about 17,000 Gaza residents to enter and work in Israel, the food, fuel and medical supplies that people in Gaza use all first pass through Israel.
  Fact 0.0: Israel has granted permits to about 17,000 Gaza residents to enter and work in Israel.
  Fact 0.1: The food, fuel and medical supplies that people in Gaza use all first pass through Israel.
Sentence 1: The understandings that Israel reached with Hamas in 2012 were made "to stop the fighting", Ortal says, and to get life back to normal as quickly as possible.
  Fact 1.0: Israel reached understandings with Hamas in 2012.
  Fact 1.1: Ortal says the understandings were made "to stop the fighting".
  Fact 1.2: Ortal says the understandings were made to get life back to normal as quickly as possible.
Sentence 2: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 2.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.

Retrieved items:
<item 1>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
With them and with the US, we established a kind of task force, which is not subject to any country or unifying body, countries that are working in
coordination, and meet every few days to exchange intelligence, and advance the struggle."

In general, following an appeal by IMPA to its counterparts around the world, there has been a dramatic increase in intelligence coming to Israel from abroad.
"The amount of information doubled in the month of fighting, compared with the monthly average. Before the war, we received about 24 intelligence reports a
month from parallel authorities, compared with 50 today."

Countries around the world cooperating with Israel to fight terrorism sounds almost ideal, but it’s hard not to wonder where they were before.

"Many countries felt this had nothing to do with them, that it wasn’t their struggle.
</item 1>

<item 2>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
However, IMPA’s goal, Ostrovitch-Levi explains, is not only to stop the individual raising of funds for terror financing, but the entire interrelated chain,
from the fund-raiser to the transferrer, and on through to the recipient. "We have the ability to build the intelligence scenario, synchronize all the sources
of information received, and delineate the money trail from abroad to Hamas. We’re interested not only in who is running the campaign, the association or
entity, but where the money is coming from, and where it’s going. We need to stop all of the elements in this network."

Ever since October 7, everyone has been discussing and analyzing the question of about the intelligence agencies, and why there was no warning. Where was IMPA
before?

"Hamas collects lots of money all the time, and the security agencies deter money transfers. There was no sudden increase in the raising of billions, and you
must also remember that most of the money is raised globally, not in Israel.
</item 2>

<item 3>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
Dozens of our staff have been called up to reserve duty, in large part managers working regularly in the fight against terror financing."

Since then, most of IMPA’s work has focused on identifying and blocking the terror money pipelines. "We work very reactively - we see a fundraising campaign,
try to deter it, and stop it from happening again. There are no breaks, because they are constantly active. It’s a daily struggle. Our analysts have been
sitting on the social networks from day two of the war, just manually monitoring visible sources of information on these networks to locate fundraising
campaigns for terrorist organizations operating in Gaza."

Three days after the war began, IMPA also issued a call to all global regulated entities for increased vigilance towards terrorism financing campaigns in
response to the ongoing war and state of emergency, and to report all activities that raise suspicion of terror support and terror financing, in an effective
and immediate manner.
</item 3>

<item 4>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
While Israel has granted permits to about 17,000 Gaza residents to enter and work in Israel, the food, fuel and medical supplies that people in Gaza use all
first pass through Israel.

Israel controls two physical checkpoints along Gaza, which monitor both the entry and exit of people and trucks. Israel limits the kind and quantity of
materials that pass into Gaza. And the blockades generally prohibit Gazans who do not have work permits or special clearance – for medical purposes, for example
– from entering Israel.

Israel’s restrictions through the blockade intensified since Hamas’ surprise attack on 20 Israeli towns and several military bases on Oct. 7, with Israel then
announcing a broad blockade of imports into Gaza. This stopped all food, fuel and medical supplies from entering the region.
</item 4>

<item 5>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
RELATED ARTICLES Tech experts battle to shut down Gaza financial pipeline

"Overnight, we started receiving hundreds of reports about unusual activity by customers of financial institutions in the State of Israel," she says. Before
October 7, IMPA received about 100 reports a day dealing with terror financing. That number jumped to about 1,000. "There is a 900% increase in reports on
terror financing, and we don't waste time, everything happens very, very quickly. As we receive the information, we simultaneously pass it on to the operational
professionals who locate the money pipelines, and block them. Some information will not necessarily lead to terror financing, but we don’t have time to waste.
Bank CEOs call and draw our attention to the information they convey to us that warrants special attention."

Has information from banks and financial institutions led to blocking the money pipeline for terrorism?

"There was very high-quality intelligence that we passed on to the security agencies. We don't always know the results."
</item 5>

<item 6>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
We don't always know the results."

"Gaza could be more beautiful than Dubai "

The network to block terror financing funds includes many entities, including private citizens. "The world's financial intelligence authorities are now at the
heart of this struggle, and we’re working together to thwart the financing channels that we’ve located. In addition, we’ve been approached by many experts from
the private sector; people with intelligence experience because they’ve worked in these organizations, or have high-tech experience, and have mobilized like
crazy. Israeli entrepreneurs in Silicon Valley have contacted us; they write code, search for information, and give us lots of information. We aggregate and
examine everything."

IMPA is charged with rooting out the sources of funding for terrorism, working together with the National Bureau for Counter Terror Financing of Israel (NBCTF)
in the Ministry of Defense, which is the coordinating body of all activities to thwart terror financing, the Shin Bet, Military Intelligence Directorate (AMAN),
and all other relevant intelligence and security agencies.
</item 6>

<item 7>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
In an article five years ago, Ortal, currently in active military service an unavailable for interview, wrote, "Israeli restraint as regards proactive thwarting
of offensive capabilities in the Gaza Strip stems from a clear Israeli strategy that strives to reduce the influence of the enemy on the current routine of life
in Israel, even at the price of worsening of the threat in the future." Ortal pointed out that this policy was natural and that "a Western nation cannot lead
its life in an endless state of emergency. But alongside the understanding of the need for a containment policy, it must be recognized that the risks are high,
as evidenced by the steeply worsening intensity of conflicts between Israel and the terrorist organizations in Gaza."

According to Ortal, in recent decades, Hamas and Hezbollah developed ballistic missiles and terror tunnels in response to Israeli policy that sought minimal
friction with the enemy through separation and withdrawal from the security zone in southern Lebanon, and the Gaza Strip.
</item 7>

<item 8>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
This is a real-time deterrence," reveals IMPA director general Adv. Ilit Ostrovitch-Levi. "And then there was a domino effect. Following this blocking, more
countries began to look around, and blocked funds intended for terror. All this information, which is now shared at an international level, leads to the closing
of more and more Hamas funding pipelines, in real time. With worldwide cooperation, Israel’s security and enforcement authorities are thwarting the transfer of
millions of dollars earmarked for terrorism, every day."

Headed by Ostrovitch-Levi since May 2022, IMPA is an intelligence body that conducts the most complex of international economic investigations, and fights
terror financing routinely. On October 7, Ostrovitch-Levi received a painful reminder of the vital importance of her work. "Already on that Saturday night, with
Israeli under heavy rocket fire, we talked about preparing ourselves for the campaign we’re now entering, the fight against terror financing channels.
</item 8>

<item 9>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Ortal points out that between Operation Pillar of Defense in 2012 and Operation Protective Edge in 2014, Israel significantly reduced targeted countermeasures
against terrorist operatives, and none focused on Hamas operatives. According to his analysis, this was because, prior to the Iron Dome coming into use in 2011,
the most dangerous threat posed by Gaza developed in the period preceding Operation Protective Edge. The understandings that Israel reached with Hamas in 2012
were made "to stop the fighting", Ortal says, and to get life back to normal as quickly as possible. This, after air raid sirens sounded in Tel Aviv and
Jerusalem for the first time since the Yom Kippur War, and the education system shut down for more than a week.

Even after Iron Dome, Israel found it difficult to stand up against the significant disruption of daily life on the home front.
</item 9>

<item 10>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
They’ve given us valuable information that we’ve passed on to the security agencies, about organizations they didn’t necessarily know about, and people backing
them, all in real time. At the same time, they’ve issued alerts to increase vigilance among the US public about terror financing."

On October 18, the US Treasury Department's Office of Foreign Assets Control (OFAC) announced that it had imposed sanctions on 10 key Hamas operatives and
financial facilitators in Gaza and elsewhere including Sudan, Turkey, Algeria, and Qatar. The statement read, "This action targets members managing assets in a
secret Hamas investment portfolio, a Qatar-based financial facilitator with close ties to the Iranian regime, a key Hamas commander, and a Gaza-based virtual
currency exchange and its operator… To date, the Treasury has targeted nearly 1,000 individuals and entities connected to terrorism and terrorist financing by
the Iranian regime and its proxies, including Hamas, Hezbollah, and other Iran-aligned terrorist groups in the region."
</item 10>
=== END ITEM id=a6a3837bd8daff73 ===

=== ITEM id=b75a66b2d4bcf13a ===
Gold sentences and facts:
Sentence 0: Not only has he produced at a high level, but his play has been instrumental in the Vikings turning things around after a 1-4 start.
  Fact 0.0: He has produced at a high level.
  Fact 0.1: His play has been instrumental in the Vikings turning things around.
  Fact 0.2: The Vikings had a 1-4 start.
Sentence 1: But they've also seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero completions of 20-plus yards from tight ends).
  Fact 1.0: The team under discussion has seen the fourth-lowest ADOT from tight ends.
  Fact 1.1: The ADOT from tight ends is 5.0 yards per target.
  Fact 1.2: Teams seem to know they will not be able to succeed on downfield shots against the team under discussion.
  Fact 1.3: The team under discussion has had zero completions of 20-plus yards from tight ends.

Retrieved items:
<item 1>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
And Edwards follows it up with the Bears 3rd INT of the game ‼️



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/2d4C1s2O09 — NFL (@NFL) November 28, 2023

10:06 p.m. — After a hold by Jaylon Johnson gifts the Vikings a first down, they still can't capitalize. Ty Chandler can't make T.J. Edwards miss on third down,
so Minnesota will go for it on fourth down.

10:02 p.m. — The running game is working, the passing game is not for the Vikings. Mattison has 19 yards three plays into this drive for Minnesota.

Bears 6, Vikings 3

9:58 p.m.
</item 1>

<item 2>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
No one has scored a touchdown yet tonight.

10:40 p.m. — Addison's catch stands, and an Alexander Mattison run has the Vikings at the edge of field goal range. They want more than three points, but it's
early enough that they absolutely could settle for a field goal if it got to that point.

10:37 p.m. — Matt Eberflus throws the challenge flag on a nice grab by Addison to give the Vikings a first down. This one looks like it will be tough to
overturn...

10:34 p.m. — Wow, Jordan Addison would've had a walk-in touchdown, but Dobbs' throw takes him out of bounds. Addison couldn't have been more wide open.

10:33 p.m. — Brandon Powell has been the Vikings' biggest playmaker at two key points tonight. He goes 18 yards here to put Minnesota near midfield.

10:29 p.m.
</item 2>

<item 3>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
8:00 p.m. — Will the Josh Dobbs show continue tonight? He’s had good command of the Vikings’ offense in three games since a trade out of Arizona, flashing an
especially strong connection with T.J. Hockenson.

7:45 p.m. — It’s 15 degrees in Minnesota with a feels like of 2, so the roof will come in handy tonight at U.S. Bank Stadium.

7:25 p.m. — Vikings legend Adrian Peterson is in the building to watch his former team.

7:17 p.m. — The Bears enter tonight with the No. 1 and No. 4 picks in the 2024 NFL Draft, and they'll remain in those slots with or without a loss thanks to the
strength of schedule tiebreaker.

7:10 p.m. — Heading into Week 13, tonight's game is the difference between the No. 6 and No. 7 seed for the Vikings.
</item 3>

<item 4>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
In an effort not to overwork their backs, who won't have it easy against the Vikings, Kmet could handle more short-area throws than normal. His ADOT over the
past two games has been 7.64, which isn't far from his season average of 6.69.

VIKINGS: Allow the fewest yards per catch to rival tigh ends (5.9 yards), even keeping a lid on Travis Kelce last week (6.7 yards per catch). But they've also
seen the fourth-lowest ADOT from tight ends (5.0 yards per target), almost as if teams know they won't be able to succeed on downfield shots against them (zero
completions of 20-plus yards from tight ends). These numbers are tough but if Kmet gets the target volume we've seen him have over the past couple of weeks he
can still put up palatable numbers. And, tight ends have scored three times in the Vikings past three games.
</item 4>

<item 5>
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
</item 5>

<item 6>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
9:30 p.m. — The Bears' defense has suddenly morphed back into an elite unit on these last two plays, forcing intentional grounding by Dobbs and pressuring him
heavily on the following play. It's 3rd & 24 for the Vikings, who might just settle for a gain of a few yards to set up the field goal.

9:27 p.m. — Another first down by K.J. Osborn drives the Vikings into Bears territory, and the Vikings suddenly find themselves 13 yards from the end zone after
Kyler Gordon is called for interfering with T.J. Hockenson. The Vikings have a chance to take the lead with 38 seconds on the clock.

9:25 p.m. — What an effort by Brandon Powell to turn around and make a 28-yard grab while falling to the ground. That's the biggest offensive play of the night
for the Vikings.

A perfect catch from Brandon Powell on a perfect pass
</item 6>

<item 7>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
When it mattered most, though, Fields was poised and sealed the Bears' first win over an NFC North opponent since 2021.

While Chicago's hopes are still extraordinarily distant at 4-8, the Vikings saw their playoff chances take a hit on Monday night. Minnesota (6-6) dropped to the
No. 7 seed in the NFC with the loss, with the Packers, Rams and Saints all just a half-game behind in the wild-card race. While a win would have put the Vikings
in position to compete for the division crown down the stretch, just holding onto a playoff spot will now be a battle for Kevin O'Connell's squad.

MORE: Latest updates on Justin Jefferson's return date

The Sporting News tracked live updates from the "Monday Night Football" game between the Bears and Vikings. Check out all of the key moments you might have
missed.

Bears vs.
</item 7>

<item 8>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
LAST WEEK: Total disaster. In a gamescript that actually gave Minnesota a chance to -- gasp! -- run the ball, Kirk Cousins barely threw the ball and Addison
went catchless on one target. There was also a major reduction in snaps played, down to 58%. His season-low was 56% from Week 1.

MATCHUP: With the Vikings capable of running the ball, the offense went with heavier formations. Backup tight end Josh Oliver played about as many snaps as he
did in Week 1 and third tight end Johnny Mundt saw a season-high 17% of snaps. Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least
69% of their snaps.

THIRD MAN: Last week did affirm that Addison may actually be the No. 3 wide receiver in the offense behind Justin Jefferson and K.J.
</item 8>

<item 9>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
— Dobbs is nearly intercepted by Jaylon Johnson (again) on third down. The last turnover wasn't Dobbs' fault, but this one should've been a pick. The Bears'
defense is all over the Vikings tonight, and Minnesota can't get out of its own way.

9:13 p.m. — Addison, trying to make up for the drop that turned into an interception, looks like he goes for 17 yards, but replay shows his knee was clearly
down. It becomes a much smaller 4-yard gain.

9:09 p.m. — Fields comes under pressure on third down and misses D.J. Moore. He's starting to miss these receivers after a hot start. Somehow, the Vikings will
have a chance to tie this game or take a lead before halftime.

9:04 p.m. — Dobbs is picked off again!
</item 9>

<item 10>
Bears vs. Vikings live score, updates, highlights from NFL 'Monday Night Football' game (Sporting News, 2023-11-27T23:32:04+00:00)
The Bears defense comes up big with a 4th down stop



📺: #CHIvsMIN on ESPN/ABC

📱: Stream on #NFLPlus https://t.co/J5tLNJHyh1 pic.twitter.com/yndUqwgIbf — NFL (@NFL) November 28, 2023

9:47 p.m. — The Bears answer right back and sack Dobbs, with Sweat and Brisker in there to take him down.

9:46 p.m. — Alexander Mattison has drawn the ire of Vikings fans all season, but he just got Minnesota's second half of to a booming start with a 21-yard run.

End of first half: Bears 3, Vikings 3

9:32 p.m. FIELD GOAL — Greg Joseph drills it from 34 yards out, and this mess of a game is all tied at three entering the half.

9:30 p.m.
</item 10>
=== END ITEM id=b75a66b2d4bcf13a ===
