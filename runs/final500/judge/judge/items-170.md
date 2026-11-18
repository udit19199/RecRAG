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

=== ITEM id=47e14416b6db79dc ===
Gold sentences and facts:
Sentence 0: It was a Michigan quarterback that led the Buccaneers to the promised land last time so perhaps lightning strikes twice.
  Fact 0.0: A Michigan quarterback led the Buccaneers to the promised land last time.
  Fact 0.1: Lightning may strike twice.
Sentence 1: Source: The Big Ten is reviewing the lengthy written responses from both Michigan and Jim Harbaugh.
  Fact 1.0: The source for this is the Big Ten.
  Fact 1.1: The Big Ten is reviewing lengthy written responses.
  Fact 1.2: The written responses are from Michigan.
  Fact 1.3: The written responses are from Jim Harbaugh.

Retrieved items:
<item 1>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
According to ESPN.com, Stalions allegedly purchased tickets to more than 35 games involving Big Ten and potential CFP opponents.

Photos also surfaced os a person who resembles Stalions on the Central Michigan sideline during the Chippewas' opener against Michigan State on Sept. 1. Central
Michigan has yet to confirm who that person was, another layer to a wild Stalions story that has supplied endless meme and conspiracy material on social media.

Stalions might be the fall guy who saved Michigan from further wrath from the Big Ten and NCAA or the reason the Wolverines' national championship season
unraveled. That fate is yet to be determined, but at minimum he will be one of those controversial names we'll never forget, like Harvey Updyke and Nevin
Shapiro.

Jim Harbaugh

On one hand, Harbaugh could serve his second suspension within one season and further NCAA punishment could be coming down the line.
</item 1>

<item 2>
2024 NFL Mock Draft: Bengals' Joe Burrow gets another playmaker; four QBs go in Round 1, but Pats opt for WR (CBSSports.com, 2023-11-17T20:07:09+00:00)
The Falcons have added some older veterans at the position but it is not a viable long-term solution.

Round 1 - Pick 10 Laiatu Latu EDGE UCLA • Sr • 6'5" / 265 lbs Projected Team Washington PROSPECT RNK 11th POSITION RNK 1st After moving on from Chase Young and
Montez Sweat, Washington has a need for rushing the passer. Laiatu Latu may end up being the best out of this draft class.

Round 1 - Pick 11 J.J. McCarthy QB Michigan • Jr • 6'3" / 202 lbs Projected Team Tampa Bay PROSPECT RNK 21st POSITION RNK 4th Baker Mayfield is replaced in
Tampa Bay with J.J. McCarthy. It was a Michigan quarterback that led the Buccaneers to the promised land last time so perhaps lightning strikes twice.
</item 2>

<item 3>
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
</item 3>

<item 4>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
— Pete Thamel (@PeteThamel) November 8, 2023

Tuesday, Nov. 7

Michigan says three other Big Ten schools shared decoded Wolverines signals

While Michigan tries to navigate the storm around its program, the Wolverines are also trying to make one thing clear: they're hardly alone in trying to
decipher opponents' signals.

Michigan has shared evidence with the Big Ten and NCAA showing staff members at Ohio State, Purdue and Rutgers shared intelligence on the Wolverines' signals
prior to last year's Michigan-Purdue Big Ten Championship Game, The Athletic reported.

Documents breaking down Michigan's signals for numerous plays were compiled by Purdue staffers ahead of the game based on information provided by Ohio State and
Rutgers football personnel, the report says. The person who supplied the documents told The Athletic he spent 10 to 12 hours per week trying to decode signals
based on game video and said the practice of sharing decoded signals with other schools is common.
</item 4>

<item 5>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
Public reprimands/fines are common penalties under B1G sportsmanship policy. — Adam Rittenberg (@ESPNRittenberg) November 9, 2023

Michigan's 10-page letter to Big Ten released

Yahoo Sports' Dan Wetzel posted Michigan's full 10-page letter to the Big Ten on X (formerly Twitter). The letter outlines Michigan's belief that the Big Ten
would be acting too hastily in suspending Harbaugh or levying any punishment toward Michigan before the completion of a full NCAA investigation.

In the letter, Michigan said the university has not had the chance to review all the evidence and respond, which oversteps the due process allowed for Michigan.
It said Pettiti should not have unilateral authority to punish violations without first having a completed investigation and then a response from the Compliance
and Reinstatement Committee, which would recommend penalties for rules violations.
</item 5>

<item 6>
Michigan sign-stealing scandal news: Latest updates on possible Jim Harbaugh suspension from NCAA cheating investigation (Sporting News,
2023-11-06T23:58:04+00:00)
However, the primary difference is Michigan is accused of using in-person scouting, which has been prohibited under NCAA rules for nearly 30 years, to acquire
its information.

Monday, Nov. 6

No connection between Buckeyes coach, Michigan investigation

Ohio State coach Ryan Day and his family have no connection to the Big Ten and NCAA investigations into Michigan, Yahoo Sports reported.

A report emerged over the weekend indicating Day may have had some involvement in the probe into the Buckeyes' arch-rivals, but the NCAA told the Big Ten there
are no ties that it knows of between the Ohio State coach and the investigation.

NCAA shares findings with Big Ten as Harbaugh suspension looms

Yahoo Sports reported Monday that the NCAA shared its investigation findings with the Big Ten, noting that the findings "do not connect the in-person scouting
and recording of opponents’ sidelines to Harbaugh."

Still, the conference is considering a suspension for Harbaugh and could hand down punishments before the end of the week.
</item 6>

<item 7>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Helmet technology

CBSSports.com reported on Oct. 20 that the NCAA rules committee is giving teams the option of coach-to-player helmet communication to relay plays from the
sideline during bowl season this year. Tablets will inevitably follow, and here is the sliver of good that will come from this scandal.

It's about damn time. Death to sandwich-board playcalling, please. Bring on those playsheets where coaches simply have to cover their mouth. This is one
instance where emulating the NFL game is overdue.

Sherrone Moore

If Moore is going to be a head coach, then Saturday’s victory against the Nittany Lions was a fitting audition. The Wolverines rushed for 227 yards – including
32 straight run plays – in grinding out a top-10 victory at one of the toughest environments in the FBS at Beaver Stadium.

Moore was emotional - complete with NSFW language – in the aftermath.

Sherrone Moore filled in for Jim Harbaugh as Michigan's coach today.
</item 7>

<item 8>
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
</item 8>

<item 9>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
Will the Harbaugh-Petitti battle lead to an unsettling end for Michigan? President Santa Ono and athletic director Warde Manuel have voiced their support for
Harbaugh, but that has been more about due process than admitting to any wrongdoing.

It has thrust the Wolverines – a talented team led by J.J. McCarthy, Blake Corum and a defense that allows 7.5 points per game – into a villain role, complete
with visits from Ric Flair. That only works when you are winning. One loss, and all this could unravel, and the Terps and Buckeyes still remain on the schedule.

Will a national championship outweigh the stain on their reputation? The answer is almost always yes for the fans, but the program might feel a different
long-term result.
</item 9>

<item 10>
Michigan sign-stealing scandal winners and losers, from Jim Harbaugh to Tony Petitti (Sporting News, 2023-11-11T22:50:13+00:00)
On the other, he could emerge as a martyr for Michigan fans given there is reportedly no proof yet he knew of Stalion's rule-breaking.

Harbaugh will have a temporary restraining order hearing on Nov. 17, according to ESPN.com. That will determine whether he returns to the field for the final
two games against Maryland and No. 1 Ohio State.

It's an inescapable fact that Harbaugh is 31-3 – and Michigan is 35-3 – since the program bottomed out at the end of the 2020 season. While it is almost
impossible to quantify how much the alleged in-person scouting and sign-stealing would have helped the Wolverines over the past two seasons, it will be a tough
stain to erase knowing those violations are now on the Big Ten record.

What will Harbaugh do now? That is almost impossible to predict given his combative personality. Would he sign a contract extension and continue to fight the
NCAA? Will Harbaugh finally take the parachute to the NFL?
</item 10>
=== END ITEM id=47e14416b6db79dc ===

=== ITEM id=4e4bf3e6ea150394 ===
Gold sentences and facts:
Sentence 0: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 0.0: Sportsbooks are also gathering this information.
  Fact 0.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.
Sentence 1: If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their liability.
  Fact 1.0: Sportsbooks might adjust the odds to balance their liability if a significant amount of money is placed on one team or participant.

Retrieved items:
<item 1>
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
</item 3>

<item 4>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on one side might indicate that the
initial odds were potentially "off," prompting an adjustment. Event Dynamics: For live betting, where bets are placed while the event is in progress, the
evolving dynamics of the match can lead to frequent changes in the odds.

Sportsbooks adjust moneyline odds to ensure they remain profitable, balance their book, and reflect the most current information about the event.

Related Pages:
</item 4>

<item 5>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
This includes your initial $10 wager, along with a profit of $15.30.

It's essential to note that this calculation gives you the total payout, not just your earnings. If you're keen on tracking your profits from betting, remember
to subtract your original stake from the payout to get an accurate figure.

Why do moneyline odds change?

Lines at the best sports betting apps change quickly due to two primary reasons:

Public betting

New information

Before setting odds for an event, oddsmakers gather extensive data. Once they finalize the odds, they release them to the public.

As a bettor, you can bet immediately or wait for potential shifts in the odds, although predicting these shifts is challenging.

For instance, when examining odds for the next Super Bowl champion released shortly after the previous Super Bowl, these odds are based mostly on the recently
concluded season.

As the new season approaches and teams make changes, odds adjust based on perceived championship chances. As games progress, lines continue to evolve based on
team performances.

Public betting can also sway lines.
</item 5>

<item 6>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 6>

<item 7>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.
</item 7>

<item 8>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Moreover, information and public perception play a role. If a piece of news or a prevailing sentiment starts to gain traction about a particular team's
increased chances of winning, this can influence how people bet, leading sportsbooks to adjust their lines accordingly.

Point spread odds can and do change based on a combination of actual betting patterns, unexpected events, and shifts in public perception. Savvy bettors keep a
close eye on these line moves, as they can offer insights into where the value lies in a particular bet.
</item 8>

<item 9>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Line shopping is the key that unlocks the best odds and lines at top sports betting sites.

The process is as simple as checking the lines and odds for your prospective bet at multiple online sports betting sites. The result is quite often better odds
and improved lines.

What is line shopping, and why is it worth your time? Find the answers below in our complete guide to line shopping.

How do sportsbooks set odds and lines?

The best sportsbooks employ oddsmakers and oddsmaking services to set their lines and odds.

Other sports betting apps simply copy those lines. The oddsmakers and oddsmaking services use computer models and algorithms that consider analytics,
statistics, injury and roster news, performance, matchup data, and more to develop the lines and odds.

Betting lines and odds are then subject to change based on the betting public’s reaction and any additional information that comes to light ahead of the game.
Sportsbooks will move the line or change the odds if the public bets heavily on one side.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=4e4bf3e6ea150394 ===

=== ITEM id=1f6160aa8b13c267 ===
Gold sentences and facts:
Sentence 0: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.0: The guide is a Diablo 4 guide.
  Fact 0.1: The guide gathers the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.2: The guide simplifies the best Diablo 4 Sorcerer builds for season 2.
Sentence 1: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2.
  Fact 1.0: The guide is a Diablo 4 guide.
  Fact 1.1: The guide gathers the best Diablo 4 Barbarian builds for season 2.
  Fact 1.2: The guide simplifies the best Diablo 4 Barbarian builds for season 2.

Retrieved items:
<item 1>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
dealing X% of its damage to enemies

(Sunken Ruins dungeon in Scosglen) — Hammer of the Ancients quakes outwards, dealing X% of its damage to enemies Aspect of the Expectant (Underroot dungeon in
Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X%

(Underroot dungeon in Scosglen) — Attacking enemies with a Basic Skill increases the damage of your next Core Skill cast by X% up to X% Edgemaster’s Aspect
(Oldstones dungeon in Scosglen) — Skills deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary
Resource

Barbarian Season 2 endgame build — Double Swing

This build is based off of AaronActionRPG’s Double Swing Barbarian endgame guide at IcyVeins, and is a general endgame guide capable of finding success in all
activities.
</item 1>

<item 2>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
If you don’t want to end up dead more often than not, make sure to invest heavily in defensive skills while making your way toward one of the six Key Passives
we listed above.

Looking for other Diablo 4 builds? We have explainers on the Barbarian, Rogue, Druid, and Necromancer classes.
</item 2>

<item 3>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
As such, you’ll need to unlock new skills before enhancing existing ones:

Frenzy / Enhanced Frenzy Hammer of the Ancients / Enhanced Hammer of the Ancients / Furious Hammer of the Ancients Combat Frenzy Rallying Cry / Enhanced
Rallying Cry / Tactical Rallying Cry Hammer of the Ancients (ranks 2-3) Leap / Enhanced Leap / Power Leap Hammer of the Ancients (ranks 4-5) Death Blow /
Enhanced Death Blow / Warrior’s Death Blow / Death Blow (rank 2-5) Wrath of the Berserker / Prime Wrath of the Berserker / Supreme Wrath of the Berserker Pit
Fighter (ranks 1-3) Tempered Fury Furious Impulse (ranks 1-3) Walking Arsenal Hamstring No Mercy Expose Vulnerability (ranks 1-3) No Mercy (ranks 2-3) Swiftness
(ranks 1-3) Thick Skin (ranks 1-3) Counteroffensive (ranks 1-3) Defensive
</item 3>

<item 4>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
The idea here is that you’re constantly swapping between your skills and weapons to give yourself massive damage buffs.

When making a build, you want to have one of these Key Passives in mind and choose your skills accordingly. Thankfully skills that share certain aspects (like
Bleed) are color-coded on the skill tree, and you can even use the in-game search function to find them all. Other Barbarian builds let you take advantage of
mechanics like Thorns or Shouts. But they still typically end with one of these Key Passives.

Looking for other Diablo 4 builds? We have explainers on the Sorcerer, Rogue, Druid, and Necromancer classes.
</item 4>

<item 5>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Sorcerers are the magic-slinging wizards of Sanctuary in Diablo 4. But with each element they can harness bringing its own unique effect, it can be easy to get
overwhelmed when trying to design a Sorcerer build. That’s where we come in, with our suggestions of the best Sorcerer builds for Diablo 4 season 2.

Sorcerers are capable of harnessing lightning to deal with crowds, ice to keep the hordes at bay, and fire to burn the demons of hell. But they’re also able to
enchant their own spells with unique effects, intricately combining abilities for devastating results.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2. If you need even more information on item affixes or other
more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.
</item 5>

<item 6>
Graph query row:
article: The best Sorcerer builds and skills in Diablo 4
relationship: PUBLISHED_BY
publisher: Polygon
publication_date: 2023-10-18T21:43:17+00:00
</item 6>

<item 7>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
Recharging Aspect (Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana

(Zenith in Fractured Peaks) — Each time Chain Lightning bounces off you, gain X Mana Aspect of Control (Sunken Library in Kehjistan) — You deal X% more damage
to Immobilized, Stunned, or Frozen enemies

(Sunken Library in Kehjistan) — You deal X% more damage to Immobilized, Stunned, or Frozen enemies Edgemaster’s Aspect (Oldstones dungeon in Scosglen) — Skills
deal X% increased damage based on Primary Resource when cast, receiving the maximum benefit while you have full Primary Resource

Sorcerer Season 2 endgame build — Firewall

This build is based off of Northwar’s Sorcerer endgame guide at MaxRoll.gg, and is a general endgame guide capable of finding success in all activities.
</item 7>

<item 8>
The best Barbarian builds and skills in Diablo 4 (Polygon, 2023-10-19T17:16:31+00:00)
Barbarians are a relatively simple class in Diablo 4: They smash stuff. However, Diablo 4 is underscored by math that’s so complex that even the simple act of
smashing stuff becomes a dizzying equation. That’s where we come in, with our suggestions of the best Barbarian builds for Diablo 4 season 2.

Barbarians are big, muscular, and powerful. They excel at using Bleed to kill their enemies over time. And their unique weapon-swapping system means they get to
have more Legendary powers at their disposal than any other class.

In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2. If you need even more information on item affixes or
other more complex ideas, we’ve been sure to link out to the more detailed versions of these builds, as well as where to find their original creators.
</item 8>

<item 9>
The best Sorcerer builds and skills in Diablo 4 (Polygon, 2023-10-18T21:43:17+00:00)
refer to Northwar’s original guide.

Paragon Boards for Sorcerer Firewall build

The Paragon Board system is very complex, and you’ll want to carefully pair your Glyphs with specific boards for your class:

Starter Board / Reinforced Glyph Burning Instinct / Destruction Glyph Searing Heat / Adept Glyph Static Surge / Control Glyph Enchantment Master / Exploit

You should level your Glyphs to 15 in this specific order:

Destruction Control Adept Exploit Reinforced

You can path your own way through the Paragon Boards following these basic guidelines. However, for the best results, follow Northwar’s path exactly.

How to make a great Sorcerer build

Sorcerers are Diablo 4’s pure caster class, and use elemental magic to differentiate from the spells of the Druid and Necromancer. The Sorcerer has six Key
Passives, but only three elements, which is what you’ll base your build around:

Lightning

Fire

Ice

Lightning builds utilize Lightning skills to deal with large groups of enemies at once.
</item 9>

<item 10>
Graph query row:
article: The best Barbarian builds and skills in Diablo 4
relationship: PUBLISHED_BY
publisher: Polygon
publication_date: 2023-10-19T17:16:31+00:00
</item 10>
=== END ITEM id=1f6160aa8b13c267 ===

=== ITEM id=179679f3c96aa610 ===
Gold sentences and facts:
Sentence 0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.
  Fact 0.0: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue.
  Fact 0.1: Paradigm was told that Bankman-Fried did not want investors as directors.
  Fact 0.2: Bankman-Fried planned on having a board with experts.
Sentence 1: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 1.0: Rehn declared to the jury that all of that, all of it, was built on lies.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth.
  Fact 2.2: The prosecution said Bankman-Fried committed fraud to achieve power and influence.
  Fact 2.3: The defense argued that the FTX founder acted in good faith.
  Fact 2.4: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.5: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
You can also follow along with her and the TechCrunch team’s coverage in the Chain Reaction newsletter, which drops Thursdays at 12 p.m. PT, and the Chain
Reaction podcast, which will hit your feed every Thursday.

How to follow the SBF trial

The trial kicked off with jury selection October 3. From there, access to the proceedings is limited, as no devices are allowed; reports are being physically
put together within the courtroom (or sent to overflow rooms).

Beyond our daily coverage, additional expertise and commentary will live on TechCrunch+. There, you can find stories like this in-depth breakdown of what to
expect from both sides of the SBF case, where the prosecution and defense could gain ground or fall short in their arguments and what the takeaways were from
the opening arguments.

SBF trial: What we learned in week three

The third week of the trial featured a couple of noteworthy moments for the prosecution, including FTX’s former general counsel Can Sun.
</item 1>

<item 2>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The prosecution painted a picture of an empire “built on lies” and made a point to show actual pictures of SBF alongside rich and powerful figures to illustrate
the peak of his power and influence prior to FTX’s implosion.

“This man stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he
emptied the accounts of ordinary customers, too. He bought himself wealth, power and influence.”

The defense, on the other hand, depicted SBF and his colleagues as in over their heads, without ill-will as FTX collapsed around them.

“Sam and his colleagues were building the plane as they were flying it,” defense attorney Mark Cohen said. “They had to figure out how to navigate a world where
they were running FTX, building out its systems, dealing with hacking threats, managing the credit risk of their customers, managing hundreds of employees, all
while building up their actual exchange.”

Read a recap of the prosecution and defense’s full arguments here.
</item 2>

<item 3>
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
</item 3>

<item 4>
Graph query row:
title: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand
publisher: Fortune
publication_date: 2023-10-04T23:32:53+00:00
related_facts: [{'entity': 'Matt Huang', 'relationship': 'MENTIONS'}, {'entity': 'Damian Williams', 'relationship': 'MENTIONS'}, {'entity': 'Gary Wang',
'relationship': 'MENTIONS'}, {'entity': 'Caroline Ellison', 'relationship': 'MENTIONS'}, {'entity': 'crypto', 'relationship': 'MENTIONS'}, {'entity': 'alleged
crimes', 'relationship': 'MENTIONS'}, {'entity': 'Thane Rehn', 'relationship': 'MENTIONS'}, {'entity': 'Larry David', 'relationship': 'MENTIONS'}, {'entity':
'stealing customer funds', 'relationship': 'MENTIONS'}, {'entity': 'theft', 'relationship': 'MENTIONS'}, {'entity': 'Mark Cohen', 'relationship': 'MENTIONS'},
{'entity': 'Ponzi scheme', 'relationship': 'MENTIONS'}, {'entity': 'crypto markets', 'relationship': 'MENTIONS'}, {'entity': 'Bitcoin', 'relationship':
'MENTIONS'}, {'entity': 'Dogecoin', 'relationship': 'MENTIONS'}, {'entity': 'attempted withdrawal of cash and crypto', 'relationship': 'MENTIONS'}, {'entity':
'Adam Yedidia', 'relationship': 'MENTIONS'}, {'entity': 'Jane Street', 'relationship': 'MENTIONS'}, {'entity': 'Sequoia Capital', 'relationship': 'MENTIONS'},
{'entity': 'Brooklyn', 'relationship': 'MENTIONS'}, '(+22 more)']
</item 4>

<item 5>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 5>

<item 6>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.
</item 6>

<item 7>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 7>

<item 8>
Graph query row:
title: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
publisher: TechCrunch
publication_date: 2023-10-06T17:22:02+00:00
related_facts: [{'entity': 'Josh Naftalis', 'relationship': 'MENTIONS'}, {'entity': 'Naftalis', 'relationship': 'MENTIONS'}, {'entity': 'Jacquelyn Melinek',
'relationship': 'MENTIONS'}, {'entity': 'Chain Reaction podcast', 'relationship': 'MENTIONS'}, {'entity': 'Apple Podcasts', 'relationship': 'MENTIONS'},
{'entity': 'bankruptcy', 'relationship': 'MENTIONS'}, {'entity': 'Chain Reaction newsletter', 'relationship': 'MENTIONS'}, {'entity': 'jury selection',
'relationship': 'MENTIONS'}, {'entity': 'Robert Boroujerdi', 'relationship': 'MENTIONS'}, {'entity': 'Third Point', 'relationship': 'MENTIONS'}, {'entity':
'University of Notre Dame', 'relationship': 'MENTIONS'}, {'entity': 'Easton', 'relationship': 'MENTIONS'}, {'entity': 'FTX Ventures', 'relationship':
'MENTIONS'}, {'entity': 'Chinese officials', 'relationship': 'MENTIONS'}, {'entity': 'Paying Chinese officials $150 million to restore access to $1 billion in
frozen trading accounts', 'relationship': 'MENTIONS'}, {'entity': 'charities', 'relationship': 'MENTIONS'}, {'entity': 'SBF Trial', 'relationship': 'MENTIONS'},
{'entity': 'FTX collapse', 'relationship': 'MENTIONS'}, {'entity': 'SBF', 'relationship': 'MENTIONS'}, {'entity': 'Matt Huang', 'relationship': 'MENTIONS'},
'(+27 more)']
</item 8>

<item 9>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
Much of said funding came from customers, and Easton testified that he found that user deposits were used by both FTX and Alameda to purchase real estate, make
investments or direct funding toward political causes and charities.

SBF trial: What we learned in week two

The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to
repay debts to lenders, using them as a line of credit under the instruction of SBF. Ellison also went into detail about how a $2 billion venture fund, FTX
Ventures, was funded by money that had previously been allocated to Alameda from third-party lenders.

And though there are no charges of bribery in the case, Ellison also testified to paying Chinese officials $150 million to restore access to $1 billion in
frozen trading accounts.

SBF trial: What we learned in the week one

As expected, the opening statements in the SBF trial were spicy.
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 10>
=== END ITEM id=179679f3c96aa610 ===

=== ITEM id=fb2b033469629be0 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that all of that, all of it, was built on lies.
Sentence 1: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 1.0: Wang testified.
  Fact 1.1: Wang heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account.
  Fact 1.2: Bankman-Fried's permission to keep withdrawing money applied as long as the amount was less than FTX's total trading revenue.
Sentence 2: He was the richest person in the world under 30 [and] he was going to spend it to save humanity from extinction,” he said.
  Fact 2.0: The speaker said he was the richest person in the world under 30.
  Fact 2.1: The speaker said he was going to spend his wealth to save humanity from extinction.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 3.1: The defense countered that the FTX founder acted in good faith.
  Fact 3.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 3.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 1>

<item 2>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
As a result of today’s testimony, I am both more and less confused. I am clearer on what probably happened at FTX. What I don’t understand is why Gary Wang did
it.

Wang was the co-founder and chief technology officer of FTX, as well as the co-founder of Alameda Research. He owned only 10 percent of Alameda and 17 percent
of FTX. He got a $200,000 a year salary, with no performance bonuses. (Okay, sure, at times he was a billionaire on paper.) By contrast, Bankman-Fried owned 90
percent of Alameda and 65 percent of FTX.

Yet Wang’s testimony suggests he took bizarre risks for his own relatively small share. Bankman-Fried didn’t code, so none of his fingerprints are on the
technical changes that let FTX allegedly defraud its customers and investors. Meanwhile, Wang supervised the code behind a key prosecution claim: that Alameda
Research was afforded secret, special privileges with FTX’s funds.
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
</item 4>

<item 5>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 5>

<item 6>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
</item 6>

<item 7>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 7>

<item 8>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
</item 8>

<item 9>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Oh, but he was testifying to Congress about keeping customer funds safe.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Who was making decisions to spend $8 billion of customer funds? Bankman-Fried couldn’t recall knowing anything about it. Were there rules or requirements for
how money borrowed from FTX would be returned? Were there rules for risk management? “I was concerned with overall risk management,” Bankman-Fried said.

But it was the testimony about June 2022 that resonated the most to me. Didn’t Bankman-Fried ask what “fiat@ftx” was? He did.
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
A substantial chunk had been spent on various investments, including in the celebrity-connected venture firm K5, trading firm Modulo (co-founded by one of
Bankman-Fried’s ex-girlfriends), and AI company Anthropic.

Another $413 million had gone directly to Bankman-Fried himself, through his wholly-owned company Paper Bird. “Outflows to insiders,” which suggested Singh’s
not-quite-loans were included, was $192 million; real estate was another $228 million; and the donations to Gabe Bankman-Fried’s Guarding Against Pandemics,
combined with the philanthropy from the FTX Foundation, totaled less than $40 million.

It is easy to lie with words and harder to lie with money.
</item 10>
=== END ITEM id=fb2b033469629be0 ===

=== ITEM id=a3f7de11e20c1a72 ===
Gold sentences and facts:
Sentence 0: In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and effectively.
  Fact 0.0: Scientists point to a number of recent developments in the long term.
  Fact 0.1: The recent developments have raised hopes that dementia may be possible to tackle more directly.
  Fact 0.2: The recent developments have raised hopes that dementia may be possible to tackle more effectively.
Sentence 1: Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies.
  Fact 1.0: Health coach Susan Saunders found herself parenting her mother.
  Fact 1.1: Susan Saunders's mother had dementia.
  Fact 1.2: Susan Saunders's own daughters were babies at the time.

Retrieved items:
<item 1>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
It took 10 years for Mike Mackenzie-Smith to get a diagnosis of young-onset Alzheimer’s.

Another grim illustration of the problem facing patients is provided by Graeme Armstrong. His wife, Trina, began having trouble recognising faces and reading
telephone numbers in 2006 and was told, after a CT scan three years later, that she had probably had a stroke – although her symptoms did not suggest such a
diagnosis. It took another three years before she was diagnosed with posterior cortical atrophy, a rare variant of Alzheimer’s disease that affects the way the
brain interprets information from the eyes.

“Had we had an accurate diagnosis four years earlier, Trina could have been placed on the correct medication, which may have been more effective and helped her
day to day,” Armstrong said.

One solution sought by doctors involves the creation of blood tests which could pinpoint the disease quickly and effectively. “These are under research, but we
are several years away from having them in widespread use,” said Thomas.
</item 1>

<item 2>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
However there are long waiting times – up to two years on average – for appointments at these centres.

In addition, diagnosing Alzheimer’s and other forms of dementia is usually based on pen-and-paper tests, followed by lumbar punctures and brain scans before a
final diagnosis.

About 65% of cases are confirmed this way. The remaining third of dementia cases are never diagnosed. Yet patients can only expect to receive treatments –
including the new drugs – if their condition is identified.

Eleanor Mackenzie-Smith’s father, Mike, has young-onset Alzheimer’s. His first symptoms were spotted 17 years ago, when Eleanor was 11. “However, it took over
10 years from the start of his symptoms and four separate tests between 2009 and 2017 to get my dad’s final diagnosis, when he was 65. It was distressing not
knowing what was happening. Too many families, like mine, have had to watch as dementia takes hold of our loved ones while we are left without a diagnosis and
the support and access to treatments.”
</item 2>

<item 3>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
People in Britain could benefit from a key medical breakthrough next year. They may be given access to the first drugs ever developed to slow the impact of
Alzheimer’s disease.

The first of these medicines – lecanemab – was recently approved in the US and Japan, where treatments using it have already been launched. A second drug,
donanemab, is expected to follow soon, and next year it is anticipated that the UK medical authorities will consider both of them for approval in Britain.

The prospect has raised hopes that, after years of effort, scientists may be closing in on ways to directly tackle the UK’s dementia crisis. About a million
people are living with the condition in this country, and this is expected to rise to about 1.7 million by 2040 – with potentially grim consequences. Last year
dementia took the lives of 66,000 people in England and Wales, and it is now the leading cause of death in Britain, with Alzheimer’s accounting for two-thirds
of cases.
</item 3>

<item 4>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
“Anyone with older parents should be urging them out for a walk five days a week – and when they have a companion, the walk also doubles as a social occasion,”
he explained. “If they can walk for longer than 20 minutes, even better. Our study showed that the risk of depression fell in proportion to the time spent
walking.”

As well as a daily walk, Laird recommends vitamin D and B12 supplements – both of which appear to moderate the chance of depression in older people. “Depression
matters,” he added. “Not just because it reduces quality of life, but because in older people there seems to be a link between depression and dementia which
we’re still unpacking.”

Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies. “I’d advise anyone with a parent in the
early stages of dementia to investigate all the available research trials. Start by joining Dementia Research, a nationwide register that matches volunteers to
research programmes.”
</item 4>

<item 5>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Until now doctors have only been able to prescribe medicines that help patients manage their symptoms, so the arrival of the first drugs that treat the actual
cause of the condition has been welcomed – although experts have warned that their use should be treated with some caution.

After decades of research, these are the first drugs to improve patients’ lives directly. If nothing else, they suggest we’re on the right road to tackling
Alzheimer’s

“The new drugs slow down the development of Alzheimer’s by six months to a year and are useful only for those in the early stages of the condition, so they are
certainly not miracle medicines,” said David Thomas, head of policy at Alzheimer’s Research UK.

“However, after decades of research, they are the first to improve patients’ lives directly, and that is a justifiable cause for excitement. If nothing else,
they suggest we are probably on the right road to tackling Alzheimer’s.”

This point was backed by neurologist Cath Mummery, of the Dementia Research Centre at University College London.
</item 5>

<item 6>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and
effectively. One major challenge they have faced is the problem of getting drugs to pass through the blood-brain barrier, which controls the movement of ions
and molecules from our bodies to our brains. That makes it difficult to get drugs into the central nervous system to tackle problems such as amyloid plaques.

“However, researchers are developing active transport methods for getting drugs across the blood-brain barrier,” said Mummery. “For example, we are now
exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact.”

Such developments will take years to realise, scientists caution, and a great deal needs to be done in the short term to deal with dementia.

“It is clear we have taken a step in the right direction but there is much more to do before we can have successfully dealt with these processes that are
happening to our brains,” said Thomas.
</item 6>

<item 7>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It is a challenge, and a very important one.”
</item 7>

<item 8>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Both drugs are to be considered for approval in the UK next year. The Medicines & Healthcare products Regulatory Agency (MHRA) will first decide if they are
safe and effective, then the National Institute for Health and Care Excellence (Nice) will rule on whether they offer value for money.

Both medicines are expensive – lecanemab costs about $25,000 (£19,700) a year – and are given by regular intravenous infusions. “That is a challenge from a
health-service point of view, because you need to find the space and time to put someone in an infusion suite to treat them,” added Thomas.

However, the main problem facing doctors is the difficulty involved in pinpointing dementia not just in its early stages but even in its later phases. Most
cases are first presented to GPs, who will then refer patients to memory clinics for a dementia test. However there are long waiting times – up to two years on
average – for appointments at these centres.
</item 8>

<item 9>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It has been a very long, hard road, but finally we have something positive to look at. That is very welcome.”

Alzheimer’s disease is triggered by the buildup of a protein called amyloid in the brain, although symptoms may not appear for decades after this accretion has
started. Scientists have tried for more than 20 years to find ways to stop amyloid from forming these plaques, in the hope this would stop the progression of
the disease.

Lecanemab, which is produced by Japanese pharmaceutical company Eisai, and donanemab, produced by Eli Lilly of the US, are the first medicines to achieve this
aim – though they only slow but do not ultimately halt the disease’s progression.

Graeme Armstrong with wife Trina, who has a rare form of Alzheimer’s.

Both drugs are to be considered for approval in the UK next year.
</item 9>

<item 10>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“Meanwhile, we have got to get the NHS to a state where it is much more focused and better organised in getting better and earlier dementia diagnoses. That will
be crucial in our battle against the condition.”

Mummery agreed: “By the time you have dementia, you will have had Alzheimer’s disease developing in your brain for at least 20 years. Dementia is the latest
stage; we need to pick up signals much earlier. So we need to be thinking about how to diagnose people at the very earliest stages of disease, when they may
only have very, very subtle symptoms.

We are now exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact Cath Mummery

“Our current service isn’t good for doing that, and we need to develop brain health clinics where we can pinpoint the disease much earlier, then help to build
resilience against dementia in a patient when it is only in a very early stage.”
</item 10>
=== END ITEM id=a3f7de11e20c1a72 ===
