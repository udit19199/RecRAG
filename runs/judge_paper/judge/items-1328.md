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

=== ITEM id=96f33b4c1fab804e ===
Gold sentences and facts:
Sentence 0: Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana.
  Fact 0.0: The U.S. will travel to GEODIS Park in Nashville, Tennessee on October 17.
  Fact 0.1: The U.S. will play a friendly against Ghana at GEODIS Park.
Sentence 1: Fellow import Milton Doyle and Majok Deng each scored 17 for Tasmania, who recorded their second win over Sydney in as many attempts this season.
  Fact 1.0: Milton Doyle is a fellow import.
  Fact 1.1: Majok Deng is a fellow import.
  Fact 1.2: Milton Doyle scored 17 for Tasmania.
  Fact 1.3: Majok Deng scored 17 for Tasmania.
  Fact 1.4: Tasmania recorded their second win over Sydney this season.
  Fact 1.5: Tasmania's second win over Sydney this season came in as many attempts as the team made this season.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Jamaica vs Canada result

Score Goal scorers Jamaica 1 Shamar Nicholson (58) Canada 2 Jonathan David (45 +2), Stephen Eustaquio (85)

Starting lineups:

Jamaica (4-4-2, right to left): Blake (GK) — Leigh, Pinnock, Bernard, Lembikisa — Decordova-Reid, Latibeaudiere, Johnson, Bailey — Antonio, Nicholson

Canada (4-4-2, right to left): Borjan (GK) — Laryea, Vitoria, Miller, Adekugbe — Buchanan, Kone, Eustaquio, Davies — Larin, David

In a game rearranged for Saturday after heavy rain flooded parts of Kingston's Independence Park on Friday, David capitalised on a skilful turn and cross from
Cyle Larin to score his 26th international goal as part of a dominant first half in which his side had four of the five attempts on target.
</item 3>

<item 4>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
The decision about how to replace Adams in the formation was complex — too much so to be resolved in one afternoon.

Reyna was exceptional throughout the first half: less influential than he might have been if given greater access to the ball but evincing a welcome taste for
covering ground to become helpful on defense or to retrieve the ball and launch an attack. Weston McKennie was his typical self, particularly when he could
connect with star winger Christian Pulisic along the left side.

Young Yunus Musah was deployed in the defensive midfield role, and it did not work. Whenever Germany were able to quell an American surge in the first half and
regain possession, they plowed with ease through through the center of the park. He is elite at maintaining possession when operating as a box-to-box
midfielder, and lately his club play, with Pulisic at AC Milan, has seen Musah in more of an attacking role.
</item 4>

<item 5>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
While Christian Pulisic starred on the left, Tim Weah was equally devastating down the right, dazzling with electric pace and good link-up.

Weah's biggest moment came when he scorched Robin Gosens on the wing in the first half, knocking it past the German defender and then beating him with
blistering speed to get back on the ball. Weah's statistical haul don't quite do him justice, but he was a threat throughout his 66 minutes of action.

ST: Folarin Balogun — 7

U.S. fans are salivating for Folarin Balogun to get more touches up front, and as Gregg Berhalter foreshadowed pre-match, they made a clear attempt to do that.
Gio Reyna and Tim Weah did the most of the work in that regard,

LW: Christian Pulisic — 9

There's no doubting who remains the most electric and incisive U.S.
</item 5>

<item 6>
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
</item 6>

<item 7>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
The Reggae Boyz went closest before David broke the deadlock, Aston Villa attacker Leon Bailey hitting the post with a shot from Bobby Decordova-Reid's corner.

Nicholson was aghast when his finish was initially ruled out for offside, only for a VAR intervention and pitchside monitor check by referee Mary Penso to
justify Jamaica's celebrations in their technical area.

The Spartak Moscow forward had raced on to Daniel Johnson's quickly taken freekick and converted coolly to give Jamaica hope of extending their unbeaten run to
six matches.

As well as a Copa America berth, the winners in three days' time will advance to the semifinals of the Nations League, where their opponents will be one of
fellow quarterfinalists the USA, Trinidad and Tobago, Honduras, Mexico, Costa Rica and Panama.

FULL TIME: @CanadaSoccerEN 🇨🇦 defeats Jamaica in the first leg of the #CNL Quarter-Finals!
</item 7>

<item 8>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Canada lineup (4-4-2, right to left): Borjan (GK) — Laryea, Vitoria, Miller, Adekugbe – Buchanan, Kone, Eustaquio, Davies – Larin, David

How to watch Jamaica vs Canada

Here's how to watch this CONCACAF Nations League match across selected areas of the world's major regions:

Region TV Streaming USA — Paramount+ Canada Fubo Canada, OneSoccer CONCACAF Go app UK — CONCACAF Go, YouTube Australia — CONCACAF Go, YouTube India — CONCACAF
Go, YouTube Hong Kong — CONCACAF Go, YouTube Malaysia — CONCACAF Go, YouTube Singapore — CONCACAF Go, YouTube New Zealand — CONCACAF Go, YouTube

USA: This game is available for streaming via Paramount+.

Canada: Fans in Canada can watch on Fubo Canada, OneSoccer and via CONCACAF's Go app and YouTube channel.
</item 8>

<item 9>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Sobey and Smith then knocked down shots as the Bullets lead ballooned to 20.

The Wildcats did make a charge with a Pinder-inspired 11-0 run in the fourth to cut the deficit to three, but the Bullets answered with strong inside finishes
from Harrison and Bannan to secure the win.

United flex muscles in road win over JackJumpers

Melbourne United have had to withstand a hot start and finish from the Tasmania JackJumpers but in between were outstanding on the way to a 90-82 victory in the
NBL.

Advertisement

The league’s top two teams locked horns at a sold out MyState Bank Arena on Friday night and it was the JackJumpers who raced out of the blocks with a 17-4
start, but United took control from there.
</item 9>

<item 10>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But we held on.”

NBA and NBL champion Ian Clark top-scored with 19 points, with rookie Kyle Bowen adding a career-best 17 as did Jo Lual-Acuil along with 10 rebounds.

Advertisement

Jack McVeigh led the way for Tasmania with 20 points and seven rebounds, with Jordon Crawford contributing 16 points and four assists, and Milton Doyle 12
points and five assists.

Tasmania coach Scott Roth was philosophical afterwards, having given up 18 offensive rebounds and lost the battle on the glass by 16.

“We got a little uncharacteristic with our shot selection, which was a credit to them,” he said.

“Then they beat us up on the glass. It’s just little things when you get in those kind of games that are going to make the separation points.”
</item 10>

<item 11>
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
</item 11>

<item 12>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Which looks like this 😅👇 pic.twitter.com/iPhqIdvgCo — OneSoccer (@onesoccer) November 17, 2023

1 hour to kickoff: Lineups have yet to be released, and the likelihood of this match being played seems to be thinning by the moment. The Canada Soccer social
media admin just went on Instagram Live to show the conditions at Independence Park, and they are apocalyptic. The individual said on the live social media
broadcast that "it's not looking likely" the game will be played.

The @CanadaSoccerEN admin just went live on Instagram to showcase the state of the pitch ahead of tonight’s #CanMNT match😂
</item 12>

<item 13>
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
</item 13>

<item 14>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Even if one can argue he has the skill set to succeed at the No. 6, he did not appear comfortable transitioning on short notice.

MORE: USA vs. Germany player ratings

“Tyler’s a great player. It’s not nice to talk about guys that aren't here, though,” Berhalter told reporters afterward. “I’d rather talk about the guys that
are on the field.”

He didn’t, though, relative to how this group filled Adams’ huge vacancy. He was pleased to talk about the pressure the USMNT applied in the first half,
including a ball played over the top of Germany’s defense to Pulisic early in the game that was stopped by an offside call both player and coach disputed, and
Pulisic’s brilliant goal in the 27th minute that involved him driving to his right and blasting a curving shot while four defenders converged and ace goalkeeper
Marc-Andre Ter Stegen watched it sail into the upper righthand corner.
</item 14>

<item 15>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Meanwhile, Fabrizio Romano also shot it down from a journalistic standpoint.

There isn't enough of a break for him to reasonably integrate with Barcelona, and at 36 years old he's sure to welcome the extended time off.

Tata Martino on rumors of Messi playing on loan with Barcelona this winter:



"Is he going for fun? Maybe he's going to visit Barcelona. Other than that, I don't know anything else about it." pic.twitter.com/qJwoJOwkC6 — Felipe Cárdenas
(@FelipeCar) October 8, 2023

2 hours to kickoff: Lionel Messi's club Inter Miami were eliminated from MLS playoff contention after their latest defeat, a 1-0 loss to FC Cincinnati last
weekend. Messi had missed the previous four matches but returned to the field for the final 30 minutes. He was powerless to turn the tide of the game, after
Inter Miami had missed so many gilt-edged chances in the first half.
</item 15>

<item 16>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
That means viewers in the USA can watch this match on Universo, with streaming on Fubo as well as the company's premium subscription services.

The match is not set to be available globally to regions including Australia, Canada, and the United Kingdom, as CONMEBOL has yet to sell broadcast rights to a
provider in those territories.

Other territories with broadcast options for this match include Malaysia (Astro Supersport), Argentina (TyC Sports), Brazil (SporTV 2), France (L'Equipe Web),
Spain (Movistar), and Saudi Arabia (SSC).
</item 16>

<item 17>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
player is, as Christian Pulisic once again backed up his nickname of 'Captain America'. The first half stunner was emblematic of his performance, but beyond
that he constantly proved a threat to German's out-of-position right-back Jonathan Tah.

Pulisic completed 27 of 29 passes, a tidy haul for a player of his attacking intent, and created two chances in addition to getting the goal. His corners remain
awful, and it's a wonder why he is still on duty for that role, but otherwise Pulisic was a menace.

Christian Pulisic with an absolute banger 🚀



Watch USA vs.
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
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
The ousted Nations League quarterfinalists will compete for two more qualification spots, and the calibre of teams in those playoffs is likely to be weaker,
with the USA looking almost certain not to need that route following their 3-0 first-leg win over Trinidad and Tobago.

Eustaquio's first international goal since 2021 also ended Canada's run of two consecutive defeats and gave them a second win in seven matches.

Reaction from Milan Borjan:



"It feels amazing. What the boys did today, in this heat and on this pitch, is unbelievable. We need to come out again and show the brotherhood in Toronto. We
need a full stadium there to give us that support."
</item 19>

<item 20>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
In either setup, the forward line will shift horizontally to allow a midfield player to drive into space, and given options once halted to either play into the
crowded forward line or release a bombing full-back to the off side, both of which can stretch or confuse an opponent's defensive unit.

Having spent time in the Red Bull system, Nagelsmann also deploys a complicated press that, like Gregg Berhalter, starts with the lone central striker. The goal
is to force the opponent to play the ball into overload traps out wide, or if the opponent plays centrally, a defensive midfielder would step up at the base of
a diamond (with two midfielders on the sides and the striker at the top) which would overwhelm an opposition midfield trio.

How will Germany play under Julian Nagelsmann?

The question, now, is how all this will translate to the German national team.

Looking at Julian Nagelsmann's first squad of 26 players, the biggest question is which formation he will play.
</item 20>

<item 21>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
14 MID Pascal Gross Brighton (ENG) 2 0 MID Ilkay Gundogan Barcelona (SPA) 69 17 MID Jonas Hoffman Bayer Leverkusen 22 4 MID Joshua Kimmich Bayern Munich 80 6
MID Jamal Musiala Bayern Munich 23 1 MID Leroy Sane Bayern Munich 55 13 MID Florian Wirtz Bayer Leverkusen 10 0 FWD Kevin Behrens Union Berlin 0 0 FWD Niklas
Fullkrug Borussia Dortmund 9 7 FWD Kai Havertz Arsenal (ENG) 39 13 FWD Thomas Muller Bayern Munich 123 45

Julian Nagelsmann tactics, style of play

Through his time at Bayern Munich, Julian Nagelsmann preferred a 4-2-3-1 formation with Joshua Kimmich and Leon Goretzka holding down the lynchpin central
midfield spots.
</item 21>

<item 22>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
That looked painful for the Red Bulls goalie, but he's ok to continue.

It's a bit surprising that Lautaro got away with no booking from that really dangerous decision.

13th min: Chance, Argentina! Oh no, it's a terribly bad miss from Nico Gonzalez. The Argentine winger breaks down the left, and with the Paraguay defense
lagging behind, he's got plenty of room to shoot across the face of goal from a somewhat tight angle.

The "shot" is so bad it's hard to tell if he was trying to put it on frame or square it for a teammate at the back post, but it's a tragic attempt either way,
as there was nobody anywhere close at the far stick to meet it.

9th min: Paraguay get their first attacking move playing long to the striker, but after a few nervy moments the Argentina defense recovers and gets back on the
ball.

4th min: GOAL! ARGENTINA!
</item 22>

<item 23>
Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match (Sporting News, 2023-10-12T21:11:00+00:00)
However, with the 2026 World Cup field — and by extension, qualifying across all federations — expanding this cycle, Los Guaranies will consider themselves real
contenders for a spot and thus points will be at a premium in every match.

LIVE BLOG: Follow along with updates, highlights, and commentary as Argentina meet Paraguay

Argentina vs Paraguay live stream, TV channel

Following an agreement at the 11th hour of the September international break, broadcast rights for Argentina's home World Cup qualifiers in the United States
have been secured by Telemundo and Univision. That means viewers in the USA can watch this match on Universo, with streaming on Fubo as well as the company's
premium subscription services.

The match is not set to be available globally to regions including Australia, Canada, and the United Kingdom, as CONMEBOL has yet to sell broadcast rights to a
provider in those territories. The match is, however, available on FIFA+ for free in select territories.
</item 23>

<item 24>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
So many national team fans sided with Reyna during the controversy that developed as the team competed at last winter's World Cup. They got what they argued for
in this game, but not what they wished.
</item 24>

<item 25>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Reaction from Steven Vitoria:



"It feels great to get our first win here in 35 years. It wasn't easy, far from it. Still a lot to be done but we're one step closer to where we want to
go."#CANMNT pic.twitter.com/3YYwSNCgFm — Canada Soccer (@CanadaSoccerEN) November 18, 2023

Lille striker David had two promising early opportunities, poking a shot into Milan Borjan's arms under pressure from two markers and applying a looping header
to an Alphonso Davies cross, forcing the goalkeeper to backpedal and claim.

Even if they are knocked out on Wednesday, there will be another chance for the losers of the tie to reach the Copa America.
</item 25>

<item 26>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Every match ad free, live and on demand in 4K UHD

The most likely outcome is that Fiji wins easily and the Wallabies will be pushing confirm on their off-season holiday plans.﻿

"We've only got ourselves to blame, we put ourselves in this position," Wallabies centre Samu Kerevi, who was born in Fiji, said after an open training session
in front of 1000 keen locals at the Andrezieux-Boutheon rugby club on Saturday.

"We'll sit down as a team and watch together on Sunday night. I think we understand the situation ... our results and the way we played, we put ourselves in
this position. So the team understands that and we have our belief in anything's possible at the World Cup.
</item 26>

<item 27>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
Despite being without star big Aron Baynes (suspended) and import guard Shannon Scott (hamstring), the Bullets soared to a second straight victory on Sunday and
snapped a quirky streak in the process.

It was their first win after seven straight losses in Sunday fixtures, in a run that dates back to last season.

Far more importantly, the result levelled Brisbane’s current season record at 4-4 after five rounds.

But it didn’t come easily as fourth-quarter specialists Tasmania trimmed a seven-point deficit back to a single point when Majok Deng drove to the rack with 34
seconds remaining.

Advertisement

The two sides traded blows in the dying stages and the JackJumpers almost sent the game to overtime when they pinched Sam McDaniel’s inbounds pass on the final
play.

Jack McVeigh got a hand in to force the turnover and Tasmania found Jordon Crawford, but his decent look at an attempted game-leveller rimmed out.
</item 27>

<item 28>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
ET (7:30 a.m. PT) tomorrow morning .

Nothing else was announced as far as what should happen if the pitch remains water logged. That means one of two things: either the federations haven't sorted
that out yet, or they are looking to keep that in-house as not to risk publicly announce plans for something that never ends up happening.

Here's to hoping that won't be necessary anyways, and tomorrow morning's kickoff can proceed as planned!

UPDATE (8:25 p.m. ET): **GAME POSTPONED UNTIL TOMORROW**

The match has been called off for the evening, and they will try again tomorrow. Journalists have announced that the match has been pushed back until 10:30 a.m.
ET / 7:30 a.m. PT. No official word from the federation yet, but this seems as concrete as ever.

They tried to wait it out tonight, but it just wasn't to be.
</item 28>

<item 29>
Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match (Sporting News, 2023-10-12T21:11:00+00:00)
SGT New Zealand Fri, Oct. 13 11 a.m. NZST

Argentina vs Paraguay starting lineups & team news

Argentina head coach Lionel Scaloni named a bloated 36-man squad for the matches against Paraguay and Peru this month, but Angel Di Maria, Paulo Dybala, and
Juan Foyth were forced to withdraw due to injury. Lisandro Martinez is also unavailable after having to undergo another foot operation.

Now that Inter Miami's playoff run is officially over, Argentina can feel a little better about deploying Lionel Messi at full 90-minute use despite his recent
fitness issues, knowing his club season is nearly over and he'll have months of rest upcoming. Messi missed the last match against Bolivia, and missed four
Inter Miami games after, but he returned for a half-hour of action in their final fixture before the international break. However, he doesn't start, but is
reportedly fit enough to come off the bench for most or all of the second half.
</item 29>

<item 30>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
He got to answer critics like Mark Bosnich and Craig Foster as well – bluntly rejecting their negativity with the dismissive “who cares?”. That would have
tasted sweet.

On Tuesday, FA CEO James Johnson, who held firm on Arnold 20 months ago, got to congratulate the coach on his impending record-breaking match against Bangladesh
as 2026 World Cup qualifying starts on Thursday.

Advertisement

“Graham is one of the greatest contributors to the Socceroos,” Johnson said in a statement.

“He’s been a constant in the Australian football landscape for more than 30 years as both a player and coach, being a part of some of the Socceroos most
memorable moments.

“Anyone who knows Arnie will tell you he bleeds green and gold. His love for the Socceroos and football in Australia is both infectious and inspiring.
</item 30>

<item 31>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
🎥: @onesoccer pic.twitter.com/XzBldBSXfX — Canada Soccer (@CanadaSoccerEN) November 17, 2023

2 hours to kickoff: While the Canadian federation is reeling from the departure of John Herdman, one player is happy to see change happen at the top. Central
midfielder Mark-Anthony Kaye had been left out of multiple squads towards the end of Herdman's tenure, and he believes his relationship with the former Canada
head coach was the reason.

Me and John had some really tough conversations where he blatantly told me I wasn’t good enough to be part of the team. That was hard for me to accept,” Kaye
said to Sportsnet, after also posting to social media an honest self-reflection following his New England Revolution's elimination from the MLS playoffs a week
ago.

Heartfelt post by Mark-Anthony Kaye.
</item 31>

<item 32>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Sergino Dest allowed the ball to deflect off his foot and into dangerous territory at the top of the box, de la Torre became a spectator with a too-great view
and central defender Chris Richards missed two opportunities to shut down the threat. McKennie, central defender Tim Ream and right back Joe Scally all bore
some responsibility for the second.

“When you watch the goals back — I had the opportunity to do that in the coaches’ office — it’s really these fine moments, these split seconds where you need to
be well-positioned, not open up any gaps in your back line,” Berhalter said. “And that’s where it went wrong.”

With the attempt to transform Musah into some semblance of Adams, though, it almost was certain to go wrong. The USMNT conjured a more compelling attack because
of Reyna’s presence, but the ability to control the game went missing.
</item 32>

<item 33>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Sporting News reviews how each USMNT player performed across the 90 minutes.

MORE: Recap the 3-1 defeat for the USMNT against Germany in Hartford

USA player ratings vs Germany

USA starting lineup (4-3-3, right to left): 1. Turner (GK) — 19. Scally, 4. Richards (Carter-Vickers, 66'), 13. Ream, 2. Dest — 7. Reyna (de la Torre, 46'), 6.
Musah, 8. McKennie (Cardoso, 75') — 21. Weah (Aaronson, 66'), 20. Balogun (Pepi, 66'), 10. Pulisic (Paredes, 75').
</item 33>

<item 34>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
There are no natural right-backs on the roster, meaning he will either have to pidgeonhole a center-back (Antonio Rudiger or Niklas Sule) out wide, sacrifice
Josh Kimmich's presence in midfield to move him back, or move Pascal Gross back. None of those are ideal options.

Whether Nagelsmann deploys a 4-2-3-1 or a 3-2-4-1, the tactical approach will likely remain the same.

Getting the best out of young superstar Jamal Musiala will be one of the most important aspects of Nagelsmann's tenure, as Musiala has blossomed into a
world-class talent at Bayern Munich, but the 20-year-old has just one goal in 23 international caps to this point.

Nagelsmann will also look to introduce a few overlooked players, starting with Robert Andrich who could factor into the midfield setup as a traditional No. 6,
especially as Ilkay Gundogan enters the twilight of his career.
</item 34>

<item 35>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Along the back, Nagelsmann has oddly selected five center-backs compared to just two natural full-backs in David Raum and Robin Gosens, both left-sided players,
possibly indicating a desire to play a three-CB system, although it's hard to know for sure as Nagelsmann preferred a 4-2-3-1 for much of his club managerial
career.

One of the center-backs present in the squad is Mats Hummels, who has not been called in since 2021 and dropped for the entirety of Hansi Flick's time in
charge. This came at the expense of Hummels' Borussia Dortmund teammate Nico Schlotterbeck.

First national team call-ups were handed to Kevin Behrens of Union Berlin, Chris Fuhrich of Stuttgart, and Robert Andrich of Bayer Leverkusen.
</item 35>

<item 36>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 36>

<item 37>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
“I think he played a good game, showed his quality. I think everyone in the first half played well. So it was good to see him, good to have him on the field.”

It was not good to not have him on the field, because the collapse from a 1-1 halftime tie not only followed Reyna’s departure, but probably was connected. The
U.S. was able to threaten the opposing goal enough with Reyna on the field to cause the opposition to concern itself with defending more often than it
preferred. Replacement Luca de la Torre dulled the attack and did nothing to dissuade Germany’s rampage toward the American net.

There was plenty of blame to be shared on the goals by Niclas Fullkrug in the 58th minute and Jamal Musiala three minutes later.
</item 37>

<item 38>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s own messaging. Crocker, in a September meeting with U.S. reporters along with Cone and U.S. Soccer CEO J.T. Batson, said he hoped to have a new head
coach in place by December.

But the initial contact with Hayes was made a couple of months ago, early into the search, with all three top-level executives from U.S. Soccer involved in
those talks. The trio also described the interview process to journalists in that September meeting – a U.S. soccer statement describes it as involving
“psychometrics and abstract reasoning tests, in-depth discussions of strategy, coaching philosophy and the current player pool, as well as evaluation on the
reactions to pressure, culture-building and interactions with players and staff.” USMNT coach Gregg Berhalter went through a similar process, including an
abstract reasoning test, when he was re-hired by Crocker earlier in 2023.
</item 38>

<item 39>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
RENTSCHLER FIELD, HARTFORD — The United States appeared well positioned to fight with a European power through 45 minutes, but the house of cards came crashing
down after the break as the U.S. men's national team succumbed to a 3-1 defeat to Germany in Hartford.

Presented with a chance to test themselves against a world-class opponent on home soil, the U.S. entered the break level as a Christian Pulisic stunner was
cancelled out by Ilkay Gundogan's poaching.

Yet after halftime, as Gio Reyna was withdrawn from his first start back under Gregg Berhalter to build his fitness, the U.S. midfield was overrun and the
defense committed a multitude of errors to allow the Germans to gain full control of proceedings.

"It’s frustrating because its just little moments," said U.S. goalkeeper Matt Turner after the match. "Little moments made a big difference at the World Cup,
and it’s kinda the same story here.”
</item 39>

<item 40>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.
</item 40>

<item 41>
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
</item 41>

<item 42>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
.@pulisic with a SCREAMER to open the scoring 😱 pic.twitter.com/KlG0oMeNPm — U.S. Men's National Soccer Team (@USMNT) October 14, 2023

“I think we created some problems,” Berhalter said. “And then just the threat we had a lot in the first half was really good. It put Germany on their heels at
times. But to me, it’s also about controlling the tempo of the game, and that’s what we really need to learn.”

It was a victory of sorts for Reyna to play 45 minutes after he’d gone only 27 in his one club appearance this season for Borussia Dortmund. He’d been injured
in June while playing for the U.S. and missed all of preseason training and nearly two months of the Bundesliga season.

“The goal is to send him back to Dortmund in a really good spot,” Berhalter said.
</item 42>

<item 43>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Pulisic (Paredes, 75').

GK: Matt Turner — 8

The only goal Matt Turner conceded in the first half came on a play Turner initially saved, before being slammed home by Ilkay Gundogan as the rest of the U.S.
defense just stood there and watched.

Turner made a host of excellent saves, and while Germany certainly didn't dazzle in the finishing department, he was a critical part of the U.S. performance as
the defense melted in front of him. The second half was overwhelming for the U.S. back line, and Turner had little answer for the waves of pressure, but he
still made a number of good stops, particularly just after the break with the game still level.

A quality performance from the U.S. goalkeeper, although he could have done better to help marshal a totally scattered defensive line.

Matt Turner says no ❌



Watch USA vs.
</item 43>

<item 44>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
With the United States set to test themselves against one of Europe's most powerful sides, the opponents are entering a transitional phase as well.

Germany are considered one of the preeminent national team programs in the world, but recent years have humbled Die Mannschaft. They have been eliminated in the
group stage of two straight World Cup tournaments, an unthinkable slip in stature given they had never been bounced from any World Cup that early in history
prior to 2018.

With Hansi Flick unable to steer the team to safer waters in Qatar, Julian Nagelsmann has taken over as new Germany head coach.

At just 36 years old, Nagelsmann is a rising star, having spent three successful seasons at Hoffenheim followed by two years at RB Leipzig, which catapulted him
to the managerial post at Bayern Munich in 2021. He steered the Bavarian giants to the Bundesliga title in his first season, but he was sacked by the club in
March of 2023 amidst a furious league title fight.
</item 44>

<item 45>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
NEW PODCAST! Michael Atkinson and Sam Worthington preview a thrilling final round of pool games to decide the quarter-finalists at the Rugby World Cup in
France﻿

He wanted to regain form at club level ﻿with Urayasu D-Rocks but his hand remained up for Test selection.

"I chose to leave Australian shores and understand the ruling around that and what it holds," he said.

"I've got to perform for my club first and if you get the call up, you get the call up. If you don't, I'll take my three months break and enjoy that."

Kerevi is an unabashed Jones fan.

Eddie Jones with fans during a Wallabies training session at Stade Roger Baudras. (Getty)

"He's up at 4am working hard. We catch him in the gym, he's working hard and I swear he doesn't even sleep.
</item 45>

<item 46>
Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match (Sporting News, 2023-10-12T21:11:00+00:00)
local time in Buenos Aires.

Here's how that time translates across some of the major territories:

Date Kickoff time USA/Canada Thu, Oct. 12 7 p.m. ET USA/Canada Thu, Oct. 12 4 p.m. PT UK Fri, Oct. 13 12 a.m. BST Australia Fri, Oct. 13 9 a.m. AEST India Fri,
Oct. 13 4:30 a.m. IST Hong Kong Fri, Oct. 13 7 a.m. HKT Malaysia Fri, Oct. 13 7 a.m. MYT Singapore Fri, Oct. 13 7 a.m. SGT New Zealand Fri, Oct. 13 11 a.m.
</item 46>

<item 47>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 47>

<item 48>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Bailey is adjudged to have been offside.

1 min: Jamaica 0-0 Canada

Glorious-looking conditions in Kingston this morning, thankfully, with sunshine and a picturesque, tree-filled backdrop to this stadium.

It's a quiet start between these sides, Jamaica in their familiar yellow and green home strip, Canada in dark blue.

20 mins to kickoff: Here's how Canada prepared.

Time to get loose 🏃 pic.twitter.com/fWUbJQQiyp — Canada Soccer (@CanadaSoccerEN) November 18, 2023

40 mins to kickoff: Jamaica's lineup (4-4-2, right to left): Blake (GK) – Leigh, Pinnock, Bernard, Lembikisa – Decordova-Reid, Latibeaudiere, Johnson, Bailey –
Antonio, Nicholson

1 hour to kickoff: Team news!
</item 48>

<item 49>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Advertisement

The hiring process included multiple rounds of evaluation, with the list of candidates becoming smaller each time. The first pass was driven purely by data,
which was then whittled down to a double-digit list Crocker was considering as of September, and then a final shortlist, which also included Tony Gustavsson,
head coach of Australia. Multiple sources confirmed both Hayes and Gustavsson flew to the U.S. for interviews.

One source who was briefed on the situation said the federation had also checked in on the availability of Sarina Wiegman, despite clear messaging from both the
English FA and Wiegman herself in August. “I’m staying out of it. I’ve heard it (from the press officer) but no, I’m with England and I’m really happy with
England, and I have a contract until 2025,” Wiegman said. A representative from her camp declined to comment for this story.
</item 49>

<item 50>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 50>

<item 51>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Jamaica's first-ever victory on Canadian soil has seen the Reggae Boyz fight back to reach the CONCACAF Nations League semifinal and qualify for Copa America
2024.

A 3-2 win in the second leg at BMO Field in Toronto allowed Jamaica to progress on away goals after a 4-4 aggregate tie across the two legs.

After Canada opened the scoring through Alphonso Davies they held a comfortable 3-1 aggregate lead, but that advantage eroded in 15 second-half minutes as
Jamaican striker Shamar Nicholson scored a brace, before Fulham attacker Bobby De Cordova-Reid converted the penalty that sealed progression.

Canada will now have to defeat Trinidad and Tobago in a one-off playoff in March to reach the Copa America 2024 tournament to be hosted in the U.S.

MORE: Highlights, recap of Canada's narrow win in the first leg

The Sporting News followed the Canada vs. Jamaica match live, providing score updates, commentary and highlights as they happened.
</item 51>

<item 52>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
CB: Chris Richards — 4

Statistically, Chris Richards had a pretty good match, missing with just one pass of his 42 attempted and not being beaten at any point by any German attackers.
Yet his lack of volume speaks... well, volumes.

Richards was off the pace for much of his 65 minutes, and failed to get involved as his teammates were beaten repeatedly on the dribble. It was Richards'
positioning that let him down, and he should have been more involved with the U.S. under siege in such an open match.

CB: Tim Ream — 5

Chris Richards often left Tim Ream on an island, and the 36-year-old just could not cope with the waves of German attackers.

The third German goal was emblematic of this lack of chemistry between the two U.S.
</item 52>

<item 53>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
Oct 8th, 2023

Oct 8th, 2023

S﻿AINT-ETIENNE: The Wallabies will have to endure a more tortuous Sunday night watch than even the worst rom com as their Rugby World Cup fate is decided by
Fiji and Portugal in Toulouse.

Australia will gather together in their plush Saint-Galmier hotel at 9pm local time (6am Monday AEDT) to view what will almost certainly be their World Cup
execution.

The Wallabies are currently sitting second in Pool C but a single bonus point would be enough for Fiji to leapfrog them and advance to the quarter-finals in
Marseille.

Watch all the action from Rugby World Cup 2023 on the home of rugby, Stan Sport.
</item 53>

<item 54>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
The Bayern star saw his shot saved onto the crossbar before the ball spun in on the goal line.

This gave Canada a 3-1 aggregate lead and if it wasn't for Jamaican stopper Andre Blake, there could have easily been a couple more goals, making two great
saves from Cyle Larin prior to Davies' opener.

However, the home side couldn't get the second goal and Jamaica grew in confidence, particularly after half-time.

In three second-half minutes, the tie was blown apart by two goals from Shamar Nicholson. A Canadian mistake handed the first to him on a plate before Demarai
Gray's individual brilliance set up the second for a tap-in.

With the scoreline 2-1 on the night, the match was destined for extra time and penalties with a 3-3 aggregate.

But Jamaica could barely contemplate possible glory before Ismael Kone's header made it 2-2 on the night and gave Canada a single-goal aggregate advantage.
</item 54>

<item 55>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 55>

<item 56>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
There's a choice for Nagelsmann to make up front, as Niklas Fullkrug has been prolific in his short national team stint, but Thomas Muller is an experienced
veteran who is still performing at a high level.

Injuries for Nagelsmann to work around include Serge Gnabry who is out with a broken arm, and Timo Werner who has been touch and go at RB Leipzig with a back
problem.

Germany predicted starting lineup vs. USA (4-2-3-1): ter Stegen (GK) — Kimmich, Rudiger, Sule, Raum — Gundogan, Goretzka — Sane, Musiala, Brandt — Fullkrug.

Germany second-choice lineup (4-2-3-1): Leno (GK) — Gross, Tah, Thiaw, Gosens — Andrich, Kimmich — Hoffman, Wirtz, Fuhrich — Muller.

German national team squad for USA, Mexico friendlies

Pos Player Club Caps Goals GK
</item 56>

<item 57>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
The striker can only poke a low shot gently towards goalkeeper Andre Blake, who gathers.

7 mins: Jamaica 0-0 Canada

There isn't a great deal of action to report in the sunshine. Here's how Canada looked before kickoff.

4 mins: Jamaica 0-0 Canada

Jamaica should have scored! Leon Bailey is through on goal from a long ball but produces a tame, rushed finish that drops comfortably wide of Milan Borjan's
goal with the goalkeeper well off his line.

Manager Heimir Hallgrimsson turns and shows his frustration on the sideline, while Bailey - justifiably - looks sheepish. That was well short of the quality
he's capable of.

2 mins: Jamaica 0-0 Canada

Leon Bailey canters along the left flank and sends in a dangerous low cross which Canada do well to clear. Bailey is adjudged to have been offside.
</item 57>

<item 58>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
MORE: Watch Premier League matches live with Fubo in USA | Watch EPL in Canada

TV channel Streaming USA Universo Fubo, Telemundo Deportes en Vivo, Universo NOW UK — — Australia — — Canada — — India — — New Zealand — — Malaysia Astro
Supersport Astro Go, sooka Singapore — — Hong Kong — —

Argentina vs Paraguay betting odds & lines

In Canada, bet CONMEBOL at Sports Interaction: See the latest odds for Argentina vs Paraguay here
</item 58>

<item 59>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Sporting director Matt Crocker made the final decision to hire Hayes after a search process that began in August, following the team’s surprising exit in the
round of 16 at the World Cup and the subsequent departure of head coach Vlatko Andonovski.

“She has tremendous energy and an insatiable will to win,” U.S. Soccer president Cindy Parlow Cone said in a statement. “Her experience in the USA, her
understanding of our soccer landscape and her appreciation of what it means to coach this team makes her a natural fit for this role and we could not be more
pleased to have her leading our women’s national team forward.”

Though Hayes is seen as one of the world’s top coaches in women’s soccer, the appointment still comes as something of a surprise. Here’s how the deal got done.

Details of the deal

At least part of the surprise surrounding Hayes’ hire – and the six-month runway before she officially takes charge – is down to U.S. Soccer’s own messaging.
</item 59>

<item 60>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Jamaica lineup (4-4-2, right to left): Blake (GK) – Leigh, Pinnock, Bernard, Lembikisa – Decordova-Reid, Latibeaudiere, Johnson, Bailey – Antonio, Nicholson

Attacking midfielder Liam Millar was out with a calf injury for Canada, whose interim head coach Biello says his players are on a "mission" to progress.

"We know that these matches ​will be​ against a solid opponent in difficult conditions," Biello added. "That is why ​ we decided on an experienced and seasoned
group of players who are used to high-pressure games.”
</item 60>

<item 61>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Y Julián es muy molesto. Corre y marca. Son dos grandes 9".



🗣️ Mario Kempes, en @TyCSports. pic.twitter.com/SIFzdIfrM9 — TR SPORTS ®️ (@trsports_) October 11, 2023

Argentina vs Paraguay starting lineups & team news

Argentina head coach Lionel Scaloni named a bloated 36-man squad for the matches against Paraguay and Peru this month, but Angel Di Maria, Paulo Dybala, and
Juan Foyth were forced to withdraw due to injury. Lisandro Martinez is also unavailable after having to undergo another foot operation.

Now that Inter Miami's playoff run is officially over, Argentina can feel a little better about deploying Lionel Messi at full 90-minute use despite his recent
fitness issues, knowing his club season is nearly over and he'll have months of rest upcoming.
</item 61>

<item 62>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Matt Turner says no ❌



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/x2HslCk3s2 — B/R Football (@brfootball) October 14, 2023

RB: Joe Scally — 6

Joe Scally was tidy in possession, and while he offered very little in the attack, that was largely by design as they didn't need him on that right side with
Tim Weah cooking.

Scally defended ok, and wasn't guilty of any big mistakes, which is more than most of the other U.S. defenders can say. It wasn't a standout performance by any
means, and the whole back line has some soul searching to do, but others were far more culpable than the Borussia Monchengladbach man.
</item 62>

<item 63>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Canada took a major step towards Copa America 2024 qualification as Stephen Eustaquio's 85th-minute strike gave them victory in their CONCACAF Nations League
quarterfinal first leg and a first win in Jamaica for 35 years.

Porto midfielder Eustaquio slotted in from Richie Laryea's surging run and low delivery, restoring the lead Canada had initially established through Jonathan
David during first-half added time.

Shamar Nicholson fired in Jamaica's equaliser 13 minutes after the break but their opponents will seal their place at Copa America if they avoid defeat in the
second leg at Ontario's BMO Field on Tuesday.
</item 63>

<item 64>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Dest was lively up front as he combined with Christian Pulisic and worked very well on the ball. He got Mats Hummels booked on a counter-attack and delivered a
few very quality crosses.

Dest was also guilty of some of the game's worst defensive errors. He was completely non-existent as Leroy Sane blazed by him on the first Germany goal, and he
nearly got his team in trouble again right before halftime by holding onto the ball too long after making an interception rather than just clearing it out of
danger. Filling in for Antonee Robinson at left-back rather than his natural right, Germany targeted him on multiple occasions, to great profit.

He had a few good moments guarding Leroy Sane in the second half, but he was a top culprit for Germany's second goal by playing Niklas Fullkrug onside. If the
U.S.
</item 64>

<item 65>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Gomez does well to dispossess Messi, but the Argentine is first back on the loose ball, expertly shielding the ball with his body, and Gomez obliges by barging
in from behind.

With the set-piece, Messi hits the woodwork! Carlos Coronel is helpless to do anything but watch, rooted to the spot, and he sees it sail by him and strike the
outside of the left post!

Gio Lo Celso comes on for Argentina to replace Rodrigo de Paul, who had a very good match.

89th min: As this game sputters towards a conclusion, Emiliano Martinez has set a new national team record for a shutout streak, crossing 609 minutes of play
without conceding a goal. It's an impressive mark nonetheless, but in fairness, he's had almost nothing to do tonight, making just one (admittedly impressive)
save.

Argentina have not conceded a single goal since the 2022 World Cup final. Four minutes of stoppage time are shown.
</item 65>

<item 66>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 66>

<item 67>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
In March last year Graham Arnold seemed a dead man walking – the Socceroos World Cup qualification campaign was lurching towards failure and the manager was
being battered from all sides.

Arnold’s team had finished seven points off automatic qualification for Qatar after a shocking run of one win in seven games left them facing a Asian playoff
against United Arab Emirates and then another against Peru to make the finals.

While rumours swirled and ex-players and fans bayed for his blood, Football Australia took the pragmatic course and stuck by Arnold. It likely had more to do
with the lateness of the hour and financial considerations than any overwhelming faith in Arnie to run around the fortunes.

Facebook Twitter Whatsapp Reddit Email Share

But he did just that with qualification – and an excellent run at the finals – bowing out to the eventual champions Argentina and after nearly taking them to
extra time.
</item 67>

<item 68>
England thrash Scotland but cannot avoid Team GB Olympic heartbreak (The Guardian, 2023-12-05T21:56:19+00:00)
That rare foray forward was urged on by a vocal Scotland crowd, in which interesting conversations could be heard. After Lauren Hemp raced free on the right a
hopeful “Come on Lauren” was met with a thick, cheery Scottish “Shut up!” behind the press box.

If it was friendly in the stands it was far less so on the pitch. England went close to a second, with the Dutch having taken the lead against Belgium, when
James weaved into the box before delivering to Hemp at the far post who lashed her effort off the base of a post from close range.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy .
</item 68>

<item 69>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Argentina completely and utterly dominated the first 45 minutes, but have precious little to show for it.

The hosts pummeled Carlos Coronel's net, and while the Red Bulls goalkeeper has made a few saves, it's mostly been Argentina's wastefulness that has seen them
fail to follow up their early opener. Rodrigo de Paul hit the post, Alexis Mac Allister probably should have had a penalty, and Nico Gonzalez has missed a few
great chances from the left.

Overall, Lionel Scaloni will be pleased with his side's performance, but the match hangs in the balance on the scoresheet. In reality, thought, Paraguay need
vast improvement if they are to snatch a result after being meticulously carved to pieces.
</item 69>

<item 70>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
But Goulding caught fire to lead United to a 52-51 at the main break.

Defence was key in the second half as both sides tightened up, with Melbourne holding the visitors to just 15 points in the final period.

Travers returned after half-time, finishing with six points and a game-high 11 rebounds.

Advertisement

Ariel Hukporti retained his spot in Melbourne’s starting five ahead of Jo Lual-Acuil Jr and impressed with seven points and 10 rebounds, including four on the
offensive end.

Bullets ride late luck to win over Tasmania

Nathan Sobey has peeled off a game-high 28 points as Brisbane survived a late Tasmania fightback to continue their NBL resurgence with a tense 90-87 win.
</item 70>

<item 71>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.
</item 71>

<item 72>
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
</item 72>

<item 73>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
What is the style of play, what’s the plan?

“He needs to take a step back, ten Hag, and say what do I need to do with these players I have available in order to first of all stop leaking goals, because
that’s a problem, and secondly, score goals – how do I get the next goal? Four games without a goal, it’s a long time.”

Advertisement

DEADLOCK BROKEN! West Ham lead against Manchester United! ????



That assist from Lucas Paquetá though… ????



Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/Qqq7zugQmj — Optus Sport (@OptusSport) December 23, 2023

Ten Hag tried to put a positive spin on the performance by claiming to BBC that they were the better side before the first goal.
</item 73>

<item 74>
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
</item 74>

<item 75>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Bayern Munich have their first game back against Mainz on Saturday, as does much of the Bundesliga.

"I don't know if I'd be interested in playing against Mexico in America before [the Euros hosted] in Germany," said Bayern Munich manager Thomas Tuchel prior to
the international break. "I don't know if anyone can explain this in a way that I can understand. I don't know who decided it, but it has been decided.

"There is a difference between a busy schedule and everything away from home and in different time zones. This is at the end of the endurance. No coach in the
world at this level will say otherwise."

"I don't understand the trip to the USA — for a variety of reasons," said former Germany international Lothar Matthaus to German outlet Bild back in the summer.
"I would have tried to create a good atmosphere in Germany before the Euros.
</item 75>

<item 76>
NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers (The Roar | Sports Writers Blog,
2023-10-26T21:11:37+00:00)
The guys are going in the right direction. “I’m not disappointed, I’m stoked.”

Bullets shoot down weakened Wildcats

Advertisement

The Brisbane Bullets have shrugged off the absence of two important players to outwork the Wildcats at their home RAC Arena in Perth and produce a performance
full of heart for their 84-79 NBL win.

Both teams were 2-4 coming into Friday’s match and on losing slides, but it was the Wildcats who had the pressure building, even with the Bullets being without
key pair, the suspended Aron Baynes and injured Shannon Scott.

Brisbane scored the first seven points before opening up a 20-point lead in the third quarter and going on to win by five after holding out against a furious
late charge from the home team.

The team effort of the Bullets was outstanding as they got another 17 points from captain Nathan Sobey.
</item 76>

<item 77>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The third German goal was emblematic of this lack of chemistry between the two U.S. center-backs: Richards stepped up to guard a free Niklas Fullkrug who got by
Sergino Dest, and Ream was unaware as that left Leroy Sane free on goal himself. These two just didn't have it, and Ream was powerless to pick up the slack.

Jamal Musiala dances through the USA defense and Germany lead 3-1 😮‍💨



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/PdjsjkObZR — B/R Football (@brfootball) October 14, 2023

LB: Sergino Dest — 4

If it were possible to give Sergino Dest two different scores, one for the attacking half and one for the defensive half, we'd do it, but that's not how this
works.
</item 77>

<item 78>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Nagelsmann's dismissal at Bayern Munich came as a shock to much of the European football landscape, as the club had performed exceptionally in the Champions
League and was still very much in the title race, sitting just a point behind Dortmund. Now, with a chance for redemption at the helm of the national team,
Nagelsmann will see his time begin with a friendly against the United States.

MORE: A look at the 23-man squad for Gregg Berhalter as USMNT play Germany & Ghana

Germany squad, players for USA friendly

Julian Nagelsmann will be very familiar with much of the 26-man German national team squad, as five players hail from his former club Bayern Munich, including
critical midfield duo Josh Kimmich and Leon Goretzka, who were instrumental in his setup at Allianz Arena.
</item 78>

<item 79>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
UK and Australia: The match is available to stream via CONCACAF's Go app and YouTube channel.

MORE: MLS Cup playoffs schedule

Jamaica vs Canada betting odds & lines
</item 79>

<item 80>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
German national team squad for USA, Mexico friendlies

Pos Player Club Caps Goals GK Oliver Baumann Hoffenheim 0 0 GK Bernd Leno Fulham (ENG) 9 0 GK Marc-Andre ter Stegen Barcelona (SPA) 36 0 GK Kevin Trapp
Eintracht Frankfurt 7 0 DEF Robin Gosens Union Berlin 18 2 DEF Mats Hummels Borussia Dortmund 76 5 DEF David Raum RB Leipzig 18 0 DEF Antonio Rudiger Real
Madrid 62 2 DEF Niklas Sule Borussia Dortmund 47 1 DEF Jonathan Tah Bayer Leverkusen 17 0 DEF Malick Thiaw AC Milan (ITA) 2 0 MID Robert Andrich Bayer
Leverkusen 0 0 MID Julian Brandt Borussia Dortmund 44 3 MID Chris Fuhrich VfB Stuttgart 0 0 MID Leon Goretzka Bayern Munich 53 14 MID Pascal Gross Brighton
(ENG) 2 0 MID Ilkay Gundogan Barcelona
</item 80>

<item 81>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 81>

<item 82>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Antonio was replaced by Salernitana striker Trivante Stewart, but he struggled and was himself withdrawn in the 81st minute. Demarai Gray could step in, but he
himself is struggling for fitness, and while he remains with the team he was not part of the first leg matchday squad. Another option is Colorado Switchbacks
forward Romario Williams, who scored 15 goals in 29 USL Championship matches this season.

Other injuries for Jamaica to contend with include Cory Burke and Amari'i Bell, who both miss out on this international window.

Jamaica projected starting lineup (4-3-3): Blake (GK) — Leigh, Bernard, Pinnock, Lembikisa — Latibeaudiere, D. Johnson, Bailey — Reid, Nicholson, R. Williams.
</item 82>

<item 83>
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
</item 83>

<item 84>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 84>

<item 85>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"
</item 85>

<item 86>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/wdM7tdgfb9#OptusSport pic.twitter.com/n7dqm1qnUH — Optus Sport (@OptusSport) December 23, 2023

Advertisement

Arsenal led through defender Gabriel’s fourth-minute header from Martin Odegaard’s free kick on Saturday (local time), but Mohamed Salah beat David Raya at his
near post to level in the 29th.

Liverpool came closest in the second half and hit the woodwork twice, through Trent Alexander-Arnold and substitute Harvey Elliott.

Mikel Arteta’s team go into Christmas a point clear of Liverpool and third-placed Aston Villa.

Liverpool defender Kostas Tsimikas broke his collarbone when he collided with Jurgen Klopp on the sideline.

Tsimikas was challenged by Arsenal forward Bukayo Saka and stumbled into Klopp, who fell on top of him.
</item 86>

<item 87>
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
</item 87>

<item 88>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Messi missed the last match against Bolivia, and missed four Inter Miami games after, but he returned for a half-hour of action in their final fixture before
the international break. However, he doesn't start, but is reportedly fit enough to come off the bench for most or all of the second half.

Nicolas Tagliafico started both matches in September, and while Marcos Acuna is back with the team following an injury, the former retains his starting spot.
Messi's club teammate Facundo Farias was also included on the roster but it'd be surprising if he saw the field.

Argentina starting lineup (4-3-3): E. Martinez (GK) — Molina, Romero, Otamendi, Tagliafico — De Paul, Mac Allister, E. Fernandez — J. Alvarez, L. Martinez, N.
Gonzalez.
</item 88>

<item 89>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
It's a Concacaf QUARTER-FINAL with Copa América on the LINE! 🔥 pic.twitter.com/sF7P7IkL0v — Canada Soccer (@CanadaSoccerEN) November 17, 2023

1 hour 45 mins to kick: Last time these two teams played, it was a momentous occasion for Canada. Les Rouges ran away as 4-0 winners, qualifying for the 2022
World Cup in the process, their first trip to the big FIFA dance in over 30 years.

#CANMNT's last time out vs. Jamaica 🇯🇲



A convincing 4-0 win as we qualified for the @FIFAWorldCup 💪
</item 89>

<item 90>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
"I would have tried to create a good atmosphere in Germany before the Euros. That's why I would be in all 10 European Championship stadiums for the friendly
matches and would have shown myself there once as a DFB selection. That can create a mood. But what do we want in the US at this point?"
</item 90>

<item 91>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
“I will commit to being the catalyst that will set the tone for our group for every game, to help restore our club to where it deserves to be, I am confident of
that. We will be back and better.”#NERevs pic.twitter.com/A0mVyGdaAZ — Seth (@SethMan31) November 10, 2023

Jamaica vs Canada lineups & team news

Jamaica manager Hallgrimsson also described his side as being on a mission, but they faced it without New York Red Bulls forward Cory Burke because of a groin
problem.

Karoy Anderson, who plays for English club Charlton Athletic, was called up for the first time. "I was just shocked," the midfielder told the Addicks' website.
"I was looking at it for a long time just taking it in and I told my mum and she started crying."
</item 91>

<item 92>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Celtic defender had 22 touches and gave away possession just two times, completing two dribbles and making two clearances. Carter-Vickers was not a target
for Germany to attack nearly as much as the other three defenders on the pitch, a strong nod to his abilities.

65th min: Brenden Aaronson — 7

Probably the the U.S. substitute to make the biggest impact, Brenden Aaronson was bright in the attack, dazzling with his dribbling skills and helping usher the
U.S. forward amidst a second-half onslaught from Germany. A good showing from the Union Berlin winger who needs confidence at the moment.

65th min: Ricardo Pepi — 5

It wasn't an ideal situation for Ricardo Pepi to enter in, coming on with the U.S. shell-shocked from the quickfire Germany double. Still, he provided very
little up front, with just eight touches in 24 minutes of action.
</item 92>

<item 93>
Argentina vs Paraguay live score, result, updates, highlights from CONMEBOL World Cup qualifying (Sporting News, 2023-10-12T23:22:00+00:00)
Lionel Messi's Inter Miami teammate Diego Gomez was named to the Paraguay roster but was forced to withdraw due to injury.

Paraguay starting lineup (5-4-1): C. Coronel (GK) — I. Ramirez, G. Gomez, Balbuena, Alonso, Espinoza — R Sanchez, Campuzano, Almiron, Bareiro — Sosa.

Paraguay subs (12): Rojas (GK), Espinola (GK), R. Rojas, Alderete, Avalos, Sanabria, Villaba, Galarza, M. Rojas, A. Romero, Gimenez, Ojeda.

How to watch Argentina vs Paraguay

Following an agreement at the 11th hour of the September international break, broadcast rights for Argentina's home World Cup qualifiers in the United States
have been secured by Telemundo and Univision.
</item 93>

<item 94>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
“I was able to put lessons from that time into practice, develop and eventually return to the Socceroos much more prepared for the role. It’s been a privilege
to be part of the national team again with the benefit of that experience, and I’m immensely proud of what this group has been able to achieve in the last four
years.

“In my opinion, there’s no greater honour than representing your nation on the international stage. I look at the young players coming through the Socceroos
system now full of pride in the work they’ve put in to get where they are, and the journey that’s ahead of them.

“I’ve always loved being a part of the Socceroos culture and my desire to see Australia fulfil its footballing potential is still driving me to take this group
to further success.”

Australia starts the road to north America in Melbourne at AAMI Park on Thursday night, before travelling to the Middle East to play Palestine next Wednesday
(1.00am AEDT).
</item 94>

<item 95>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He was constantly unsure of where he should be as Germany built forward with incisive pace, and he was one of many U.S. players caught unaware on the first-half
Ilkay Gundogan goal.

McKennie didn't put a first-half pass wrong, going 18/18 before the break and 28/30 overall, but his positioning was all wrong out of possession. McKennie was a
sieve as the U.S. midfield was invisible defensively, leaving their defenders completely unshielded, particularly at the top of the penalty area.

It was only a matter of time before the floodgates opened, and that they did after halftime. McKennie was eventually withdrawn in what felt like one of Gregg
Berhalter's only tactical substitutions of the match.
</item 95>

<item 96>
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
</item 96>

<item 97>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
However, as the 2022/23 season progressed and results weren't as easy to come by, Nagelsmann switched to a 3-2-4-1 formation. This allowed him to deploy the
likes of Alphonso Davies and Kingsley Coman very high up the pitch to provide width, while letting Jamal Musiala and Thomas Muller cook centrally behind a
striker.

That 3-2-4-1 formation is flexible horizontally, as it allows one full-back to push high up the pitch while the other slots in along the back line, and thus
sees the two central No. 10 players alternate pushing wide or pinching in to compensate.

This tactical setup is becoming popular with top-tier coaches across Europe, and has been used at various stages by the likes of Pep Guardiola, Stefano Pioli,
Mikel Arteta, Edin Terzic, and many others.
</item 97>

<item 98>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Jamaica match live, providing score updates, commentary and highlights as they happened.

Canada vs. Jamaica result (full-time)

Score Aggregate Goal scorers Canada 2 4 Alphonso Davies (25)

Ismael Kone (69) Jamaica 3 4 Shamar Nicholson (63), (66)

Bobby De Cordova-Reid (78)

Kickoff: 7:30 p.m. local (7:30 p.m ET / 4:30 p.m.
</item 98>

<item 99>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
Spain's World Cup-winning midfielder spent five years with Vissel Kobe after spending his entire club career until that point with Barcelona, where he played
alongside Messi, Sergio Busquets and Jordi Alba during an era of astonishing success in Catalonia.

"Vissel Kobe is a powerhouse club in Asia coming off of a historic season after winning the J1 League, so we're thrilled about this opportunity to sharpen our
team in a match like this one. We continue to look for the best ways to prepare for 2024, and feel confident this match and this tour will help us do that,"
said Chief Soccer Officer and Sporting Director Chris Henderson.

Boyhood Messi club to face Inter Miami in USA

After their Asia trip, Inter Miami will host a friendly at DRV PNK Stadium against Newell's Old Boys.

The team from Argentina hold a special place in the heart of Messi and Martino.
</item 99>

<item 100>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Gio Reyna looked sharp in the starting lineup for the United States men’s national team, in all the ways a young man can: his unyielding energy, his surprising
eagerness to cover the whole of the midfield, the periodic flashes of skill when the opportunity presented itself, even the bleach-blond hair that allowed
viewers to spot him in an instant.

There was one substantial element missing from his performance, though, in an immense friendly Saturday against Germany: being able to endure from starting to
finishing.

It was kind of like that for the USMNT as a whole, as they blew an early lead and fell, 3-1, in East Hartford, Conn.

MORE: USMNT undone by errors in loss to Germany

It is quite possible this was the most challenging friendly the U.S.
</item 100>
=== END ITEM id=96f33b4c1fab804e ===
