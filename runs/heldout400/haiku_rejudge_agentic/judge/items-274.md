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

This file has 3 items.

=== ITEM id=d6c25119dfe37329 ===
Gold sentences and facts:
Sentence 0: United have also conceded 14 goals in the Champions League group stages, a new and unwanted record for the club that could get worse when they face Bayern in the final match.
  Fact 0.0: Manchester United have conceded 14 goals in the Champions League group stages.
  Fact 0.1: Conceding 14 goals in the Champions League group stages is a new and unwanted record for Manchester United.
  Fact 0.2: The record could get worse when Manchester United face Bayern in their final match.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether.
  Fact 1.1: Manchester United gave a limp display against Bayern at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win at Old Trafford.
  Fact 1.3: Kingsley Coman's 1-0 win was for Thomas Tuchel's side.

Retrieved items:
<item 1>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
The performance this season was so bad that they find themselves out of Europe altogether as the bottom team in the group, although some at the club might find
solace from that. There wasn’t much else. The second place they aimed for has instead gone to Copenhagen, a club with a fraction of United’s budget. That is
where the damage was done.

While that meant there was only ever a slim hope of going through – and the Danish side’s victory over Galatasaray ensured any win would be irrelevant – the
club’s rich legacy in this competition was built on such defiance. There was absolutely none of it here.

There was nothing much at all. Where was the energy? Where was the intensity? The Stretford End eventually implored United to “attack attack attack” but the
complete absence of any such impulses eventually left some of the stadium booing the side off at the end.

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Galatasaray
- Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage -MENTIONS-> Galatasaray
- Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign -MENTIONS-> Galatasaray
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Galatasaray
- Galatasaray -PARTICIPATED_IN-> 3-3 draw with Galatasaray

Entities: Galatasaray, Manchester United, Copenhagen, Bayern Munich, The Independent, Stretford End, Manchester United out of Europe after defeat to Bayern
Munich
</item 1>

<item 2>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
In 2020‑21, they lost their final two group games and were eliminated, while back in 2015-16, Wolfsburg won the deciding group match to knock out United.

Mauro Icardi celebrates scoring the winner for Galatasaray in their 3-2 win at Old Trafford in early October. Photograph: Darren Staples/AFP/Getty Images

When it comes to pressure in the Champions League of late, United do not cope well and there are few events trickier than a night in Istanbul. Pallister smiled
at a bell boy at the team hotel and got a finger across the throat in response, Brian McClair had a cleaner bang on his bedroom ceiling in the night and other
players received numerous phone calls to disrupt their sleep. The current squad could face the same level of malevolence and need to be prepared.

United have done little learning in recent times, repeating the same failures in Europe’s elite competition.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Champions League
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Alex Ferguson
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Eric Cantona
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> riot police
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Turkey

Entities: Manchester United face Galatasaray with high hopes but bad memories, Galatasaray, Wolfsburg, Mauro Icardi, Pallister, Brian McClair, Darren Staples,
Galatasaray's 3-2 win at Old Trafford, Manchester United's final two group games, 2015-16 deciding group match, Champions League, AFP, Manchester United, Old
Trafford, The Guardian
</item 2>

<item 3>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
There was not even the sense of heroic failure that accompanied their quarter-final exit to Bayern in 2010. There was just crushing failure. Because it wasn’t
Bayern who eliminated them. It was Galatasaray, who won at Old Trafford and scored six goals over 180 minutes against United. It was FC Copenhagen, who
qualified instead of them. Rasmus Hojlund’s former club will be in the last 16. His current one won’t. Copenhagen will be the anomalies in the draw, the
outsiders with the spot that had seemed reserved for United.

Manchester United’s defeat means they miss out on a spot in the Europa League (PA)

Instead, they are cast out of continental competition altogether. Europe’s sinking superpower have become just the fourth English team to prop up a Champions
League group when it concluded. There was something predictable about the manner of their demise; their defence was unlocked by Harry Kane, with a lovely flick.

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Galatasaray
- Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage -MENTIONS-> Galatasaray
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Galatasaray
- Manchester United out of Europe after defeat to Bayern Munich -MENTIONS-> Galatasaray
- Galatasaray -PARTICIPATED_IN-> 3-3 draw with Galatasaray

Entities: Galatasaray, Champions League, Bayern, Rasmus Hojlund, Manchester United, Erik ten Hag, Old Trafford, FC Copenhagen, Europa League, quarter-final exit
to Bayern, Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign, Harry Kane, Europe, The Independent
</item 3>

<item 4>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
The 1990s were a hell of a decade. They still are around here.

'We've lost decisive players': Ten Hag says injuries were key to Champions League exit – video

So much for the good news then. The bad news is that Ten Hag’s United have now compiled arguably the worst performance by any English team in a Champions League
group stage. OK, it’s no Blackburn Rovers. But there are extra marks here for United’s pedigree, for the fact the group really wasn’t much of a group, and for
United managing to finish bottom, shipping 15 goals in the process, more than any other English team ever.

Plus of course there were just so many moments of farce along the way, a six-part saga of chaos, collapse and laughter in the dark. We remember Onana diving out
of the way of the ball in Munich. We remember Onana also diving out of the way of the ball in Istanbul.

Graph facts:
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Blackburn Rovers
- The Soccer Star on the Verge of Breaking an 89-Year-Old Record -MENTIONS-> Blackburn Rovers
- Jon Dahl Tomasson -WORKS_FOR-> Blackburn Rovers
- Blackburn Rovers -PARTICIPATED_IN-> Chelsea vs Blackburn Rovers
- Blackburn Rovers -PARTICIPATED_IN-> Blackburn Rovers vs Cardiff City

Entities: Blackburn Rovers, Onana, Manchester United, Champions League, The Guardian, Istanbul, Barney Ronay, Ten Hag’s Manchester United stand up and die at
club stuck in the past tense | Barney Ronay, Munich, Ten Hag
</item 4>

<item 5>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
This is the current reality for United. They obviously have many more issues than the goalkeeper – as illustrated by the goal itself – but a strange season for
Ten Hag’s primary signing only continues. United’s problems meanwhile combined. There were so many errors leading to the goal, to go with all of those that have
characterised this entire European campaign.

That can now be talked about definitively, since it has been decisively ended. It is a low but just part of a decade of disappointment in the competition that
has seen this club rise to its most emotional heights.

This game never threatened to do that. United never threatened at all. They’d been involved in so many ludicrously chaotic matches that the biggest of all ended
up being a non-event.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Manchester United
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Manchester United
- Everton players and fans will be fired up by points penalty, asserts Sean Dyche -MENTIONS-> Manchester United
- Carabao Cup fixtures, draws, results, TV channel, and live streams for every round of EFL League Cup 2023/24 -MENTIONS-> Manchester United
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Manchester
United

Entities: Manchester United, Bayern Munich, The Independent, Manchester United out of Europe after defeat to Bayern Munich, European campaign, Ten Hag
</item 5>

<item 6>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
United now have to beat Bayern Munich and hope Galatasaray draw with FC Copenhagen on the final matchday to progress, and it's the third time in the Champions
League this season that they have been in a winning position and then failed to emerge victorious.

Eight points have been dropped when they've been in the lead, and no other team has suffered a worse fate this term.

Captain Fernandes appears to be fed up of their inability to hold onto results as he told TNT Sports: "Every time we concede a goal we had many chances to score
again.

"I think at the end of the game we had like four chances to score a goal but we have to get our heads up and understand that sometimes it's about finding
another pass, finding the player that is free.

"Trying to get the best one in the best position and take the chance ourselves, we have to look for the team first and after, for ourselves."

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> final match
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Kerem
Akturkoglu
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> 3-3 draw with
Galatasaray
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Hakim Ziyech
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Ziyech

Entities: Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe, TNT Sports,
Galatasaray, Fernandes, Manchester United, FC Copenhagen, Champions League, talkSPORT, Europe, Bayern Munich
</item 6>

<item 7>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Another infamous European night at Old Trafford. It should be a landmark low, but almost the most deflating element of this 1-0 defeat to Bayern Munich was how
grimly routine it all felt.

Manchester United, one of the grandest and richest names in this competition, have again gone out of the Champions League at the group stage. It is the third
time it has happened since Sir Alex Ferguson, meaning it’s their most common round of exit in that period.

Graph facts:
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Champions League
- Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United -MENTIONS-> Champions League
- Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage -MENTIONS-> Champions League
- Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world -MENTIONS-> Champions League
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Champions League

Entities: Champions League, Manchester United, Old Trafford, Sir Alex Ferguson, Bayern Munich, The Independent, 1-0 defeat to Bayern Munich, Manchester United
out of Europe after defeat to Bayern Munich
</item 7>

<item 8>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
4 Fernandes criticised their lack of ability to kill games off Credit: Getty

He added: "It's too bad to be honest, I don't want to be too negative but for what we've been doing in the Champions League is not enough, we have to control
our games every time.

"It's not the first game we've had like this, even with FC Copenhagen, I understand we had one player less but we scored to make it 3-2 and we had to secure the
result.

"We have to understand in these kind of games, in these kind of stages, you have to be smart enough and manage the game better."

United have also conceded 14 goals in the Champions League group stages, a new and unwanted record for the club that could get worse when they face Bayern in
the final match.

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Hakim Ziyech
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Kerem
Akturkoglu
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> 3-3 draw with
Galatasaray
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> TNT Sports
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Ziyech

Entities: Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe, final match,
Bayern, Fernandes, Manchester United, FC Copenhagen, Champions League, talkSPORT, Europe
</item 8>

<item 9>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Manchester United's domestic form has certainly shown signs of improvement in recent weeks, but that's far from the case in Europe.

Erik ten Hag's side have once again dropped points from a winning position as they drew 3-3 with Galatasaray in the Champions League, a damaging result in their
bid to reach the last 16.

4 United were once again the victims of their own demise Credit: Getty

Goals from Alejandro Garnacho and Bruno Fernandes fired them into a two-goal lead within 18 minutes before Chelsea loanee Hakim Ziyech struck back.

Scott McTominay scored what he thought would have been a crucial third shortly after the break but United failed to hold on.

Another free-kick from Ziyech caught Andre Onana off guard once again and a strong strike from Kerem Akturkoglu levelled the scoring.

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Ziyech
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> defence
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> final match
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> TNT Sports
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Gabby
Agbonlahor

Entities: Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe, Hakim Ziyech,
Kerem Akturkoglu, 3-3 draw with Galatasaray, Galatasaray, Chelsea, Alejandro Garnacho, Manchester United, Erik ten Hag, Champions League, Scott McTominay,
talkSPORT, Bruno Fernandes, Europe, Andre Onana
</item 9>

<item 10>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
They're easy saves and he's cost his team.

"If Onana doesn't make those two mistakes, I think United go on to win 4-1.

"He's so laid back on the ball that he can't switch on, he's late to dive for the free-kicks."

4 United now face a huge task in their hope of reaching the last 16 Credit: AFP

United's 12 goals in the group stage is the fifth highest tally out of any club but their defensive woes have been their downfall, and former defender Rio
Ferdinand cites a lack of balance as to why.

The 2007/08 Champions League winner with Man United said: "I think they've got to look at themselves.

"You can't go into a Champions League campaign and score nine goals in three away games and have one point at the end of it. I think that shows you that there's
a lack of balance in the team at the moment.

Graph facts:
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Hakim Ziyech
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> Kerem
Akturkoglu
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> 3-3 draw with
Galatasaray
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> TNT Sports
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> final match

Entities: Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe, Rio Ferdinand,
2007/08 Champions League, AFP, Onana, Manchester United, talkSPORT, Europe
</item 10>
=== END ITEM id=d6c25119dfe37329 ===

=== ITEM id=8e554d93c93efb79 ===
Gold sentences and facts:
Sentence 0: The Bears' defense is all over the Vikings tonight, and Minnesota can't get out of its own way.
  Fact 0.0: The Bears' defense is all over the Vikings tonight.
  Fact 0.1: Minnesota cannot get out of its own way.
Sentence 1: Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least 69% of their snaps.
  Fact 1.0: Week 4 was the first time all year the Vikings did not throw the ball on at least 69% of their snaps.
Sentence 2: 11:03 p.m. — The Vikings are using their two timeouts, but there isn't much they'll be able to do once this hits fourth down.
  Fact 2.0: At 11:03 p.m., the Vikings are using their two timeouts.
  Fact 2.1: There is not much the Vikings will be able to do once this hits fourth down.
Sentence 3: The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running game for Kevin O'Connell and the defense overall under Brian Flores.
  Fact 3.0: Josh Dobbs is the new man on the Vikings.
  Fact 3.1: Josh Dobbs keeps producing the same offensive results the Vikings enjoyed with Kirk Cousins.
  Fact 3.2: Josh Dobbs's results have more help from the running game under Kevin O'Connell.
  Fact 3.3: Josh Dobbs's results have more help from the defense overall under Brian Flores.

Retrieved items:
<item 1>
Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback (Sporting News, 2023-12-07T22:06:00+00:00)
Matthew Stafford (@ Ravens), Geno Smith (@ 49ers), and Josh Dobbs (@ Raiders) are all risky against some of the league's better pass defenses. Stafford faces a
sound Ravens stop unit that entered their Week 13 bye allowing the second-fewest FPPG to QBs.

There's a chance the Vikings opt to start fellow QB Nick Mullens over Dobbs after his debacle last Monday night against the Bears, but if Minnesota elects to
stick with Dobbs, his rocky Week 12 outing concerns us against a Raiders defense that's been one of the more efficient pass defenses since Week 5, sitting
eighth in dropback EPA.

On the other hand, after an elite Week 13 effort in Seattle's one-possession loss to Dallas, fantasy owners might go looking to chase points with Smith, but
replicating last week's showing against the 49ers is a completely different ask for a QB who's underwhelmed this season.

Graph facts:
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit -MENTIONS-> Raiders
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders

Entities: Raiders, Vikings, Josh Dobbs, Dallas, fantasy football, Ravens, Sporting News, Geno Smith, Bears, 49ers, Matthew Stafford, Nick Mullens, Seattle,
Seattle's one-possession loss to Dallas, Pass defenses
</item 1>

<item 2>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, The New York Times, Eagles, Jalen Hurts, DeVonta Smith, Super Bowl, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders, A.J.
Brown, Dallas Goedert
</item 2>

<item 3>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Brady, Eagles, Hurts, Super Bowl, Zach Wilson, Jets, N.F.L., Manning, Rodgers, Eagles game, the exec, a veteran coach, WPA (win
probability added), two-minute
</item 3>

<item 4>
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

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> New York Giants
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> New York Giants
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> New York Giants
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> New York Giants
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> New York Giants

Entities: New York Giants, The New York Times, Kansas City, Minnesota Vikings, Super Bowl, N.F.L., N.F.L. Executives on the Biggest Questions Facing Super Bowl
Contenders, Denver, Green Bay, playoffs, Week 9 bye, football
</item 4>

<item 5>
Graph query row:
publisher: CBSSports.com
team: Atlanta
game: Week 4
page: NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
passage: NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Terrell on Collins and make things a little harder on the Texans. Terrell shadowed a receiver for the first time this season in Week 4, contributing to Calvin
Ridley's modest day. It ultimately would mean multiple bodies on Dell.

Sit Him (Lineup Decisions) Projections powered by Sportsline Kyle Pitts TE ATL Atlanta • #8

Age: 23 • Experience: 3 yrs. Weekly Breakdown MATCHUP vs HOU ATL -2 O/U 41.5 OPP VS TE 18th PROJ PTS 7.6 TE RNK 13th YTD Stats REC 11 TAR 21 REYDS 121 TD 0
FPTS/G 5.8 If Coach Smith is right, then holding on to Pitts in Fantasy makes sense. But it's hard to see the good in his game so far this season beyond him
finding holes in zone coverage.
</item 5>

<item 6>
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

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Vikings, Minnesota, Atlanta, Dolphins, Desmond Ridder, Super Bowl, Justin Jefferson, N.F.L., N.F.L. Executives on the Biggest
Questions Facing Super Bowl Contenders, Kirk Cousins, quarterback, two-minute drill, Giants, draft
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Especially when you face a defensive coordinator who says, ‘You show me you can beat me throwing the ball outside the numbers — best of luck, and to do that,
you are going to have to take a five-step drop and I’m going to have Myles Garrett chasing you every play.'”

Advertisement

• Kansas City Chiefs: Is the explosive passing game gone for good?

What I think: The Chiefs have never scored less per game on offense through six games or a full season since Patrick Mahomes became their quarterback. They have
also never enjoyed larger average point differentials (9.8 per game), because their defense is so much better. I think the pass offense will become slightly
more explosive as Mahomes gets reps with his receivers, but I’m not expecting a huge shift.

The Chiefs are averaging 9.5 plays per touchdown drive when starting at least 70 yards from the opponent’s goal line, per TruMedia.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Patrick Mahomes, Super Bowl, N.F.L., TruMedia, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders, Myles
Garrett, Kansas City Chiefs
</item 7>

<item 8>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Lamar Jackson
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Lamar Jackson
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Lamar Jackson
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Lamar Jackson
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Lamar Jackson

Entities: Lamar Jackson, The New York Times, Baltimore Ravens, Super Bowl, N.F.L., N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders
</item 8>

<item 9>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Detroit Lions: Have they fixed their defense?

What I think: The Lions are 11.3 EPA per game better on defense from last season to this season, the second-largest gain for any team since last season. The
schedule has something to do with that, but the 2022 defense was so bad, it couldn’t stop Carolina. Detroit will win the NFC North and return to the playoffs,
which will make this season successful no matter what happens from there. But we won’t find out until the postseason just how much better the defense might be,
because the remaining schedule features so many weak offenses. That includes Chicago twice, Green Bay, Denver and Las Vegas.

Exec comment: “Yes, I do buy them. This is a league full of teams who do not really know who they are week to week. Detroit knows who they are, and they play to
their identity. Anytime you know who you are and play to that style, you have a chance to be dangerous.”

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Carolina, Chicago, Super Bowl, Detroit Lions, N.F.L., Las Vegas, N.F.L. Executives on the Biggest Questions Facing Super Bowl
Contenders, Denver, Green Bay, playoffs, NFC North
</item 9>

<item 10>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Philadelphia Eagles
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Philadelphia Eagles
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Philadelphia Eagles
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Philadelphia Eagles
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Philadelphia Eagles

Entities: Philadelphia Eagles, The New York Times, Travis Kelce, Mahomes, Jalen Hurts, Super Bowl, Chiefs, N.F.L., Chris Jones, explosive play rates, explosive
passing, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders
</item 10>
=== END ITEM id=8e554d93c93efb79 ===

=== ITEM id=8298a20b11a5a5d5 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that 'all of that, all of it, was built on lies'.
Sentence 1: And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.
  Fact 1.0: Bankman-Fried left Jane Street.
  Fact 1.1: Jane Street is a high-frequency trading firm.
  Fact 1.2: Bankman-Fried, the former billionaire, got his start in finance at Jane Street after MIT.
  Fact 1.3: Bankman-Fried persuaded Yedidia to join him as a trader at Alameda.
  Fact 1.4: Bankman-Fried later persuaded Yedidia to join him as a developer at FTX.
Sentence 2: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 2.0: Bankman-Fried, known as SBF, has pleaded not guilty to some seven charges.
  Fact 2.1: The charges are of fraud and conspiracy.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 3.1: The defense argued that the FTX founder Bankman-Fried acted in good faith.
  Fact 3.2: The defense argued that Bankman-Fried never meant to commit fraud or steal.
  Fact 3.3: The defense argued that Bankman-Fried basically got in over his head.

Retrieved items:
<item 1>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.

Graph facts:
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> cryptocurrency
- The people who ruined the internet -MENTIONS-> cryptocurrency
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> cryptocurrency
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> cryptocurrency
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> cryptocurrency

Entities: cryptocurrency, FTX trial, Southern District of New York, Christopher LaVigne, Withers, transactions, Bankman-Fried’s tweets, indictment, The FTX
trial is bigger than Sam Bankman-Fried, FTX, Sam Bankman-Fried, The Verge
</item 1>

<item 2>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Caroline Ellison, Bankman-Fried, Alameda Research, wire fraud, Nishad Singh, embezzlement, money laundering, conspiracy, political
donations, In the end, the FTX trial was about the friends screwed along the way, Joe Bankman, FTX trial, FTX, Yedidia
</item 2>

<item 3>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Assistant United States Attorney Nicolas Roos questions Gary Wang during Sam Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt cryptocurrency
exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

An $8 billion bug

Adam Yedidia, who was the prosecution's second witness on Wednesday, continued his testimony on Thursday. Yedidia met Bankman-Fried in college at MIT, and the
pair remained close friends. Yedidia, assuming a robotic posture on the stand, worked out of FTX's Hong Kong office from January to October of 2021 and then in
the Bahamas until last year's collapse. In his testimony, he referred to a group Signal thread called "People of the House," referring to Bankman-Fried's $35
million penthouse, where many employees lived.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Adam Yedidia, New York City, Signal, Bahamas, Jane Rosenberg, FTX, Nicolas Roos, Sex, Signal messages, and sabotaging FTX's code: SBF execs
and Bahamas roommates tell all in court, Federal Court, People of the House, Sam Bankman-Fried's fraud trial, $35 million penthouse, Sam Bankman-Fried, Reuters
</item 3>

<item 4>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Ellison
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Ellison
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Ellison

Entities: Ellison, New York, Jane Rosenberg, FTX, U.S. attorney's office, courthouse, pleading guilty, Wang's testimony, Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court, Wang, hearing, Sam Bankman-Fried, Reuters, Manhattan, CNBC
</item 4>

<item 5>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Ellison is expected to take the stand under a similar arrangement.

U.S. District Judge Lewis Kaplan presides as Gary Wang testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

Born in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to the Massachusetts Institute of Technology to study math and computer
science. He worked at Google after college. Wang, who first met Bankman-Fried during high school at a summer camp, owned 10% of Alameda, while his boss owned
the other 90%. Wang told the court about the advantages that Alameda received by having code baked into FTX's software that allowed special access to the crypto
exchange.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, New York City, Lewis Kaplan, Alameda, Minnesota, cryptocurrency, Signal, Bahamas, Ellison, China, Jane Rosenberg, FTX, Sex, Signal
messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court, FTX software, fraud trial of Sam Bankman-Fried
</item 5>

<item 6>
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

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Damian Williams
- Damian Williams -SAID-> financial fraud
- Damian Williams -PARTICIPATED_IN-> Press conference
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto

Entities: Damian Williams, crypto, alleged crimes, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take
the stand, Fortune, FTX, the government, Sam Bankman-Fried, the trial, MIT, fraud
</item 6>

<item 7>
Graph query row:
person: Sam Bankman-Fried
relationship: MENTIONS
related: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
report: None
publisher: None
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Adam Yedidia, New York City, Bahamas, Jane Rosenberg, FTX, Nicolas Roos, Sam Bankman-Fried fraud trial, Federal Court in New York City,
financial crimes, Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court, Sam Bankman-Fried, Manhattan, Reuters,
CNBC
</item 8>

<item 9>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 9>

<item 10>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Bankman-Fried faces seven federal charges, including wire fraud, securities fraud and money laundering, that could put him in prison for the rest of his life.
Thus far, Bankman-Fried, 31, has remained mostly quiet in court intently listening to witnesses and at times writing notes to his attorneys. But as Wang
testified against him, Bankman-Fried looked visibly upset, shifting his gaze from his former friend to the ground, and at one point putting his head in his
hands.

Sam Bankman-Fried listens as Assistant U.S. Attorney Nicolas Roos questions Gary Wang during Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Wang, 30, was technology chief for FTX, which spiraled into bankruptcy in November.
</item 10>
=== END ITEM id=8298a20b11a5a5d5 ===
