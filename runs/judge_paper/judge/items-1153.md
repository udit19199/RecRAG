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

=== ITEM id=b7550c4902e9b9ef ===
Gold sentences and facts:
Sentence 0: For all their own flaws Bayern are basically a good version of whatever it is United are attempting to piece together.
  Fact 0.0: Bayern have their own flaws.
  Fact 0.1: Bayern are basically a good version of whatever Manchester United are attempting to piece together.
Sentence 1: Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's side.
  Fact 1.0: Manchester United are out of Europe altogether.
  Fact 1.1: Manchester United put in a limp display against Bayern at Old Trafford.
  Fact 1.2: Kingsley Coman sealed a 1-0 win at Old Trafford.
  Fact 1.3: Kingsley Coman's 1-0 win was for Thomas Tuchel's side.

Retrieved items:
<item 1>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.

“After the break, we were more compact and certainly more energetic, hungrier and more determined,” Milan’s coach, Stefano Pioli, told DAZN. “That is a great
sign when playing every three days and it was a fully deserved victory.”

Milan suffered a potential injury blow as the midfielder Ruben Loftus-Cheek limped off in the first half. “It would be a shame if we lost Ruben, he felt a
twinge near the groin area,” Pioli said. “Unfortunately, injuries tend to come when players are in good shape.”
</item 1>

<item 2>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
after newsletter promotion

This is what Ten Hag has overseen in recent weeks as the protracted sale of the club has entered a new phase, with Sir Jim Ratcliffe’s bid to buy 25% for an
inflated £1.3bn in return for control of football policy expected to go through imminently. But despite any implications this may have for the futures of Ten
Hag’s players, the manager himself does not think the matter is a distraction.

Bruno Fernandes volleys home for Manchester United against Burnley – one of two goals he has scored for the club this season. Photograph: Matthew
Peters/Manchester United/Getty Images

“Of course, they are committed to the club but especially they are committed to this team,” Ten Hag says. “They want to be successful and are therefore aware
you need [to focus on] the team, your teammates. That is how the players are thinking and they are not thinking about strategic reviews or structures or
whatever.”

On Sunday at 3.30pm we will find out just how focused United’s players are.
</item 2>

<item 3>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Photograph: Michael Regan/Getty Images

Ensuring a working relationship with the Glazers

For paying an overpriced £1.3bn for his 25% Ratcliffe obtains a major element of control but still the Glazers remain majority owners. The 71-year-old oversees
football policy but these do not exist in a vacuum given that the commercial side affects this and vice versa. Ratcliffe accepts Manchester United have an
impressive revenue-based model that makes the club akin to a cash machine but he is not enamoured of how the money generated is invested – primarily in the
squad. What happens then, for example, if Ten Hag wants a new midfield ace, Ratcliffe agrees but the Glazers cite a downturn in income and thus seek to block
the signing?
</item 3>

<item 4>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
In 2020‑21, they lost their final two group games and were eliminated, while back in 2015-16, Wolfsburg won the deciding group match to knock out United.

Mauro Icardi celebrates scoring the winner for Galatasaray in their 3-2 win at Old Trafford in early October. Photograph: Darren Staples/AFP/Getty Images

When it comes to pressure in the Champions League of late, United do not cope well and there are few events trickier than a night in Istanbul. Pallister smiled
at a bell boy at the team hotel and got a finger across the throat in response, Brian McClair had a cleaner bang on his bedroom ceiling in the night and other
players received numerous phone calls to disrupt their sleep. The current squad could face the same level of malevolence and need to be prepared.

United have done little learning in recent times, repeating the same failures in Europe’s elite competition.
</item 4>

<item 5>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Uefa’s description of Onana intrigues because it suggests that what Guardiola introduced to the domestic game – a No 1 in Ederson who performs as a de facto
extra defender to maintain fluidity along the backline – has evolved further. The goalkeeper was so advanced as to be a No 6 (rather than, the technical report
posited, a “sweeper-keeper”) who can take up residence in the middle third of his half as his team pushes deep into enemy territory.

Onana did so against City for Internazionale but has this encounter with the champions come too soon for him to function as “the extra man” Ten Hag identifies?
“It also depends on the opposition, how strong is the press, for instance, and how good is the team around him,” the United manager says.

“So if it’s possible – in my philosophy he has to do it every game [but] it is not always possible, so you have to adapt and this what I make clear.
</item 5>

<item 6>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
With United out of Europe, there are questions about Erik ten Hag’s ability to coach the team (AFP via Getty Images)

These are not the sounds we are used to hearing after that Champions League theme. Something does stir when the anthem is played at Old Trafford amid the
lights. Well, for most, that is, except the players.

This is why there are growing questions for Erik ten Hag himself, even if everyone is constantly keen to put it in a wider context.

Manchester United should not be performing like this on what is supposed to be the classic European night at Old Trafford. This just reminded that all they are
going to have for some time are their memories.

It was another utterly forgettable display.

Bayern, already through but perhaps looking to make a point after their own 5-1 defeat to Eintracht Frankfurt, didn’t even need to step it up to win. They were
quite low-intensity themselves, Kingsley Coman’s winning goal coming from the forward almost walking through the middle of their team.
</item 6>

<item 7>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
They defeated Honved before needing a late intervention from Cantona to avoid a deficit heading to Istanbul, where they were greeted at the airport by a horde
of Galatasaray supporters providing a wall of noise and the infamous “Welcome to Hell” banner.

“That was our first foray in the Champions League and we were expected to go through,” says the former United defender Gary Pallister, who was injured for the
game and so had to watch proceedings from the stands. “The hostility is something [to learn from] – I never experienced anything like that in the rest of my
career. You look at that and think it can’t get any worse.”

Thirty years on United head to Turkey knowing they have rarely been convincing in the Champions League in recent years. A 3-1 away win against Paris
Saint-Germain in the 2018-19 round of 16 after losing the home leg 2-0 was the last time they won a crunch match.
</item 7>

<item 8>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
After all that pressure, all that tender hope, it looked a bit like a training goal.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

And that was pretty much that. From here United’s season continues to narrow. They can’t win the league now, can’t win the Champions League, can’t win the
Carabao Cup.
</item 8>

<item 9>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
The fears came true and they lost 3-0 in the Manchester derby. They had some hope in their next match in the EFL Cup round of 16 match against Newcastle United.

The boys in blue 🤩 pic.twitter.com/0awNdmkXtT — Manchester City (@ManCity) November 3, 2023

Especially since they defeated the Magpies last term in the final of the EFL Cup. However, the reigning Carabao Cup champions could not defend their title any
longer. They lost by another 3-0 margin to Newcastle United. With 5 wins and 5 defeats in the domestic league, they are now sitting in 8th position in the
table. It still seems that coach Erik Ten Hag has the backing of the management. However, he will still have to pull up his team from the doom as soon as
possible.

Fulham are in a worse situation than the Red Devils.
</item 9>

<item 10>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
It was already a drab game, with a muted crowd.

It didn’t help that Bruno Fernandes blazed United’s best opportunity over the bar after good work from Aaron Wan Bissaka. It looked like it was going to be one
of those games. It certainly wasn’t one of those big European nights at Old Trafford.

On 71 minutes, Bayern managed what they had been half-heartedly threatening all game. They also did it at near walking pace. Kingsley Coman had weaved his way
through the centre, for the ball to eventually be worked to Harry Kane. The English forward clipped an artful ball back through for Coman who finished
emphatically.

Kingsley Coman’s goal ensured Manchester United were knocked out of Europe (PA)

It was almost too easy for him. Why did Andre Onana stay so close to his line? If that sounds like the set-up for a joke… well, you can write your own
punchline.

This is the current reality for United.
</item 10>

<item 11>
European football: Eintracht Frankfurt hit Bayern for five in famous thrashing (The Guardian, 2023-12-09T20:46:25+00:00)
Eintracht put the ball in the net again, five minutes after the restart, to make it 4-1, outplaying Bayern with another textbook quick break and with Ebimbe
beating Manuel Neuer for his second goal of the afternoon. Ansgar Knauff then slotted in to make it 5-1 on the hour as Frankfurt became the first team in 48
years to score five goals in one hour against Bayern in the Bundesliga since their own 6-0 win in 1975.

Quick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhone or the Google Play store on Android
by searching for 'The Guardian'.

If you already have the Guardian app, make sure you’re on the most recent version.

In the Guardian app, tap the Menu button at the bottom right, then go to Settings (the gear icon), then Notifications.

Turn on sport notifications. Was this helpful? Thank you for your feedback.
</item 11>

<item 12>
European football: Eintracht Frankfurt hit Bayern for five in famous thrashing (The Guardian, 2023-12-09T20:46:25+00:00)
Internazionale defeated lowly visitors Udinese 4-0 with a dominant performance highlighted by a superb long-range effort from Lautaro Martínez to regain the
Serie A top spot on Saturday. Inter moved two points clear of second-placed Juventus, who had a 1-0 home win over champions Napoli on Friday.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion



Inter took the lead with a penalty by Hakan Calhanoglu in the 36th minute after a foul by defender Nehuén Pérez on Martínez which was reviewed by VAR.
</item 12>

<item 13>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Nostalgia underpinned Solskjaer’s management. Ten Hag cannot call upon his personal history in such a way; the Champions League final he almost reached was with
Ajax. His past does not offer the protection that means his name will always be sung at Old Trafford.

And, 24 years after Solskjaer’s stunning intervention, another United Champions League campaign finished against Bayern Munich. Not in glory, but ignominy. Like
Solskjaer, Kingsley Coman has a winner in a Champions League final; this time the winger finished off United, condemning them to a seventh home defeat of the
season. In a sense, it may have been the most respectable: United were vaguely competent on the night, but utterly incompetent over the course of a wretched
campaign.

There was no hint of a repeat of their most famous comeback against Bayern, ensuring Solskjaer’s name will forever be bracketed with Teddy Sheringham’s.
</item 13>

<item 14>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 14>

<item 15>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
Christian Pulisic’s equaliser threatened to condemn them to the Europa League. Samuel Chukwueze’s winner instead meant that spot goes to AC Milan. The group of
death went down to the dying minutes, but Newcastle have perished. They had savoured a return to Europe but there will be no continental trips for them in the
spring of 2024.

And, after Kylian Mbappe’s controversial 98th-minute leveller in Paris, they can rue the loss of another lead. Once again, they may feel entitled to believe
they should have won, but for different reasons. For much of the night, they were rampant and had Milan reeling. Yet, last season’s Champions League
semi-finalists hung in the game, scored twice and hit the woodwork twice more. For Newcastle, a third consecutive defeat may come with the sense that their
injuries came at a cost.
</item 15>

<item 16>
Man United vs Chelsea head to head record, all-time results between Premier League giants (Sporting News, 2023-12-05T20:30:03+00:00)
Chelsea have won the Champions League and Europa League twice each, while United have won the former trophy three times and the latter once.

United's tally of 12 FA Cup triumphs is eight more than Chelsea have claimed, but both sets of supporters will be hoping their teams can become superpowers
again after barren periods by their 21st-century standards.

The rise of mega-wealthy Manchester City to become arguably the best team in the world has not helped, and nor has Liverpool's resurgence under Jurgen Klopp.
United finished well off the pace albeit in a respectable third at the end of 2022/23, a season in which Chelsea came 12th – their lowest final position for 29
years.
</item 16>

<item 17>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
"So I think there's a lot of question marks in the performances and the way they're going about things at the moment but balance for me is one of the key
aspects that they've got to really find.

"We mentioned in commentary, scars from throwing away leads, those are hard to get away, you've got to really manage games and he mentioned that himself Bruno,
you've got to go out managing games correctly.

"They've got to do that not over one or two or three games, they've got to do that over a period of games now so they can erase those scars that they've created
and they'e grown through bad results and throwing away leads, especially in Europe."
</item 17>

<item 18>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?
</item 18>

<item 19>
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
</item 19>

<item 20>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
As the seconds ticked down at the end of a Champions League season‑ender that felt like a gentle, even quite tender, act of sporting euthanasia, the only noise
inside Old Trafford was the sound of the Bayern Munich fans singing an impressively sustained version of Football’s Coming Home. A little later they sang Is
This A Library? Actually, no, it’s more of a museum. They chanted “Auf Wiedersehen” at the departing members of the home support who, frankly, had done pretty
well to stick it out to that point.

The good news for Erik ten Hag at the end of this low-fi 1-0 defeat is that United’s players didn’t lie down and die at Old Trafford. Instead they stood up and
died, running hard and creating a kind of simulacrum of a functioning elite team.

This was not a group of players in revolt or drained of spirit. They looked like what they are, a muddled creation, stitched together out of parts and off-cuts.
</item 20>

<item 21>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
29 Galatasaray 3-3 Manchester United Nov. 29 Bayern Munich 0-0 FC Copenhagen Dec. 12 FC Copenhagen 1-0 Galatasaray Dec. 12 Manchester United 0-1 Bayern Munich

Group B

Team GP W D L GD P Arsenal (Q) 6 4 1 1 +12 13 PSV (Q) 6 2 3 1 -2 9 Lens 6 2 2 2 -5 8 Sevilla 6 0 2 4 -5 2

Arsenal finished top of the pile and their 6-0 thrashing of Lens on matchday five confirmed them as winners of Group B.

PSV's comeback win over Sevilla meant have also qualified thanks to Lens' defeat. The Ligue 1 side beat their LaLiga opponents 2-1 to make sure of a Europa
League berth as PSV and Arsenal played out an entertaining 1-1 draw.
</item 21>

<item 22>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
Bayern Munich struck twice in the second half to come from two goals down and snatch a 2-2 draw at RB Leipzig but the champions dropped to third in the
Bundesliga with their second draw of the season.

Goals from Leroy Sané and Harry Kane cancelled out Leipzig’s first-half lead but the result left Bayern on 14 points, two behind the leaders, Bayer Leverkusen.
Leipzig, who host Manchester City in the Champions League next week, are fifth, a further point behind.

“Two very different halves from our perspective,” the Bayern coach, Thomas Tuchel, said. “In the first we started well but only for the first two minutes and
missed that huge chance.

“We were too slow, made too many mistakes and were not courageous enough. After the break we showed a different mentality. It was a bit wild, with a lot of risk
but it is a positive result, not to lose after being 2-0 down. We showed character.” Bayern travel to Copenhagen in the Champions League next week.
</item 22>

<item 23>
Man United vs Chelsea head to head record, all-time results between Premier League giants (Sporting News, 2023-12-05T20:30:03+00:00)
Under the brash Jose Mourinho – an antagonist to his legendary United counterpart Sir Alex Ferguson – Chelsea embarrassed third-placed United in 2004/05,
winning the trophy while finishing 18 points ahead of them and, at one point, forcing Ferguson's players to give them a guard of honour at Old Trafford because
of the timing of their title win.

United closed the gap to finish eight points behind Chelsea in second the following season, then returned to the top in 2006/07, the deposed champions coming
six points short of their total.

Two points separated the teams as United won the Premier League again in 2007/08, and there would be one more changing of the guard to come when Chelsea
finished first above United in 2009/10 – only for United to finish top a year later.

That ultra-tight Champions League final encapsulated how close the rivalry had become in a competitive sense, and both teams have won the Premier League since
then, leaving United on a record 20 titles and Chelsea on six.
</item 23>

<item 24>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
Paris Saint-Germain join them in the knockouts after a 1-1 draw between the teams at Signal Iduna Park — a result that also confirmed Dortmund as group winners.

Newcastle were briefly in second spot during the final matchday before letting their lead slip to AC Milan, who snatched the Europa League spot in the process.

The Rossoneri finished level on points with PSG but behind them based on their head-to-head tiebreaker.

Results

Date Fixture/Result Sept. 19 Milan 0-0 Newcastle Sept. 19 PSG 2-0 Dortmund Oct. 4 Dortmund 0-0 Milan Oct. 4 Newcastle 4-1 PSG Oct. 25 PSG 3-0 Milan Oct. 25
Newcastle 0-1 Dortmund Nov. 7 Dortmund 2-0 Newcastle Nov. 7 Milan 2-1 PSG Nov. 28 PSG 1-1 Newcastle Nov. 28 Milan 1-3 Dortmund Dec.
</item 24>

<item 25>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
19 Young Boys 1-3 RB Leipzig Sept. 19 Man City 3-1 Red Star Belgrade Oct. 4 RB Leipzig 1-3 Man City Oct. 4 Red Star Belgrade 2-2 Young Boys Oct. 25 RB Leipzig
3-1 Red Star Belgrade Oct. 25 Young Boys 1-3 Man City Nov. 7 Man City 3-0 Young Boys Nov. 7 Red Star Belgrade 1-2 RB Leipzig Nov. 28 Man City 3-2 RB Leipzig
Nov. 28 Young Boys 2-0 Red Star Belgrade Dec. 13 RB Leipzig 2-1 Young Boys Dec. 13 Red Star Belgrade 2-3 Man City

Group H

Team GP W D L GD P 1. Barcelona (Q) 6 4 0 2 +6 12 2.
</item 25>

<item 26>
European football: Eintracht Frankfurt hit Bayern for five in famous thrashing (The Guardian, 2023-12-09T20:46:25+00:00)
A stunning goal from Real Betis’s Aitor Ruibal cancelled out Jude Bellingham’s strike for Real Madrid early in the second half to earn the home side a 1-1 draw
in La Liga. Real Madrid remain top of the table with 39 points, one ahead of Girona, who visit third-placed Barcelona on Sunday.

Real Betis’s Aitor Ruibal (centre) celebrates after scoring the equaliser against Real Madrid. Photograph: Jose Breton/AP

Bellingham gave the visitors the lead with a fine low finish after Brahim Díaz’s brilliant chipped pass on the edge of the box eight minutes after half-time.
The English midfielder reached 15 goals in 17 games across all competitions this season.

Betis bounced back in the 66th minute when Ruibal equalised with a long-range rocket into the far corner.
</item 26>

<item 27>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
With the United States set to test themselves against one of Europe's most powerful sides, the opponents are entering a transitional phase as well.

Germany are considered one of the preeminent national team programs in the world, but recent years have humbled Die Mannschaft. They have been eliminated in the
group stage of two straight World Cup tournaments, an unthinkable slip in stature given they had never been bounced from any World Cup that early in history
prior to 2018.

With Hansi Flick unable to steer the team to safer waters in Qatar, Julian Nagelsmann has taken over as new Germany head coach.

At just 36 years old, Nagelsmann is a rising star, having spent three successful seasons at Hoffenheim followed by two years at RB Leipzig, which catapulted him
to the managerial post at Bayern Munich in 2021. He steered the Bavarian giants to the Bundesliga title in his first season, but he was sacked by the club in
March of 2023 amidst a furious league title fight.
</item 27>

<item 28>
European football: Eintracht Frankfurt hit Bayern for five in famous thrashing (The Guardian, 2023-12-09T20:46:25+00:00)
Eintracht Frankfurt pulverised Bayern Munich 5-1 with a sensational three-goal first-half performance to snap the champions’ unbeaten run in the Bundesliga this
season. In an explosive first half, Eintracht completely overran Bayern and exposed their error-prone backline, scoring three times in 24 minutes.

Omar Marmoush opened their account in the 12th minute, drilling in the rebound after Fares Chaibi had hit the crossbar following a miskick from Bayern’s
Noussair Mazraoui. After yet more slapdash defending from Bayern, Éric Dina Ebimbe doubled their lead in the 31st minute and Hugo Larsson made it 3-0 five
minutes later, benefiting from a Joshua Kimmich error. Kimmich made amends just before the break with a powerful shot from 20 yards to cut the deficit but it
did not stop the home side.
</item 28>

<item 29>
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
</item 29>

<item 30>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
As Manchester United's season continues to slip away, the pressure around Dutch boss Erik ten Hag continues to get more palpable.

The latest slip saw the Red Devils crash out of Europe as they fell 1-0 to Bayern Munich on the final day of Champions League group-stage play, finishing bottom
of their four-team group to leave them eliminated from Europe in their worst performance in the competition since 2005.

On top of that, they set the record for most goals against in a Champions League group stage by an English club, with 15 conceded in their six games. Their
defensive record has collapsed as the season goes on: from a Premier League standpoint, they sit eighth in the English top flight, but their -5 goal
differential is the worst of any club in the top 10.

With Jim Ratcliffe having now completed the formalities around his investment into the club, taking control of 25% of his boyhood club after a year of
negotiations, the pressure will be piled on even more.
</item 30>

<item 31>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Old Trafford (and the training ground)

To find an apt shorthand for the lurching ship that is United, end the search at the roof of their stadium where on Manchester’s (many) wet days rain leaks
through. Old Trafford remains a storied venue but as it continues to crumble the tale of disrepair has become a central narrative in its recent history. As
Premier League titles have become scarce (zero in a decade) the tale of neglect of a ground that dates to 1910 is emblematic of the Glazers’ quasi‑absentee
proprietorship. Ratcliffe will invest $300m (£237m) into infrastructure but this is small beer when considering the multibillion-pound facelift or new venue
required. There is also the cramped Carrington base: as on Sir Matt Busby Way, the footprint is large at the training ground but the configuration and
facilities are ageing – badly.
</item 31>

<item 32>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Which, as the German champions have not lost in 40 group-stage games in this competition and United have not beaten anyone resembling a top team this season,
always felt unlikely.

This was far from their worst performance of the season, nowhere near as bad as Bournemouth or Crystal Palace or Newcastle, or the other defeat to Newcastle.
They were not as self-destructive as they had been in Denmark and Turkey and at home to Galatasaray. But they still lost.

Harry Maguire’s injury forced him off the pitch as 36-year-old Jonny Evans replaced him (EPA)

And they still went out, with four points, in fourth place. Go back a couple of months and Raphael Varane had said United could win the Champions League. On a
night when Ten Hag seemed to revise his opinion of the quadruple Champions League winner, the defender may need a rethink. Instead, United won a lone game in it
and even that required a penalty save to deny FC Copenhagen a draw.
</item 32>

<item 33>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
The UEFA Champions League has reached the end of its group stage, with 16 precious places in the first knockout round now claimed.

Manchester City are looking to go back-to-back after their triumph in last season's Champions League, potentially becoming the first English club since
Nottingham Forest in 1980 to successfully defend the trophy.

The likes of Bayern Munich, Real Madrid and Barcelona have also enjoyed strong group-stage campaigns and will surely be in the mix towards the sharp end of the
competition. Atletico Madrid and Real Sociedad will complete the Spanish contingent in the next round.

Manchester United and Newcastle United will play no further part in Europe, though, having been eliminated by finishing bottom of their groups. Paris
Saint-Germain will be in the Round of 16, though.

The Sporting News reviews the group stage ahead of its final match night and examines what teams from each group need to qualify for the Round of 16.
</item 33>

<item 34>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
The England captain is the forward United neglected to bid for in the summer. The one they did buy, Hojlund, has at least struck in the Champions League but in
domestic competitions he has been outscored 18-0 by Kane.

Yet a rookie centre-forward is not the culprit in chief. For Ten Hag, this has been a chastening setback; with Sir Jim Ratcliffe set to finalise his investment,
with United listing in the Premier League, the chances are the Dutchman has managed them in the Champions League for the last time.

Go back four years and it required a 96th minute goal by Lucas Moura to stop him from reaching the final. Now the less heralded Lukas Lerager signalled United’s
demise, putting Copenhagen ahead against Galatasaray when United needed a draw to preserve their own chances of qualification; that, and a victory over Bayern.
</item 34>

<item 35>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
"The first is a transition moment, we spoke about quick release and we don't block the middle.

"The second is an unnecessary free-kick but then we have to defend the free-kick better. The third one we are very good and organised but we got run off and we
are in an overload there.

"So, our mistakes can never happen in the Champions League, you get punished for it, so we have to learn from that."

However, Aston Villa legend Gabby Agbonlahor insists Onana needs to take a significant chunk of the blame for United's collapse on Wednesday night.

"I feel sorry for Onana," Agbonlahor told talkSPORT. "There's a clip of him throwing his gloves to the ground after the game. He's angry with himself.

"But those two free-kicks... you can't score those. They're easy saves and he's cost his team.
</item 35>

<item 36>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Manchester United's domestic form has certainly shown signs of improvement in recent weeks, but that's far from the case in Europe.

Erik ten Hag's side have once again dropped points from a winning position as they drew 3-3 with Galatasaray in the Champions League, a damaging result in their
bid to reach the last 16.

4 United were once again the victims of their own demise Credit: Getty

Goals from Alejandro Garnacho and Bruno Fernandes fired them into a two-goal lead within 18 minutes before Chelsea loanee Hakim Ziyech struck back.

Scott McTominay scored what he thought would have been a crucial third shortly after the break but United failed to hold on.

Another free-kick from Ziyech caught Andre Onana off guard once again and a strong strike from Kerem Akturkoglu levelled the scoring.
</item 36>

<item 37>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
No family member has given a press conference or interview during their 18-year tenure. Ratcliffe has arrived on a ticket of, partly, being the local lad made
good (Failsworth, where he was born, is next to Newton Heath, United’s birthplace) so if he proves as silent as the Americans then this will be: a) poor PR; and
b) just plain poor. Football fans can be classed as the most loyal citizens of any society but even the faith of United enthusiasts has been stretched by the
cold relationship created by the Glazers’ stance regarding communication. A tap-in, then, for Ratcliffe is to remedy this by opening and maintaining a regular
line into supporters.

The Glazer family have little relationship with the Manchester United fans – something Ratcliffe would do well to avoid.
</item 37>

<item 38>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
Theirs was another colossal effort, a far better performance than at Everton or Tottenham, but while Eddie Howe had the luxury of one attacking substitute – in
Alexander Isak – three replacements combined for Milan’s decider: first Luka Jovic found Noah Okafor and he picked out Chukwueze, who placed a shot past Martin
Dubravka.

It was a sign of Newcastle’s determination to win, not to settle for the point and the Europa League place, that they were short-staffed at the back because
centre-back Fabian Schar had charged into the Milan box in open play. Likewise, in a frantic finale, they almost conceded a third: with Dubravka out of his
goal, Theo Hernandez almost scored from the halfway line and Fikayo Tomori hit the post.

As they wonder what might have been, they can reflect on the turning point of Pulisic’s strike, around the time Warren Zaire-Emery drew Paris Saint-Germain
level against Borussia Dortmund.
</item 38>

<item 39>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Bayern Munich have their first game back against Mainz on Saturday, as does much of the Bundesliga.

"I don't know if I'd be interested in playing against Mexico in America before [the Euros hosted] in Germany," said Bayern Munich manager Thomas Tuchel prior to
the international break. "I don't know if anyone can explain this in a way that I can understand. I don't know who decided it, but it has been decided.

"There is a difference between a busy schedule and everything away from home and in different time zones. This is at the end of the endurance. No coach in the
world at this level will say otherwise."

"I don't understand the trip to the USA — for a variety of reasons," said former Germany international Lothar Matthaus to German outlet Bild back in the summer.
"I would have tried to create a good atmosphere in Germany before the Euros.
</item 39>

<item 40>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
This is the current reality for United. They obviously have many more issues than the goalkeeper – as illustrated by the goal itself – but a strange season for
Ten Hag’s primary signing only continues. United’s problems meanwhile combined. There were so many errors leading to the goal, to go with all of those that have
characterised this entire European campaign.

That can now be talked about definitively, since it has been decisively ended. It is a low but just part of a decade of disappointment in the competition that
has seen this club rise to its most emotional heights.

This game never threatened to do that. United never threatened at all. They’d been involved in so many ludicrously chaotic matches that the biggest of all ended
up being a non-event.
</item 40>

<item 41>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
“Sometimes players need time, they’re human beings, they need to settle, Sonny had a great track record at Hamburg and Leverkusen but there was this narrative
that started to grow internally and externally that he wasn’t a good finisher.

6 Mitchell is becoming an increasingly recognisable face

“I remember telling a coach ‘you’re wrong, his history shows in two different clubs he’s a very astute finisher off both sides’.”

Mitchell remains good friends with Rangnick from their time together at Leipzig, and last season made headlines when he was spotted at a youth team game while
the German was interim coach at Man United.

So astute is Mitchell in his observations, he even laid out a blueprint for United's success while at Monaco.

“I think context is important to measure what Manchester United's current situation is, but I think they do need to get to the point of really putting down what
Manchester United needs to be in the modern game,” he explained.

"We are a long time from the Sir Alex Ferguson era.
</item 41>

<item 42>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
We showed character.” Bayern travel to Copenhagen in the Champions League next week.

Leipzig struck twice in six minutes with Loïs Openda putting them in front in the 20th when he completed a quick break. They scored again with Castello Lukeba
stabbing in after the goalkeeper Sven Ulreich misjudged a corner.

Bayern earned a 57th-minute penalty against the run of play for a handball and Kane cut the deficit with his eighth league goal. Only Manchester City’s Erling
Haaland, when he played for Borussia Dortmund, scored more often in his first six Bundesliga games with nine goals.

The Bavarians’ comeback was complete in the 70th with Kane clearing the ball in his own half and Musiala finding Sané for the equaliser.

Bellingham celebrates scoring Real Madrid’s third goal during their 3-0 away league win over Girona.
</item 42>

<item 43>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
On the other hand, Antonee Robinson, Tim Team, and Bernd Leno will be joining the team again to help them hold the fort in the back.

Manchester United Premier League Form

L W L W W L

Manchester United Form in All Competitions

L W W W L L

Fulham Premier League Form

W D L W L D

Fulham Form in All Competitions

W L W L D W

Match Details

Match – Man United vs Fulham

Date – 4th November, 2023

Time – 6:00 PM IST

Live Telecast – Premier League’s rights are owned by the Star Sports Network. Hence the matches will be broadcast on various Star Sports channels. The fans will
get to watch English commentary on Star Sports Select and apart from that regional fans will also get the opportunity to watch the match in their favorite
language.
</item 43>

<item 44>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
It shows that while Ten Hag's side do win the ball high up the pitch, opponents do tend to keep the ball as well as play it around United's pressing forwards as
well.

What is PPDA? PPDA is the number of opposition passes allowed outside of the pressing team's own defensive third, divided by the number of defensive actions by
the pressing team outside of their own defensive third. A lower figure indicates a higher level of pressing, while a higher figure indicates a lower level of
pressing.

The other pressing concern United have, which Ten Hag did reference in Friday's press conference, is finishing chances when they win the ball high up.

United have the fourth-highest amounts of shots after winning the ball in their final third, but the glaring statistic is no goals from 104 high turnovers.

As expected, Manchester City have the highest amount of build-up attacks in the Premier League this season, so if United do win the ball high up in Sunday's
derby, they need to start taking those chances.
</item 44>

<item 45>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
Galatasaray knew that if they beat Copenhagen at Parken they would go through but Lukas Lerager scored the only goal in a 1-0 win for the hosts

Manchester United are out of Europe altogether after a limp display against Bayern at Old Trafford, where Kingsley Coman sealed a 1-0 win for Thomas Tuchel's
side.

Results

Date Fixture/Result Sept. 20 Galatasaray 2-2 FC Copenhagen Sept. 20 Bayern Munich 4-3 Manchester United Oct. 3 Manchester United 2-3 Galatasaray Oct. 3 FC
Copenhagen 1-2 Bayern Munich Oct. 24 Galatasaray 1-3 Bayern Munich Oct. 24 Manchester United 1-0 FC Copenhagen Nov. 8 Bayern Munich 2-1 Galatasaray Nov. 8 FC
Copenhagen 4-3 Manchester United Nov. 29 Galatasaray 3-3 Manchester United Nov.
</item 45>

<item 46>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Photograph: Adam Vaughan/EPA

They held the ball nicely, probed in neat, clipped, mannered attacks. A lot seemed to hinge on how desperately Bayern felt the need to twist that scalpel,
whether they could really be bothered to take this thing to the wall. United defended well. Their waves of pressing drew waves of applause. Everyone here seemed
to be trying.

But with 58 minutes gone Copenhagen scored and United’s hopes began to slip beneath the waves. Just over 10 minutes later Bayern made it 1-0 in Manchester.
Kingsley Coman got it, granted a huge expanse of time and space in front of the United goal, enough to set himself, as though taking a penalty, and belt the
ball into the corner. The moment was made by a delightfully sneaky round‑the‑corner pass from Kane. After all that pressure, all that tender hope, it looked a
bit like a training goal.
</item 46>

<item 47>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Erik ten Hag smiles when asked if André Onana can be the “holding midfielder” for Manchester United described in Uefa’s technical report of the goalkeeper’s
display for Internazionale in May’s Champions League final.

“First the job of a keeper is to save, is to [guard] his goal, that’s his main job and don’t forget this,” says United’s manager. “But, yeah, he is a keeper who
we can use as an extra player and if we are developing [more] he can be more progressive and even higher up the pitch – an extra player, so that’s an
advantage.”

Onana’s last game for the Italian club was the 1-0 defeat by Manchester City in Istanbul as United’s crosstown rivals claimed a first Champions League and so
clinched the treble. Yet if he has recent experience of facing Pep Guardiola’s side, the 27-year-old will be a derby debutant against City in Sunday’s 191st
edition of the local squabble that will attract a global audience of millions.
</item 47>

<item 48>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
Garnacho scored with an incredible overhead kick and Mainoo impressed on what was his full Premier League debut, which could earn him a start in Turkey and an
education in European football.

Around them are players who have succeeded on hostile nights. But if they thought Goodison Park was unwelcoming, they will be in for a surprise on Wednesday and
will need to find communal coping mechanisms. Fail to do so and they will just have to live and learn for next time.
</item 48>

<item 49>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“I think that it is important sometimes that you have to go through those moments to realise what it takes. There are elements that happen in a football match
and elements that happen in the buildup to a football match, so you have to experience all those things if you want to be successful.”

The win against Everton on Sunday was defined by two teenagers, Alejandro Garnacho and Kobbie Mainoo.
</item 49>

<item 50>
Man United vs Chelsea head to head record, all-time results between Premier League giants (Sporting News, 2023-12-05T20:30:03+00:00)
17, 2020 Stamford Bridge Chelsea 0-2 Man United League Cup Oct. 30, 2019 Stamford Bridge Chelsea 1-2 Man United Premier League Aug. 11, 2019 Old Trafford Man
United 4-0 Chelsea

Do Man United vs Chelsea have a rivalry?

Although the matches between United and Chelsea have tended to be closely contested, they do not have as fierce a rivalry as the Red Devils do with the likes of
Manchester City and Liverpool, or the west London club holds with neighbours such as Tottenham and Arsenal.

Abramovich's takeover gave Chelsea a new level of wealth that temporarily turned them into the dominant force in the English game, briefly supplanting United,
who had won the title in eight of the previous 12 Premier League seasons.
</item 50>

<item 51>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
Manchester United are yet to recover from their large number of long-term injuries.

Man United vs Fulham Live: Premier League LIVE – MUN vs FUL LIVE – Manchester United are travelling to Greater London for their 11th Premier League match of the
season. Fulham will host the Red Devils in the hopes of finally winning against them for the first time since 2019. It doesn’t seem like they will get a better
chance considering Man United’s horrible run of form. However, the Red Devils came away as winners in both of their last games on road. Hence, the job will not
be easy for Fulham.

Also Read

Manchester United will have to look for a way to get back to winning after 2 major defeats. They were on a 3-match winning run in all competitions when they
faced Man City in their last Premier League match. There was little hope of winning that match considering their form. The fears came true and they lost 3-0 in
the Manchester derby.
</item 51>

<item 52>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
Harry Kane, for so long billed as a potential United player, didn’t score but did set it up.

This game showed one reason why he is at Germany’s biggest club rather than England’s. His side are through with ease, among the favourites for the competition
even if they do look lacking in a few areas.

Harry Kane’s assist was another reminder of what could have been for United (PA)

Not that you could judge from this game. It was almost a kickabout for them.

Bayern were still having more of the ball and the attacks. This was another tension for Ten Hag to overcome on the night. United were playing in quite a
controlled way when they were going to eventually have to go for it, but Bayern were still getting through that with relative ease.

It meant the team almost looked like they were in two minds, to go with how thoughts were always going to be in Copenhagen. The news came through shortly before
the hour that Lukas Lerager had scored for the Danish side, although it didn’t really change the atmosphere or the energy.
</item 52>

<item 53>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
It is 30 years since Manchester United’s most famous trip to Istanbul with Champions League elimination on the line. Galatasaray created an unwelcome hell that
Cerberus would have been proud of and a night of sin ended with Alex Ferguson’s side being dumped out of Europe.

United, realistically, required a win to progress after drawing 3‑3 in the first leg of the second‑round tie at Old Trafford but the atmosphere in Turkey
subdued the visitors and the second leg stuttered to a goalless draw. Eric Cantona was sent off late on for his frustrations with the referee and the home
team’s addiction to timewasting. There was a post-match melee in the tunnel involving the United squad and riot police as the trip ended like it began – with
intimidation. Many of United’s team froze, unable to deal with the ferocious and vociferous atmosphere they faced.

The match on Wednesday is not a knockout game but in truth the scenario is not much different.
</item 53>

<item 54>
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
</item 54>

<item 55>
Man United vs Chelsea head to head record, all-time results between Premier League giants (Sporting News, 2023-12-05T20:30:03+00:00)
Arguably the most memorable game between the pair is the 2008 UEFA Champions League final, when United goalkeeper Edwin van der Sar saved Nicholas Anelka's
penalty to give his side victory following a 1-1 draw after extra time.

Man United vs Chelsea all-time results (all competitions)

Games Man United

wins Chelsea

wins Draws 193 82 55 56

Man United vs Chelsea recent results

Ahead of their visit to Old Trafford in December 2023, Chelsea’s most recent Premier League away win against Manchester United was a 1-0 win in May 2013.

At that point, they had won more Premier League games at Old Trafford (six) than any other side, but they proceeded to draw six and lose four of their next 10
trips to M16.

Manchester United are currently 11 Premier League games unbeaten against Chelsea since November 2017. However, the fact that seven of those matches ended all
square has helped to make the game the most drawn fixture in Premier League history.
</item 55>

<item 56>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
But Martínez opened the scoring in the 62nd minute when he collected Marcus Thuram’s low cross and chipped a finish over Guillermo Ochoa.

skip past newsletter promotion Sign up to Soccer with Jonathan Wilson Free weekly newsletter Jonathan Wilson brings expert analysis on the biggest stories from
European soccer Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

The substitute Lautaro Martínez converts a penalty against Salernitana. Photograph: Massimo Pica/EPA

Three minutes later Mateusz Legowski’s apparent equaliser was disallowed due to an offside.
</item 56>

<item 57>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Finally you get to manage the everyday metrics on which you will be judged; team, results, style, energy, messaging. Win things, improve players, create a
coherent and happy internal culture: but do it during an unceasing takeover circus while star players sulk and brief against you, while the ghost of Hamlet’s
father still skulks in the eaves. This felt like the end of one thing, perhaps even the beginning of the end of something else.
</item 57>

<item 58>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
There was not even the sense of heroic failure that accompanied their quarter-final exit to Bayern in 2010. There was just crushing failure. Because it wasn’t
Bayern who eliminated them. It was Galatasaray, who won at Old Trafford and scored six goals over 180 minutes against United. It was FC Copenhagen, who
qualified instead of them. Rasmus Hojlund’s former club will be in the last 16. His current one won’t. Copenhagen will be the anomalies in the draw, the
outsiders with the spot that had seemed reserved for United.

Manchester United’s defeat means they miss out on a spot in the Europa League (PA)

Instead, they are cast out of continental competition altogether. Europe’s sinking superpower have become just the fourth English team to prop up a Champions
League group when it concluded. There was something predictable about the manner of their demise; their defence was unlocked by Harry Kane, with a lovely flick.
</item 58>

<item 59>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
United have done little learning in recent times, repeating the same failures in Europe’s elite competition. It would not be a disgrace to lose against a good
Galatasaray team, especially given Erik ten Hag is missing key players, but if the worst does happen, everyone will need to come out of the experience stronger.
Regardless of the result, it should be a night of education when it comes to the demands of the Champions League.

Six of United’s squad that failed in 1993 went on to win the competition six years later. “It did galvanise us,” says Mike Phelan, who started in central
midfield 30 years ago. “Winning the Cup Winners’ Cup took us to another level and we felt pretty comfortable with each other and felt we could develop into a
serious team, and I think going into the Champions League was one of those moments where we expected to do well – the experiences along the way were great and
took us to the next level further down the line.
</item 59>

<item 60>
European football: Eintracht Frankfurt hit Bayern for five in famous thrashing (The Guardian, 2023-12-09T20:46:25+00:00)
It started badly for the hosts with Mats Hummels sent off for a last-man foul at the edge of the box in the 15th minute, then a Ramy Bensebaini own goal put the
visitors in front just after the half-hour.

Dortmund equalised with the last kick of the first half when Niklas Suele volleyed in from the far post. But it was only a matter of time until Leipzig scored
again and they did it with Christoph Baumgartner in the 54th.

With Dortmund taking risks in search of an equaliser, Leipzig launched a counterattack and Yussuf Poulsen rifled in before Niclas Füllkrug scored deep in
stoppage time for some last-gasp drama. Leipzig are fourth on 29 points and now have a four-point lead over Dortmund in fifth place.
</item 60>

<item 61>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
In either setup, the forward line will shift horizontally to allow a midfield player to drive into space, and given options once halted to either play into the
crowded forward line or release a bombing full-back to the off side, both of which can stretch or confuse an opponent's defensive unit.

Having spent time in the Red Bull system, Nagelsmann also deploys a complicated press that, like Gregg Berhalter, starts with the lone central striker. The goal
is to force the opponent to play the ball into overload traps out wide, or if the opponent plays centrally, a defensive midfielder would step up at the base of
a diamond (with two midfielders on the sides and the striker at the top) which would overwhelm an opposition midfield trio.

How will Germany play under Julian Nagelsmann?

The question, now, is how all this will translate to the German national team.

Looking at Julian Nagelsmann's first squad of 26 players, the biggest question is which formation he will play.
</item 61>

<item 62>
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
</item 62>

<item 63>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
“Unfortunately, injuries tend to come when players are in good shape.”

In France, AS Monaco beat Marseille 3-2 to move top of Ligue 1, thanks to a brace from the midfielder Maghnes Akliouche and a powerful strike from Folarin
Balogun.

Marseille, who appointed Gennaro Gattuso as coach this week, took the lead after 30 secondsthrough Iliman Ndiaye’s goal, only for the hosts to respond minutes
later when Akliouche fired a low shot past Pau López.

The visitors regained the lead in the 18th minute when Ndiaye centred for Samuel Gigot to send a left-footed shot past Philipp Köhn, but Monaco equalised five
minutes later as Akliouche found Balogun who scored with a fierce finish.

Monaco sealed the win in the second half with another great strike from Akliouche.
</item 63>

<item 64>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
The manager’s future

Erik ten Hag’s team are treading water at precisely the wrong moment: as Sir Jim Ratcliffe takes charge of Manchester United’s sporting department before making
sweeping changes in the posts that will directly affect the manager. Richard Arnold has left despite officially staying until the end of the year and into his
role may come Jean-Claude Blanc, the CEO of Ratcliffe’s Ineos Sport, as the man Ten Hag reports to. Ten Hag’s line manager may change too, as Ratcliffe
considers whether a new football director is required, a position held by John Murtough. Ten Hag may have a first clue of his status in Ratcliffe’s new empire
if he is sounded out for his view of a prime factor in his ability to coach and manage the side to success: United’s transfer policy. This leads us to Murtough
…

Will Murtough survive?

The football director has no plans to depart but whether he remains in situ may be a matter for Ratcliffe.
</item 64>

<item 65>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
28 PSG 1-1 Newcastle Nov. 28 Milan 1-3 Dortmund Dec. 13 Dortmund 1-1 PSG Dec. 13 Newcastle 1-2 Milan

Group G

Team GP W D L GD P 1. Man City (Q) 6 6 0 0 +11 18 2. RB Leipzig (Q) 6 4 0 2 +3 12 3. Young Boys 6 1 1 4 -6 4 4. Red Star Belgrade 6 0 1 5 -8 1

Manchester City and RB Leipzig confirmed their knockout places after four games played and City's 3-2 comeback win over the Bundesliga team on November 28
clinched top spot in Group G for Pep Guardiola. Young Boys drop into the Europa League.

Results

Date Fixture/Result Sept. 19 Young Boys 1-3 RB Leipzig Sept.
</item 65>

<item 66>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

It was the sequel Newcastle United did not want, the second cruel comeback in as many Champions League games. For some 45 minutes, including half-time, they
could savour the sight of a table that showed them in second, permitting them to imagine a date with Real Madrid or Bayern Munich in the knockout stages. And
then, little more than half an hour, they were ejected from Europe altogether.

Christian Pulisic’s equaliser threatened to condemn them to the Europa League.
</item 66>

<item 67>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
The 42-year-old has also led the Red Devils before on a caretaker basis, installed back in 2021 after Ole Gunnar Solskjaer's dismissal and leading the club to
some big wins over Arsenal in league play and Villarreal in the Champions League before Ralf Rangnick took over.

Hansi Flick

The former German national-team manager only lasted two years on the job with Die Mannschaft, before their failures at the 2022 World Cup caught up with him.
Yet Flick has big club experience, having led Bayern Munich first as an assistant under Niko Kovac and then for two years as manager, winning the treble in
2019/20 en route to being named UEFA Coach of the Year.

Yet Flick's lack of experience of any kind outside Germany — he has just one year abroad, as an assistant at Red Bull Salzburg — coupled with his struggles at
the national-team level could give some pause at bringing him to England for such a project position.
</item 67>

<item 68>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Onana has been criticised for his role in Galatasaray's second goal as he allowed Ziyech's shot to spill under him and into the net, although Erik ten Hag says
collectively the defence isn't up to scratch.

"We're winning and then we're losing, we should have taken three points," the United boss told TNT Sports. "But there's also in more games, at home against
Galatasaray, away in Copenhagen, we play so good so I have to make a big compliment to the team.

"But also in the same moment I have to criticise the team and us as coaching staff, me as a manager. We aren't defending good enough, we are leading 3-1 and we
can't afford such mistakes because that makes the difference."

4 Onana had another disappointing performance in goal for United Credit: Getty

"Every goal has it's own story," he added.
</item 68>

<item 69>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

They sang the name of Erik ten Hag’s predecessor. Not in a gesture of dissent, but as a reminder of a time when Manchester United were Europe’s best and an
evening without which a certain Norwegian would never have become their manager two decades later. After all, who put the ball in the Germans’ net? Ole Gunnar
Solskjaer. But that was a different millennium, a different United. Nostalgia underpinned Solskjaer’s management.
</item 69>

<item 70>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
Martínez then got his second goal after 77 minutes when he confidently finished a pass from Nicolò Barella with a first-time strike.

Martínez completed his hat-trick from the penalty spot with five minutes remaining after Matteo Lovato pulled Thuram’s shirt. He concluded the evening by
converting Carlos Augusto’s cross from 10 yards out.

Inter lead Serie A with 18 points, the same as their neighbours, Milan, who had briefly topped the table earlier in the day after beating Lazio 2-0 thanks to
second-half goals from Christian Pulisic and Noah Okafor. Rafael Leão played a vital role in both goals, with the 24-year-old Portugal international providing
two assists in a menacing display.

Quick Guide How do I sign up for sport breaking news alerts? Show Download the Guardian app from the iOS App Store on iPhone or the Google Play store on Android
by searching for 'The Guardian'.

If you already have the Guardian app, make sure you’re on the most recent version.
</item 70>

<item 71>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
As Nick Pope had denied Milan victory in San Siro with a string of saves, Stefano Pioli’s team could testify to the difference brilliant goalkeeping can make.
Now they benefited from it. Newcastle had 17 shots, eight on target, but Maignan was only beaten once. Milan had two shots on target but Dubravka conceded
twice.

The loss of the injured Pope may have proved pivotal. But perhaps seven-time winners, and a side who reached the last four last year, displayed greater knowhow.
Maybe they simply had greater strength in depth on a night when Kieran Trippier and Anthony Gordon were taken off because they seemed exhausted.

Newcastle can nevertheless depart the competition with heads held high. One win and five points seem meagre tallies for their contribution to the competition.
They have proof it can be unforgiving, reasons to return. Their first Champions League campaign for two decades will surely not be their last for another 20
years.
</item 71>

<item 72>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
They still have the FA Cup, and the extended pursuit of fourth or even fifth place, although the state of the coefficient means the thing that might keep
Manchester United out of Europe next season could be Manchester United being so bad in Europe this season. Maybe City can revive their fortunes by winning the
Champions League again.

Ten Hag will take the blame of course for the early exit and plenty of mistakes have been made, not least in the players the manager has been allowed to sign.
In reality assessing the job any United manager does with this team involves applying a series of filters. To be United manager is to undertake three concurrent
roles.

First you must manage the past, which is constantly in the room and which skews every act, achievement and expectation. Second you must manage the dysfunction
of the present, which expresses itself in every detail from a leaking stadium roof to the deep ills of a playing squad peopled with squatters and long‑term
missteps.
</item 72>

<item 73>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Manchester United’s century old stadium needs an upgrade. Photograph: Greig Cowie/Shutterstock

Communications

The Glazers treat a microphone or TV camera as United fans do Liverpool or Manchester City: as a fierce enemy. No family member has given a press conference or
interview during their 18-year tenure.
</item 73>

<item 74>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
"We are a long time from the Sir Alex Ferguson era. Erik ten Hag is a top coach from Ajax but they need a top blueprint not only for now but for the next five
years, and work towards that, and sometimes on that journey there are hard moments when you don't get the right results, even though you're doing the right
things.

"You have to know that, over that period, the consistency of the decision-making will yield a good end result. I think we've seen that both Manchester City and
Liverpool, that when you adopt that long-term strategy you can get real consistency and sustainable success.

"I think that's the biggest thing for Manchester United; putting down their identity of what they want to be today, but most crucially what they want to be in
five years' time, and where they want to be, and what they want to look like; from everything from young players, to the style of play, to the whole culture
peace in Manchester United."
</item 74>

<item 75>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
The performance this season was so bad that they find themselves out of Europe altogether as the bottom team in the group, although some at the club might find
solace from that. There wasn’t much else. The second place they aimed for has instead gone to Copenhagen, a club with a fraction of United’s budget. That is
where the damage was done.

While that meant there was only ever a slim hope of going through – and the Danish side’s victory over Galatasaray ensured any win would be irrelevant – the
club’s rich legacy in this competition was built on such defiance. There was absolutely none of it here.

There was nothing much at all. Where was the energy? Where was the intensity? The Stretford End eventually implored United to “attack attack attack” but the
complete absence of any such impulses eventually left some of the stadium booing the side off at the end.
</item 75>

<item 76>
Champions League permutations, scenarios: What teams need to qualify from groups for 2023/24 Round of 16 stage (Sporting News, 2023-12-12T21:28:00+00:00)
MORE: Champions League Round of 16 schedule, draw & qualified teams

Champions League permutations, scenarios: What teams needed to qualify

Group A

Team GP W D L GD P Bayern Munich (Q) 6 5 0 0 +6 16 Copenhagen (Q) 6 2 2 2 0 8 Galatasaray 6 1 2 3 -3 5 Manchester United 6 1 1 4 -3 4

Bayern Munich coasted into the knockout rounds but Copenhagen, Galatasaray and Manchester United all entered the final matchweek with a chance of joining the
Bundesliga giants.
</item 76>

<item 77>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
I think that shows you that there's a lack of balance in the team at the moment.

"Yes they can get goals and yes they are attacking at will and creating chances, many of them coming through Bruno. But I think defensively, they just made
mistakes, I think the goalkeeper today has to hold his hands up and say, 'Yep, I was at fault for a couple of goals.

"He should do better. But, I think the manager will be scratching his head and thinking, 'The game plan first-half worked perfectly,' and then all of a sudden
they score a goal, it changes the dynamic of the game.

"They were 2-0 up, 2-1, and you're thinking, 'Wow, I set the team up perfectly well, they execute it but then a mistake cost us,' and at 3-1, the same again.
</item 77>

<item 78>
European football: Eintracht Frankfurt hit Bayern for five in famous thrashing (The Guardian, 2023-12-09T20:46:25+00:00)
Turn on sport notifications. Was this helpful? Thank you for your feedback.

“There is no need for an explanation,” said Bayern’s midfielder Thomas Müller. “We have to congratulate Frankfurt who deserved the win. They took the fight out
of us in that first half hour. When you lose 5-1 then a reaction has to follow. We won’t lose our heads now. We don’t have enough time but we will bounce back.”

Bayern, who have a game in hand after last week’s game against Union Berlin was postponed due to snow, remain in second place on 32 points, with the leaders,
Bayer Leverkusen, on 35, travelling to third-placed VfB Stuttgart, who have 30 points, on Sunday.

Later, RB Leipzig beat 10-man Borussia Dortmund 3-2 to tighten their grip on fourth place and put a further dent in their opponents’ title hopes.
</item 78>

<item 79>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
Indeed, Ratcliffe wants United back at "the very top of world football".

The Sporting News has laid out all the options should Man United move on from Ten Hag and look for a new beginning, in the hopes of salvaging the 2023/24
season.

MORE: How Man United produced their worst Champions League group stage in 18 years

Man United next manager: Assessing candidates to replace Erik ten Hag

If Manchester United decide to pull the plug on Erik ten Hag's managerial tenure at Old Trafford, there aren't any perfect options for his replacement.

That allows us to speculate some as to who could slot in as the next man hoping to take United back to former glories, which still evade the club since the
departure of Sir Alex Ferguson over a decade ago.

Zinedine Zidane

The former Real Madrid manager has been out of work since departing the Spanish club in 2021, clearly waiting for the right position to open up.
</item 79>

<item 80>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Even if you’re not playing well, if the team is not playing well, or you’re not playing to the standards you expect, you still have to win – so find a way to
win, be pragmatic.”

United have been so of late, claiming victories over Sheffield United, Brentford and FC Copenhagen in their last three matches. But Ten Hag’s “depends on the
opposition” and “how good is the team around him” comments sound like hints. In contrast to Inter’s Simone Inzaghi, the United manager may harbour doubts about
deploying Onana high up the pitch against City.

The Dutchman is still building his team and may be wary of the uneven form of United’s new keeper. Bought for £47.2m, Onana arrived on a ticket of being a
high-grade shot-stopper, technically impressive too with the ball at his feet. But until impressing against Brentford and Copenhagen, Onana had bombed, costing
United goals and points.
</item 80>

<item 81>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
Manchester United managers since Sir Alex Ferguson

If Manchester United were to appoint a new manager, that person would be the ninth coach to take charge of a United team following Sir Alex Ferguson's
retirement in May 2013, including caretaker/interim managers Ryan Giggs, Ralf Rangnick and Carrick.

Manager Honours Games David Moyes Community Shield 51 Ryan Giggs (caretaker) — 4 Louis van Gaal FA Cup 103 Jose Mourinho Community Shield, Europa League, League
Cup 144 Ole Gunnar Solskjaer — 168 Michael Carrick (caretaker) — 3 Ralf Rangnick (interim) — 29 Erik ten Hag League Cup *

*Still manager of the club
</item 81>

<item 82>
Manchester United face Galatasaray with high hopes but bad memories (The Guardian, 2023-11-27T22:30:42+00:00)
The match on Wednesday is not a knockout game but in truth the scenario is not much different. Defeat against Galatasaray would again end United’s Champions
League’s campaign prematurely while a win, assuming Copenhagen lose against Bayern Munich, would put qualifying for the latter stages back in their hands.
United will face the same atmosphere as they did 30 years ago because the home fans know victory would turn their subsequent trip to Denmark into a potential
shootout for who goes through with Bayern, with the achievement of knocking out a club with the storied history of United only adding to the pleasure.

In 1993 everyone was on a learning curve at United. It was their first return to Europe’s top table after an absence of 24 years. Much of the squad had won the
European Cup Winners’ Cup in 1990‑91 but this was a clear step up.
</item 82>

<item 83>
European roundup: Bellingham helps Real Madrid go top while Bayern draw (The Guardian, 2023-09-30T21:17:47+00:00)
Photograph: David S.Bustamante/Soccrates Images/Getty Images

In Spain Real Madrid outclassed Girona 3-0 to return to the top of LaLiga, thanks to goals from Joselu, Aurélien Tchouaméni and Jude Bellingham.

Real moved on to 21 points from eight games, one ahead of Barcelona, who beat Sevilla 1-0 at home on Friday. Girona dropped to third.

“I feel very good. I have freedom of movement and I’m enjoying football. Whenever I put on this shirt I always try to do my best,” said Bellingham.

Real will also be buoyed up by the performance of Vinícius Jr, who looked lively in his first start since returning from a thigh muscle injury, before being
taken off in the 68th minute. Real’s coach, Carlo Ancelotti, said: “It was a solid, convincing game.
</item 83>

<item 84>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
Erik ten Hag has disagreed with Jamie Carragher's comments that Manchester United are going backwards this season - by saying the statistics show that his team
has actually improved.

United host local rivals Manchester City in a Super Sunday derby this weekend, live on Sky Sports with kick-off at 3.30pm, in the latest acid test to see how
far Ten Hag's side have improved from last season.

In his column in The Daily Telegraph, Carragher said United's summer signings have led to the club becoming "stale" - and that Ten Hag should now be questioned
about whether he can get the best out of his squad.

But in his pre-match press conference for the City game, Ten Hag hit back at those claims.

"In possession we have to do better, but there are reasons and I don't go into them, but everyone is seeing why. But there are also facts so I disagree [with
Carragher], we are much better," said Ten Hag.

"In high ball regains, we are a top of the Premier League.
</item 84>

<item 85>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
They will always have the past at Old Trafford, but Evans may be consigned to it; perhaps, in time, Ten Hag, too.
</item 85>

<item 86>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
In the second half United’s back five was André Onana, Aaron Wan-Bissaka, Jonny Evans, Raphaël Varane and Diogo Dalot, an entirely random collection of players
signed, at various times, by Ten Hag, José Mourinho, Ole Gunnar Solskjær and Alex Ferguson. Presumably Daley Thompson, Michael Portillo and Howard from Take
That were busy.

By the end United had five academy players on the pitch, ages ranged between 17 and 35, in outline evidence of internal good health, in reality another symptom
of a machine that wants to function properly but has for so long been made to judder along at half-speed. Ever feel like you’re starting to slip into the past
tense for good? When VAR happens at Old Trafford there isn’t even a VAR screen to say “VAR Check”. The 1990s were a hell of a decade. They still are around
here.
</item 86>

<item 87>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
He was heavily rumoured to be in line for the France position once the 2022 World Cup finished, but Didier Deschamps surprised many by signing a new contract.

Zidane would be a top choice for United, as he would be for many top clubs with a managerial vacancy, but there's questions around whether he would be open to
signing on. The 51-year-old told Marca back in June of 2022 that he isn't exactly thrilled about a Premier League job due to the language barrier.

When asked specifically about a possible job opening in Manchester, Zidane said: "I understand English, but I don't master it completely. I know that there are
coaches who go to clubs without speaking the language, but I work in a different way. Many elements come into play to win. It is a global context." Does he
still feel the same way?
</item 87>

<item 88>
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
</item 88>

<item 89>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
We should do this on a consistent basis.

"We don't take the benefit in the attacking transition moments, we should be because the ability in the team can take benefit from it. We pay attention to it,
we need to grow. Then games will become easier."

So, what do the stats say?

Sky Sports' Sam Blitz:

Ten Hag is indeed right - Manchester United do rank very highly in their pressing stats in the final two-thirds of the pitch.

United are, as he says, "top of the Premier League" in terms of possessions won in the final third and high turnovers. He is slightly wrong in terms of winning
the ball in the middle third of the pitch - with United second behind Everton in that statistical table.

However, there is still work to be done in terms of improving United's pressing. Their passing permitted per defensive actions (PPDA) tally ranks 10th - so,
average - in the league.
</item 89>

<item 90>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
However, as the 2022/23 season progressed and results weren't as easy to come by, Nagelsmann switched to a 3-2-4-1 formation. This allowed him to deploy the
likes of Alphonso Davies and Kingsley Coman very high up the pitch to provide width, while letting Jamal Musiala and Thomas Muller cook centrally behind a
striker.

That 3-2-4-1 formation is flexible horizontally, as it allows one full-back to push high up the pitch while the other slots in along the back line, and thus
sees the two central No. 10 players alternate pushing wide or pinching in to compensate.

This tactical setup is becoming popular with top-tier coaches across Europe, and has been used at various stages by the likes of Pep Guardiola, Stefano Pioli,
Mikel Arteta, Edin Terzic, and many others.
</item 90>

<item 91>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
We remember Onana also diving out of the way of the ball in Istanbul. Casemiro’s red card, lost leads in Manchester, Marcus Rashford unfairly convicted of a
venomous stamp in Copenhagen, when frankly, he’s just not that kind of semi-interested footballer.

And yet with both group games blank at half-time United were still just one goal away here from going through to the last 16, which really would have made for
an excellent punchline. It wasn’t to be. For all their own flaws Bayern are basically a good version of whatever it is United are attempting to piece together.
Kimmich‑Goretzka‑Musiala is an A grade version of Amrabat‑McTominay‑Fernandes. Manuel Neuer is the thing Onana is trying so earnestly to be. Bayern actually did
go ahead and sign Harry Kane.

Harry Kane (centre) and his Bayern Munich teammates applaud fans after their Champions League victory at Old Trafford.
</item 91>

<item 92>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
There are no natural right-backs on the roster, meaning he will either have to pidgeonhole a center-back (Antonio Rudiger or Niklas Sule) out wide, sacrifice
Josh Kimmich's presence in midfield to move him back, or move Pascal Gross back. None of those are ideal options.

Whether Nagelsmann deploys a 4-2-3-1 or a 3-2-4-1, the tactical approach will likely remain the same.

Getting the best out of young superstar Jamal Musiala will be one of the most important aspects of Nagelsmann's tenure, as Musiala has blossomed into a
world-class talent at Bayern Munich, but the 20-year-old has just one goal in 23 international caps to this point.

Nagelsmann will also look to introduce a few overlooked players, starting with Robert Andrich who could factor into the midfield setup as a traditional No. 6,
especially as Ilkay Gundogan enters the twilight of his career.
</item 92>

<item 93>
Man United vs Fulham: Red Devils determined to get back to winning ways in PL (Insidesport, 2023-11-03T21:58:48+00:00)
Fulham are in a worse situation than the Red Devils. After a mixed set of results with 3 wins, 3 draws, and 4 defeats, they are sitting in 14th position in the
league table. However, they are still unbeaten in their last two games in all competitions. They won their last match by a 3-1 margin against Ipswich Town in
the EFL Cup round of 16. Before that, they drew 1-1 against Brighton in their last Premier League match. However, they will have a tough time holding onto the
streak in this match.

Manchester United Team News

Man United have been fighting a battle to line up their best XI as players fall victim to one injury after another. Casemiro, who got injured during the
international break, got back on the field against Newcastle United in the EFL Cup match. However, he was taken off at half-time with a new hamstring issue. He
will probably remain sidelined for weeks.
</item 93>

<item 94>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
The 1990s were a hell of a decade. They still are around here.

'We've lost decisive players': Ten Hag says injuries were key to Champions League exit – video

So much for the good news then. The bad news is that Ten Hag’s United have now compiled arguably the worst performance by any English team in a Champions League
group stage. OK, it’s no Blackburn Rovers. But there are extra marks here for United’s pedigree, for the fact the group really wasn’t much of a group, and for
United managing to finish bottom, shipping 15 goals in the process, more than any other English team ever.

Plus of course there were just so many moments of farce along the way, a six-part saga of chaos, collapse and laughter in the dark. We remember Onana diving out
of the way of the ball in Munich. We remember Onana also diving out of the way of the ball in Istanbul.
</item 94>

<item 95>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Varane was summoned from his inexplicable exile, after a 10-game absence when both Luke Shaw and Jonny Evans were preferred at centre-back to a World Cup
winner. He performed with an assurance to make his omission all the odder, but it has scarcely been the only U-turn in selection; with Harry Maguire and Scott
McTominay, Ten Hag is compiling a group who were first out of his plans and then back in. For a manager who likes to project an image of strong, decisiveness
leadership, he has had to reverse course rather too often. But then his team are going backwards.

When Coman scored, United became the first Premier League side to concede 15 goals in a Champions League group stage. When Maguire hobbled off, holding his
groin, and Evans came on, United had a man who played in United’s 2007-08 Champions League-winning campaign, dragooned into service 16 years later.
</item 95>

<item 96>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
14 MID Pascal Gross Brighton (ENG) 2 0 MID Ilkay Gundogan Barcelona (SPA) 69 17 MID Jonas Hoffman Bayer Leverkusen 22 4 MID Joshua Kimmich Bayern Munich 80 6
MID Jamal Musiala Bayern Munich 23 1 MID Leroy Sane Bayern Munich 55 13 MID Florian Wirtz Bayer Leverkusen 10 0 FWD Kevin Behrens Union Berlin 0 0 FWD Niklas
Fullkrug Borussia Dortmund 9 7 FWD Kai Havertz Arsenal (ENG) 39 13 FWD Thomas Muller Bayern Munich 123 45

Julian Nagelsmann tactics, style of play

Through his time at Bayern Munich, Julian Nagelsmann preferred a 4-2-3-1 formation with Joshua Kimmich and Leon Goretzka holding down the lynchpin central
midfield spots.
</item 96>

<item 97>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
Ten Hag on misfiring strikers: We need to be better

Please use Chrome browser for a more accessible video player Manchester United boss Erik ten Hag speaks on the improvement he's seen in goalkeeper Andre Onana
and insists he believes Marcus Rashford can get back to scoring goals

United go into Sunday's derby with a slight striker scoring problem. City forward Erling Haaland has scored more Premier League goals this season (9) than
United's five forwards have managed in all competitions (8).

United's top scorer this season is actually defensive midfielder Casemiro with four strikes so far.

Asked if their forwards are to blame for this frontline problem, Ten Hag replied: "Fault? It's co-operation. It has to click.

"We have shown in some games, we have so many in overload positions to the opponents goal and we don't net or don't even hit the target. We should be doing this
better."

It has been a sobering start to the season for Marcus Rashford, who has one goal so far this term for United.
</item 97>

<item 98>
Manchester United out of Europe after defeat to Bayern Munich (The Independent - Sports, 2023-12-12T22:38:05+00:00)
Sign up to Miguel Delaney’s Reading the Game newsletter sent straight to your inbox for free Sign up to Miguel’s Delaney’s free weekly newsletter Please enter a
valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and updates from The Independent. Read our
privacy notice Thanks for signing up to the

Football email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Another infamous European night at Old Trafford. It should be a landmark low, but almost the most deflating element of this 1-0 defeat to Bayern Munich was how
grimly routine it all felt.

Manchester United, one of the grandest and richest names in this competition, have again gone out of the Champions League at the group stage. It is the third
time it has happened since Sir Alex Ferguson, meaning it’s their most common round of exit in that period.
</item 98>

<item 99>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
"In high ball regains, we are a top of the Premier League. In the middle [of the pitch] ball regains, we are top. The pressing is very good. So that's not the
truth, what he [Carragher] is telling."

Please use Chrome browser for a more accessible video player FREE TO WATCH: Highlights from Sheffield United’s Premier League clash with Manchester United

In a fortnight where United have picked up three straight wins against Brentford, Sheffield United and FC Copenhagen, albeit via far from convincing
performances, Christian Eriksen said United still need to find their "balance and routines" this season.

Asked about those comments, Ten Hag said: "We are in the right direction. We have a way to go. I see positives but I see negatives.

"What Christian is referring to is we are not consistently in possession on the level where we can be. Like Arsenal away, the first 30 minutes in Bayern Munich
away. We should do this on a consistent basis.
</item 99>

<item 100>
Man United vs Chelsea head to head record, all-time results between Premier League giants (Sporting News, 2023-12-05T20:30:03+00:00)
Manchester United would love a repeat of last season's 4-1 home win over Chelsea when the Blues visit Old Trafford this Wednesday.

The pressure on United has grown again following their lacklustre 1-0 defeat at Newcastle on Saturday and another setback against a Chelsea side – who beat
Brighton & Hove Albion 3-2 the following day – would worsen the misery around Old Trafford.

Two heavyweights of world football, United and Chelsea are two of only seven teams to have won the modern Premier League and are only behind Liverpool when it
comes to English success with European trophies.

Chelsea's recent record against United is poor, but they have won the Premier League twice since United last did so in the 2012/13 season and could exploit the
poor form of Ten Hag's side here.

The Sporting News looks at a rivalry that stretches back decades and has been sharply in focus over recent years.
</item 100>
=== END ITEM id=b7550c4902e9b9ef ===

=== ITEM id=cf0960b64443c554 ===
Gold sentences and facts:
Sentence 0: As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse gases.
  Fact 0.0: The subject referred to as 'they' is never going to replace the need for reducing emissions of CO2 and other greenhouse gases.
Sentence 1: He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about 90%, according to his math.
  Fact 1.0: He stopped flying.
  Fact 1.1: He became a vegetarian.
  Fact 1.2: He ditched gasoline-powered cars.
  Fact 1.3: He drives a Tesla.
  Fact 1.4: He cut his personal emissions by about 90%, according to his math.

Retrieved items:

=== END ITEM id=cf0960b64443c554 ===

=== ITEM id=7d1b13bb1f29cc2b ===
Gold sentences and facts:
Sentence 0: Unfortunately, she did say that she had conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine.
  Fact 0.0: The witness said she had conversations with investors as part of their due diligence.
  Fact 0.1: Alameda was taking on losses from FTX to keep FTX's balance sheet pristine.
Sentence 1: It was not, however, possible for those accounts to avoid liquidation, as Wang testified Alameda could do — or to have an overall negative balance.
  Fact 1.0: It was not possible for those accounts to avoid liquidation.
  Fact 1.1: Wang testified that Alameda could avoid liquidation for those accounts.
  Fact 1.2: It was not possible for those accounts to have an overall negative balance.
Sentence 2: It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy and sell cryptocurrencies.
  Fact 2.0: The argument described Alameda as a 'very innovative, successful company'.
  Fact 2.1: The argument's speaker claimed the business practices between Alameda and FTX were reasonable.
  Fact 2.2: The argument's speaker claimed Alameda acted legally as an FTX customer.
  Fact 2.3: The argument's speaker claimed Alameda acted legally as an FTX payment processor.
  Fact 2.4: The argument's speaker claimed Alameda acted legally as an FTX market maker.
  Fact 2.5: A market maker is a financial entity that acts as a trading partner for customers looking to buy and sell cryptocurrencies.

Retrieved items:

=== END ITEM id=7d1b13bb1f29cc2b ===

=== ITEM id=f5c6b0c8d915bd4e ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google “siphons off” news publishers’ content.
  Fact 0.2: The case argues that Google siphons off news publishers’ readers.
  Fact 0.3: The case argues that Google siphons off news publishers’ ad revenue.
  Fact 0.4: The case argues that Google does this through anticompetitive means.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will also suggest that Google hid items from discovery.
  Fact 1.1: Epic will suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats.
  Fact 1.2: Google will counter this as well.
  Fact 1.3: Google will say it has provided the court with thousands of chat logs.
  Fact 1.4: Google will say it has provided the court with millions of documents.

Retrieved items:

=== END ITEM id=f5c6b0c8d915bd4e ===

=== ITEM id=ddd6d425f9823f57 ===
Gold sentences and facts:
Sentence 0: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 0.0: Sportsbooks are also gathering this information.
  Fact 0.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.
Sentence 1: That allows sportsbooks to earn the profit built into the odds and lines without worrying about which side wins.
  Fact 1.0: The situation described in the preceding sentence allows sportsbooks to earn the profit built into the odds and lines.
  Fact 1.1: Sportsbooks can earn the profit built into the odds and lines without worrying about which side wins.

Retrieved items:
<item 1>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The #Springboks are gearing up for another tough challenge this week with a semi-final clash against England 🇿🇦🏴󠁧󠁢󠁥󠁮󠁧󠁿#StrongerTogether #RWC2023
pic.twitter.com/HLSnnjWggB — Springboks (@Springboks) October 18, 2023

England are the only side still competing who are yet to lose a game, and that will stand them in good stead coming into this semi-final. Dominant wins over
Argentina, Japan and Chile sealed passage to the last eight, and after being given an almighty scare by Samoa in their final pool match, England overcame a
Fijian fightback in Marseille to triumph 30-24. Despite being outscored by three tries to two, captain Owen Farrell kicked 20 points, sending his side into the
last four at the expense of 2023's breakout stars.
</item 1>

<item 2>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The Springboks have done it tough, too, having already faced two of the tournament favourites on their way to the semi-finals. Comfortable defeats of Scotland,
Romania and Tonga couldn't mask the disappointment of the 13-8 loss to Ireland, which consigned South Africa to second in Pool B. However, with Ireland sent
home by the All Blacks in the quarters, the Bokke showed their World Cup mettle to edge the hosts 29-28 at the Stade de France, sneaking past Les Bleus in one
of the best games of rugby you're ever likely to see.
</item 2>

<item 3>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The latter is winning only his seventh England cap but gets the starting nod alongside Maro Itoje, with Genge, Chessum and the rest of the replacements hoping
to make an impact sizeable enough to overcome the Springboks' infamous 'bomb squad'.

Their time is now 🙌



Here is the @EnglandRugby team to take on reigning champions South Africa in the semi-final in Paris#RWC2023 | #ENGvRSA pic.twitter.com/nkf6rXBPUU — Rugby World
Cup (@rugbyworldcup) October 19, 2023

England starting XV: Freddie Steward, Jonny May, Joe Marchant, Manu Tuilagi, Elliot Daly, Owen Farrell, Alex Mitchell; Joe Marler, Jamie George, Dan Cole, Maro
Itoje, George Martin, Courtney Lawes, Tom Curry, Ben Earl.
</item 3>

<item 4>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The last time Cheslin Kolbe played England at a Rugby World Cup... 🇿🇦 🏆#RWC2023 | #ENGvRSA pic.twitter.com/Riw2uaSaKa — Rugby World Cup (@rugbyworldcup) October
19, 2023

Both sides' journeys have seen twists, turns, and bumps in the road, but it's led them both to within 80 minutes of a Rugby World Cup final. England will no
doubt be looking to echo the spirit of 2019, when they beat the All Blacks 19-7 to reach the final, but in South Africa, they face a side who have taken their
game to new heights just to reach the semis.

MORE: 2023 Rugby World Cup final: Details, kick-off time and every team's route

What time does England vs South Africa kick off?
</item 4>

<item 5>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.
</item 5>

<item 6>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
Click below for more ⤵️ https://t.co/LeUuw4kPfP — RugbyPass (@RugbyPass) October 19, 2023

Marcus Smith hardly put a foot wrong at fullback against Fiji, but the world champions are an entirely different proposition for England to deal with at the
back. The high-ball security and hulking frame of Freddie Steward returns to the No. 15 jersey in one of three changes, while at inside centre, Manu Tuilagi
starts hoping to repeat his heroics of four years ago, when his early try in a dominant semi-final performance took England past the All Blacks.

England's other two changes come in the tight five, with Steve Borthwick making some big calls ahead of this game. Joe Marler starts ahead of vice-captain Ellis
Genge at loosehead prop, and Ollie Chessum is replaced in the second-row by George Martin.
</item 6>

<item 7>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'
</item 7>

<item 8>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England replacements: Theo Dan, Ellis Genge, Kyle Sinckler, Ollie Chessum, Billy Vunipola, Danny Care, George Ford, Ollie Lawrence.

The Springboks are unchanged from that epic quarter-final with France, having all played their part as Jacques Nienaber's side hung on in the second half. The
contribution of backs Kurt-Lee Arendse, Damian de Allende and Cheslin Kolbe saw all three cross for first-half tries, with Kolbe's coming off the back of a
charged-down conversion from France's Thomas Ramos. Preventing those two points proved crucial in the end.

The key contribution, though, came from Eben Etzebeth, who bulldozed his way over to score the only five-pointer of the second half and wrestle the game back
into South African hands.
</item 8>

<item 9>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby
</item 9>

<item 10>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
In short, this semi-final requires a huge step up in quality for Borthwick's side, and that's why we think that the Springboks will be victorious at the Stade
de France. South Africa have already shown they can match, and beat, the world's best, and we think that they'll do the latter in Saint-Denis.
</item 10>
=== END ITEM id=ddd6d425f9823f57 ===
