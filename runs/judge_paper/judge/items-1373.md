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

=== ITEM id=9f1bdb31c9640ac3 ===
Gold sentences and facts:
Sentence 0: “I’m here until the end,” she said in her press conference on Friday.
  Fact 0.0: A female speaker said 'I'm here until the end' in her press conference on Friday.
Sentence 1: He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with the London club went so poorly.
  Fact 1.0: A man has been out of work since his tenure with the Blues came to a close back in April.
  Fact 1.1: It would be a hard sell to fans for this man to return, since his time with the London club went so poorly.

Retrieved items:
<item 1>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
"I had to set some standards"



Erik ten Hag claims there was not a good culture among staff and players before he joined Man United 😬pic.twitter.com/DJWe42mDXY — Sky Sports Premier League
(@SkySportsPL) December 5, 2023

Man United vs Chelsea lineups

Midfielder Casemiro and defender Lisandro Martinez remain out for Manchester United, while Christian Eriksen may not return in 2023.

Mason Mount missed United's defeat at Newcastle United on Saturday and Jonny Evans has not played for almost a month because of a thigh issue. Amad Diallo and
Tyrell Malacia have long-term injuries.

Man United lineup (4-2-3-1, right to left): Onana (GK) — Dalot, Maguire, Lindelof, Shaw — Amrabat, McTominay — Antony, Bruno Fernandes, Garnacho — Hojlund.
</item 1>

<item 2>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Certain things [here] are different,” shrugs Ten Hag when asked why the No 1 is taking time to settle in England. “OK it is 11-v-11, and it is about attacking,
defending, transition. But there is more: the approach to the game, the speed of the game, the intensity. The varieties in styles. So there is absolutely a
difference and players need to adjust, need time to do so.”

Last season’s derby at Old Trafford ended in a 2-1 win for United. Bruno Fernandes’s equaliser was controversial with Marcus Rashford offside but not so the
goal was chalked off as he had not touched the ball. So far this term the captain has scored two goals in 12 United appearances and Ten Hag has no doubt he can
be a major factor in Sunday’s showdown.

“If he does [score] again, I’ll be happy,” the manager says. “Bruno is an [effective] player when we have a lot of the ball.
</item 2>

<item 3>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

“It’s how you are and about your character,” Pochettino said. “He feels football. I am 51 and still sometimes I feel that decisions are no good or unfair. You
are competitive. Like with Mikel or myself or other coaches that played football, in our minds, we still believe we can play. Still, we have the animal
instinct. We cannot be thinkers.”

For Pochettino, it must be about channeling the emotion in the right way and not suppressing it. More broadly, he is in a better place after the wins at Fulham
and Burnley, which followed the Carabao Cup victory over Brighton on the day that James was charged.
</item 3>

<item 4>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
The specifics of how that will work without Hayes’ involvement remain a mystery. It would be understandable for players to feel like they are still auditioning
for an absent director until May rolls around, while still knowing they must perform at the USWNT standard.

Along these same lines, there is at least the suggestion that the federation could be willing to sacrifice coherent preparation for this upcoming Olympic
tournament to focus more on the longer-term project of the 2027 World Cup. That itself represents a marked change from the expectations and pressure of constant
performance and winning that the team is known for.

Is that a good or a bad thing? Perhaps a little bit of both. The USWNT shouldn’t be ruled out of contention for the Olympics by any stretch, but this past
summer’s World Cup did reveal that the problems facing the team are far more foundational than just poor coaching decisions or the strange midfield chemistry.
Balancing realism with the pressure to win feels like a much more sustainable path forward for the USWNT.
</item 4>

<item 5>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
It’s game management.

Advertisement

“They go ahead and they’re already winning the game but they try and win it again and again and leave themselves open. Jarrod Bowen was outstanding out there
and you can really exploit them. They leave space and they make poor decisions on the ball and it’s costing them at the moment.”

Spurs had rebounded from three consecutive defeats with a stirring 3-3 draw at Manchester City over the weekend, but are now outside the Champions League
qualification spots in fifth place – and level with Manchester United.

1 – Tottenham are the first side in Premier League history to…



…fail to win five consecutive games despite going 1-0 up in each match.



…lose three consecutive home games despite going up 1-0 up in each match.
</item 5>

<item 6>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 6>

<item 7>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
That chance proved to be an early warning which Chelsea failed to heed as Sarr flicked on a sixth-minute diagonal pass for Kulusevski, who calmly rifled into
the near-right corner to put Tottenham back top of the table at that point.

Pedro Porro almost doubled Spurs’ lead a minute later as he rifled upon goal from Kulusevski’s pullback pass of Brennan Johnson’s cross, yet Sanchez was able to
make a comfortable save.

Chelsea’s first chance eventually came in the 11th minute as Jackson forced a save from Guglielmo Vicario, before Son briefly extended the hosts lead two
minutes later with a tap-in but he was ruled offside via his right shoulder position.

Tottenham soon afterwards began to lose their composure and Romero was lucky not to be sent off in the 21st minute, after he retaliated with a kick on Levi
Colwill following an initial shove on the Argentinean.
</item 7>

<item 8>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer that has continued through the start of November.

Unless the situation changes drastically, Hayes will only have two camps, including four friendlies, with the USWNT ahead of the 2024 Olympics in Paris. She’ll
miss three international windows between Tuesday’s announcement and her planned start date, including the 2024 CONCACAF Gold Cup in February and March.

U.S. Soccer has a plan in place for the transition. Current interim head coach Twila Kilgore will continue in the role through May and will remain as a
permanent assistant coach under Hayes after the swap is complete. U.S. Soccer said that Kilgore and the rest of the technical staff are working on a handoff
plan for Hayes.

“This is a unique situation, but the team is in safe hands with Twila,” Crocker said. “Her stewardship will be crucial during this period as we are focused on
success at the Olympics.
</item 8>

<item 9>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream #PLGoalRush LIVE, or on-demand after, here ???? https://t.co/qE2Kd4Kh3N#OptusSport #PL pic.twitter.com/uRXhc07AAo — Optus Sport (@OptusSport) December
23, 2023

Richarlison scored for a third consecutive match against his old club Everton as Ange Postecoglou’s Tottenham ensured they would be in the EPL top four at
Christmas after a hard-fought 2-1 home win.

Sean Dyche’s visitors crucially had a Dominic Calvert-Lewin effort ruled out by VAR in the 51st minute which could have changed the complexion of the match.

Son Heung-min added to Richarlison’s opener and Andre Gomes scored a later consolation for Everton.

Burnley took all three points with a deserved 2-0 victory over hosts Fulham as Rebecca Welch became the first female referee to take charge of an EPL match.
</item 9>

<item 10>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Emma has endorsed Twila, she will be a key part of Emma’s staff when she arrives and moving forward, and we are excited for what’s to come with our USWNT
program.”

GO DEEPER How Emma Hayes' winning ways at Chelsea can benefit USWNT on the field

It’s still an extremely tight turn for the Olympics, with 18 days for Hayes to get situated with the team ahead of the tournament, between the two international
windows from May 27 to June 4, then July 8 to 16. The Olympic tournament will start on July 25.

Advertisement

There are, of course, logistical questions about roster selection over the next few months. Some of those may be answered relatively soon, with the roster for
the upcoming camp that begins at the end of the month imminent. The greater challenge will likely be ongoing player evaluation over the next six months, at a
time of great transition within the squad. The specifics of how that will work without Hayes’ involvement remain a mystery.
</item 10>

<item 11>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
"In high ball regains, we are a top of the Premier League. In the middle [of the pitch] ball regains, we are top. The pressing is very good. So that's not the
truth, what he [Carragher] is telling."

Please use Chrome browser for a more accessible video player FREE TO WATCH: Highlights from Sheffield United’s Premier League clash with Manchester United

In a fortnight where United have picked up three straight wins against Brentford, Sheffield United and FC Copenhagen, albeit via far from convincing
performances, Christian Eriksen said United still need to find their "balance and routines" this season.

Asked about those comments, Ten Hag said: "We are in the right direction. We have a way to go. I see positives but I see negatives.

"What Christian is referring to is we are not consistently in possession on the level where we can be. Like Arsenal away, the first 30 minutes in Bayern Munich
away. We should do this on a consistent basis.
</item 11>

<item 12>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Stream LIVE, or on-demand after, here ???? https://t.co/09W1bq24Ur#OptusSport #PL pic.twitter.com/IcKGdmJv4O — Optus Sport (@OptusSport) December 23, 2023

Manchester United legend Peter Schmeichel lashed the team’s lack of a plan and blasted them for looking slow against the Hammers.

“It was just in many ways irrelevant. There was no plan to it,” he said. “West Ham got an easy win which is a little bit silly to say.

“When I look at the team, I don’t know what the identity of the team is. What is the style of play, what’s the plan?
</item 12>

<item 13>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
Pedro Porro, Cristian Romeero, Mickey van de Ven (Hojbjerg 45+1′), Destiny Udogie, Yves Bissouma, Pape Matar Sarr (Bentancur 61′), Dejan Kulusevski (Skipp 61′),
James Maddison (Royal 45+1′), Brennan Johnson (Dier 34′), Son Heung-min

Substitutes: Oliver Skipp, Pierre-Emile Hojbjerg, Richarlison, Bryan Gil, Emerson Royal, Eric Dier, Giovani Lo Celso, Fraser Forster, Rodrigo Bentancur

Chelsea: Robert Sanchez, Reece James (Gusto 77′), Axel Disasi, Thiago Silva, Levi Colwill (Cucurella 45′), Moises Caicedo, Enzo Fernandez, Cole Palmer, Conor
Gallagher, Raheem Sterling, Nicolas Jackson

Substitutes: Marc Cucurella, Benoit Badiashile,
</item 13>

<item 14>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
“I thought we were poor in both areas with our finishing and also both goals we conceded were terrible.

There’s no point in feeling sorry for ourselves, looking for a cuddle anywhere. There’s only one way to change our circumstances and that is to come here on
Sunday and put in a performance, not just play good football but go out there and show some conviction about ourselves as a team.

Advertisement

“Sometimes we can disguise how we’re going by playing some nice stuff but like I said from day one, that’s not what I’m about.

“I want to win and that’s why I came to this football club and that’s the message. We’ve still got a long way to go, I’ve said that from the start, we’re still
right at the beginning what we need to create and days like today just give me further evidence and fuel of how much we need to do.”

James Ward-Prowse puts West Ham ahead against Tottenham ????
</item 14>

<item 15>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
And they boast a brilliant record against the Blues of late, winning six out of their last seven meetings.

They have triumphed in their last three trips to Stamford Bridge, and haven't lost in a London derby since May 2022.

As for Chelsea, they go into their match against the Gunners hopeful of a result having won their last three games in all competitions.

Both teams face injury doubts, with Arsenal to potentially be without Bukayo Saka and William Saliba, while it remains to be seen if Reece James and Armando
Broja feature for the hosts.

9 Saka missed Arsenal's 1-0 victory over City at the Emirates last time out Credit: Getty

Sheffield United desperate for points against lacklustre Manchester United

With just one point on the board, Sheffield United's season could not be going much worse given they are rock bottom.

With a fixture against Manchester United on the cards, though, they may feel their hopes could be looking up given Erik ten Hag's men disappointing season.
</item 15>

<item 16>
Everton players and fans will be fired up by points penalty, asserts Sean Dyche (The Guardian, 2023-11-24T22:30:32+00:00)
You can’t guarantee it’s made a difference [to players’ resolve] yet because we haven’t played any games. All I know is that the players know what we’re doing.
They’ve referenced that to me probably for the last six weeks.”

Everton’s appeal against the 10-point deduction will be heard by a different three-person panel before the end of the season. Dyche, however, wants his team to
forget about the appeal and accept the reality of being joint-bottom of the Premier League on four points. That outlook, he believes, will help Everton focus on
hauling themselves out of the relegation zone.

Abdoulaye Doucouré celebrates Everton’s win at Crystal Palace, the club’s sixth victory in nine games. Photograph: Jacques Feeney/Offside/Getty Images

On the prospect of survival, Dyche said: “I’ve never lacked confidence in my own ability, in my staff’s ability or the players’ ability.
</item 16>

<item 17>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Even if you’re not playing well, if the team is not playing well, or you’re not playing to the standards you expect, you still have to win – so find a way to
win, be pragmatic.”

United have been so of late, claiming victories over Sheffield United, Brentford and FC Copenhagen in their last three matches. But Ten Hag’s “depends on the
opposition” and “how good is the team around him” comments sound like hints. In contrast to Inter’s Simone Inzaghi, the United manager may harbour doubts about
deploying Onana high up the pitch against City.

The Dutchman is still building his team and may be wary of the uneven form of United’s new keeper. Bought for £47.2m, Onana arrived on a ticket of being a
high-grade shot-stopper, technically impressive too with the ball at his feet. But until impressing against Brentford and Copenhagen, Onana had bombed, costing
United goals and points.
</item 17>

<item 18>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 18>

<item 19>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
That he can shout down the noise in the short term while building something in the long term. Right now, in the raptured afterglow of a big win, it feels like a
gamble being won. But best to come back and check in a few days, just to make sure.
</item 19>

<item 20>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
Arsenal represent an acid test of the progress and the start of a daunting run in the league; after them, Chelsea face Brentford, Tottenham, Manchester City,
Newcastle, Brighton and Manchester United. Pochettino, though, pushed back against the notion that he was seeking a first statement win at the club, something
to ignite his tenure. All he wants are points to drive his team up from 11th in the table.

Pochettino has appeared mired in a selection crisis since his arrival at Chelsea; James only one element of it and he will be without seven others through
injury against Arsenal.

“The moment we have all the squad at a high level and fit, we can say: ‘OK, we are going to Arsenal, Manchester City and we are going to beat you.’ At the
moment, with the circumstances, I don’t think this game is going to be a statement to say: ‘Be careful, Chelsea is coming,’ or something like this.”
</item 20>

<item 21>
Arteta reveals Pochettino influence on his career | Poch: He is family (Sky Sports, 2023-10-20T21:30:00+00:00)
Havertz will command particular focus after his early struggles at Arsenal. With just one goal in 12 appearances for the club there are questions about his form
and role.

Please use Chrome browser for a more accessible video player Ahead of Kai Havertz's return to Stamford Bridge, Sam Blitz takes a forensic look at his tactical
role at Arsenal

Armando Broja will miss Chelsea's game against Arsenal due to "irritation" in his knee. Late checks will be made on the fitness of Reece James, Axel Disasi and
Nicolas Jackson, while Benoit Badiashile could return from a long-term injury.

Enzo Fernandez and Moises Caicedo should be available despite travelling to South America during the international break.

Ben Chilwell, Carney Chukwuemeka, Christopher Nkunku, Romeo Lavia, Wesley Fofana and Trevoh Chalobah remain unavailable.

Arsenal will make a late call on the fitness of Bukayo Saka, William Saliba and Leandro Trossard.
</item 21>

<item 22>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
All three players left Nice this summer. Newcastle’s sporting director, Dan Ashworth, and Paul Mitchell, Monaco’s sporting director until March, are among those
mentioned in dispatches as a potential Murtough replacement.

Can any CEO make United a best-in-class elite club again?

Blanc, who has an MBA from Harvard, is billed as the “Lionel Messi of business” by Nice’s chief executive, Fabrice Bocquet. At 60 years old Blanc has been the
Juventus CEO and president, CEO at Paris Saint‑Germain and French Tennis, and the executive overseeing the Tour de France. But once he or whoever becomes
Arnold’s successor is in place, it will dawn rapidly that United are a one-off hydra-headed beast of a club which is treacherously difficult to control. And
there is this, too: across town Manchester City have accrued six championships since United’s last in 2013 and boast a commercial operation that leaves
Ratcliffe’s new concern light years behind.
</item 22>

<item 23>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
Colwill though found himself hooked off at half-time by Mauricio Pochettino for Marc Cucurella, due to his ill-tempered first-half antics.

Upon the restart, Chelsea quickly made their one-man advantage count but their recurring end product issues reared its head, as Palmer rifled wide in the 47th
minute in a fast start from the Blues.

Sterling then broke forward seven minutes later but his shot was blocked by Vicario, before he seconds later broke forward on a counter-attack, which drew a
foul from Udogie who was consequently dismissed to put Spurs down to nine men with 35 minutes left on the clock.

If Chelsea thought that decision boosted their prospects of a second goal then poor finishing looked to be their Achilles heel, as Jackson flicked over the bar
from James’ cutback pass following a short free-kick.

Jackson continued to push for his third goal of the campaign but his 58th minute shot was punched away by Vicario, when then blocked Palmer’s rebound shot.
</item 23>

<item 24>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
Mauricio Pochettino said he made Reece James Chelsea captain because of ‘his personality, his leadership, the respect he has from the club’. Photograph: Darren
Walsh/Chelsea FC/Getty Images

There is a reason why Pochettino calls Arteta “family”. They joined PSG at the same time, early 2001; Pochettino was 28, Arteta 18. They hit it off immediately,
Pochettino wowed by Arteta’s charisma and his reading of the game. “Already, he was a coach,” Pochettino said, and he has no qualms with what some perceive as
Arteta’s histrionic touchline behaviour.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 24>

<item 25>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
They’ll always have Paris. Throughout this Nations League campaign this has been an England team fighting its own sense of entropy, still giving everything,
still trying the same plans and processes in the hope that eventually something would click. And at 9.34pm on a briskly cold Wembley night, it finally did. Ella
Toone’s injury-time goal earned England three points and kept their faintly glowing Olympic dream alive. But somehow it also did so much more.

It breathed life into a team and a cycle, and a moment that for so much of the evening had felt like it was drawing to its natural conclusion provided a rousing
encore to a cast preparing for its final curtain. There will be new challenges and new frontiers to conquer soon enough, a fresh set of European Championship
qualifiers in the spring, but for now this iteration of the Lionesses still has tricks up its sleeve, still has talent and creativity to burn, still has a fire
that can scorch any team on the planet.
</item 25>

<item 26>
Everton players and fans will be fired up by points penalty, asserts Sean Dyche (The Guardian, 2023-11-24T22:30:32+00:00)
Players reacted to last Friday’s news by sharing defiant messages on their WhatsApp group. After a run of six wins in nine games in all competitions, that had
lifted Everton to 14th in the table prior to being plunged back to 19th, their resolve to maintain momentum and haul the club out of danger was relayed to the
manager at Friday’s meeting.

“The feeling was very positive in the sense of building on the mentality that’s been there lately,” said Dyche. “We all spoke about that, saying this hasn’t
changed it and it’s building regardless of this current situation. The players are clear-minded. They have their WhatsApp group and when the news broke they put
it straight on there: ‘Right lads, just another challenge, this is how we take it on’. The group is taking ownership and that’s a powerful thing.

“I got from them the sense that: ‘What we’re building, we don’t let anything disrupt that and we don’t let anything get in the way’.
</item 26>

<item 27>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
Photograph: Michael Regan/Getty Images

Ensuring a working relationship with the Glazers

For paying an overpriced £1.3bn for his 25% Ratcliffe obtains a major element of control but still the Glazers remain majority owners. The 71-year-old oversees
football policy but these do not exist in a vacuum given that the commercial side affects this and vice versa. Ratcliffe accepts Manchester United have an
impressive revenue-based model that makes the club akin to a cash machine but he is not enamoured of how the money generated is invested – primarily in the
squad. What happens then, for example, if Ten Hag wants a new midfield ace, Ratcliffe agrees but the Glazers cite a downturn in income and thus seek to block
the signing?
</item 27>

<item 28>
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
</item 28>

<item 29>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
That would have been too simple. Rather for their next game at Burnley on 7 October when, according to Pochettino, James was back in contention for selection.

The 23-year-old was not called up by England for the internationals over this past window and the idea was that he would work to be ready for Saturday’s home
derby against Arsenal. And yet there have been stories of an injury setback, even that he may need surgery to get to the root of his hamstring issues.

Pochettino bustled his way through those questions on Friday and the best read on James’s status was that he would be back in the squad for Arsenal, most likely
as a substitute. Gusto is available again after his own suspension. “It [James’s hamstring] is much better,” Pochettino said. “He is progressing well, still not
at his best, 100%. But maybe to help the team at some point …. maybe, he is close to be there.”
</item 29>

<item 30>
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
</item 30>

<item 31>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
However, names such as Alli, Heung-Min Son, Toby Alderweireld, Eric Dier and Kieran Trippier set the side up for a bright future, before chairman Daniel Levy
got in Mitchell’s way.

Leaving what he called his ‘dream job’ after less than two years, it was reportedly due to Levy’s over-involvement in negotiations and the recruitment process,
something which wasn’t an issue at his next club – Leipzig.

Ralf Rangnick landed Mitchell and more success followed. Signings such as Dani Olmo, Christopher Nkunku and Amadou Haidara again showcased what Mitchell can do
with the shackles off, and again the results followed.

6 Getting Nkunku from PSG was one of Mitchell's biggest coups Credit: Getty

Many of his signings starred in a run to the Champions League semi-finals in 2020, and in 2022 won the club’s first ever trophy with a DFB Pokal.
</item 31>

<item 32>
Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe (TalkSport,
2023-11-29T23:40:49+00:00)
Manchester United's domestic form has certainly shown signs of improvement in recent weeks, but that's far from the case in Europe.

Erik ten Hag's side have once again dropped points from a winning position as they drew 3-3 with Galatasaray in the Champions League, a damaging result in their
bid to reach the last 16.

4 United were once again the victims of their own demise Credit: Getty

Goals from Alejandro Garnacho and Bruno Fernandes fired them into a two-goal lead within 18 minutes before Chelsea loanee Hakim Ziyech struck back.

Scott McTominay scored what he thought would have been a crucial third shortly after the break but United failed to hold on.

Another free-kick from Ziyech caught Andre Onana off guard once again and a strong strike from Kerem Akturkoglu levelled the scoring.
</item 32>

<item 33>
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
</item 33>

<item 34>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
But maybe to help the team at some point …. maybe, he is close to be there.”

It has not been straightforward and it seems to remain that way. Yet there were two takeaways from Pochettino on James – with a common thread running through
them. First, James could be like a coiled spring when he finally returns to action. And second, Pochettino was in no mood to criticise him for his indiscipline.
On the contrary. He was happier to talk up James’s passion.

“It’s part of the business,” Pochettino said. “He was upset about the situation [against Villa] and then you need to understand that he is so committed to this
project from the beginning. He feels really bad because he cannot help the team [because of his injury].

“He made a mistake, we all make them and he was really sorry about it. I think he’s not going to repeat the situation. This is a young guy that wants to play
football.
</item 34>

<item 35>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
Sarina Wiegman talks a lot about defending against counterattacks, because this is basically the key to the whole enterprise. Cut off the counter, and with
England’s technical quality they can easily dominate. It is also the essential measure of how the team is functioning as a collective: the efficiency of the
press, the closing of the angles, the command of space. Lineth Beerensteyn’s two goals were the sort England simply were not conceding four months ago.

Mary Earps shows her frustration after conceding a second goal to the Netherlands. Photograph: Tom Jenkins/The Guardian

So England passed the ball around a bit and gave it away a bit and essentially ran around trying to feel normal again, and on countless occasions Kelly or Hemp
would win a first or second ball, look up and see a penalty area empty of white shirts.
</item 35>

<item 36>
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
</item 36>

<item 37>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
They will always have the past at Old Trafford, but Evans may be consigned to it; perhaps, in time, Ten Hag, too.
</item 37>

<item 38>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Wilson Odobert and Sander Berge scored to lift Burnley off the bottom of the table.

Dominic Solanke scored a hat-trick to earn Bournemouth a 3-2 win at Nottingham Forest and ruin Nuno Espirito Santo’s first game in charge at the City Ground.

Advertisement

Forest, who scored through Anthony Elanga and Chris Wood, had defender Willy Boly controversially sent off in the first half.

They were level until Solanke headed home in the fifth minute of stoppage time.

© AAP
</item 38>

<item 39>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
This was a win that felt a lot tighter than it really was: United inexplicably hanging on at the end despite enjoying 28 shots and an xG tally of 4.07. For long
periods United played some of their most authentically thrilling football of the season, a doctrine of relentless attack that had Diogo Dalot flying forward
from right-back even as the minutes ticked down.

It was also a game that felt more chaotic than it really was. In fact the slippery, slingy nature of the play was largely by design: United hollowing out their
midfield in order to throw bodies forward, Chelsea hollowing out theirs in order to leave men up for the breakaway. It was a game that hung on United’s
determination to keep allowing Chelsea counterattacks, and Chelsea’s determination to keep messing them up. It was a game, above all, that hung on the curious
main character energy of Scott McTominay.
</item 39>

<item 40>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
As Manchester United's season continues to slip away, the pressure around Dutch boss Erik ten Hag continues to get more palpable.

The latest slip saw the Red Devils crash out of Europe as they fell 1-0 to Bayern Munich on the final day of Champions League group-stage play, finishing bottom
of their four-team group to leave them eliminated from Europe in their worst performance in the competition since 2005.

On top of that, they set the record for most goals against in a Champions League group stage by an English club, with 15 conceded in their six games. Their
defensive record has collapsed as the season goes on: from a Premier League standpoint, they sit eighth in the English top flight, but their -5 goal
differential is the worst of any club in the top 10.

With Jim Ratcliffe having now completed the formalities around his investment into the club, taking control of 25% of his boyhood club after a year of
negotiations, the pressure will be piled on even more.
</item 40>

<item 41>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
“I was always breaking her pipes,” he recalled. He has come a long way since then; particularly in 2023. “Yeah, it’s been crazy,” he said. “The changes have
happened so fast. From winning the Euros to signing for Chelsea then getting signed up here.”

Palmer chats with Arsenal winger Bukayo Saka (The FA/Getty)

His part in England Under-21s’ European Championship win may have caught Gareth Southgate’s eye. His immediate decision was whether he had done enough to
impress Pep Guardiola. There had been interest in taking him on loan, from Brighton and Burnley and RB Leipzig. City decided that he could only leave on a
permanent deal and Chelsea forked out £42.5m. “Who knows what would have happened if I had stayed?” Palmer mused. “Maybe I would have played more, maybe not.
But I think the decision to go to Chelsea so far is paying off.”

That is an understatement.
</item 41>

<item 42>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
Theirs was another colossal effort, a far better performance than at Everton or Tottenham, but while Eddie Howe had the luxury of one attacking substitute – in
Alexander Isak – three replacements combined for Milan’s decider: first Luka Jovic found Noah Okafor and he picked out Chukwueze, who placed a shot past Martin
Dubravka.

It was a sign of Newcastle’s determination to win, not to settle for the point and the Europa League place, that they were short-staffed at the back because
centre-back Fabian Schar had charged into the Milan box in open play. Likewise, in a frantic finale, they almost conceded a third: with Dubravka out of his
goal, Theo Hernandez almost scored from the halfway line and Fikayo Tomori hit the post.

As they wonder what might have been, they can reflect on the turning point of Pulisic’s strike, around the time Warren Zaire-Emery drew Paris Saint-Germain
level against Borussia Dortmund.
</item 42>

<item 43>
Ten Hag responds to Carra: Man Utd have actually improved (Sky Sports, 2023-10-28T21:30:00+00:00)
But Ten Hag says he has full faith that the England forward, who scored 30 goals last season, can find his scoring boots again.

"If you give him the trust and I play him every game, most games also he finishes the game. I have strong belief he will return to scoring a lot of goals," said
Ten Hag.

"It also has to do with the co-operation, with the movement around and the distribution to him. We have to work on that and that's my focus point."
</item 43>

<item 44>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
“We’ll see what the appeal brings. The mentality is key for me, we took a knock but there have been a lot of knocks here over the last couple of years.

Advertisement

“But the team mentality is growing. I can’t emphasise it enough, the commitment to each other and the connection is a powerful thing because I know we’ve got
quality.”

Eddie Howe’s side are proving one of the most inconsistent in the league, albeit with a lengthy injury list.

In the last month, they have defeated Manchester United, Arsenal and Chelsea – as well as drawing at French heavyweights PSG in the Champions League – but have
now lost comfortably to two cellar dwellers in Everton and Bournemouth.

“We have to do better,” said Howe.

“There are reasons behind every performance and we have struggled to change players. That is a big miss with the quality of players out. The longer you do it,
the harder it gets.
</item 44>

<item 45>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
The 44-year-old Italian has led the tiny club to new heights, qualifying for the Europa League for the first time in club history and establishing themselves as
top-four contenders, building upon the success under Graham Potter. Yet could Potter's failure at Chelsea cause some trepidation around another Brighton success
story moving on up?

Michael Carrick

This list wouldn't be complete without mention of the likeliest of those with direct Manchester United links. Former United midfielder Michael Carrick, who made
316 appearances for the club over a 12-year Old Trafford tenure, is at Championship club Middlesbrough, his first permanent managerial appointment. They sit
11th in the second tier as of mid-December, just three points back of the playoff spots in a crowded top half.

Carrick's lack of experience is glaring, but hiring someone with deep knowledge of the club could be very enticing, and the fact he hasn't bombed in his first
job is notable.
</item 45>

<item 46>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
But he also pops up in the channels, on the wing, either supporting the man in possession or offering himself for the quick switch of play.

Great times for a great guy. But, of course, if you commit a midfielder to this role, then you leave gaps elsewhere. One of the reasons Chelsea found it so easy
to play through United was that, with the forwards pushed high and McTominay as the secret sauce in the final third, there were open spaces every time they
played through the press. Chelsea’s goal was a case in point: two simple vertical passes straight through the centre via Mykhailo Mudryk to Cole Palmer, passes
that McTominay could see coming but could do precious little to stop.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 46>

<item 47>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Advertisement

The hiring process included multiple rounds of evaluation, with the list of candidates becoming smaller each time. The first pass was driven purely by data,
which was then whittled down to a double-digit list Crocker was considering as of September, and then a final shortlist, which also included Tony Gustavsson,
head coach of Australia. Multiple sources confirmed both Hayes and Gustavsson flew to the U.S. for interviews.

One source who was briefed on the situation said the federation had also checked in on the availability of Sarina Wiegman, despite clear messaging from both the
English FA and Wiegman herself in August. “I’m staying out of it. I’ve heard it (from the press officer) but no, I’m with England and I’m really happy with
England, and I have a contract until 2025,” Wiegman said. A representative from her camp declined to comment for this story.
</item 47>

<item 48>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s own messaging. Crocker, in a September meeting with U.S. reporters along with Cone and U.S. Soccer CEO J.T. Batson, said he hoped to have a new head
coach in place by December.

But the initial contact with Hayes was made a couple of months ago, early into the search, with all three top-level executives from U.S. Soccer involved in
those talks. The trio also described the interview process to journalists in that September meeting – a U.S. soccer statement describes it as involving
“psychometrics and abstract reasoning tests, in-depth discussions of strategy, coaching philosophy and the current player pool, as well as evaluation on the
reactions to pressure, culture-building and interactions with players and staff.” USMNT coach Gregg Berhalter went through a similar process, including an
abstract reasoning test, when he was re-hired by Crocker earlier in 2023.
</item 48>

<item 49>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Sporting director Matt Crocker made the final decision to hire Hayes after a search process that began in August, following the team’s surprising exit in the
round of 16 at the World Cup and the subsequent departure of head coach Vlatko Andonovski.

“She has tremendous energy and an insatiable will to win,” U.S. Soccer president Cindy Parlow Cone said in a statement. “Her experience in the USA, her
understanding of our soccer landscape and her appreciation of what it means to coach this team makes her a natural fit for this role and we could not be more
pleased to have her leading our women’s national team forward.”

Though Hayes is seen as one of the world’s top coaches in women’s soccer, the appointment still comes as something of a surprise. Here’s how the deal got done.

Details of the deal

At least part of the surprise surrounding Hayes’ hire – and the six-month runway before she officially takes charge – is down to U.S. Soccer’s own messaging.
</item 49>

<item 50>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
This is a young guy that wants to play football. He was frustrated and he made a mistake. That’s it. Finish.”

Pochettino made James his captain because of “his personality, his leadership, the respect he has from the club and his teammates … he has Chelsea DNA”. He
described him as an appointment “for the present and the future – we cannot change captain every season, every six months”.

It is also easy to imagine that Pochettino sees something of himself in James; the self-belief, the running on the emotions of a match day. Pochettino made the
point that he had got himself booked for dissent during the Burnley game. And then there is his connection with the Arsenal manager, Mikel Arteta; there are
similarities, albeit this relationship is rather deeper, going back to their time together as players at Paris Saint-Germain.
</item 50>

<item 51>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
Chelsea continued to press and eventually got their reward as Sterling broke forward up the right flank at the start of the 75th minute, and found Jackson
unmarked to rifle his pass into the left corner to put the Blues ahead at Pochettino’s former team.

Tottenham however seemingly responded three minutes later as substitute, Rodrigo Bentancur flicked a free-kick behind for Eric Dier to volley into the top-left
corner, but he was adjudged offside as Bentancur nodded the ball into his path.

Chelsea eventually clung on for a fourth league win of the season after Bentancur narrowly put a 86th minute header wide, as Spurs valiantly tried to chase an
unsuccessful equaliser in which Son saw a 93rd minute shot denied by Sanchez.

Jackson subsequently rounded off Tottenham’s nightmare evening a minute later as he turned Gallagher’s pass into the net to seal a triumphant return to North
London for Pochettino and his Blues, before he completed his hat-trick two minutes later.
</item 51>

<item 52>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
After all that pressure, all that tender hope, it looked a bit like a training goal.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

And that was pretty much that. From here United’s season continues to narrow. They can’t win the league now, can’t win the Champions League, can’t win the
Carabao Cup.
</item 52>

<item 53>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
Nicolas Jackson

Substitutes: Marc Cucurella, Benoit Badiashile, Mykhailo Mudryk, Noni Madueke, Lesley Ugochukwu, Malo Gusto, Dorde Petroivic, Ian Maatsen, Deivid Washington

Referee: Michael Oliver

Like this: Like Loading...
</item 53>

<item 54>
City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge (The Roar | Sports Writers Blog, 2023-11-12T20:48:50+00:00)
Manchester City won't care. Limbs in the away end at Stamford Bridge.



WATCH | https://t.co/HPhtrVhoVE#OptusSport pic.twitter.com/DWcenGZudc — Optus Sport (@OptusSport) November 12, 2023

Chelsea boss Mauricio Pochettino said: “That is why we always say the Premier League is the best in the world. We want to do something special here and this is
how we need to play. We scored four goals against the best team in the world. It’s good to go into the international break like this.

“The approach of the game was good. We were brave. That is the idea of football. I am so proud and happy. The players never gave up. The reaction of the players
was amazing.”

Pochettino struggled to stay composed during the game and apologised for his behaviour.
</item 54>

<item 55>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 55>

<item 56>
Another cruel comeback sees Newcastle perish in group of death (The Independent - Sports, 2023-12-13T22:16:07+00:00)
Newcastle failed to clear as the ball ricocheted around their box before Olivier Giroud found Pulisic to poke in a shot. The American had been a target for
Newcastle last January; how they must now regret the fact he joined Milan in the summer.

Then, with 11 minutes remaining, Rafael Leao sprinted clear and shot against the post. If this had threatened to be the derby of the depleted, Milan had Leao
back for the first time in a month. Newcastle had Callum Wilson able to start, but their goal came from a less regular source, with a thunderbolt from
Joelinton.

Joelinton thumped home Newcastle’s goal (Reuters)

One of the great success stories of Howe’s reign, the misfit forward who was reinvented as a talismanic midfielder gave Newcastle a deserved lead. Joelinton’s
rasping, rising shot from the edge of the box flew past Mike Maignan. He had taken Miguel Almiron’s pass and teed himself up.
</item 56>

<item 57>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
GO DEEPER USWNT's Emma Hayes hire could create a problem in preparation for the Olympics

What’s next for Chelsea?

Questions surrounding Hayes’ future have cropped up previously in her tenure at Chelsea. She was linked to several jobs in the men’s English Football League in
the past and has always been interested in managing Spain’s national team, although there was never an official conversation with Spain’s federation. So it was
a question of when, not if, Hayes would leave. Still, news of her decision came as a shock to her staff and players.

A few staff members were told on the morning of Nov. 4 before Chelsea’s away game at Aston Villa kicked off at 12.30 p.m. UK time. Most of the other staff
members found out with the players in the post-match meeting minutes before the official club statement, which Hayes had no hand in writing and did not even see
before publication, was released at 3 p.m.
</item 57>

<item 58>
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
</item 58>

<item 59>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Man United subs (12): To be confirmed

Ben Chilwell remains unavailable for Chelsea and Malo Gusto has been missing with a knee injury, but fellow defender Reece James is available again after
suspension, as is Marc Cucurella.

Romeo Lavia and Christopher Nkunku are likely to still be sidelined and Conor Gallagher is banned after receiving two yellow cards against Brighton. Carney
Chukwuemeka, Trevoh Chalobah and Wesley Fofana are out.

Chelsea lineup (4-3-3, right to left): Sanchez (GK) — Disasi, Silva, Colwill, Cucurella — Sterling, Fernandez, Caicedo — Mudryk, Palmer, Jackson.
</item 59>

<item 60>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Already a top tier European owner of OGC Nice, there has been some success, but nothing compared to their French riviera neighbours, where Ratcliffe appears set
to look for inspiration.

Paul Mitchell recently left the principality of Monaco, but he could well be back there in meetings with Ratcliffe who is planning to appoint a sporting
director.

Mitchell fulfilled that role at Monaco after previous stints at Leipzig, Tottenham, Southampton and MK Dons, becoming one of football’s most respected
executives.

A former player in England’s lower leagues, Mitchell was previously linked with Chelsea and Liverpool, and much like Ratcliffe, he’s also from Greater
Manchester.

Starring mainly for Wigan and MK Dons, it was with the latter where Mitchell was moved upstairs, and made his first big move by spotting future England regular
Dele Alli after being named their head of recruitment in 2010.
</item 60>

<item 61>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
Uefa’s description of Onana intrigues because it suggests that what Guardiola introduced to the domestic game – a No 1 in Ederson who performs as a de facto
extra defender to maintain fluidity along the backline – has evolved further. The goalkeeper was so advanced as to be a No 6 (rather than, the technical report
posited, a “sweeper-keeper”) who can take up residence in the middle third of his half as his team pushes deep into enemy territory.

Onana did so against City for Internazionale but has this encounter with the champions come too soon for him to function as “the extra man” Ten Hag identifies?
“It also depends on the opposition, how strong is the press, for instance, and how good is the team around him,” the United manager says.

“So if it’s possible – in my philosophy he has to do it every game [but] it is not always possible, so you have to adapt and this what I make clear.
</item 61>

<item 62>
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
</item 62>

<item 63>
Everton players and fans will be fired up by points penalty, asserts Sean Dyche (The Guardian, 2023-11-24T22:30:32+00:00)
I’ve never lacked confidence since I’ve been here and, let’s face it, I’ve had some pretty tough times. But it’s not just about confidence. You’ve got to win
games, and this is just another big challenge.

“It is currently a momentary thing because we don’t know what the appeal brings, we don’t know how that changes things again. But until that period, and I have
made it clear to the players, be under no illusions because there are no guarantees things will change so we have to commit to what’s going on now as if that is
the reality. It might be changed but, until it’s changed, we have to commit to this reality. There are no guarantees that what goes before guarantees what comes
next. We have to keep pushing like we have been doing because we have lost 10 points and we have to get them back as quick as we can.”
</item 63>

<item 64>
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
</item 64>

<item 65>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“Of course, I’m disappointed to hear things being said in the press. I want to make sure I maintain my own professionalism in everything I do.”

Advertisement

U.S Soccer’s annual financial reports reveal the salaries of their head coaches and other executives. While Berhalter received a new contract this year, his
previous deal that ran from April 2021 to March 2022, earned him $1.6 million, including $300,000 in bonuses. During that same time, Andonovski earned $446,495,
of which only $50,000 was bonus money. With Hayes expected to earn close to $2 million per year in her deal, this will likely create a knock-on effect for other
international women’s coaches negotiating their next contracts.

U.S. Soccer’s rollout of their new head coach has not been an easy one for the federation’s communications staff, considering that Hayes is essentially
unavailable for any formal ceremonies or media appearances until her time with Chelsea is complete.
</item 65>

<item 66>
Erik ten Hag could pay the price for Manchester United’s incompetent Champions League campaign (The Independent - Sports, 2023-12-12T22:36:20+00:00)
Which, as the German champions have not lost in 40 group-stage games in this competition and United have not beaten anyone resembling a top team this season,
always felt unlikely.

This was far from their worst performance of the season, nowhere near as bad as Bournemouth or Crystal Palace or Newcastle, or the other defeat to Newcastle.
They were not as self-destructive as they had been in Denmark and Turkey and at home to Galatasaray. But they still lost.

Harry Maguire’s injury forced him off the pitch as 36-year-old Jonny Evans replaced him (EPA)

And they still went out, with four points, in fourth place. Go back a couple of months and Raphael Varane had said United could win the Champions League. On a
night when Ten Hag seemed to revise his opinion of the quadruple Champions League winner, the defender may need a rethink. Instead, United won a lone game in it
and even that required a penalty save to deny FC Copenhagen a draw.
</item 66>

<item 67>
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
</item 67>

<item 68>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Manchester United’s century old stadium needs an upgrade. Photograph: Greig Cowie/Shutterstock

Communications

The Glazers treat a microphone or TV camera as United fans do Liverpool or Manchester City: as a fierce enemy. No family member has given a press conference or
interview during their 18-year tenure.
</item 68>

<item 69>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
And as they rallied so spectacularly to salvage a match that looked cooked after 35 minutes, running on fumes and adrenaline, sustained by the sizzle of the
Wembley crowd, the cold air on their skin and the flickering candle of qualification for Paris next summer, it was as if a kind of energy was surging within
them.

An energy that we have not really glimpsed from them since they reassembled this autumn, barely a month on from the trauma of Sydney, still a little dazed and
concussed.

Beth Mead came on for Chloe Kelly at half-time, Georgia Stanway moved further forward and made herself a menace in the penalty area, and the old buddies Toone
and Alessia Russo teamed up off the bench. The Dutch retreated further and further into Wembley’s wide open spaces, giving Lauren James the time and territory
to dictate.
</item 69>

<item 70>
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
</item 70>

<item 71>
Pochettino looks past James mistake and stakes future on his ‘Chelsea DNA’ (The Guardian, 2023-10-20T21:30:28+00:00)
It was the moment when Reece James simply could not handle the frustration. Named by Mauricio Pochettino as the Chelsea captain, James had not played since the
opening weekend draw with Liverpool at Stamford Bridge when he injured a hamstring. Now, he seethed from the sideline as his team lost 1-0 at home to Aston
Villa, a game in which his replacement, Malo Gusto, was controversially sent off.

It was 24 September, Chelsea’s second home Premier League defeat on the spin after the one against Nottingham Forest and they were booed off again. James would
lose his cool with an official in the tunnel and, when a Football Association charge was announced three days later for abusive conduct, it felt like only a
matter of time before he was given a one-match ban.

It did not come in time for Chelsea’s visit to Fulham on 2 October, when James was still injured. That would have been too simple.
</item 71>

<item 72>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
“This is a huge honor to be given the opportunity to coach the most incredible team in world football history,” she said in a statement on Tuesday. “The
feelings and connection I have for this team and for this country run deep. I’ve dreamed about coaching the USA for a long time so to get this opportunity is a
dream come true. I know there is work to do to achieve our goals of winning consistently at the highest levels. To get there, it will require dedication,
devotion and collaboration from the players, staff and everyone at the U.S. Soccer Federation.”

Looking ahead for the USWNT

Hayes’ appointment will have an immediate impact, even if she’s not immediately present. The particulars around timing and the plan moving forward have been one
of the areas of discussion between Hayes and her representatives, Chelsea and U.S. Soccer that has continued through the start of November.
</item 72>

<item 73>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
However, it's also possible he decides to stay on for the 2026 World Cup cycle, particularly if England have a good tournament in Germany. In that instance, he
would only be available for double duty on a temporary basis, and that would simply be too messy to outweigh the benefits. Brazil are currently in that boat,
and it's all gone wrong for Fernando Diniz despite his Copa Libertadores victory with Fluminense.

Mauricio Pochettino

In the event Todd Boehly proves to sport just as itchy a trigger finger as his successor Roman Abramovich at Chelsea, it's possible that Mauricio Pochettino
becomes available should Manchester United look to move on from Erik ten Hag. But would he really be a sellable hire, having just failed in London after bombing
at PSG and departing Tottenham in acrimonious manner? It seems unlikely that United would be interested in someone who demands significant club control and will
have just been fired from a struggling league rival.
</item 73>

<item 74>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Chelsea Women had created a frenzy. On Nov. 4, the English club released a statement saying their coach Emma Hayes was leaving at the end of the season to
“pursue a new opportunity outside of the Women’s Super League and club football.” Hayes had just entered her second decade in charge of the club, and few knew
where she would land next.

Advertisement

That same day The Athletic, among others, reported that Hayes’ next job would be with the U.S. women’s national team, leading a four-time World Cup and Olympic
gold medal-winning program into a new era. On Tuesday, U.S. Soccer made her appointment official.

Hayes, who previously won six WSL titles in England, will become the 10th full-time coach of the U.S., but not until her final season with Chelsea is complete
next May.
</item 74>

<item 75>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
Speaking to BBC post-match, Tottenham manager, Ange Postecoglou said: “There was a fair bit going on, a lot of VAR intervention, a hard game for me to analyse
but I am very proud of the players. Right to the end they were trying to create something for us.

“The players couldn’t give any more than they did, part of this game is will, effort and determination and they showed that.”

Tottenham now sit second in the table, whilst Chelsea’s win lifted them up to tenth on goal difference from fellow London rivals, Crystal Palace.

Embed from Getty Images

Having slipped to second in the table over the weekend, Tottenham started with high energy and fluidity which saw Pape Matar Sarr play Son Heung-min through in
the fifth minute, but the South Korea international saw his shot saved by Robert Sanchez.
</item 75>

<item 76>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
Palmer eventually stepped up in the 35th minute and Vicario got a touch on the Englishman’s penalty, yet could only deflect it onto the right post, which saw
the ball ricochet into the bottom-right corner to pull the West London visitors level.

Chelsea quickly pushed on after their equaliser against an imploding Spurs defence but failed to find a breakthrough goal before half-time, whilst the hosts
lost Romero’s central defensive teammate – Mickey van de Ven to injury alongside James Maddison in two huge blows.

Tottenham however were lucky to not be down to nine men at the break after Sarr and Colwill came to blows in the tenth minute of injury time, yet Oliver decided
to just yellow card both players than send either player off.

Reece James was then lucky not to be dismissed for Chelsea in the 11th minute of injury-time after he elbowed Udogie in the face, yet VAR deemed it a simple
incident and not a yellow or red card offence.
</item 76>

<item 77>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
That is an understatement. And if there seemed a risk involved in swapping the stability of City for the seeming madhouse of Stamford Bridge, there was evidence
of more cool-headed thinking from a 21-year-old. Chelsea have been manic buyers but he said: “I did look at how many players were there but I don’t think people
realise how many players they actually got rid off, as well.”

The departures of, among others, Mason Mount, Kai Havertz, Christian Pulisic and Joao Felix, meant there could be opportunities in the attacking midfield
positions. “I looked at the squad and thought: ‘If I go there and know what I can do then I’ll have the chance of playing’,” he said. “And thankfully it’s
happened.” If there seemed plenty of traffic in the queue for places in Southgate’s attack, Palmer seems to have accelerated past several rivals. Now an England
debut could follow in the next week.
</item 77>

<item 78>
City old boy has final say for Chelsea in eight-goal thriller, supreme Salah leads Liverpool surge (The Roar | Sports Writers Blog, 2023-11-12T20:48:50+00:00)
Sheffield United clawed their way off the bottom of the table with their 1-1 draw at Brighton, while Tomas Soucek grabbed a late winner as West Ham came from
behind at the London Stadium to beat Nottingham Forest 3-2.
</item 78>

<item 79>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
It is a global context." Does he still feel the same way?

Graham Potter

While Potter's time at Chelsea crashed and burned, many still rate the former Brighton & Hove Albion boss highly and he could be in line for a project job such
as this. Indeed, there have previously been rumours that Sir Jim Ratcliffe, the INEOS founder on the brink of purchasing a 25 per cent stake in United, would
like Potter at the club. He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with
the London club went so poorly.

Roberto De Zerbi

One of the fastest rising managers in the game, Brighton manager Roberto De Zerbi is known for his tactical innovation in transitional play, and his name has
cropped up a number of times when big clubs have an opening.
</item 79>

<item 80>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
after newsletter promotion

And clarity is perhaps what this team has been lacking most of late: clarity of selection, clarity of formation, clarity of mission and motivation. How much
prestige should we really attach to the Nations League, anyway? Is it primarily an Olympic qualification shootout or something worth winning in its own right?
England ended up trying to answer these questions on the hoof, and somehow it was only when their options had finally clarified down to one – survival or
oblivion – that they recovered their poise and verve.

This entire post-pandemic international cycle – a Euros, a World Cup, an Olympics and another Euros squeezed into three years – feels like a kind of sadism
dressed up as a panorama of opportunity. There is a school of thought out there that a fallow summer might not have been the worst thing for this team: time to
reflect and process, time to breathe, time to refresh.

Instead, they now travel to Glasgow on Tuesday for one last push. The minds are still frazzled.
</item 80>

<item 81>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
A representative from her camp declined to comment for this story.

Crocker said in September that the final interviews would include lengthy technical and tactical assessments, as well as questions to determine the candidates’
cultural fit. He and the federation stayed fairly consistent on their desired start date since the head coach role opened in August, but that became one of the
major concessions made by U.S. Soccer in selecting Hayes.

She’s headed Stateside 🇺🇸 U.S. Soccer has appointed Emma Hayes as the 10th full-time head coach in #USWNT history » https://t.co/sR7qfSnWVx Welcome, Emma!
pic.twitter.com/WYn6Sg9RmX — U.S. Women's National Soccer Team (@USWNT) November 14, 2023

Hayes will remain exclusively with Chelsea through the end of their WSL campaign and the Champions League season. She will not work with the U.S. in
international windows.
</item 81>

<item 82>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
United have now won their past 19 Premier League games at Old Trafford when scoring first, including all 16 under their Dutch manager.

MORE: Every game on Amazon in December and how to watch

Ten Hag: Crisis? What crisis?

The United manager faced a demanding week amid reports indicating disunity within his team and his players did exceptionally well to block out the noise and
focus on footballing matters.

When asked about a potential crisis, he said: “Crisis? Not for us. We keep calm. We know where we have to build. We are going in the right direction. This team
is capable to be successful.”

And United were successful tonight, thanks to McTominay. Ten Hag explained how his midfielder has managed to become more involved in an attacking sense this
season: "He has a very good smell when to arrive and a very good finish.

"It is not a frustration, in the end, we won but we could have made our lives much easier," he said.
</item 82>

<item 83>
Manchester divided as ten Hag pressure mounts after United Hammered, Ange-ball pays off again as Spurs sink Everton (The Roar | Sports Writers Blog,
2023-12-23T21:19:19+00:00)
Goals from West Ham’s Jarrod Bowen and Mohammed Kudus extended Manchester United’s miserable run in the English Premier League.

Bowen bundled home at the second attempt in the 72nd minute, his 11th league goal this season, and Kudus struck his fourth goal in as many games to settle
matters.

It was another worrying setback for manager Erik ten Hag as United suffered a 13th defeat in 26 games in all competitions.

Facebook Twitter Whatsapp Reddit Email Share

A defensive howler from Manchester United and Mohammed Kudus capitalises! ????



What a finish from the West Ham man, but Kobbie Mainoo won't want to see this one back… ????



Stream LIVE, or on-demand after, here ????
</item 83>

<item 84>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Only leaving the side to reach the pinnacle of his job, finally stepping up from a head of recruitment to become sporting director at Monaco, Mitchell showed
just how big picture he is with an interview in 2021.

“I want to continue to drive standards in each and every department,” he said. “Psychology, sports science, nutrition, medicine, the way we travel, the way we
refuel.

“Everything, every element I think we need to keep pushing to its maximum to try and be the best we possibly can.”

Having worked in football management for over a decade, Mitchell’s career has always been about the long term for his club’s, with the young signings he’s made
often only paying off after he’s left.

One in particular was 2022/23 Premier League top scorer Son, who perfectly sums up his philosophy.

“I took quite a lot of criticism in my first year at Tottenham for some of the decisions and Sonny was kind of at the centre of that,” he explained.
</item 84>

<item 85>
Tottenham’s Unbeaten Run Ends With 4-1 Home Loss to Chelsea (Sport Grill, 2023-11-06T22:18:35+00:00)
The 22 year-old first turned in Connor Gallagaher’s 94th minute pass followed by a clinical strike from Palmer’s pass in the 97th minute, which secured Jackson
his first Premier League hat-trick of his career.

Postecoglou meanwhile suffered a first home defeat as a manager since he began his 52-match unbeaten home run in November 2020 during his time at Yokohama F.
Marinos, then two-year spell at Celtic plus now Tottenham.

What’s Next?

Tottenham continue their push for the Premier League title with a trip to Wolves on Saturday 11 November, with kick-off at Molineux set for 12:30pm GMT live on
TNT Sport 1 in the UK.

Chelsea host Spurs’ title rivals, Manchester City on Sunday 12 November at 4:30pm at Stamford Bridge, live on Sky Sports in the UK.

Team News

Tottenham: Guglielmo Vicario, Pedro Porro, Cristian Romeero,
</item 85>

<item 86>
Ten Hag’s Manchester United stand up and die at club stuck in the past tense | Barney Ronay (The Guardian, 2023-12-12T23:53:03+00:00)
Finally you get to manage the everyday metrics on which you will be judged; team, results, style, energy, messaging. Win things, improve players, create a
coherent and happy internal culture: but do it during an unceasing takeover circus while star players sulk and brief against you, while the ghost of Hamlet’s
father still skulks in the eaves. This felt like the end of one thing, perhaps even the beginning of the end of something else.
</item 86>

<item 87>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
Imagine what Manchester United could do if Erik ten Hag hadn’t already lost the dressing room. Yes, it’s been another of those weeks at Old Trafford: rancour,
rumour and recrimination, barbs in the press and barbs at the press, defeat at Newcastle followed by this stirring recovery against Chelsea, and the startling
realisation that United are now three points behind Manchester City. It’s still only Thursday, by the way.

Perhaps ultimately this win only buys Ten Hag a few days’ grace, a warm flume of goodwill that lasts only as long as it takes for Dominic Solanke to run through
on goal for Bournemouth on Saturday afternoon. Back in the grip of crisis, back in the now-familiar lexicon of surly unsourced stories about dressing room
discontent. Such is the way of things at football’s most reliable content provider. This beast must always eat.

But block out the noise for just a moment, as of course United cannot, and something interesting is happening here.
</item 87>

<item 88>
Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew (The Guardian, 2023-12-06T23:45:56+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

On another night, perhaps Chelsea score from one of these counterattacks. Perhaps in another game United come up against a better finisher than Nicolas Jackson,
who for all his talent looks like a man playing football in a hall of mirrors. This is the high-wire act that Ten Hag is enacting: a daring gamble that United
will gain more going forward than they lose going back.

That once Lisandro Martínez and Raphaël Varane come back, once he brings the defensive line up 10 yards, once Rasmus Højlund hits some form, once Mason Mount
gets up to speed, lawless abandon will begin to resemble a sustainable attacking blueprint.
</item 88>

<item 89>
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
</item 89>

<item 90>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
This began the race to confirm Hayes had been selected as the USWNT head coach.

Talks between U.S. Soccer and Hayes’ representatives continued even after Chelsea’s press release. The federation’s board convened late on Saturday, Nov. 4 to
approve the selection, even without the final details of the contract settled or signed.

At the end of it all, the sides have agreed to a deal that will make Hayes the highest-paid women’s football coach in the world — though her salary is not tied
to equal compensation with Berhalter. While her salary is in the same range as the USMNT head coach, it’s thought to be a reflection of the market value for
Hayes. With reports that Chelsea was prepared to quadruple her salary to keep her, Hayes herself danced around the details in her first media availability with
the club.

“I believe in private conversations,” she said. “Of course, I’m disappointed to hear things being said in the press.
</item 90>

<item 91>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
James Ward-Prowse puts West Ham ahead against Tottenham ????



Lyenoma Udogie’s back pass to Guglielmo Vicario is short of the mark and Ward-Prowse seizes the moment.



Stadium silenced.



LIVE | https://t.co/0TogbUnBLK#PL #OptusSport pic.twitter.com/7h3q96W086 — Optus Sport (@OptusSport) December 7, 2023

On Optus Sport, ex-Spurs boss Tim Sherwood tore strips off Spurs.

“They’re quietly having a bad run, it’s five without a win now,” he said.

“They’ve led in every game and I’ve said when they take the lead, I wouldn’t say the game is over. It’s not good enough.

“But they’ve changed. The last three or four years they’ve been watching some dross there but it’s better to watch. It’s game management.
</item 91>

<item 92>
Man United next manager: Assessing candidates to replace Erik ten Hag if Red Devils sack Dutch boss (Sporting News, 2023-12-13T20:52:15+00:00)
In the event he does return quickly, he's been heavily linked to Real Madrid as a possible replacement for the Carlo Ancelotti, who is expected to become the
next permanent Brazil head coach.

It's hard to imagine that Scaloni, who seemingly craves stability and routine, has no experience at club level, and lacks ties to the Premier League or Old
Trafford (beyond a brief loan spell with West Ham as a player in 2006) would be a match for Manchester United in any capacity.

Gareth Southgate

You can sense a trend here as another currently employed national-team manager is mentioned in England boss Southgate. While he's been linked to the United job
before, it would take a perfect storm for him to be installed, whether on a permanent basis or as an interim stopgap option.

Some believe he could be considered next summer once the Euros are over: it has been speculated that Southgate could leave his England post following next
summer's tournament, with his contract expiring in December next year.
</item 92>

<item 93>
Ten Hag demands both positivity and pragmatism from erratic Onana (The Guardian, 2023-10-28T21:30:25+00:00)
“Bruno is an [effective] player when we have a lot of the ball. Then he is at his best. And there are games against top teams when you have less of the ball. So
then it is about taking benefit from [any attacking] moments and for the rest you have to work hard for the team. And to get a result because that is the most
important, that the team is winning.”

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Newsletters may contain
info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google reCaptcha to protect our
website and the Google Privacy Policy and Terms of Service apply.
</item 93>

<item 94>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
James had a hand in all three goals, and as Toone ghosted in at the far post to provide the final flourish, Wembley screeched out one last call to arms, one
last push, one last effort.

For all this, it is probably worth pointing out that for large parts of the game England were not very good. And we were reminded that for all their creativity
and verve going forward perhaps the biggest flaw in this side is a lack of presence in the two penalty areas, the physicality and the alpha energy to do the
grunt work, to convert the crosses and to head them away.

Jess Carter had a poor game and was withdrawn for Esme Morgan after half-time, Alex Greenwood over-committed for the first Dutch goal and even the
all-conquering Mary Earps plummeted to Earth with a poor first-half error.

But beyond personnel, a basic brittleness seems to have crept into this team, a sense of melting surfaces and cracks in the ice.
</item 94>

<item 95>
Man United vs Chelsea final score, result and highlights as Scott McTominay brace gives Ten Hag vital win (Sporting News, 2023-12-06T23:27:00+00:00)
Ten Hag has just told MUTV that Chelsea do not "like to defend" and that United must take advantage of that this evening.

45 mins until kickoff: Raheem Sterling has failed to score in any of his 24 matches against Man United. Could tonight be the ex-Liverpool and Man City player's
night?

1 hour until kickoff: Team news is in! The eyebrow-raising decision from Erik ten Hag is that Marcus Rashford starts from the bench. Mauricio Pochettino starts
the former Man City man, Cole Palmer.

1 hour 30 mins until kickoff: Chelsea have won three of their last four Premier League away games but are without a league win at Old Trafford since 2013...

2 hours until kickoff: It's been a relatively busy start to the week off the pitch for Erik ten Hag, it's time to see what happens on it.

"I had to set some standards"
</item 95>

<item 96>
Ange aghast as Spurs suffer ‘unbelievable’ West Ham turnaround – with two of the worst goals you’ll see (The Roar | Sports Writers Blog,
2023-12-07T22:28:31+00:00)
Kieran Trippier makes the mistake and the Everton man makes him pay.



It’s a massive goal for the Toffees and Goodison Park goes wild.



LIVE | https://t.co/W8Ud2jDSwD#PL #OptusSport pic.twitter.com/RVSGfbKcKc — Optus Sport (@OptusSport) December 7, 2023

In the other match, Everton trounced Newcastle 3-0 to move clear of the relegation zone despite their ten point penalty for financial irregularities.

They left it late, with goals from Dwight McNeil, Abdoulaye Doucoure and Brazilian striker Beto enough to down the Magpies.

“Without those 10 points, we’d be in an amazing position,” said Toffees boss Sean Dyche. “20 points would be amazing relatively to the last two seasons here.

“We’ll see what the appeal brings.
</item 96>

<item 97>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
England and Chelsea captain Millie Bright was “devastated,” and most players were understandably sad — many of whom owe their career progression to Hayes — but
know they still have a job to do this season.

“It’s been a bit of a whirlwind. As a player and a person, I was devastated. I’ve been here nine years under Emma and the things I’ve learned,” Bright said.
“She’s a mentor, a coach, a friend, a life coach. It’s more than just football playing under her.”

Bright credits Hayes for her on and off-field guidance. (Photo by Marc Atkins, Getty Images)

Hayes’ American assistant Denise Reddy, born in New Jersey, is likely to follow her across the pond. The former United States Under-20 coach has remained
faithful to her friend of 20 years and voluntarily quit her job as assistant at Chicago Red Stars in 2010 when Hayes was fired as head coach. Chelsea’s general
manager Paul Green will stay at the club.
</item 97>

<item 98>
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
</item 98>

<item 99>
Everton players and fans will be fired up by points penalty, asserts Sean Dyche (The Guardian, 2023-11-24T22:30:32+00:00)
I think they can sense it again. That’s what I think. I think they sense we are getting back to an earthiness, a determined Everton group of players who will
really have a go.”

Dyche declined to comment on the implications of Everton’s punishment for Manchester City, who have been charged with 115 financial breaches by the Premier
League, and deny any wrongdoing, and Chelsea, who are also under investigation by the football authorities.
</item 99>

<item 100>
From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray (The Guardian, 2023-12-25T10:00:20+00:00)
No family member has given a press conference or interview during their 18-year tenure. Ratcliffe has arrived on a ticket of, partly, being the local lad made
good (Failsworth, where he was born, is next to Newton Heath, United’s birthplace) so if he proves as silent as the Americans then this will be: a) poor PR; and
b) just plain poor. Football fans can be classed as the most loyal citizens of any society but even the faith of United enthusiasts has been stretched by the
cold relationship created by the Glazers’ stance regarding communication. A tap-in, then, for Ratcliffe is to remedy this by opening and maintaining a regular
line into supporters.

The Glazer family have little relationship with the Manchester United fans – something Ratcliffe would do well to avoid.
</item 100>
=== END ITEM id=9f1bdb31c9640ac3 ===

=== ITEM id=0bab42eff071c501 ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 0.1: Caroline Ellison was the CEO of Alameda.
  Fact 0.2: Bankman-Fried used Caroline Ellison as a front.
  Fact 0.3: Bankman-Fried had 'secret access' to customer money.
  Fact 0.4: The customer money included both cash and crypto.
  Fact 0.5: The government claimed this.
Sentence 1: He was concerned because it "seemed like a lot of money" from FTX customers was at risk.
  Fact 1.0: The person was concerned about the risk.
  Fact 1.1: The person was concerned because the money seemed like a lot of money.
  Fact 1.2: The money was from FTX customers.
  Fact 1.3: The money from FTX customers was at risk.

Retrieved items:
<item 1>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Ensure the site has a responsible gaming section.

The overall user experience should be smooth and hassle-free.

Safety and Security:

Top NBA betting sites prioritize users' safety.

Ensure your personal information is secure.

Some sportsbooks excel in ensuring cybersecurity; always go for those with solid reputations.

How to find the best NBA lines

Not all sportsbook betting odds are created equal. If you're willing to put in some legwork, you can shop for the best NBA lines to maximize your potential
return.

Movement in the odds depends on multiple factors, including team news, injuries, public betting, recent play, and even the number of games a team plays in a
row.

Oddsmakers use all the available information to craft their lines, and you can find value in shopping around for the most favorable lines and odds.

You can use our NBA odds page (see link above) to find the best lines for your prospective bet, or you can open accounts at multiple sportsbooks and do the
shopping yourself.
</item 1>

<item 2>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 2>

<item 3>
Moneyline Betting - What is the Moneyline and How Does it Work? (Sporting News, 2023-11-01T21:25:48+00:00)
New Information: Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an
event. When such information becomes public, sportsbooks may adjust the odds accordingly.

Changes in factors like player injuries, player substitutions, or even changes in weather conditions can affect the perceived outcome of an event. When such
information becomes public, sportsbooks may adjust the odds accordingly. Professional Bettors: Large wagers from sharp bettors or professional gamblers can
cause the lines to shift. These individuals often have insights or information that can influence the odds.

Large wagers from sharp bettors or professional gamblers can cause the lines to shift. These individuals often have insights or information that can influence
the odds. Initial Miscalculation: Sometimes, the initial odds set by oddsmakers may be perceived as offering value to bettors. In such cases, heavy betting on
one side might indicate that the initial odds were potentially "off," prompting an adjustment.
</item 3>

<item 4>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Either way, the line shopping strategy is designed to help you maximize your potential return.

NBA live betting

Thanks to the increased popularity of online sports betting apps, in-game betting has exploded nationwide. Also known as live betting, in-game betting allows
you to bet on a game while it’s unfolding.

The action is fast, and the odds are constantly shifting, making for a new way to enjoy the flow of a game while still staying engaged as a bettor.

Depending on the sportsbook, there is usually a strong variety of live betting opportunities. You’ll find game and player prop bets like whether the next bucket
will be worth one, two, or three points.

You can bet on the team leading at the end of the current quarter or on whether there will be a comeback victory. You can even boil live betting down to the
results of the next possession.

The most popular sportsbook apps, such as DraftKings, FanDuel, and BetMGM, offer NBA in-game betting, though each has its lines and betting opportunities.
</item 4>

<item 5>
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
</item 5>

<item 6>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
That means checking the posted lines and odds for the same bets at different sportsbooks.

You can do this for moneylines, point spreads, totals bets, prop bets, and more. Finding the best line or better odds on any bet is like finding money.

You can shop around using multiple sportsbook apps on your phone or live odds feeds like the one above. Pick a side, then use the drop-down menus and tabs to
find the odds and lines for that bet.

Finally, bet when you see an opportunity at better odds or an improved line compared to the competition.

Line shopping for futures

People tend to bet futures at their favorite online sportsbook. However, shopping for better odds on any future bet is worth your time. Most game lines are
similar at different sportsbooks, but the odds in futures markets can vary significantly.

Sportsbooks sometimes have higher liabilities for specific teams and offer better odds on others to lay that off. Timing is everything.

Get it right, and you can easily take advantage of changes in different futures markets due to liability.
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
</item 8>

<item 9>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Favorites to win the game are shown negative odds, such as -135, while the underdogs will have positive odds, like +165.

NBA over/under bets: Also known as totals. These bets are focused on the combined point total of the two teams playing in a game. (Winners and losers don’t
matter.) For example, the over/under could be set at 211.5 points in the Knicks vs. Heat matchup. If you were to pick the over on the bet, you’d need the two
teams to combine for 212 points or more to win. They need to score 211 points or fewer if you went with the under.

NBA props and futures

NBA player and team props are growing in popularity. Sportsbooks are taking note and providing more NBA props than ever.

After all, it’s much easier to focus on a single player and how he will perform than the entire team. Player props could focus on how many points a player
scores or how many rebounds he pulls down.
</item 9>

<item 10>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Player props could focus on how many points a player scores or how many rebounds he pulls down. These are typically provided in an over/under format.

Then, team props can range from which team will score the first bucket to the total points scored in a quarter to which team will lead at halftime. The
selection is often robust, especially when high-profile teams play.

Futures bets are exactly what their name implies: bets on events that will happen in the future. Common futures markets include:

and much more. You can usually bet on NBA futures odds throughout the year, including well before the season starts and throughout the regular season. The
further away from the event, the more likely you will find heavier odds in your favor.

Is NBA betting legal in the U.S.?

In the US, the landscape of sports betting, specifically NBA betting, has evolved considerably. While NBA betting is permitted, its legality varies across the
nation. The significant turning point came in 2018 when the US Supreme Court made a pivotal decision.
</item 10>
=== END ITEM id=0bab42eff071c501 ===

=== ITEM id=9ac93fe189c81c4a ===
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
<item 1>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Johnson, however, noted that ketamine is a "moderately addictive substance."

He told Fox News Digital, "Even long-term users typically only develop a behavioral or psychological dependence. Physical dependence can occur, but usually the
withdrawal signs are mild to moderate."

Dr. Sandhya Prashad, president of the American Society Of Ketamine Physicians, Psychotherapists and Practitioners (ASKP3) in Florida, said there is "an
extremely low risk of addiction, craving or withdrawal" when ketamine is used in low doses in a monitored clinical setting.

"Abuse of ketamine occurs when it is self-administered in high or frequent doses without supervision or monitoring, which leads to addiction ," she told Fox
News Digital.

Myth No 3: Ketamine can be self-prescribed and purchased online

Ketamine should only be obtained and administered by a licensed and experienced medical doctor, Johnson said.
</item 1>

<item 2>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Myth No. 1: Ketamine is a new treatment

In reality, the drug has been used for depression, post-traumatic stress disorder and pain for more than 10 years, according to Dr. Patrick Sullivan, medical
director of Initia Nova Medical Solutions in New Jersey .

The FDA approved ketamine for anesthesia in both humans and animals since 1970, the doctor noted.

After successful clinical trials , the FDA approved the drug in 2019 as a medication for treatment-resistant depression.

OZEMPIC AND WEGOVY WEIGHT LOSS DRUGS COULD HELP REDUCE ALCOHOL USE DISORDER SYMPTOMS, STUDY SUGGESTS

"There are hundreds of studies showing it to be safe and effective in the outpatient setting," Sullivan told Fox News Digital.

Myth No. 2: Ketamine is powerfully addictive

Ketamine is "legally recognized as having the potential for abuse and for both psychological and physical dependence," according to the American Addiction
Centers website.
</item 2>

<item 3>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
"First, if a patient has access to use it at home for pain or mood, this can increase the likelihood of developing a physical or psychological dependence,"
Sullivan told Fox News Digital.

Experts recommend that home use of ketamine is only prescribed for select patients — who should be closely monitored with regular face-to-face office visits.

"Another way it can become dangerous is if a patient uses too large of a dose in an unsupervised setting, where they may put themselves at risk for accidents,
such as falls or drownings," said Sullivan.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

Mixing ketamine with other medicines can also be dangerous, Johnson warned.

For example, when mixed with opiates, it can have a stronger sedative effect, and combining ketamine with stimulants can worsen cardiovascular effects, he said.

In particular, ketamine mixed with buprenorphine can enhance sedation and lead to unresponsiveness, said Johnson.
</item 3>

<item 4>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
They would follow each other in groups and ceased going round in circles. There were some other behaviors, such as attention to a specific task and sleeping,
that were unaffected, but overall the results were promising and according to Yoshizawa, suggest dopamine could be key to how the diet affects behavior.

“We suspect that due to this behavior variation, probably the target of the ketogenic diet is the dopaminergic system,” he said. The challenge now is teasing
out all the complex factors that may be involved.

Diet, environment, and genes – a complex recipe

According to Yoshizawa, there are two plausible ideas as to how the ketones produced by a ketogenic diet are acting on behavior. The first involves the
mitochondria, which use either carbs or fat to produce energy in our cells, and the other involves epigenetics, which simple refers to any non-genetic influence
which turns genes on and off.

Ketones are known to create detectable increases in gene expression in cells.
</item 4>

<item 5>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
ULTRAPROCESSED FOODS WITH ‘FEEL-GOOD CHEMICALS’ COULD BE AS ADDICTIVE AS CIGARETTES AND DRUGS, STUDY SUGGESTS

"These can produce strong cardiac effects of a very high blood pressure and pulse, and can lead to serious medical complications," he said.

Myth No. 4: Intranasal or micro-dosing is safe for recreational use

While using small doses of ketamine may seem safer than the larger doses given via IV, Johnson warned that when this drug is self-administered, there is a
danger of exceeding safe limits.

"The total cumulative dose could end up being higher than a standard dose of a medically administered intravenous dose," he told Fox News Digital.

Intranasal use can also lead to "more aggressive drug-seeking" behaviors, Johnson warned, as the "high" of taking the ketamine is paired with cues in the user’s
environment — which are not present in a medical office.
</item 5>

<item 6>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
After Matthew Perry’s cause of death was listed as "acute effects of ketamine" on Friday, experts are speaking out about misconceptions surrounding the drug.

(Other conditions that contributed to Perry's death, according to the autopsy report, included "coronary artery disease [and] buprenorphrine effects." Also,
"prescription medications and loose pills" were found at the residence, the report said.)

Ketamine is primarily used as an anesthetic during surgical procedures — but in recent years it's been used as a remedy for treatment-resistant depression .

WHAT IS KETAMINE, THE DRUG THAT KILLED MATTHEW PERRY ON OCTOBER 28?

The Alcohol and Drug Foundation (ADF) describes ketamine as a "dissociative drug," which means it causes people to feel "separated or detached" from their
bodies or physical surroundings.

While it is also used illegally as a recreational drug , experts say ketamine is generally not dangerous when used as prescribed.
</item 6>

<item 7>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
"To use ketamine, it is important to establish a clear diagnosis of severe depression or anxiety and/or post-traumatic stress disorder," he told Fox News
Digital. "This requires an expert and cannot be self-diagnosed properly."

"Abuse of ketamine occurs when it is self-administered in high or frequent doses without supervision or monitoring, which leads to addiction."

Those who think they may benefit from ketamine treatment should consult a doctor, Johnson advised.

"Ketamine is generally safe when administered by trained health professionals in a medical setting," Johnson said.

"In those settings, it is best delivered by the IV route, where the dose can be calculated accurately."

When purchasing ketamine outside a doctor’s prescription, there is also the risk of receiving a tainted product, experts warned. While people might assume
they're purchasing straight ketamine, the drug is often mixed with stimulants like cocaine or phencyclidine (PCP), Johnson noted.
</item 7>

<item 8>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Ketones are known to create detectable increases in gene expression in cells. Pulling apart exactly how things like, diet, environment, genes and
neurotransmitters are linked is incredibly difficult but could reveal which pathways are best to target for autism treatments or could identify a specific
ketone which works more efficiently than others.

While the cavefish are not an exact model for human autism this work shows how these experiments shine light on the biology and evolution of these behavioral
conditions and offer new directions for researchers to follow.

Reference: Motoko Iwashita, et al. Metabolic shift toward ketosis in asocial cavefish increases social-like affinity, BMC Biology (2023), DOI:
10.1186/s12915-023-01725-9

Feature image credit: Masato Yoshizawa

Correction: This article was edited on November 2, 2023. Ryan Lee was originally described as working at the University of Hawai‘i Mānoa.
</item 8>

<item 9>
5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors (FOX News - Health, 2023-12-19T01:50:33+00:00)
Buprenorphine, a drug that is prescribed to treat opioid use disorder , was in Perry’s system at the time of his death.

Guidelines to be published soon

ASKP3 announced that it will soon publish guidelines for safe use of the drug.

"In the wake of Matthew Perry’s autopsy report, we are committed to creating and publishing guidelines for at-home ketamine use, which should only be practiced
with a very specific patient profile and only in the context of a close relationship with a physician and in-office follow-up," the organization said in a
released statement.

CLICK HERE TO GET THE FOX NEWS APP

It is important for people to recognize the distinction between different uses of the drug, according to Prashad.

"In reality, the ways in which ketamine is used for anesthesia versus depression versus abuse are all very different in terms of dosage and frequency."
</item 9>

<item 10>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
When thinking about treatments for conditions like autism spectrum disorder, diet is likely not high on the list. But clues from an eyeless species of
cave-dwelling fish point to a link between a trendy weight loss diet, the brain, social behavior, and autism.

The ketogenic diet is designed to induce a physiological state called ketosis. This happens when the body no longer has enough carbohydrates to burn for energy
and switches to fat as an energy source. Breaking down fat for energy produces a group of molecules called ketones, which travel throughout the body.

High in protein and fat but low in carbs, the ketogenic diet is popular for its weight loss effects but also appears to have some neuroprotective and
anti-inflammatory effects too. Children with epilepsy are sometimes placed on this diet because it helps reduce the frequency of seizures.

The exact mechanism behind these neurological effects is still a mystery and there have been reports of long-term side effects in people trying the diet outside
of clinical settings.
</item 10>
=== END ITEM id=9ac93fe189c81c4a ===
