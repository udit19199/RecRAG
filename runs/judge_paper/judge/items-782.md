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

This file has 2 items.

=== ITEM id=8189eedecf94475b ===
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
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.
</item 1>

<item 2>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Dest was lively up front as he combined with Christian Pulisic and worked very well on the ball. He got Mats Hummels booked on a counter-attack and delivered a
few very quality crosses.

Dest was also guilty of some of the game's worst defensive errors. He was completely non-existent as Leroy Sane blazed by him on the first Germany goal, and he
nearly got his team in trouble again right before halftime by holding onto the ball too long after making an interception rather than just clearing it out of
danger. Filling in for Antonee Robinson at left-back rather than his natural right, Germany targeted him on multiple occasions, to great profit.

He had a few good moments guarding Leroy Sane in the second half, but he was a top culprit for Germany's second goal by playing Niklas Fullkrug onside. If the
U.S.
</item 2>

<item 3>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 3>

<item 4>
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
</item 4>

<item 5>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
So many national team fans sided with Reyna during the controversy that developed as the team competed at last winter's World Cup. They got what they argued for
in this game, but not what they wished.
</item 5>

<item 6>
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
</item 6>

<item 7>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
While Christian Pulisic starred on the left, Tim Weah was equally devastating down the right, dazzling with electric pace and good link-up.

Weah's biggest moment came when he scorched Robin Gosens on the wing in the first half, knocking it past the German defender and then beating him with
blistering speed to get back on the ball. Weah's statistical haul don't quite do him justice, but he was a threat throughout his 66 minutes of action.

ST: Folarin Balogun — 7

U.S. fans are salivating for Folarin Balogun to get more touches up front, and as Gregg Berhalter foreshadowed pre-match, they made a clear attempt to do that.
Gio Reyna and Tim Weah did the most of the work in that regard,

LW: Christian Pulisic — 9

There's no doubting who remains the most electric and incisive U.S.
</item 7>

<item 8>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Celtic defender had 22 touches and gave away possession just two times, completing two dribbles and making two clearances. Carter-Vickers was not a target
for Germany to attack nearly as much as the other three defenders on the pitch, a strong nod to his abilities.

65th min: Brenden Aaronson — 7

Probably the the U.S. substitute to make the biggest impact, Brenden Aaronson was bright in the attack, dazzling with his dribbling skills and helping usher the
U.S. forward amidst a second-half onslaught from Germany. A good showing from the Union Berlin winger who needs confidence at the moment.

65th min: Ricardo Pepi — 5

It wasn't an ideal situation for Ricardo Pepi to enter in, coming on with the U.S. shell-shocked from the quickfire Germany double. Still, he provided very
little up front, with just eight touches in 24 minutes of action.
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.
</item 9>

<item 10>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
“I think he played a good game, showed his quality. I think everyone in the first half played well. So it was good to see him, good to have him on the field.”

It was not good to not have him on the field, because the collapse from a 1-1 halftime tie not only followed Reyna’s departure, but probably was connected. The
U.S. was able to threaten the opposing goal enough with Reyna on the field to cause the opposition to concern itself with defending more often than it
preferred. Replacement Luca de la Torre dulled the attack and did nothing to dissuade Germany’s rampage toward the American net.

There was plenty of blame to be shared on the goals by Niclas Fullkrug in the 58th minute and Jamal Musiala three minutes later.
</item 10>

<item 11>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Sergino Dest allowed the ball to deflect off his foot and into dangerous territory at the top of the box, de la Torre became a spectator with a too-great view
and central defender Chris Richards missed two opportunities to shut down the threat. McKennie, central defender Tim Ream and right back Joe Scally all bore
some responsibility for the second.

“When you watch the goals back — I had the opportunity to do that in the coaches’ office — it’s really these fine moments, these split seconds where you need to
be well-positioned, not open up any gaps in your back line,” Berhalter said. “And that’s where it went wrong.”

With the attempt to transform Musah into some semblance of Adams, though, it almost was certain to go wrong. The USMNT conjured a more compelling attack because
of Reyna’s presence, but the ability to control the game went missing.
</item 11>

<item 12>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He was constantly unsure of where he should be as Germany built forward with incisive pace, and he was one of many U.S. players caught unaware on the first-half
Ilkay Gundogan goal.

McKennie didn't put a first-half pass wrong, going 18/18 before the break and 28/30 overall, but his positioning was all wrong out of possession. McKennie was a
sieve as the U.S. midfield was invisible defensively, leaving their defenders completely unshielded, particularly at the top of the penalty area.

It was only a matter of time before the floodgates opened, and that they did after halftime. McKennie was eventually withdrawn in what felt like one of Gregg
Berhalter's only tactical substitutions of the match.
</item 12>

<item 13>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 13>

<item 14>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
That would allow them to rotate Kamal Miller out, who picked up a yellow card in the first leg, keeping him safe for the semifinals.

In would step Alistair Johnston and Derek Cornelius, with Steven Vitoria retained alongside them.

The only injury absence for Canada at the moment is striker Lucas Cavallini, who withdrew from the squad prior to the two matches, replaced by Jacen
Russell-Rowe.

Canada projected starting lineup (3-5-2): Borjan (GK) — Cornelius, Vitoria, Johnston — Laryea, Eustaquio, Kone, Kaye, Davies — David, Larin.

Jamaica saw star striker Michail Antonio go down with a gutting ACL injury in the first leg, and he now begins a lengthy rehab process. West Ham are reportedly
furious about how it was handled, angered by both the conditions of the pitch in Jamaica and the fact he played another 10 minutes after sustaining the injury.
</item 14>

<item 15>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
CB: Chris Richards — 4

Statistically, Chris Richards had a pretty good match, missing with just one pass of his 42 attempted and not being beaten at any point by any German attackers.
Yet his lack of volume speaks... well, volumes.

Richards was off the pace for much of his 65 minutes, and failed to get involved as his teammates were beaten repeatedly on the dribble. It was Richards'
positioning that let him down, and he should have been more involved with the U.S. under siege in such an open match.

CB: Tim Ream — 5

Chris Richards often left Tim Ream on an island, and the 36-year-old just could not cope with the waves of German attackers.

The third German goal was emblematic of this lack of chemistry between the two U.S.
</item 15>

<item 16>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Antonio was replaced by Salernitana striker Trivante Stewart, but he struggled and was himself withdrawn in the 81st minute. Demarai Gray could step in, but he
himself is struggling for fitness, and while he remains with the team he was not part of the first leg matchday squad. Another option is Colorado Switchbacks
forward Romario Williams, who scored 15 goals in 29 USL Championship matches this season.

Other injuries for Jamaica to contend with include Cory Burke and Amari'i Bell, who both miss out on this international window.

Jamaica projected starting lineup (4-3-3): Blake (GK) — Leigh, Bernard, Pinnock, Lembikisa — Latibeaudiere, D. Johnson, Bailey — Reid, Nicholson, R. Williams.
</item 16>

<item 17>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
🎥: @onesoccer pic.twitter.com/XzBldBSXfX — Canada Soccer (@CanadaSoccerEN) November 17, 2023

2 hours to kickoff: While the Canadian federation is reeling from the departure of John Herdman, one player is happy to see change happen at the top. Central
midfielder Mark-Anthony Kaye had been left out of multiple squads towards the end of Herdman's tenure, and he believes his relationship with the former Canada
head coach was the reason.

Me and John had some really tough conversations where he blatantly told me I wasn’t good enough to be part of the team. That was hard for me to accept,” Kaye
said to Sportsnet, after also posting to social media an honest self-reflection following his New England Revolution's elimination from the MLS playoffs a week
ago.

Heartfelt post by Mark-Anthony Kaye.
</item 17>

<item 18>
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
</item 18>

<item 19>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 19>

<item 20>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
The decision about how to replace Adams in the formation was complex — too much so to be resolved in one afternoon.

Reyna was exceptional throughout the first half: less influential than he might have been if given greater access to the ball but evincing a welcome taste for
covering ground to become helpful on defense or to retrieve the ball and launch an attack. Weston McKennie was his typical self, particularly when he could
connect with star winger Christian Pulisic along the left side.

Young Yunus Musah was deployed in the defensive midfield role, and it did not work. Whenever Germany were able to quell an American surge in the first half and
regain possession, they plowed with ease through through the center of the park. He is elite at maintaining possession when operating as a box-to-box
midfielder, and lately his club play, with Pulisic at AC Milan, has seen Musah in more of an attacking role.
</item 20>

<item 21>
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
</item 21>

<item 22>
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
</item 22>

<item 23>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 23>

<item 24>
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
</item 24>

<item 25>
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
</item 25>

<item 26>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Sporting News reviews how each USMNT player performed across the 90 minutes.

MORE: Recap the 3-1 defeat for the USMNT against Germany in Hartford

USA player ratings vs Germany

USA starting lineup (4-3-3, right to left): 1. Turner (GK) — 19. Scally, 4. Richards (Carter-Vickers, 66'), 13. Ream, 2. Dest — 7. Reyna (de la Torre, 46'), 6.
Musah, 8. McKennie (Cardoso, 75') — 21. Weah (Aaronson, 66'), 20. Balogun (Pepi, 66'), 10. Pulisic (Paredes, 75').
</item 26>

<item 27>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The third German goal was emblematic of this lack of chemistry between the two U.S. center-backs: Richards stepped up to guard a free Niklas Fullkrug who got by
Sergino Dest, and Ream was unaware as that left Leroy Sane free on goal himself. These two just didn't have it, and Ream was powerless to pick up the slack.

Jamal Musiala dances through the USA defense and Germany lead 3-1 😮‍💨



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/PdjsjkObZR — B/R Football (@brfootball) October 14, 2023

LB: Sergino Dest — 4

If it were possible to give Sergino Dest two different scores, one for the attacking half and one for the defensive half, we'd do it, but that's not how this
works.
</item 27>

<item 28>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Even if one can argue he has the skill set to succeed at the No. 6, he did not appear comfortable transitioning on short notice.

MORE: USA vs. Germany player ratings

“Tyler’s a great player. It’s not nice to talk about guys that aren't here, though,” Berhalter told reporters afterward. “I’d rather talk about the guys that
are on the field.”

He didn’t, though, relative to how this group filled Adams’ huge vacancy. He was pleased to talk about the pressure the USMNT applied in the first half,
including a ball played over the top of Germany’s defense to Pulisic early in the game that was stopped by an offside call both player and coach disputed, and
Pulisic’s brilliant goal in the 27th minute that involved him driving to his right and blasting a curving shot while four defenders converged and ace goalkeeper
Marc-Andre Ter Stegen watched it sail into the upper righthand corner.
</item 28>

<item 29>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 29>

<item 30>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
“I will commit to being the catalyst that will set the tone for our group for every game, to help restore our club to where it deserves to be, I am confident of
that. We will be back and better.”#NERevs pic.twitter.com/A0mVyGdaAZ — Seth (@SethMan31) November 10, 2023

Jamaica vs Canada lineups & team news

Jamaica manager Hallgrimsson also described his side as being on a mission, but they faced it without New York Red Bulls forward Cory Burke because of a groin
problem.

Karoy Anderson, who plays for English club Charlton Athletic, was called up for the first time. "I was just shocked," the midfielder told the Addicks' website.
"I was looking at it for a long time just taking it in and I told my mum and she started crying."
</item 30>
=== END ITEM id=8189eedecf94475b ===

=== ITEM id=439ed55d9310e403 ===
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
Fans react to most ‘heartbreaking’ scene in The Crown (The Independent - Life and Style, 2023-11-17T09:35:32+00:00)
It is in this scene that The Crown depicts the doting mother’s final conversation with her sons.

William, then 15, and Harry, then 12, inquire over the phone when their mother will be coming back home from Paris. “I’m coming home tomorrow, and I’m seeing
you both tomorrow night,” Diana replies in the episode, which also takes place on the same day as her fatal car crash later that evening. The boys then ask
Diana whether she plans on marrying Dodi, considering their relationship has been plastered all over the tabloids.

Diana’s poignant last words to William and Harry shown in The Crown scene

“Well, you know better than to believe the papers,” she tells William, before adding: “I’m emphatically not going to marry Dodi. To be honest, I can’t wait to
come home.”

When William sweetly asks his mother if she’s doing “okay”, Diana candidly tells her eldest son: “I’m okay. It’s just a bit mad here.
</item 1>

<item 2>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
While it was lovely to be taken care of, we all tend to regress when we’re living with our parents. It’s impossible not to feel infantilised, and there’s a real
tipping point where you start to lean into learnt helplessness. I stayed for four months before I moved out to house-sit a friend’s place. Then I launched
myself into dating to try to find a man to have a family with. After the roller-coaster of dating, I decided to start fertility treatment on my own, and by the
time it began, I’d found a stable job and was renting an apartment. When I got pregnant, I was elated but also terrified about how I would raise a kid on my
own, financially and emotionally. Luckily, after I gave birth, my parents invited me to live with them, so I moved back for the first three months of my son’s
life.
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
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and
effectively. One major challenge they have faced is the problem of getting drugs to pass through the blood-brain barrier, which controls the movement of ions
and molecules from our bodies to our brains. That makes it difficult to get drugs into the central nervous system to tackle problems such as amyloid plaques.

“However, researchers are developing active transport methods for getting drugs across the blood-brain barrier,” said Mummery. “For example, we are now
exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact.”

Such developments will take years to realise, scientists caution, and a great deal needs to be done in the short term to deal with dementia.

“It is clear we have taken a step in the right direction but there is much more to do before we can have successfully dealt with these processes that are
happening to our brains,” said Thomas.
</item 4>

<item 5>
Shayda director Noora Niasari on family violence, Iranian liberation and taking her first film to the Oscars (The Guardian, 2023-10-01T14:00:47+00:00)
It’s not about us any more. That feels liberating and cathartic.”

skip past newsletter promotion Sign up to Saved for Later Free newsletter Catch up on the fun stuff with Guardian Australia's culture and lifestyle rundown of
pop culture, trends and tips Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more
information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy
. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

The world is seeing the strength of Iranian women now Noora Niasari

After screenings, adult men have approached her with their own tales of growing up in shelters; other audiences have said they left with a deeper understanding
of different Iranian women’s experiences.

It was important to Niasari to shade in light along with the darkness of family violence.
</item 5>

<item 6>
Mayo Clinic sees AI as 'transformative force' in health care, appoints Dr. Bhavik Patel as chief AI officer (FOX News - Health, 2023-10-10T09:30:18+00:00)
ARTIFICIAL INTELLIGENCE HELPS DOCTORS PREDICT PATIENTS’ RISK OF DYING, STUDY FINDS: ‘SENSE OF URGENCY’

It’s important to protect data privacy and security as well, Patel noted, due to the large amounts of information that must be fed to AI models.

While AI's capabilities are "immense and ever-growing," Patel emphasized that AI’s role is to be an ally.

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

"Humans caring for humans is the bedrock of health care — AI strengthens that foundation, but doesn’t replace it," he said.

"AI can offer insights and assist in decision-making, but human touch, intuition and empathy cannot be replicated by algorithms."

CLICK HERE TO GET THE FOX NEWS APP

While AI promises a "new frontier" in health care — Patel calls for a balance of "enthusiasm and caution."
</item 6>

<item 7>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The researchers hope that these findings will lead to human trials.

The goal is to obtain funding to start a pilot study of patients next year.

"Accessible interventions like this have great translational potential, as they are available and are usually very well-tolerated," Desplats said.

The goal is to obtain funding to start a pilot study of patients next year.

"This may have special importance for the elderly , as time-restricted eating does not require limitation in calories or change in the diet, but may offer
important benefits ranging from metabolic and sleep regulation to potential cognitive improvement."

TIME-RESTRICTED EATING NO MORE BENEFICIAL THAN CALORIC RESTRICTION IN OBESE PATIENTS, STUDY SAYS

Adopting a new dietary pattern should not be a replacement for medical care , however, as Desplats emphasized the importance of patients' discussions about
options with their clinicians.
</item 7>

<item 8>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
However, with a “traditional” approval for Eisai-partnered lecanemab for Alzheimer’s (compared to accelerated approval for adacanumab), Biogen has the first
approved treatment, putting adacanumab on the backburner.

Cognitive function improvement is a critical marker for Alzheimer’s treatment and lecanemab delivers lower degeneration compared to placebo on this count. The
treatment reduced clinical decline by 27 per cent at 18 months on clinical dementia scoring and on daily living scale delivered a 37 per cent improvement
compared to placebo. As is evident, the development is satisfactory but a lot more progress is left to be made on Alzheimer’s treatment curve.

Takeaways for Investors

Analysing pharma stocks in the US rests on three factors: drugs about to expire, drugs in ramping up stage and the pipeline.
</item 8>

<item 9>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
We figured we’d be with my in-laws for three to four weeks, but we didn’t realise how bad the rental crisis in the region is. We’ve been offering more than the
asking price and/or offering more rent upfront to secure a place, but as there’s a preference to rent to families or those who’ve rented before, we keep getting
knocked back. Our initial “quick stay” with Dane’s parents has stretched out month after month. We’re living in one of their spare rooms and have our boxes
stored in another.

I love my parents-in-law, so there have certainly been upsides. We’re spending a lot of quality time together and they’re really lovely about trying to give us
as much space as possible, often booking weekends away so that Dane and I get the house to ourselves for a couple of days.
</item 9>

<item 10>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
One major limitation of the study is that it uses animal models that only represent certain aspects of disease and "do not present comorbidities or other
clinical diseases associated with aging, which are very common in Alzheimer’s patients," noted Desplats.

"Furthermore, the study was done under strictly controlled conditions that did not resemble the variety of people’s habits and environments," she added.

THE 8 BIGGEST ALZHEIMER’S DISEASE MYTHS — AND THE TRUTHS BEHIND THEM

The researchers hope that the public recognizes the importance of the circadian clock in regulating many aspects of health .

"Sustaining healthy sleep and eating habits that are more synchronized with the natural environment are vital in a society that is constantly exposed to food,
light and shift work," said Desplats.

Previous studies have supported this association between fasting and reduced Alzheimer’s risk.
</item 10>

<item 11>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
“One way to start could be to emphasize the use of fresh, local products and ingredients in Mediterranean-style meals,” said Sotos-Prieto. “[But] changing
dietary habits and lifestyle is a gradual process, and affordability is also an important question.”

Other questions remain as to how governments should invest in introducing new healthy habits, which ones should be given preference, and how much change is it
enough to start seeing benefits.

The task is not simple since culture can define anything from ingredients and recipes to different social habits and work-life schedules. But Sotos-Prieto is
optimistic.

“I think that picking realistic goals, even if small, and sustaining them over time can make a difference [in public health],” concluded Sotos-Prieto.

Reference: Maroto-Rodriguez J. et al., Association of a Mediterranean Lifestyle With All-Cause and Cause-Specific Mortality: A Prospective Study from the UK
Biobank, Mayo Clinic proceedings (2023).
</item 11>

<item 12>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Perhaps we should think of it as self-parenting.

Good groups for elders

Age UK has a database of social activities and groups for older folk

The Walking Football Association keeps keen amateur footballers playing long into their 50s and 60s

Action for Elders offers social and exercise programmes

The Centre for Ageing Better provides information on working and living as an older person

u3a, a charity collective, provides local opportunities for people no longer in work to acquire new skills and knowledge

Annabel Streets is the co-author of The Age-Well Project: Easy Ways to a Longer, Healthier, Happier Life (Piatkus) and co-founder of The Age-Well Project
</item 12>

<item 13>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
While it was wonderful to have the emotional and practical support, after three months I felt that If I didn’t get out there to live as an independent person, I
would quickly lose any sense of what I was capable of. But I’m incredibly grateful for the time I had living with my parents; I know it’s a huge privilege to
have them as a safety net. I ended up writing a memoir, Inconceivable, about my journey to become a solo mum – and the complexities of being a woman in your
late 30s who has to start over and move back in with your parents. I think while we’re all embracing this contemporary feminist movement where women can create
families of their own making, we’re also realising that we need help, that parenthood can’t happen alone.”
</item 13>

<item 14>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
In the last few years I’ve witnessed friends and family dashing to help parents with poorly designed online banking systems, unruly fridge alarms, “upgraded”
software that has changed beyond recognition, apps and phones that don’t function as expected, and numerous other technical challenges.

Frankly, technology is often baffling to me, too. Luckily I have kids who delight in parenting me – advising, explaining and guiding me on all things digital.
But studies suggest that this helpfulness might not be as beneficial as we think. Grappling with new and baffling things forces the ageing brain to work in
novel ways. A recent study from the University of Texas involving more than 200 elderly adults found that those who engaged with technology had better cognition
and memory than their non-engaging peers.

Digital engagement – whatever the device – protects the brain

Even the stress we experience while battling with our computer can be helpful, sharpening our mental acuity. The study authors concluded that digital engagement
– whatever the device – protects the brain.
</item 14>

<item 15>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
And Sherri Nordlie Geerts was her mother.

Bazella emailed Geerts and they started corresponding regularly.

"I lost 39 years … There’s a lot of grieving with that." — Diane Bazella

"We were slowly getting to know each other through email," Bazella said. "She had a lot on her plate as a caregiver to her husband, so we were slowly
establishing our mother-daughter relationship."

A LIFE IN PICTURES: FROM BIRTH INTO ADULTHOOD, A FATHER TAKES HIS SON’S PHOTO EVERY DAY FOR DECADES

Finally, on Christmas Day 2021, Bazella and Geerts heard each other’s voices over the phone for the first time.

They met in person in July 2022.

Since then, they have been talking on the phone frequently, sometimes for hours at a time.

‘Same sense of humor’

During their conversations, Geerts opened up about having a baby when she was just 18 years old.
</item 15>

<item 16>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Both drugs are to be considered for approval in the UK next year. The Medicines & Healthcare products Regulatory Agency (MHRA) will first decide if they are
safe and effective, then the National Institute for Health and Care Excellence (Nice) will rule on whether they offer value for money.

Both medicines are expensive – lecanemab costs about $25,000 (£19,700) a year – and are given by regular intravenous infusions. “That is a challenge from a
health-service point of view, because you need to find the space and time to put someone in an infusion suite to treat them,” added Thomas.

However, the main problem facing doctors is the difficulty involved in pinpointing dementia not just in its early stages but even in its later phases. Most
cases are first presented to GPs, who will then refer patients to memory clinics for a dementia test. However there are long waiting times – up to two years on
average – for appointments at these centres.
</item 16>

<item 17>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
“If the idea of my being in that place was to heal, that was not the effect.”

She’s not scared of anything after that experience, she writes, “but it doesn’t make me feel strong; it makes me sad. I shouldn’t be this strong.”

#FreeBritney protesters in 2021. Photograph: Étienne Laurent/EPA

A nurse shows Britney the blossoming fan movement that is spreading awareness of its suspicions that all is not right with her conservatorship. Come June 2021,
she calls 911 to report her father for conservatorship abuse days before a probate court hearing on the arrangement. “My voice had been used for me and against
me so many times that I was afraid nobody would recognise it now if I spoke freely,” she writes. When she is granted her freedom in November 2021, she says she
feels shock, relief, elation, sadness and joy.
</item 17>

<item 18>
Looking for a book to read? Here are 14 tips for October (The Sydney Morning Herald, 2023-10-01T21:04:20+00:00)
His latest brings back Nell Buchanan and Ivan Lucie, whom we met in Treasure & Dirt and Tilt, for a case of murder involving one of the seven families who rule
the Yuwonderie district like a fiefdom. You can expect dark secrets to emerge and herrings to be red.

Life as We Knew it

Aisha Dow & Melissa Cunningham, Scribe, $35. October 3 This account of how the pandemic affected Australia is a timely publication coming only a few days after
Daniel Andrews, the man who approved probably the world’s longest cumulative lockdown, left office. While many people may not want to revisit those stressful
years, Aisha Dow and Melissa Cunningham, journalists with Nine newspapers, manage to make this a riveting story of discovery, dismay, and recovery. Ahead of the
government’s impending inquiry, you might find some answers here.
</item 18>

<item 19>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Being social, says Laird, is as important as trying new things: “Research shows that those who are more social are also more resilient to chronic conditions and
often have a stronger immune system.”

Anyone with older parents should be urging them out for a walk five days a week Dr Eamon Laird, health and ageing researcher

Depression among the elderly is on the rise and often goes unnoticed. Gerontologists now believe that it is one of the most common psychiatric disorders among
the elderly – with half of all cases going undiagnosed. When elderly parents become depressed (many of them come from a generation not accustomed to talking
about their mental health), it can be difficult to know how to respond. But Laird’s latest report suggests that a daily walk could help. His study of more than
4,000 older adults found that as little as 20 minutes of daily walking could cut the risk of depression by up to 43%.
</item 19>

<item 20>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
The study authors concluded that digital engagement – whatever the device – protects the brain. These findings reflect earlier studies that discovered that
older people who made active use of technology were less likely to get dementia.

Dr Eamon Laird, researcher in health and ageing at Limerick university, agrees that we should be encouraging older parents to try new things. And the further
out of their comfort zone they feel, the better. “It’s always good to keep the mind active and fresh,” he told me. “New challenges can help build and maintain
new brain connections and can be good for brain and overall health.”

If older people use technology to keep in touch with extended family and friends, that’s a double whammy. When my 84-year-old mother recently joined Instagram,
she not only expanded her grey matter but she also found an effective means of staying in touch with her grandchildren.
</item 20>

<item 21>
England choose survival over oblivion and a chance of summer trip to France | Jonathan Liew (The Guardian, 2023-12-01T22:57:40+00:00)
These are issues not so much of tactics but cohesion and energy: the timing of runs, the sharpness required not just to anticipate where the ball is going to
be, but actually get there. You can still chase when you’re running on fumes, but to create: this needs an idea and a vision, the clarity of mind to see the
picture before it has materialised.

skip past newsletter promotion Sign up to Moving the Goalposts Free weekly newsletter No topic is too small or too big for us to cover as we deliver a weekly
roundup of the wonderful world of women’s football Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside
parties. For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply.
</item 21>

<item 22>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Based on these findings, Longo recommends that people talk to their neurologist about considering fasting-mimicking diets and other nutrition-based approaches
in support of standard-of-care drugs and therapies.

"Especially for Alzheimer’s disease, for which pharmacological interventions have had very limited efficacy, it will be very important to continue to study
fasting-mimicking diets and other dietary interventions that can make drugs more effective," he added.

Desplats, lead author of the UC San Diego study, told Fox News Digital that Longo’s studies are on a similar track, demonstrating the connection between
metabolism, fasting and brain health.

"The main difference is that we tested an approach that does not require calorie reduction, which may be more sustainable and safer for older people whose
metabolisms are already impacted by aging changes," she said.

COLORADO WOMAN WITH ALZHEIMER’S IS STILL VIBRANT AFTER MORE THAN 4 YEARS: ‘I REFUSE TO SURRENDER’

Dr.
</item 22>

<item 23>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Because fasting-mimicking diets can be done in longer periods, they can help the body go into higher levels of ketosis, Antoun noted, "which is wonderful for
the brain."

CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER

(Ketosis, which is when the body burns fat for energy instead of glucose, has been shown to have cognitive benefits.)

"Prolonged fasting can also help with leaky gut, which then can aid the brain," added Antoun.

In addition, intermittent fasting can help to control blood sugar , which can slow down the progression of Alzheimer's, the doctor said.

Over the past several decades, there has been a steady increase in diagnoses of Alzheimer's disease, Antoun noted.

"However, when we are able to detect neurodegenerative conditions early, we can take positive steps that may slow the process and mitigate any major side
effects of the disorder," he said.
</item 23>

<item 24>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The goal of the restricted eating is to restore the circadian rhythm, the researchers said, which could help counteract the numerous disruptions that
Alzheimer’s patients face when it comes to sleep and schedule-related confusion.

HIDDEN BELLY FAT COULD SIGNAL ALZHEIMER’S DISEASE RISK 15 YEARS BEFORE SYMPTOMS SHOW UP, STUDY FINDS

"Our study emphasizes the power of feeding timing in aligning the circadian clock and its impact on the brain," said senior study author Paula Desplats, PhD,
professor in the Department of Neurosciences at UC San Diego School of Medicine, in a statement to Fox News Digital.

"We were hopeful to see some improvements in pathology, but did not expect such profound effects in reducing plaques and inflammation and improving memory," she
went on.

"The effects of time-restricted feeding on lower amyloid pathology were also measurable in blood using markers found in the clinic, which was another important
finding."

The researchers hope that these findings will lead to human trials.
</item 24>

<item 25>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
This comes after several failed attempts, and one controversial false start in the recent years.

Alzheimer’s disease is characterised by amyloid beta deposits on brain cells leading to shrinking and loss of cognitive abilities. Despite a clear marker of
amyloid plaques and a target area to work on, pharma companies failed to address plaque removal or a statistical improvement in cognitive functioning after
plaque removal, till the recent approval. Pfizer, Eli lilly, Merck and many others faced failure in the space. The amyloid theory itself was in question.

Biogen’s adacanumab had a peculiar start. The company itself stopped development in 2019, but it surprisingly pursued a high-dose study into 2021. Later, FDA
approved adacanumab with an accelerated approval for high-dose version, which faced sharp criticism. Three advisers resigned over the approval of the drug that
could have cost $56,000 per annum to administer.
</item 25>

<item 26>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
During prolonged periods of fasting, cells undergo a process called autophagy, which involves removing and recycling damaged cellular components, Antoun said.

CLICK HERE TO GET THE FOX NEWS APP

"This process is essential for maintaining proper cellular function and can help protect the brain from neurodegenerative diseases such as Alzheimer’s," he
explained.

More than six million Americans of all ages have Alzheimer's, and every 67 seconds, someone in America develops the disease, according to the Alzheimer’s
Association.
</item 26>

<item 27>
Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes (FOX News - Health, 2023-10-18T09:30:30+00:00)
Mackenzie Parks, a licensed clinical social worker at Children's Nebraska, said the families she encountered were "touched and flattered" that the hospital
would offer follow-up support.

"The letters I’ve received in return are really telling about the impact the program has on their well-being and hope for the future," she told Fox News
Digital.

"I also feel that it helps families know that we truly do care about them when we’re assessing them — not just because it’s our job, but because we really want
them to be safe," Parks added.

TEEN GIRLS SPEND MORE TIME ON ‘SENSITIVE’ SOCIAL MEDIA CONTENT THAT CAN HARM MENTAL HEALTH, REPORT SAYS

Eva Doescher, a licensed clinical social worker in Nebraska, shared similar feedback on her past Caring Contacts experiences.

"Caring Contacts gives me a chance to slow down and learn from the powerful, short moments that I get to connect with patients and their families," she told Fox
News Digital.
</item 27>

<item 28>
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
</item 28>

<item 29>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
It’s been proven that what people eat can help prevent or slow Alzheimer’s disease — but what about when they eat?

Participating in intermittent (time-restricted) fasting could lead to a reduced risk of cognitive deterioration, a recent study published in the journal Cell
Metabolism suggests.

Researchers at University of California San Diego School of Medicine adjusted the feeding schedule of certain groups of mice so that they only ate within
six-hour windows each day.

ALZHEIMER’S BLOOD TEST COULD HIT THE MARKET IN EARLY 2024, RESEARCHERS SAY

(This would be equivalent to 14 hours of fasting for humans, the researchers noted.)

Compared to a control group of mice that ate on demand, the fasting mice showed improvements in memory, were less hyperactive in the evenings and had fewer
sleep disruptions .

They also were shown to have less buildup of amyloid proteins in the brain, which is a hallmark of Alzheimer’s disease.
</item 29>

<item 30>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Parenting no longer starts and stops with our children. Nor is it confined to those who have children. In a time of unrelenting change and ever-extending life,
most of us will – at some stage – find ourselves “parenting” our own parents.

Indeed, many of us – particularly those who had families later – will find ourselves simultaneously parenting our kids and our parents. In one breath we’ll be
begging our children to swap French fries for vegetables, and in the next breath we’ll be urging our parents to exchange cake for sardines. Little wonder
today’s midlifers are known as the sandwich generation.

The relentless pace of technology – accelerated by the Covid-19 pandemic – has made everything infinitely more complicated, leaving many older (and not-so-old)
people baffled and frustrated.
</item 30>

<item 31>
‘Is this worth it?’: The difficult questions the Shameless founders have had to ask each other (The Age, 2023-10-07T18:00:00+00:00)
How worried are they about getting it wrong? Turns out, not very. Andrews and McDonald know what good advice looks like, and how to give it. “I think the best
advice is often not a statement – it’s a question,” says Andrews. “It’s listening to what someone’s dilemma is, then posing a question that helps them
re-frame.” As two women who are constantly being asked for advice, what are the questions they ask themselves? “What’s the next thing?” says Andrews, without
hesitation. “I’m constantly wondering, what’s the next show that isn’t out there.”

Advertisement
</item 31>

<item 32>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It is a challenge, and a very important one.”
</item 32>

<item 33>
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
</item 33>

<item 34>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
Considering the 6 million patients in AD in the US, assuming 50 per cent are mild to moderate (applicable group according to FDA label) and 15 per cent on
treatment by year 7, the drug alone can rake in $12 billion (halved to account for Eisai’s share) by 2030 in cumulative sales for a company whose market cap is
$33 billion trading at 14 times FY24 EPS.
</item 34>

<item 35>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
“Dairy, fish or non-processed meat can help maintain muscle strength and function, reducing the risk of frailty as we age.”

In truth, anyone over 50 would do well to follow these simple guidelines: engage with something new every day, take a daily walk of at least 20 minutes,
socialise regularly, take a daily multivitamin for seniors and check the protein content of our meals. Perhaps we should think of it as self-parenting.
</item 35>

<item 36>
Looking for a book to read? Here are 14 tips for October (The Sydney Morning Herald, 2023-10-01T21:04:20+00:00)
Particularly family, and particularly her mother, who took her own life at 46 and about whom Davidson has been unable previously to write to her own
satisfaction. She is acutely aware that her version of the truth “can bury another’s”, but this one is hers.

Stone Yard Devotional

Charlotte Wood, Allen & Unwin, $32.99. October 3 Another reckoning with a mother’s death, this time in a novel from the sure hands of Charlotte Wood. A woman
arrives at some sort of religious community having turned her back on her husband, their life and the metropolis. What is she fleeing and what are her demons?
It’s not all peace and tranquillity in the community, which has collective crises to deal with, while her previous life won’t leave her alone. Written in a
spare prose, the consistently brilliant Wood delivers yet again. Will Murdoch help Trump again?
</item 36>

<item 37>
Merck, Novo Nordisk, Gilead, Biogen and more: Here are the investment opportunities in global Big Pharma (Business Line, 2023-11-18T14:00:36+00:00)
Antibodies deal with viruses but fail to do so in HIV infection. By engineering bNAbs in a manner similar to mABs for cancer or vaccines (where antibody
reaction to neutered virus generates immune protection), HIV cure may be on the horizon.

Gilead’s several Phase-2 programmes with ART combination are in progress. Studies as recently as February 2023 presented proof of concept — 90 per cent
participants showed virological suppression at 26 weeks. Incidentally, Gilead’s Sovaldi and Harvoni were the first cures for Hepatitis C. The World Health
Organization (WHO) aims to eradicate the virus by 2030.

Alzheimer’s

The degenerative disease of the brain has proved to be a difficult challenge for pharmaceutical companies. Leqembi (lecanemab), developed by Eisai in
collaboration with Biogen, was the first and only approved drug for Alzheimer’s in July 2023.
</item 37>

<item 38>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“Meanwhile, we have got to get the NHS to a state where it is much more focused and better organised in getting better and earlier dementia diagnoses. That will
be crucial in our battle against the condition.”

Mummery agreed: “By the time you have dementia, you will have had Alzheimer’s disease developing in your brain for at least 20 years. Dementia is the latest
stage; we need to pick up signals much earlier. So we need to be thinking about how to diagnose people at the very earliest stages of disease, when they may
only have very, very subtle symptoms.

We are now exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact Cath Mummery

“Our current service isn’t good for doing that, and we need to develop brain health clinics where we can pinpoint the disease much earlier, then help to build
resilience against dementia in a patient when it is only in a very early stage.”
</item 38>

<item 39>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
We’ve also worked out our perfect system for sharing the load; they’ll cook dinner for half the week, for example, and we’ll cook the other half, so the
division of labour has been a positive experience. Loading The most challenging aspects? They’re retired, so they’re home a lot. And I work from home, so it’s a
little tougher for me than it is for Dane, who works in an office. I get lots of invitations from my mother-in-law to join her and her friends for lunch or
coffee when I’m supposed to be working, which is probably not the worst thing to complain about. Despite our love for his parents, the anxiety Dane and I have
been feeling has increased as time’s gone by. So much so that we’ve just agreed to move into a friend’s holiday home, sight unseen, even though we’ll have to
move back in with Dane’s parents during the times the friend wants to use it, such as at Christmas.
</item 39>

<item 40>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
DOI: 10.1016/j.mayocp.2023.05.031

Feature image credit: Brooke Lark on Unsplash
</item 40>

<item 41>
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
</item 41>

<item 42>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Previous studies have supported this association between fasting and reduced Alzheimer’s risk.

Last year, a study led by the USC Leonard Davis School of Gerontology in Los Angeles also found that mice that followed a fasting diet showed lower levels of
"plaques and tangles" in the brain, reduced brain inflammation and better performance on cognitive tests.

"Cycles of a fasting-mimicking diet (FMD) had strong effects against Alzheimer’s symptoms in two different types of Alzheimer’s mouse models," the lead author
of that study, USC biogerontologist Dr. Valter Longo, told Fox News Digital.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain."

"In mice, the effects of FMD cycles were very strong and wide, meaning they affected both learning and memory and also Alzheimer’s pathology peptides/proteins
(amyloid and tau)."
</item 42>

<item 43>
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
</item 43>

<item 44>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Her life under the conservatorship is a horror story

It feels fitting that The Woman in Me is being released a week before Halloween. No matter how much we already know about the 13 years that Britney lived under
a conservatorship, her recounting of its imposition, life under it and her inability to escape it is nothing short of a horror story. It is appalling, like
something out of a Victorian novel, not the very real experiences of a thirtysomething in the last decade.

When she fears that Federline is going to take away access to her kids, she locks herself in a bathroom with one of her sons. “Before I knew what was happening,
a Swat team in black suits burst through the bathroom door as if I’d hurt someone,” she writes, and she is subject to a forced hospital hold. Soon after, her
mother invites her to her beach house because “the cops are after you”. When she arrives, so does another Swat team.
</item 44>

<item 45>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
Start by joining Dementia Research, a nationwide register that matches volunteers to research programmes.”

Even if your parents don’t require such intensive “parenting”, a simple supplement could help. A recent study from Columbia University involving 2,200 adults
aged 60 and over found that an inexpensive multivitamin formulated specifically for older people improved cognition, memory and focus. In fact, the results were
so marked that the researchers said the supplement could slow age-related memory decline by three years.

But while most of us can probably persuade our parents to take a multivitamin, suggesting they change their diet might be more complicated. Appetites fade as we
age, chewing becomes more difficult and those living alone don’t always feel inclined to cook. The important thing, say both Saunders and Laird, is that older
people consume plenty of protein.

“Older adults should be taking at least 1 to 1.5g of protein per kilogram of body weight,” says Laird.
</item 45>

<item 46>
The best shows to stream in December (The Sydney Morning Herald, 2023-11-30T22:00:00+00:00)
With stories in The Age and Sydney Morning Herald, as well as on 60 Minutes, journalists Nick McKenzie and Chris Masters pushed to bring the truth to the
public, navigating divisions in the SAS and eventually the defamation trial that Roberts-Smith brought to court and lost. It’s been a compelling saga, and Dora
Weekley’s documentary should include hitherto unknown details and unheard commentary.

Advertisement

I still haven’t got over the nightmarish sounds of a spinal surgery going very wrong in the first season, but nonetheless Dr Death (December 22) is back for a
second serve of malpractice and monstrous medical egos. This American true crime drama is now an anthology. Every season is a different doctor whose abuse of
privilege posed a criminal risk to their patients. The new season will cover Paolo Macchiarini, an Italian surgeon and researcher who was accused of laboratory
fraud and manipulative behaviour.
</item 46>

<item 47>
Billie Jean King sees progress in her crusade for more investment women’s sports: 'We’re kind of at a tipping point' (Fortune, 2023-10-13T22:26:03+00:00)
We need to have a league where the very best players will play. We went to Toronto and I did an opening speech about trailblazers. It was amazing because the
families were crying, the players were crying, they said ‘we’ve never been treated like this, it’s amazing, we feel like pros for the first time.’ There were a
lot of little kids there. Kids are going to have an amazing opportunity that the generations before them never had. All three of their networks had it on. It’s
a religion up there.

How is investment in women’s sports changing?

I’m asking CEOs and everyone now — ‘do you invest as much in women as you do in men?’ Then it usually gets quiet. But I must say it’s better than it used to be.
We’re really lucky to be with this investment group. The male allies we’ve had through the years have made such a difference. They have the money and the power.
</item 47>

<item 48>
At 37, Alexandra found herself broke, heartbroken and living with her parents (The Age, 2023-11-25T18:00:00+00:00)
It isn’t ideal, but it’s one step back towards independence. If we’d known then what we know now, we’d never have sold our home.” “It’s a shame we don’t value
multi-generational living more than we do”

Job insecurity and financial practicality encouraged Rachel Claire, a 32-year-old photographer, to move back in with her parents in 2019. “The great thing about
living with your parents when you’re an adult yourself is that you don’t take any of it for granted.” “My parents and I have always been close. When I first
moved out of home at the age of 22, my parents would always find reasons to pop by, whether it was just a quick visit after work or to fix things or mow my
lawn. I could say they struggled with becoming empty-nesters, but I think it’s more accurate to say we’ve just always enjoyed each other’s company.
</item 48>

<item 49>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It has been a very long, hard road, but finally we have something positive to look at. That is very welcome.”

Alzheimer’s disease is triggered by the buildup of a protein called amyloid in the brain, although symptoms may not appear for decades after this accretion has
started. Scientists have tried for more than 20 years to find ways to stop amyloid from forming these plaques, in the hope this would stop the progression of
the disease.

Lecanemab, which is produced by Japanese pharmaceutical company Eisai, and donanemab, produced by Eli Lilly of the US, are the first medicines to achieve this
aim – though they only slow but do not ultimately halt the disease’s progression.

Graeme Armstrong with wife Trina, who has a rare form of Alzheimer’s.

Both drugs are to be considered for approval in the UK next year.
</item 49>

<item 50>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Joseph Antoun, CEO and chairman of the nutritech company L-Nutra in California, was also involved in the USC study.

"In these studies, mice fed the fasting-mimicking diet showed improved cognitive performance, motor learning and both short- and long-term memory, and promoted
the growth of new neurons in the brain," he told Fox News Digital via email.

"Fasting-mimicking diets have the potential to drastically reduce many aging and disease risk factors by reducing neuroinflammation or superoxide production in
the brain."

Fasting can help the brain function in a myriad of ways, said Antoun.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain," he told Fox News Digital in an emailed
statement.

Time-restricted eating can also improve the quality of sleep, he noted, which can help the brain recover better.
</item 50>
=== END ITEM id=439ed55d9310e403 ===
