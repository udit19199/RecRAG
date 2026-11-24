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

=== ITEM id=78c1f2bdcf499c61 ===
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
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
RENTSCHLER FIELD, HARTFORD — The United States appeared well positioned to fight with a European power through 45 minutes, but the house of cards came crashing
down after the break as the U.S. men's national team succumbed to a 3-1 defeat to Germany in Hartford.

Presented with a chance to test themselves against a world-class opponent on home soil, the U.S. entered the break level as a Christian Pulisic stunner was
cancelled out by Ilkay Gundogan's poaching.

Yet after halftime, as Gio Reyna was withdrawn from his first start back under Gregg Berhalter to build his fitness, the U.S. midfield was overrun and the
defense committed a multitude of errors to allow the Germans to gain full control of proceedings.

"It’s frustrating because its just little moments," said U.S. goalkeeper Matt Turner after the match. "Little moments made a big difference at the World Cup,
and it’s kinda the same story here.”
</item 1>

<item 2>
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
</item 2>

<item 3>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
UK and Australia: The match is available to stream via CONCACAF's Go app and YouTube channel.

MORE: MLS Cup playoffs schedule

Jamaica vs Canada betting odds & lines
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
</item 5>

<item 6>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Celtic defender had 22 touches and gave away possession just two times, completing two dribbles and making two clearances. Carter-Vickers was not a target
for Germany to attack nearly as much as the other three defenders on the pitch, a strong nod to his abilities.

65th min: Brenden Aaronson — 7

Probably the the U.S. substitute to make the biggest impact, Brenden Aaronson was bright in the attack, dazzling with his dribbling skills and helping usher the
U.S. forward amidst a second-half onslaught from Germany. A good showing from the Union Berlin winger who needs confidence at the moment.

65th min: Ricardo Pepi — 5

It wasn't an ideal situation for Ricardo Pepi to enter in, coming on with the U.S. shell-shocked from the quickfire Germany double. Still, he provided very
little up front, with just eight touches in 24 minutes of action.
</item 6>

<item 7>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
There's a choice for Nagelsmann to make up front, as Niklas Fullkrug has been prolific in his short national team stint, but Thomas Muller is an experienced
veteran who is still performing at a high level.

Injuries for Nagelsmann to work around include Serge Gnabry who is out with a broken arm, and Timo Werner who has been touch and go at RB Leipzig with a back
problem.

Germany predicted starting lineup vs. USA (4-2-3-1): ter Stegen (GK) — Kimmich, Rudiger, Sule, Raum — Gundogan, Goretzka — Sane, Musiala, Brandt — Fullkrug.

Germany second-choice lineup (4-2-3-1): Leno (GK) — Gross, Tah, Thiaw, Gosens — Andrich, Kimmich — Hoffman, Wirtz, Fuhrich — Muller.

German national team squad for USA, Mexico friendlies

Pos Player Club Caps Goals GK
</item 7>

<item 8>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 8>

<item 9>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
🎥: @onesoccer pic.twitter.com/XzBldBSXfX — Canada Soccer (@CanadaSoccerEN) November 17, 2023

2 hours to kickoff: While the Canadian federation is reeling from the departure of John Herdman, one player is happy to see change happen at the top. Central
midfielder Mark-Anthony Kaye had been left out of multiple squads towards the end of Herdman's tenure, and he believes his relationship with the former Canada
head coach was the reason.

Me and John had some really tough conversations where he blatantly told me I wasn’t good enough to be part of the team. That was hard for me to accept,” Kaye
said to Sportsnet, after also posting to social media an honest self-reflection following his New England Revolution's elimination from the MLS playoffs a week
ago.

Heartfelt post by Mark-Anthony Kaye.
</item 9>

<item 10>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Matt Turner says no ❌



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/x2HslCk3s2 — B/R Football (@brfootball) October 14, 2023

RB: Joe Scally — 6

Joe Scally was tidy in possession, and while he offered very little in the attack, that was largely by design as they didn't need him on that right side with
Tim Weah cooking.

Scally defended ok, and wasn't guilty of any big mistakes, which is more than most of the other U.S. defenders can say. It wasn't a standout performance by any
means, and the whole back line has some soul searching to do, but others were far more culpable than the Borussia Monchengladbach man.
</item 10>

<item 11>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Antonio was replaced by Salernitana striker Trivante Stewart, but he struggled and was himself withdrawn in the 81st minute. Demarai Gray could step in, but he
himself is struggling for fitness, and while he remains with the team he was not part of the first leg matchday squad. Another option is Colorado Switchbacks
forward Romario Williams, who scored 15 goals in 29 USL Championship matches this season.

Other injuries for Jamaica to contend with include Cory Burke and Amari'i Bell, who both miss out on this international window.

Jamaica projected starting lineup (4-3-3): Blake (GK) — Leigh, Bernard, Pinnock, Lembikisa — Latibeaudiere, D. Johnson, Bailey — Reid, Nicholson, R. Williams.
</item 11>

<item 12>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
"I would have tried to create a good atmosphere in Germany before the Euros. That's why I would be in all 10 European Championship stadiums for the friendly
matches and would have shown myself there once as a DFB selection. That can create a mood. But what do we want in the US at this point?"
</item 12>

<item 13>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The third German goal was emblematic of this lack of chemistry between the two U.S. center-backs: Richards stepped up to guard a free Niklas Fullkrug who got by
Sergino Dest, and Ream was unaware as that left Leroy Sane free on goal himself. These two just didn't have it, and Ream was powerless to pick up the slack.

Jamal Musiala dances through the USA defense and Germany lead 3-1 😮‍💨



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/PdjsjkObZR — B/R Football (@brfootball) October 14, 2023

LB: Sergino Dest — 4

If it were possible to give Sergino Dest two different scores, one for the attacking half and one for the defensive half, we'd do it, but that's not how this
works.
</item 13>

<item 14>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
.@pulisic with a SCREAMER to open the scoring 😱 pic.twitter.com/KlG0oMeNPm — U.S. Men's National Soccer Team (@USMNT) October 14, 2023

“I think we created some problems,” Berhalter said. “And then just the threat we had a lot in the first half was really good. It put Germany on their heels at
times. But to me, it’s also about controlling the tempo of the game, and that’s what we really need to learn.”

It was a victory of sorts for Reyna to play 45 minutes after he’d gone only 27 in his one club appearance this season for Borussia Dortmund. He’d been injured
in June while playing for the U.S. and missed all of preseason training and nearly two months of the Bundesliga season.

“The goal is to send him back to Dortmund in a really good spot,” Berhalter said.
</item 14>

<item 15>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Bayern Munich have their first game back against Mainz on Saturday, as does much of the Bundesliga.

"I don't know if I'd be interested in playing against Mexico in America before [the Euros hosted] in Germany," said Bayern Munich manager Thomas Tuchel prior to
the international break. "I don't know if anyone can explain this in a way that I can understand. I don't know who decided it, but it has been decided.

"There is a difference between a busy schedule and everything away from home and in different time zones. This is at the end of the endurance. No coach in the
world at this level will say otherwise."

"I don't understand the trip to the USA — for a variety of reasons," said former Germany international Lothar Matthaus to German outlet Bild back in the summer.
"I would have tried to create a good atmosphere in Germany before the Euros.
</item 15>

<item 16>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
“I think he played a good game, showed his quality. I think everyone in the first half played well. So it was good to see him, good to have him on the field.”

It was not good to not have him on the field, because the collapse from a 1-1 halftime tie not only followed Reyna’s departure, but probably was connected. The
U.S. was able to threaten the opposing goal enough with Reyna on the field to cause the opposition to concern itself with defending more often than it
preferred. Replacement Luca de la Torre dulled the attack and did nothing to dissuade Germany’s rampage toward the American net.

There was plenty of blame to be shared on the goals by Niclas Fullkrug in the 58th minute and Jamal Musiala three minutes later.
</item 16>

<item 17>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Sporting News reviews how each USMNT player performed across the 90 minutes.

MORE: Recap the 3-1 defeat for the USMNT against Germany in Hartford

USA player ratings vs Germany

USA starting lineup (4-3-3, right to left): 1. Turner (GK) — 19. Scally, 4. Richards (Carter-Vickers, 66'), 13. Ream, 2. Dest — 7. Reyna (de la Torre, 46'), 6.
Musah, 8. McKennie (Cardoso, 75') — 21. Weah (Aaronson, 66'), 20. Balogun (Pepi, 66'), 10. Pulisic (Paredes, 75').
</item 17>

<item 18>
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
</item 18>

<item 19>
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
</item 19>

<item 20>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.
</item 20>

<item 21>
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
</item 21>

<item 22>
Watch Argentina vs Paraguay live stream, TV channel, lineups, betting odds for World Cup qualifying match (Sporting News, 2023-10-12T21:11:00+00:00)
local time in Buenos Aires.

Here's how that time translates across some of the major territories:

Date Kickoff time USA/Canada Thu, Oct. 12 7 p.m. ET USA/Canada Thu, Oct. 12 4 p.m. PT UK Fri, Oct. 13 12 a.m. BST Australia Fri, Oct. 13 9 a.m. AEST India Fri,
Oct. 13 4:30 a.m. IST Hong Kong Fri, Oct. 13 7 a.m. HKT Malaysia Fri, Oct. 13 7 a.m. MYT Singapore Fri, Oct. 13 7 a.m. SGT New Zealand Fri, Oct. 13 11 a.m.
</item 22>

<item 23>
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
</item 23>

<item 24>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Jamaica's first-ever victory on Canadian soil has seen the Reggae Boyz fight back to reach the CONCACAF Nations League semifinal and qualify for Copa America
2024.

A 3-2 win in the second leg at BMO Field in Toronto allowed Jamaica to progress on away goals after a 4-4 aggregate tie across the two legs.

After Canada opened the scoring through Alphonso Davies they held a comfortable 3-1 aggregate lead, but that advantage eroded in 15 second-half minutes as
Jamaican striker Shamar Nicholson scored a brace, before Fulham attacker Bobby De Cordova-Reid converted the penalty that sealed progression.

Canada will now have to defeat Trinidad and Tobago in a one-off playoff in March to reach the Copa America 2024 tournament to be hosted in the U.S.

MORE: Highlights, recap of Canada's narrow win in the first leg

The Sporting News followed the Canada vs. Jamaica match live, providing score updates, commentary and highlights as they happened.
</item 24>

<item 25>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
With the United States set to test themselves against one of Europe's most powerful sides, the opponents are entering a transitional phase as well.

Germany are considered one of the preeminent national team programs in the world, but recent years have humbled Die Mannschaft. They have been eliminated in the
group stage of two straight World Cup tournaments, an unthinkable slip in stature given they had never been bounced from any World Cup that early in history
prior to 2018.

With Hansi Flick unable to steer the team to safer waters in Qatar, Julian Nagelsmann has taken over as new Germany head coach.

At just 36 years old, Nagelsmann is a rising star, having spent three successful seasons at Hoffenheim followed by two years at RB Leipzig, which catapulted him
to the managerial post at Bayern Munich in 2021. He steered the Bavarian giants to the Bundesliga title in his first season, but he was sacked by the club in
March of 2023 amidst a furious league title fight.
</item 25>

<item 26>
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
</item 26>

<item 27>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
The ousted Nations League quarterfinalists will compete for two more qualification spots, and the calibre of teams in those playoffs is likely to be weaker,
with the USA looking almost certain not to need that route following their 3-0 first-leg win over Trinidad and Tobago.

Eustaquio's first international goal since 2021 also ended Canada's run of two consecutive defeats and gave them a second win in seven matches.

Reaction from Milan Borjan:



"It feels amazing. What the boys did today, in this heat and on this pitch, is unbelievable. We need to come out again and show the brotherhood in Toronto. We
need a full stadium there to give us that support."
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
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
CB: Chris Richards — 4

Statistically, Chris Richards had a pretty good match, missing with just one pass of his 42 attempted and not being beaten at any point by any German attackers.
Yet his lack of volume speaks... well, volumes.

Richards was off the pace for much of his 65 minutes, and failed to get involved as his teammates were beaten repeatedly on the dribble. It was Richards'
positioning that let him down, and he should have been more involved with the U.S. under siege in such an open match.

CB: Tim Ream — 5

Chris Richards often left Tim Ream on an island, and the 36-year-old just could not cope with the waves of German attackers.

The third German goal was emblematic of this lack of chemistry between the two U.S.
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

<item 31>
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
</item 31>

<item 32>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.
</item 32>

<item 33>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
While Christian Pulisic starred on the left, Tim Weah was equally devastating down the right, dazzling with electric pace and good link-up.

Weah's biggest moment came when he scorched Robin Gosens on the wing in the first half, knocking it past the German defender and then beating him with
blistering speed to get back on the ball. Weah's statistical haul don't quite do him justice, but he was a threat throughout his 66 minutes of action.

ST: Folarin Balogun — 7

U.S. fans are salivating for Folarin Balogun to get more touches up front, and as Gregg Berhalter foreshadowed pre-match, they made a clear attempt to do that.
Gio Reyna and Tim Weah did the most of the work in that regard,

LW: Christian Pulisic — 9

There's no doubting who remains the most electric and incisive U.S.
</item 33>

<item 34>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 34>

<item 35>
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
</item 35>

<item 36>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Sergino Dest allowed the ball to deflect off his foot and into dangerous territory at the top of the box, de la Torre became a spectator with a too-great view
and central defender Chris Richards missed two opportunities to shut down the threat. McKennie, central defender Tim Ream and right back Joe Scally all bore
some responsibility for the second.

“When you watch the goals back — I had the opportunity to do that in the coaches’ office — it’s really these fine moments, these split seconds where you need to
be well-positioned, not open up any gaps in your back line,” Berhalter said. “And that’s where it went wrong.”

With the attempt to transform Musah into some semblance of Adams, though, it almost was certain to go wrong. The USMNT conjured a more compelling attack because
of Reyna’s presence, but the ability to control the game went missing.
</item 36>

<item 37>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
14 MID Pascal Gross Brighton (ENG) 2 0 MID Ilkay Gundogan Barcelona (SPA) 69 17 MID Jonas Hoffman Bayer Leverkusen 22 4 MID Joshua Kimmich Bayern Munich 80 6
MID Jamal Musiala Bayern Munich 23 1 MID Leroy Sane Bayern Munich 55 13 MID Florian Wirtz Bayer Leverkusen 10 0 FWD Kevin Behrens Union Berlin 0 0 FWD Niklas
Fullkrug Borussia Dortmund 9 7 FWD Kai Havertz Arsenal (ENG) 39 13 FWD Thomas Muller Bayern Munich 123 45

Julian Nagelsmann tactics, style of play

Through his time at Bayern Munich, Julian Nagelsmann preferred a 4-2-3-1 formation with Joshua Kimmich and Leon Goretzka holding down the lynchpin central
midfield spots.
</item 37>

<item 38>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Gio Reyna looked sharp in the starting lineup for the United States men’s national team, in all the ways a young man can: his unyielding energy, his surprising
eagerness to cover the whole of the midfield, the periodic flashes of skill when the opportunity presented itself, even the bleach-blond hair that allowed
viewers to spot him in an instant.

There was one substantial element missing from his performance, though, in an immense friendly Saturday against Germany: being able to endure from starting to
finishing.

It was kind of like that for the USMNT as a whole, as they blew an early lead and fell, 3-1, in East Hartford, Conn.

MORE: USMNT undone by errors in loss to Germany

It is quite possible this was the most challenging friendly the U.S.
</item 38>

<item 39>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He was constantly unsure of where he should be as Germany built forward with incisive pace, and he was one of many U.S. players caught unaware on the first-half
Ilkay Gundogan goal.

McKennie didn't put a first-half pass wrong, going 18/18 before the break and 28/30 overall, but his positioning was all wrong out of possession. McKennie was a
sieve as the U.S. midfield was invisible defensively, leaving their defenders completely unshielded, particularly at the top of the penalty area.

It was only a matter of time before the floodgates opened, and that they did after halftime. McKennie was eventually withdrawn in what felt like one of Gregg
Berhalter's only tactical substitutions of the match.
</item 39>

<item 40>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 40>

<item 41>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
So many national team fans sided with Reyna during the controversy that developed as the team competed at last winter's World Cup. They got what they argued for
in this game, but not what they wished.
</item 41>

<item 42>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
German national team squad for USA, Mexico friendlies

Pos Player Club Caps Goals GK Oliver Baumann Hoffenheim 0 0 GK Bernd Leno Fulham (ENG) 9 0 GK Marc-Andre ter Stegen Barcelona (SPA) 36 0 GK Kevin Trapp
Eintracht Frankfurt 7 0 DEF Robin Gosens Union Berlin 18 2 DEF Mats Hummels Borussia Dortmund 76 5 DEF David Raum RB Leipzig 18 0 DEF Antonio Rudiger Real
Madrid 62 2 DEF Niklas Sule Borussia Dortmund 47 1 DEF Jonathan Tah Bayer Leverkusen 17 0 DEF Malick Thiaw AC Milan (ITA) 2 0 MID Robert Andrich Bayer
Leverkusen 0 0 MID Julian Brandt Borussia Dortmund 44 3 MID Chris Fuhrich VfB Stuttgart 0 0 MID Leon Goretzka Bayern Munich 53 14 MID Pascal Gross Brighton
(ENG) 2 0 MID Ilkay Gundogan Barcelona
</item 42>

<item 43>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Along the back, Nagelsmann has oddly selected five center-backs compared to just two natural full-backs in David Raum and Robin Gosens, both left-sided players,
possibly indicating a desire to play a three-CB system, although it's hard to know for sure as Nagelsmann preferred a 4-2-3-1 for much of his club managerial
career.

One of the center-backs present in the squad is Mats Hummels, who has not been called in since 2021 and dropped for the entirety of Hansi Flick's time in
charge. This came at the expense of Hummels' Borussia Dortmund teammate Nico Schlotterbeck.

First national team call-ups were handed to Kevin Behrens of Union Berlin, Chris Fuhrich of Stuttgart, and Robert Andrich of Bayer Leverkusen.
</item 43>

<item 44>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Nagelsmann's dismissal at Bayern Munich came as a shock to much of the European football landscape, as the club had performed exceptionally in the Champions
League and was still very much in the title race, sitting just a point behind Dortmund. Now, with a chance for redemption at the helm of the national team,
Nagelsmann will see his time begin with a friendly against the United States.

MORE: A look at the 23-man squad for Gregg Berhalter as USMNT play Germany & Ghana

Germany squad, players for USA friendly

Julian Nagelsmann will be very familiar with much of the 26-man German national team squad, as five players hail from his former club Bayern Munich, including
critical midfield duo Josh Kimmich and Leon Goretzka, who were instrumental in his setup at Allianz Arena.
</item 44>

<item 45>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 45>

<item 46>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 46>

<item 47>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
player is, as Christian Pulisic once again backed up his nickname of 'Captain America'. The first half stunner was emblematic of his performance, but beyond
that he constantly proved a threat to German's out-of-position right-back Jonathan Tah.

Pulisic completed 27 of 29 passes, a tidy haul for a player of his attacking intent, and created two chances in addition to getting the goal. His corners remain
awful, and it's a wonder why he is still on duty for that role, but otherwise Pulisic was a menace.

Christian Pulisic with an absolute banger 🚀



Watch USA vs.
</item 47>

<item 48>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Dest was lively up front as he combined with Christian Pulisic and worked very well on the ball. He got Mats Hummels booked on a counter-attack and delivered a
few very quality crosses.

Dest was also guilty of some of the game's worst defensive errors. He was completely non-existent as Leroy Sane blazed by him on the first Germany goal, and he
nearly got his team in trouble again right before halftime by holding onto the ball too long after making an interception rather than just clearing it out of
danger. Filling in for Antonee Robinson at left-back rather than his natural right, Germany targeted him on multiple occasions, to great profit.

He had a few good moments guarding Leroy Sane in the second half, but he was a top culprit for Germany's second goal by playing Niklas Fullkrug onside. If the
U.S.
</item 48>

<item 49>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
In either setup, the forward line will shift horizontally to allow a midfield player to drive into space, and given options once halted to either play into the
crowded forward line or release a bombing full-back to the off side, both of which can stretch or confuse an opponent's defensive unit.

Having spent time in the Red Bull system, Nagelsmann also deploys a complicated press that, like Gregg Berhalter, starts with the lone central striker. The goal
is to force the opponent to play the ball into overload traps out wide, or if the opponent plays centrally, a defensive midfielder would step up at the base of
a diamond (with two midfielders on the sides and the striker at the top) which would overwhelm an opposition midfield trio.

How will Germany play under Julian Nagelsmann?

The question, now, is how all this will translate to the German national team.

Looking at Julian Nagelsmann's first squad of 26 players, the biggest question is which formation he will play.
</item 49>

<item 50>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 50>
=== END ITEM id=78c1f2bdcf499c61 ===

=== ITEM id=cc92c298d53718cf ===
Gold sentences and facts:
Sentence 0: For now, Griffais explained, it's important to Valve that developers don't need to "split their attention" between different performance targets.
  Fact 0.0: Griffais explained that it is important to Valve that developers do not need to 'split their attention' between different performance targets.
  Fact 0.1: Griffais said this applies for now.
Sentence 1: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 1.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 1.1: Valve has Steam Deck OLED units ready to ship on that same day.

Retrieved items:
<item 1>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 1>

<item 2>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
The PDP Victrix BFG Wireless Controller for PlayStation 5, PS4, and PC in its Call of Duty: Modern Warfare II-branded black-and-gold design is on sale at Amazon
for $169.99 ($30 off). I know this controller looks very Extra (and it is), but it’s a very versatile DualSense alternative if you like FPS and fighting games.
It’s got multi-stage hair triggers, and its sticks and face buttons can be swapped out for an Xbox-style layout or as a six-button fightpad.

Verge Deals on X (formerly known as Twitter) / Join more than 51,000 followers and keep up with the best daily tech deals with @vergedeals Follow us!

Wireless gaming headsets, Laptops, and PC accessories

Asus ROG Flow Z13 $ 893 $ 1300 31 % off $ 893 $ 893 $ 1300 31 % off The ROG Flow Z13 is a gaming tablet with a detachable keyboard, kickstand, and plenty of
chops for QHD gaming.
</item 2>

<item 3>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
(Parts will once again be sold through iFixit.) However, I did enjoy the new power supply, with its faster charging (45 minutes gets you from 20% to 80%) and
2.5-meter cable that stretches the length of my couch.

One small, charming detail: The carrying case now has a tinier carrying case within it. If you want to be extra cautious, stick to the original bulky case. But
if you need to toss the Steam Deck in your backpack for a day trip? Go with the slim case that has funny little thumbstick nipples!

All of the tweaks, large and small, amount to a perfect handheld — made better by the price, which remains the same as the original Steam Deck available today,
just that the OLED will come with more stuff.
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The panel has a wider color gamut (110% DCI-P3, for the visual obsessives) and the beautiful, inky blacks that OLED dorks like myself adore.

Steam Deck OLED’s bigger, better battery Valve’s promise of 30-50% more battery raised one obvious question: Is that comparing the OLED model with a Steam Deck
running its original firmware (which had poor battery life), or a Steam Deck using the latest firmware? Valve’s answer: the latter! So if you’re a Steam Deck
owner, expect those big battery improvements over your current Steam Deck experience.

The OLED panel also has a 90 Hz refresh rate — perfect for frame doubling from 45 frames per second. In case that jargon means nothing to you, suffice it to say
that gameplay on the Steam Deck OLED looks smoother.
</item 4>

<item 5>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
For developers, this means the device has the same performance targets as the original and there’s no need to create games specifically for the new hardware. In
the end, the device weighs about 30 grams less than the original model, or the equivalent of five quarters.

Engadget

In use

Which leads us to the bad news for my tiny hand gang. The Steam Deck OLED is the same monstrous size as the original: 11.7 inches long, 4.5 inches tall and
nearly 2 inches thick. The thumbsticks on the new model are actually 1mm taller than the first version. This was one of my main complaints about the Steam Deck
when I first reviewed it: I have smaller-than-average hands, and the Steam Deck looks and feels absolutely ridiculous when I’m holding it. It’s just massive, no
matter how long my manicure gets. However, the OLED model feels more balanced than the original. It’s less top-heavy, and it does technically weigh less.
</item 5>

<item 6>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
</item 6>

<item 7>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Valve will be selling three models:

$649 - 1 TB OLED (up from 512 GB)

$529 - 512 GB OLED (up from 256 GB)

$399 - 256 GB non-OLED Steam Deck (up from 64 GB)

Plus, for a limited time, the company will offer a special edition 1 TB Steam Deck OLED made with a translucent shell for $679 — only in the U.S. and Canada.
Though let’s be real: This thing will sell like hotcakes and Valve will reconsider the definition of “limited supply.” Have I considered ordering one of these
limited editions? Yes. But I am also practicing restraint. Like, how many Steam Decks should one human have?

I suppose that’s the question, really. If you already have a Steam Deck, is the OLED model technically an incremental improvement? Yes. But you might find that
the screen and the battery justify the double dip.
</item 7>

<item 8>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 8>

<item 9>
Teenage Engineering's K.O. II sampler proves the company can do cost-friendly cool (Engadget, 2023-12-19T15:00:38+00:00)
There's something of a theme running through Teenage Engineering's recent products. That theme is you need more money. The Field range represents the Swedish
company’s most exclusive music making gear. So when its website teased a new product with a colorful countdown, the wallets of Teenies everywhere braced for
impact. Once that timer hit all zeros, the big reveal turned out to be the EP-133 sampler. Or, to give it its full name, the EP-133 K.O. II 64MB Sampler
Composer. The real surprise though, was that it both looked cool and, at $299, was reasonably priced.

Fadergate

The countdown was really just the start. Barely 24 hours after the K.O. II was revealed units started landing in buyers’ hands. Within days, YouTube was awash
with first look videos and tutorials. Before our review unit even showed up, several users were complaining that theirs had defective faders.
</item 9>

<item 10>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 10>

<item 11>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Even with improvements to the battery, thermals and weight distribution, the Steam Deck OLED is still more of a Wii U than a Vita. It’s a beast of a handheld,
and it makes the most sense as a portable console to use around the house, rather than a truly travel-ready device. The new model even comes with an extra-long,
2.5m charging cable, designed specifically with couch-based players in mind. I didn’t test out the docking station with the new Steam Deck, but Valve recently
released a SteamOS update that lets players take advantage of HDR and VRR on external displays, which only drives home the Wii U-ness of it all.

As far as software, you know the drill by now. Valve has rolled out labels for Steam games signifying how well they’ll work on the handheld, with four levels:
unknown, unsupported, playable and verified. Playable games generally come with a small caveat, like having to deploy the on-screen keyboard, while verified
games have been fully optimized for portable play.
</item 11>

<item 12>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
And critically, Microsoft has a habit of making its big blockbuster games available to play on Game Pass, for a monthly fee, on the first day they release.

Image source, EPA-EFE Image caption, Bethesda, which is owned by Microsoft, launched its new game Starfield in 2023 - but only on Xbox and PC

It's up to gamers to decide for themselves whether it's a better deal to buy the game or pay to access it.

But it is worth noting this concerned the UK regulator so much it made sure Activision Blizzard games couldn't be streamed exclusively on Xbox for 15 years.

That might be because Microsoft has form here - when the hotly anticipated Starfield came out in September 2023 to critical acclaim, it was available
exclusively on PC and Xbox, and playable on Game Pass from day one.

If games are still on PlayStation, what's in it for Microsoft?

Even with games still able to appear on PlayStation, it's Microsoft who will be winning out.
</item 12>

<item 13>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Here are the best Prime Day deals we could find gathered in one place.

Amazon's second Prime Day-like sale event for 2023 kicked off in the wee hours this morning, and it's turning out to be a great opportunity to get holiday
shopping done early. Or, if you missed out on any of the deals from Prime Day in July, this year's "Prime Big Deal Days" will let you make up for that. The sale
has brought back many of the same steep discounts we saw during the summer as well as new sales. We've also put together category roundups for deals on
headphones, tablets, TVs, video games and other gadgets.
</item 13>

<item 14>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 14>

<item 15>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The Steam Deck OLED now includes a Wi-Fi 6E radio (great if you have a modern router and connection to make use of it) and a dedicated Bluetooth module, which
allows for improved audio quality with wireless headphones, the option to wake the Steam Deck from a controller, and the capability to connect many Bluetooth
controllers for multiplayer. The haptics are noticeably improved, with a wider frequency range. And the touchscreen is markedly better — it’s responsive to the
point where I don’t feel the need to plug in a USB-C keyboard anytime I need to type a few words.

A review for spec specifics If you’re the sort of person who likes the nitty-gritty technical details of screens and batteries, we also recommend reading The
Verge’s review.

Valve also claims that the Steam Deck OLED is more repairable, which, mercifully, I didn’t have the need to test for myself. (Parts will once again be sold
through iFixit.)
</item 15>

<item 16>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 16>

<item 17>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management. My palms have gotten toasty playing the OLED
model, but they haven’t broken a sweat like they often do with the original. The new fan may be larger, but it’s not any louder, and it clearly works a treat.

The Steam Deck OLED has a new Wi-Fi 6E module and improved antennas that should enable faster downloads for anyone with a compatible router. I don’t have a 6E
router or multi-gigabit Wi-Fi, but with a regular Wi-Fi 6 router, the top download speed I saw on the new Steam Deck was 562mbps, which was about 10 percent
faster than the old Steam Deck, and 10 percent slower than the speeds I get on my MacBook Pro. The device uses Bluetooth 5.3 and adds a dedicated antenna, which
should improve functionality when using Wi-Fi and Bluetooth at the same time.
</item 17>

<item 18>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.
</item 18>

<item 19>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 19>

<item 20>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Because while Valve hasn’t upped the horsepower, it has improved practically everything else. The Steam Deck OLED has longer battery life, faster Wi-Fi, and a
stand-alone Bluetooth module. And the screen isn’t just OLED; it’s HDR OLED — the first HDR OLED screen on any gaming portable, with a peak brightness of 1,000
nits for HDR content and 600 nits for SDR (for comparison, the original screen’s peak brightness was 400 nits). And somehow, with all of these additions, the
Steam Deck OLED is lighter.

So, goodbye, old Steam Deck; hello, new Steam Deck.

The Steam Deck OLED delivers on Valve’s original promise

OLED is in the name for a reason.

The OLED panel in the new Steam Deck is lighter, thinner, and more power efficient than its LCD predecessor. It’s also bigger and faster. Though the device has
the same shape as the original Steam Deck, smaller screen bezels allow for a display that’s slightly larger, at 7.4 inches compared to 7 inches.
</item 20>

<item 21>
Teenage Engineering's K.O. II sampler proves the company can do cost-friendly cool (Engadget, 2023-12-19T15:00:38+00:00)
Before our review unit even showed up, several users were complaining that theirs had defective faders. Enough folks were having this problem that it quickly
became known as “fadergate.” One brave creator even took their unit apart and, possibly, discovered the cause — the internal pins were bent and not making a
connection. I asked Teenage Engineering about the issue and will update this story once I hear back.

Some buyer’s theorized that the issue might be caused by the fact that the K.O. II ships without the caps on its rotaries and fader. They come loose in the box
to enable the packaging to be flatter, but the cap for the fader is unusually tight fitting. This led to speculation that the enthusiasm required to push this
cap down might be putting too much pressure on components inside, opening up the cruel possibility of users breaking their own devices before they even got to
play with them.
</item 21>

<item 22>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This investment alone means any time you buy a game built on Epic’s Unreal Engine, Tencent (and Sony) is getting a cut. Tencent is the biggest investor in
games, with thousands of tendrils across the industry — if you played something this year, Tencent was probably involved.

On a smaller scale, companies like Netflix and Devolver Digital have also dabbled in acquisitions recently. Devolver started buying studios in 2020, and it now
owns Croteam, Dodge Roll, Doinksoft, Firefly Studios, Nerial and System Era Softworks. Annapurna Interactive bought South African studio 24 Bit Games in
November. Netflix launched its Games division in 2021, and it’s already purchased four studios, including Oxenfree developer Night School and Alphabear company
Spry Fox.

Night School co-founder Sean Krankel told Engadget in June that the move to Netflix was a boon for the studio, providing financial security, a dedicated working
space and plenty of marketing support for its projects.
</item 22>

<item 23>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The improved 90 Hz refresh rate falls short of the Asus ROG Ally’s 120 Hz, though it’s worth considering that the higher the frame rate, the faster the battery
drain — assuming you can even spot the difference between 90 Hz and 120 Hz.

You know what’s more interesting than a list of technical specs? Actually seeing the screen. It’s gorgeous. On Sunday morning, I picked at Sega’s Like a Dragon
Gaiden: The Man Who Erased His Name while watching football and Christmas movies with my family. The new Yakuza game looks fantastic on high settings at 60 fps,
its night sequences flaunting the OLED’s dark blacks and the abundance of neon signage showcasing the HDR.

I got so distracted pulverizing bad dudes that it took me some time to notice the Steam Deck OLED’s most welcome feat. When playing AAA games on the original
Steam Deck, I had always kept the handheld plugged into an outlet. But with the OLED model, a couple of hours had passed before I realized I’d forgotten to
connect the charger.
</item 23>

<item 24>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Valve Valve Steam Deck OLED SCORE 89 Engadget 89 Critics - Not yet scored N/A Users - Not yet scored N/A Pros Improved battery life

Bright and smooth visuals

Better thermals

Feels less top-heavy

Long charging cable Cons Still ridiculously large

Seriously it’s so big

Valve Valve Steam Deck OLED This isn't the Steam Deck 2 some are waiting for, it's just a welcome improvement on an already good device. $549 at Steam

OLED screen

The HDR OLED display looks fantastic. It features a wider P3 color gamut, pure blacks and a peak brightness of 1,000 nits, compared with the original max of 400
nits. In action, this means the screen’s bezels meld with the black of any game, creating a more polished frame for retro aspect ratios and providing room for
higher-contrast colors. The screen on the new model doesn’t support variable refresh rates (VRR), but it now tops out at 90Hz, while the original could only hit
60Hz.
</item 24>

<item 25>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 25>

<item 26>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
It sports drift-free Hall effect sticks and is compatible with PC, Nintendo Switch, and Android.

8BitDo Ultimate Bluetooth Controller $ 56 $ 70 20 % off $ 56 $ 56 $ 70 20 % off The 8BitDo Ultimate Bluetooth Controller comes with its own charging dock,
features remappable controls, and is compatible with the Nintendo Switch, Steam Deck, and Windows PCs thanks to its Bluetooth and 2.4GHz connectivity. Unlike
the 2.4GHz-only version, it features drift-free Hall effect sticks. $56 at Amazon (black)

The RGB-ified PowerA Spectra Infinity Enhanced Wired Controller for Xbox and PC is on sale at Amazon for $29.99 ($15 off). In addition to its colorful lights,
it’s one of the cheapest controllers around to sport hair trigger lockouts.

for Xbox and PC is on sale at Amazon for $29.99 ($15 off). In addition to its colorful lights, it’s one of the cheapest controllers around to sport hair trigger
lockouts.
</item 26>

<item 27>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is almost here, but we wouldn't blame you if you've felt like it's been here for a while. What was once a one-day event filled with limited-time
deals has grown into a month-long event with a plethora of sales to sift through each day. This year, Amazon, Target, Best Buy and Walmart and several other
retailers kicked off their Black Friday deals early, with some starting as soon as late October. If you're overwhelmed by the amount of discounts showing up in
your feeds, you're not alone.



To help, we at Engadget have collected all of the best Black Friday deals you can shop right now, primarily on tech we've tested and reviewed. These are the
best tech deals you can get this year, on the gadgets we know to be worth your money.
</item 27>

<item 28>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$893 at Amazon

Sony InZone H3 $ 58 $ 100 42 % off $ 58 $ 58 $ 100 42 % off The InZone H3 is the wired (3.5mm / USB-A) model in Sony’s latest lineup of gaming headsets, which
are compatible with both PlayStation and PC. You can customize the audio profile of InZone headsets with the InZone companion app and Sony’s Spatial Sound
Personalizer. $58 at Amazon$60 at Best Buy

Elgato Stream Deck Mini $ 59 $ 80 26 % off $ 59 $ 59 $ 80 26 % off Elgato’s Stream Deck Mini is a smaller, six-key version of its popular macro pads. Its LCD
keys can be customized to offer quick one-button macros during a livestream, or it can simply control shortcuts on your computer and connect to your smart home.
</item 28>

<item 29>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The immediate reality is a bit of a mess and a little complicated, but this, Valve says, will be the Steam Deck line-up moving forward:

256GB Steam Deck LCD ($399)

512GB Steam Deck OLED ($549)

1TB Steam Deck OLED ($649)

As a result of the changes, the 64GB LCD Steam Deck (previously sold at $399) will drop to $349, and the 512GB LCD (previously $649) will drop to $499. Once
Valve sells through its inventory of both, those lines will be discontinued. Valve says it will continue to support all LCD models with software updates.

So, the long-term lineup starts at the same $399 price, which gets you the old LCD Steam Deck, but with 256GB of storage instead of 64GB. The new step-up model
costs $20 more at $549, but comes with an OLED display and the improvements outlined above, plus double the storage.
</item 29>

<item 30>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 30>

<item 31>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
You can get this two-pack of 100 W-ready, six-foot-long, USB-C-to-C cables at Amazon for $15.99.

Related The best Nintendo Switch accessories

Best power bank for your Steam Deck

Each Steam Deck console includes a 45 W USB-C wall adapter, so it’s a good idea to find a power bank that can meet that power level when you’re on the go.
Thankfully, some options exist that can do this. One thing to note during your search is that only batteries that support the PD 3.0 spec, including the two
options below, can recharge the Steam Deck at its fastest speed.

Related The best Steam Deck accessories

Best portable charger for your laptop

The number of laptops that can recharge via USB-C is growing, making it possible to use the same charger to refill your laptop as you do your phone or Nintendo
Switch. For instance, Apple’s entire MacBook lineup allows it, as do several popular Windows laptops, like the Dell XPS 13, Razer’s Blade gaming laptop, and
more.
</item 31>

<item 32>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
Similar to Walmart and Amazon, be aware that items at Target shipped or sold by third-party retailers can result in longer lead times. Product listings usually
spell out your options for pickup, same-day delivery, and standard shipping — complete with estimated availability times — before you add them to your cart.
</item 32>

<item 33>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 33>

<item 34>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 34>

<item 35>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Hades is my most-played game on the original Steam Deck, so I used that to test out Valve’s battery life claims. Though it doesn’t support HDR, a game like
Hades pops on the OLED, and its animations look more fluid than on the LCD version. The OLED lasted four hours and 23 minutes before dying. With the same
settings (800p with a 60 fps cap and similar brightness) the original model died after two hours and 42 minutes. That’s a 62 percent improvement, surpassing
Valve’s own estimates, though batteries degrade and I’ve had my LCD Steam Deck since the device’s launch in early 2022. In a stress test that involved playing
Elden Ring with everything maxed out, the OLED battery lasted two hours and 20 minutes, about 40 percent longer than the original.

Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management.
</item 35>

<item 36>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$400 at Amazon (refurbished)

Jsaux Mod Case for Steam Deck $ 24 $ 30 20 % off $ 24 $ 24 $ 30 20 % off A protective modular case system for Valve’s Steam Deck. It consists of a base case
with grippy texture, a built-in kickstand, a removable lid, and a rear mounting system that can attach a power bank using an adjustable strap or Jsaux’s add-on
cooling fan. $24 at Amazon

Jsaux’s 45W USB-C charger with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a
Nintendo Switch or just about any USB-C phone or tablet. It’s just $15.19 ($3 off) right now at Amazon.

with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a Nintendo Switch or just about
any USB-C phone or tablet.
</item 36>

<item 37>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
It's our top pick in our best microSD card guide because it was the most consistent of the ones we tested, with some of the best overall speeds and all of the
important ratings that a good card should have: U3, V30 and A2. It'll be a great pick for your tablet, smartphone or Nintendo Switch, and it comes with a
full-sized adapter in case you need to use it with a camera or connect it to your computer.

Backbone One

The Backbone One is a well-built gamepad that snaps around the sides of your phone and works immediately with just about any game that supports controller
input. It’s also convenient for cloud streaming games from a console or PC. Right now both its USB-C and Lightning models are down to $70, which is $30 off.
That’s not an all-time low, but discounts on the device are relatively uncommon. We gave the One a positive review last year, and we currently highlight it in
our guide to the best handheld gaming systems.
</item 37>

<item 38>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 38>

<item 39>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
Meanwhile Sophie, who was named Ubisoft's content creator of the year, told the BBC she "just wants the games to win".

"We want everyone to be able to play video games that they want to play, we want them to be accessible for everyone," she said.

"Obviously, we're going to have our opinions, but if the end result is good games, that's fine with us."

This Instagram post cannot be displayed in your browser. Please enable Javascript or try a different browser. View original content on Instagram The BBC is not
responsible for the content of external sites. Skip instagram post by cadaea Allow Instagram content? This article contains content provided by Instagram. We
ask for your permission before anything is loaded, as they may be using cookies and other technologies. You may want to read Meta’s Instagram cookie policy,
external and privacy policy, external before accepting. To view this content choose ‘accept and continue’. Accept and continue The BBC is not responsible for
the content of external sites.
</item 39>

<item 40>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
Email (required) Sign up By submitting your email, you agree to our Terms and Privacy Notice . This site is protected by reCAPTCHA and the Google Privacy Policy
and Terms of Service apply.
</item 40>

<item 41>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is smooth, crisp and bright, like a perfect lemon tart. (Even then, I do not recommend licking the screen.)

Battery life

One of the downsides of the first Steam Deck was its short battery life, which maxed out at eight hours, and that was only under ideal conditions. Valve says
the new Steam Deck battery will last 30 to 50 percent longer, depending on how you’re playing, and it charges faster — the company claims it will go from 20
percent to 80 percent in 45 minutes.

Custom AMD APU

The Steam Deck OLED has a die-shrunk version of the custom AMD APU found in the original Steam Deck — same CPU and GPU, just more efficient. This is the process
Sony takes advantage of to build the PS4 Slim and PS5 Slim, but instead of collapsing the Steam Deck around smaller hardware, Valve chose to add a bigger
battery and larger fan to its existing chassis.
</item 41>

<item 42>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
It’s less top-heavy, and it does technically weigh less. These small adjustments have made a noticeable difference during my playtime, and the device feels
slightly more manageable as a handheld — or maybe I’m just too distracted by all of its bright, pretty colors to care about the cramping in my palms.

The thumbsticks on the new handheld have an improved texture that shouldn’t turn gray after a few months’ use, and they feel perfectly serviceable. In my
hands-on time so far, I've found that the OLED touchscreen responds immediately to input, and the haptic feedback on the built-in trackpads feels even more
precise, populating in reactive pin-pricks under the textured plastic.

I spent a few cozy nights on my couch swapping between the LCD and OLED Steam Decks, and honestly, after I’m done with this review, I’ll probably retire my
original model. The Steam Deck OLED is sharper and more responsive; indie games, AAA games and plain old Steam menus look so much better on the new hardware.
</item 42>

<item 43>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$59 at Amazon

Storage for consoles and PCs

A massive 1TB SanDisk Extreme microSD card is now $89.99 $79.99 ($13 off) at Amazon. That can store a whole lot of games on a Steam Deck or Nintendo Switch.

Samsung Evo Select microSD card (512GB) $ 30 $ 40 25 % off $ 30 $ 30 $ 40 25 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $30 at Amazon

Western Digital WD_Black SN850X SSD $ 70 $ 180 61 % off $ 70 $ 70 $ 180 61 % off The WD_Black SN850X is one of the fastest M.2 drives around, with a sequential
read speed of up to 7,300 MB/s and a write speed of up to 6,600 MB/s.
</item 43>

<item 44>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 44>

<item 45>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 45>

<item 46>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
These days, Steam generally peaks at over 30 million, with valleys of 20 million. Fortnite’s valleys are closer to 3.8 million.]

Tim Sweeney says he plays Jellie in Fortnite. Image: Epic Games

Okay, let’s do a lightning round. What’s your favorite Fortnite skin?

I play as Jellie! Jellyfish. You know, the big tentacles? It’s just so cool!

Will Epic ever announce Unreal Tournament 3 X?

Forever in motion, the future is.

Why is Fortnite still not playable on Steam Deck?

If we only had a few more programmers. It’s the Linux problem. I love the Steam Deck hardware. Valve has done an amazing job there; I wish they would get to
tens of millions of users, at which point it would actually make sense to support it.
</item 46>

<item 47>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 47>

<item 48>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.
</item 48>

<item 49>
The Verge’s 2023 holiday gift guide for kids (The Verge, 2023-12-18T23:14:07+00:00)
Price: $119.99 Amazon Best Buy

8BitDo SN30 Pro (Special Edition) The 8BitDo SN30 is more than a small controller best suited for tiny hands. It also oozes nostalgic vibes to make you feel a
bit of what mom and dad did while playing the original SNES — albeit, with modern games on PC or the Nintendo Switch. What’s more, its see-through color schemes
are sure to entice just about everyone, regardless of what year you were born.

Price: $44.99+ Amazon Walmart 8BitDo

Nintendo Switch Lite (Isabelle’s Aloha Edition) The Steam Deck may have ushered in a new era of handheld gaming, but it’s a bit pricey and overbearing for a
child. That said, there’s still no substitute for the simplicity of Nintendo’s Switch Lite and its travel-friendly 5.5-inch display.
</item 49>

<item 50>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 50>
=== END ITEM id=cc92c298d53718cf ===
