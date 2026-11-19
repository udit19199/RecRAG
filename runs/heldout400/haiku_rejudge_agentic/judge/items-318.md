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

=== ITEM id=c3405ccc88030503 ===
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
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.

Graph facts:
- McRae reveals hidden message he carried under collar of his grand final shirt -MENTIONS-> Adams
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Adams
- Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat -MENTIONS-> Adams
- NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable -MENTIONS-> Adams
- Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Adams

Entities: Adams, Ghana, Gregg Berhalter, ankle injury, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, Johnny
Cardoso, Internacional, Lennard Maloney, Heidenheim, October friendlies, sports hernia, September camp, October squad call-up, USMNT, Sporting News
</item 1>

<item 2>
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

Graph facts:
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Oman
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Oman
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana

Entities: Oman, Ghana, Gregg Berhalter, Celtic, AC Milan, October friendlies against Germany and Ghana, Friendlies against Uzbekistan and Oman, Uzbekistan,
Malik Tillman, PSV Eindhoven, Alex Zendejas, Club America, Ethan Horvath, Nott. Forest, USA squad for October friendlies 2023: USMNT roster for friendly matches
vs Germany, Ghana
</item 2>

<item 3>
Graph query row:
article: USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana
publication_date: 2023-10-04T22:42:00+00:00
publisher: Sporting News
connected_entities: ['Adams', 'Wolfsburg', 'hamstring injury', 'Oman', 'Ghana', 'Juventus', 'Fulham', 'Bournemouth', 'Leeds United', 'Gregg Berhalter',
'Connecticut', '2024 Copa America', 'New England Revolution', 'Union Berlin', 'Celtic', 'Dortmund', 'AC Milan', 'ankle injury', "U.S. men's national team",
'2026 FIFA World Cup', 'Euro 2024 qualifying', 'friendly against Ghana', 'GEODIS Park', 'Nashville', 'World Cup tournament in 2006', 'World Cup tournament in
2010', 'World Cup tournament in 2014', 'October friendlies against Germany and Ghana', 'Friendlies against Uzbekistan and Oman', 'Uzbekistan', 'Malik Tillman',
'PSV Eindhoven', 'Alex Zendejas', 'Club America', 'Ethan Horvath', 'Nott. Forest', 'friendly matches vs Germany, Ghana', 'USA', 'Cameron Carter-Vickers',
'Sergino Dest', 'DeJuan Jones', 'Kristoffer Lund', 'Palermo', 'Kevin Paredes', 'Tim Ream', 'Chris Richards', 'Miles Robinson', 'Gladbach', 'Johnny Cardoso',
'Internacional', 'Luca de la Torre', 'Celta Vigo', 'Lennard Maloney', 'Heidenheim', 'Weston McKennie', 'Yunus Musah', 'October friendlies', 'Tyler Adams',
'October camp', 'Zack Steffen', 'Sean Johnson', 'Josh Sargent', 'Daryl Dike', 'leg fracture', 'Achilles tear', 'October friendlies 2023', 'sports hernia',
'September camp', 'October squad call-up', 'Gio Reyna', 'summer Nations League duty', 'domestic violence investigation of Berhalter', 'domestic violence',
'Berhalter', 'Reyna', 'friendly against Germany', 'USMNT', 'Christian Pulisic', 'EFL Cup', 'Tennessee', 'international break', 'Crystal Palace', 'Germany',
'Folarin Balogun', 'Atlanta United', '2022 World Cup', 'Arsenal', 'Julian Nagelsmann', 'Borussia Dortmund', 'Bayern Munich', 'Berlin', 'Matt Turner',
'Hartford', 'Joe Scally', 'Tim Weah', 'Antonee Robinson', 'Brenden Aaronson', 'Ricardo Pepi', 'Zoom']
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

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- French soccer players back African mobile video network StarNews Mobile in $3M round -MENTIONS-> Ghana
- AfroFuture -LOCATED_IN-> Ghana

Entities: Ghana, Gregg Berhalter, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies, leg fracture,
Gio Reyna, summer Nations League duty, domestic violence investigation of Berhalter, domestic violence, USMNT, Sporting News, Germany, 2022 World Cup, Borussia
Dortmund
</item 4>

<item 5>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> hamstring injury
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> hamstring injury
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> hamstring injury
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> hamstring injury
- Jamaal Williams -MENTIONS-> hamstring injury

Entities: hamstring injury, Ghana, Juventus, Gregg Berhalter, Union Berlin, AC Milan, Malik Tillman, PSV Eindhoven, Alex Zendejas, Club America, USA squad for
October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies, Tyler Adams, October camp, Gio Reyna
</item 5>

<item 6>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
Nagelsmann's dismissal at Bayern Munich came as a shock to much of the European football landscape, as the club had performed exceptionally in the Champions
League and was still very much in the title race, sitting just a point behind Dortmund. Now, with a chance for redemption at the helm of the national team,
Nagelsmann will see his time begin with a friendly against the United States.

MORE: A look at the 23-man squad for Gregg Berhalter as USMNT play Germany & Ghana

Germany squad, players for USA friendly

Julian Nagelsmann will be very familiar with much of the 26-man German national team squad, as five players hail from his former club Bayern Munich, including
critical midfield duo Josh Kimmich and Leon Goretzka, who were instrumental in his setup at Allianz Arena.

Graph facts:
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Gregg Berhalter
- Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat -MENTIONS-> Gregg Berhalter
- USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana -MENTIONS-> Gregg Berhalter
- USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah -MENTIONS-> Gregg Berhalter
- Gregg Berhalter -SAID-> Antonee Robinson

Entities: Gregg Berhalter, Dortmund, Champions League, Ghana, USMNT, Sporting News, Germany, Julian Nagelsmann, Josh Kimmich, Leon Goretzka, Allianz Arena, USA
friendly, title race, What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager, Bayern Munich
</item 6>

<item 7>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Wolfsburg
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Juventus
- Wolfsburg -PARTICIPATED_IN-> 2015-16 deciding group match
- Manchester United -COMPETES_WITH-> Wolfsburg
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Juventus

Entities: Wolfsburg, Juventus, Fulham, New England Revolution, Celtic, AC Milan, Ghana, Malik Tillman, PSV Eindhoven, USA squad for October friendlies 2023:
USMNT roster for friendly matches vs Germany, Ghana, friendly matches vs Germany, Ghana, USA, Cameron Carter-Vickers, Sergino Dest, DeJuan Jones
</item 7>

<item 8>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- French soccer players back African mobile video network StarNews Mobile in $3M round -MENTIONS-> Ghana
- AfroFuture -LOCATED_IN-> Ghana

Entities: Ghana, Dortmund, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies 2023, Berhalter,
Reyna, USMNT, Sporting News, Germany, Zoom
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> hamstring injury
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> hamstring injury
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> hamstring injury
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> hamstring injury
- Jamaal Williams -MENTIONS-> hamstring injury

Entities: hamstring injury, Ghana, Bournemouth, Leeds United, Celtic, ankle injury, USA squad for October friendlies 2023: USMNT roster for friendly matches vs
Germany, Ghana, Cameron Carter-Vickers, Zack Steffen, Sean Johnson, Josh Sargent, Daryl Dike, leg fracture, Achilles tear, October friendlies 2023
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.

Graph facts:
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> Gregg Berhalter
- Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat -MENTIONS-> Gregg Berhalter
- What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager -MENTIONS-> Gregg Berhalter
- USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah -MENTIONS-> Gregg Berhalter
- Gregg Berhalter -SAID-> Antonee Robinson

Entities: Gregg Berhalter, Connecticut, 2024 Copa America, U.S. men's national team, 2026 FIFA World Cup, Euro 2024 qualifying, Ghana, friendly against Ghana,
GEODIS Park, Nashville, World Cup tournament in 2006, World Cup tournament in 2010, World Cup tournament in 2014, USA squad for October friendlies 2023: USMNT
roster for friendly matches vs Germany, Ghana, friendly against Germany
</item 10>
=== END ITEM id=c3405ccc88030503 ===

=== ITEM id=b560cc3acb2da477 ===
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
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> hamstring injury
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> hamstring injury
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> hamstring injury
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> hamstring injury
- Jamaal Williams -MENTIONS-> hamstring injury

Entities: hamstring injury, Ghana, Juventus, Gregg Berhalter, Union Berlin, AC Milan, Malik Tillman, PSV Eindhoven, Alex Zendejas, Club America, USA squad for
October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies, Tyler Adams, October camp, Gio Reyna
</item 1>

<item 2>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- French soccer players back African mobile video network StarNews Mobile in $3M round -MENTIONS-> Ghana
- AfroFuture -LOCATED_IN-> Ghana

Entities: Ghana, Dortmund, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies 2023, Berhalter,
Reyna, USMNT, Sporting News, Germany, Zoom
</item 2>

<item 3>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.

Graph facts:
- Chicago Fire vs Inter Miami live score, result, updates, highlights from Lionel Messi in MLS match -MENTIONS-> World Cup
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> World Cup
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -MENTIONS-> World Cup
- UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead -MENTIONS-> World Cup
- THE GREATEST! Boks beat All Blacks in instant classic to claim record FOURTH RWC in controversial, dramatic final -MENTIONS-> World Cup

Entities: World Cup, Gregg Berhalter, Tyler Adams, Gio Reyna, USMNT, 2026 World Cup, Gio Reyna looks good in start for USMNT vs. Germany, but those involved in
the finish are punished in 3-1 defeat, USMNT vs. Germany, Sporting News, Germany, North America
</item 3>

<item 4>
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

Graph facts:
- USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana -MENTIONS-> Johnny Cardoso
- Kevin Paredes -WORKS_FOR-> Wolfsburg
- USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana -MENTIONS-> Kevin Paredes
- Kevin Paredes -PART_OF-> USA
- Johnny Cardoso -WORKS_FOR-> Internacional

Entities: Kevin Paredes, Johnny Cardoso, Luca de la Torre, Lennard Maloney, USMNT, Pulisic, Sporting News, Germany, USA vs Germany player ratings: USMNT
defensive errors blight positive performance by Pulisic, Weah, Weah, player ratings, defensive errors, USA vs Germany
</item 4>

<item 5>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.

Graph facts:
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> hamstring injury
- Vikings vs. Broncos live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> hamstring injury
- Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line -MENTIONS-> hamstring injury
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> hamstring injury
- Jamaal Williams -MENTIONS-> hamstring injury

Entities: hamstring injury, Ghana, Bournemouth, Leeds United, Celtic, ankle injury, USA squad for October friendlies 2023: USMNT roster for friendly matches vs
Germany, Ghana, Cameron Carter-Vickers, Zack Steffen, Sean Johnson, Josh Sargent, Daryl Dike, leg fracture, Achilles tear, October friendlies 2023
</item 5>

<item 6>
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

Graph facts:
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Oman
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Oman
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana

Entities: Oman, Ghana, Gregg Berhalter, Celtic, AC Milan, October friendlies against Germany and Ghana, Friendlies against Uzbekistan and Oman, Uzbekistan,
Malik Tillman, PSV Eindhoven, Alex Zendejas, Club America, Ethan Horvath, Nott. Forest, USA squad for October friendlies 2023: USMNT roster for friendly matches
vs Germany, Ghana
</item 6>

<item 7>
Graph query row:
entity: Johnny Cardoso
report: USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana
published_on: 2023-10-04T22:42:00+00:00
status: USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.
</item 7>

<item 8>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.

Graph facts:
- McRae reveals hidden message he carried under collar of his grand final shirt -MENTIONS-> Adams
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Adams
- Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat -MENTIONS-> Adams
- NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable -MENTIONS-> Adams
- Jets vs. Raiders live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Adams

Entities: Adams, Ghana, Gregg Berhalter, ankle injury, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, Johnny
Cardoso, Internacional, Lennard Maloney, Heidenheim, October friendlies, sports hernia, September camp, October squad call-up, USMNT, Sporting News
</item 8>

<item 9>
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

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Ghana
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> Ghana
- ‘AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience.’ -MENTIONS-> Ghana
- French soccer players back African mobile video network StarNews Mobile in $3M round -MENTIONS-> Ghana
- AfroFuture -LOCATED_IN-> Ghana

Entities: Ghana, Gregg Berhalter, USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana, October friendlies, leg fracture,
Gio Reyna, summer Nations League duty, domestic violence investigation of Berhalter, domestic violence, USMNT, Sporting News, Germany, 2022 World Cup, Borussia
Dortmund
</item 9>

<item 10>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> Wolfsburg
- From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray -MENTIONS-> Juventus
- Wolfsburg -PARTICIPATED_IN-> 2015-16 deciding group match
- Manchester United -COMPETES_WITH-> Wolfsburg
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Juventus

Entities: Wolfsburg, Juventus, Fulham, New England Revolution, Celtic, AC Milan, Ghana, Malik Tillman, PSV Eindhoven, USA squad for October friendlies 2023:
USMNT roster for friendly matches vs Germany, Ghana, friendly matches vs Germany, Ghana, USA, Cameron Carter-Vickers, Sergino Dest, DeJuan Jones
</item 10>
=== END ITEM id=b560cc3acb2da477 ===

=== ITEM id=7713a6044516c7a6 ===
Gold sentences and facts:
Sentence 0: Research shows that even a small reduction in sleep has been shown to impact your immune system, however.
  Fact 0.0: Research shows that even a small reduction in sleep impacts the immune system.
Sentence 1: “This sex difference in insomnia emerges after puberty, suggesting that hormonal differences such as menstruation, pregnancy and menopause may be the cause of this increased risk in women,” Dr Browning says.
  Fact 1.0: Dr Browning says this sex difference in insomnia emerges after puberty.
  Fact 1.1: Dr Browning says the sex difference suggests hormonal differences may be the cause of the increased insomnia risk in women.
  Fact 1.2: Dr Browning names menstruation, pregnancy and menopause as examples of hormonal differences.
  Fact 1.3: Dr Browning says hormonal differences such as menstruation, pregnancy and menopause may cause this increased risk in women.

Retrieved items:
<item 1>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Lavender oil. Eating kiwis. Masturbation. Everyone has their own “hacks” when it comes to falling asleep in 2023; the days of counting sheep are long gone.
Jennifer Aniston’s top tip for battling insomnia is to “sleep with my phone at least five feet away from me”, while Martha Stewart recommends focusing on your
breathing.

Graph facts:
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Apple
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Getty Images
- Amazon’s latest Echo Frames are more style than substance -MENTIONS-> phone
- Dr Browning -SAID-> insomnia
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Reddit

Entities: Jennifer Aniston, Martha Stewart, Lavender oil, kiwis, Masturbation, phone, breathing, Lifestyle Edit email, insomnia, Sleepy tea, counting backwards
and dense earplugs: Inside the lives of insomniacs, The Independent
</item 1>

<item 2>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
In her 1973 feminist novel Fear of Flying, author Erica Jong notes that, sometimes, the only way to overcome sleeplessness is by pretending you don’t care about
nodding off. “Then sleep became piqued, like a rejected lover, and crept up to try to seduce you.”

In a world where human sleep patterns have increasingly been plunged into chaos, people are having to turn to ever more creative – and often bizarre – means of
getting their 40 winks in. Today, the internet is littered with tips and tricks that promise a sound sleep, but not all of them are effective and none are
foolproof. (Kiwis, by the way, contain serotonin and antioxidants, which is why it’s thought they might have sleep-promoting benefits.) The result is a society
with a growing problem on its hands: persistent sleeplessness – otherwise known as insomnia.

Graph facts:
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Apple
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Getty Images
- Dr Browning -SAID-> insomnia
- Here are over 40 last-minute gifts you can still nab before Christmas -MENTIONS-> internet
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Reddit

Entities: kiwis, Erica Jong, Fear of Flying, human sleep patterns, internet, serotonin, antioxidants, insomnia, Sleepy tea, counting backwards and dense
earplugs: Inside the lives of insomniacs, The Independent
</item 2>

<item 3>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
The Society has urged the UK government to adopt British Summer Time on a permanent basis.

“RoSPA is in favour of this proposal, and is calling for the government to adopt British Summer Time (GMT+1) all year,” the society states.

Dark, icy roads can be dangerous (Getty)

“This would mean road users will no longer experience the sudden onset of darkness during their autumn commutes, potentially saving many lives.”

And they’re not alone. The Royal Society for the Prevention of Accidents also supports the call to abolish DST, citing figures from the RAC Foundation which
state that road traffic collisions increase by 19 per cent in the fortnight after putting the clocks back one hour from British Summer Time, reducing by 11 per
cent when we put the clocks forward.

Increased immunity

Getting a good night’s sleep during the winter months is essential in protecting us against the seasonal increase in colds and flu. Research shows that even a
small reduction in sleep has been shown to impact your immune system, however.

Graph facts:
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> sleep
- Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ -MENTIONS-> sleep
- Introduction of British Summer Time -PART_OF-> British Summer Time
- UK government -PARTICIPATED_IN-> Introduction of British Summer Time
- What about the Mediterranean lifestyle reduces mortality? Scientists find out -MENTIONS-> sleep

Entities: The Royal Society for the Prevention of Accidents, UK government, British Summer Time, DST, RAC Foundation, road traffic collisions, colds and flu,
sleep, When do the clocks change in 2023?, The Independent, UK, Getty, immune system
</item 3>

<item 4>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
“One of the most important aspects of a healthy sleep schedule is waking up around the same time each morning, so try to limit yourself to a one-hour window if
you need to adjust when you get up in the morning and when you go to sleep at night. One hour sounds like an insignificant amount of time, but it can completely
throw our circadian rhythm, or more simply known as internal clock, off balance,” he says.

Despite this, a spokesperson for Sleep School, a science-based sleep education and therapy service, told The Independent that it is “impartial” on the extent to
which the clocks changing impacts sleep, arguing that the important factor is the duration of time asleep and a consistent schedule.

Fewer depressive episodes

A Danish study from 2017 of more than 185,000 people over 15 years found that the transition from summer time to standard time was associated with an 11 per
cent rise in depressive episodes. Conversely, the switch from winter to summer time was found to have no effect.

Graph facts:
- Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ -MENTIONS-> circadian rhythm
- Paula Desplats -SAID-> circadian rhythm
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> sleep
- Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ -MENTIONS-> sleep
- What about the Mediterranean lifestyle reduces mortality? Scientists find out -MENTIONS-> sleep

Entities: circadian rhythm, sleep, Sleep School, Danish study, depressive episodes, clocks changing, summer time, standard time, When do the clocks change in
2023?, The Independent
</item 4>

<item 5>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
The symptoms of insomnia can last a long time; a longitudinal study conducted in the UK found that 69 per cent of respondents affected by the sleep disorder at
the start of the study still had insomnia 12 months later.

This may go to show how complex treating insomnia can be. Explaining the appeal of quick fixes, Dr Browning says, “In the short term you’re changing something,
so [they] can often [work] like a placebo. If hunger is keeping you awake, eating before bed will help. If your mattress is uncomfortable, sleeping on the floor
will help.” But the problem isn’t just that these methods don’t necessarily work – it’s that they could be used to exploit vulnerable insomniacs.

“I’m a sleep therapist, my doctorate is in insomnia and it’s very, very frustrating,” she says. “People who have insomnia are desperate.

Graph facts:
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Getty Images
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Reddit
- Dr Browning -SAID-> Cognitive Behavioural Therapy for Insomnia
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Apple
- Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs -MENTIONS-> Google

Entities: insomnia, Dr Browning, longitudinal study, Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs, The Independent, UK
</item 5>

<item 6>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
The series was inspired by creator Natalie Walter’s own experience with insomnia, and her late-night Facebook chats with Steve Burge, her co-writer on Still Up.
“As part of the research for [the show] we visited a sleep clinic, which was fascinating,” she tells me. “Some evolutionary biologists think the reason humans
have different sleep patterns is that during our early history, a few members of the tribe would always be awake and vigilant during the night, and so could
protect everyone else from predators. Not that I’d be much use fighting off a woolly mammoth.”

Just like depression or anxiety, insomnia can often be triggered by something. Whether it’s a life event as monumental as becoming a new parent, or something as
commonplace as jet lag, there are any number of things that can prompt a bout of sleeplessness. The issue, explains sleep therapist Dr Lindsay Browning, is that
insomnia can often remain long after the trigger has gone.

Graph facts:
- A scientist reckons with climate grief -MENTIONS-> anxiety
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -MENTIONS-> anxiety
- Suicide prevention program at Nebraska children's hospital emphasizes encouraging, 'life-saving' notes -MENTIONS-> anxiety
- At 37, Alexandra found herself broke, heartbroken and living with her parents -MENTIONS-> anxiety
- 5 myths about ketamine, the drug tied to Matthew Perry's death, according to doctors -MENTIONS-> anxiety

Entities: Natalie Walter, Steve Burge, Lindsay Browning, jet lag, anxiety, insomnia, Sleepy tea, counting backwards and dense earplugs: Inside the lives of
insomniacs, Still Up, The Independent, sleep patterns, depression
</item 6>

<item 7>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Live your life healthier and happier with our free Living Well newsletter, packed with wellness advice, practical tips and nutritious recipes Live your life
healthier and happier with our free weekly Living Well newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to
be emailed about offers, events and updates from The Independent. Read our privacy notice Thanks for signing up to the

Living Well email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Winter is coming, as the clocks go backwards one hour at 2am on Sunday 29 October.

This will give us an extra hour in bed as we prepare for the colder months when the sun sets earlier and days get significantly shorter.

The clocks won’t go forward again until Sunday 31 March 2024.

Even so, the debate as to whether we should continue to comply with Daylight Saving Time (DST) protocols continues to run rife.

Graph facts:
- When do the clocks change in 2023? -MENTIONS-> carbon emissions
- When do the clocks change in 2023? -MENTIONS-> Aoife Foley
- When do the clocks change in 2023? -MENTIONS-> circadian rhythm
- When do the clocks change in 2023? -MENTIONS-> public safety
- When do the clocks change in 2023? -MENTIONS-> Queen’s University Belfast

Entities: Living Well newsletter, Daylight Saving Time (DST), Clocks go backwards one hour, Clocks go forward, When do the clocks change in 2023?, The
Independent
</item 7>

<item 8>
Graph query row:
article: When do the clocks change in 2023?
published_on: 2023-10-24T05:50:17+00:00
relationship: MENTIONS
related_entity: immune system
</item 8>

<item 9>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Conversely, the switch from winter to summer time was found to have no effect.

People frequently report an increase in SAD and depressive symptoms in winter (Getty/iStock)

“This study shows that the transition from summer time to standard time was associated with an increase in the incidence rate of unipolar depressive episodes,”
the study authors concluded. “Distress associated with the sudden advancement of sunset, marking the coming of a long period of short days, may explain this
finding.”

A drop in heart attacks

Not everyone is convinced, however. Charlie Morley, author of Wake Up to Sleep, cites a 2014 study by the College of Cardiology, which reported a significant
drop in heart attacks in the 24 hours immediately following turning back the clocks.

“Fascinatingly, each year when the clocks go back for Daylight Saving and 1.6 billion people across 70 countries gain an extra hour in bed, there is a 21 per
cent decrease in heart attacks the next day,” he says.

Graph facts:
- When do the clocks change in 2023? -MENTIONS-> carbon emissions
- When do the clocks change in 2023? -MENTIONS-> Living Well newsletter
- When do the clocks change in 2023? -MENTIONS-> circadian rhythm
- When do the clocks change in 2023? -MENTIONS-> public safety
- When do the clocks change in 2023? -MENTIONS-> Daylight Saving Time (DST)

Entities: Charlie Morley, Wake Up to Sleep, College of Cardiology, 2014 study, heart attacks, SAD, depressive symptoms, unipolar depressive episodes, Daylight
Saving, transition from summer time to standard time, When do the clocks change in 2023?, The Independent
</item 9>

<item 10>
When do the clocks change in 2023? (The Independent - Life and Style, 2023-10-24T05:50:17+00:00)
Research shows that even a small reduction in sleep has been shown to impact your immune system, however.

Making the move from daylight saving time interferes with your natural circadian rhythm, the 24-hour biological cycle that is influenced by morning light and
evening darkness, however.

“It’s hard to understate the importance of a good night’s sleep,” says Giulia Guerrini, lead pharmacist of digital pharmacy Medino. “Having too much or too
little sleep can have adverse side effects, so you should aim to get between seven and eight hours of sleep as consistently as possible.”

Never underestimate the value of a good night’s sleep (Getty)

“One of the biggest disruptors to our sleep is when UK clocks go forward one hour at 1am on the last Sunday in March and go back one hour at 2am on the last
Sunday in October, which is rapidly approaching on 30 October.

Graph facts:
- Royal Society for the Prevention of Accidents -SAID-> daylight saving time
- Paula Desplats -SAID-> circadian rhythm
- Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ -MENTIONS-> circadian rhythm
- European parliament vote to scrap DST -MENTIONS-> daylight saving time
- The best Warlock subclass, feats, and build in Baldur’s Gate 3 -MENTIONS-> sleep

Entities: circadian rhythm, daylight saving time, Giulia Guerrini, Medino, sleep, UK clocks go back, When do the clocks change in 2023?, The Independent, UK,
immune system
</item 10>
=== END ITEM id=7713a6044516c7a6 ===
