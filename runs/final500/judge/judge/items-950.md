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

=== ITEM id=d0925dacde99ffa0 ===
Gold sentences and facts:
Sentence 0: The Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League ( NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their fourth NFL title , after winning in 1948 , 1949 , and 1960 .
  Fact 0.0: The Philadelphia Eagles are a professional American football franchise.
  Fact 0.1: The Philadelphia Eagles are based in Philadelphia, Pennsylvania.
  Fact 0.2: The Philadelphia Eagles compete in the National Football League (NFL).
  Fact 0.3: The Philadelphia Eagles are a member club of the league's National Football Conference (NFC) East division.
  Fact 0.4: The Philadelphia Eagles are Super Bowl champions.
  Fact 0.5: The Philadelphia Eagles won Super Bowl LII.
  Fact 0.6: Super Bowl LII was the Eagles' fourth NFL title.
  Fact 0.7: The Eagles had won NFL titles in 1948, 1949, and 1960 before Super Bowl LII.
Sentence 1: Philadelphia Eagles Current season Established July 8 , 1933 ; 84 years ago ( July 8 , 1933 ) First season : 1933 Play in Lincoln Financial Field Philadelphia , Pennsylvania Headquartered in the NovaCare Complex Philadelphia , Pennsylvania Logo Wordmark League / conference affiliations National Football League ( 1933 -- present ) Eastern Division ( 1933 -- 1949 ) American Conference ( 1950 -- 1952 ) Eastern Conference ( 1953 -- 1969 ) Capitol Division ( 1967 -- 1969 ) National Football Conference ( 1970 -- present ) NFC East ( 1970 -- present ) Current uniform Team colors Midnight green , Silver , Black , White Fight song Fly , Eagles Fly Mascot Swoop Personnel Owner ( s ) Jeffrey Lurie Christina Weiss Lurie Chairman Jeffrey Lurie CEO Jeffrey Lurie President Don Smolenski General manager Howie Roseman Head coach Doug Pederson Team history Philadelphia Eagles ( 1933 -- 1942 ; 1944 -- present ) Phil - Pitt `` Steagles '' ( 1943 ) Team nicknames The Birds Championships League championships ( 4 ) NFL championships ( pre-1970 AFL -- NFL merger ) ( 3 ) 1948 , 1949 , 1960 Super Bowl championships ( 1 ) 2017 ( LII ) Conference championships ( 4 ) NFL Eastern : 1960 NFC : 1980 , 2004 , 2017 Division championships ( 13 ) NFL East : 1947 , 1948 , 1949 NFC East : 1980 , 1988 , 2001 , 2002 , 2003 , 2004 , 2006 , 2010 , 2013 , 2017 Playoff appearances ( 25 ) NFL : 1947 , 1948 , 1949 , 1960 , 1978 , 1979 , 1980 , 1981 , 1988 , 1989 , 1990 , 1992 , 1995 , 1996 , 2000 , 2001 , 2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 ) Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present )
  Fact 1.0: The Philadelphia Eagles were established on July 8, 1933.
  Fact 1.1: The Eagles' first season was 1933.
  Fact 1.2: The Eagles play at Lincoln Financial Field in Philadelphia, Pennsylvania.
  Fact 1.3: The Eagles are headquartered in the NovaCare Complex in Philadelphia, Pennsylvania.
  Fact 1.4: The Eagles' team colors are midnight green, silver, black, and white.
  Fact 1.5: The Eagles' fight song is 'Fly, Eagles Fly'.
  Fact 1.6: The Eagles' mascot is Swoop.
  Fact 1.7: The Eagles are owned by Jeffrey Lurie and Christina Weiss Lurie.
  Fact 1.8: Jeffrey Lurie is the chairman of the Eagles.
  Fact 1.9: Jeffrey Lurie is the CEO of the Eagles.
  Fact 1.10: Don Smolenski is the president of the Eagles.
  Fact 1.11: Howie Roseman is the general manager of the Eagles.
  Fact 1.12: Doug Pederson is the head coach of the Eagles.
  Fact 1.13: The Eagles are nicknamed The Birds.
  Fact 1.14: The Eagles have been in the National Football League since 1933.
  Fact 1.15: The Eagles were in the Eastern Division from 1933 to 1949.
  Fact 1.16: The Eagles were in the American Conference from 1950 to 1952.
  Fact 1.17: The Eagles were in the Eastern Conference from 1953 to 1969.
  Fact 1.18: The Eagles were in the Capitol Division from 1967 to 1969.
  Fact 1.19: The Eagles have been in the National Football Conference since 1970.
  Fact 1.20: The Eagles have been in the NFC East since 1970.
  Fact 1.21: The Philadelphia Eagles team existed from 1933 to 1942 and from 1944 to the present.
  Fact 1.22: The Phil-Pitt 'Steagles' team existed in 1943.
  Fact 1.23: The Eagles have won 4 league championships.
  Fact 1.24: The Eagles have won 3 NFL championships before the AFL-NFL merger.
  Fact 1.25: The Eagles won NFL championships in 1948, 1949, and 1960.
  Fact 1.26: The Eagles have won 1 Super Bowl championship.
  Fact 1.27: The Eagles won Super Bowl LII in 2017.
  Fact 1.28: The Eagles have won 4 conference championships.
  Fact 1.29: The Eagles won the NFL Eastern conference championship in 1960.
  Fact 1.30: The Eagles won NFC conference championships in 1980, 2004, and 2017.
  Fact 1.31: The Eagles have won 13 division championships.
  Fact 1.32: The Eagles won NFL East division championships in 1947, 1948, and 1949.
  Fact 1.33: The Eagles won NFC East division championships in 1980, 1988, 2001, 2002, 2003, 2004, 2006, 2010, 2013, and 2017.
  Fact 1.34: The Eagles have made 25 playoff appearances.
  Fact 1.35: The Eagles made NFL playoff appearances in 1947, 1948, 1949, 1960, 1978, 1979, 1980, 1981, 1988, 1989, 1990, 1992, 1995, 1996, 2000, 2001, 2002, 2003, 2004, 2006, 2008, 2009, 2010, 2013, and 2017.
  Fact 1.36: The Eagles played home games at Baker Bowl from 1933 to 1935.
  Fact 1.37: The Eagles played home games at Philadelphia Municipal Stadium in 1936 to 1939 and in 1941.
  Fact 1.38: The Eagles played home games at Connie Mack Stadium in 1940 and from 1942 to 1957.
  Fact 1.39: The Eagles played home games at Franklin Field from 1958 to 1970.
  Fact 1.40: The Eagles played home games at Veterans Stadium from 1971 to 2002.
  Fact 1.41: The Eagles have played home games at Lincoln Financial Field since 2003.

Retrieved items:
<item 1>
The Philadelphia Eagles, nicknamed “The Birds,” have competed since 1933, except as the Phil-Pitt “Steagles” in 1943. They belong to the NFL’s NFC East. Their
championships include NFL titles in 1948, 1949, and 1960, and Super Bowl LII in 2017. They have won four conference titles, 13 division titles, and made 25
playoff appearances. Retired numbers include 5, 15,
</item 1>

<item 2>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 2>

<item 3>
The Philadelphia Eagles’ history features notable moments including the Fog Bowl, “4th and 26,” the Philly Special, and Super Bowl XXXIX. Their rivals include
the Cowboys, Giants, Redskins, and Steelers. They have 13 division, four conference, and four league championships, including Super Bowl LII in 2017. Retired
numbers include 5, 15, 20, 40, 44, 60, 70, 92, and 99. They
</item 3>

<item 4>
Doug Pederson
Despite becoming major playoff underdogs due to the loss of Wentz , Foles filled in admirably as the starter , allowing Philadelphia to make it to Super Bowl
LII , their first Super Bowl appearance since the 2004 season . Eventual Super Bowl MVP Foles led the team in a 41 - 33 win over the New England Patriots ,
giving them their first Lombardi Trophy in franchise history and their first league championship since 1960 . Head coaching record ( edit ) Team Year Regular
season Postseason Won Lost Ties Win % Finish Won Lost Win % Result PHI 2016 7 9 0 . 438 4th in NFC East -- -- -- -- PHI 2017 13 0 . 813 1st in NFC East 0 1.000
Won Super Bowl LII Total 20 12 0 .
</item 4>

<item 5>
Philadelphia Eagles
This would be Philadelphia 's first Super Bowl win and their fourth league championship ( their first championship since 1960 ) . Foles won Super Bowl MVP going
28 -- 43 with 373 passing yards , 3 touchdowns , and an interception . Nick Foles became the first backup quarterback to start and win a Super Bowl since his
opponent Tom Brady won as the backup for Drew Bledsoe in 2001 .
</item 5>

<item 6>
Super Bowl XXXIX
The Eagles were trying to win their first NFL title since 1960 and the first championship for the city of Philadelphia since Moses Malone 's `` fo ' , fi ' , fo
' '' during the 76ers run to the 1983 NBA Championship . With President George W. Bush being inaugurated for a second time in January , the Eagles were also
trying to end a losing streak -- teams in the city of Philadelphia had lost six straight championships during presidential inauguration years , beginning with
the 76ers loss in 1977 . The streak included the Eagles in Super Bowl XV in 1981 . 1989 was not included in that streak , but 1977 , 1981 , 1985 ( Flyers ) ,
1993 ( Phillies ) , 1997 ( Flyers ) , and 2001 ( 76ers ) were . Television and entertainment ( edit ) The game was televised in the United States by Fox , with
play - by - play announcer Joe Buck and color commentators Cris Collinsworth and Troy Aikman . Pam Oliver ( Patriots sideline ) and Chris Myers ( Eagles
sideline ) served as sideline reporters .
</item 6>

<item 7>
New England (14–2) defeated Philadelphia (13–3) 24–21 in Super Bowl XXXIX, earning consecutive titles and its third championship in four years. After a 14–14
tie through three quarters, Corey Dillon’s touchdown and Adam Vinatieri’s field goal put the Patriots ahead. Donovan McNabb’s late touchdown pass to Greg Lewis
narrowed the gap, but Philadelphia’s comeback fell short. New England forced four turnovers; MVP Deion Branch had 133 yards and a
</item 7>

<item 8>
On February 6, 2005, the New England Patriots defeated the Philadelphia Eagles 24–21 at Jacksonville’s Alltel Stadium, earning their third Super Bowl title in
four seasons. After a 14–14 tie entering the fourth quarter, Corey Dillon scored and Adam Vinatieri added a field goal. Donovan McNabb’s late touchdown to Greg
Lewis brought Philadelphia within three, but the Patriots held on. McNabb threw three interceptions; Tom Brady passed for 236 yards and two touchdowns.
</item 8>

<item 9>
Super Bowl XXXIX
( 1960 -- 1969 ) Seasons ( 58 ) 1960 1961 1962 1963 1964 1965 1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988
1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Championship seasons in
bold Philadelphia Eagles Founded in 1933 Based and headquartered in Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players
Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop
Curse of Billy Penn The Lombardi Curse Invincible Philadelphia Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders
</item 9>

<item 10>
Philadelphia Eagles
In 2003 , they won the NFC first seed , but Westbrook went down in Week 17 , culminating in a loss to the Carolina Panthers in their 3rd straight NFC
Championship . 2004 season and Super Bowl XXXIX In 2004 , the Philadelphia Eagles had their best season since 1960 , going 13 -- 1 before resting their starters
and losing their next 2 , clinching the 1st seed for the second year in a row . McNabb set career highs , completing 64 % of his passes for 3,875 yards , though
he did n't play all 16 games . McNabb became the first quarterback ever to throw more than 30 touchdowns and fewer than 10 interceptions in a season . His
success could be attributed to the fact that he had a reliable receiver , Terrell Owens , who got 1,200 yards and 14 touchdowns in 14 games . After beating the
Vikings and Falcons the Eagles advanced to Super Bowl XXXIX , where they dueled the New England Patriots .
</item 10>
=== END ITEM id=d0925dacde99ffa0 ===

=== ITEM id=25d1a15fec1a8cc6 ===
Gold sentences and facts:
Sentence 0: The Raiders had a great chance to score a touchdown early in the game after cornerback Charles Woodson intercepted Buccaneers quarterback Brad Johnson 's pass on the third play of the game and returned it 12 yards to the Tampa Bay 36 - yard line . However , six plays later , Tampa Bay defensive end Simeon Rice sacked Raiders quarterback Rich Gannon on third down , forcing Oakland to settle for kicker Sebastian Janikowski 's 40 - yard field goal to give them a 3 -- 0 lead .
  Fact 0.0: Cornerback Charles Woodson intercepted a pass thrown by Buccaneers quarterback Brad Johnson.
  Fact 0.1: The interception happened on the third play of the game.
  Fact 0.2: Charles Woodson returned the interception 12 yards to the Tampa Bay 36-yard line.
  Fact 0.3: The Raiders had a great chance to score a touchdown early in the game after the interception.
  Fact 0.4: Six plays after the interception, Tampa Bay defensive end Simeon Rice sacked Raiders quarterback Rich Gannon on third down.
  Fact 0.5: The sack forced Oakland to settle for a field goal.
  Fact 0.6: Kicker Sebastian Janikowski kicked a 40-yard field goal for Oakland.
  Fact 0.7: The field goal gave Oakland a 3-0 lead.
Sentence 1: Still , even Gruden had trouble getting the offense in sync during his first year as Tampa Bay 's head coach . In 2002 , the Buccaneers ranked 25th in the league in total yards gained ( 5,222 ) . Quarterback Brad Johnson made the Pro Bowl , completing 281 out of 451 passes for 3,049 yards , 22 touchdowns , and only 6 interceptions . Running back Michael Pittman led the team in rushing with 718 yards and one touchdown , and caught 59 passes for 477 yards . Pro Bowl fullback Mike Alstott had 548 rushing yards and 5 touchdowns , and also had 35 receptions for 242 yards and 2 touchdowns . Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5 touchdowns , while wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns .
  Fact 1.0: Even Gruden had trouble getting the offense in sync during his first year as Tampa Bay's head coach.
  Fact 1.1: In 2002, the Buccaneers ranked 25th in the league in total yards gained.
  Fact 1.2: The Buccaneers gained 5,222 total yards in 2002.
  Fact 1.3: Quarterback Brad Johnson made the Pro Bowl.
  Fact 1.4: Brad Johnson completed 281 of 451 passes for 3,049 yards.
  Fact 1.5: Brad Johnson threw 22 touchdowns.
  Fact 1.6: Brad Johnson threw only 6 interceptions.
  Fact 1.7: Running back Michael Pittman led the Buccaneers in rushing with 718 yards and one touchdown.
  Fact 1.8: Michael Pittman caught 59 passes for 477 yards.
  Fact 1.9: Pro Bowl fullback Mike Alstott had 548 rushing yards and 5 touchdowns.
  Fact 1.10: Mike Alstott had 35 receptions for 242 yards and 2 touchdowns.
  Fact 1.11: Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5 touchdowns.
  Fact 1.12: Wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns.

Retrieved items:
<item 1>
Super Bowl XXXVII
But that changed when the team hired Tony Dungy as head coach in 1996 . Dungy , along with his defensive coordinator Monte Kiffin , rebuilt the defense around a
core group of players , such as defensive lineman Warren Sapp , linebacker Derrick Brooks , and defensive backs Ronde Barber and John Lynch . By 1997 , Tampa
Bay ranked 3rd in the league in total yards allowed , 2nd in 1998 , and 3rd in 2000 . With one of the best defenses in the league , Dungy helped guide the
Buccaneers to four playoff appearances in his six years as the team 's head coach . But the team always had one of the worst offenses in the league , and this
was a major factor in their playoff losses . Thus , Dungy was fired and replaced by Gruden . Still , even Gruden had trouble getting the offense in sync during
his first year as Tampa Bay 's head coach . In 2002 , the Buccaneers ranked 25th in the league in total yards gained ( 5,222 ) .
</item 1>

<item 2>
Super Bowl XXXVII
Better still for the Buccaneers was that Oakland had n't changed their audible - calling signals that Gruden himself had installed , thus tipping off plays
repeatedly . Johnson finished the game with 18 out of 34 completions for 215 yards and 2 touchdowns , with 1 interception , along with 10 rushing yards .
Pittman was the top rusher of the game with 129 yards . Alstott was the game 's second leading rusher with 15 yards and a touchdown , and had 5 receptions for
43 yards . Wide receiver Keyshawn Johnson was the Buccaneers ' leading receiver with 6 catches for 69 yards . Smith recorded 2 interceptions , 94 return yards ,
and 2 touchdowns . He also added another 23 yards on a kickoff return . Gannon finished the game 24 out of 44 for 272 yards and 2 touchdowns , but was
intercepted a Super Bowl record 5 times . Garner was their leading rusher , but with only 10 yards , and caught 7 passes for 51 yards . Rice was the Raiders '
leading receiver of the game with 5 catches for 77 yards and a touchdown .
</item 2>

<item 3>
Super Bowl XXXVII
Aided with Pittman 's gains of 6 and 19 yards , the Buccaneers scored their first touchdown on a 2 - yard run from fullback Mike Alstott , increasing their lead
to 13 -- 3 . Then with 3 : 45 left in the half , Tampa Bay drove 77 yards , assisted by a pair of catches by Alstott for 28 total yards . Johnson finished the
drive with a 5 - yard touchdown pass to wide receiver Keenan McCardell to give the Buccaneers a 20 -- 3 halftime lead . Third quarter ( edit ) Tampa Bay
continued to dominate the game for most of the third quarter . The Buccaneers forced the Raiders to punt on the opening drive of the second half . Next , Tampa
Bay marched 89 yards on a 14 - play drive that took 7 : 52 off the clock , and ended with Johnson 's 8 - yard touchdown pass to McCardell to increase their lead
to 27 -- 3 .
</item 3>

<item 4>
Super Bowl XXXVII
Quarterback Brad Johnson made the Pro Bowl , completing 281 out of 451 passes for 3,049 yards , 22 touchdowns , and only 6 interceptions . Running back Michael
Pittman led the team in rushing with 718 yards and one touchdown , and caught 59 passes for 477 yards . Pro Bowl fullback Mike Alstott had 548 rushing yards and
5 touchdowns , and also had 35 receptions for 242 yards and 2 touchdowns . Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5
touchdowns , while wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns . However , the Buccaneers ' defense was still the strength
of the team , leading the NFL in total defense ( 252.8 yards per game ) , pass defense ( 155.6 yards per game ) , points allowed ( 12.3 points per game ) ,
passing touchdowns allowed ( 10 ) , interceptions ( 31 ) , and opponent passer rating ( 48.4 ) . Brooks , Lynch , Sapp , and defensive end Simeon Rice all had
Pro Bowl years .
</item 4>

<item 5>
Super Bowl XXXVII
January 15 , 2003 . Retrieved August 27 , 2012 . Jump up ^ Mitchell , Fred ( January 27 , 2003 ) . `` In their fashion , Bucs fans laugh '' . Chicago Tribune .
Retrieved August 27 , 2012 . Jump up ^ Vivinetto , Gina ( January 27 , 2003 ) . `` Super Bowl XXXVII : Superstars belt out favorites , or lip synch them '' .
St. Petersburg Times . ^ Jump up to : George , Thomas ( January 27 , 2003 ) . `` Bucs steal the treasure '' . The New York Times . Retrieved August 27 , 2012 .
^ Jump up to : `` Tampa Bay Buccaneers 48 vs. Oakland Raiders 21 '' . pro-football - reference . Retrieved August 27 , 2012 . Jump up ^ Cunningham , Michael ;
Skolnick , Ethan J. ( January 27 , 2003 ) . `` Treu Gets The Call In Robbins ' Absence '' . South Florida Sun - Sentinel .
</item 5>

<item 6>
Tampa Bay defeated Oakland 48–21 in Super Bowl XXXVII on January 26, 2003, in sunny San Diego. Charles Woodson’s early interception set up Sebastian
Janikowski’s 40-yard field goal, but Tampa Bay rallied and dominated. The Buccaneers had 24 first downs and 150 rushing yards, compared with Oakland’s 11 and
19. The teams scored 46 second-half points, a Super Bowl record, and 69 total points, tying the
</item 6>

<item 7>
The Tampa Bay Buccaneers, founded in 1976, won the NFC South six times, the NFC title and Super Bowl XXXVII in 2002. Hall of Famers include Lee Roy Selmon,
Steve Young, Warren Sapp and Derrick Brooks; numbers 55, 63 and 99 are retired. The Oakland Raiders, founded in 1960, played in Los Angeles from 1982–94 and won
three Super Bowls (1976, 1980, 198
</item 7>

<item 8>
Super Bowl XXXVII
Then on third and 18 from the Oakland 29 - yard line , Tampa Bay linebacker Derrick Brooks intercepted a pass from Gannon and returned it 44 yards for a
touchdown , giving the Buccaneers a 41 -- 21 lead with only 1 : 18 left , and leading Buccaneers radio announcer Gene Deckerhoff to make his famous call of ``
The dagger 's in , we 're gon na win the Super Bowl ! '' . A few plays later , with the Raiders now playing for pride , Dwight Smith intercepted a tipped pass
and returned it 50 yards for a touchdown with only two seconds remaining . Gramatica kicked the extra point , and the Buccaneers were up 48 -- 21 . Raiders
defensive tackle Chris Cooper returned Gramatica 's kickoff 6 yards before being tackled by Jack Golden , ending the game . The Buccaneers won their first -
ever Super Bowl with a 48 - 21 victory . Gannon said after the game that his performance was `` nightmarish '' .
</item 8>

<item 9>
The Buccaneers defeated the Raiders 48–21 in Super Bowl XXXVII. Tampa Bay dominated possession (37:14–22:46) and scored five interceptions, returning them for
172 yards; Oakland had five turnovers. Raiders QB Rich Gannon threw for 272 yards but was intercepted five times. Buccaneers QB Brad Johnson passed for 215
yards, while Michael Pittman rushed for 124. Jerry Rice led Oakland with 77 receiving yards; Keyshawn Johnson led Tampa Bay with
</item 9>

<item 10>
On January 26, 2003, Tampa Bay beat Oakland 48–21 in Super Bowl XXXVII, winning its first championship. The Buccaneers’ top-ranked defense intercepted Rich
Gannon five times, returning three picks for touchdowns, and held Oakland to 19 rushing yards. Derrick Brooks and Dwight Smith scored interception-return
touchdowns; Dexter Jackson, who intercepted Gannon twice, was named MVP. The game was nicknamed the “Gruden Bowl” after Jon Gruden moved from Oakland to Tampa
Bay
</item 10>
=== END ITEM id=25d1a15fec1a8cc6 ===

=== ITEM id=476753499ba774cc ===
Gold sentences and facts:
Sentence 0: Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait * Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org
  Fact 0.0: The Organization of the Petroleum Exporting Countries (OPEC) has its headquarters in Vienna, Austria.
  Fact 0.1: OPEC's official language is English.
  Fact 0.2: OPEC is an international cartel.
  Fact 0.3: OPEC has 14 member states as of May 2017.
  Fact 0.4: The OPEC member states listed are Algeria, Angola, Ecuador, Equatorial Guinea, Gabon, Iran, Iraq, Kuwait, Libya, Nigeria, Qatar, Saudi Arabia, the United Arab Emirates, and Venezuela.
  Fact 0.5: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela are marked with an asterisk, which the table defines as founding member.
  Fact 0.6: The OPEC leader is Secretary General Mohammed Barkindo.
  Fact 0.7: OPEC was established in Baghdad, Iraq.
  Fact 0.8: The OPEC statute is dated September 1960 and was in effect from January 1961.
  Fact 0.9: OPEC's prices are indexed as US dollars per barrel (US$/bbl).
  Fact 0.10: OPEC's website is OPEC.org.
Sentence 1: Organization of the Petroleum Exporting Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria . As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies .
  Fact 1.0: OPEC is an intergovernmental organization of 14 nations as of February 2018.
  Fact 1.1: OPEC is pronounced OH-pek (IPA: ˈoʊpɛk), or OPEP in several other languages.
  Fact 1.2: OPEC was founded in 1960 in Baghdad by its first five members: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela.
  Fact 1.3: OPEC has been headquartered in Vienna, Austria since 1965.
  Fact 1.4: As of 2016, the 14 OPEC countries accounted for an estimated 44 percent of global oil production.
  Fact 1.5: As of 2016, the 14 OPEC countries held an estimated 73 percent of the world's 'proven' oil reserves.
  Fact 1.6: OPEC has a major influence on global oil prices.
  Fact 1.7: Global oil prices were previously determined by American-dominated multinational oil companies.
Sentence 2: Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa 1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 -- 1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa 1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,000 Iraq Middle East 1960 -- 37,202,572 437,072 4,451,516 143,069,000,000 Kuwait Middle East 1960 -- 4,052,584 17,820 2,923,825 101,500,000,000 Libya North Africa 1962 -- 6,293,253 1,759,540 384,686 48,363,000,000 Nigeria West Africa 1971 -- 185,989,640 923,768 1,999,885 37,070,000,000 Qatar Middle East 1961 -- 2,569,804 11,437 1,522,902 25,244,000,000 Saudi Arabia Middle East 1960 -- 32,275,687 2,149,690 10,460,710 266,578,000,000 United Arab Emirates Middle East 1967 -- 9,269,612 83,600 3,106,077 97,800,000,000 Venezuela South America 1960 -- 31,568,179 912,050 2,276,967 299,953,000,000 OPEC Total 478,498,000 12,150,695 35,221,740 1,209,103,000,000 World Total 7,618,511,000 510,072,000 80,622,287 1,650,585,000,000 OPEC Percent 6 % 2 % 44 % 73 %
  Fact 2.0: Algeria (North Africa): membership from 1969; population 40,606,052 (2016 est.); area 2,381,740 km²; oil production 1,348,361 bbl/day (2016); proven reserves 12,200,000,000 bbl (2016).
  Fact 2.1: Angola (Southern Africa): membership from 2007; population 28,813,463; area 1,246,700 km²; oil production 1,769,615 bbl/day; proven reserves 8,423,000,000 bbl.
  Fact 2.2: Ecuador (South America): membership 1973 to 1992 and from 2007; population 16,385,068; area 283,560 km²; oil production 548,421 bbl/day; proven reserves 8,273,000,000 bbl.
  Fact 2.3: Equatorial Guinea (Central Africa): membership from 2017; population 1,221,490; area 28,051 km²; oil production 227,000 bbl/day; proven reserves 1,100,000,000 bbl.
  Fact 2.4: Gabon (Central Africa): membership 1975 to 1995 and from 2016; population 1,979,786; area 267,667 km²; oil production 210,820 bbl/day; proven reserves 2,000,000,000 bbl.
  Fact 2.5: Iran (Middle East): membership from 1960; population 80,277,428; area 1,648,000 km²; oil production 3,990,956 bbl/day; proven reserves 157,530,000,000 bbl.
  Fact 2.6: Iraq (Middle East): membership from 1960; population 37,202,572; area 437,072 km²; oil production 4,451,516 bbl/day; proven reserves 143,069,000,000 bbl.
  Fact 2.7: Kuwait (Middle East): membership from 1960; population 4,052,584; area 17,820 km²; oil production 2,923,825 bbl/day; proven reserves 101,500,000,000 bbl.
  Fact 2.8: Libya (North Africa): membership from 1962; population 6,293,253; area 1,759,540 km²; oil production 384,686 bbl/day; proven reserves 48,363,000,000 bbl.
  Fact 2.9: Nigeria (West Africa): membership from 1971; population 185,989,640; area 923,768 km²; oil production 1,999,885 bbl/day; proven reserves 37,070,000,000 bbl.
  Fact 2.10: Qatar (Middle East): membership from 1961; population 2,569,804; area 11,437 km²; oil production 1,522,902 bbl/day; proven reserves 25,244,000,000 bbl.
  Fact 2.11: Saudi Arabia (Middle East): membership from 1960; population 32,275,687; area 2,149,690 km²; oil production 10,460,710 bbl/day; proven reserves 266,578,000,000 bbl.
  Fact 2.12: United Arab Emirates (Middle East): membership from 1967; population 9,269,612; area 83,600 km²; oil production 3,106,077 bbl/day; proven reserves 97,800,000,000 bbl.
  Fact 2.13: Venezuela (South America): membership from 1960; population 31,568,179; area 912,050 km²; oil production 2,276,967 bbl/day; proven reserves 299,953,000,000 bbl.
  Fact 2.14: OPEC total: population 478,498,000; area 12,150,695 km²; oil production 35,221,740 bbl/day; proven reserves 1,209,103,000,000 bbl.
  Fact 2.15: World total: population 7,618,511,000; area 510,072,000 km²; oil production 80,622,287 bbl/day; proven reserves 1,650,585,000,000 bbl.
  Fact 2.16: OPEC percent of the world: population 6%; area 2%; oil production 44%; proven reserves 73%.

Retrieved items:
<item 1>
OPEC
A statement released by OPEC on 10 September 2008 confirmed Indonesia 's withdrawal , noting that OPEC `` regretfully accepted the wish of Indonesia to suspend
its full membership in the organization , and recorded its hope that the country would be in a position to rejoin the organization in the not - too - distant
future . '' 2008 production dispute ( edit ) Countries by net oil exports ( 2008 ) The differing economic needs of OPEC member states often affect the internal
debates behind OPEC production quotas . Poorer members have pushed for production cuts from fellow members , to increase the price of oil and thus their own
revenues . These proposals conflict with Saudi Arabia 's stated long - term strategy of being a partner with the world 's economic powers to ensure a steady
flow of oil that would support economic expansion . Part of the basis for this policy is the Saudi concern that overly expensive oil or unreliable supply will
drive industrial nations to conserve energy and develop alternative fuels , curtailing the worldwide demand for oil and eventually leaving unneeded barrels in
the ground . To this point , Saudi Oil Minister Yamani famously remarked in 1973 : `` The Stone Age did n't end because we ran out of stones . ''
</item 1>

<item 2>
OPEC
As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving
OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies . OPEC 's stated mission is ``
to coordinate and unify the petroleum policies of its member countries and ensure the stabilization of oil markets , in order to secure an efficient , economic
and regular supply of petroleum to consumers , a steady income to producers , and a fair return on capital for those investing in the petroleum industry . ''
The organization is also a significant provider of information about the international oil market . As of May 2017 , OPEC 's members are Algeria , Angola ,
Ecuador , Equatorial Guinea , Gabon , Iran , Iraq , Kuwait , Libya , Nigeria , Qatar , Saudi Arabia ( the de facto leader ) , United Arab Emirates , and
Venezuela , while Indonesia is a former member . Two - thirds of OPEC 's oil production and reserves are in its six Middle Eastern countries that surround the
oil - rich Persian Gulf .
</item 2>

<item 3>
OPEC
Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa
1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 --
1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa
1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,
</item 3>

<item 4>
OPEC
In May 2008 , Indonesia announced that it would leave OPEC when its membership expired at the end of that year , having become a net importer of oil and being
unable to meet its production quota . It rejoined the organization in January 2016 , but announced another `` temporary suspension '' of its membership at year
- end when OPEC requested a 5 percent production cut . Some commentators consider that the United States was a de facto member of OPEC during its formal
occupation of Iraq , due to its leadership of the Coalition Provisional Authority in 2003 -- 2004 . But this is not borne out by the minutes of OPEC meetings ,
as no US representative attended in an official capacity . Observers ( edit ) Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia ,
and other oil - exporting nations have attended many OPEC meetings as observers . This arrangement serves as an informal mechanism for coordinating policies .
</item 4>

<item 5>
OPEC
Jump up ^ The UAE was founded in December 1971 . Its OPEC membership originated with the Emirate of Abu Dhabi . Lapsed members ( edit ) Country Region
Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Indonesia Southeast Asia 1962 -- 2008
, Jan - Nov 2016 261,115,456 1,904,569 833,667 3,692,500,000 For countries that export petroleum at relatively low volume , their limited negotiating power as
OPEC members would not necessarily justify the burdens imposed by OPEC production quotas and membership costs . Ecuador withdrew from OPEC in December 1992 ,
because it was unwilling to pay the annual US $2 million membership fee and felt that it needed to produce more oil than it was allowed under its OPEC quota at
the time , although it rejoined in October 2007 . Similar concerns prompted Gabon to suspend membership in January 1995 ; it rejoined in July 2016 .
</item 5>

<item 6>
OPEC
269,612 83,600 3,106,077 97,800,000,000 Venezuela South America 1960 -- 31,568,179 912,050 2,276,967 299,953,000,000 OPEC Total 478,498,000 12,150,695
35,221,740 1,209,103,000,000 World Total 7,618,511,000 510,072,000 80,622,287 1,650,585,000,000 OPEC Percent 6 % 2 % 44 % 73 % ^ Jump up to : One petroleum
barrel ( bbl ) is approximately 42 U.S. gallons , or 159 liters , or 0.159 m , varying slightly with temperature . To put the production numbers in context , a
supertanker typically holds 2,000,000 barrels ( 320,000 m ) , and the world 's current production rate would take approximately 56 years to exhaust the world 's
current proven reserves . ^ Jump up to : The five founding members attended the first OPEC conference in September 1960 .
</item 6>

<item 7>
OPEC
OPEC - wikipedia OPEC Jump to : navigation , search Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official
language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait *
Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq
Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org Organization of the Petroleum Exporting
Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded
in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria .
</item 7>

<item 8>
OPEC
9 2003 -- 2011 volatility 3.10 2008 production dispute 3.11 2014 -- 2017 oil glut 3.12 2017 - 2018 Production cut 4 See also 5 References 6 External links
Membership ( edit ) Current member countries ( edit ) As of May 2017 , OPEC has 14 member countries : six in the Middle East ( Western Asia ) , six in Africa ,
and two in South America . According to the U.S. Energy Information Administration ( EIA ) , OPEC 's combined rate of oil production ( including gas condensate
) represented 44 percent of the world 's total in 2016 , and OPEC accounted for 73 percent of the world 's `` proven '' oil reserves , including 48 percent from
just the six Middle Eastern members : Approval of a new member country requires agreement by three - quarters of OPEC 's existing members , including all five
of the founders . In October 2015 , Sudan formally submitted an application to join , but it is not yet a member . Country Region Membership Years Population (
2016 est . )
</item 8>

<item 9>
OPEC
During 1961 -- 1975 , the five founding nations were joined by Qatar ( 1961 ) , Indonesia ( 1962 -- 2008 , rejoined 2014 - 2016 ) , Libya ( 1962 ) , United Arab
Emirates ( originally just the Emirate of Abu Dhabi , 1967 ) , Algeria ( 1969 ) , Nigeria ( 1971 ) , Ecuador ( 1973 -- 1992 , rejoined 2007 ) , and Gabon ( 1975
-- 1994 , rejoined 2016 ) . By the early 1970s , OPEC 's membership accounted for more than half of worldwide oil production . Indicating that OPEC is not
averse to further expansion , Mohammed Barkindo , OPEC 's Acting Secretary General in 2006 , urged his African neighbors Angola and Sudan to join , and Angola
did in 2007 , followed by Equatorial Guinea in 2017 . Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia , and other oil - exporting
nations have attended many OPEC meetings as observers , as an informal mechanism for coordinating policies .
</item 9>

<item 10>
OPEC
However , their influence on international trade is periodically challenged by the expansion of non-OPEC energy sources , and by the recurring temptation for
individual OPEC countries to exceed production ceilings and pursue conflicting self - interests . Contents ( hide ) 1 Membership 1.1 Current member countries
1.2 Lapsed members 1.3 Observers 1.4 Leadership and decision - making 1.5 International cartel 1.6 Conflicts 2 Market information 2.1 Publications and research
2.2 Crude oil benchmarks 2.3 Spare capacity 3 History and impact 3.1 Post-WWII situation 3.2 1959 -- 1960 anger from exporting countries 3.3 1960 -- 1975
founding and expansion 3.4 1973 -- 1974 oil embargo 3.5 1975 -- 1980 Special Fund , now OFID 3.6 1975 hostage siege 3.7 1979 -- 1980 oil crisis and 1980s oil
glut 3.8 1990 -- 2003 ample supply and modest disruptions 3.9 2003 -- 2011 volatility 3.
</item 10>
=== END ITEM id=476753499ba774cc ===

=== ITEM id=e50922fa8d21099d ===
Gold sentences and facts:
Sentence 0: In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript 3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder ( built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of components that included charting , advanced UI , and data services ( Flex Data Services ) .
  Fact 0.0: In 2007, Adobe's first version release was Adobe Flash CS3 Professional.
  Fact 0.1: Adobe Flash CS3 Professional was the ninth major version of Flash.
  Fact 0.2: Adobe Flash CS3 Professional introduced the ActionScript 3.0 programming language.
  Fact 0.3: ActionScript 3.0 supported modern programming practices.
  Fact 0.4: ActionScript 3.0 enabled business applications to be developed with Flash.
  Fact 0.5: Adobe Flex Builder was built on Eclipse.
  Fact 0.6: Adobe Flex Builder targeted the enterprise application development market.
  Fact 0.7: Adobe Flex Builder was released in 2007.
  Fact 0.8: Flex Builder included the Flex SDK.
  Fact 0.9: The Flex SDK is a set of components that included charting, advanced UI, and data services.
  Fact 0.10: The data services in the Flex SDK are called Flex Data Services.

Retrieved items:
<item 1>
Adobe Flash
Adobe Flash - Wikipedia Adobe Flash Jump to : navigation , search For the application / animation - maker , see Adobe Animate . For other uses , see Adobe Flash
( disambiguation ) . Adobe Flash Developer ( s ) Adobe Systems Macromedia FutureWave Target platform ( s ) Web browsers , iOS , Android , Windows , macOS Editor
software Adobe Animate , Flash Builder , FlashDevelop , Powerflasher FDT , Flash Catalyst Scaleform Player software Flash Player Adobe AIR OpenFL Scaleform
Gnash Lightspark Format ( s ) SWF FLV FLA Programming language ( s ) ActionScript Application ( s ) Animation Browser games Rich Internet applications Desktop
applications Mobile apps Mobile games console games computer games Status Active ( Until 2020 ) License Proprietary Adobe Flash is a multimedia software
platform used for production of animations , rich Internet applications , desktop applications , mobile applications , mobile games and embedded web browser
video players . Flash displays text , vector graphics and raster graphics to provide animations , video games and applications . It allows streaming of audio
and video , and can capture mouse , keyboard , microphone and camera input . Artists may produce Flash graphics and animations using Adobe Animate .
</item 1>

<item 2>
Adobe Flash
Adobe AIR was upgraded to support 64 - bit computers , and to allow developers to add additional functionality to the AIR runtime using AIR Native Extensions (
ANE ) . In 2014 , Adobe AIR reached a milestone with over 100,000 unique applications built , and over 1 billion installations logged across the world ( May
2014 ) . Adobe AIR was voted the Best Mobile Application Development product at the Consumer Electronics Show on two consecutive years ( CES 2014 and CES 2015 )
. In 2016 , Adobe renamed Flash Professional , the primary authoring software for Flash content , to Adobe Animate to reflect its growing use for authoring
HTML5 content in favour of Flash content . Open Screen Project ( edit ) On May 1 , 2008 , Adobe announced the Open Screen Project , with the intent of providing
a consistent application interface across devices such as personal computers , mobile devices , and consumer electronics .
</item 2>

<item 3>
Adobe Flash
FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse PageMaker Photoshop Photoshop Lightroom Premiere Pro Reader
Shockwave Player Server LiveCycle Flash Media Server BlazeDS ColdFusion JRun PhoneGap Build Technologies ActionScript Adobe AIR Adobe Flash Adobe Flex Adobe
Shockwave Adobe Font Folio Digital Negative ( DNG ) Authorware FlashPaper Portable Document Format ( PDF ) PostScript Web services Acrobat.com Adobe Solutions
Network Adobe Photoshop Express Adobe Premiere Express Adobe Spark Adobe Stock Acquisitions Aldus Macromedia Scene7 Omniture Behance Fotolia People Bruce Chizen
Charles Geschke Shantanu Narayen John Warnock Del Yocam Commons Retrieved from `` https://en.wikipedia.org/w/index.php?title=Adobe_Flash&oldid=821675124 ''
Categories : Adobe Flash American inventions Cross-platform software Computing platforms C++ software Mac OS multimedia software MacOS multimedia software
Windows multimedia software Macromedia software Adobe software 1993 software Video game development software Hidden categories : Webarchive template wayback
links Pages using web citations with no URL Pages using citations with accessdate and no URL Use mdy dates from April 2012 Articles containing potentially dated
statements from February 2009 All articles containing potentially dated statements Articles containing potentially
</item 3>

<item 4>
Adobe Flash Player
( SWC ) Software ( list ) Players Adobe AIR Adobe Flash Player OpenFL Gnash Lightspark Gameswf Tamarin Animation tools Adobe Animate Toon Boom Moho GoAnimate
Stencyl Programming tools FlashDevelop Powerflasher FDT Apache Flex Haxe CrossBridge Adobe Scout FlashFirebug Libraries Starling Framework Away3D Flare3D
Stage3D PureMVC Ming List of Flex frameworks Converters Adobe Flash Media Live Encoder SWFTools swfmill Google Swiffy Adobe Wallaby Server - side Adobe Flash
Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC Scaleform GFx SWiSH Max Swift 3D
Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash controversy Flash animation films
television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software Macromedia Adobe Systems Adobe
Systems Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code
Edge Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse
</item 4>

<item 5>
Graph query row:
entity: Adobe Flash CS3 Professional
relationship: RELATED_TO
related: ActionScript 3.0
kind: programming language
</item 5>

<item 6>
Adobe Flash
The Apache Flex ActionScript 3.0 compiler generates SWF files from ActionScript 3 files . Flex was the primary ActionScript 3 compiler and was actively
developed by Adobe before it was donated to Apache Software Foundation in 2011 . Adobe Animate -- primarily used to design graphics and animation , but supports
ActionScript scripting and debugging . Adobe Flash Builder -- enterprise application development & debugging , contains the Flex SDK with UI and charting
components . Adobe Scout -- a visual profiler to optimize performance of Flash content . CrossBridge -- a free SDK to cross-compile C++ code to run in Flash
Player . Third - party tools ( edit ) Third - party development tools have been created to assist developers in creating software applications and video games
with Flash . FlashDevelop is a free and open - source Flash ActionScript IDE , which includes a project manager and debugger for building applications on Flash
Player and Adobe AIR . Powerflasher FDT is a commercial ActionScript IDE similar to FlashDevelop . Haxe is an open source , high - level object - oriented
programming language geared towards web - content creation that can compile SWF files from Haxe programs .
</item 6>

<item 7>
Adobe Flash
Artists may produce Flash graphics and animations using Adobe Animate . Software developers may produce applications and video games using Adobe Flash Builder ,
FlashDevelop , Flash Catalyst , or any text editor when used with the Apache Flex SDK . End - users can view Flash content via Flash Player ( for web browsers )
, AIR ( for desktop or mobile apps ) or third - party players such as Scaleform ( for video games ) . Adobe Flash Player ( supported on Microsoft Windows ,
macOS and Linux ) enables end - users to view Flash content using web browsers . Adobe Flash Lite enabled viewing Flash content on older smartphones , but has
been discontinued and superseded by Adobe AIR . The ActionScript programming language allows the development of interactive animations , video games , web
applications , desktop applications and mobile applications . Programmers can implement Flash software using an IDE such as Adobe Animate , Adobe Flash Builder
, Adobe Director , FlashDevelop and Powerflasher FDT . Adobe AIR enables full - featured desktop and mobile applications to be developed with Flash and
published for Windows , macOS , Android , iOS , Xbox One , PlayStation 4 , and Nintendo Wii U and Switch .
</item 7>

<item 8>
Adobe Flash
Adobe Wallaby Server - side Adobe Flash Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC
Scaleform GFx SWiSH Max Swift 3D Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash
controversy Flash animation films television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software
Macromedia Adobe Systems Adobe Creative Suite and Creative Cloud Current products Acrobat After Effects Animate Audition Bridge Dreamweaver Flash Builder
Illustrator InCopy InDesign Lightroom Media Encoder Muse Photoshop Prelude Premiere Pro Spark XD Creative Cloud services Behance Business Catalyst Story Typekit
Former products Contribute Device Central Edge Edge Animate Encore Flash Catalyst Fireworks GoLive ImageReady OnLocation Soundbooth Stock Photos Ultra Adobe
eLearning Suite Current products Captivate Flash Dreamweaver Photoshop Acrobat Presenter Soundbooth Shared applications Bridge Device Central Adobe Systems
Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code Edge
Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse PageMaker
Photoshop
</item 8>

<item 9>
Adobe Flash
CreateJS is library that while available separately was also adopted by Adobe as a replacement for Wallaby in CS6 . Unlike Wallaby , which was a standalone
program , the `` Toolkit for CreateJS '' only works as a plug - in inside Flash Professional ; it generates output for the HTML5 canvas , animated with
JavaScript . Around December 2013 , the toolkit was integrated directly into Flash Professional CC . Criticisms ( edit ) Mobile support ( edit ) Websites built
with Adobe Flash will not function on most modern mobile devices running Google Android or iOS ( iPhone , iPad ) . The only alternative is using HTML5 and
responsive web design to build websites that support both desktop and mobile devices . However , Flash is still used to build mobile games using Adobe AIR .
Such games will not work in mobile web browsers , but must be installed via the appropriate app store . Vendor dependence ( edit ) See also : Vendor lock - in
The reliance on Adobe for decoding Flash makes its use on the World Wide Web a concern -- the completeness of its public specifications are debated , and no
complete implementation of Flash is publicly available in source code form with a license that permits reuse .
</item 9>

<item 10>
Adobe Flash
, blend modes ( similar to Adobe Photoshop ) , and advanced features for FLV video . Adobe ( edit ) Macromedia was acquired by Adobe Systems in 2005 , and the
entire Macromedia product line including Flash , Dreamweaver , Director / Shockwave , Fireworks ( which has since been discontinued ) and Authorware is now
handled by Adobe . In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript
3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder (
built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of
components that included charting , advanced UI , and data services ( Flex Data Services ) . In 2008 , Adobe released the tenth version of Flash , Adobe Flash
CS4 . Flash 10 improved animation capabilities within the Flash editor , adding a motion editor panel ( similar to Adobe After Effects ) , inverse kinematics (
bones ) , basic 3D object animation , object - based animation , and other text and graphics features .
</item 10>
=== END ITEM id=e50922fa8d21099d ===

=== ITEM id=b90dd5fdb9ee1018 ===
Gold sentences and facts:
Sentence 0: John James McCoy is a fictional character in the television drama Law & Order . He was created by Michael S. Chernuchin and portrayed by Sam Waterston from 1994 until the end of the series in 2010 . He is the second - longest tenured character on the show ( 16 seasons ) , after Lt. Anita Van Buren ( 17 seasons ; portrayed by S. Epatha Merkerson ) . He appeared in 368 episodes of Law & Order , four episodes of Law & Order : Special Victims Unit , two episodes of Law & Order : Trial by Jury , two episodes of Homicide : Life on the Street , and the made - for - TV movie Exiled .
  Fact 0.0: John James McCoy is a fictional character in the television drama Law & Order.
  Fact 0.1: John James McCoy was created by Michael S. Chernuchin.
  Fact 0.2: John James McCoy was portrayed by Sam Waterston from 1994 until the end of the series in 2010.
  Fact 0.3: John James McCoy is the second-longest tenured character on the show, with 16 seasons.
  Fact 0.4: John James McCoy's tenure ranks just after Lt. Anita Van Buren, who had 17 seasons.
  Fact 0.5: Lt. Anita Van Buren was portrayed by S. Epatha Merkerson.
  Fact 0.6: John James McCoy appeared in 368 episodes of Law & Order.
  Fact 0.7: John James McCoy appeared in four episodes of Law & Order: Special Victims Unit.
  Fact 0.8: John James McCoy appeared in two episodes of Law & Order: Trial by Jury.
  Fact 0.9: John James McCoy appeared in two episodes of Homicide: Life on the Street.
  Fact 0.10: John James McCoy appeared in the made-for-TV movie Exiled.
Sentence 1: Jack McCoy brings 24 years of experience with him as he is appointed executive assistant district attorney by Adam Schiff ( Steven Hill ) in the season - five premiere episode `` Second Opinion '' . He quickly establishes himself as a more unconventional , ruthless litigator than his predecessor , Ben Stone ( Michael Moriarty ) . He often bends -- and sometimes breaks -- trial rules to get convictions , finds tenuous rationales for charging defendants with crimes when the original charges fail to stick , and charges innocent people to frighten them into testifying against others . McCoy is found in contempt of court 80 times for such behavior , and his tactics occasionally incur negative publicity for the DA 's office . His underlying motivation , however , is not , he maintains , corruption , but a sincere desire to see justice done . To that end , McCoy has gone after defendants accused of perverting the justice system to arrange wrongful convictions with just as much determination as his more mundane cases . Such aggressive actions in the courts have earned him the nickname `` Hang ' em High McCoy '' . He has subsequently developed a reputation with both colleagues and rival attorneys , once being referred to as `` the top of the legal food chain '' by a rival attorney during a trial .
  Fact 1.0: Jack McCoy brings 24 years of experience with him.
  Fact 1.1: Jack McCoy is appointed executive assistant district attorney by Adam Schiff.
  Fact 1.2: Adam Schiff is played by Steven Hill.
  Fact 1.3: Jack McCoy is appointed in the season-five premiere episode 'Second Opinion'.
  Fact 1.4: Jack McCoy quickly establishes himself as a more unconventional, ruthless litigator than his predecessor.
  Fact 1.5: Jack McCoy's predecessor is Ben Stone.
  Fact 1.6: Ben Stone is played by Michael Moriarty.
  Fact 1.7: Jack McCoy often bends trial rules to get convictions.
  Fact 1.8: Jack McCoy sometimes breaks trial rules to get convictions.
  Fact 1.9: Jack McCoy finds tenuous rationales for charging defendants with crimes when the original charges fail to stick.
  Fact 1.10: Jack McCoy charges innocent people to frighten them into testifying against others.
  Fact 1.11: Jack McCoy is found in contempt of court 80 times for such behavior.
  Fact 1.12: Jack McCoy's tactics occasionally incur negative publicity for the DA's office.
  Fact 1.13: Jack McCoy's underlying motivation is not corruption, he maintains.
  Fact 1.14: Jack McCoy's underlying motivation is a sincere desire to see justice done.
  Fact 1.15: Jack McCoy has gone after defendants accused of perverting the justice system to arrange wrongful convictions.
  Fact 1.16: Jack McCoy pursues those defendants with just as much determination as his more mundane cases.
  Fact 1.17: Jack McCoy's aggressive actions in the courts have earned him the nickname 'Hang 'em High McCoy'.
  Fact 1.18: Jack McCoy has developed a reputation with both colleagues and rival attorneys.
  Fact 1.19: Jack McCoy was once referred to as 'the top of the legal food chain' by a rival attorney during a trial.
Sentence 2: `` Entitled '' ( onscreen ) : McCoy makes his first appearance on the series . McCoy , working with his assistant , Abbie Carmichael ( Angie Harmon ) , assists the Special Victims Unit in solving a cold case , that Briscoe handled years ago with his former partner , Detective Mike Logan ( Chris Noth ) .
  Fact 2.0: The episode 'Entitled' (onscreen) is where McCoy makes his first appearance on the series.
  Fact 2.1: McCoy works with his assistant, Abbie Carmichael.
  Fact 2.2: Abbie Carmichael is played by Angie Harmon.
  Fact 2.3: McCoy assists the Special Victims Unit in solving a cold case.
  Fact 2.4: Briscoe handled the cold case years ago.
  Fact 2.5: Briscoe handled the cold case with his former partner, Detective Mike Logan.
  Fact 2.6: Detective Mike Logan is played by Chris Noth.

Retrieved items:
<item 1>
Jack McCoy
He was created by Michael S. Chernuchin and portrayed by Sam Waterston from 1994 until the end of the series in 2010 . He is the second - longest tenured
character on the show ( 16 seasons ) , after Lt. Anita Van Buren ( 17 seasons ; portrayed by S. Epatha Merkerson ) . He appeared in 368 episodes of Law & Order
, four episodes of Law & Order : Special Victims Unit , two episodes of Law & Order : Trial by Jury , two episodes of Homicide : Life on the Street , and the
made - for - TV movie Exiled . Waterston 's performance as McCoy on the New York - based series was so popular that it resulted in him being declared a ``
Living Landmark '' by the New York Landmarks Conservancy , along with fellow longtime series cast member Jerry Orbach ( who portrayed the popular police
detective Lennie Briscoe for 12 years ) . Contents ( hide ) 1 Character overview 2 Personal life 3 Notable conflicts 4 In Law & Order : Special Victims Unit 4.1
As executive assistant district attorney 4.1.
</item 1>

<item 2>
Jack McCoy
Reception ( edit ) Entertainment Weekly television critic Ken Tucker has praised Law & Order 's creator Dick Wolf for putting McCoy at the center of `` some of
the best episodes of the immortal series ' 19th season . '' Tucker elaborates how the character , riding `` herd over a couple of stubborn young bucks --
assistant DAs Mike Cutter ( Linus Roache ) and Connie Rubirosa ( Alana de la Garza ) -- McCoy argues , bellows orders , and croaks with outrage when his charges
disobey his legal advice . ''
</item 2>

<item 3>
Jack McCoy
wikipedia.org/w/index.php?title=Jack_McCoy&oldid=838021970 '' Categories : Law & Order characters Fictional lawyers Fictional American people of Irish descent
Fictional characters introduced in 1994 Fictional district attorneys Fictional victims of child abuse Crossover characters in television Hidden categories :
Pages using deprecated image syntax Articles using Infobox character with multiple unlabeled fields Talk Contents About Wikipedia Deutsch Hrvatski 日本 語
Português Srpskohrvatski / српскохрватски Edit links This page was last edited on 24 April 2018 , at 13 : 14 . About Wikipedia
</item 3>

<item 4>
Jack McCoy
District attorney 's office timeline ( edit ) Time period Executive assistant district attorney ( EADA ) Assistant district attorney ( ADA ) District attorney (
DA ) 1994 -- 1996 Jack McCoy Claire Kincaid Adam Schiff 1996 -- 1998 Jamie Ross 1998 -- 2000 Abbie Carmichael 2000 -- 2001 Nora Lewin 2001 -- 2002 Serena
Southerlyn 2002 -- 2005 Arthur Branch 2005 -- 2006 Alexandra Borgia 2006 -- 2007 Connie Rubirosa 2008 -- 2010 Michael Cutter Jack McCoy Appearances on other TV
shows ( edit ) Homicide : Life on the Street Season 6 Episode 5 : `` Baby ,
</item 4>

<item 5>
Jack McCoy
Although McCoy insists that he is a prosecutor , and not a politician , McCoy replaces Branch as the DA immediately thereafter . In Law & Order : Special
Victims Unit ( edit ) McCoy has appeared in four episodes of Law & Order : Special Victims Unit ; however , he is often referred to in the series , and his
actions affect the ADAs working with the Special Victims Unit . As executive assistant district attorney ( edit ) Season one ( edit ) `` Entitled '' ( onscreen
) : McCoy makes his first appearance on the series . McCoy , working with his assistant , Abbie Carmichael ( Angie Harmon ) , assists the Special Victims Unit
in solving a cold case , that Briscoe handled years ago with his former partner , Detective Mike Logan ( Chris Noth ) . Season two ( edit ) `` Baby Killer '' (
mention ) : After a murder intertwines with ADA Alexandra Cabot 's ( Stephanie March ) sex - crimes case , Chief Assistant District Attorney Charlie Phillips (
Jeffrey DeMunn ) , Lewin 's second in command , hands the murder case to McCoy . As district attorney ( edit ) Season 9 ( edit ) `` Blinded '' ( onscreen ) :
McCoy had been promoted to district attorney .
</item 5>

<item 6>
Graph query row:
person: Jack McCoy
relationship: RELATED_TO
fact: None
context: None
work: Law & Order
firstAppearanceEpisode: Second Opinion
year: September 21, 1994
</item 6>

<item 7>
Jack McCoy
Jack McCoy - wikipedia Jack McCoy Jump to : navigation , search For the politician and labor activist , see Jack E. McCoy . Jack McCoy Law & Order character
First appearance `` Second Opinion '' `` Entitled '' ( SVU ) Last appearance `` Rubber Room '' ( Law & Order ) `` The Undiscovered Country '' ( SVU ) Portrayed
by Sam Waterston Time on show 1994 -- 2010 , 2018 Seasons 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 Credited appearances 368
episodes ( L&O ) 2 episodes ( HLOTS ) 4 episodes ( SVU ) 2 episodes ( TBJ ) 376 episodes ( total ) Preceded by Ben Stone ( EADA ) Arthur Branch ( DA ) ( L&O )
Succeeded by Michael Cutter ( EADA ) Information Family Rebecca McCoy ( daughter ) John James McCoy is a fictional character in the television drama Law & Order
.
</item 7>

<item 8>
Jack McCoy
1 Season one 4.1. 2 Season two 4.2 As district attorney 4.2. 1 Season 9 4.2. 2 Season 10 4.2. 3 Season 11 4.2. 4 Season 12 4.2. 5 Season 13 4.2. 6 Season 19 5
Reception 6 District attorney 's office timeline 7 Appearances on other TV shows 8 References Character overview ( edit ) Jack McCoy brings 24 years of
experience with him as he is appointed executive assistant district attorney by Adam Schiff ( Steven Hill ) in the season - five premiere episode `` Second
Opinion '' . He quickly establishes himself as a more unconventional , ruthless litigator than his predecessor , Ben Stone ( Michael Moriarty ) . He often bends
-- and sometimes breaks -- trial rules to get convictions , finds tenuous rationales for charging defendants with crimes when the original charges fail to stick
, and charges innocent people to frighten them into testifying against others . McCoy is found in contempt of court 80 times for such behavior , and his tactics
occasionally incur negative publicity for the DA 's office .
</item 8>

<item 9>
Jack McCoy
By the end of the 18th season , however , McCoy discovers that Shalvoy is involved in a prostitution scandal that is tied to a murder case he is prosecuting .
Angered and disappointed , McCoy orders Cutter to start investigating Shalvoy , who retaliates by lending his support to McCoy 's opponent in the election . In
the last episode of season 19 , `` The Drowned and the Saved '' , Cutter uncovers evidence that Shalvoy tried to buy a Senate seat for his wife , and threatens
him with public disgrace unless he resigns . McCoy 's opponent suddenly has no patron , giving McCoy good prospects for victory . In the opening episode of
season 20 , McCoy was revealed to have won the election ; he serves as DA for the remainder of this final season of Law & Order . Personal Life ( edit ) While
he is a brilliant legal mind , McCoy has more than a few personal demons . He was abused by his father , an Irish Chicago policeman who had also beat Jack 's
mother , and who eventually died of cancer . McCoy says that his determination and unyielding work ethic are a byproduct of having been harshly punished by his
father for losing at anything .
</item 9>

<item 10>
Jack McCoy
His underlying motivation , however , is not , he maintains , corruption , but a sincere desire to see justice done . To that end , McCoy has gone after
defendants accused of perverting the justice system to arrange wrongful convictions with just as much determination as his more mundane cases . Such aggressive
actions in the courts have earned him the nickname `` Hang ' em High McCoy '' . He has subsequently developed a reputation with both colleagues and rival
attorneys , once being referred to as `` the top of the legal food chain '' by a rival attorney during a trial . Following the 17th season ( 2006 -- 2007 ) ,
Jack McCoy was appointed interim district attorney , taking over from Arthur Branch ( Fred Thompson ) . McCoy 's appearance on Law & Order : Special Victims
Unit on the November 13 , 2007 , episode `` Blinded '' , marked his first appearance in the Law & Order universe as district attorney . The replacement for his
former position is Michael Cutter ( Linus Roache ) , a prosecutor with a penchant for recklessness not unlike McCoy 's own in his younger days . This
occasionally presents political difficulties for the new district attorney .
</item 10>
=== END ITEM id=b90dd5fdb9ee1018 ===

=== ITEM id=fd5bcb4561dde320 ===
Gold sentences and facts:
Sentence 0: The Quebec Winter Carnival or the Carnaval de Québec is a pre-Lenten festival held in Quebec City . After being held intermittently since 1894 , the Carnaval de Québec has been celebrated annually since 1955 . That year Bonhomme Carnaval , the mascot of the festival , made his first appearance . Up to one million people attended the Carnaval de Québec in 2006 making it the largest winter festival in the world .
  Fact 0.0: The Quebec Winter Carnival (Carnaval de Québec) is a pre-Lenten festival.
  Fact 0.1: The Carnaval de Québec is held in Quebec City.
  Fact 0.2: The Carnaval de Québec was held intermittently after 1894.
  Fact 0.3: The Carnaval de Québec has been celebrated annually since 1955.
  Fact 0.4: In 1955, Bonhomme Carnaval, the mascot of the festival, made his first appearance.
  Fact 0.5: Up to one million people attended the Carnaval de Québec in 2006.
  Fact 0.6: In 2006, the Carnaval de Québec was the largest winter festival in the world.

Retrieved items:
<item 1>
God Save the Queen
Queen 's Printer for Canada . Retrieved 5 July 2010 . Jump up ^ Remembrance Day ( PDF ) , Vancouver Board of Parks and Recreation , 11 November 2009 , archived
from the original ( PDF ) on 7 July 2011 , retrieved 5 July 2010 Jump up ^ `` Honours and Salutes '' . Queen 's Printer for Canada . Retrieved 4 March 2017 . ^
Jump up to : Bélanger , Claude . `` The Quebec History Encyclopedia '' . In Marianopolis College . National Anthem of Canada . Montreal : Marianopolis College .
Retrieved 5 July 2010 . ^ Jump up to : Department of Canadian Heritage . `` National Anthem : O Canada > Parliamentary Action '' . Queen 's Printer for Canada .
Retrieved 29 June 2010 . ^ Jump up to : Department of National Defence ( 1 April 1999 ) , The Honours , Flags and Heritage Structure of the Canadian Forces (
PDF ) , Ottawa : Queen 's Printer for Canada , p.
</item 1>

<item 2>
God Save the Queen
Jump up ^ MacLeod 2008 , pp. 54 , I ^ Jump up to : Kallmann , Helmut , `` The Canadian Encyclopedia '' , in Marsh , James Harley , Encyclopedia of Music in
Canada > Musical Genres > National and royal anthems , Toronto : Historica Foundation of Canada , archived from the original on 10 October 2011 , retrieved 25
June 2010 Jump up ^ Office of the Lieutenant Governor of Nova Scotia . `` History of the Lieutenant Governor > Royal Salute > Royal Salute ( Formerly known as
the Vice-Regal Salute ) '' . Queen 's Printer for Nova Scotia . Archived from the original on 30 April 2010 . Retrieved 25 June 2010 . Jump up ^ Hoiberg , Dale
( ed . ) , `` Encyclopædia Britannica '' , O Canada , Toronto : Encyclopædia Britannica , Inc. , retrieved 25 June 2010 Jump up ^ Citizenship and Immigration
Canada ( 2009 ) . Discover Canada ( PDF ) . Ottawa : Queen 's Printer for Canada . p. 2 .
</item 2>

<item 3>
Quebec Winter Carnival
Quebec winter carnival - wikipedia Quebec winter carnival Jump to : navigation , search This article needs additional citations for verification . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2018 ) ( Learn how and when to
remove this template message ) Quebec Winter Carnival Carnaval de Québec Bonhomme Carnaval in 2011 Genre Festival Begins End of January Ends Mid February
Frequency Annually Location ( s ) Quebec City Country Canada Website Carnaval de Québec The Quebec Winter Carnival or the Carnaval de Québec is a pre-Lenten
festival held in Quebec City . After being held intermittently since 1894 , the Carnaval de Québec has been celebrated annually since 1955 . That year Bonhomme
Carnaval , the mascot of the festival , made his first appearance . Up to one million people attended the Carnaval de Québec in 2006 making it the largest
winter festival in the world .
</item 3>

<item 4>
Quebec Winter Carnival
Frommer 's Canada : With the best hiking & outdoor adventures . John Wiley & Sons . p. 300 . ISBN 978 - 0 - 470 - 04457 - 5 . Jump up ^
https://www.theglobeandmail.com/life/travel/activities-and-interests/the-snowman-behind-the-legend-get-to-know-quebecs-beloved-bonhomme/article37716316/ Quebec
Winter Carnaval virtual tour in panoramic photography Tornoipee-wee.qc.ca Fissnowboard.com Icehotel-canada.com Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Quebec_Winter_Carnival&oldid=822352977 '' Categories : Tourist attractions in Quebec City Festivals in Quebec City
Winter festivals in Canada Mardi Gras 1894 establishments in Quebec Hidden categories : Articles needing additional references from January 2018 All articles
needing additional references Quebec articles missing geocoordinate data All articles needing coordinates Talk Contents About Wikipedia Deutsch Español Français
한국어 日本 語 Português Svenska Edit links This page was last edited on 25 January 2018 , at 21 : 54 .
</item 4>

<item 5>
Quebec Winter Carnival
Bonhomme is described as a seven - foot - tall , four - hundred pound snowman sporting a red cap , black buttons and a ceinture fléchée . It is traditional to
drink Caribou , a hot alcoholic beverage , to keep warm . The public auction is a fundraising event in aid of the carnival . This auction features many goods
and services donated for silent auction and live auction . Feasts and restaurants ( edit ) The Business Leaders ' Luncheon , organized by the Québec City
Chamber of Commerce . The Calgary Stampede Flap Jack Breakfast . In 2008 , carnival - goers were served by Prime Minister Stephen Harper . The ' Restaurant
Partners ' Campaign is a 179 - day promotion during which Québec City restaurants offer customers a special menu for a fixed price throughout the carnival (
including appetizer , soup , or salad , a main course , and a dessert ) . Races and tournaments ( edit ) A sleigh race in which drivers and their horses take
part in a single - and double - harness race on the Plains of Abraham . An Ice canoe race on the St. Lawrence river .
</item 5>

<item 6>
Quebec Winter Carnival
An Ice canoe race on the St. Lawrence river . The Québec International Pee - Wee Hockey Tournament ( not officially part of the carnival program since 1977 ) .
The Snowboard World Cup in Quebec City ( not part of the carnival official program ) . Also not part of the official program but worth a visit or a stay , the
Quebec City Ice Hotel is open every year from early January to late March with its bar , nightclub , exhibition galleries and ceremonial chapel . Gallery ( edit
) Ice sculpture ( 2005 ) Family venue ( 2006 ) Ice Palace ( 2009 ) Parade in Montreal ( 2011 ) Canoe race ( 2011 ) Ice Palace ( 2011 ) Toboggan at Terrasse
Dufferin References ( edit ) Jump up ^ Daniel Shafto ( 1 January 2009 ) . Carnival . Infobase Publishing . p. 80 . ISBN 978 - 1 - 4381 - 2660 - 9 . Jump up ^
Hilary Davidson ; Paul Karr ; Herbert Bailey Livesey ; Bill McRae ; Donald Olson ( 14 August 2006 ) .
</item 6>

<item 7>
God Save the Queen
Ottawa : Queen 's Printer for Canada . p. 2 . ISBN 978 - 1 - 100 - 12739 - 2 . Retrieved 16 January 2011 . Jump up ^ Alberta Police and Peace Officers '
Memorial Day 2009 Order of Service , Queen 's Printer for Alberta , 27 September 2009 access - date = requires url = ( help ) Jump up ^ Department of Canadian
Heritage . `` Ceremonial and Canadian Symbols Promotion > The National Flag of Canada '' . Queen 's Printer for Canada . Retrieved 5 July 2010 . Jump up ^ Royal
Canadian Legion Dominion Command ( 4 November 2009 ) . `` National Remembrance Day Ceremony '' . Royal Canadian Legion . Archived from the original on 12 March
2010 . Retrieved 5 July 2010 . Jump up ^ Department of Veterans Affairs . `` Canada Remembers > Partnering Opportunities > Community Engagement Partnership Fund
> Nova Scotia > Community Engagement Partnership Fund : Nova Scotia '' . Queen 's Printer for Canada . Retrieved 5 July 2010 .
</item 7>

<item 8>
Quebec Winter Carnival
About Wikipedia
</item 8>

<item 9>
Quebec Winter Carnival
Contents ( hide ) 1 Activities and attractions 2 Feasts and restaurants 3 Races and tournaments 4 Gallery 5 References Activities and attractions ( edit ) The
most famous attractions of this winter festival are the night - time and daytime parades led by mascot Bonhomme Carnaval . The parades wind through the upper
city , decorated for the occasion with lights and ice sculptures . Numerous public and private parties , shows and balls are held across the city , some of them
outside in the bitter cold , testimony to the Québécois ' fabled joie de vivre . Other major events include : A masquerade ball with up to 400 participants at
the grand ballroom of the Château Frontenac . The opening and closing ceremonies taking place at the Ice Palace before thousands of participants , Bonhomme and
the mayor of Quebec . Outdoor sport events ( snowboarding , ice canoe , snowshoes , hockey , dog - sledding , etc. , some of them part of World Championship
tournaments ) inside and outside the city . Free outdoor public banquets ( brunch , breakfast , etc . ) . .
</item 9>

<item 10>
Quebec Winter Carnival
Free outdoor public banquets ( brunch , breakfast , etc . ) . . The Canadian , Québécois , International and Student artist snow sculpture contests on the
Plains of Abraham , the main setting of the carnival . The Plains are a public city park and stay open for leisure activities , including snowshoeing and
cross-country skiing trails , during carnival time . Part of the Plains around the Citadel is transformed into an outdoor winter amusement park with various
family - themed activities , including the display of the three main snow - sculpting contests ( Canada 's provinces , Quebec 's regions , International ) and
the traditional bikini snow bath event ( bain de neige ) Outdoor dance parties are held at the Ice Palaces . Kiosks and other outlets in the city sell the
Bonhomme effigy tag that grants admission into most of the events , although some are free outside the main site . Most commercial main streets are decorated
and some bars and restaurants set up a winter patio in front of their establishments . Bonhomme -- short for bonhomme de neige ( `` snowman '' ) is the official
ambassador of the festivities , the castle lord of the Ice Palace .
</item 10>
=== END ITEM id=fd5bcb4561dde320 ===
