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

This file has 11 items.

=== ITEM id=37b901c6f7f05db3 ===
Gold sentences and facts:
Sentence 0: Midland F1 Racing (also known as MF1 Racing) was a Formula One constructor and racing team.
  Fact 0.0: Midland F1 Racing was a Formula One constructor.
  Fact 0.1: Midland F1 Racing was a racing team.
  Fact 0.2: Midland F1 Racing is also known as MF1 Racing.
Sentence 1:  In 2007, the team competed as Spyker F1, and in 2008 was sold to Indian businessman Vijay Mallya and was renamed Force India F1.
  Fact 1.0: In 2007, the team competed as Spyker F1.
  Fact 1.1: In 2008, the team was sold to Indian businessman Vijay Mallya.
  Fact 1.2: In 2008, the team was renamed Force India F1.
Sentence 2: The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.
  Fact 2.0: The Spyker F1 Team was known as the Etihad Aldar Spyker F1 Team for sponsorship reasons.
  Fact 2.1: The Spyker F1 Team was a Formula One team.
  Fact 2.2: The Spyker F1 Team competed in the 2007 Formula One World Championship.
  Fact 2.3: The Spyker F1 Team was created by Spyker Cars after their buyout of the short-lived Midland F1 team.
  Fact 2.4: Midland F1 was formerly known as Jordan Grand Prix.

Retrieved items:
<item 1>
Midland F1 Racing
Midland F1 Racing (also known as MF1 Racing) was a Formula One constructor and racing team.

 It competed in the 2006 Formula One season with drivers Christijan Albers and Tiago Monteiro.

 The team was created by the renaming of Jordan Grand Prix after its purchase by Canadian businessman, and owner of the Midland Group, Alex Shnaider.

 The team was registered as the first Russian Formula One team, reflecting Shnaider's roots, although it continued to be based in the United Kingdom, at
Jordan's Silverstone factory.

 Towards the end of the 2006 season, the team was sold to Spyker Cars N.V.; the team raced in its last three Grands Prix under the official name "Spyker MF1
Racing".

 In 2007, the team competed as Spyker F1, and in 2008 was sold to Indian businessman Vijay Mallya and was renamed Force India F1.
</item 1>

<item 2>
2013–14 Midlands football leagues (levels 9–10)
The 2013–14 Midland Football Alliance season was the 20th and final in the history of Midland Football Alliance, a football competition in England.

 The 2013–14 Midland Football Combination season (known as the 2013–14 Athium Midland Football Combination for sponsorship reasons) was the 77th and final in
the history of Midland Football Combination and it was a feeder to the Midland Football Alliance.

 The 2013–14 West Midlands (Regional) League season was the 114th in the history of the West Midlands (Regional) League.

 It was also a feeder league to the Midland Football Alliance.

 At the end of the season the Midland Alliance and the Midland Combination merged to form the Midland Football League.
</item 2>

<item 3>
Speedway Motorsports
Speedway Motorsports, Inc. (SMI) is an American corporation that owns and manages racing facilities that host NASCAR, IndyCar Series, NHRA, World of Outlaws and
other motor racing series.

 The company's headquarters are located at Charlotte Motor Speedway, in the suburbs of Charlotte, NC.

 SMI owns nine racing facilities with a combined seating capacity of approximately 885,000.

 In addition to operating racetracks, SMI owns Performance Racing Network (PRN), U.S. Legends Cars International, and co-owns Motorsports Authentics.
</item 3>

<item 4>
Marussia F1
The Marussia F1 Team (subsequently Manor Marussia F1 Team) was an Anglo-Russian Formula One racing team and constructor which was based in Banbury, Oxfordshire
in the United Kingdom.

 The team was operated by Manor Motorsport (formerly Marussia Manor Racing), which was previously a subsidiary of Marussia Motors, a now defunct sports car
manufacturer which was based in Moscow.

 The team originally started racing in under the "Virgin Racing" name; the following year Virgin adopted Marussia as a title sponsor becoming "Marussia Virgin
Racing" until being fully rebranded as the "Marussia F1 Team" for .
</item 4>

<item 5>
Spyker F8-VII
The Spyker F8-VII (subsequently known as the Force India VJM01) was a Formula One car, constructed by Spyker F1 that competed in the 2007 Formula One World
Championship.

 A "B Specification" car named the Spyker F8-VIIB was launched at the Italian Grand Prix and used for the remainder of the 2007 season.

 The engine of Spyker F8-VII car was Ferrari 056 despite the team opted for 2006-spec engine due to cost reasons.

 For the 2008 World Championship, Force India used a slightly modified version of the F8-VIIB, called the Force India VJM01 named after team owners Vijay
Mallya, Jan Mol and Michiel Mol.

 The VJM01 used 2007-spec Ferrari 056 engines instead of 2008-spec.

 The F8-VII was the only car constructed by Spyker F1 in their own right after their take-over from the struggling Midland F1 team part way through 2006.
</item 5>

<item 6>
Spyker F1
The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World
Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.

 The change to the Spyker name was accompanied by a switch in racing livery from the red and white previously used by Midland, to an orange and silver
scheme—already seen on the Spyker Spyder GT2-R—orange being the national colour and the auto racing colour of the Netherlands.

 At the end of the 2007 season the team was sold and renamed Force India.
</item 6>

<item 7>
1999 British Grand Prix
The 1999 British Grand Prix was a Formula One motor race held on 11 July 1999 at the Silverstone Circuit near Silverstone, England.

 It was the eighth race of the 1999 Formula One season.

 The 60-lap race was won by McLaren driver David Coulthard after he started from third position.

 Eddie Irvine finished second for the Ferrari team and Williams driver Ralf Schumacher came in third.
</item 7>

<item 8>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 8>

<item 9>
Simtek
Simtek (Simulation Technology) was an engineering consultancy firm and Formula One racing team.

 The Formula One (F1) engineering consultancy arm, Simtek Research, was founded in 1989 by Max Mosley and Nick Wirth.

 It originally was involved in many areas of Formula One, including wind tunnel construction and chassis building for third parties.

 Simtek Grand Prix, the racing team, launched in 1993 and competed in the 1994 and 1995 seasons achieving a best result of ninth place.

 With large debts and a lack of sponsorship money, Simtek went into voluntary liquidation in June 1995.
</item 9>

<item 10>
2005 Chinese Grand Prix
The 2005 Chinese Grand Prix was the final Formula One motor race of the 2005 Formula One season which took place on 16 October 2005 at the Shanghai
International Circuit.

 This was the second Chinese Grand Prix to be held since the event's 2004 inception.

 The race was won by the new World Champion, Renault's Fernando Alonso.

 McLaren driver Kimi Räikkönen was four seconds behind in second position, a reflection of their season long duel for the championship.

 Toyota driver Ralf Schumacher was third.

 Renault won the Constructors' Championship at this race; they had led McLaren by two points before the start of the race.

 This was the final race for Antônio Pizzonia, the Minardi and Jordan teams, although both teams continued into 2006 under different names (Scuderia Toro Rosso
and Midland F1 Racing respectively).

 This was also last win for a car equipped with a 6-speed gearbox and with a V10 engine.
</item 10>
=== END ITEM id=37b901c6f7f05db3 ===

=== ITEM id=015659198f9433ce ===
Gold sentences and facts:
Sentence 0:  He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.
  Fact 0.0: He won three singles titles.
  Fact 0.1: He won 26 doubles titles.
  Fact 0.2: He won the Australian Open 1978 doubles title with Wojtek Fibak.
  Fact 0.3: He won the Australian Open 1980 title with fellow countryman Mark Edmondson.
  Fact 0.4: He won the Australian Open 1981 title with fellow countryman Mark Edmondson.
  Fact 0.5: He won the Roland Garros 1986 title with fellow countryman Mark Edmondson.
  Fact 0.6: He was a runner-up at the Australian Open 1985 with fellow countryman Mark Edmondson.
  Fact 0.7: Mark Edmondson is a fellow countryman of his.
Sentence 1:  In his career, he won nine doubles titles, but no singles titles.
  Fact 1.0: In his career, he won nine doubles titles.
  Fact 1.1: In his career, he won no singles titles.

Retrieved items:
<item 1>
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 1>

<item 2>
1988 Volvo International – Doubles
Jorge Lozano and Todd Witsken won in the final 6–3, 7–6 against Pieter Aldrich and Danie Visser.
</item 2>

<item 3>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 3>

<item 4>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 4>

<item 5>
Evonne Goolagong career statistics
This is a list of the main career statistics of Australian former tennis player Evonne Goolagong.

 During her career, which lasted from 1967 to 1983, Goolagong won seven singles titles at a Grand Slam event and was a runner-up on 11 occasions.

 In addition she won five Grand Slam doubles titles, partnering Margaret Court, Peggy Michel and Helen Gourlay, as well as one mixed doubles title with Kim
Warwick.

 In total she won 82 singles titles, 46 doubles titles and 4 mixed doubles titles.

 She achieved a No. 1 singles ranking for a two-week period in April–May 1976 although this was only officially recognized in 2007.

 She was a member of the Australian Federation Cup teams that won the cup in 1971, 1973 and 1974.
</item 5>

<item 6>
Jorge Lozano
In Davis Cup play, represented Mexico for 15 years, 1981–1995, won 12 doubles matches, and lost 12 as well.

 He lost 11 singles matches, and won 8, making his overall win / loss record at the Davis Cup 20–23.

 Lozano won his first title at Forest Hills in 1988, with his partner Todd Witsken, and won his last at Athens in 1993.

 He resides in Guadalajara, Mexico where he directs a tennis academy for kids and also coaches the men and women tennis team at the University Tec de Monterrey.

 He is currently the captain of the Mexican Davis Cup Team.
</item 6>

<item 7>
1994 Athens International – Doubles
Horacio de la Peña and Jorge Lozano were the defending champions, but did not participate this year.
</item 7>

<item 8>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 8>

<item 9>
1990 Stockholm Open – Doubles
Jorge Lozano and Todd Witsken were the defending champions, but lost in the quarterfinals this year.
</item 9>

<item 10>
1989 Volvo International – Doubles
Jorge Lozano and Todd Witsken were the defending champions but did not compete that year.
</item 10>
=== END ITEM id=015659198f9433ce ===

=== ITEM id=1598f23ba2a18c21 ===
Gold sentences and facts:
Sentence 0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
  Fact 0.0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia.
  Fact 0.1: The 1997 European Athletics Junior Championships were held on July 24-27.
Sentence 1: Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.
  Fact 1.0: Ljubljana is the capital of Slovenia.
  Fact 1.1: Ljubljana is the largest city of Slovenia.
  Fact 1.2: Ljubljana is also known by other, historical names.

Retrieved items:
<item 1>
1968 European Junior Games
The 1968 European Junior Games was the third edition of the biennial athletics competition for European athletes aged under twenty.

 It was the final edition of the competition under that name and was succeeded by the European Athletics Junior Championships in 1970.

 The event was held in Leipzig, part of the German Democratic Republic at that time.

 The competition was mainly between Eastern European countries, with Belgium, Austria, Netherlands and Greece being present from the west.
</item 1>

<item 2>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 2>

<item 3>
1964 European Junior Games
The 1964 European Junior Games was the first edition of the biennial athletics competition for European athletes aged under twenty.

 It was an unofficial competition without sanction from the European Athletic Association.

 The event was held at the 10th-Anniversary Stadium in Warsaw, Poland.

 The competition eventually led to the creation of the European Athletics Junior Championships in 1970.
</item 3>

<item 4>
1997 European Athletics Junior Championships
The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
</item 4>

<item 5>
European Athletics U20 Championships
The European Athletics U20 Championships (formerly named the European Athletics Junior Championships up to 2015) are the European championships for athletes who
are 19 years of age or under, which is the age range recognised by the IAAF as junior athletes.

 The event is currently organized by the European Athletic Association.
</item 5>

<item 6>
2015 European Athletics Junior Championships
The 2015 European Athletics Junior Championships were the 23rd edition of the biennial European U20 athletics championships.

 They were held in Eskilstuna, Sweden from 16 July to 19 July.
</item 6>

<item 7>
Ljubljana
Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.

 It has been the cultural, educational, economic, political, and administrative center of independent Slovenia since 1991.

 Its central geographic location within Slovenia, transport connections, concentration of industry, scientific and research institutions, and cultural tradition
are contributing factors to its leading position.
</item 7>

<item 8>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 8>

<item 9>
Amela Terzić
Amela Terzić (Serbian Cyrillic: Амела Терзић, born 2 January 1993 in Priboj, FR Yugoslavia) is a Serbian middle-distance runner.

 She won two gold medals at 2011 European Athletics Junior Championships in Tallinn and was the junior champion at the 2012 European Cross Country
Championships.

 She has also been a medallist in the 1500 metres at the World Junior and World Youth Championships and a gold medalist at the 2013 European U23 Championship.
</item 9>

<item 10>
Graph query row:
event: 1997 European Athletics Junior Championships
held_on: July 24–27
located_in: Ljubljana
event_connection: LOCATED_IN
country: Slovenia
location_connection: LOCATED_IN
</item 10>
=== END ITEM id=1598f23ba2a18c21 ===

=== ITEM id=92a521ae96b8a53d ===
Gold sentences and facts:
Sentence 0: Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.
  Fact 0.0: Corn Ranch is a spaceport.
  Fact 0.1: Corn Ranch is located in the West Texas town of Van Horn, Texas.
  Fact 0.2: Flight tests of the New Shepard are carried out at Corn Ranch.
  Fact 0.3: Flight tests of the New Shepard are carried out by Blue Origin.
Sentence 1: Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its headquarters in Kent, Washington.
  Fact 1.0: Blue Origin is an American privately funded aerospace manufacturer.
  Fact 1.1: Blue Origin is a spaceflight services company.
  Fact 1.2: Blue Origin was set up by Amazon.com founder Jeff Bezos.
  Fact 1.3: Blue Origin has its headquarters in Kent, Washington.

Retrieved items:
<item 1>
Corn Ranch
Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.

 The 165,000-acre (670 km²) land parcel was purchased by Internet billionaire Jeff Bezos.

 The first flight test took place on November 13, 2006 with the goal of providing commercial tourist flights.
</item 1>

<item 2>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 2>

<item 3>
H-T Ranch
H-T Ranch is a historic ranch complex 10 mi west of Amidon, North Dakota.

 The ranch originally consisted of ten buildings; however, only the ranch house and log barn survive.

 The ranch served as the headquarters of the Little Missouri Cattle Company, which was run by Arthur Clark Hidekoper.

 Hidekober established the ranch in the 1880s, and by the end of the decade, it had become the "most notable" ranch in the area.

 The ranch house, called Shackford, was built in 1896; its uncommon style resembles a bungalow but also borrows from other styles such as the Stick style.

 By 1906, the "Fargo Forum" described the ranch as "the biggest and most important [ranch] in the state" and "one of the largest horse raising outlets in the
world".

 In the same year, Hidekoper sold the 70,000 acre ranch; the sale was the largest land deal in North Dakota history.

 After the sale, a land company reduced the ranch to 5000 acre ; it was later used as a dude ranch in the 1920s.
</item 3>

<item 4>
SyberJet Aircraft
SyberJet Aircraft is the manufacturer of the SJ30 light business jet.

 The company's headquarters is in Cedar City, Utah adjacent to the Cedar City Regional Airport with additional engineering offices and manufacturing, service,
repair and fatigue test facilities near and on the San Antonio International Airport in San Antonio, Texas.
</item 4>

<item 5>
Northern California TRACON
Northern California TRACON (NCT) (Terminal Radar Approach Control), or NorCal TRACON for short, is an air traffic control facility that provides safety alerts,
separation, and sequencing of air traffic arriving, departing, and transiting the airspace and airports in Northern California.

 Located in Rancho Cordova near Sacramento, NCT controls airspace over 19000 square miles, and serves Reno International Airport, Sacramento International
Airport, San Jose International Airport, Oakland International Airport, and San Francisco International Airport, plus 19 other smaller airports with air traffic
control towers.

 NCT is the 3rd busiest TRACON in America.

 NorCal TRACON is the step between local control (in an airport's control tower) and Air Route Traffic Control Center (ARTCC), in this case, Oakland Center
(ICAO code: ZOA).

 San Francisco International Airport is the 2nd largest airport in California and the largest airport serving Northern California.
</item 5>

<item 6>
Blue Origin
Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its
headquarters in Kent, Washington.

 The company is developing technologies to enable private human access to space with the goal to dramatically lower costs and increase reliability.

 Blue Origin is employing an incremental approach from suborbital to orbital flight, with each developmental step building on its prior work.

 The company motto is ""Gradatim Ferociter"", Latin for "Step by Step, Ferociously".

 Blue Origin is developing a variety of technologies, with a focus on rocket-powered Vertical Takeoff and Vertical Landing (VTVL) vehicles for access to
suborbital and orbital space.

 The company's name refers to the blue planet, Earth, as the point of origin.
</item 6>

<item 7>
GoJet Airlines
GoJet Airlines LLC is a company headquartered in Bridgeton, Missouri, United States.

 Wholly owned by Trans States Holdings, it has 1670 employees.

 It operates commuter feeder services under the United Express and Delta Connection names.

 Go Jet Airlines has crew bases at Lambert-Saint Louis International Airport, O'Hare International Airport, Denver International Airport, and Raleigh-Durham
International Airport.

 Flights are currently operated out of United's hubs at O'Hare International Airport and Denver International Airport, as well as Lambert-Saint Louis
International Airport.

 GoJet's Delta Connection flights currently operate out of Lambert-Saint Louis International Airport , Detroit Metropolitan Airport and Raleigh-Durham
International Airport.

 Its call sign "Lindbergh" is named after aviation pioneer Charles Lindbergh, who flew the "Spirit of St. Louis" solo across the Atlantic Ocean in 1927, the
first person to do so.
</item 7>

<item 8>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 8>

<item 9>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 9>

<item 10>
M7 Aerospace
M7 Aerospace LP is an aerospace company with its headquarters on the property of San Antonio International Airport in Uptown San Antonio, Texas, United States.
</item 10>
=== END ITEM id=92a521ae96b8a53d ===

=== ITEM id=702c3ebec16e919c ===
Gold sentences and facts:
Sentence 0: Sasson Gabai (Hebrew: ששון גבאי‎ ‎ ; born 24 November 1947) is an Israeli actor.
  Fact 0.0: Sasson Gabai's Hebrew name is ששון גבאי.
  Fact 0.1: Sasson Gabai was born on 24 November 1947.
  Fact 0.2: Sasson Gabai is an Israeli actor.
Sentence 1:  Winner of an Ophir Award, Asia Pacific Screen Award (nominated), European Film Awards and Jerusalem Film Festival in 2007 as Best Actor in a highly acclaimed Israeli film "The Band's Visit".
  Fact 1.0: Sasson Gabai won an Ophir Award.
  Fact 1.1: Sasson Gabai was nominated for an Asia Pacific Screen Award.
  Fact 1.2: Sasson Gabai won a European Film Awards award.
  Fact 1.3: Sasson Gabai won a Jerusalem Film Festival award.
  Fact 1.4: These awards were won in 2007.
  Fact 1.5: The awards were for Best Actor.
  Fact 1.6: The award-winning role was in the highly acclaimed Israeli film The Band's Visit.
Sentence 2: The Band's Visit (Hebrew: ביקור התזמורת - "Bikur Ha-Tizmoret") is a 2007 Israeli film directed by Eran Kolirin.
  Fact 2.0: The Band's Visit is a 2007 Israeli film.
  Fact 2.1: The Band's Visit is written in Hebrew as ביקור התזמורת.
  Fact 2.2: The Hebrew title of The Band's Visit is Bikur Ha-Tizmoret.
  Fact 2.3: The Band's Visit was directed by Eran Kolirin.

Retrieved items:
<item 1>
Beyond the Sea (1991 film)
Beyond the Sea (translit.

 Me'ever Layam) is a 1991 Israeli drama film directed by Jacob Goldwasser.

 It stars Aryeh Moskona, Dafna Rechter, Moti Giladi.

 The film, a family drama set in the 1960s, was critically acclaimed, and won the Ophir Award for Best Film.

 The film was selected as the Israeli entry for the Best Foreign Language Film at the 64th Academy Awards, but was not accepted as a nominee.
</item 1>

<item 2>
Sh'Chur
Sh'Chur is a 1994 Israeli drama film starring Gila Almagor, Ronit Elkabetz and Hanna Azoulay-Hasfari.

 It was written by Hanna Azoulay Hasfari and directed by her partner Shmuel Hasfari.

 Sh'Chur received critical acclaim and was the 1994 official Israeli submission for the Academy Award for Best Foreign Language Film.

 It was also awarded the Ophir Award for best film by the Israeli Film and Television Academy.

 After its release, the film garnered various discussions in the Israeli press over its representation of the Moroccan community in Israel.
</item 2>

<item 3>
The Band's Visit
The Band's Visit (Hebrew: ביקור התזמורת - "Bikur Ha-Tizmoret") is a 2007 Israeli film directed by Eran Kolirin.
</item 3>

<item 4>
Foxtrot (2017 film)
Foxtrot (Hebrew: פוֹקְסטְרוֹט‎ ‎ ) is a 2017 Israeli drama film directed by Samuel Maoz.

 It was screened in the competition section of the 74th Venice International Film Festival where it won the Grand Jury Prize Silver Lion.

 It was also screened in the Special Presentations section at the 2017 Toronto International Film Festival.

 It won the Ophir Award for Best Film, therefore becoming the Israeli entry for the Best Foreign Language Film at the 90th Academy Awards.
</item 4>

<item 5>
Joseph Cedar
Yossef (Joseph) Cedar (Hebrew: יוסף סידר; born August 31, 1968) is an Israeli film director and screenwriter.

 He has won a Silver Bear and an Ophir Award for Best Director, and an Ophir Award for writing a Best Screenplay.

 He also won the best screenplay award at the 2011 Cannes Film Festival for his film "Footnote" (2011).
</item 5>

<item 6>
Yana's Friends
Yana's Friends (Hebrew: החברים של יאנה, "HaHaverim shel Yana"‎ ‎ ) is a 1999 Israeli film directed by Arik Kaplun.

 script editor: Savi Gabizon.

 Critically acclaimed, it won 10 Israeli Academy Awards including the Ophir Award for Best Picture.

 It also won the Crystal Globe at the Karlovy Vary International Film Festival in 1999.

 The film has a very rare 100% rating on the film website Rotten Tomatoes based on 30 reviews.
</item 6>

<item 7>
Hunting Elephants
Hunting Elephants (Hebrew: לצוד פילים‎ ‎ , "Latzud Pilim", "lit."

 To Hunt Elephants) is a 2013 Israeli crime comedy film directed by Reshef Levi.

 It stars, among others, Sasson Gabai, Moni Moshonov and Patrick Stewart.

 It was released to theaters in Israel on July 4, 2013.
</item 7>

<item 8>
The Farewell Party
The Farewell Party (Hebrew: מיתה טובה‎ ‎ Mita Tova) is a 2014 Israeli drama film about the use of a euthanasia device, directed by Tal Granit and Sharon Maymon.

 It was nominated for the Ophir Award for Best Film.

 The film was screened in the Venice Days section of the 71st Venice International Film Festival and has been selected to be screened in the Contemporary World
Cinema section at the 2014 Toronto International Film Festival.

 The film was also screened in the 44th edition of the International Film Festival Rotterdam, reaching the second place in the IFFR audience award.
</item 8>

<item 9>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 9>

<item 10>
Sasson Gabai
Sasson Gabai (Hebrew: ששון גבאי‎ ‎ ; born 24 November 1947) is an Israeli actor.

 Winner of an Ophir Award, Asia Pacific Screen Award (nominated), European Film Awards and Jerusalem Film Festival in 2007 as Best Actor in a highly acclaimed
Israeli film "The Band's Visit".
</item 10>
=== END ITEM id=702c3ebec16e919c ===

=== ITEM id=ce11d2fc205f9856 ===
Gold sentences and facts:
Sentence 0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.
  Fact 0.0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music.
  Fact 0.1: More Hot Rocks was released by ABKCO Records.
  Fact 0.2: ABKCO Records is run by former manager Allen Klein.
  Fact 0.3: Allen Klein usurped control of the Rolling Stones' Decca/London material in 1970.
  Fact 0.4: More Hot Rocks was released after the Rolling Stones departed from Decca.
  Fact 0.5: More Hot Rocks was released after the Rolling Stones departed from Allen Klein.
Sentence 1: Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording artists.
  Fact 1.0: Allen Klein was born December 18, 1931.
  Fact 1.1: Allen Klein died July 4, 2009.
  Fact 1.2: Allen Klein was an American businessman.
  Fact 1.3: Allen Klein was a music publisher.
  Fact 1.4: Allen Klein was a writers' representative.
  Fact 1.5: Allen Klein was a filmmaker.
  Fact 1.6: Allen Klein was a record label executive.
  Fact 1.7: Allen Klein was most noted for his tough persona.
  Fact 1.8: Allen Klein was most noted for aggressive, innovative negotiation tactics.
  Fact 1.9: Many of Allen Klein's negotiation tactics established higher industry standards for compensating recording artists.
Sentence 2:  He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming one of the most powerful individuals in the music industry during his era.
  Fact 2.0: Allen Klein first scored massive monetary windfalls for Buddy Knox.
  Fact 2.1: Allen Klein first scored massive contractual windfalls for Buddy Knox.
  Fact 2.2: Allen Klein first scored massive monetary windfalls for Jimmy Bowen.
  Fact 2.3: Allen Klein first scored massive contractual windfalls for Jimmy Bowen.
  Fact 2.4: Buddy Knox and Jimmy Bowen were one-hit rockabillies of the late 1950s.
  Fact 2.5: Allen Klein later parlayed his early successes into a position managing Sam Cooke.
  Fact 2.6: Allen Klein eventually managed the Beatles.
  Fact 2.7: Allen Klein eventually managed the Rolling Stones.
  Fact 2.8: Allen Klein managed the Beatles and the Rolling Stones simultaneously.
  Fact 2.9: Allen Klein managed many other artists.
  Fact 2.10: Allen Klein became one of the most powerful individuals in the music industry during his era.

Retrieved items:
<item 1>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 1>

<item 2>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 2>

<item 3>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 3>

<item 4>
Hot Rocks 1964–1971
Hot Rocks 1964–1971 is the first compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who gained control of the
band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1971, it proved to be The Rolling Stones' biggest-selling release of their career and an enduring and popular retrospective.
</item 4>

<item 5>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 5>

<item 6>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 6>

<item 7>
Bigtop Records
Bigtop Records was an American record label started by music executive Johnny Bienstock and the major music publisher Hill & Range Music and was co-owned along
with Big Top Record Distributors (sic).

 Hit artists included Del Shannon, Johnny and the Hurricanes, Lou Johnson, Sammy Turner, Don and Juan, and Toni Fisher.

 Big Top Record Distributors also distributed Paul Case's Dunes Records label in the early 60's, which had hits from Ray Peterson ("Corrina, Corrina") and
Curtis Lee ("Pretty Little Angel Eyes"), both records produced by Phil Spector.

 Bell Records briefly distributed Bigtop prior to the label closing, around 1965.

 "Bigtop" also released two Mad Magazine music parody themed albums; "Mad Twists Rock 'N' Roll" and "Fink Along With Mad" in 1963.
</item 7>

<item 8>
More Hot Rocks (Big Hits &amp; Fazed Cookies)
More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who
usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1972 as the follow-up to the hugely successful "Hot Rocks 1964–1971", it was another success for Klein and ABKCO.
</item 8>

<item 9>
Sargent House
Sargent House is an American management company and record label based in Los Angeles.

 It was founded in June 2006 by Cathy Pellow, who is a music video commissioner for Atlantic Records and also owns a music video production company called
Refused TV.

 Pellow began managing the rock outfit RX Bandits in 2006.

 The band needed to release its new album "...And the Battle Begun", so Pellow decided to launch her own record label (also to be able to do things differently
from the traditional model of record labels).

 From that point onward, Pellow continued to manage bands and release records, all under the Sargent House banner.
</item 9>

<item 10>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 10>
=== END ITEM id=ce11d2fc205f9856 ===

=== ITEM id=6fda49fde8bcbe8a ===
Gold sentences and facts:
Sentence 0: Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of Pinocchio" by Carlo Collodi.
  Fact 0.0: Pinocchio is a 1940 American film.
  Fact 0.1: Pinocchio is an animated musical fantasy film.
  Fact 0.2: Pinocchio was produced by Walt Disney Productions.
  Fact 0.3: Pinocchio is based on the Italian children's novel 'The Adventures of Pinocchio'.
  Fact 0.4: The novel 'The Adventures of Pinocchio' was written by Carlo Collodi.
Sentence 1:  It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
  Fact 1.0: Pinocchio was the second animated feature film produced by Disney.
  Fact 1.1: Pinocchio was made after the success of 'Snow White and the Seven Dwarfs'.
  Fact 1.2: Snow White and the Seven Dwarfs is from 1937.
Sentence 2: Home on the Range is a 2004 American animated musical western comedy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.
  Fact 2.0: Home on the Range is a 2004 American film.
  Fact 2.1: Home on the Range is an animated musical western comedy film.
  Fact 2.2: Home on the Range was produced by Walt Disney Feature Animation.
  Fact 2.3: Home on the Range was released by Walt Disney Pictures.

Retrieved items:
<item 1>
Pinocchio (1957 TV-musical)
Pinocchio, a 1957 television production of "Pinocchio", is a live (and live-action) musical version starring Mickey Rooney in the title role of the puppet who
wishes to become a real boy.

 Based on the novel by Carlo Collodi which also inspired the classic Walt Disney animated film, this version featured a now-forgotten new score by Alec Wilder
and William Engvick.

 It was telecast once on NBC as a television special, and, as far as is known, never rebroadcast by NBC, or even restaged with a different cast as was Rodgers
and Hammerstein's "Cinderella".

 Nor has it ever been issued on VHS or DVD.

 Other notable actors who appeared in the special included Walter Slezak (as Geppetto), Fran Allison (as the Blue Fairy), Martyn Green (as the Fox), Jerry
Colonna (as a Ringmaster), and Stubby Kaye as a Town Crier, a role he repeated in Wilder and Engvick's 1958 television musical, "Hansel and Gretel".

 "Pinocchio" was directed by noted Broadway choreographer Hanya Holm.
</item 1>

<item 2>
Home on the Range (2004 film)
Home on the Range is a 2004 American animated musical western comedy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.

 The 45th Disney animated feature film, it was the last 2D animated Disney film released until "The Princess and the Frog" in 2009.

 Named after the popular country song of the same name, "Home on the Range" features the voices of Roseanne Barr, Judi Dench, Jennifer Tilly, Cuba Gooding Jr.,
Randy Quaid, and Steve Buscemi.

 The film is set in the Old West, and centers on a mismatched trio of dairy cows—brash, adventurous Maggie; prim, proper Mrs. Caloway; and ditzy, happy-go-lucky
Grace.

 The three cows must capture an infamous cattle rustler named Alameda Slim for his bounty in order to save their idyllic farm from foreclosure.

 Aiding them in their quest is Lucky Jack, a feisty, peg-legged rabbit, but a selfish horse named Buck, eagerly working in the service of Rico, a famous bounty
hunter, seeks the glory for himself.
</item 2>

<item 3>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 3>

<item 4>
Frankie Darro
Frankie Darro (born Frank Johnson, Jr.

 December 22, 1917 – December 25, 1976) was an American actor and later in his career a stuntman.

 He began his career as a child actor in silent films, progressed to lead roles and co-starring roles in adventure, western, dramatic, and comedy films, and
later became a character actor and voice-over artist.

 He is known for his role as Lampwick, the unlucky boy turned into a donkey in Walt Disney's 2nd animated feature, "Pinocchio", which was originally released in
February 1940.
</item 4>

<item 5>
Return to Never Land
Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by
Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.

 The film is a sequel to Walt Disney Feature Animation 1953 film "Peter Pan", It is based on J. M. Barrie's novel "Peter and Wendy", and had a worldwide gross
of $109 million.
</item 5>

<item 6>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 6>

<item 7>
Get a Horse!
Get a Horse!

 is a 2013 American 3D animated slapstick comedy short film, produced by Walt Disney Animation Studios.

 Combining black-and-white hand-drawn animation and color CGI animation, the short features the characters of the late 1920s "Mickey Mouse" cartoons, and
features archival recordings of Walt Disney in his posthumous role as Mickey Mouse.

 It is the first original Mickey Mouse theatrical animated short since "Runaway Brain" (1995), and the first appearance of Oswald the Lucky Rabbit in a Disney
animated production in 85 years.
</item 7>

<item 8>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 8>

<item 9>
Pinocchio (1940 film)
Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of
Pinocchio" by Carlo Collodi.

 It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
</item 9>

<item 10>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 10>
=== END ITEM id=6fda49fde8bcbe8a ===

=== ITEM id=247eb91f04e83f02 ===
Gold sentences and facts:
Sentence 0:  In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
  Fact 0.0: In Crash, there is no betting, unlike in Brag.
  Fact 0.1: In Crash, players aim to reach a total of 11 points.
  Fact 0.2: In Crash, the 11 points are gained over successive deals.
Sentence 1: Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.
  Fact 1.0: Three-card brag is a 16th-century British card game.
  Fact 1.1: Three-card brag is the British national representative of the vying or 'bluffing' family of gambling games.

Retrieved items:
<item 1>
Three card brag
Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.

 Brag is a direct descendant of the Elizabethan game of Primero and one of the several ancestors to poker, just varying in betting style and hand rankings.
</item 1>

<item 2>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 2>

<item 3>
Écarté
Écarté is a two-player card game originating from France, the word literally meaning "discarded".

 It is a trick-taking game, similar to whist, but with a special and eponymous discarding phase.

 It is closely related to Euchre, a card game played mainly in the United States.

 Écarté was popular in the 19th century, but is now rarely played.
</item 3>

<item 4>
Quadrille (card game)
Quadrille is a card game that was popular in the 18th century.

 A variant of the Spanish card game Ombre, it is played by four players in pairs, with a deck of 40 cards (the 8's, 9's and 10's being removed).

 By the end of the 19th century, the card game had fallen out of fashion.
</item 4>

<item 5>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 5>

<item 6>
Put (card game)
Put is an English tavern trick-taking card game first recorded in the 16th century and later castigated by 17th century moralists as one of ill repute.

 It belongs to a very ancient family of card games and clearly relates to a group known as Trut, "Truque", also "Tru", and the South American game Truco.

 Its more elaborate version is the Spanish game of Truc, which is still much played in many parts of Southern France and Spain.
</item 6>

<item 7>
Ombre
The historical importance of Ombre in the field of playing cards is the fact that it was the first card game in which a trump suit was established by bidding
rather than by the random process of turning the first card of the stock.

 This game developed from Triunfo, though it was from L'Hombre that the idea of bidding was adopted into other card games such as Skat, and Tarot, which owes
Hombre a good portion of its betting system as well.

 The game continued to be in vogue almost in every corner of Europe from the late 17th through the 18th centuries.
</item 7>

<item 8>
Triomphe
Triomphe (French for triumph) is a card game dating from the late 15th century.

 It most likely originated in France or Spain (as triunfo) and later spread to the rest of Europe.

 When the game arrived in Italy, it shared a similar name with the pre-existing game and deck known as "trionfi" (tarot).

 While trionfi has a fifth suit that acts as permanent trumps, triomphe randomly selects one of the existing four suits as trumps.

 Another common feature of this game is the robbing of the stock.

 Triomphe became so popular that during the 16th century the earlier game of trionfi was gradually renamed tarocchi, tarot, or tarock.

 This game is the origin of the English word "trump" and is the ancestor of many trick-taking games like Euchre (via Écarté) and Whist (via Ruff and Honours).
</item 8>

<item 9>
Egyptian Ratscrew
Egyptian Ratscrew (also known as Egyptian Ratkiller, Egyptian War, and by other names) is a card game of the matching family of games.

 The game is similar to the 19th century British card game Beggar-My-Neighbour, with the added concept of "slapping" cards when certain combinations are played,
similar to and perhaps borrowed from Slapjack.
</item 9>

<item 10>
Crash (card game)
Crash is a British card game extension of Nine-card Brag.

 In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
</item 10>
=== END ITEM id=247eb91f04e83f02 ===

=== ITEM id=075d2a1d885e8380 ===
Gold sentences and facts:
Sentence 0:  He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
  Fact 0.0: He received an excellent education in Arabic.
  Fact 0.1: He received an excellent education in Greek.
  Fact 0.2: He studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
Sentence 1: Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.
  Fact 1.0: Abul ʿAla Al-Maʿarri's Arabic name is أبو العلاء المعري (Abū al-ʿAlāʾ al-Maʿarrī).
  Fact 1.1: Abul ʿAla Al-Maʿarri's full name is أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري (Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī).
  Fact 1.2: Abul ʿAla Al-Maʿarri was born in December 973.
  Fact 1.3: Abul ʿAla Al-Maʿarri died in May 1057.
  Fact 1.4: Abul ʿAla Al-Maʿarri was blind.
  Fact 1.5: Abul ʿAla Al-Maʿarri was an Arab philosopher.
  Fact 1.6: Abul ʿAla Al-Maʿarri was an Arab poet.
  Fact 1.7: Abul ʿAla Al-Maʿarri was an Arab writer.

Retrieved items:
<item 1>
Al-Mutanabbi
Al-Mutanabbi had a great pride in himself through his poetry.

 Among the topics he discussed were courage, the philosophy of life, and the description of battles.

 Many of his poems were and still are widely spread in today's Arab world and are considered to be proverbial.
</item 1>

<item 2>
Al-Mutanabbi
Abu at-Tayyib Ahmad bin Al-Husayn al-Mutanabbi al-Kindi (Arabic: أبو الطيب أحمد بن الحسين المتنبّي الكندي‎ , "Abū ṭ-Ṭayyib ʾAḥmad bin al-Ḥusayn al-Muṫanabbī
al-Kindī " ‎ ) (915 – 23 September 965 CE) was an Arab poet.

 He is considered as one of the greatest poets in the Arabic language and is the most prominent and most influential poet in the Arab world and much of his work
has been translated into over 20 languages worldwide.

 Much of his poetry revolves around praising the kings he visited during his lifetime.

 Some consider his 326 poems to be a great representation of his life story.

 He started writing poetry when he was nine years old.

 He is well known for his sharp intelligence and wittiness.

 Al-Mutanabbi had a great pride in himself through his poetry.
</item 2>

<item 3>
Antarah ibn Shaddad
Antarah ibn Shaddad (Arabic: عنترة بن شداد العبسي‎ ‎ , "ʿAntarah ibn Shaddād al-ʿAbsī";  525–608), also known as ʿAntar, was a pre-Islamic Arab knight and poet,
famous for both his poetry and his adventurous life.

 His chief poem forms part of the "Mu'allaqāt", the collection of seven "hanging odes" legendarily said to have been suspended in the Kaaba.

 The account of his life forms the basis of a long and extravagant romance.
</item 3>

<item 4>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 4>

<item 5>
Abd Allah ibn Abbas
Abd Allah ibn Abbas (Arabic: عبد الله ابن عباس‎ ‎ ) or ′Abd Allah ibn al-′Abbas otherwise called (Ibn Abbas; Al-Habr; Al-Bahr; The Doctor; The Sea) was born c.
619 CE.

 He was the son of Al-‘Abbas ibn ‘Abd al-Muttalib, an uncle of the Islamic prophet Muhammad, and a nephew of the Maymunah bint al-Harith, who later became
Muhammad's wife.

 He was one of Muhammad's cousins and one of the early Qur'an scholars.
</item 5>

<item 6>
Shaykh Muhammad ibn Thaleb
Muhammad ibn Thalib ibn Abd Allah ibn Ni`mat Allah ibn Sadr ad-Din ibn Shaykh Baha' ad-Din ash-Shirazi (Arabic: محمد ابن طالب عبدالله ابن نعمات الله ابن صدر
الدين ابن شيخ بهاءالدين الشيرازي‎ ‎ ) was a 15th-century Persian physician from Shiraz, Iran.
</item 6>

<item 7>
Ibn al-Khatib
Lisan ad-Din ibn al-Khatib (Born 16 November 1313, Loja– died 1374, Fes, Morocco) (Full name Muhammad ibn Abd Allah ibn Said ibn Ali ibn Ahmad al-Salmani) was a
Moroccan Arab polymath poet, writer, historian, philosopher, physician and politician from Emirate of Granada.

 Some of his poems decorate the walls of the Alhambra in Granada.
</item 7>

<item 8>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 8>

<item 9>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 9>

<item 10>
Al-Maʿarri
Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad
ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.

 Al-Maʿarri held and expressed an irreligious world view which was met with controversy, but in spite of it, he is regarded as one of the greatest classical
Arabic poets.
</item 10>
=== END ITEM id=075d2a1d885e8380 ===

=== ITEM id=772a2d28689a626f ===
Gold sentences and facts:
Sentence 0:  Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any affiliation with the work.
  Fact 0.0: The film takes its title from an album by Leonard Cohen.
  Fact 0.1: Ira Nadel wrote a biography of Leonard Cohen.
  Fact 0.2: The subject of the film is not Leonard Cohen.
  Fact 0.3: The subject of the film has no affiliation with the work.
Sentence 1: Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and painter.
  Fact 1.0: Leonard Norman Cohen was born on September 21, 1934.
  Fact 1.1: Leonard Norman Cohen died on November 7, 2016.
  Fact 1.2: Leonard Norman Cohen was a Canadian singer.
  Fact 1.3: Leonard Norman Cohen was a Canadian songwriter.
  Fact 1.4: Leonard Norman Cohen was a Canadian musician.
  Fact 1.5: Leonard Norman Cohen was a Canadian poet.
  Fact 1.6: Leonard Norman Cohen was a Canadian novelist.
  Fact 1.7: Leonard Norman Cohen was a Canadian painter.

Retrieved items:
<item 1>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 1>

<item 2>
Daddy's Hands (band)
Daddy's Hands was a Canadian experimental rock band formed in Victoria, British Columbia, Canada, by singer/guitarist Dave Wenger, bassist Emily Bauslaugh,
drummer Jonah Fortune, and saxophonist Jonathan Pollard.

 Dave Wenger and Emily Bauslaugh had been involved in the British Columbian hardcore/emo scene before forming the band, being in bands M Blanket and Ache Hour
Credo, and the band Floragore respectively.

 Daddy's Hands has released two demo tapes, one EP and two full-length albums.

 The band gigged extensively in the 90s, even playing a radio show with Nardwuar.

 Despite achieving little commercial success, the band has been referred to as highly influential in the West Coast music scene.

 Lead singer Wenger, highly regarded by some critics, has been referred to as Montreal's Peter Laughner.
</item 2>

<item 3>
Love Can't Wait (Nick Carter song)
"Love Can't Wait" is a song by American singer-songwriter Nick Carter, released as the lead Canadian single from his second studio album, "I'm Taking Off".

 The single was released on June 21, 2011.

 The song only appears on the Canadian version of the album, and has not been released in any other territories.

 The music video for the song features Carter performing the song with the backdrop of a motorway system.

 The music video also featured Canadian pop star Shawn Desman, who co-wrote the song.

 The single peaked at #2 on the Canadian Singles Chart, based entirely on download sales.

 Carter ran a competition for amateur mixers to remix the track, with the winning remixer earning $1500 and their remix appearing on Carter's remix album, "I'm
Taking Off: Relaunched and Remixed".
</item 3>

<item 4>
Various Positions Tour
The Various Positions Tour was a concert tour by Leonard Cohen, in support of his album "Various Positions", released in 1984 (Canada) and 1985 in Europe and
the United States.
</item 4>

<item 5>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 5>

<item 6>
Various Positions (film)
Various Positions is a 2002 film by Vancouver, BC, lawyer and filmmaker Ori Kowarsky, starring Carly Pope and Tygh Runyan.

 "Various Positions" won the 2002 Prix de Montréal at the Montreal World Film Festival.

 Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any
affiliation with the work.
</item 6>

<item 7>
New World Man
"New World Man" is a track from the 1982 album "Signals" by Canadian rock band Rush.

 The song was the last and quickest composed song on the album, stemming from a suggestion by then-Rush producer Terry Brown to even out the lengths of the two
sides of the cassette version.

 It went to #1 (on the "RPM" national singles chart) in Canada, where it remained for two weeks in October 1982.

 Less successful in the United States, it nonetheless remains Rush's only American Top 40 hit, peaking at #21 on the "Billboard" singles chart for three weeks
in October and November 1982.

 It also topped the "Billboard" Top Tracks chart for two weeks (their first single to do so).

 "New World Man" also reached #42 in the UK; a remixed version released as a double A-side with "Countdown" later reached #36 in the UK in early 1983.
</item 7>

<item 8>
List of songs recorded by Motion City Soundtrack
American rock band Motion City Soundtrack recorded songs for six studio albums, various soundtracks, compilations, and non-album singles.

 The majority of the group's original material was written by vocalist and guitarist Justin Pierre.

 The band also recorded cover versions of other artists' songs at various points throughout their career, including by the Beatles, Limbeck, the Police, R.E.M.,
the Rentals, Rilo Kiley, Trampled by Turtles, and the Weakerthans.

 In all, the group recorded 107 songs, nine of which were covers.
</item 8>

<item 9>
Guillaume Morissette
Guillaume Morissette (born 1984) is a Canadian fiction writer and poet based in Montreal, Quebec.

 His work has frequently been associated with the Alt Lit movement, with Dazed & Confused magazine describing him as "Canada's Alt Lit poster boy."

 He has published stories, poems and essays online and in print, in venues such as "Maisonneuve", "Little Brother", "Broken Pencil", "Shabby Doll House" and
"Thought Catalog", and was listed as one of CBC Books' "Writers to Watch" for 2014.
</item 9>

<item 10>
For Those Who Think Young (album)
For Those Who Think Young, appearing on the album cover as (for those who think young) and originally to be entitled for those who think jung, was the third
album by Canadian new wave band Rough Trade; it was released in 1981 (True North TN-48 in Canada; Boardwalk NB-33261-1, US; Big Time, UK; CBS 85385, The
Netherlands).

 It climbed to #9 in Canada on the "RPM" Top 50 Albums Chart on November 7, 1981 (putting it at #1 on the CANCON Chart listing), and held the position for three
weeks, dropping out of the Top 50 after sixteen weeks on February 6 of the following year.

 It was certified gold in Canada by the CRIA on November 1, 1981.

 The single "All Touch" gave the band its biggest commercial success, reaching #12 in Canada on the "RPM" Top 50 Singles Chart (#2 on the CANCON Chart) and #58
on the U.S. ""Billboard" Hot 100".
</item 10>
=== END ITEM id=772a2d28689a626f ===

=== ITEM id=850ff39e718254f3 ===
Gold sentences and facts:
Sentence 0:  In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
  Fact 0.0: In 2003, a person referred to as "he" received an award for his actions during the Battle of Qala-i-Jangi.
  Fact 0.1: The Battle of Qala-i-Jangi took place in late November to early December 2001.
Sentence 1: The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November 25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
  Fact 1.0: The Battle of Qala-i-Jangi is also incorrectly referred to as the "Battle of Mazar-i-Sharif".
  Fact 1.1: The Battle of Qala-i-Jangi was a prisoner-of-war camp uprising.
  Fact 1.2: The Battle of Qala-i-Jangi took place between November 25 and December 1, 2001.
  Fact 1.3: The Battle of Qala-i-Jangi took place in northern Afghanistan.
  Fact 1.4: The Battle of Qala-i-Jangi followed the armed intervention by United States-led coalition forces.
  Fact 1.5: The United States-led coalition forces intervened with armed forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan.
  Fact 1.6: The coalition forces had accused the Taliban's Islamic Emirate of Afghanistan of harboring al-Qaeda operatives.

Retrieved items:
<item 1>
Sayeed Salahudeen
Syed Mohammed Yusuf Shah, popularly known as Syed Salahudeen, is the head of Hizb-ul-Mujahideen, a terrorist group operating in Kashmir, and head of an alliance
of anti-India militant groups, the United Jihad Council, that works to annex the Indian state of Jammu and Kashmir to Pakistan .

 Salahuddin vowed to block any peaceful resolution to the Kashmir conflict, threatened to train more Kashmiri suicide bombers, and vowed to turn the Kashmir
valley “into a graveyard for Indian forces.”

 He is listed on the NIA Most Wanted list.

 On 26 June 2017, he was named as a Specially Designated Global Terrorist by the US Department Of State.

 After being designated as a Global terrorist, he addressed the Pakistani Media at Muzaffarabad’s Centre Press Club, claiming that "The declaration was a joint
move by the US, Israel, and India to express their animosity towards Pakistan."
</item 1>

<item 2>
The Great War for Civilisation
The Great War for Civilisation: The Conquest of the Middle East is a book published in 2005 by the award-winning English journalist Robert Fisk.

 The book is a compilation of many of the articles Fisk wrote when he was serving as a correspondent in the Middle East for "The Times" and "The Independent".

 The book revolves around several key themes regarding the history of the modern Middle East: the Arab-Israeli conflict, the Soviet invasion of Afghanistan, the
Persian Gulf War as well as the 2003 Iraq War (United States invasion of Iraq) as well as other regional conflicts such as the Armenian Genocide and the
Algerian Civil War.

 The "Great War for Civilisation" is the second book Fisk has written about the Middle East with the first one, "Pity the Nation", (Nation Books, 2002) being
about the Lebanese Civil War.
</item 2>

<item 3>
Iraultza
Iraultza (Basque for Revolution) was a small Basque militant armed group of leftist tendency.

 Seven of the group would die preparing explosive devices.

 On June 27, 1986, a bomb claimed by Iraultza mangled a young worker called José Miguel Moros Peña in Portugalete.

 The victim would die in Cruces Hospital (Barakaldo) on August 13, 1986.

 The group is also known for being described by the United States Government as "probably [having] committed more bombings against American business interests
than any other European terrorist group".

 Soon after, the group, which had been active for about a decade with small attacks of explosives that never aimed to cause any personal injuries, announced its
dissolution.

 When the group was operational, it is estimated that it had approximately 20 members.
</item 3>

<item 4>
The Good Terrorist
The Good Terrorist is a 1985 political novel written by the British novelist Doris Lessing.

 It was first published in September that year by Jonathan Cape in the United Kingdom and by Alfred A. Knopf in the United States.

 The book's protagonist is the naïve drifter Alice, who squats with a group of radicals in London and is drawn into their terrorist activities.
</item 4>

<item 5>
National Intelligence Reform Medal
The National Intelligence Reform Medal is an award of the United States Intelligence Community's National Intelligence Awards Program that recognizes
extraordinary service in implementing the Intelligence Reform and Terrorism Prevention Act.

 The medal ranks below the National Intelligence Superior Service Medal, but above the National Intelligence Exceptional Achievement Medal.

 Established 23 May 2007 with the creation of the National Intelligence Awards Program, it was retired in November 2010.
</item 5>

<item 6>
Battle of Qala-i-Jangi
The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November
25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's
Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
</item 6>

<item 7>
Enemy Combatant (book)
Enemy Combatant is a memoir by British Muslim, Moazzam Begg, co-written by Victoria Brittain, former Associate Foreign Editor for "The Guardian", about Begg's
detention by the government of the United States of America in Bagram Detention Facility and at Camp Echo, Guantanamo Bay and his life prior to that detention.

 It was published in Britain as "Enemy Combatant: A British Muslim's Journey To Guantanamo and Back" (ISBN  ), and in the US as "Enemy Combatant: My
Imprisonment at Guantanamo, Bagram, and Kandahar" (ISBN  ).

 In the US, the foreword was written by David Ignatius of The Washington Post.
</item 7>

<item 8>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 8>

<item 9>
Combat Terrorist Organization
Combat Terrorist Organization or Autonomous Combat Terrorist Organization (rus.

 Боевая Террористическая Организация) was a Russian radical Neo-Nazi terrorist group based in Saint Petersburg from 9 August 2003 to 2006.

 It was formed by two members of the "Mad Crowd" skinhead organization, Dmitriy Borovikov and Voevodin Alexey.
</item 9>

<item 10>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 10>
=== END ITEM id=850ff39e718254f3 ===
