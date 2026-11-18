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

=== ITEM id=c64c766eb121ce57 ===
Gold sentences and facts:
Sentence 0:  The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
  Fact 0.0: The ground was home to the Boston Reds.
  Fact 0.1: The Boston Reds played in the Players' League in 1890.
  Fact 0.2: The Boston Reds played in the American Association in 1891.
Sentence 1:  It is home to the Port of Boston, a major shipping facility in the northeastern United States.
  Fact 1.0: The ground is home to the Port of Boston.
  Fact 1.1: The Port of Boston is a major shipping facility.
  Fact 1.2: The Port of Boston is in the northeastern United States.

Retrieved items:
<item 1>
Transportation in Boston
The Boston transportation system includes roadway, subway, regional rail, air, and sea options for passenger and freight transit in Boston, Massachusetts.

 The Massachusetts Port Authority (Massport) operates the Port of Boston, which includes a container shipping facility in South Boston, and Logan International
Airport, in East Boston.

 The Massachusetts Bay Transportation Authority (MBTA) operates bus, subway, short distance rail, and water ferry passenger services throughout the city and
region.

 Amtrak operates passenger rail service to and from major northeastern cities.

 A major bus terminal at South Station is served by varied intercity bus companies.

 The city is bisected by major highways I-90 and I-93, the intersection of which has undergone a major renovation, nicknamed the Big Dig.
</item 1>

<item 2>
Braves Field
Braves Field was a baseball park in the Northeastern United States, located in Boston, Massachusetts.

 Today the site is home to Nickerson Field on the campus of Boston University.

 The stadium was home of the Boston Braves of the National League from 1915–1952, prior to the Braves' move to Milwaukee in 1953.

 The stadium hosted the 1936 Major League Baseball All-Star Game and Braves home games during the 1948 World Series.

 The Boston Red Sox used Braves Field for their home games in the 1915 and 1916 World Series since the stadium had a larger seating capacity than Fenway Park.

 Braves Field was the site of Babe Ruth's final season, playing for the Braves in 1935.

 From 1929 to 1932, the Boston Red Sox played select regular season games periodically at Braves Field.

 On May 1, 1926, Braves Field hosted the longest baseball game in history – 26 Innings.

 The game ended in a 1–1 tie.
</item 2>

<item 3>
New Bedford Bay Sox
The New Bedford Bay Sox are a baseball team that plays in the New England Collegiate Baseball League, a collegiate summer baseball league located in the
northeastern United States region of New England.

 The team is located in New Bedford, Massachusetts.

 New Bedford's Paul Walsh Field serves as the home field of the Bay Sox.

 Th NECBL's players use wooden bats, as opposed to aluminum, and are college baseball players recruited from the top intercollegiate baseball programs in the
United States.

 The franchise has appeared in league finals four times in their history, all as the Torrington Twisters (1997, 1998, 2003, 2006), and have twice played host to
the NECBL All-Star Game as the Twisters (1998, 2008).

 Pat and Beth O'Connor, owners of Little Fenway and Little Wrigley joined the ownership team of the New Bedford Bay Sox in October, 2011, and signed a 3-year
lease to play at Paul Walsh Field.

 In November 2011, former MLB player Rick Miller was announced as the 2012 Bay Sox manager.
</item 3>

<item 4>
Deering Oaks
Deering Oaks is a 55 acre public park in Portland, Maine, which has a baseball diamond, tennis courts, a playground, and a pond.

 It is located west of downtown Portland and is bordered by Deering Avenue on the west, Forest Avenue on the east, Park Avenue to the south and Interstate 295
to the north.

 State Street bisects the park.

 Access is from State Street, Deering Avenue, or Park Avenue.

 The Portland Farmers Market is located on the Park Avenue side of the park.

 The park is listed on the National Register of Historic Places.
</item 4>

<item 5>
Newark Bay
Newark Bay is a tidal bay at the confluence of the Passaic and Hackensack Rivers in northeastern New Jersey.

 It is home to the Port Newark-Elizabeth Marine Terminal, the largest container shipping facility in Port of New York and New Jersey, the third largest and one
of the busiest in the United States.

 An estuary, it is periodically dredged to accommodate ocean-going ships.
</item 5>

<item 6>
Congress Street Grounds
Congress Street Grounds is a former baseball ground located in Boston, Massachusetts.

 The ballpark, as the name implies, was along Congress Street, near the intersection of Thompson Place, and not far from the Fort Point Channel on South Boston
Flats, a newly filled in piece of land on Boston Harbor.

 The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
</item 6>

<item 7>
New Bedford Bay Sox
Miller was succeeded as manager prior to the 2015 season by Westport, MA native and former Boston Red Sox draft pick Kyle Fernandes.

 Fernandes has led the Bay Sox to back-to-back postseason appearances in his two seasons as manager, most recently losing in a three-game series to the eventual
NECBL Champions the Mystic Schooners.
</item 7>

<item 8>
Boston Harbor
Boston Harbor is a natural harbor and estuary of Massachusetts Bay, and is located adjacent to the city of Boston, Massachusetts.

 It is home to the Port of Boston, a major shipping facility in the northeastern United States.
</item 8>

<item 9>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 9>

<item 10>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 10>
=== END ITEM id=c64c766eb121ce57 ===

=== ITEM id=bc97ebc600b6a03d ===
Gold sentences and facts:
Sentence 0: Gasland is a 2010 American documentary written and directed by Josh Fox.
  Fact 0.0: Gasland is a documentary.
  Fact 0.1: Gasland is from 2010.
  Fact 0.2: Gasland is American.
  Fact 0.3: Gasland was written by Josh Fox.
  Fact 0.4: Gasland was directed by Josh Fox.
Sentence 1: To Shoot an Elephant is a 2009 documentary film about the 2008-2009 Gaza War directed by Alberto Arce and Mohammad Rujailahk.
  Fact 1.0: To Shoot an Elephant is a 2009 documentary film.
  Fact 1.1: To Shoot an Elephant is about the 2008-2009 Gaza War.
  Fact 1.2: To Shoot an Elephant was directed by Alberto Arce.
  Fact 1.3: To Shoot an Elephant was directed by Mohammad Rujailahk.

Retrieved items:
<item 1>
Hunting Elephants
Hunting Elephants (Hebrew: לצוד פילים‎ ‎ , "Latzud Pilim", "lit."

 To Hunt Elephants) is a 2013 Israeli crime comedy film directed by Reshef Levi.

 It stars, among others, Sasson Gabai, Moni Moshonov and Patrick Stewart.

 It was released to theaters in Israel on July 4, 2013.
</item 1>

<item 2>
Olympia (1938 film)
Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium
in Berlin, Germany.

 The film was released in two parts: "Olympia 1.

 Teil — Fest der Völker" (Festival of Nations) and "Olympia 2.

 Teil — Fest der Schönheit" (Festival of Beauty).

 It was the first documentary feature film of the Olympic Games ever made.

 Many advanced motion picture techniques, which later became industry standards but which were groundbreaking at the time, were employed —including unusual
camera angles, smash cuts, extreme close-ups, placing tracking shot rails within the bleachers, and the like.

 The techniques employed are almost universally admired, but the film is controversial due to its political context.

 Nevertheless, the film appears on many lists of the greatest films of all time, including "Time" magazine's "All-Time 100 Movies."
</item 2>

<item 3>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 3>

<item 4>
Gasland
Gasland is a 2010 American documentary written and directed by Josh Fox.

 Nominated for an Academy Award for Best Documentary in 2011, the film focuses on communities in the United States affected by natural gas drilling and,
specifically, a method of horizontal drilling into shale formations known as hydraulic fracturing.

 The film was a key mobilizer for the anti-fracking movement, and "brought the term 'hydraulic fracturing' into the nation's living rooms" according to "The New
York Times".

 The film was nominated for the Academy Award for Best Documentary Feature.
</item 4>

<item 5>
Elephant (album)
Elephant is the fourth album by the American alternative rock duo The White Stripes.

 Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the
Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album
charts.
</item 5>

<item 6>
The Last Lions
The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.

 It was shot at the Okavango Delta in Botswana.

 The film premiered at the Palm Springs International Film Festival in January 2011 and was released in select theaters on February 18, 2011.

 The film follows in the tradition of other National Geographic big cat films, such as "India: Land of the Tiger" and "Eye of the Leopard".
</item 6>

<item 7>
The entries survey documentaries, films, TV, and film-related concepts: Los Angeles hip-hop, Aboriginal country music, Latin American politics, oceans, nerd
culture, Iraqi life and war, Bougainville, education, disability, poetry, surfing, and family relationships. Featured works include *Gasland*, *Baghdad ER*,
*War/Dance*, *First Life*, and *The Coconut Revolution*. Several received major awards or nominations. The material also profiles filmmaker Wayne Coles-Janess
and defines
</item 7>

<item 8>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 8>

<item 9>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 9>

<item 10>
To Shoot an Elephant
To Shoot an Elephant is a 2009 documentary film about the 2008-2009 Gaza War directed by Alberto Arce and Mohammad Rujailahk.
</item 10>
=== END ITEM id=bc97ebc600b6a03d ===

=== ITEM id=c5f91ec2e73330d7 ===
Gold sentences and facts:
Sentence 0:  Levitt & Sons developed the Monmouth Heights section from 1965-1969.
  Fact 0.0: Levitt & Sons developed the Monmouth Heights section from 1965 to 1969.
Sentence 1:  The company is most famous for having built the town of Levittown, New York.
  Fact 1.0: Levitt & Sons is most famous for having built the town of Levittown, New York.

Retrieved items:
<item 1>
Taylors Mills, New Jersey
Taylors Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The main road that runs through the area is named for the community, Taylors Mills Road.
</item 1>

<item 2>
Monmouth Heights at Marlboro, New Jersey
Monmouth Heights at Marlboro is an unincorporated community located within Marlboro Township in Monmouth County, New Jersey, United States.

 The housing development was created in the 1960s by Levitt & Sons though some protests occurred by long-time residents of Marlboro Township at the time of its
creation.

 The community is located along New Jersey Route 79 across the highway from Marlboro High School.
</item 2>

<item 3>
Oakland Mills, New Jersey
Oakland Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The area is named for Oakland Mill, a mill along the Manalapan Brook operated by the Boo family.

 Some areas within the Manalapan Brook valley and surround forestland are preserved while the remainder of the area surrounding Oakland Mills has since been
developed into housing developments.
</item 3>

<item 4>
Hanbys Corner, Delaware
Hanbys Corner is an unincorporated community in New Castle County, Delaware, United States.

 It is located at the intersection of Delaware Route 3 (Marsh Road) and Delaware Route 92 (Naamans Road), in Brandywine Hundred.

 The area is named for Richard G. Hanby, who first purchased the 125 acre parcel from the descendants of William Penn in 1753.

 His descendants included several notable figures in the local political scene including Samuel Winfield Hanby (1817-1892) who was elected as a State
Representative in 1874, Jacob Klose Hanby (1839-1932) who was Samuel's son and was elected State Representative in 1904, Robert Johnson Hanby (1834-1898), who
served in the 124th PA infantry during the Civil War and was elected State Senator in 1896, and Florence Wood Hanby (1870-1963), who was Robert's
daughter-in-law and was the first woman elected to the Delaware House of Representatives in 1924.

 In addition Wayne Hanby and James Hanby have both served as Justices of the Peace for New Castle County.
</item 4>

<item 5>
HMS Monmouth
Seven ships of the Royal Navy have been named HMS "Monmouth".

 Monmouth was the name of a castle and is now the name of a town in Wales; the name also recognises James Scott, 1st Duke of Monmouth, the "Black Duke".

 The first HMS "Monmouth" was named after the town but whilst the second one was being used the Monmouth Rebellion took place and this caused some concern to
the crew.

 Since that time, all of "Monmouth"s have had their names painted in black and the name and the crest are covered because of the shame of the Duke of Monmouth's
execution.

 Today the officers of the latest HMS "Monmouth" carry black handkerchiefs and the ship is known as the Black Duke.

 However the town of Monmouth still gives the freedom of the town to the ship's crew and talks of the "enduring link" between the town and the ship.
</item 5>

<item 6>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 6>

<item 7>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 7>

<item 8>
Riverview, New Brunswick
Riverview is a Canadian town in Albert County, New Brunswick, Canada.

 Riverview is located on the south side of the Petitcodiac River, across from the larger cities of Moncton and Dieppe.

 Riverview has an area of 34 km2 , and a population density of 564.6 PD/km2 .

 Riverview's slogan is "A Great Place To Grow".

 With a population of 19,667, Riverview is the fifth largest municipality in New Brunswick, having a larger population than the cities of Edmundston, Bathurst,
Campbellton, and Miramichi, despite its designation of "town".
</item 8>

<item 9>
Great Neck, NY, encompasses nine North Hempstead villages/hamlets (~40,000 residents), five ZIP codes and one school district; Saddle Rock is its Blue Ribbon
K–5 school. Floral Park village had 15,863 residents in 2010. Old Country Road serves central Nassau shopping and partly borders Hempstead/North Hempstead. Long
Island’s 75-mile outer barrier separates lagoons from the Atlantic. The Nassau Herald serves Five Towns. Other entries describe
</item 9>

<item 10>
Monmouth Heights at Manalapan, New Jersey
Monmouth Heights at Manalapan is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 Levitt & Sons developed the Monmouth Heights section from 1965-1969.

 Residents of the 685 homes in the development are required to join the Monmouth Heights Community Association, which provides pools and other amenities.
</item 10>
=== END ITEM id=c5f91ec2e73330d7 ===

=== ITEM id=74e16b5623962535 ===
Gold sentences and facts:
Sentence 0: Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil War.
  Fact 0.0: Am Rong was born in 1929.
  Fact 0.1: Am Rong died in May 1975.
  Fact 0.2: Am Rong was Cambodian.
  Fact 0.3: Am Rong was a soldier.
  Fact 0.4: Am Rong was a filmmaker.
  Fact 0.5: Am Rong acted as a spokesman on military matters for the Khmer Republic.
  Fact 0.6: Am Rong acted as a spokesman during the Cambodian Civil War.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is American.
  Fact 1.2: Ava Marie DuVernay is a director.
  Fact 1.3: Ava Marie DuVernay is a screenwriter.
  Fact 1.4: Ava Marie DuVernay is a film marketer.
  Fact 1.5: Ava Marie DuVernay is a film distributor.

Retrieved items:
<item 1>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 1>

<item 2>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 2>

<item 3>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 3>

<item 4>
Graph query row:
person: Am Rong
birth_date: 1929
</item 4>

<item 5>
Am Rong
Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil
War.

 Western journalists commented on the irony of his name as he gave briefings which "painted a rosy picture of the increasingly desperate situation on the
ground" during the war.
</item 5>

<item 6>
Dana Ivey
Dana Robins Ivey (born August 12, 1941) is an American actress.

 She is a five-time Tony Award nominee for her work on Broadway, and won the 1997 Drama Desk Award for Outstanding Featured Actress in a Play for her work in
both "Sex and Longing" and "The Last Night of Ballyhoo".

 Her film appearances include "The Color Purple" (1985), "Dirty Rotten Scoundrels" (1988), "The Addams Family" (1991), "Two Weeks Notice" (2002), "Rush Hour 3"
(2007) and "The Help" (2011).
</item 6>

<item 7>
A Wrinkle in Time (2018 film)
A Wrinkle in Time is an upcoming American science-fiction fantasy adventure film directed by Ava DuVernay and written by Jennifer Lee.

 It is based on the 1963 novel of the same name by Madeleine L'Engle.

 The film stars Oprah Winfrey, Reese Witherspoon, Mindy Kaling, Storm Reid, Zach Galifianakis and Chris Pine.

 Principal photography on the film began on November 2, 2016, in Los Angeles, California.
</item 7>

<item 8>
Emayatzy Corinealdi
Emayatzy Evett Corinealdi ( ; born January 14, 1980) is an American actress.

 She is best known for her leading role in the Ava DuVernay film "Middle of Nowhere" (2012).
</item 8>

<item 9>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 9>

<item 10>
Graph query row:
person: Ava DuVernay
birth_date: August 24, 1972
</item 10>
=== END ITEM id=74e16b5623962535 ===

=== ITEM id=ffa929e68e289748 ===
Gold sentences and facts:
Sentence 0: John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
  Fact 0.0: John André lived from 2 May 1750 to 2 October 1780.
  Fact 0.1: John André was a British Army officer.
  Fact 0.2: John André was hanged as a spy by the Continental Army during the American Revolutionary War.
  Fact 0.3: John André was hanged for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York, to the British.
Sentence 1: Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American Continental Army, and later defected to the British Army.
  Fact 1.0: Benedict Arnold lived from January 14, 1741 (Old Style: January 3, 1740) to June 14, 1801.
  Fact 1.1: Benedict Arnold was a general during the American Revolutionary War.
  Fact 1.2: Benedict Arnold fought for the American Continental Army.
  Fact 1.3: Benedict Arnold later defected to the British Army.

Retrieved items:
<item 1>
John André
John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting
Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
</item 1>

<item 2>
Siege of Yorktown
The Siege of Yorktown, also known as the Battle of Yorktown, the Surrender at Yorktown, German Battle or the Siege of Little York, ending on October 19, 1781,
at Yorktown, Virginia, was a decisive victory by a combined force of American Continental Army troops led by General George Washington and French Army troops
led by the Comte de Rochambeau over a British Army commanded by British peer and Lieutenant General Charles Cornwallis.

 The culmination of the Yorktown campaign, the siege proved to be the last major land battle of the American Revolutionary War in the North American theater, as
the surrender by Cornwallis, and the capture of both him and his army, prompted the British government to negotiate an end to the conflict.

 The battle boosted faltering American morale and revived French enthusiasm for the war, as well as undermining popular support for the conflict in Great
Britain.
</item 2>

<item 3>
William Farrar Smith
William Farrar Smith (February 17, 1824 – February 28, 1903), known as ‘Baldy’ Smith, was a Union general in the American Civil War, notable for attracting the
extremes of glory and blame.

 He was praised for his gallantry in the Seven Days Battles and the Battle of Antietam, but was demoted for insubordination after the disastrous defeat at the
Battle of Fredericksburg.

 As chief engineer of the Army of the Cumberland, he achieved recognition by restoring a supply line that saved that army from starvation and surrender, known
the “Cracker Line”, that helped Union troops to success in the Chattanooga Campaign in the autumn of 1863.

 Leading the first operation against Petersburg, Smith’s hesitation, possibly illness-related, cost the Union a prime opportunity for a quick end to the war,
and he was relieved of command.
</item 3>

<item 4>
Battle of Germantown
The Battle of Germantown was a major engagement in the Philadelphia campaign of the American Revolutionary War.

 It was fought on October 4, 1777, at Germantown, Pennsylvania, between the British Army led by Sir William Howe, and the American Continental Army, with the
2nd Canadian Regiment, under George Washington.
</item 4>

<item 5>
Lewis Nicola
Lewis Nicola (1717 – August 9, 1807) was an Irish-born American military officer, merchant, and writer who held various military and civilian positions
throughout his career.

 Nicola is most notable for authoring the Newburgh letter, which urged George Washington to assume a royal title.

 Born in Dublin, Ireland, Nicola had been an officer in the British Army, serving in Europe before immigrating to the Thirteen Colonies.

 Establishing a residence in Philadelphia with his family, Nicola opened a library in 1767 and was active in colonial philosophical organizations.

 As a result of his work to establish the American Philosophical Society, he was elected as one of its curators.

 When the American Revolution broke out, Nicola offered his services to the colonial government, which eventually appointed him to various positions with local
forces.
</item 5>

<item 6>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 6>

<item 7>
Benedict Arnold
Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American
Continental Army, and later defected to the British Army.

 While a general on the American side, he obtained command of the fortifications at West Point, New York (which after 1802 would become the site of the U.S.
Military Academy), overlooking the cliffs at the Hudson River (upriver from British-occupied New York City), and planned to surrender them to British forces.

 This plan was exposed in September 1780.

 He was commissioned into the British Army as a brigadier general.
</item 7>

<item 8>
Benjamin Logan
Benjamin Logan (c.1742 – December 11, 1802) was an American pioneer, soldier, and politician from Shelby County, Kentucky.

 As colonel of the Kentucky County, Virginia militia during the American Revolutionary War, he was second-in-command of all the militia in Kentucky.

 He was also a leader in Kentucky's efforts to become a state.

 His brother, John Logan, was the first state treasurer of Kentucky.
</item 8>

<item 9>
List of Continental Army units
The Continental Army was the national army of first the Thirteen Colonies, and then the independent United States, during the American Revolutionary War,
established by a resolution of the Congress on June 14, 1775, three days before the Battle of Bunker Hill, where it saw its first action under that title.

.

 The Continental Congress took a number of steps in the spring of 1775 to create the army in response to the Battles of Lexington and Concord in April and the
seizure of Fort Ticonderoga in May.

 The units composing the Continental Army changed frequently, especially in the first two years of the war.

 From 1777 to the close of the war, the organization of the Continental Army became progressively more systematic and sophisticated.

 The Continental Army that served at Yorktown in 1781 bore very little resemblance to the Continental Army that blockaded Boston in 1775.
</item 9>

<item 10>
Benjamin Lincoln
Benjamin Lincoln (January 24, 1733 (O.S. January 13, 1732) – May 9, 1810) was an American army officer.

 He served as a major general in the Continental Army during the American Revolutionary War.

 Lincoln is notable for being involved in three major surrenders during the war: his participation in the Battles of Saratoga (sustaining a wound shortly
afterward) contributed to John Burgoyne's surrender of a British army, he oversaw the largest American surrender of the war at the 1780 Siege of Charleston,
and, as George Washington's second in command, he formally accepted the British surrender at Yorktown.
</item 10>
=== END ITEM id=ffa929e68e289748 ===

=== ITEM id=54a670bb244f5710 ===
Gold sentences and facts:
Sentence 0:  The trains were built between 1994 and 2009 – for the most part by Talbot, part of Bombardier Inc., with some railcars built by De Dietrich.
  Fact 0.0: The trains were built between 1994 and 2009.
  Fact 0.1: The trains were built for the most part by Talbot.
  Fact 0.2: Talbot is part of Bombardier Inc.
  Fact 0.3: Some railcars were built by De Dietrich.
Sentence 1: Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.
  Fact 1.0: Bombardier Inc. is a multinational aerospace and transportation company.
  Fact 1.1: Bombardier Inc. is based in Montreal, Quebec, Canada.

Retrieved items:
<item 1>
Transportation in Jacksonville, Florida
The Jacksonville transportation network includes ground, air, and sea options for passenger and freight transit.

 The Jacksonville Port Authority (Jaxport) operates the Port of Jacksonville, which includes container shipping facilities at "Blount Island Marine Terminal",
the "Talleyrand Marine Terminal" and the "Dames Point Marine Terminal".

 Jacksonville Aviation Authority managers Jacksonville International Airport in Northside, as well as several smaller airports.

 The Jacksonville Transportation Authority (JTA) operates bus, people mover, and park-n-ride services throughout the city and region.

 A major bus terminal at the intermodal Rosa Parks Transit Station serves as JTA's main transit hub.

 Various intercity bus companies terminate near Central Station.

 Amtrak operates passenger rail service to and from major cities throughout North America.

 The city is bisected by major highways, I-95 and I-10, I-295 creates a full beltway around the city.
</item 1>

<item 2>
Northern Transportation Company
Northern Transportation Company Limited (NTCL) is a marine transportation company in the Canadian and American Arctic owned by Norterra, a holding company that
is owned by the Inuvialuit of the Northwest Territories.

 While primarily a marine freight hauler on Canadian rivers, like the Mackenzie and the Hay and along the Arctic coast of Canada, they are also a petroleum
wholesaler.

 Its head office is now in the town of Hay River, Northwest Territories.

 Today, it primarily utilizes tugs and barges.
</item 2>

<item 3>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 3>

<item 4>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 4>

<item 5>
Marquette Transportation Company
The Marquette Transportation Company is a marine transportation company based in Paducah, Kentucky, United States.
</item 5>

<item 6>
Toland, Pennsylvania
Toland is less than 1/4 of a mile long, and it is located less than half a mile from the Appalachian trail crossing.

 The closest town is Mount Holly Springs, Pennsylvania located (3 miles north of Toland).

 The village is roughly 10 miles south of Carlisle, Pennsylvania, 7 miles south east of Boiling Springs, Pennsylvania, and 15 miles South of Gettysburg,
Pennsylvania.

 The only business in Toland is the Cherokee Campground formerly known as the Tagg Run Campground.

 The campground restaurant is now closed.

 Toland Mission is a small non-denominational church that can hold up to 74 persons.

 It was originally built by the owner of Beetem Lumber Company in Carlisle for the families of the community of Toland.

 A one-room school house near the church originally served the community, but it was converted to a home when residents' children were transported to a township
school.

 Local people in Toland usually travel to the town of Mount Holly Springs for necessities such as gas, food, and toiletries.
</item 6>

<item 7>
Toland, Pennsylvania
Toland is an unincorporated community in Cumberland County, Pennsylvania, United States.

 The very small village is located on Pine Grove Road, east of Mountain Creek Campground.

 Michaux State Forest, the site of Laurel Lake, Fuller Lake and Pine Grove Furnace State Park lie a few miles to the west near the intersection of Pine Grove
Road and PA route 233.

 Toland has a mailing address of Gardners Pennsylvania, because the size of the community doesn't warrant a post-office, nor an official incorporated name.

 What is now a sand pit operation was originally a clay bank mining operation that was the reason for the location of the village of Toland.

 Less than 50 people reside in the close-knit community, which was built for the clay bank company workers in the first quarter of the 20th century.

 The original community of Toland, Pennsylvania consisted of 11 duplex houses, built side by side along Pine Grove Road, with less than 0.17 of an acre of land
to each.

 Additional homes have been built since.

 When the community was originally built there was only a common well with a hand pump for all of the families to draw from.
</item 7>

<item 8>
New York Transportation Company
The New York Transportation Company (originally New York Electrical Vehicle Transportation Company) was a company incorporated in New Jersey in 1899 as the 'New
York Electrical Vehicle Transportation Company' which changed its name to the "New York Transportation Company" in 1902 and was wound up in 1936.

 From 1922 it was controlled by the 'Fifth Avenue Bus Securities Corporation' which was incorporated in Delaware in 1922 which was controlled by The Omnibus
Corporation also incorporated in Delaware.
</item 8>

<item 9>
Merchants Transportation Company
The Merchants Transportation Company was a shipping firm that operated on Puget Sound from 1905 to 1929.

 This company should not be confused with the similarly named Merchants Transportation Company of Olympia, formed in 1874 and a completely separate firm.
</item 9>

<item 10>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
</item 10>
=== END ITEM id=54a670bb244f5710 ===

=== ITEM id=5bc7d74603de44c2 ===
Gold sentences and facts:
Sentence 0: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 0.0: Warrel Dane was born on March 7, 1969.
  Fact 0.1: Warrel Dane is the lead singer of the band Sanctuary.
  Fact 0.2: Sanctuary is an American power metal band.
Sentence 1: Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.
  Fact 1.0: Roy Sætre Khantatat was born on March 12, 1970.
  Fact 1.1: Roy Sætre Khantatat is a Norwegian singer-songwriter.

Retrieved items:
<item 1>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 1>

<item 2>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 2>

<item 3>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 3>

<item 4>
Graph query row:
person: Warrel Dane
birth_date: March 7, 1969
relationship: MEMBER_OF
related_entity: Nevermore
</item 4>

<item 5>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 5>

<item 6>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 6>

<item 7>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 7>

<item 8>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 8>

<item 9>
Roy Khan
Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.

 Commonly known as Roy Khan or simply Khan, he was the lead vocalist for the symphonic power metal band Kamelot from 1997 to his departure in 2011.

 He co-wrote most of Kamelot's songs with the band's guitarist and founder Thomas Youngblood during his tenure.

 Prior to joining Kamelot, Khan was lead singer for the Norwegian progressive metal band Conception from 1991 until 1997 when they disbanded.
</item 9>

<item 10>
Graph query row:
person: Roy Sætre Khantatat
birth_date: March 12, 1970
relationship: MEMBER_OF
related_entity: Kamelot
</item 10>
=== END ITEM id=5bc7d74603de44c2 ===

=== ITEM id=3b188bf578f05aec ===
Gold sentences and facts:
Sentence 0:  They were first year members of the Mountain West Conference.
  Fact 0.0: They were first-year members of the Mountain West Conference.
Sentence 1: The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I Football Bowl Subdivision (FBS) (formerly I-A).
  Fact 1.0: The Mountain West Conference (MW) is a collegiate athletic conference.
  Fact 1.1: The Mountain West Conference is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA).
  Fact 1.2: The Mountain West Conference is affiliated with the NCAA Division I Football Bowl Subdivision (FBS).
  Fact 1.3: The Football Bowl Subdivision (FBS) was formerly called I-A.

Retrieved items:
<item 1>
2012–13 Wagner Seahawks men's basketball team
The 2012–13 Wagner Seahawks men's basketball team represented Wagner College during the 2012–13 NCAA Division I men's basketball season.

 The Seahawks were led by the youngest men's head coach in NCAA Division I, Bashir Mason, who was 28 when he was elevated from an assistant position in March
2012 following the departure of Dan Hurley for Rhode Island.

 The Seahawks played their home games at Spiro Sports Center and were members of the Northeast Conference.

 They finished the season 19–12, 12–6 in NEC play to finish in a three way tie for second place.

 They advanced to the semifinals of the Northeast Tournament where they lost to Long Island.

 For the second consecutive year, despite a winning record, Wagner choose not to participate in a post season tournament.
</item 1>

<item 2>
Women's National Collegiate Athletic Association
The Women's National Collegiate Athletic Association (WNCAA) is an athletic association in the Philippines exclusively for women.

 It was founded in 1970.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 Its men's counterpart is the Men's National Collegiate Athletic Association, founded in 2004.
</item 2>

<item 3>
Men's National Collegiate Athletic Association
The Men's National Collegiate Athletic Association is an athletic association in the Philippines, and is the direct counterpart of the older Women's National
Collegiate Athletic Association (founded in 1970).

 The Men's National Collegiate Athletic Association (MNCAA) is exclusive for men.

 The MNCAA was founded in 2004.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 The competing sports of the tournament are Basketball, Volleyball, Futsal, Table Tennis, Badminton and Streetdance.
</item 3>

<item 4>
Wayne State Warriors football
The Wayne State Warriors football team is the college football team at Wayne State University in Detroit, Michigan.

 The Wayne State football team played their first game in October 1918.

 The Wayne State Warriors have competed in the Great Lakes Intercollegiate Athletic Conference since 1999 (and previously from 1975-1989), and are currently a
Division II member of the National Collegiate Athletic Association (NCAA).

 Wayne State plays their home games at Tom Adams Field at Wayne State Stadium.

 All Wayne State games are broadcast on WDTK radio.
</item 4>

<item 5>
Mountain West Conference
The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I
Football Bowl Subdivision (FBS) (formerly I-A).

 The MW officially began operations in July 1999.

 Geographically, the MW covers a broad expanse of the Western United States, with member schools located in California, Colorado, Hawaii, Idaho, Nevada, New
Mexico, Utah, and Wyoming.

 Craig Thompson has served as Commissioner of the MW since its founding in 1999.
</item 5>

<item 6>
2012 Hawaii Warriors football team
The 2012 Hawaii Warriors football team represented the University of Hawaii at Manoa in the 2012 NCAA Division I FBS football season.

 They were led by first-year head coach Norm Chow and played their home games at Aloha Stadium.

 They were first year members of the Mountain West Conference.

 They finished the season 3–9, 1–7 in Mountain West play to finish in a tie for ninth place.
</item 6>

<item 7>
List of Pacific Tigers head football coaches
The Pacific Tigers college football team represented University of the Pacific (CA).

 The Tigers competed in the National Collegiate Athletic Association (NCAA) College Division in the years 1937–1968.

 In 1969, the team moved to National Collegiate Athletic Association (NCAA) Division I and stayed there until disbanding after the 1995 season.
</item 7>

<item 8>
The entries span college and professional football, plus other athletics. Georgia Southern won six FCS titles, 10 SoCon championships, and the Sun Belt outright
in its 2014 FBS debut. Featured season records include Alabama (1997: 4–7; 1983: 8–4; 1906: 5–1; 1905: 6–4), Auburn’s inaugural team (1892: 2–2), and
Connecticut Aggies (189
</item 8>

<item 9>
2011–12 Wagner Seahawks men's basketball team
The 2011–12 Wagner Seahawks men's basketball team represented Wagner College during the 2011–12 NCAA Division I men's basketball season.

 The Seahawks, led by second year head coach Dan Hurley, played their home games at Spiro Sports Center and are members of the Northeast Conference.

 They finished the season 25–6, 15–3 in NEC play to finish in second place.

 They lost in the semifinals of the NEC Basketball Tournament to Robert Morris.

 Despite having 25 wins, the Seahawks did not accept an invitation to a post season tournament.
</item 9>

<item 10>
Hawaii Rainbow Warriors baseball
The Hawaii Rainbow Warriors baseball team is a varsity intercollegiate athletic team of the University of Hawaii at Manoa in Honolulu, Hawaii, United States.

 The team is a member of the Big West Conference, which is part of the National Collegiate Athletic Association's Division I.

 Hawaii's first baseball team was fielded in 1923.

 The team plays its home games at Les Murakami Stadium in Honolulu, Hawaii.

 The Rainbow Warriors are coached by Mike Trapasso.
</item 10>
=== END ITEM id=3b188bf578f05aec ===

=== ITEM id=e83f2f313b1cb1cd ===
Gold sentences and facts:
Sentence 0: Ecopack Ltd is a PET bottles blowing company in Pakistan.
  Fact 0.0: Ecopack Ltd is a PET bottles blowing company.
  Fact 0.1: Ecopack Ltd is located in Pakistan.
Sentence 1: Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and in combination with glass fiber for engineering resins.
  Fact 1.0: Polyethylene terephthalate is also written as poly(ethylene terephthalate).
  Fact 1.1: Polyethylene terephthalate is commonly abbreviated as PET.
  Fact 1.2: Polyethylene terephthalate is commonly abbreviated as PETE.
  Fact 1.3: Polyethylene terephthalate is sometimes abbreviated as PETP, which is obsolete.
  Fact 1.4: Polyethylene terephthalate is sometimes abbreviated as PET-P, which is obsolete.
  Fact 1.5: Polyethylene terephthalate is the most common thermoplastic polymer resin of the polyester family.
  Fact 1.6: Polyethylene terephthalate is used in fibers for clothing.
  Fact 1.7: Polyethylene terephthalate is used in containers for liquids and foods.
  Fact 1.8: Polyethylene terephthalate is used in thermoforming for manufacturing.
  Fact 1.9: Polyethylene terephthalate is used in combination with glass fiber for engineering resins.

Retrieved items:
<item 1>
Polyethylene naphthalate
Polyethylene naphthalate (poly(ethylene 2,6-naphthalate or PEN) is a polyester with good barrier properties (even better than Polyethylene terephthalate).

 Because it provides a very good oxygen barrier, it is particularly well-suited for bottling beverages that are susceptible to oxidation, such as beer.

 It is also used in making high performance sailcloth.

 It also has been found to show supreme scintillation properties and is expected to replace classic plastic scintillators.
</item 1>

<item 2>
Polyethylene 2,5-furandicarboxylate
Polyethylene 2,5-furandicarboxylate, also named poly(ethylene 2,5-furandicarboxylate), polyethylene furanoate and poly(ethylene furanoate) and generally
abbreviated as PEF, is a polymer that can be produced by polycondensation of 2,5-furandicarboxylic acid (FDCA) and monoethylene glycol (MEG).

 As an aromatic polyester from ethylene glycol it is a chemical analogue of polyethylene terephthalate (PET) and polyethylene naphthalate (PEN).

 PEF has been described in (patent) literature since 1951, but has gained renewed attention since the US department of energy proclaimed its building block,
FDCA, as a potential bio-based replacement for purified terephthalic acid (PTA) in 2004.
</item 2>

<item 3>
Polyethylene terephthalate
Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common
thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and
in combination with glass fiber for engineering resins.
</item 3>

<item 4>
Ecopack
Ecopack Ltd is a PET bottles blowing company in Pakistan.
</item 4>

<item 5>
Two-liter bottle
The two-liter bottle is a common container for soft drinks.

 These bottles are produced from polyethylene terephthalate, also known as PET plastic, using the blow molding process.

 Bottle labels consist of a printed, tight-fitted plastic sleeve.

 A resealable screw-top allows the contents to be used at various times while retaining carbonation.
</item 5>

<item 6>
Xenoy
Xenoy is a blend of plastics with many industrially-useful properties.

 It is typically polyester (polybutylene terephthalate, PBT, or polyethylene terephthalate, PET) and polycarbonate (PC): it is often labeled PBT+PC or PET+PC.
</item 6>

<item 7>
PTT, made from 1,3-propanediol and terephthalic acid or dimethyl terephthalate, is used in carpet fibers. PEF, produced from FDCA and ethylene glycol, is a
PET/PEN analogue attracting interest as a potentially bio-based alternative. PET is widely used for fibers, packaging, and bottles; two-liter soft-drink bottles
are blow-molded PET with sleeves and resealable caps. BoPET is a
</item 7>

<item 8>
Polytrimethylene terephthalate
Polytrimethylene terephthalate (PTT), is a polyester synthesized and patented in 1941.

 It is produced by a method called condensation polymerization or transesterification.

 The two monomer units used in producing this polymer are: 1,3-propanediol and terephthalic acid or dimethyl terephthalate.

 Similar to polyethylene terephthalate, the PTT is used to make carpet fibers.
</item 8>

<item 9>
BoPET
BoPET (Biaxially-oriented polyethylene terephthalate) is a polyester film made from stretched polyethylene terephthalate (PET) and is used for its high tensile
strength, chemical and dimensional stability, transparency, reflectivity, gas and aroma barrier properties, and electrical insulation.
</item 9>

<item 10>
Polycyclohexylenedimethylene terephthalate
Polycyclohexylenedimethylene terephthalate (PCT) is a thermoplastic polyester formed from the polycondensation of terephthalic acid and cyclohexanedimethanol.

Its chemical structure is similar to that of polyethylene terephthalate (PET), with which it shares properties like dimensional stability and chemical
resistance.

 PCT is also particularly resistant to high temperatures and hydrolysis.

 The melting point is 545 °F (285 °C).

 Common brand names are "Thermx" (Ticona), "Eastar" (Eastman) and "SkyPURA" (SK Chemicals).
</item 10>
=== END ITEM id=e83f2f313b1cb1cd ===

=== ITEM id=c039c9aead424a8d ===
Gold sentences and facts:
Sentence 0: Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.
  Fact 0.0: Old Smokey is a euphemistic name for the state prison electric chair in New Jersey.
  Fact 0.1: The New Jersey state prison electric chair, called Old Smokey, is on display at the New Jersey State Police Museum.
Sentence 1:  The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
  Fact 1.0: The chair's most notorious victim was Richard Hauptmann.
  Fact 1.1: Richard Hauptmann was executed in the chair.
  Fact 1.2: Richard Hauptmann was found guilty of abducting and killing Charles Augustus Lindbergh, Jr.
  Fact 1.3: The abduction and killing of Charles Augustus Lindbergh, Jr. happened in 1932.
  Fact 1.4: Charles Augustus Lindbergh, Jr. was 20 months old.
  Fact 1.5: Charles Augustus Lindbergh, Jr. was the son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh.
  Fact 1.6: The case is referred to as the Lindbergh kidnapping (see Lindbergh kidnapping).
Sentence 2:  The Lindbergh kidnapping became known as "The Crime of the Century".
  Fact 2.0: The Lindbergh kidnapping became known as 'The Crime of the Century'.

Retrieved items:
<item 1>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 1>

<item 2>
Smiley face murder theory
The Smiley face murder theory (variations include Smiley face murders, Smiley face killings, Smiley face gang, and others) is a theory advanced by two retired
New York City detectives, Kevin Gannon and Anthony Duarte, that a number of young men found dead in bodies of water across several Midwestern American states
over the last decade did not accidentally drown, as concluded by law enforcement agencies, but were victims of a serial killer or killers.

 The term "smiley face" became connected to the alleged murders when it was made public that the police had discovered graffiti depicting a smiley face near
locations where they think the killer dumped the bodies in at least a dozen of the cases.

 The response of law enforcement investigators and other experts to Gannon and Duarte's theory has been largely skeptical.
</item 2>

<item 3>
The Big House (1930 film)
The Big House is a 1930 American Pre-Code crime drama film directed by George W. Hill, released by Metro-Goldwyn-Mayer, and starring Chester Morris, Wallace
Beery, Lewis Stone and Robert Montgomery.

 The supporting cast features Leila Hyams, George F. Marion, J. C. Nugent, Karl Dane and Tom Kennedy.

 The story and dialogue were written by Frances Marion, with additional dialogue by Joe Farnham and Martin Flavin.

 The story was inspired by a spate of prison riots in 1929 and resulting federal investigation.

 In response, George Hill wrote a twenty-seven page story treatment called "The Reign of Terror: A Story of Crime and Punishment".

 Irving Thalberg gave the go ahead for the screenplay and assigned Frances Marion to work with George Hill.
</item 3>

<item 4>
Old Smokey
Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.

 The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus
Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
</item 4>

<item 5>
Sten Forshufvud
Sten Gabriel Bernhard Forshufvud (9 February 1903 – 25 June 1985) was a Swedish dentist and physician, and amateur toxicologist (expert on poisons) who
formulated and supported the controversial theory that Napoleon was assassinated by a member of his entourage while in exile.

 He wrote a book, in Swedish, about this in 1961, which was translated the following year as "Who Killed Napoleon?"

 He later published his ideas in English, in the 1983 book "Assassination At St. Helena: The Poisoning Of Napoleon Bonaparte," a book on whose authorship Ben
Weider, co-author (with David Hapgood) of the book "The Murder Of Napoleon," published the year earlier, which also advanced Forshufvud's ideas, collaborated.
</item 5>

<item 6>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 6>

<item 7>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 7>

<item 8>
Johnny Jack Nounes
Johnny Jack Nounes, also known as the "Beau Brummell of Galveston", was an organized crime boss in Galveston, Texas, United States, during the early 1900s.

 He, with one-armed George Musey, led the Downtown Gang, one of the two gangs which controlled most of the Galveston Crime Syndicate until the early 1930s.

 They fought for control of the island against the rival Beach Gang led by Ollie Quinn and Dutch Voight.

 As the prohibition era began, his gang came to be one of the dominant forces in the Galveston Crime Syndicate.

 Galveston became the main port of entry for liquor supply in Texas and many parts of the Midwest.
</item 8>

<item 9>
Richard Hauptmann
Bruno Richard Hauptmann (November 26, 1899 – April 3, 1936) was a German-born carpenter who was convicted of the abduction and murder of the 20-month-old son of
aviator Charles Lindbergh and his wife Anne Morrow Lindbergh.

 The Lindbergh kidnapping became known as "The Crime of the Century".

 Hauptmann proclaimed his innocence to the end, but he was convicted of first degree murder and executed in 1936 in the electric chair at the New Jersey State
Prison.
</item 9>

<item 10>
Old Sparky
Old Sparky is the nickname of the electric chairs in Arkansas, Connecticut, Florida, Georgia, Illinois, Kentucky, Nebraska, New York, Ohio, Oklahoma, South
Carolina, Texas, Virginia, and West Virginia.

 Old Smokey was the nickname of the electric chairs used in New Jersey, Pennsylvania, and Tennessee.
</item 10>
=== END ITEM id=c039c9aead424a8d ===

=== ITEM id=eeeb38b624293a41 ===
Gold sentences and facts:
Sentence 0:  Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.
  Fact 0.0: Qionghai is called a city.
  Fact 0.1: Qionghai refers to a large land area in Hainan.
  Fact 0.2: The area of Qionghai in Hainan was once a county.
Sentence 1:  It has an area of 1692 km2 and in 2010, it had a population of 483,217.
  Fact 1.0: Qionghai has an area of 1692 km2.
  Fact 1.1: Qionghai had a population of 483,217 in 2010.
Sentence 2: Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.
  Fact 2.0: Suining has the Sichuanese Pinyin Xunin.
  Fact 2.1: Suining is a prefecture-level city.
  Fact 2.2: Suining is in eastern Sichuan province.
  Fact 2.3: Sichuan province is in Southwest China.
Sentence 3:  In 2002, Suining had a population of 658,798.
  Fact 3.0: Suining had a population of 658,798 in 2002.

Retrieved items:
<item 1>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 1>

<item 2>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 2>

<item 3>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 3>

<item 4>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 4>

<item 5>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 5>

<item 6>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 6>

<item 7>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 7>

<item 8>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 8>

<item 9>
Qionghai
Qionghai () is one of the seven county-level cities of Hainan province, China.

 Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.

 Within this area is the main city, Qionghai City.

 It is located in the east of the island at the mouth of the Wanquan River, 86 km from the provincial capital of Haikou.

 It has an area of 1692 km2 and in 2010, it had a population of 483,217.
</item 9>

<item 10>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 10>
=== END ITEM id=eeeb38b624293a41 ===
