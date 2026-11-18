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

This file has 12 items.

=== ITEM id=b7a9b4ab154f146d ===
Gold sentences and facts:
Sentence 0: The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.
  Fact 0.0: The Bishop of Ely is the ordinary of the Diocese of Ely.
  Fact 0.1: The Diocese of Ely is a Church of England diocese.
  Fact 0.2: The Diocese of Ely is in the Province of Canterbury.
Sentence 1:  Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
  Fact 1.0: Conway became Bishop of Ely in 2010.
  Fact 1.1: Conway was translated to Bishop of Ely from the Diocese of Salisbury.
  Fact 1.2: Conway was Bishop suffragan of Ramsbury before becoming Bishop of Ely.
  Fact 1.3: Conway was Bishop suffragan of Ramsbury in the Diocese of Salisbury.
Sentence 2: The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.
  Fact 2.0: The Diocese of Salisbury is a Church of England diocese.
  Fact 2.1: The Diocese of Salisbury is in the south of England.
  Fact 2.2: The Diocese of Salisbury is within the ecclesiastical Province of Canterbury.

Retrieved items:
<item 1>
Ecclesiastical Province of Freiburg
The Ecclesiastical Province of Freiburg ("Kirchenprovinz Freiburg") or Upper Rhenish Ecclesiastical Province ("Oberrheinische Kirchenprovinz") is an
ecclesiastical province of the Roman Catholic Church in the Upper Rhine area of Germany, centring on Freiburg im Breisgau.

 It covers the Roman Catholic Archdiocese of Freiburg, the Roman Catholic Diocese of Mainz and the Roman Catholic Diocese of Rottenburg-Stuttgart, covering
large areas of Baden-Württemberg and Hesse and small parts of Rhineland-Palatinate.

 Its metropolitan bishop is the Archbishop of Freiburg - that Archdiocese and the Province were both set up in 1821 in the wake of the 1801 Concordat and the
1815 Congress of Vienna.
</item 1>

<item 2>
Bishop of Ely
The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.

 The diocese roughly covers the county of Cambridgeshire (with the exception of the Soke of Peterborough), together with a section of north-west Norfolk and has
its episcopal see in the City of Ely, Cambridgeshire, where the seat is located at the Cathedral Church of the Holy Trinity.

 The current bishop is Stephen Conway, who signs "+Stephen Elien:" (abbreviation of the Latin adjective "Eliensis", meaning "of Ely").

 The diocesan bishops now reside in Bishop's House, Ely, the former cathedral deanery.

 Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
</item 2>

<item 3>
Roman Catholic Ecclesiastical Province of Miami
The Ecclesiastical Province of Miami is a Catholic ecclesiastical province covering the U.S. state of Florida.

 Its metropolitan bishop is the Archbishop of Miami, head of the Archdiocese of Miami.

 The province additionally includes the suffragan dioceses of Orlando, Palm Beach, Pensacola-Tallahassee, St. Augustine, St. Petersburg, and Venice.
</item 3>

<item 4>
Graph query row:
person: Stephen Conway
diocese: Diocese of Ely
relationship: MEMBER_OF
role: None
status: None
ecclesiastical_province: Province of Canterbury
province_relationship: PART_OF
</item 4>

<item 5>
Ecclesiastical Province of Ontario
The Ecclesiastical Province of Ontario is one of the Anglican Church of Canada's four ecclesiastical provinces.

 It was established in 1912 out of six dioceses of the Ecclesiastical Province of Canada located in the civil Province of Ontario, and the Diocese of Moosonee
from the Ecclesiastical Province of Rupert's Land.
</item 5>

<item 6>
Dean of Chelmsford
The Dean of Chelmsford is the head ("primus inter pares" – first among equals) and chair of the Cathedral Chapter, the governing body of Chelmsford Cathedral,
the Cathedral Church of St Mary the Virgin, St Peter and St Cedd.

 Before 2000 the post was designated as a provost, which was then the equivalent of a dean at most English cathedrals.

 The cathedral is the mother church of the Diocese of Chelmsford and seat of the Bishop of Chelmsford.

 The Dean of Chelmsford is also responsible for the Chapel of St Peter-on-the-Wall at Bradwell-on-Sea, founded by St Cedd, among the oldest church buildings in
regular use in England.
</item 6>

<item 7>
Diocese of Salisbury
The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.

 The diocese covers most of Dorset (excepting the deaneries of Bournemouth and Christchurch, which fall within the Diocese of Winchester), and most of Wiltshire
(excepting a part in the north and Swindon).

 The diocese is led by the Bishop of Salisbury (Nick Holtam) and the diocesan synod.

 The bishop's seat is at Salisbury Cathedral.
</item 7>

<item 8>
Dean of Guildford
The Dean of Guildford is the head ("primus inter pares" – first among equals) and chair of the chapter of canons, the ruling body of Guildford Cathedral.

 The dean and chapter are based at the "Cathedral Church of the Holy Spirit" in Guildford.

 The cathedral is the mother church of the Diocese of Guildford and seat of the Bishop of Guildford.

 The current dean is Dianna Gwilliams, who was installed on 15 September 2013.
</item 8>

<item 9>
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 9>

<item 10>
Ecclesiastical province
An ecclesiastical province is a general term for one of the basic forms of jurisdiction in Christian Churches with traditional hierarchical structure, including
Western Christianity and Eastern Christianity.

 In general, ecclesiastical province is consisted of several dioceses (or eparchies), one of them being the archdiocese (or archeparchy), headed by metropolitan
bishop or archbishop who has ecclesiastical jurisdiction over all other bishops of the province.
</item 10>
=== END ITEM id=b7a9b4ab154f146d ===

=== ITEM id=b9c3b4a3fc02d07c ===
Gold sentences and facts:
Sentence 0: Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York, United States.
  Fact 0.0: Albany International Airport has the IATA code ALB.
  Fact 0.1: Albany International Airport has the ICAO code KALB.
  Fact 0.2: Albany International Airport has the FAA LID ALB.
  Fact 0.3: Albany International Airport is a public airport.
  Fact 0.4: Albany International Airport is seven miles (11 km) northwest of Albany.
  Fact 0.5: Albany International Airport is in Albany County, New York, United States.
Sentence 1: Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown Madison, the capital of Wisconsin.
  Fact 1.0: Dane County Regional Airport (DCRA) has the IATA code MSN.
  Fact 1.1: Dane County Regional Airport (DCRA) has the ICAO code KMSN.
  Fact 1.2: Dane County Regional Airport (DCRA) has the FAA LID MSN.
  Fact 1.3: Dane County Regional Airport is also known as Truax Field.
  Fact 1.4: Dane County Regional Airport is a civil-military airport.
  Fact 1.5: Dane County Regional Airport is located six miles northeast of downtown Madison.
  Fact 1.6: Madison is the capital of Wisconsin.

Retrieved items:
<item 1>
Paul Clyne
Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.

 A graduate of Albany Law School, he spent about 14 years as an assistant district attorney, before he was tapped by local politicians to replace the retiring
District Attorney, Sol Greenberg.

 He was defeated for re-election by David Soares, first in the Democratic Party primary election in September 2004, and then in the general election in November
2004, in which he ran on an independent line.

 After a stint teaching at the New York Prosecutors Institute, he went into private practice as a criminal defense attorney in 2007, with an office in Albany,
New York.
</item 1>

<item 2>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 2>

<item 3>
Truax Field Air National Guard Base
Truax Field Air National Guard Base (IATA: MSN, ICAO: KMSN, FAA LID: MSN) , also known as Truax Field, is a military facility located at Dane County Regional
Airport.

 It is located five miles (8 km) northeast of the center of Madison, a city in Dane County, Wisconsin, United States.
</item 3>

<item 4>
Wilmington International Airport
Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated
Wrightsboro, Cape Fear Township, New Hanover County.
</item 4>

<item 5>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 5>

<item 6>
Nashville Terminal Subdivision
The Nashville Terminal Subdivision is a railroad line owned by CSX Transportation in the U.S. State of Tennessee.

 The Subdivision is broken up into 5 sections all in Nashville, Tennessee.

 The northern part of the Terminal is in Madison, Tennessee at milepost 000/0BA 174 on the southern end of the Ex-L&N Mainline Subdivision at Monfort.

 Disptach for the Mainline Sub is known as "LD" which is part of the Cincinnati Division.

 From here in Madison begins the double track that stays for another 22 miles south to Brentwood, Tennessee.

 At milepost 000/0BA/00H 176.6, the famous Johnny Cash "Amqui" location where the Ex-L&N Evansville, Indiana line, the Henderson Subdivision meets with the
Terminal.

 Dispatch for the Henderson Sub is known as "SA" and operates on AAR58.

 There are two crossings near each other, Williams Ave and Nesbitt Lane at Amqui.
</item 6>

<item 7>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 7>

<item 8>
Hyde Park Railroad Station
The Hyde Park Railroad Station is the former New York Central Railroad station located where Crum Elbow Creek flows into the Hudson River in Hyde Park, New
York.

 A one-story wooden station was first established by the Central at the spot in 1851 by the Hudson River Railroad, connecting New York City and Albany.

 It was replaced by the existing building, built in a combination of the Mission and Spanish Revival styles by Warren and Wetmore, the railroad's preferred
architects who had also designed Grand Central Terminal and the nearby Poughkeepsie station, in 1914.
</item 8>

<item 9>
Dane County Regional Airport
Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown
Madison, the capital of Wisconsin.

 In the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, it is categorized as a small-hub primary commercial
service facility.
</item 9>

<item 10>
Air Wisconsin
Air Wisconsin Airlines is a regional airline based at Appleton International Airport in the town of Greenville, Wisconsin, United States, near Appleton.

 Air Wisconsin is the largest privately held regional airline in the United States.

 It currently operates regional jet flights as American Eagle under contract to American Airlines via a code sharing agreement, serving cities in the U.S. and
Canada with hubs at Boston Logan International Airport (BOS), Charlotte Douglas International Airport (CLT), Philadelphia International Airport (PHL) and Ronald
Reagan Washington National Airport (DCA) .

 Air Wisconsin previously operated United Express service on behalf of United Airlines followed by US Airways Express service on behalf of US Airways prior to
becoming an American Eagle regional air carrier.

 The company has announced it will once again partner with United Airlines as a United Express code sharing air carrier with primary hubs to be located at
Chicago O'Hare International Airport (ORD) and Washington Dulles International Airport (IAD) beginning in September 2017 and will be "exclusively operating" as
United Express by March 2018.
</item 10>
=== END ITEM id=b9c3b4a3fc02d07c ===

=== ITEM id=698e015b6a6c61d4 ===
Gold sentences and facts:
Sentence 0: Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.
  Fact 0.0: Orlando Brown was born on December 4, 1987.
  Fact 0.1: Orlando Brown is an American actor.
  Fact 0.2: Orlando Brown is a voice actor.
  Fact 0.3: Orlando Brown is a rapper.
  Fact 0.4: Orlando Brown is a singer.
Sentence 1:  He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
  Fact 1.0: Orlando Brown is best known for his role as Eddie Thomas in "That's So Raven".
  Fact 1.1: Orlando Brown is best known for his role as 3J Winslow in "Family Matters".
  Fact 1.2: Orlando Brown is best known for his role as Tiger in "Major Payne".
  Fact 1.3: Orlando Brown is best known for his role as Max in "Two of a Kind".
  Fact 1.4: Orlando Brown is best known for his role as Damey Wayne in the short-lived Waynehead.
  Fact 1.5: Orlando Brown is best known for his role as Dobbs in "Max Keeble's Big Move".
  Fact 1.6: Orlando Brown is best known for his role as Frankie in "Eddie's Million Dollar Cook Off".
Sentence 2: Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17, 1998.
  Fact 2.0: Family Matters is an American sitcom.
  Fact 2.1: Family Matters originated on ABC.
  Fact 2.2: Family Matters originated on ABC from September 22, 1989 to May 9, 1997.
  Fact 2.3: Family Matters moved to CBS after its ABC run.
  Fact 2.4: Family Matters was on CBS from September 19, 1997 to July 17, 1998.
Sentence 3:  A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.
  Fact 3.0: Family Matters is a spin-off of "Perfect Strangers."
  Fact 3.1: Family Matters revolves around the Winslow family.
  Fact 3.2: The Winslow family is a middle-class African American family.
  Fact 3.3: The Winslow family lives in Chicago, Illinois.

Retrieved items:
<item 1>
Jaimee Foxworth
Jaimee Monae Foxworth (born December 17, 1979) is an American actress and model.

 She is best known for her role of Judy Winslow, the youngest daughter of Carl and Harriette Winslow on the ABC/CBS sitcom "Family Matters".

 She later transitioned to pornographic films using the name Crave, starring in several titles.
</item 1>

<item 2>
Family Matters
Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17,
1998.

 A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.

 Midway through the first season, the show introduced the Winslows' nerdy neighbor Steve Urkel (Jaleel White), who quickly became its breakout character and
eventually the show's main character.

 Having run for nine seasons, "Family Matters" became the second longest-running non-animated US sitcom with a predominantly African American cast, behind only
"The Jeffersons" (11).

 Having aired 215 episodes, "Family Matters" is ranked third, behind only "Tyler Perry's House of Payne" (254), and "The Jeffersons" (253).
</item 2>

<item 3>
Lamman Rucker
Lamman Rucker (born October 6, 1971) is an American actor.

 Rucker began his career on the daytime soap operas "As the World Turns" and "All My Children", before roles in Tyler Perry's films "Why Did I Get Married?

", "Why Did I Get Married Too?

", and "Meet the Browns", and its television adaptation.

 In 2016, he began starring as Jacob Greenleaf in the Oprah Winfrey Network drama series, "Greenleaf".
</item 3>

<item 4>
In Living Color
In Living Color is an American sketch comedy television series that originally ran on Fox from April 15, 1990, to May 19, 1994.

 Brothers Keenen and Damon Wayans created, wrote and starred in the program.

 The show was produced by Ivory Way Productions in association with 20th Century Fox Television and was taped at stage 7 at the Fox Television Center on Sunset
Boulevard in Hollywood, California.

 The title of the series was inspired by the NBC announcement of broadcasts being presented "in living color" during the 1960s, prior to mainstream color
television.

 It also refers to the fact that most of the show's cast were black, unlike other sketch comedy shows such as "Saturday Night Live" whose casts are mostly
white.

 It was controversial due to the Wayans' decision to portray African-American humor from the ghetto in a time when mainstream American tastes regarding black
comedy had been set by more upscale shows such as "The Cosby Show", causing an eventual feud for control between Fox executives and the Wayans.
</item 4>

<item 5>
The Simpsons Spin-Off Showcase
"The Simpsons Spin-Off Showcase" is the twenty-fourth episode of the eighth season of "The Simpsons".

 It originally aired on the Fox network in the United States on May 11, 1997.

 The episode centers on fictional pilot episodes of non-existent television series derived from "The Simpsons", and is a parody of the tendency of networks to
spin off characters from a hit series.

 As such it includes references to many different TV series.

 The first fictional spin-off is "Chief Wiggum P.I.", a cop-drama featuring Chief Wiggum and Seymour Skinner.

 The second is "The Love-matic Grampa", a sitcom featuring Moe Szyslak who receives dating advice from Abraham Simpson, whose ghost is possessing a love testing
machine.

 The final segment is "The Simpson Family Smile-Time Variety Hour", a variety show featuring the Simpson family except for Lisa, who has been replaced.
</item 5>

<item 6>
Tyler Perry's For Better or Worse
Tyler Perry's For Better or Worse is an American television sitcom created, written and executive produced by Tyler Perry.

 The series is based on and is a TV spin-off to Perry's 2007 film "Why Did I Get Married?

" and its 2010 sequel "Why Did I Get Married Too?

".

 The show premiered on TBS on November 25, 2011 and ended on July 22, 2017.

 Led by the comical, over-the-top antics of Marcus and Angela Williams, the ensemble follows three couples: Marcus and Angela, Joseph and Leslie, and Richard
and Keisha who are at various stages of their relationships as they go through the ups-and-downs of married life and dating.

 The third season of the series premiered on September 18, 2013, on the Oprah Winfrey Network.
</item 6>

<item 7>
Darius McCrary
Darius Creston McCrary (born May 1, 1976) is an American film and television actor and singer.

 He is best known for his role as Edward "Eddie" Winslow, the oldest child of Carl and Harriette Winslow on the ABC/CBS television sitcom "Family Matters "which
ran from 1989–1998.

 Another one of McCrary's notable roles was as Scam in the 1987 comedy film "Big Shots", which was his film debut.

 He provided the voice of Jazz in the 2007 "Transformers" reboot.

 From December 2009 to October 2011, he portrayed photographer Malcolm Winters on the CBS daytime drama "The Young and the Restless".
</item 7>

<item 8>
Wingin' It
Wingin' It (originally titled Angel on Campus prior to its debut) is a Canadian teen sitcom which aired on Family.

 The series was produced by Temple Street Productions in association with Family.

 It stars Demetrius Joyette and Dylan Everett.

 This show included guest-stars from other Family Channel shows such as "The Latest Buzz".

 It was announced that the series was renewed for a third season on June 13, 2011.

 Family announced that the series is not planned for a fourth season.
</item 8>

<item 9>
Orlando Brown (actor)
Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.

 He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne
in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
</item 9>

<item 10>
Randy J. Goodwin
Randy J. Goodwin, born as Ivan Jerome Goodwin, is a Hollywood actor.

 He is perhaps best known for his role as "Kennedy Winslow" on "Fast Track" and "Max Ellis" on the show "Abby", as well as his recurring roles on "Girlfriends"
as "Davis Hamilton" and "The Vampire Diaries" as "Dr. Jonas Martin".
</item 10>
=== END ITEM id=698e015b6a6c61d4 ===

=== ITEM id=3a15f7da1b77f0b6 ===
Gold sentences and facts:
Sentence 0:  They competed at the 2015 Pacific Games, where they finished with an 0-4 record.
  Fact 0.0: The unnamed subject competed at the 2015 Pacific Games
  Fact 0.1: The unnamed subject finished the 2015 Pacific Games with an 0-4 record
Sentence 1: Basketball at the 2015 Pacific Games in Port Moresby, Papua New Guinea was held at the BSP Arena and PNG Power Dome on 3–12 July 2015.
  Fact 1.0: Basketball at the 2015 Pacific Games was held in Port Moresby, Papua New Guinea
  Fact 1.1: Basketball at the 2015 Pacific Games was held at the BSP Arena and PNG Power Dome
  Fact 1.2: Basketball at the 2015 Pacific Games was held on 3 to 12 July 2015

Retrieved items:
<item 1>
Kiribati national basketball team
The Kiribati national basketball team are the basketball side that represent Kiribati in international competitions.

 They competed at the 2015 Pacific Games, where they finished with an 0-4 record.
</item 1>

<item 2>
Men's football at the 2015 Pacific Games
The 2015 Pacific Games men's football tournament was the 14th edition of the Pacific Games men's football tournament.

 The men's football tournament was held in Port Moresby, Papua New Guinea between 3–17 July 2015 as part of the 2015 Pacific Games.

 The tournament was age-restricted and open to men's under-23 national teams only.
</item 2>

<item 3>
Basketball at the 2015 Pacific Games
Basketball at the 2015 Pacific Games in Port Moresby, Papua New Guinea was held at the BSP Arena and PNG Power Dome on 3–12 July 2015.
</item 3>

<item 4>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 4>

<item 5>
Powerlifting at the 2015 Pacific Games
Powerlifting at the 2015 Pacific Games was held from 9–10 July 2015 in the Sir John Guise Indoor Power Dome at Port Moresby, Papua New Guinea.

 The host nation Papua New Guinea was the dominant team, particularly in the women's divisions where it claimed six of the seven possible gold medals.

 Nauru was the strongest team in the men's divisions, winning four of the eight weight categories.

 Telupe Iosefa received the first ever gold medal won by Tuvalu at the Pacific Games in the 120 kg male division.
</item 5>

<item 6>
2001 Goodwill Games
The 2001 Goodwill Games was the fifth and final edition of the international multi-sport event.

 The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.

 A total around 1300 athletes took part in 14 sporting competitions.
</item 6>

<item 7>
Kiribati at the 2015 Pacific Games
Kiribati competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 A total of 86 competitors for Kiribati were listed as of 4 July 2015.
</item 7>

<item 8>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 8>

<item 9>
2015 Pacific Games
The 2015 Pacific Games was held in Port Moresby, Papua New Guinea, from 4 to 18 July 2015.

 It was the fifteenth staging of the Pacific Games.
</item 9>

<item 10>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 10>
=== END ITEM id=3a15f7da1b77f0b6 ===

=== ITEM id=df985ea5d0c99366 ===
Gold sentences and facts:
Sentence 0: Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.
  Fact 0.0: Henderson Point is a census-designated place (CDP).
  Fact 0.1: Henderson Point is in Harrison County, Mississippi, United States.
Sentence 1:  Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
  Fact 1.0: Gulfport is a city adjoining Biloxi.
  Fact 1.1: Biloxi is a county seat of Harrison County.

Retrieved items:
<item 1>
Biloxi, Mississippi
Biloxi ( ) is a city in Harrison County, Mississippi, United States.

 The 2010 United States Census recorded the population as 44,054, and in 2016 the estimated population was 45,975.

 Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
</item 1>

<item 2>
Hendersonville, North Carolina
Hendersonville is a city in Henderson County, North Carolina United States.

 This city is 22 miles south of Asheville.

 It is the county seat of Henderson County.

 Like the county, the city is named for 19th-century North Carolina Supreme Court Chief Justice Leonard Henderson.
</item 2>

<item 3>
Nashville Terminal Subdivision
The Nashville Terminal Subdivision is a railroad line owned by CSX Transportation in the U.S. State of Tennessee.

 The Subdivision is broken up into 5 sections all in Nashville, Tennessee.

 The northern part of the Terminal is in Madison, Tennessee at milepost 000/0BA 174 on the southern end of the Ex-L&N Mainline Subdivision at Monfort.

 Disptach for the Mainline Sub is known as "LD" which is part of the Cincinnati Division.

 From here in Madison begins the double track that stays for another 22 miles south to Brentwood, Tennessee.

 At milepost 000/0BA/00H 176.6, the famous Johnny Cash "Amqui" location where the Ex-L&N Evansville, Indiana line, the Henderson Subdivision meets with the
Terminal.

 Dispatch for the Henderson Sub is known as "SA" and operates on AAR58.

 There are two crossings near each other, Williams Ave and Nesbitt Lane at Amqui.
</item 3>

<item 4>
Raeford, North Carolina
Raeford is a city in Hoke County, North Carolina, United States.

 The population was 4,626 at the 2010 census.

 It is the county seat of Hoke County.

 The county was named after Confederate General Robert F. Hoke, a North Carolina native.
</item 4>

<item 5>
Henderson Point, Mississippi
Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.

 It is part of the Gulfport–Biloxi Metropolitan Statistical Area.

 The population was 170 at the 2010 census.
</item 5>

<item 6>
Paulding, Mississippi
Paulding is an unincorporated community in and one of the two county seats of Jasper County, Mississippi.

 It is the only unincorporated county seat in Mississippi.

 Settled in 1833, it was named by United States settlers in honor of Revolutionary War hero John Paulding.

 After its citizens refused to contribute to a new railroad, the community was bypassed in favor of Bay Springs, Mississippi, which was designated a railroad
stop to the west and the second county seat.

 It attracted more development and industry.
</item 6>

<item 7>
Jasper County Community Unit School District 1
Craig Carr is the principal of this school.

 Ste. Marie Elementary School is located in the southern Jasper County village of Ste. Marie, and serves students in grades one through six.

 The principal of Ste. Marie is David Parker, the principal of Willow Hill Elementary School.

 Newton Elementary School is located in the county seat of Newton, and serves most of west Jasper County's first through sixth graders under principal Travis
Wyatt.

 The latter three elementary schools feed into Jasper County Junior High School and are taught in the facility during seventh and eighth grade while being
supervised by Newton Elementary School principal Travis Wyatt before graduating into Newton Community High School.

 Students in grades nine through twelve spend their last leg of precollegiate education at this school; their principal is Ruth Kerner.

 The district superintendent is Ron Alburtus, and the district's mascot is the eagle.
</item 7>

<item 8>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 8>

<item 9>
Piney Woods Country Life School
The Piney Woods Country Life School (or The Piney Woods School) is a co-educational independent historically African-American boarding school for grades 9-12 in
Piney Woods, unincorporated Rankin County, Mississippi.

 It is 21 mi south of Jackson.

 It is one of four remaining historically African-American boarding schools in the United States.

 It is currently the largest African-American boarding school, as well as being the second oldest continually operating African-American boarding school.
</item 9>

<item 10>
Henderson, Texas
Henderson is a city in Rusk County, northeast Texas, United States.

 The population was 13,712 at the 2010 census.

 It is the county seat of Rusk County.

 Henderson is named for James Pinckney Henderson, the first governor of Texas.
</item 10>
=== END ITEM id=df985ea5d0c99366 ===

=== ITEM id=e888b80d4f14d28b ===
Gold sentences and facts:
Sentence 0: Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
  Fact 0.0: Ruggero Deodato was born on 7 May 1939.
  Fact 0.1: Ruggero Deodato is an Italian film director.
  Fact 0.2: Ruggero Deodato is an Italian screenwriter.
  Fact 0.3: Ruggero Deodato is an Italian actor.
Sentence 1: Alejandro Springall is a Mexican film director and producer.
  Fact 1.0: Alejandro Springall is a Mexican film director.
  Fact 1.1: Alejandro Springall is a Mexican producer.

Retrieved items:
<item 1>
Ruggero Deodato
Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
</item 1>

<item 2>
Antonio De Carlo
Antonio De Carlo, born (1967--) 4, 1967 (age 50 ) in Tijuana, Baja California, Mexico, is a Mexican actor also known as Antonio in the late 1980s and early
1990s.

 He won an Emmy Award in 2005-2006.

 He is president and founder of the "Fundación Cultura Sin Fronteras AC".

 After twelve years out of show business, in 2012 he returned to the soap operas performing "Magic Dragon" one of the main characters for "Miss XV", a
co-production from Televisa México and Nickelodeon directed and produced by Pedro Damian.

 He is also part of the cast of "Muchacha Italiana Viene a Casarse", directed by Pedro Damian (Televisa, México).
</item 2>

<item 3>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 3>

<item 4>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 4>

<item 5>
Cannibal Holocaust
Cannibal Holocaust is a 1980 Italian cannibal horror film directed by Ruggero Deodato from a screenplay by Gianfranco Clerici.

 It stars Carl Gabriel Yorke, Robert Kerman, Francesca Ciardi, Perry Pirkanen, and Luca Barbareschi.

 Influenced by the works of Mondo director Gualtiero Jacopetti, the film was inspired by Italian media reporting of Red Brigade terrorism.

 The coverage included news reports Deodato believed to be staged, an idea which became an integral aspect of the film's story.

 "Cannibal Holocaust" was filmed primarily in the Amazon rainforest of Colombia with indigenous tribes interacting with American and Italian actors.
</item 5>

<item 6>
Hercules, Prisoner of Evil
Hercules, Prisoner of Evil (Italian: "Ursus il terrore dei Kirghisi" , 'Ursus, Terror of the Kirghiz' ) is a 1964 Italian "peplum" film directed by Antonio
Margheriti and an uncredited Ruggero Deodato.

 Deodato, the official assistant director, replaced Margheriti as he was busy with the completion of the film "The Fall of Rome".

 Deodato actually directed most of the film in actuality but Margheriti was credited as the director.

 The film is filled with a variety of horrific themes and elements, featuring a killer werewolf, and is as much a horror film as it is a peplum.
</item 6>

<item 7>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 7>

<item 8>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 8>

<item 9>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 9>

<item 10>
Alejandro Springall
Alejandro Springall is a Mexican film director and producer.
</item 10>
=== END ITEM id=e888b80d4f14d28b ===

=== ITEM id=00dadeb63d3cd7aa ===
Gold sentences and facts:
Sentence 0: Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.
  Fact 0.0: Talking Heads were an American rock band.
  Fact 0.1: Talking Heads were formed in 1975.
  Fact 0.2: Talking Heads were formed in New York City.
  Fact 0.3: Talking Heads were active until 1991.
Sentence 1:  is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
  Fact 1.0: The group is a surf rock group.
  Fact 1.1: The group formed in Auburn, Alabama.
  Fact 1.2: The group formed in the early 1990s.
  Fact 1.3: The group came to prominence over the following decade.

Retrieved items:
<item 1>
Talking Heads: 77
Talking Heads: 77 is the debut album by the American rock band Talking Heads, released in September 1977.

 The single "Psycho Killer" reached No. 92 on the Billboard Hot 100 Chart.

 In 2003, the album was ranked No. 290 on "Rolling Stone" magazine's The 500 Greatest Albums of All Time list.
</item 1>

<item 2>
Talking Heads Africa
Talking Heads (Africa) was introduced in Cape Town in 2008 as part of the Infecting the City public art festival.

 Talking Heads has four core components that form the project.

 These include: developing a platform for conversation and exchange with and between experts; creating a network of African thought leaders; shooting
mini-documentaries that define these leaders and their contributions; developing the tools to make this model work in cities all over the African continent.
</item 2>

<item 3>
The Happenings
The Happenings are a pop music group that originated in the 1960s.

 Members of the original group, created in the spring of 1961 and initially called "The Four Grads" because all had just graduated from high school in Paterson,
New Jersey, were Bob Miranda, David Libert, Tom Giuliano, and Ralph DiVito.

 In 1968 DiVito was replaced by Bernie LaPorta and Lenny Conforti also joined to play drums in the touring band.

 Both LaPorta and Conforti took a hiatus from the northern New Jersey band, The Emerald Experience, to play and tour with The Happenings.

 That lineup performed mostly at colleges and universities until 1970, when Libert left the band to manage other groups, including George Clinton and
Parliament-Funkadelic, Living Colour, Brian Auger, Vanilla Fudge, The Runaways (Cherie Currie, Joan Jett, Lita Ford), Mother's Finest, Alice Cooper and Evelyn
"Champagne" King.
</item 3>

<item 4>
Man or Astro-man?
Man or Astro-man?

 is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
</item 4>

<item 5>
More Songs About Buildings and Food
More Songs About Buildings and Food is the second studio album by the American rock band Talking Heads, released in July 1978.

 The album was the first of three Talking Heads LPs produced by collaborator Brian Eno.

 It saw the group move musically toward a danceable style, crossing singer David Byrne's unusual delivery with new emphasis on the rhythm section (made up of
bassist Tina Weymouth and drummer Chris Frantz).
</item 5>

<item 6>
Road to Nowhere
"Road to Nowhere" is a rock song written by David Byrne for the 1985 Talking Heads album "Little Creatures."

 It also appeared on "Best of Talking Heads", "", the "Once in a Lifetime" box set and the "Brick" box set.

 The song was released as a single in 1985 and reached number 25 on the Billboard Mainstream Rock Tracks and number 6 in the British, German and South African
singles charts.

 It also made number 8 on the Dutch Top 40.
</item 6>

<item 7>
Method Man
Clifford Smith (born April 1, 1971), better known by his stage name Method Man, is an American rapper, record producer, and actor.

 He is known as a member of the East Coast hip hop collective Wu-Tang Clan.

 He is also one half of the hip hop duo Method Man & Redman.

 He took his stage name from the 1979 film "Method Man".

 In 1996, he won a Grammy Award for Best Rap Performance by a Duo or Group, for "I'll Be There for You/You're All I Need to Get By", with American R&B
singer-songwriter Mary J. Blige.
</item 7>

<item 8>
No Talking, Just Head
No Talking, Just Head is an album released in 1996 by The Heads, a band composed of Jerry Harrison, Tina Weymouth, and Chris Frantz of Talking Heads, joined by
a variety of guest singers.

 Its name may be seen as an allusion to the fact that Talking Heads' former vocalist, David Byrne, is the only member not involved.
</item 8>

<item 9>
Talking Heads
Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.

 The band comprised David Byrne (lead vocals, guitar), Chris Frantz (drums), Tina Weymouth (bass), and Jerry Harrison (keyboards, guitar).

 Described by critic Stephen Thomas Erlewine as "one of the most critically acclaimed bands of the '80s," the group helped to pioneer new wave music by
integrating elements of punk, art rock, funk, and world music with avant-garde sensibilities and an anxious, clean-cut image.
</item 9>

<item 10>
The Best of Talking Heads
The Best of Talking Heads is a 2004 greatest hits album by Talking Heads, released by Sire/Rhino/Warner Bros., and contains in all 18 tracks, from the beginning
to the end of Talking Heads' history.

 It was released the same day (August 17, 2004) as the expanded reissue of "The Name of This Band Is Talking Heads".

 The album charted at number 87 on the ARIA Charts and charted at number 96 on the Ultratop Charts in Belgium.
</item 10>
=== END ITEM id=00dadeb63d3cd7aa ===

=== ITEM id=6f97a32d9446bc3c ===
Gold sentences and facts:
Sentence 0: Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area women role models over 85 years old who remained artistically active.
  Fact 0.0: Still Kicking: Six Artistic Women of Project Arts & Longevity is a documentary film.
  Fact 0.1: Still Kicking: Six Artistic Women of Project Arts & Longevity was released in 2006.
  Fact 0.2: Still Kicking: Six Artistic Women of Project Arts & Longevity runs 32 minutes.
  Fact 0.3: Still Kicking: Six Artistic Women of Project Arts & Longevity was made by Pacific Grove filmmaker Greg Young.
  Fact 0.4: Still Kicking: Six Artistic Women of Project Arts & Longevity features six Bay Area women role models.
  Fact 0.5: The six women featured in Still Kicking are over 85 years old.
  Fact 0.6: The six women featured in Still Kicking remained artistically active.
Sentence 1: Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
  Fact 1.0: Time Bombs is a film.
  Fact 1.1: Time Bombs is a 2008 film.
  Fact 1.2: Time Bombs is a Canadian film.
  Fact 1.3: Time Bombs was produced by Productions de la ruelle.

Retrieved items:
<item 1>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 1>

<item 2>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 2>

<item 3>
The entries cover documentaries on Boston hardcore (All Ages, 2012), punk (Punk’s Not Dead, 2007; American Hardcore, 2006), bipolar disorder, artistic aging,
slavery, archaeology, bookstores, and Charles Bukowski; plus Barbara Hammer’s autobiographical Tender Fictions. Fiction films include Bakshi’s acclaimed X-rated
Heavy Traffic, Heavyweights, several American Pie and Bring It On sequels, Schrader’s Hardcore, Blue Collar, and The
</item 3>

<item 4>
Time Bomb High School
Time Bomb High School is the second album by the Reigning Sound.

 It was released in 2002 on In the Red Records.

 The album featured the original Reigning Sound lineup of Greg Cartwright on lead vocals and guitar; Alex Greene on organ, piano, guitar, and backing vocals;
Jeremy Scott on bass, and backing vocals; and Greg Roberson on drums.

  Howlin' Pelle Almqvist of The Hives commented in "Rolling Stone" that Time Bomb High School was his favorite record of 2002.
</item 4>

<item 5>
Time Bombs
Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
</item 5>

<item 6>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 6>

<item 7>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 7>

<item 8>
Still Kicking: The Fabulous Palm Springs Follies
Still Kicking: The Fabulous Palm Springs Follies is a 1997 American short documentary film directed by Mel Damski.

 It was nominated at the 70th Academy Awards for Best Documentary Short Subject.

 It features The Fabulous Palm Springs Follies, which was formerly staged at the Palm Springs, California Plaza Theatre.
</item 8>

<item 9>
The entries span film, television, documentaries, and film distribution: Philippine series Sine Novela adapts Viva classics; The Doom Generation (1995) is Gregg
Araki’s violent “Teenage Apocalypse” film; Naked Gun 33⅓ concludes its comedy trilogy, while Naked City (1958–63) is a semi-documentary police drama. Night Life
is a zombie comedy; The Runaways portrays the band’s formation and Currie-Jett relationship. Consolidated Film Industries was
</item 9>

<item 10>
The entries cover filmmakers, artists and films. Takeshi Fukunaga’s debut, *Out of My Hand* (2015), won the LA Film Festival’s top prize and earned an
Independent Spirit nomination. Hou Hsiao-hsien, a Taiwan New Wave auteur, won Venice’s Golden Lion and Cannes’ Best Director award; his films include *The
Assassin*, *Dust in the Wind* and *The Time to Live and the Time to Die*. Cinematographer Mark Lee
</item 10>
=== END ITEM id=6f97a32d9446bc3c ===

=== ITEM id=4ca355bd6e97fd91 ===
Gold sentences and facts:
Sentence 0: Dana Robins Ivey (born August 12, 1941) is an American actress.
  Fact 0.0: Dana Robins Ivey was born on August 12, 1941.
  Fact 0.1: Dana Robins Ivey is an American actress.
Sentence 1: Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.
  Fact 1.0: Two Weeks Notice is a 2002 film.
  Fact 1.1: Two Weeks Notice is an Australian-American film.
  Fact 1.2: Two Weeks Notice is a romantic comedy film.
  Fact 1.3: Two Weeks Notice stars Hugh Grant.
  Fact 1.4: Two Weeks Notice stars Sandra Bullock.

Retrieved items:
<item 1>
Legally Blonde 2: Red, White &amp; Blonde
Legally Blonde 2: Red, White & Blonde (also referred to as just Legally Blonde 2) is a 2003 American comedy film directed by Charles Herman-Wurmfeld and written
by Kate Kondell.

 The sequel to the 2001 film "Legally Blonde", the film stars Reese Witherspoon alongside an ensemble cast featuring Sally Field, Regina King, Jennifer
Coolidge, Bruce McGill, Dana Ivey, Mary Lynn Rajskub, Bob Newhart, and Luke Wilson, with Coolidge and Wilson reprising their roles from the first film.
</item 1>

<item 2>
Kim Johnston Ulrich
Kim Johnston Ulrich (born March 24, 1955) is an American actress.

 From 1983 to 1986, Ulrich played the role of Diana McColl on "As the World Turns".

 In 1988, she guest-starred in two episodes of "Werewolf".

 In 1990, Ulrich appeared on the TV series "Wings" as Carol, Brian's ex-wife.

 In 1993, Ulrich appeared in the pilot episode of "" as Dr. Antoinette Baines, a villainous scientist.

 She appeared in a third season episode of "Highlander: The Series" in 1995.

 She appeared in the first episode of season 4 of "" as a conniving wife in "Murder by Friendly Fire" in 1996.
</item 2>

<item 3>
Heather Burns
Heather Burns (born April 7, 1975) is an American actress, known for her role as Miss Rhode Island in the 2000 film "Miss Congeniality".

 Her other film appearances include "You've Got Mail" (1998), "Two Weeks Notice" (2002) and "Bewitched" (2005).
</item 3>

<item 4>
Two Weeks Notice
Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.

 The film was written and directed by Marc Lawrence, who had previously scripted two other box-office hits for Bullock, "Miss Congeniality" and "Forces of
Nature".

 Although critical response was mixed, the film received a successful box office run, both in the United States and globally.
</item 4>

<item 5>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 5>

<item 6>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 6>

<item 7>
Dana (surname)
Dana ( or ) as a surname may have several origins.

 In England, it came from "dann", the valley of a meadow, and it may mean the dweller of that valley.

 In Continental Europe, it probably came from "Dane" (or Danish, from Denmark).

 This surname is related to Danese in Italy and it can be found mostly in the Piedmont region.

 It may also be a modification of Huguenot French origin, probably a variant of "d'Aunay", of geographical origin.

 It may be also a Gaelic patronymic, since it is a common forename in Ireland.

 Dana is a relatively common surname in the US, ranking 7161 out of 88,799 in the 1990 U.S. Census.

 Dana is also the Persian word for wisdom.
</item 7>

<item 8>
Dana Ivey
Dana Robins Ivey (born August 12, 1941) is an American actress.

 She is a five-time Tony Award nominee for her work on Broadway, and won the 1997 Drama Desk Award for Outstanding Featured Actress in a Play for her work in
both "Sex and Longing" and "The Last Night of Ballyhoo".

 Her film appearances include "The Color Purple" (1985), "Dirty Rotten Scoundrels" (1988), "The Addams Family" (1991), "Two Weeks Notice" (2002), "Rush Hour 3"
(2007) and "The Help" (2011).
</item 8>

<item 9>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 9>

<item 10>
Two Weeks (The Office)
"Two Weeks" is the twenty-first episode of the fifth season of the television series "The Office", and the 93rd overall episode of the series.

 It originally aired on NBC in the United States on March 26, 2009.

 In this episode, Michael, who has given his two weeks' notice to Dunder Mifflin, tries to convince others in the office to quit and join him in starting a new
paper company.

 Meanwhile, Pam spends her day trying to put together the new photocopier and becomes frustrated with her job.
</item 10>
=== END ITEM id=4ca355bd6e97fd91 ===

=== ITEM id=8487c1bf4f790c00 ===
Gold sentences and facts:
Sentence 0: The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.
  Fact 0.0: The Bears and I is a 1974 American drama film.
  Fact 0.1: The Bears and I was directed by Bernard McEveety.
  Fact 0.2: The Bears and I was written by John Whedon.
Sentence 1:  The film was released on July 31, 1974, by Buena Vista Distribution.
  Fact 1.0: The Bears and I was released on July 31, 1974.
  Fact 1.1: The Bears and I was released by Buena Vista Distribution.
Sentence 2: Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as co-director.
  Fact 2.0: Oceans (French: Océans) is a 2009 French nature documentary film.
  Fact 2.1: Oceans was directed by Jacques Perrin.
  Fact 2.2: Oceans was produced by Jacques Perrin.
  Fact 2.3: Oceans was co-written by Jacques Perrin.
  Fact 2.4: Oceans was narrated by Jacques Perrin.
  Fact 2.5: Jacques Cluzaud was co-director of Oceans.

Retrieved items:
<item 1>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 1>

<item 2>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 2>

<item 3>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 3>

<item 4>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 4>

<item 5>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 5>

<item 6>
Foxes (soundtrack)
Foxes is the soundtrack to the 1980 film of the same name, starring Jodie Foster, Scott Baio, Sally Kellerman, Randy Quaid as well as The Runaways' lead singer
Cherie Currie.

 The double-album was released on the disco label Casablanca Records.
</item 6>

<item 7>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 7>

<item 8>
Return to the Blue Lagoon
Return to the Blue Lagoon is a 1991 American romance and adventure film starring Milla Jovovich and Brian Krause, produced and directed by William A. Graham.

 The film is a sequel to "The Blue Lagoon" (1980).

 The screenplay by Leslie Stevens was based on the novel "The Garden of God" by Henry De Vere Stacpoole.

 The original music score was composed by Basil Poledouris.

 The film's closing theme song "A World of Our Own" is performed by Surface featuring Bernard Jackson.

 The music was written by Barry Mann, and the lyrics were written by Cynthia Weil.
</item 8>

<item 9>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 9>

<item 10>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 10>
=== END ITEM id=8487c1bf4f790c00 ===

=== ITEM id=f77ad4157aee5fb7 ===
Gold sentences and facts:
Sentence 0: The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US skyscraper.
  Fact 0.0: The Woolworth Building is at 233 Broadway, Manhattan, New York City.
  Fact 0.1: The Woolworth Building was designed by architect Cass Gilbert.
  Fact 0.2: The Woolworth Building was constructed between 1910 and 1912.
  Fact 0.3: The Woolworth Building is an early US skyscraper.
Sentence 1: 1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.
  Fact 1.0: 1 New York Plaza is an office building in New York City's Financial District.
  Fact 1.1: 1 New York Plaza was built in 1969.
  Fact 1.2: 1 New York Plaza is at the intersection of South and Whitehall Streets.

Retrieved items:
<item 1>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 1>

<item 2>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 2>

<item 3>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 3>

<item 4>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 4>

<item 5>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 5>

<item 6>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 6>

<item 7>
Woolworth Building
The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US
skyscraper.

 The original site for the building was purchased by F. W. Woolworth and his real estate agent Edward J. Hogan by April 15, 1910, from the Trenor Luther Park
Estate and other owners for $1.65 million.

 By January 18, 1911, Woolworth and Hogan had acquired the final site for the project, totaling $4.5 million.

 More than a century after its construction, it remains, at 241.4 m , one of the 100 tallest buildings in the United States as well as one of the 30 tallest
buildings in New York City.

 It has been a National Historic Landmark since 1966, and a New York City landmark since 1983.
</item 7>

<item 8>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 8>

<item 9>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
</item 9>

<item 10>
F. W. Woolworth Building (Watertown, New York)
The Woolworth Building is an historic building in Watertown, New York.

 It is a contributing building in the Public Square Historic District.

 Plans for the Woolworth Building were begun in 1916 by Frank W. Woolworth, the founder of the Woolworth's chain of department stores.
</item 10>
=== END ITEM id=f77ad4157aee5fb7 ===

=== ITEM id=038960204af1a476 ===
Gold sentences and facts:
Sentence 0: Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.
  Fact 0.0: Aleksandr Ivanovsky lived from 1881 to 1968.
  Fact 0.1: Aleksandr Ivanovsky was a Russian screenwriter.
  Fact 0.2: Aleksandr Ivanovsky was a Russian film director.
  Fact 0.3: Aleksandr Ivanovsky worked during the Soviet era.
Sentence 1: Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera" (1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".
  Fact 1.0: Arthur Lubin was born on July 25, 1898.
  Fact 1.1: Arthur Lubin died on May 12, 1995.
  Fact 1.2: Arthur Lubin was an American film director.
  Fact 1.3: Arthur Lubin was an American producer.
  Fact 1.4: Arthur Lubin directed several 'Abbott & Costello' films.
  Fact 1.5: Arthur Lubin directed 'Phantom of the Opera', released in 1943.
  Fact 1.6: Arthur Lubin directed the 'Francis the Talking Mule' series.
  Fact 1.7: Arthur Lubin created the talking-horse TV series 'Mister Ed'.

Retrieved items:
<item 1>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 1>

<item 2>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 2>

<item 3>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 3>

<item 4>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 4>

<item 5>
Richard Lund
Richard Lund (9 July 1885 – 27 September 1960) was a Swedish film and theatre actor.

 He made his stage debut at Stora Teatern in Gothenburg in 1904 and later appeared in 73 films between 1912 and 1952, making his most important roles during the
silent film era.

 Among his best-known roles is that of Sir Archie in Mauritz Stiller's "Sir Arne's Treasure" (1919).

 He was the most prominent "first lover" of Swedish film during his heyday.
</item 5>

<item 6>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 6>

<item 7>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 7>

<item 8>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 8>

<item 9>
A Successful Failure
A Successful Failure (1934) is an American film directed by Arthur Lubin.

 It was Lubin's first movie as director.
</item 9>

<item 10>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 10>
=== END ITEM id=038960204af1a476 ===
